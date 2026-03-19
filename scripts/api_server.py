"""
FastAPI Server for Text-to-SQL Models (Baseline + Fine-Tuned)

Serves both the untrained base LLM and the LoRA-adapted model via REST API,
enabling side-by-side comparison in the Streamlit app and evaluation scripts.

Endpoints:
    POST /generate            - Generate SQL using the fine-tuned (LoRA) model
    POST /generate-baseline   - Generate SQL using the untrained base model
    GET  /health              - Health check

Usage:
    python scripts/api_server.py --model-path artifacts/fine_tuned_model
    python scripts/api_server.py --model-path artifacts/fine_tuned_model --port 8000
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

# Disable torch dynamo to avoid numpy compatibility issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


app = FastAPI(title="Analytics Copilot - SQL Generation API")

# Global state
_finetuned_model = None
_base_model = None
_tokenizer = None
_device = None
_base_model_name = None


class GenerateRequest(BaseModel):
    question: str
    schema_context: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.0


class GenerateResponse(BaseModel):
    sql: str
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    baseline_loaded: bool
    device: str


def format_prompt(question: str, schema_context: str) -> str:
    """Format the prompt in the same Alpaca-style template used during training."""
    instruction = (
        "You are a Snowflake SQL expert. Given a natural language question about "
        "the Olist Brazilian E-Commerce dataset, generate a correct Snowflake SQL query.\n\n"
    )
    if schema_context:
        instruction += f"{schema_context}\n\n"
    instruction += "Return ONLY the SQL query, no explanations."

    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{question}\n\n"
        f"### Response:\n"
    )


def extract_sql(text: str) -> str:
    """Extract SQL from model output, removing prompt artifacts."""
    if "### Response:" in text:
        text = text.split("### Response:")[-1]

    text = re.sub(r'```(?:sql)?', '', text, flags=re.IGNORECASE)
    text = text.replace('```', '')

    sql_match = re.search(r'\b(SELECT|WITH)\b', text, re.IGNORECASE)
    if sql_match:
        text = text[sql_match.start():]

    text = re.split(r'\n\n[A-Z]', text)[0]
    text = text.split('\n\n###')[0]
    text = text.split('\n\nNote:')[0]

    text = text.rstrip(';').strip()
    return text


def _generate_with_model(model, request: GenerateRequest) -> str:
    """Run inference on a given model."""
    prompt = format_prompt(request.question, request.schema_context)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=max(request.temperature, 0.01),
            do_sample=request.temperature > 0,
            top_p=0.9 if request.temperature > 0 else 1.0,
            pad_token_id=_tokenizer.pad_token_id,
        )

    full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_sql(full_text)


@app.post("/generate", response_model=GenerateResponse)
async def generate_finetuned(request: GenerateRequest):
    """Generate SQL using the LoRA fine-tuned model."""
    start = time.time()
    sql = _generate_with_model(_finetuned_model, request)
    latency = (time.time() - start) * 1000

    return GenerateResponse(
        sql=sql,
        model="fine-tuned-lora",
        latency_ms=round(latency, 1),
    )


@app.post("/generate-baseline", response_model=GenerateResponse)
async def generate_baseline(request: GenerateRequest):
    """Generate SQL using the untrained base model (no LoRA)."""
    start = time.time()
    sql = _generate_with_model(_base_model, request)
    latency = (time.time() - start) * 1000

    return GenerateResponse(
        sql=sql,
        model=f"baseline ({_base_model_name})",
        latency_ms=round(latency, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _finetuned_model is not None else "model_not_loaded",
        model_loaded=_finetuned_model is not None,
        baseline_loaded=_base_model is not None,
        device=str(_device) if _device else "none",
    )


def load_models(model_path: str, base_model_id: str = None, use_4bit: bool = False):
    """Load both the base model and the LoRA-adapted model."""
    global _finetuned_model, _base_model, _tokenizer, _device, _base_model_name

    model_path = Path(model_path)

    # Detect device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    print(f"Device: {_device}")

    # Determine base model
    if base_model_id is None:
        config_path = model_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            base_model_id = config.get("base_model", "codellama/CodeLlama-7b-Instruct-hf")
        else:
            base_model_id = "codellama/CodeLlama-7b-Instruct-hf"

    _base_model_name = base_model_id
    print(f"Base model: {base_model_id}")
    print(f"LoRA adapter: {model_path}")

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Model loading kwargs
    model_kwargs = {"trust_remote_code": True}
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
    elif _device.type == "cpu":
        model_kwargs["torch_dtype"] = torch.float32
    else:
        model_kwargs["torch_dtype"] = torch.float16

    # Load base model (kept as-is for baseline)
    print("Loading base model (baseline)...")
    _base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    _base_model.eval()

    # Load fine-tuned model (base + LoRA adapter)
    print("Loading LoRA adapter (fine-tuned)...")
    base_for_lora = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    _finetuned_model = PeftModel.from_pretrained(base_for_lora, str(model_path))
    _finetuned_model.eval()

    if _device.type not in ("cpu",) and not use_4bit:
        _base_model = _base_model.to(_device)
        _finetuned_model = _finetuned_model.to(_device)

    print("Both models loaded successfully!")


def main():
    parser = argparse.ArgumentParser(description="FastAPI server for baseline + fine-tuned SQL models")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--base-model", default=None, help="Base model ID (auto-detected from training config)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    print("=" * 60)
    print("  Analytics Copilot - SQL Generation API Server")
    print("  (Baseline + Fine-Tuned LoRA)")
    print("=" * 60)

    load_models(args.model_path, args.base_model, args.use_4bit)

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"  POST http://localhost:{args.port}/generate           (fine-tuned)")
    print(f"  POST http://localhost:{args.port}/generate-baseline  (untrained base)")
    print(f"  GET  http://localhost:{args.port}/health")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
