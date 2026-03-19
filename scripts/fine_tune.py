"""
LoRA Fine-Tuning Script for Text-to-SQL Domain Adaptation

Fine-tunes a base LLM (CodeLlama-7B-Instruct) using LoRA (Low-Rank Adaptation)
on the Olist Text-to-SQL instruction dataset.

This script can run on:
- Google Colab (T4 GPU, free tier) with 4-bit quantization
- Local machine with NVIDIA GPU (16GB+ VRAM recommended)
- CPU (very slow, for testing only)

Usage:
    # Default: CodeLlama-7B-Instruct with LoRA
    python scripts/fine_tune.py

    # Custom base model
    python scripts/fine_tune.py --base-model mistralai/Mistral-7B-Instruct-v0.3

    # CPU-only test run (1 epoch, small batch)
    python scripts/fine_tune.py --cpu --epochs 1

    # Resume from checkpoint
    python scripts/fine_tune.py --resume artifacts/fine_tuned_model/checkpoint-100

Requirements:
    pip install torch transformers peft datasets accelerate bitsandbytes
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


PROJECT_ROOT = Path(__file__).parent.parent


def format_example(example: dict, tokenizer) -> str:
    """Format a single instruction example into the chat/instruct template."""
    # Use a simple Alpaca-style template that works across models
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}{tokenizer.eos_token}"
    )
    return text


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 1024):
    """Tokenize the dataset for training."""
    def tokenize_fn(examples):
        texts = [
            format_example({"instruction": inst, "input": inp, "output": out}, tokenizer)
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # For causal LM, labels = input_ids (the model predicts next token)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


def load_instruction_data() -> tuple[Dataset, Dataset]:
    """Load train and validation datasets."""
    train_path = PROJECT_ROOT / "data" / "instruction_train.json"
    val_path = PROJECT_ROOT / "data" / "instruction_val.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Text-to-SQL")
    parser.add_argument(
        "--base-model",
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "fine_tuned_model"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (slow)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    if args.no_4bit:
        args.use_4bit = False

    print("=" * 60)
    print("  LoRA Fine-Tuning for Text-to-SQL Domain Adaptation")
    print("=" * 60)
    print(f"Base model:    {args.base_model}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank:     {args.lora_r}")
    print(f"LoRA alpha:    {args.lora_alpha}")
    print(f"4-bit quant:   {args.use_4bit}")
    print(f"Device:        {'CPU' if args.cpu else 'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"Output:        {args.output_dir}")
    print()

    # ── Load Data ──────────────────────────────────────────────────
    print("Loading instruction dataset...")
    train_ds, val_ds = load_instruction_data()
    print(f"  Train: {len(train_ds)} examples")
    print(f"  Val:   {len(val_ds)} examples")

    # ── Load Tokenizer ─────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Tokenize Data ──────────────────────────────────────────────
    print("Tokenizing datasets...")
    train_tokenized = tokenize_dataset(train_ds, tokenizer, args.max_length)
    val_tokenized = tokenize_dataset(val_ds, tokenizer, args.max_length)

    # ── Load Model ─────────────────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    model_kwargs = {"trust_remote_code": True}

    if args.cpu:
        model_kwargs["torch_dtype"] = torch.float32
    elif args.use_4bit and torch.cuda.is_available():
        # QLoRA: 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if args.use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ─────────────────────────────────────────────────
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training Arguments ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4 if not args.cpu else 1,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available() and not args.cpu,
        bf16=False,
        report_to="none",
        seed=42,
        no_cuda=args.cpu,
        remove_unused_columns=False,
    )

    # ── Data Collator ──────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # ── Trainer ────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────
    print("\nStarting training...")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save ───────────────────────────────────────────────────────
    print(f"\nSaving fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config for reproducibility
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\nFine-tuning complete!")
    print(f"Model saved to: {args.output_dir}")
    print(f"To serve: python scripts/api_server.py --model-path {args.output_dir}")


if __name__ == "__main__":
    main()
