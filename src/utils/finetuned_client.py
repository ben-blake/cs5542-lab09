"""
Model API Client

HTTP client for calling both the fine-tuned (LoRA) and baseline (untrained)
models via the FastAPI endpoint. Used alongside the Cortex-based sql_generator.
"""

import warnings
from typing import Any

import requests

from src.utils.config import get_config


def generate_sql_finetuned(
    question: str,
    schema_context: list[dict[str, Any]],
    api_url: str = None,
) -> str:
    """
    Generate SQL using the fine-tuned model via FastAPI.

    Args:
        question: User's natural language question
        schema_context: Relevant tables from schema linker
        api_url: FastAPI server URL (default from config.yaml)

    Returns:
        Generated SQL string, or empty string on error.
    """
    if api_url is None:
        cfg = get_config()
        api_url = cfg.get("finetuned", {}).get("api_url", "http://localhost:8000")

    # Format schema context as text for the model
    schema_text = _format_schema_for_prompt(schema_context)

    try:
        response = requests.post(
            f"{api_url}/generate",
            json={
                "question": question,
                "schema_context": schema_text,
                "max_new_tokens": 256,
                "temperature": 0.0,
            },
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("sql", "")

    except requests.ConnectionError:
        warnings.warn(
            f"Cannot connect to fine-tuned model API at {api_url}. "
            "Start it with: python scripts/api_server.py --model-path artifacts/fine_tuned_model"
        )
        return ""
    except Exception as e:
        warnings.warn(f"Error calling fine-tuned model API: {e}")
        return ""


def generate_sql_baseline(
    question: str,
    schema_context: list[dict[str, Any]],
    api_url: str = None,
) -> str:
    """
    Generate SQL using the untrained base model (no LoRA) via FastAPI.

    Args:
        question: User's natural language question
        schema_context: Relevant tables from schema linker
        api_url: FastAPI server URL

    Returns:
        Generated SQL string, or empty string on error.
    """
    if api_url is None:
        cfg = get_config()
        api_url = cfg.get("finetuned", {}).get("api_url", "http://localhost:8000")

    schema_text = _format_schema_for_prompt(schema_context)

    try:
        response = requests.post(
            f"{api_url}/generate-baseline",
            json={
                "question": question,
                "schema_context": schema_text,
                "max_new_tokens": 256,
                "temperature": 0.0,
            },
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("sql", "")

    except requests.ConnectionError:
        warnings.warn(f"Cannot connect to baseline model API at {api_url}.")
        return ""
    except Exception as e:
        warnings.warn(f"Error calling baseline model API: {e}")
        return ""


def check_finetuned_api(api_url: str = None) -> bool:
    """Check if the model API (both fine-tuned and baseline) is available."""
    if api_url is None:
        cfg = get_config()
        api_url = cfg.get("finetuned", {}).get("api_url", "http://localhost:8000")

    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except Exception:
        return False


def _format_schema_for_prompt(schema_context: list[dict[str, Any]]) -> str:
    """Format schema context into text for the fine-tuned model prompt."""
    if not schema_context:
        return ""

    lines = ["Available tables:"]
    for table in schema_context:
        table_name = table.get("table_name", "UNKNOWN")
        if not table_name.startswith("ANALYTICS_COPILOT"):
            table_name = f"ANALYTICS_COPILOT.RAW.{table_name}"

        cols = []
        for col in table.get("columns", []):
            cols.append(f"{col['column_name']} ({col.get('data_type', '')})")

        lines.append(f"- {table_name}: {', '.join(cols)}")

    return "\n".join(lines)
