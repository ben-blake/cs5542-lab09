"""
Evaluation Script: Baseline (Untrained Llama) vs Domain-Adapted (Fine-Tuned LoRA) Model

Runs a set of evaluation queries through both models and compares:
- Execution accuracy (does the SQL run without errors?)
- Domain relevance (uses fully qualified table names, correct Snowflake syntax)
- Latency

Usage:
    # Compare both models (requires Snowflake + FastAPI server running)
    python scripts/evaluate_adaptation.py

    # Evaluate fine-tuned model only
    python scripts/evaluate_adaptation.py --model finetuned

    # Evaluate baseline (untrained) only
    python scripts/evaluate_adaptation.py --model baseline
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.snowflake_conn import get_session, close_session
from src.utils.config import get_config
from src.agents.schema_linker import link_schema
from src.agents.validator import validate_and_execute
from src.utils.finetuned_client import (
    generate_sql_finetuned,
    generate_sql_baseline,
    check_finetuned_api,
)


# 15 evaluation queries spanning easy/medium/hard
EVAL_QUERIES = [
    # Easy
    {"question": "How many orders are in the database?", "difficulty": "easy"},
    {"question": "What is the average review score?", "difficulty": "easy"},
    {"question": "How many sellers are in São Paulo state?", "difficulty": "easy"},
    {"question": "What is the total payment value for credit card payments?", "difficulty": "easy"},
    {"question": "How many distinct product categories are there?", "difficulty": "easy"},
    # Medium
    {"question": "What is the total revenue by customer state?", "difficulty": "medium"},
    {"question": "What are the top 5 product categories by number of orders?", "difficulty": "medium"},
    {"question": "What is the average delivery time in days by customer state?", "difficulty": "medium"},
    {"question": "How many orders per month were placed in 2018?", "difficulty": "medium"},
    {"question": "What is the average review score by product category?", "difficulty": "medium"},
    # Hard
    {"question": "What is the month-over-month growth rate in total payments?", "difficulty": "hard"},
    {"question": "Find the top 3 sellers by revenue in each state", "difficulty": "hard"},
    {"question": "What percentage of orders were delivered late?", "difficulty": "hard"},
    {"question": "Rank customer states by total number of orders", "difficulty": "hard"},
    {"question": "Calculate customer lifetime value and rank top 20 customers", "difficulty": "hard"},
]


def evaluate_query(session, question: str, model: str, schema_context: list) -> dict:
    """Evaluate a single query with the specified model."""
    result = {
        "question": question,
        "model": model,
        "sql_generated": "",
        "execution_success": False,
        "row_count": 0,
        "error": None,
        "latency_ms": 0,
        "uses_qualified_names": False,
    }

    start = time.time()

    try:
        # Generate SQL
        if model == "baseline":
            sql = generate_sql_baseline(question, schema_context)
        else:
            sql = generate_sql_finetuned(question, schema_context)

        result["sql_generated"] = sql or ""

        if not sql or not sql.strip():
            result["error"] = "Empty SQL generated"
            result["latency_ms"] = (time.time() - start) * 1000
            return result

        # Check for fully qualified table names
        result["uses_qualified_names"] = "ANALYTICS_COPILOT.RAW." in sql.upper()

        # Validate and execute
        cfg = get_config()
        max_retries = cfg.get("sql_generator", {}).get("max_retries", 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_sql, exec_result = validate_and_execute(
                session, sql, question, schema_context, max_retries=max_retries
            )

        result["sql_generated"] = final_sql

        if isinstance(exec_result, str):
            result["error"] = exec_result
        else:
            result["execution_success"] = True
            df = exec_result.to_pandas()
            result["row_count"] = len(df)

    except Exception as e:
        result["error"] = str(e)

    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result


def run_evaluation(models: list[str]):
    """Run evaluation across all queries and models."""
    session = get_session()

    all_results = []

    for i, query in enumerate(EVAL_QUERIES, 1):
        question = query["question"]
        difficulty = query["difficulty"]
        print(f"\n[{i}/{len(EVAL_QUERIES)}] ({difficulty}) {question}")

        # Get schema context once (shared by both models)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            schema_context = link_schema(session, question)

        if not schema_context:
            print("  Schema linking failed, skipping")
            for model in models:
                all_results.append({
                    "question": question,
                    "difficulty": difficulty,
                    "model": model,
                    "sql_generated": "",
                    "execution_success": False,
                    "row_count": 0,
                    "error": "Schema linking returned empty",
                    "latency_ms": 0,
                    "uses_qualified_names": False,
                })
            continue

        for model in models:
            print(f"  [{model}] ", end="", flush=True)
            result = evaluate_query(session, question, model, schema_context)
            result["difficulty"] = difficulty

            status = "PASS" if result["execution_success"] else "FAIL"
            print(f"{status} ({result['latency_ms']:.0f}ms, {result['row_count']} rows)")
            if result["error"]:
                print(f"    Error: {result['error'][:100]}")

            all_results.append(result)

    close_session()
    return all_results


def print_summary(results: list[dict], models: list[str]):
    """Print evaluation summary table."""
    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)

    for model in models:
        model_results = [r for r in results if r["model"] == model]
        total = len(model_results)
        passed = sum(1 for r in model_results if r["execution_success"])
        qualified = sum(1 for r in model_results if r["uses_qualified_names"])
        avg_latency = sum(r["latency_ms"] for r in model_results) / max(total, 1)

        print(f"\n  Model: {model.upper()}")
        print(f"  {'Execution Accuracy:':<30} {passed}/{total} ({passed/total*100:.0f}%)")
        print(f"  {'Uses Qualified Names:':<30} {qualified}/{total} ({qualified/total*100:.0f}%)")
        print(f"  {'Avg Latency:':<30} {avg_latency:.0f}ms")

        # By difficulty
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in model_results if r["difficulty"] == diff]
            if diff_results:
                diff_pass = sum(1 for r in diff_results if r["execution_success"])
                print(f"  {'  ' + diff.capitalize() + ':':<30} {diff_pass}/{len(diff_results)}")

    # Side-by-side comparison if both models
    if len(models) == 2:
        print(f"\n  COMPARISON")
        print(f"  {'Query':<55} {'Baseline':<10} {'Fine-Tuned':<10}")
        print(f"  {'-'*75}")

        questions_seen = set()
        for r in results:
            q = r["question"]
            if q in questions_seen:
                continue
            questions_seen.add(q)

            bl_r = next((x for x in results if x["question"] == q and x["model"] == "baseline"), None)
            ft_r = next((x for x in results if x["question"] == q and x["model"] == "finetuned"), None)

            b_status = "PASS" if bl_r and bl_r["execution_success"] else "FAIL"
            f_status = "PASS" if ft_r and ft_r["execution_success"] else "FAIL"

            short_q = q[:52] + "..." if len(q) > 55 else q
            print(f"  {short_q:<55} {b_status:<10} {f_status:<10}")

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "finetuned", "both"], default="both")
    args = parser.parse_args()

    if args.model == "both":
        models = ["baseline", "finetuned"]
    else:
        models = [args.model]

    # Check API availability
    if any(m in models for m in ("finetuned", "baseline")):
        if not check_finetuned_api():
            print("WARNING: Model API is not available.")
            print("Start it with: python scripts/api_server.py --model-path artifacts/fine_tuned_model")
            sys.exit(1)

    print("=" * 80)
    print("  Analytics Copilot - Domain Adaptation Evaluation")
    print(f"  Models: {', '.join(models)}")
    print(f"  Queries: {len(EVAL_QUERIES)}")
    print("=" * 80)

    results = run_evaluation(models)
    print_summary(results, models)

    # Save results
    output_path = Path(__file__).parent.parent / "artifacts" / "adaptation_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
