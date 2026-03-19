"""
Evaluate Analytics Copilot Accuracy

This script evaluates the Analytics Copilot's performance using golden queries.
It runs each question through the full copilot pipeline (schema_linker →
sql_generator → validator) and measures accuracy and latency.

The script:
1. Loads golden queries from data/golden_queries.json
2. For each question:
   - Runs through full copilot pipeline
   - Measures execution success and latency
   - Captures errors if any
3. Calculates metrics:
   - Execution Accuracy: % of queries that execute successfully
   - Average Latency: mean time from question to result
4. Generates detailed report:
   - Overall statistics
   - Per-difficulty breakdown
   - Failed queries with error messages
5. Saves results to data/evaluation_report.json

Usage:
    python scripts/evaluate.py [--limit N] [--difficulty LEVEL]

    Options:
        --limit N           Evaluate only first N questions (default: all)
        --difficulty LEVEL  Evaluate only questions of this difficulty
                           (easy/medium/hard, default: all)

Example:
    python scripts/evaluate.py --limit 10 --difficulty easy
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.snowflake_conn import get_session, close_session
from src.utils.config import get_config
from src.agents.schema_linker import link_schema
from src.agents.sql_generator import generate_sql
from src.agents.validator import validate_and_execute
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def main():
    """Main entry point for copilot evaluation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate Analytics Copilot accuracy using golden queries'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Evaluate only first N questions (default: all)'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['easy', 'medium', 'hard'],
        default=None,
        help='Evaluate only questions of this difficulty (default: all)'
    )

    args = parser.parse_args()

    logger.info("Starting evaluation")
    print("="*70)
    print("ANALYTICS COPILOT EVALUATION")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.limit:
        print(f"Question limit: {args.limit}")
    if args.difficulty:
        print(f"Difficulty filter: {args.difficulty}")

    print("="*70)
    print()

    try:
        # Load golden queries
        print("Loading golden queries from data/golden_queries.json...")
        golden_queries = load_golden_queries()

        # Apply filters
        if args.difficulty:
            golden_queries = [
                q for q in golden_queries
                if q.get('difficulty') == args.difficulty
            ]
            print(f"Filtered to {len(golden_queries)} {args.difficulty} questions")

        if args.limit:
            golden_queries = golden_queries[:args.limit]
            print(f"Limited to first {len(golden_queries)} questions")

        print(f"✓ Loaded {len(golden_queries)} questions to evaluate\n")

        if len(golden_queries) == 0:
            print("No questions to evaluate. Exiting.")
            sys.exit(0)

        # Connect to Snowflake
        print("Connecting to Snowflake...")
        session = get_session()
        print("✓ Connected successfully\n")

        # Run evaluation
        print("="*70)
        print("RUNNING EVALUATION")
        print("="*70)
        print()

        results = evaluate_questions(session, golden_queries)

        # Calculate metrics
        print("\n" + "="*70)
        print("CALCULATING METRICS")
        print("="*70)
        print()

        metrics = calculate_metrics(results)
        logger.info("Evaluation accuracy: %.1f%% (%d/%d)", metrics['execution_accuracy'],
                    metrics['successful_questions'], metrics['total_questions'])
        print_report(metrics)

        # Save results
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        print()

        save_report(metrics, results)

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Exiting...")
        sys.exit(1)

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Always close the session
        print("\nClosing Snowflake connection...")
        close_session()


def load_golden_queries() -> list[dict]:
    """
    Load golden queries from data/golden_queries.json file.

    Returns:
        list[dict]: List of question objects with keys:
            - id: Question ID
            - question: Natural language question
            - sql_query: Expected SQL query (for reference)
            - difficulty: Difficulty level
            - tables_used: Tables referenced
            - verified: Whether query was validated

    Raises:
        FileNotFoundError: If golden_queries.json doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """

    project_root = Path(__file__).parent.parent
    cfg = get_config()
    golden_path = cfg.get("evaluation", {}).get("golden_queries_path", "data/golden_queries.json")
    json_path = project_root / golden_path

    if not json_path.exists():
        raise FileNotFoundError(
            f"Golden queries file not found at {json_path}\n"
            f"Please run scripts/generate_golden.py first to generate questions."
        )

    with open(json_path, 'r') as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError("golden_queries.json must contain a JSON array")

    return questions


def evaluate_questions(session, golden_queries: list[dict]) -> list[dict]:
    """
    Evaluate all golden queries through the copilot pipeline.

    For each question:
    1. Start timer
    2. Run schema_linker to find relevant tables
    3. Run sql_generator to generate SQL
    4. Run validator to validate and execute
    5. Stop timer and record result

    Args:
        session: Active Snowflake Snowpark Session
        golden_queries: List of question objects to evaluate

    Returns:
        list[dict]: List of result objects with keys:
            - id: Question ID
            - question: Original question
            - expected_sql: Golden SQL query (for reference)
            - difficulty: Difficulty level
            - tables_used: Expected tables
            - generated_sql: SQL generated by copilot
            - success: Boolean - did query execute successfully?
            - latency_seconds: Time from question to result
            - error_message: Error message if failed (None if success)
            - schema_context: Tables found by schema_linker
    """

    results = []
    total = len(golden_queries)

    for i, query in enumerate(golden_queries, 1):
        print(f"[{i}/{total}] Evaluating: {query['question'][:60]}...")

        result = {
            'id': query.get('id', i),
            'question': query['question'],
            'expected_sql': query.get('sql_query', ''),
            'difficulty': query['difficulty'],
            'tables_used': query.get('tables_used', ''),
            'generated_sql': '',
            'success': False,
            'latency_seconds': 0.0,
            'error_message': None,
            'schema_context': []
        }

        try:
            # Start timing
            start_time = time.time()

            # Step 1: Schema Linker - find relevant tables
            print(f"  → Running schema_linker...")
            schema_context = link_schema(session, query['question'])

            if not schema_context or len(schema_context) == 0:
                # Schema linker found no relevant tables
                end_time = time.time()
                result['latency_seconds'] = end_time - start_time
                result['success'] = False
                result['error_message'] = "Schema linker found no relevant tables"
                print(f"  ✗ Failed: No relevant tables found")
                results.append(result)
                continue

            result['schema_context'] = [
                {'table_name': t['table_name'], 'relevance_score': t['relevance_score']}
                for t in schema_context
            ]

            print(f"  → Found {len(schema_context)} relevant tables")

            # Step 2: SQL Generator - generate query
            print(f"  → Running sql_generator...")
            generated_sql = generate_sql(session, query['question'], schema_context)

            if not generated_sql or not generated_sql.strip():
                # SQL generator failed
                end_time = time.time()
                result['latency_seconds'] = end_time - start_time
                result['success'] = False
                result['error_message'] = "SQL generator returned empty query"
                print(f"  ✗ Failed: No SQL generated")
                results.append(result)
                continue

            result['generated_sql'] = generated_sql
            print(f"  → Generated SQL ({len(generated_sql)} chars)")

            # Step 3: Validator - validate and execute
            print(f"  → Running validator...")
            cfg = get_config()
            max_retries = cfg.get("sql_generator", {}).get("max_retries", 3)
            final_sql, execution_result = validate_and_execute(
                session, generated_sql, query['question'], schema_context, max_retries=max_retries
            )

            # Stop timing
            end_time = time.time()
            result['latency_seconds'] = end_time - start_time

            # Update with final SQL (may be different if retries occurred)
            result['generated_sql'] = final_sql

            # Check if execution succeeded
            if isinstance(execution_result, str):
                # Result is an error message
                result['success'] = False
                result['error_message'] = execution_result
                print(f"  ✗ Failed: {execution_result[:50]}...")
            else:
                # Result is a DataFrame - success!
                result['success'] = True
                result['error_message'] = None
                print(f"  ✓ Success ({result['latency_seconds']:.2f}s)")

        except Exception as e:
            # Unexpected error
            end_time = time.time()
            result['latency_seconds'] = end_time - start_time
            result['success'] = False
            result['error_message'] = f"Unexpected error: {str(e)}"
            print(f"  ✗ Failed: {str(e)[:50]}...")

        results.append(result)
        print()  # Blank line between questions

    return results


def calculate_metrics(results: list[dict]) -> dict:
    """
    Calculate evaluation metrics from results.

    Computes:
    - Overall execution accuracy (% successful)
    - Average latency (mean time per query)
    - Accuracy by difficulty level
    - Average latency by difficulty level
    - Failed queries with error details

    Args:
        results: List of evaluation result objects

    Returns:
        dict: Metrics dictionary with keys:
            - total_questions: Total number evaluated
            - successful_questions: Number that succeeded
            - failed_questions: Number that failed
            - execution_accuracy: Percentage (0-100)
            - average_latency: Mean seconds per query
            - by_difficulty: Dict of metrics per difficulty level
            - failed_queries: List of failed query details
    """

    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful

    # Calculate overall metrics
    accuracy = (successful / total * 100) if total > 0 else 0.0
    avg_latency = (
        sum(r['latency_seconds'] for r in results) / total
        if total > 0 else 0.0
    )

    # Calculate metrics by difficulty
    by_difficulty = {}
    for difficulty in ['easy', 'medium', 'hard']:
        difficulty_results = [r for r in results if r['difficulty'] == difficulty]

        if len(difficulty_results) == 0:
            continue

        diff_total = len(difficulty_results)
        diff_successful = sum(1 for r in difficulty_results if r['success'])
        diff_accuracy = (diff_successful / diff_total * 100) if diff_total > 0 else 0.0
        diff_latency = (
            sum(r['latency_seconds'] for r in difficulty_results) / diff_total
            if diff_total > 0 else 0.0
        )

        by_difficulty[difficulty] = {
            'total': diff_total,
            'successful': diff_successful,
            'failed': diff_total - diff_successful,
            'accuracy': diff_accuracy,
            'average_latency': diff_latency
        }

    # Get failed query details
    failed_queries = [
        {
            'id': r['id'],
            'question': r['question'],
            'difficulty': r['difficulty'],
            'error_message': r['error_message'],
            'generated_sql': r['generated_sql'][:200] + '...' if len(r['generated_sql']) > 200 else r['generated_sql']
        }
        for r in results if not r['success']
    ]

    return {
        'total_questions': total,
        'successful_questions': successful,
        'failed_questions': failed,
        'execution_accuracy': accuracy,
        'average_latency': avg_latency,
        'by_difficulty': by_difficulty,
        'failed_queries': failed_queries
    }


def print_report(metrics: dict) -> None:
    """
    Print evaluation report to console.

    Displays:
    - Overall statistics
    - Per-difficulty breakdown
    - Failed queries (if any)

    Args:
        metrics: Metrics dictionary from calculate_metrics()
    """

    print("OVERALL RESULTS")
    print("-" * 70)
    print(f"Total Questions:      {metrics['total_questions']}")
    print(f"Successful:           {metrics['successful_questions']}")
    print(f"Failed:               {metrics['failed_questions']}")
    print(f"Execution Accuracy:   {metrics['execution_accuracy']:.1f}%")
    print(f"Average Latency:      {metrics['average_latency']:.2f} seconds")
    print()

    # Per-difficulty breakdown
    print("BREAKDOWN BY DIFFICULTY")
    print("-" * 70)

    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty not in metrics['by_difficulty']:
            continue

        diff_metrics = metrics['by_difficulty'][difficulty]
        print(f"\n{difficulty.upper()}:")
        print(f"  Total:      {diff_metrics['total']}")
        print(f"  Successful: {diff_metrics['successful']}")
        print(f"  Failed:     {diff_metrics['failed']}")
        print(f"  Accuracy:   {diff_metrics['accuracy']:.1f}%")
        print(f"  Avg Latency: {diff_metrics['average_latency']:.2f}s")

    print()

    # Failed queries
    if metrics['failed_queries']:
        print("FAILED QUERIES")
        print("-" * 70)

        for i, failed in enumerate(metrics['failed_queries'], 1):
            print(f"\n{i}. Question #{failed['id']} ({failed['difficulty']})")
            print(f"   Q: {failed['question'][:60]}...")
            print(f"   Error: {failed['error_message'][:100]}...")
            if failed['generated_sql']:
                print(f"   SQL: {failed['generated_sql'][:80]}...")
    else:
        print("No failed queries! 🎉")


def save_report(metrics: dict, results: list[dict]) -> None:
    """
    Save evaluation report to data/evaluation_report.json.

    Saves both metrics summary and detailed results for each question.

    Args:
        metrics: Metrics dictionary from calculate_metrics()
        results: List of evaluation result objects
    """

    project_root = Path(__file__).parent.parent
    cfg = get_config()
    report_path = cfg.get("evaluation", {}).get("report_path", "artifacts/evaluation_report.json")
    json_path = project_root / report_path
    json_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_questions': metrics['total_questions']
        },
        'metrics': metrics,
        'detailed_results': results
    }

    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved to: {json_path}")


if __name__ == '__main__':
    main()
