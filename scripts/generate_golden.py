"""
Generate Golden Query Benchmarks

This script generates synthetic question-SQL pairs using Snowflake Cortex LLM
for benchmarking the Analytics Copilot. It creates questions of varying difficulty
levels (easy, medium, hard) across all Olist e-commerce tables.

The script:
1. Connects to Snowflake using credentials from .env
2. Retrieves table schemas from INFORMATION_SCHEMA
3. Uses Cortex LLM to generate realistic question-SQL pairs
4. Validates generated SQL using EXPLAIN
5. Writes results to both:
   - METADATA.GOLDEN_QUERIES table (for few-shot learning)
   - data/golden_queries.json file (for evaluation script)

Usage:
    python scripts/generate_golden.py [--count N] [--verify]

    Options:
        --count N    Number of questions to generate (default: 50)
        --verify     Verify generated SQL by executing it (default: False)

Example:
    python scripts/generate_golden.py --count 60 --verify
"""

import sys
import os
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Add project root to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.snowflake_conn import get_session, close_session
from src.utils.config import get_config


def main():
    """Main entry point for golden query generation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate golden query benchmarks using Cortex LLM'
    )
    cfg = get_config()
    default_count = cfg.get("evaluation", {}).get("default_count", 50)
    parser.add_argument(
        '--count',
        type=int,
        default=default_count,
        help=f'Number of questions to generate (default: {default_count})'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify generated SQL by executing it (validates results)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.count < 1:
        print(f"Error: --count must be at least 1 (got {args.count})")
        sys.exit(1)

    print("="*70)
    print("GOLDEN QUERY GENERATION")
    print("="*70)
    print(f"Target count: {args.count} questions")
    print(f"Verification: {'Enabled' if args.verify else 'Disabled'}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()

    try:
        # Connect to Snowflake
        print("Connecting to Snowflake...")
        session = get_session()
        print("✓ Connected successfully\n")

        # Get table schemas
        print("Retrieving table schemas from Olist database...")
        table_schemas = get_table_schemas(session)
        print(f"✓ Found {len(table_schemas)} tables\n")

        # Display table summary
        print("Tables available for query generation:")
        for table_name in sorted(table_schemas.keys()):
            col_count = len(table_schemas[table_name])
            print(f"  - {table_name} ({col_count} columns)")
        print()

        # Calculate questions per difficulty level from config
        dist = cfg.get("evaluation", {}).get("question_distribution", {})
        easy_pct = dist.get("easy", 0.4)
        medium_pct = dist.get("medium", 0.4)
        easy_count = int(args.count * easy_pct)
        medium_count = int(args.count * medium_pct)
        hard_count = args.count - easy_count - medium_count

        print(f"Question distribution:")
        print(f"  - Easy: {easy_count} questions (simple SELECT, WHERE, ORDER BY)")
        print(f"  - Medium: {medium_count} questions (JOINs, GROUP BY, aggregations)")
        print(f"  - Hard: {hard_count} questions (complex analytics, window functions, CTEs)")
        print()

        # Generate questions for each difficulty level
        all_questions = []

        print("Generating questions...")
        print("-" * 70)

        # Easy questions
        print(f"\nGenerating {easy_count} EASY questions...")
        easy_questions = generate_questions(
            session, table_schemas, 'easy', easy_count, args.verify
        )
        all_questions.extend(easy_questions)
        print(f"✓ Generated {len(easy_questions)} easy questions")

        # Medium questions
        print(f"\nGenerating {medium_count} MEDIUM questions...")
        medium_questions = generate_questions(
            session, table_schemas, 'medium', medium_count, args.verify
        )
        all_questions.extend(medium_questions)
        print(f"✓ Generated {len(medium_questions)} medium questions")

        # Hard questions
        print(f"\nGenerating {hard_count} HARD questions...")
        hard_questions = generate_questions(
            session, table_schemas, 'hard', hard_count, args.verify
        )
        all_questions.extend(hard_questions)
        print(f"✓ Generated {len(hard_questions)} hard questions")

        print("-" * 70)
        print(f"\nTotal questions generated: {len(all_questions)}")
        print()

        # Save to Snowflake table
        print("Saving to METADATA.GOLDEN_QUERIES table...")
        save_to_snowflake(session, all_questions)
        print("✓ Saved to Snowflake")

        # Save to JSON file
        print("\nSaving to data/golden_queries.json file...")
        save_to_json(all_questions)
        print("✓ Saved to JSON file")

        # Print summary statistics
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Total questions: {len(all_questions)}")
        print(f"  Easy: {sum(1 for q in all_questions if q['difficulty'] == 'easy')}")
        print(f"  Medium: {sum(1 for q in all_questions if q['difficulty'] == 'medium')}")
        print(f"  Hard: {sum(1 for q in all_questions if q['difficulty'] == 'hard')}")

        if args.verify:
            verified_count = sum(1 for q in all_questions if q['verified'])
            print(f"\nVerified queries: {verified_count}/{len(all_questions)}")

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user. Exiting...")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Always close the session
        print("\nClosing Snowflake connection...")
        close_session()


def get_table_schemas(session) -> dict:
    """
    Retrieve schema information for all Olist tables.

    Queries INFORMATION_SCHEMA to get column names and data types for all
    tables in the RAW schema. This information is used to construct prompts
    for the LLM to generate realistic SQL queries.

    Args:
        session: Active Snowflake Snowpark Session

    Returns:
        dict: Mapping of table_name to list of column dicts:
        {
            "ORDERS": [
                {"name": "order_id", "type": "VARCHAR"},
                {"name": "customer_id", "type": "VARCHAR"},
                ...
            ],
            ...
        }
    """

    query = """
    SELECT
        table_name,
        column_name,
        data_type
    FROM ANALYTICS_COPILOT.INFORMATION_SCHEMA.COLUMNS
    WHERE table_schema = 'RAW'
    ORDER BY table_name, ordinal_position
    """

    result = session.sql(query).collect()

    # Group columns by table
    schemas = {}
    for row in result:
        table_name = row['TABLE_NAME']
        col_name = row['COLUMN_NAME']
        data_type = row['DATA_TYPE']

        if table_name not in schemas:
            schemas[table_name] = []

        schemas[table_name].append({
            'name': col_name,
            'type': data_type
        })

    return schemas


def generate_questions(
    session,
    table_schemas: dict,
    difficulty: str,
    count: int,
    verify: bool
) -> list[dict]:
    """
    Generate question-SQL pairs for a specific difficulty level.

    Uses Snowflake Cortex LLM to generate realistic business questions and
    corresponding SQL queries. The LLM is prompted with table schemas and
    examples of the desired difficulty level.

    Args:
        session: Active Snowflake Snowpark Session
        table_schemas: Dictionary of table schemas from get_table_schemas()
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        count: Number of questions to generate
        verify: If True, validates SQL using EXPLAIN and optionally executes

    Returns:
        list[dict]: List of question objects with keys:
            - id: Sequential ID starting from 1
            - question: Natural language question
            - sql_query: Generated SQL query
            - difficulty: Difficulty level
            - tables_used: Comma-separated list of tables
            - verified: Boolean indicating if SQL was validated
    """

    questions = []

    # Generate questions in batches of 5 to avoid timeout
    batch_size = 5
    batches = (count + batch_size - 1) // batch_size

    for batch_num in range(batches):
        questions_in_batch = min(batch_size, count - len(questions))

        print(f"  Batch {batch_num + 1}/{batches}: Generating {questions_in_batch} questions...")

        try:
            # Build prompt for this difficulty level
            prompt = build_generation_prompt(table_schemas, difficulty, questions_in_batch)

            # Escape single quotes for SQL
            escaped_prompt = prompt.replace("'", "''")

            # Call Cortex LLM
            cfg = get_config()
            llm_model = cfg.get("llm", {}).get("model", "llama3.1-70b")
            cortex_query = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                '{llm_model}',
                '{escaped_prompt}'
            ) AS response
            """

            result = session.sql(cortex_query).collect()

            if not result or len(result) == 0:
                warnings.warn(f"Cortex returned no results for {difficulty} batch {batch_num + 1}")
                continue

            response_text = result[0]['RESPONSE']

            # Parse the JSON response
            batch_questions = parse_llm_response(response_text, difficulty, verify, session)

            # Add sequential IDs
            for q in batch_questions:
                q['id'] = len(questions) + 1
                questions.append(q)

            print(f"    ✓ Generated {len(batch_questions)} questions")

        except Exception as e:
            warnings.warn(f"Error generating batch {batch_num + 1}: {str(e)}")
            continue

    return questions


def build_generation_prompt(table_schemas: dict, difficulty: str, count: int) -> str:
    """
    Build the LLM prompt for generating questions.

    Creates a detailed prompt that includes:
    - System instructions
    - Table schemas
    - Examples for the difficulty level
    - Output format specification

    Args:
        table_schemas: Dictionary of table schemas
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        count: Number of questions to generate

    Returns:
        str: Complete prompt for the LLM
    """

    # Format table schemas
    schema_text = "AVAILABLE TABLES AND COLUMNS:\n\n"
    for table_name, columns in table_schemas.items():
        schema_text += f"Table: RAW.{table_name}\n"
        for col in columns:
            schema_text += f"  - {col['name']} ({col['type']})\n"
        schema_text += "\n"

    # Difficulty-specific instructions
    if difficulty == 'easy':
        difficulty_instructions = """
EASY QUESTIONS - Requirements:
- Use single table queries only (no JOINs)
- Simple SELECT with WHERE, ORDER BY, LIMIT
- Basic aggregations (COUNT, SUM, AVG, MAX, MIN)
- Simple filtering conditions
- No complex logic or window functions

Examples:
1. "How many orders were placed in 2017?"
2. "What is the average product weight?"
3. "Show the top 10 most expensive products by price"
4. "How many customers are from São Paulo?"
5. "What is the total payment value across all orders?"
"""
    elif difficulty == 'medium':
        difficulty_instructions = """
MEDIUM QUESTIONS - Requirements:
- Use JOINs between 2-3 tables
- GROUP BY with aggregations
- Multiple filtering conditions
- Subqueries allowed
- Date functions and calculations
- String functions

Examples:
1. "What is the total revenue by product category?"
2. "Show the top 5 customers by total spending with their city"
3. "Calculate the average delivery time by order status"
4. "Which sellers have the highest average review scores?"
5. "What percentage of orders were delivered late by month?"
"""
    else:  # hard
        difficulty_instructions = """
HARD QUESTIONS - Requirements:
- Complex JOINs across 3+ tables
- Window functions (ROW_NUMBER, RANK, LAG, LEAD)
- CTEs (WITH clauses)
- Complex aggregations and calculations
- Multiple nested subqueries
- Advanced date/time logic
- CASE statements with complex logic

Examples:
1. "Find the month-over-month growth rate in revenue by product category"
2. "Rank sellers by total sales with a running total per state"
3. "Calculate the customer lifetime value (CLV) and rank customers"
4. "Find products that had increased sales in each consecutive quarter"
5. "Identify the top 3 products per category based on review score and sales volume"
"""

    prompt = f"""You are a data analyst generating realistic business questions and SQL queries for the Olist Brazilian e-commerce dataset.

{schema_text}

{difficulty_instructions}

CRITICAL INSTRUCTIONS:
1. Generate EXACTLY {count} unique, realistic business questions
2. Each question should have a corresponding valid Snowflake SQL query
3. Use proper Snowflake SQL syntax (NOT MySQL or PostgreSQL)
4. Use uppercase for SQL keywords (SELECT, FROM, WHERE, etc.)
5. Qualify column names with table aliases when using JOINs
6. All table names must be prefixed with RAW schema (e.g., RAW.ORDERS)
7. Questions should be realistic business questions that a user would actually ask
8. Ensure variety in the types of questions (don't repeat similar patterns)

OUTPUT FORMAT:
Return a JSON array with exactly {count} objects, each with this structure:
{{
  "question": "Natural language question",
  "sql_query": "SELECT ... FROM RAW.table_name ...",
  "tables_used": "TABLE1,TABLE2"
}}

IMPORTANT: Return ONLY the JSON array, no explanations, no markdown, no extra text.

Generate {count} {difficulty} questions now:
"""

    return prompt


def parse_llm_response(
    response_text: str,
    difficulty: str,
    verify: bool,
    session
) -> list[dict]:
    """
    Parse LLM response and extract question-SQL pairs.

    Extracts JSON from the LLM response, validates the structure, and
    optionally verifies that the SQL is valid using EXPLAIN.

    Args:
        response_text: Raw text response from Cortex LLM
        difficulty: Difficulty level for the questions
        verify: If True, validate SQL using EXPLAIN
        session: Snowflake session (needed if verify=True)

    Returns:
        list[dict]: Parsed and validated question objects
    """

    # Try to extract JSON from response
    # The LLM might wrap it in markdown code fences
    import re

    # Remove markdown code fences if present
    json_text = response_text.strip()
    json_text = re.sub(r'```json\s*', '', json_text)
    json_text = re.sub(r'```\s*', '', json_text)
    json_text = json_text.strip()

    # Find JSON array
    # Look for patterns like [{ ... }]
    array_match = re.search(r'\[.*\]', json_text, re.DOTALL)
    if array_match:
        json_text = array_match.group(0)

    try:
        questions_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        warnings.warn(f"Failed to parse JSON response: {str(e)}")
        return []

    if not isinstance(questions_data, list):
        warnings.warn("Response is not a JSON array")
        return []

    # Process each question
    validated_questions = []

    for item in questions_data:
        if not isinstance(item, dict):
            continue

        question = item.get('question', '').strip()
        sql_query = item.get('sql_query', '').strip()
        tables_used = item.get('tables_used', '').strip()

        if not question or not sql_query:
            continue

        # Build question object
        question_obj = {
            'question': question,
            'sql_query': sql_query,
            'difficulty': difficulty,
            'tables_used': tables_used,
            'verified': False
        }

        # Optionally verify the SQL
        if verify:
            is_valid = verify_sql(session, sql_query)
            question_obj['verified'] = is_valid

            if not is_valid:
                # Skip invalid queries
                continue

        validated_questions.append(question_obj)

    return validated_questions


def verify_sql(session, sql: str) -> bool:
    """
    Verify that SQL is valid using EXPLAIN command.

    Args:
        session: Active Snowflake Snowpark Session
        sql: SQL query to verify

    Returns:
        bool: True if SQL is valid, False otherwise
    """

    try:
        explain_query = f"EXPLAIN {sql}"
        session.sql(explain_query).collect()
        return True
    except Exception:
        return False


def save_to_snowflake(session, questions: list[dict]) -> None:
    """
    Save generated questions to METADATA.GOLDEN_QUERIES table.

    Truncates the existing table and inserts all questions.

    Args:
        session: Active Snowflake Snowpark Session
        questions: List of question objects to insert
    """

    # Truncate existing data
    session.sql("TRUNCATE TABLE METADATA.GOLDEN_QUERIES").collect()

    # Insert questions one by one
    for q in questions:
        # Escape single quotes in strings
        question_escaped = q['question'].replace("'", "''")
        sql_escaped = q['sql_query'].replace("'", "''")
        tables_escaped = q['tables_used'].replace("'", "''")

        insert_query = f"""
        INSERT INTO METADATA.GOLDEN_QUERIES
        (QUESTION, SQL_QUERY, DIFFICULTY, TABLES_USED, VERIFIED)
        VALUES (
            '{question_escaped}',
            '{sql_escaped}',
            '{q['difficulty']}',
            '{tables_escaped}',
            {str(q['verified']).upper()}
        )
        """

        session.sql(insert_query).collect()


def save_to_json(questions: list[dict]) -> None:
    """
    Save questions to data/golden_queries.json file.

    Creates the data directory if it doesn't exist and writes the questions
    as a formatted JSON array.

    Args:
        questions: List of question objects to save
    """

    # Get project root and data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)

    # Write to JSON file
    json_path = data_dir / 'golden_queries.json'

    with open(json_path, 'w') as f:
        json.dump(questions, f, indent=2)

    print(f"  File location: {json_path}")


if __name__ == '__main__':
    main()
