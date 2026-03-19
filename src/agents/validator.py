"""
Validator Agent

This module implements the third agent in the Analytics Copilot pipeline.
It validates SQL queries using Snowflake's EXPLAIN command and implements
self-correction through retry loops when validation fails.

The agent:
1. Takes a generated SQL query from sql_generator
2. Validates it using EXPLAIN to check for syntax/semantic errors
3. If valid, executes the query and returns results
4. If invalid, captures the error and retries with self-correction
5. Uses a retry loop (max 3 attempts) to fix errors automatically
6. Returns either successful results or final error message

The self-correction mechanism provides error feedback to the SQL generator,
allowing it to learn from mistakes and fix issues like:
- Syntax errors
- Invalid table/column references
- Type mismatches
- Join errors
- Aggregation issues

Example Usage:
    from src.utils.snowflake_conn import get_session
    from src.agents.schema_linker import link_schema
    from src.agents.sql_generator import generate_sql
    from src.agents.validator import validate_and_execute

    session = get_session()
    question = "Show me total revenue by product category"

    # Get schema context and generate SQL
    schema_context = link_schema(session, question, limit=5)
    sql = generate_sql(session, question, schema_context)

    # Validate and execute with auto-correction
    final_sql, result = validate_and_execute(
        session, sql, question, schema_context, max_retries=3
    )

    # Check if successful
    if isinstance(result, str):
        print(f"Error: {result}")
    else:
        # result is a Snowpark DataFrame
        result.show()
"""

import warnings
from typing import Any, Optional
from snowflake.snowpark import Session, DataFrame
from .sql_generator import generate_sql


def validate_and_execute(
    session: Session,
    sql: str,
    question: str,
    schema_context: list[dict],
    max_retries: int = 3
) -> tuple[str, Any]:
    """
    Validate SQL using EXPLAIN, execute if valid, retry with error feedback if not.

    This function implements a self-correction loop that:
    1. Validates SQL using Snowflake EXPLAIN command
    2. If validation passes, executes the query and returns results
    3. If validation fails, provides error feedback to sql_generator
    4. Retries up to max_retries times with error context
    5. Returns results or final error message

    The validation approach uses EXPLAIN to catch errors without executing
    expensive queries, saving compute costs and preventing data modifications
    from invalid queries.

    Args:
        session: Active Snowflake Snowpark Session
        sql: Generated SQL query to validate (from sql_generator)
        question: Original user question (used for retry context)
        schema_context: Schema context from schema_linker (used for retries)
        max_retries: Maximum retry attempts (default 3). Set to 0 to disable
            self-correction and only try once.

    Returns:
        tuple[str, Any]: A tuple containing:
            - final_sql (str): The final SQL query attempted (may be different
              from input if retries occurred)
            - result (DataFrame or str): If successful, returns Snowpark DataFrame
              with query results. If all retries failed, returns error message string.

    Example:
        >>> sql, result = validate_and_execute(session, sql_query, question, schema)
        >>> if isinstance(result, str):
        ...     print(f"Failed: {result}")
        ... else:
        ...     result.show()  # Snowpark DataFrame

    Notes:
        - Uses EXPLAIN to validate without executing (fast, safe)
        - Automatically retries with error feedback on failures
        - Each retry gets the error message from previous attempt
        - Returns DataFrame on success, error string on failure
        - Logs all validation attempts and errors for debugging
    """

    # Input validation
    if not sql or not sql.strip():
        error_msg = "Empty SQL query provided to validator. Cannot validate empty query."
        warnings.warn(error_msg)
        return sql, error_msg

    if max_retries < 0:
        warnings.warn(f"Invalid max_retries value {max_retries}. Using 0 (no retries).")
        max_retries = 0

    print(f"\n{'='*60}")
    print(f"VALIDATOR: Starting SQL validation and execution")
    print(f"Max retries: {max_retries}")
    print(f"{'='*60}\n")

    # Track the current SQL and attempt number
    current_sql = sql
    attempt = 0
    last_error = None

    # Main validation and retry loop
    while attempt <= max_retries:
        attempt_num = attempt + 1
        print(f"Attempt {attempt_num}/{max_retries + 1}: Validating SQL...")

        # Validate the SQL using EXPLAIN
        is_valid, error_message = _validate_sql(session, current_sql)

        if is_valid:
            print(f"✓ Validation passed! SQL is syntactically and semantically correct.")
            print(f"Executing query...")

            # Execute the validated SQL
            result_df, exec_error = _execute_sql(session, current_sql)

            if exec_error is None:
                print(f"✓ Execution successful!")
                print(f"{'='*60}\n")
                return current_sql, result_df
            else:
                # Execution failed even though EXPLAIN passed
                # This is rare but can happen with runtime errors
                print(f"✗ Execution failed despite passing validation!")
                print(f"Error: {exec_error}")
                last_error = f"Execution error: {exec_error}"

                # If we have retries left, try to fix it
                if attempt < max_retries:
                    print(f"Attempting to fix execution error...")
                    current_sql = _retry_with_error_feedback(
                        session, question, schema_context, exec_error, current_sql
                    )
                    attempt += 1
                else:
                    print(f"No retries remaining. Returning error.")
                    print(f"{'='*60}\n")
                    return current_sql, last_error
        else:
            # Validation failed
            print(f"✗ Validation failed!")
            print(f"Error: {error_message}")
            last_error = error_message

            # If we have retries left, regenerate SQL with error feedback
            if attempt < max_retries:
                print(f"Retrying with error feedback...")
                current_sql = _retry_with_error_feedback(
                    session, question, schema_context, error_message, current_sql
                )
                attempt += 1
            else:
                # No more retries
                print(f"Max retries ({max_retries}) reached. Returning error.")
                print(f"{'='*60}\n")
                return current_sql, last_error

    # This should not be reached, but handle it just in case
    print(f"Unexpected loop exit. Returning last error.")
    print(f"{'='*60}\n")
    return current_sql, last_error or "Unknown error occurred during validation"


def _validate_sql(session: Session, sql: str) -> tuple[bool, Optional[str]]:
    """
    Validate SQL query using Snowflake EXPLAIN command.

    EXPLAIN analyzes the query without executing it, checking for:
    - Syntax errors
    - Invalid table or column references
    - Type mismatches
    - Permission issues
    - Semantic errors

    This is much faster and safer than executing the query directly,
    especially for expensive queries or queries that modify data.

    Args:
        session: Active Snowflake Snowpark Session
        sql: SQL query to validate

    Returns:
        tuple[bool, Optional[str]]:
            - is_valid: True if EXPLAIN succeeded, False otherwise
            - error_message: None if valid, error description if invalid

    Example:
        >>> is_valid, error = _validate_sql(session, "SELECT * FROM orders")
        >>> if is_valid:
        ...     print("Query is valid!")
        ... else:
        ...     print(f"Error: {error}")
    """

    try:
        # Use EXPLAIN to validate without executing
        explain_query = f"EXPLAIN {sql}"
        session.sql(explain_query).collect()

        # If we got here, EXPLAIN succeeded
        return True, None

    except Exception as e:
        # EXPLAIN failed - capture the error message
        error_msg = str(e)

        # Clean up the error message for better readability
        # Snowflake errors often contain stack traces and internal details
        # Extract the most relevant part
        error_msg = _extract_error_message(error_msg)

        return False, error_msg


def _execute_sql(session: Session, sql: str) -> tuple[Optional[DataFrame], Optional[str]]:
    """
    Execute validated SQL query and return results.

    This function should only be called after SQL has been validated with EXPLAIN.
    It executes the query and returns the results as a Snowpark DataFrame.

    Args:
        session: Active Snowflake Snowpark Session
        sql: Validated SQL query to execute

    Returns:
        tuple[Optional[DataFrame], Optional[str]]:
            - result_df: Snowpark DataFrame with query results, or None if error
            - error_message: None if successful, error description if failed

    Example:
        >>> df, error = _execute_sql(session, "SELECT COUNT(*) FROM orders")
        >>> if error is None:
        ...     df.show()
        ... else:
        ...     print(f"Error: {error}")

    Notes:
        - Returns DataFrame object (not collected) for lazy evaluation
        - Caller can call .collect(), .show(), or .to_pandas() as needed
        - This allows the caller to control when data is fetched
    """

    try:
        # Execute the SQL query
        result_df = session.sql(sql)

        # Return the DataFrame object (not collected yet)
        # This allows lazy evaluation and gives caller control
        return result_df, None

    except Exception as e:
        # Execution failed - capture the error
        error_msg = str(e)
        error_msg = _extract_error_message(error_msg)

        return None, error_msg


def _retry_with_error_feedback(
    session: Session,
    question: str,
    schema_context: list[dict],
    error_message: str,
    previous_sql: str
) -> str:
    """
    Retry SQL generation with error feedback for self-correction.

    This function implements the self-correction mechanism by:
    1. Taking the error message from the previous attempt
    2. Appending it to the original question with context
    3. Calling sql_generator again with the enriched prompt
    4. Returning the new SQL query to try

    The enriched prompt helps the LLM understand what went wrong and
    fix the issue in the next attempt. This is similar to how a human
    developer would debug SQL - see the error, understand it, fix it.

    Args:
        session: Active Snowflake Snowpark Session
        question: Original user question
        schema_context: Schema context from schema_linker
        error_message: Error message from previous validation/execution
        previous_sql: The SQL query that failed (for context)

    Returns:
        str: New SQL query generated with error feedback

    Notes:
        - Enhances the question with error context
        - Provides both the error and the failed SQL
        - Allows the LLM to learn from mistakes
        - May return empty string if regeneration fails
    """

    # Build an enriched question with error feedback
    # This gives the LLM context about what went wrong
    enriched_question = f"""{question}

PREVIOUS ATTEMPT FAILED WITH ERROR:
{error_message}

FAILED SQL QUERY:
{previous_sql}

Please fix the SQL query to avoid this error. Review the schema carefully and ensure:
- All table names are correct and exist in the schema
- All column names are correct and exist in their respective tables
- Column references are properly qualified with table names in JOINs
- Data types are compatible in comparisons and operations
- Aggregation functions are used correctly
- JOIN conditions reference valid foreign key relationships
"""

    print(f"\nRegenerating SQL with error feedback...")
    print(f"Error provided to LLM: {error_message[:100]}...")

    # Call sql_generator with the enriched question
    # The schema_context is the same, but the question now includes error feedback
    new_sql = generate_sql(session, enriched_question, schema_context)

    if not new_sql or not new_sql.strip():
        # Regeneration failed - return the previous SQL to avoid infinite loops
        warnings.warn(
            "SQL regeneration returned empty query. "
            "This may indicate the LLM cannot fix the error. "
            "Returning previous SQL."
        )
        return previous_sql

    print(f"New SQL generated with corrections")
    return new_sql


def _extract_error_message(error_text: str) -> str:
    """
    Extract the most relevant part of a Snowflake error message.

    Snowflake errors often include:
    - Stack traces
    - Internal error codes
    - Connection details
    - Timestamps

    This function extracts the human-readable error description
    while removing noise that doesn't help with debugging.

    Args:
        error_text: Raw error message from Snowflake

    Returns:
        str: Cleaned, relevant error message

    Example:
        >>> raw = "Error: SQL compilation error:\\nline 1 at position 14\\ninvalid identifier 'INVALID_COL'"
        >>> _extract_error_message(raw)
        "SQL compilation error: invalid identifier 'INVALID_COL'"
    """

    if not error_text:
        return "Unknown error (empty error message)"

    # Common patterns in Snowflake errors we want to extract
    # 1. SQL compilation error
    # 2. Execution error
    # 3. Invalid object name
    # 4. Type mismatch

    # Take the first line if it contains "error"
    lines = error_text.split('\n')
    for line in lines:
        if 'error' in line.lower():
            # Clean up the line
            cleaned = line.strip()

            # Remove common prefixes
            prefixes_to_remove = [
                'Error: ',
                'SnowflakeError: ',
                'Exception: ',
            ]

            for prefix in prefixes_to_remove:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]

            return cleaned

    # If no error keyword found, return first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()

    # Fallback: return first 200 chars of error
    return error_text[:200].strip()
