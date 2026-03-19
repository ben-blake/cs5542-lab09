"""
SQL Generator Agent

This module implements the second agent in the Analytics Copilot pipeline.
It uses Snowflake Cortex LLM to convert natural language questions into
valid SQL queries based on the relevant schema context.

The agent:
1. Takes a natural language question
2. Receives relevant schema context from the Schema Linker
3. Constructs a detailed prompt with schema info and optional examples
4. Calls Snowflake Cortex Complete LLM (llama3.1-70b)
5. Extracts and cleans the generated SQL query
6. Returns production-ready SQL

The agent uses few-shot learning when golden examples are provided,
improving accuracy for domain-specific query patterns.

Example Usage:
    from src.utils.snowflake_conn import get_session
    from src.agents.schema_linker import link_schema
    from src.agents.sql_generator import generate_sql

    session = get_session()
    question = "Show me total revenue by product category"

    # Get relevant schema context
    schema_context = link_schema(session, question, limit=5)

    # Generate SQL query
    sql = generate_sql(session, question, schema_context)
    print(sql)
"""

import re
import warnings
from typing import Any
from snowflake.snowpark import Session
from src.utils.config import get_config


def generate_sql(
    session: Session,
    question: str,
    schema_context: list[dict[str, Any]],
    golden_examples: list[dict[str, str]] = None
) -> str:
    """
    Generate SQL query from natural language using Cortex LLM.

    This function uses Snowflake Cortex Complete with the llama3.1-70b model
    to generate accurate SQL queries. It constructs a detailed prompt that
    includes schema context, Snowflake-specific syntax guidance, and optional
    few-shot examples.

    The generated SQL follows best practices:
    - Uses only provided tables and columns
    - Uses Snowflake SQL syntax (not MySQL/Postgres)
    - Properly qualifies columns with table names when joining
    - Uses uppercase for SQL keywords
    - Includes proper JOINs based on schema relationships

    Args:
        session: Active Snowflake Snowpark Session
        question: User's natural language question about the data
        schema_context: List of relevant tables from schema_linker, each with:
            - table_name: Name of the table
            - columns: List of column metadata (name, type, description, synonyms)
            - relevance_score: How relevant this table is to the question
        golden_examples: Optional list of example question-SQL pairs for
            few-shot learning. Each dict should have 'question' and 'sql' keys.

    Returns:
        str: Generated SQL query (cleaned, without markdown fences).
        Returns empty string if generation fails.

    Raises:
        No exceptions raised. All errors are caught and logged as warnings,
        returning an empty string to allow graceful degradation.

    Example:
        >>> schema = [
        ...     {
        ...         "table_name": "ORDERS",
        ...         "columns": [
        ...             {"column_name": "order_id", "data_type": "NUMBER",
        ...              "description": "Unique order identifier"},
        ...             {"column_name": "total_amount", "data_type": "NUMBER",
        ...              "description": "Order total in dollars"}
        ...         ],
        ...         "relevance_score": 0.95
        ...     }
        ... ]
        >>> sql = generate_sql(session, "What's the average order value?", schema)
        >>> print(sql)
        SELECT AVG(ORDERS.total_amount) AS avg_order_value
        FROM ORDERS

    Notes:
        - Uses llama3.1-70b model via SNOWFLAKE.CORTEX.COMPLETE()
        - Automatically removes markdown code fences (```sql ... ```)
        - Validates that SQL was successfully extracted
        - Returns empty string on any error (check warnings for details)
    """

    # Input validation
    if not question or not question.strip():
        warnings.warn("Empty question provided to SQL generator. Returning empty SQL.")
        return ""

    if not schema_context or len(schema_context) == 0:
        warnings.warn(
            "No schema context provided to SQL generator. "
            "Cannot generate SQL without table information. Returning empty SQL."
        )
        return ""

    try:
        # Build the complete prompt with all components
        prompt = _build_prompt(question, schema_context, golden_examples)

        # Escape single quotes for SQL string interpolation
        escaped_prompt = _escape_sql_string(prompt)

        # Call Snowflake Cortex Complete LLM
        cfg = get_config()
        llm_model = cfg.get("llm", {}).get("model", "llama3.1-70b")
        cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{llm_model}',
            '{escaped_prompt}'
        ) AS generated_sql
        """

        result = session.sql(cortex_query).collect()

        if not result or len(result) == 0:
            warnings.warn("Cortex LLM returned no results. Returning empty SQL.")
            return ""

        # Extract the generated text from the result
        generated_text = result[0]['GENERATED_SQL']

        if not generated_text or not generated_text.strip():
            warnings.warn("Cortex LLM returned empty response. Returning empty SQL.")
            return ""

        # Extract and clean the SQL query
        sql_query = _extract_sql(generated_text)

        if not sql_query or not sql_query.strip():
            warnings.warn(
                f"Failed to extract valid SQL from LLM response. "
                f"Raw response: {generated_text[:200]}... "
                f"Returning empty SQL."
            )
            return ""

        return sql_query

    except Exception as e:
        # Graceful degradation: log error and return empty SQL
        error_msg = str(e)

        # Provide helpful guidance for common errors
        if 'cortex.complete' in error_msg.lower():
            warnings.warn(
                "Cortex Complete function is not available. "
                "Ensure your Snowflake account has Cortex enabled. "
                f"Error: {error_msg}"
            )
        elif 'model not found' in error_msg.lower() or 'llama' in error_msg.lower():
            warnings.warn(
                "Model llama3.1-70b is not available. "
                "Please check available Cortex models in your region. "
                f"Error: {error_msg}"
            )
        else:
            warnings.warn(f"Error in SQL generator: {error_msg}. Returning empty SQL.")

        return ""


def _build_prompt(
    question: str,
    schema_context: list[dict[str, Any]],
    golden_examples: list[dict[str, str]] = None
) -> str:
    """
    Construct the complete LLM prompt with system role, schema, examples, and question.

    The prompt follows a structured format:
    1. System role and instructions
    2. Database schema context
    3. Few-shot examples (if provided)
    4. User's question

    Args:
        question: User's natural language question
        schema_context: List of relevant tables with column metadata
        golden_examples: Optional list of example question-SQL pairs

    Returns:
        str: Complete formatted prompt for the LLM
    """

    # Build list of available table names for the prompt
    available_tables = [t.get('table_name', '') for t in schema_context if t.get('table_name')]
    qualified_tables = [
        f"ANALYTICS_COPILOT.RAW.{t}" if not t.startswith("ANALYTICS_COPILOT") else t
        for t in available_tables
    ]
    available_tables_str = ", ".join(qualified_tables) if qualified_tables else "none"

    # System role and instructions
    system_message = f"""You are a Senior Snowflake Data Engineer. Your task is to generate accurate SQL queries based on user questions.

DATASET CONTEXT:
This database contains the Olist Brazilian E-Commerce dataset with these tables:
ORDERS, CUSTOMERS, ORDER_ITEMS, ORDER_REVIEWS, ORDER_PAYMENTS, PRODUCTS, SELLERS, GEOLOCATION, PRODUCT_CATEGORY_TRANSLATION

Key relationships:
- ORDERS → CUSTOMERS (via CUSTOMER_ID)
- ORDER_ITEMS → ORDERS (via ORDER_ID), PRODUCTS (via PRODUCT_ID), SELLERS (via SELLER_ID)
- ORDER_REVIEWS → ORDERS (via ORDER_ID)
- ORDER_PAYMENTS → ORDERS (via ORDER_ID)
- PRODUCTS → PRODUCT_CATEGORY_TRANSLATION (via PRODUCT_CATEGORY_NAME)
- SELLERS and CUSTOMERS both have CITY and STATE columns

IMPORTANT DATA MODEL NOTE:
- CUSTOMERS.CUSTOMER_ID is a per-order proxy key — each order gets its own CUSTOMER_ID row.
  A single real customer can have multiple CUSTOMER_ID values.
- CUSTOMERS.CUSTOMER_UNIQUE_ID is the true unique customer identifier.
- To count orders per real customer, GROUP BY CUSTOMER_UNIQUE_ID (not CUSTOMER_ID).
  Example: SELECT c.CUSTOMER_UNIQUE_ID, COUNT(o.ORDER_ID) AS order_count
           FROM ANALYTICS_COPILOT.RAW.ORDERS o
           JOIN ANALYTICS_COPILOT.RAW.CUSTOMERS c ON o.CUSTOMER_ID = c.CUSTOMER_ID
           GROUP BY c.CUSTOMER_UNIQUE_ID

CRITICAL RULES:
1. Use ONLY the tables listed in AVAILABLE TABLES below. Do NOT use any other table.
2. ALWAYS use fully qualified table names: ANALYTICS_COPILOT.RAW.<table_name>
   Example: ANALYTICS_COPILOT.RAW.ORDERS (NOT ANALYTICS_COPILOT.RAW.OLIST_ORDERS, NOT just ORDERS)
   Example: ANALYTICS_COPILOT.RAW.ORDER_REVIEWS (NOT ANALYTICS_COPILOT.RAW.REVIEWS)
3. Do NOT invent table names. If the table you need is not in AVAILABLE TABLES, return: "ERROR: Insufficient schema information"
4. Use Snowflake SQL syntax (NOT MySQL, PostgreSQL, or other dialects)
5. Use uppercase for ALL SQL keywords (SELECT, FROM, WHERE, JOIN, GROUP BY, etc.)
6. Always qualify column names with their table name (e.g., ANALYTICS_COPILOT.RAW.ORDERS.order_id)
7. Use proper JOIN syntax based on the key relationships listed above
8. Use appropriate aggregations (SUM, AVG, COUNT, etc.) when asking for totals or averages
9. Return ONLY the SQL query - no explanations, no markdown, no extra text
10. Do NOT use markdown code fences (```sql ... ```)
11. Use standard Snowflake date functions (TO_DATE, DATEADD, DATEDIFF, DATE_TRUNC, etc.)
    DATEDIFF syntax: DATEDIFF('day', start_col, end_col)  -- NOT DATEDIFF(end_col, start_col)
    Example: DATEDIFF('day', order_purchase_timestamp, order_delivered_customer_date)
12. For top-N-per-group queries, use a CTE with ROW_NUMBER() then WHERE row_num <= N (do NOT use QUALIFY with aliases)
13. For month-over-month calculations, aggregate first in a CTE, then apply LAG() in the outer query
14. For questions using superlatives (most, highest, lowest, best, worst, top, bottom) without a specific N, add LIMIT 20
    Example: "which customers placed the most orders" → ORDER BY order_count DESC LIMIT 20
15. Only SELECT the columns directly needed to answer the question — do NOT select all columns from a table.
    For ranking/aggregation questions, select only the grouping identifier(s) and the aggregate metric.
    BAD:  SELECT CUSTOMER_ID, CUSTOMER_UNIQUE_ID, ZIP_CODE, CITY, STATE, COUNT(*) AS order_count ...
    GOOD: SELECT CUSTOMER_UNIQUE_ID, COUNT(*) AS order_count ...

AVAILABLE TABLES (use ONLY these, with ANALYTICS_COPILOT.RAW. prefix):
{available_tables_str}

If the table you need is NOT in the list above, return: "ERROR: Insufficient schema information"
"""

    # Format schema context
    schema_text = _format_schema_context(schema_context)

    # Built-in few-shot examples for complex SQL patterns
    builtin_examples = """EXAMPLE QUERIES (correct SQL patterns to follow):

Example A - Top-N per group (use CTE + WHERE, NOT QUALIFY with aliases):
Question: Find the top 2 sellers by revenue in each state
SQL:
WITH ranked_sellers AS (
    SELECT
        ANALYTICS_COPILOT.RAW.SELLERS.SELLER_STATE,
        ANALYTICS_COPILOT.RAW.SELLERS.SELLER_ID,
        SUM(ANALYTICS_COPILOT.RAW.ORDER_ITEMS.PRICE) AS TOTAL_REVENUE,
        ROW_NUMBER() OVER (PARTITION BY ANALYTICS_COPILOT.RAW.SELLERS.SELLER_STATE ORDER BY SUM(ANALYTICS_COPILOT.RAW.ORDER_ITEMS.PRICE) DESC) AS row_num
    FROM ANALYTICS_COPILOT.RAW.SELLERS
    JOIN ANALYTICS_COPILOT.RAW.ORDER_ITEMS ON ANALYTICS_COPILOT.RAW.SELLERS.SELLER_ID = ANALYTICS_COPILOT.RAW.ORDER_ITEMS.SELLER_ID
    GROUP BY ANALYTICS_COPILOT.RAW.SELLERS.SELLER_STATE, ANALYTICS_COPILOT.RAW.SELLERS.SELLER_ID
)
SELECT SELLER_STATE, SELLER_ID, TOTAL_REVENUE FROM ranked_sellers WHERE row_num <= 2

Example B - Month-over-month growth (aggregate in CTE first, then LAG):
Question: What is the month-over-month growth rate in total payments?
SQL:
WITH monthly_totals AS (
    SELECT
        DATE_TRUNC('MONTH', ANALYTICS_COPILOT.RAW.ORDERS.ORDER_PURCHASE_TIMESTAMP) AS MONTH,
        SUM(ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS.PAYMENT_VALUE) AS TOTAL_PAYMENT
    FROM ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS
    JOIN ANALYTICS_COPILOT.RAW.ORDERS ON ANALYTICS_COPILOT.RAW.ORDER_PAYMENTS.ORDER_ID = ANALYTICS_COPILOT.RAW.ORDERS.ORDER_ID
    GROUP BY DATE_TRUNC('MONTH', ANALYTICS_COPILOT.RAW.ORDERS.ORDER_PURCHASE_TIMESTAMP)
)
SELECT
    MONTH,
    TOTAL_PAYMENT,
    LAG(TOTAL_PAYMENT) OVER (ORDER BY MONTH) AS PREV_MONTH_PAYMENT,
    (TOTAL_PAYMENT - LAG(TOTAL_PAYMENT) OVER (ORDER BY MONTH)) / NULLIF(LAG(TOTAL_PAYMENT) OVER (ORDER BY MONTH), 0) AS GROWTH_RATE
FROM monthly_totals
ORDER BY MONTH
"""

    # Add caller-provided few-shot examples if any
    examples_text = builtin_examples
    if golden_examples and len(golden_examples) > 0:
        examples_text += "\nADDITIONAL EXAMPLES:\n"
        for i, example in enumerate(golden_examples, 1):
            example_question = example.get('question', '')
            example_sql = example.get('sql', '')
            if example_question and example_sql:
                examples_text += f"Example {i}:\nQuestion: {example_question}\nSQL:\n{example_sql}\n\n"

    # Construct the complete prompt
    prompt = f"""{system_message}

DATABASE SCHEMA (column details for the available tables):
{schema_text}

{examples_text}
USER QUESTION:
{question}

Generate the SQL query now (SQL only, no explanations, no markdown):
"""

    return prompt


def _format_schema_context(schema_context: list[dict[str, Any]]) -> str:
    """
    Format the schema context into a readable text representation for the LLM.

    Converts the structured schema metadata into a clear, human-readable format
    that the LLM can easily understand and use for SQL generation.

    Args:
        schema_context: List of tables with column metadata

    Returns:
        str: Formatted schema description

    Example output:
        Table: ORDERS (Relevance: 0.95)
        - order_id (NUMBER): Unique order identifier
        - customer_id (NUMBER): Foreign key to CUSTOMERS table
        - total_amount (NUMBER): Order total in dollars
    """

    schema_lines = []

    for table_info in schema_context:
        table_name = table_info.get('table_name', 'UNKNOWN')
        columns = table_info.get('columns', [])
        relevance = table_info.get('relevance_score', 0.0)

        # Use fully qualified table name so the LLM generates correct SQL
        if not table_name.startswith("ANALYTICS_COPILOT"):
            qualified_name = f"ANALYTICS_COPILOT.RAW.{table_name}"
        else:
            qualified_name = table_name

        # Table header
        schema_lines.append(f"\nTable: {qualified_name} (Relevance: {relevance:.2f})")

        # Column details
        for col in columns:
            col_name = col.get('column_name', 'unknown')
            data_type = col.get('data_type', 'unknown')
            description = col.get('description', 'No description')
            synonyms = col.get('synonyms', '')

            # Build column line
            col_line = f"  - {col_name} ({data_type}): {description}"

            # Add synonyms if available
            if synonyms and synonyms.strip():
                col_line += f" [Synonyms: {synonyms}]"

            schema_lines.append(col_line)

    return "\n".join(schema_lines)


def _extract_sql(response_text: str) -> str:
    """
    Extract and clean SQL query from LLM response.

    The LLM may return:
    - Plain SQL (ideal case)
    - SQL wrapped in markdown code fences (```sql ... ```)
    - SQL with explanatory text before/after
    - Multiple SQL statements (we take the first one)

    This function handles all these cases and returns clean SQL.

    Args:
        response_text: Raw text response from the LLM

    Returns:
        str: Cleaned SQL query, or empty string if extraction fails
    """

    if not response_text:
        return ""

    # Remove leading/trailing whitespace
    text = response_text.strip()

    # Case 1: SQL wrapped in markdown code fences
    # Pattern: ```sql ... ``` or ``` ... ```
    code_fence_pattern = r'```(?:sql)?\s*(.*?)\s*```'
    code_fence_match = re.search(code_fence_pattern, text, re.DOTALL | re.IGNORECASE)

    if code_fence_match:
        sql = code_fence_match.group(1).strip()
        return _clean_sql(sql)

    # Case 2: Look for SQL keywords to identify the query
    # Find the first SELECT, WITH, INSERT, UPDATE, or DELETE statement
    sql_start_pattern = r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b'
    sql_start_match = re.search(sql_start_pattern, text, re.IGNORECASE)

    if sql_start_match:
        # Extract from the first SQL keyword to the end
        sql = text[sql_start_match.start():].strip()

        # Try to find the end of the SQL statement
        # Look for common ending patterns (semicolon, or end of meaningful SQL)
        # If there's explanatory text after, try to remove it

        # Remove trailing explanatory text (common patterns)
        # e.g., "SELECT ... FROM table;\n\nThis query does..."
        sql = re.split(r'\n\n[A-Z]', sql)[0]  # Split on paragraph breaks starting with capital letter
        sql = sql.split('\n\nNote:')[0]  # Remove "Note:" sections
        sql = sql.split('\n\nExplanation:')[0]  # Remove "Explanation:" sections

        return _clean_sql(sql)

    # Case 3: No clear SQL found - return the whole text and let _clean_sql try
    return _clean_sql(text)


def _clean_sql(sql: str) -> str:
    """
    Clean and normalize a SQL query string.

    Operations:
    - Remove markdown code fences
    - Trim whitespace
    - Remove trailing semicolons (Snowflake doesn't require them in Snowpark)
    - Normalize whitespace (collapse multiple spaces/newlines)

    Args:
        sql: Raw SQL string

    Returns:
        str: Cleaned SQL query
    """

    if not sql:
        return ""

    # Remove any remaining markdown code fences
    sql = re.sub(r'```(?:sql)?', '', sql, flags=re.IGNORECASE)

    # Remove trailing semicolons
    sql = sql.rstrip(';').strip()

    # Normalize whitespace (collapse multiple spaces, but preserve line structure)
    sql = re.sub(r'[ \t]+', ' ', sql)  # Multiple spaces/tabs to single space
    sql = re.sub(r'\n\s*\n', '\n', sql)  # Multiple newlines to single newline

    # Final trim
    sql = sql.strip()

    return sql


def _escape_sql_string(text: str) -> str:
    """
    Escape single quotes in text for safe SQL string interpolation.

    Snowflake SQL requires single quotes to be escaped as double single quotes ('').
    This prevents SQL injection and syntax errors when embedding text in SQL strings.

    Args:
        text: Input text that may contain single quotes

    Returns:
        str: Text with single quotes escaped

    Example:
        >>> _escape_sql_string("What's the total revenue?")
        "What''s the total revenue?"
    """
    return text.replace("'", "''")
