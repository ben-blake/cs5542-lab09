"""
Schema Linker Agent

This module implements the first agent in the Analytics Copilot pipeline.
It uses Snowflake Cortex Search (RAG) to find the most relevant database tables
and columns for a given user question.

The agent:
1. Takes a natural language question
2. Queries the Cortex Search Service (semantic search over table metadata)
3. Groups results by table name
4. Calculates relevance scores
5. Returns the top-k most relevant tables with their column metadata

This RAG-based approach ensures the SQL Generator receives only the relevant
schema context, improving both accuracy and token efficiency.

Example Usage:
    from src.utils.snowflake_conn import get_session
    from src.agents.schema_linker import link_schema

    session = get_session()
    question = "Show me total revenue by product category"
    relevant_tables = link_schema(session, question, limit=5)

    for table in relevant_tables:
        print(f"{table['table_name']}: {table['relevance_score']:.2f}")
"""

import json
import warnings
from typing import Any
from snowflake.snowpark import Session
from src.utils.config import get_config


def link_schema(session: Session, question: str, limit: int = None) -> list[dict[str, Any]]:
    """
    Find relevant database tables for a given question using Cortex Search.

    This function implements RAG (Retrieval-Augmented Generation) over the
    database schema metadata. It uses Snowflake Cortex Search Service to
    perform semantic similarity search between the user's question and
    table/column descriptions.

    The search service indexes the TABLE_DESCRIPTIONS table, which contains
    business-friendly descriptions, synonyms, and metadata for all columns
    across all tables in the database.

    Args:
        session: Active Snowflake Snowpark Session
        question: User's natural language question about the data
        limit: Maximum number of tables to return (default 5)

    Returns:
        List of dicts with table metadata, sorted by relevance score (highest first):
        [
            {
                "table_name": "ORDERS",
                "columns": [
                    {
                        "column_name": "order_id",
                        "description": "Unique identifier for each order",
                        "data_type": "VARCHAR",
                        "synonyms": "order number, order ID"
                    },
                    ...
                ],
                "relevance_score": 0.95
            },
            ...
        ]

        Returns empty list if:
        - Cortex Search Service doesn't exist
        - No relevant tables found
        - An error occurs during search

    Raises:
        No exceptions raised. All errors are caught and logged as warnings,
        returning an empty list to allow graceful degradation.

    Notes:
        - Query multiplier: Retrieves (limit * 10) columns to ensure we get
          multiple columns per table before grouping
        - Relevance scores are averaged across all columns in a table
        - Requires SCHEMA_SEARCH_SERVICE to be created in METADATA schema
        - Requires TABLE_DESCRIPTIONS table to be populated
    """

    # Load defaults from config
    if limit is None:
        cfg = get_config()
        limit = cfg.get("schema_linker", {}).get("limit", 5)

    # Input validation
    if not question or not question.strip():
        warnings.warn("Empty question provided to schema linker. Returning empty results.")
        return []

    if limit < 1:
        warnings.warn(f"Invalid limit ({limit}). Using default limit of 5.")
        limit = 5

    try:
        # Build the Cortex Search query
        # We retrieve more columns than tables (limit * multiplier) because:
        # - Multiple columns belong to the same table
        # - We want comprehensive coverage of each relevant table
        # - We'll group by table_name and calculate aggregate scores
        cfg = get_config()
        search_multiplier = cfg.get("schema_linker", {}).get("search_multiplier", 10)
        search_limit = limit * search_multiplier

        # SNOWFLAKE.CORTEX.SEARCH_PREVIEW takes 2 args:
        # 1. service name
        # 2. JSON string with "query", "columns" to return, and "limit"
        search_request = json.dumps({
            "query": question,
            "columns": ["table_name", "column_name", "description", "synonyms", "data_type", "sample_values"],
            "limit": search_limit
        })
        # Escape single quotes in the JSON string for SQL embedding
        search_request_escaped = search_request.replace("'", "''")

        search_query = f"""
        SELECT
            SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                'ANALYTICS_COPILOT.METADATA.SCHEMA_SEARCH_SERVICE',
                '{search_request_escaped}'
            ) AS search_results
        """

        # Execute the search query
        result = session.sql(search_query).collect()

        if not result or len(result) == 0:
            warnings.warn(f"Cortex Search returned no results for question: {question}")
            return []

        # Parse the JSON response
        # The result is a single row with a JSON column
        search_results_json = result[0]['SEARCH_RESULTS']
        search_data = json.loads(search_results_json)

        # Extract results array from the response
        # Response format: {"results": [...], "request_id": "..."}
        if 'results' not in search_data or len(search_data['results']) == 0:
            warnings.warn(f"No relevant tables found for question: {question}")
            return []

        results = search_data['results']

        # Group columns by table_name
        # Each result contains column-level metadata + relevance score
        tables_dict: dict[str, dict[str, Any]] = {}

        for result_item in results:
            # Extract fields from the search result
            # The 'score' field represents semantic similarity (0-1)
            table_name = result_item.get('table_name', 'UNKNOWN')
            column_name = result_item.get('column_name', 'UNKNOWN')
            description = result_item.get('description', '')
            data_type = result_item.get('data_type', 'UNKNOWN')
            synonyms = result_item.get('synonyms', '')
            relevance_score = result_item.get('score', 0.0)

            # Initialize table entry if not seen before
            if table_name not in tables_dict:
                tables_dict[table_name] = {
                    'table_name': table_name,
                    'columns': [],
                    'relevance_scores': []  # Temporary list for averaging
                }

            # Add column metadata
            tables_dict[table_name]['columns'].append({
                'column_name': column_name,
                'description': description,
                'data_type': data_type,
                'synonyms': synonyms
            })

            # Track relevance score for averaging
            tables_dict[table_name]['relevance_scores'].append(relevance_score)

        # Calculate average relevance score per table
        # Tables with more highly-relevant columns will rank higher
        for table_data in tables_dict.values():
            scores = table_data['relevance_scores']
            table_data['relevance_score'] = sum(scores) / len(scores) if scores else 0.0
            # Remove temporary scores list
            del table_data['relevance_scores']

        # Convert to list and sort by relevance score (descending)
        tables_list = list(tables_dict.values())
        tables_list.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Filter to avoid mixing incompatible datasets in the same context
        tables_list = _filter_dataset_mixing(tables_list)

        # Take top 'limit' tables, then supplement with FK partners
        result = tables_list[:limit]
        result = _supplement_related_tables(session, result, limit)

        return result

    except Exception as e:
        error_msg = str(e)

        if 'does not exist' in error_msg.lower() and 'schema_search_service' in error_msg.lower():
            warnings.warn(
                "Cortex Search Service not found. Falling back to keyword search over TABLE_DESCRIPTIONS."
            )
            return _fallback_keyword_search(session, question, limit)
        elif 'table_descriptions' in error_msg.lower():
            warnings.warn(
                "TABLE_DESCRIPTIONS table is missing or empty. "
                "Please run scripts/build_metadata.py to populate it. "
                "Returning empty results."
            )
        else:
            warnings.warn(f"Error in schema linker: {error_msg}. Falling back to keyword search.")
            return _fallback_keyword_search(session, question, limit)

        return []


def _fallback_keyword_search(session, question: str, limit: int) -> list[dict]:
    """
    Fallback when Cortex Search is unavailable.

    Queries TABLE_DESCRIPTIONS directly using ILIKE keyword matching.
    If TABLE_DESCRIPTIONS is empty or missing, returns all RAW tables.
    """
    # Common English stopwords to skip — these don't help find relevant tables
    _STOPWORDS = {
        'what', 'which', 'where', 'when', 'who', 'how', 'many', 'much',
        'have', 'been', 'that', 'this', 'with', 'from', 'they', 'each',
        'highest', 'lowest', 'average', 'total', 'count', 'find', 'show',
        'list', 'give', 'most', 'least', 'more', 'than', 'over', 'last',
        'first', 'were', 'does', 'also', 'into', 'some', 'only', 'their',
        'there', 'these', 'those', 'very', 'just', 'been', 'will', 'would',
        'could', 'should', 'across', 'between', 'within', 'using', 'based',
    }

    try:
        # Extract meaningful keywords: words >3 chars that are not stopwords
        raw_words = [w.strip("'\"?,") for w in question.lower().split()]
        keywords = [w for w in raw_words if len(w) > 3 and w not in _STOPWORDS]
        if not keywords:
            keywords = [w for w in raw_words if len(w) > 3]
        if not keywords:
            keywords = raw_words

        # Build ILIKE conditions for all meaningful keywords (no arbitrary 5-keyword limit)
        conditions = " OR ".join(
            f"LOWER(description) ILIKE '%{kw}%' OR LOWER(column_name) ILIKE '%{kw}%' OR LOWER(synonyms) ILIKE '%{kw}%'"
            for kw in keywords[:10]  # up to 10 keywords for broader coverage
        )

        sql = f"""
        SELECT table_name, column_name, description, data_type, synonyms,
               COUNT(*) OVER (PARTITION BY table_name) AS match_count
        FROM ANALYTICS_COPILOT.METADATA.TABLE_DESCRIPTIONS
        WHERE {conditions}
        ORDER BY match_count DESC, table_name
        LIMIT {limit * 15}
        """
        rows = session.sql(sql).collect()

        if not rows:
            # Last resort: return all tables grouped
            return _get_all_tables(session, limit)

        tables_dict: dict = {}
        for row in rows:
            tname = row['TABLE_NAME']
            if tname not in tables_dict:
                tables_dict[tname] = {
                    'table_name': tname,
                    'columns': [],
                    'relevance_score': 0.5,
                    '_match_count': row['MATCH_COUNT']
                }
            tables_dict[tname]['columns'].append({
                'column_name': row['COLUMN_NAME'],
                'description': row['DESCRIPTION'] or '',
                'data_type': row['DATA_TYPE'] or '',
                'synonyms': row['SYNONYMS'] or ''
            })

        # Sort tables by match count (tables with more matching columns rank higher)
        tables_list = sorted(
            tables_dict.values(),
            key=lambda t: t['_match_count'],
            reverse=True
        )
        # Clean up internal field
        for t in tables_list:
            del t['_match_count']

        result = tables_list[:limit]

        # Filter to avoid mixing incompatible datasets
        result = _filter_dataset_mixing(result)

        # Supplement with related tables needed for common joins
        result = _supplement_related_tables(session, result, limit)

        return result

    except Exception as e2:
        warnings.warn(f"Fallback keyword search failed: {e2}. Trying full table list.")
        return _get_all_tables(session, limit)


def _filter_dataset_mixing(tables: list[dict]) -> list[dict]:
    """
    Exclude SUPERSTORE_SALES — the system is Olist-only.

    SUPERSTORE_SALES is a separate US retail dataset with no shared keys with
    the Olist tables. It is excluded from all schema contexts.
    """
    return [t for t in tables if t['table_name'] != 'SUPERSTORE_SALES']


def _supplement_related_tables(session, tables: list[dict], limit: int) -> list[dict]:
    """
    Ensure FK-related tables are included alongside their partners.

    When ORDER_ITEMS is returned without ORDERS, order-timing and customer-join
    queries fail. This supplements the result with one-hop FK neighbors.
    """
    # FK relationships: if anchor table is present, ensure partners are too
    FK_PARTNERS: dict[str, list[str]] = {
        'ORDER_ITEMS': ['ORDERS'],
        'ORDER_REVIEWS': ['ORDERS'],
        'ORDER_PAYMENTS': ['ORDERS'],
        'ORDERS': ['CUSTOMERS'],
    }

    current_names = {t['table_name'] for t in tables}
    needed = set()

    for table in tables:
        for partner in FK_PARTNERS.get(table['table_name'], []):
            if partner not in current_names:
                needed.add(partner)

    if not needed:
        return tables

    # Fetch metadata for needed partner tables
    partner_tables = _fetch_tables_by_name(session, list(needed))
    tables = list(tables) + partner_tables

    return tables


def _fetch_tables_by_name(session, table_names: list[str]) -> list[dict]:
    """Fetch column metadata for specific tables by name."""
    if not table_names:
        return []

    names_str = ", ".join(f"'{n}'" for n in table_names)

    # Try TABLE_DESCRIPTIONS first (has semantic metadata)
    try:
        sql = f"""
        SELECT table_name, column_name, description, data_type, synonyms
        FROM ANALYTICS_COPILOT.METADATA.TABLE_DESCRIPTIONS
        WHERE table_name IN ({names_str})
        ORDER BY table_name
        """
        rows = session.sql(sql).collect()

        if rows:
            tables_dict: dict = {}
            for row in rows:
                tname = row['TABLE_NAME']
                if tname not in tables_dict:
                    tables_dict[tname] = {'table_name': tname, 'columns': [], 'relevance_score': 0.4}
                tables_dict[tname]['columns'].append({
                    'column_name': row['COLUMN_NAME'],
                    'description': row['DESCRIPTION'] or '',
                    'data_type': row['DATA_TYPE'] or '',
                    'synonyms': row['SYNONYMS'] or ''
                })
            return list(tables_dict.values())
    except Exception:
        pass

    # Fallback to INFORMATION_SCHEMA
    try:
        sql = f"""
        SELECT table_name, column_name, data_type
        FROM ANALYTICS_COPILOT.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = 'RAW' AND table_name IN ({names_str})
        ORDER BY table_name, ordinal_position
        """
        rows = session.sql(sql).collect()

        tables_dict = {}
        for row in rows:
            tname = row['TABLE_NAME']
            if tname not in tables_dict:
                tables_dict[tname] = {'table_name': tname, 'columns': [], 'relevance_score': 0.3}
            tables_dict[tname]['columns'].append({
                'column_name': row['COLUMN_NAME'],
                'description': '',
                'data_type': row['DATA_TYPE'],
                'synonyms': ''
            })
        return list(tables_dict.values())
    except Exception as e:
        warnings.warn(f"Could not fetch related tables {table_names}: {e}")
        return []


def _get_all_tables(session, limit: int) -> list[dict]:
    """Last-resort fallback: return all RAW tables with column info from INFORMATION_SCHEMA."""
    try:
        sql = """
        SELECT table_name, column_name, data_type
        FROM ANALYTICS_COPILOT.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = 'RAW'
        ORDER BY table_name, ordinal_position
        """
        rows = session.sql(sql).collect()

        tables_dict: dict = {}
        for row in rows:
            tname = row['TABLE_NAME']
            if tname not in tables_dict:
                tables_dict[tname] = {'table_name': tname, 'columns': [], 'relevance_score': 0.1}
            tables_dict[tname]['columns'].append({
                'column_name': row['COLUMN_NAME'],
                'description': '',
                'data_type': row['DATA_TYPE'],
                'synonyms': ''
            })

        return list(tables_dict.values())[:limit]

    except Exception as e3:
        warnings.warn(f"Could not retrieve table info: {e3}")
        return []


def _escape_sql_string(text: str) -> str:
    """
    Escape single quotes in text for safe SQL string interpolation.

    Snowflake SQL requires single quotes to be escaped as double single quotes ('').
    This prevents SQL injection and syntax errors.

    Args:
        text: Input text that may contain single quotes

    Returns:
        Text with single quotes escaped

    Example:
        >>> _escape_sql_string("What's the revenue?")
        "What''s the revenue?"
    """
    return text.replace("'", "''")
