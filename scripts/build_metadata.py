"""
Metadata Builder Script for Analytics Copilot

This script automatically generates semantic metadata for database tables using Snowflake Cortex LLM.
It queries the INFORMATION_SCHEMA to discover tables and columns, then uses Cortex's llama3.1-70b
model to generate business-friendly descriptions, synonyms, and sample values for each column.

The generated metadata is stored in ANALYTICS_COPILOT.METADATA.TABLE_DESCRIPTIONS table and is
used by the Analytics Copilot to understand schema semantics and improve natural language query
generation.

Process:
1. Query INFORMATION_SCHEMA.COLUMNS for all tables in RAW schema
2. For each table, generate a prompt for Cortex LLM
3. Call SNOWFLAKE.CORTEX.COMPLETE to generate metadata
4. Parse JSON response and insert into TABLE_DESCRIPTIONS
5. Track progress and handle errors gracefully

Required Environment Variables:
    - SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
    - SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE
    (See .env.example for details)

Usage:
    python scripts/build_metadata.py
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to allow imports from src/
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.snowflake_conn import get_session, close_session
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("build_metadata")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def get_schema_info(session, schema_name: str = 'RAW') -> Dict[str, List[Dict[str, Any]]]:
    """
    Query INFORMATION_SCHEMA to get all tables and columns in the specified schema.

    Args:
        session: Active Snowflake Snowpark session
        schema_name: Schema name to query (default: RAW)

    Returns:
        Dictionary mapping table names to list of column information dicts.
        Each column dict contains: column_name, data_type, ordinal_position

    Raises:
        Exception: If schema query fails
    """
    print(f"\nQuerying schema information for {schema_name}...")

    query = f"""
        SELECT
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema_name}'
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """

    try:
        result = session.sql(query).collect()

        # Group columns by table
        tables = {}
        for row in result:
            table_name = row['TABLE_NAME']
            column_info = {
                'column_name': row['COLUMN_NAME'],
                'data_type': row['DATA_TYPE'],
                'ordinal_position': row['ORDINAL_POSITION']
            }

            if table_name not in tables:
                tables[table_name] = []
            tables[table_name].append(column_info)

        print(f"✓ Found {len(tables)} tables with {len(result)} total columns")
        return tables

    except Exception as e:
        raise Exception(f"Failed to query schema information: {str(e)}")


def build_cortex_prompt(table_name: str, columns: List[Dict[str, Any]]) -> str:
    """
    Build a prompt for Cortex LLM to generate column metadata.

    Args:
        table_name: Name of the table
        columns: List of column information dicts

    Returns:
        Formatted prompt string for Cortex LLM
    """
    # Build column list with types
    column_list = []
    for col in columns:
        column_list.append(f"- {col['column_name']} ({col['data_type']})")

    columns_text = "\n".join(column_list)

    prompt = f"""You are a data dictionary expert. Given this database table and its columns,
generate a business-friendly description for each column.

Table: {table_name}
Columns:
{columns_text}

For each column, provide:
1. description: Business-friendly explanation of what this column contains
2. synonyms: Alternative names users might use (comma-separated)
3. sample_values: Examples or typical values (for categorical columns)

Return ONLY a JSON array of objects, one per column:
[
  {{
    "column_name": "...",
    "description": "...",
    "synonyms": "...",
    "sample_values": "..."
  }}
]"""

    return prompt


def generate_metadata_with_cortex(
    session,
    table_name: str,
    columns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Use Snowflake Cortex to generate metadata for table columns.

    Args:
        session: Active Snowflake Snowpark session
        table_name: Name of the table
        columns: List of column information dicts

    Returns:
        List of metadata dicts with column descriptions, synonyms, and sample values

    Raises:
        Exception: If Cortex call fails or response parsing fails
    """
    prompt = build_cortex_prompt(table_name, columns)

    # Escape single quotes in prompt for SQL
    prompt_escaped = prompt.replace("'", "''")

    cfg = get_config()
    llm_model = cfg.get("llm", {}).get("model", "llama3.1-70b")
    cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{llm_model}',
            '{prompt_escaped}'
        ) as response
    """

    try:
        result = session.sql(cortex_query).collect()

        if not result:
            raise Exception("Empty response from Cortex")

        response_text = result[0]['RESPONSE']

        # Parse JSON response
        # The response might have extra text, try to extract JSON array
        try:
            metadata = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')

            if start_idx == -1 or end_idx == -1:
                raise Exception(f"Could not find JSON array in response: {response_text[:200]}")

            json_text = response_text[start_idx:end_idx + 1]
            metadata = json.loads(json_text)

        if not isinstance(metadata, list):
            raise Exception(f"Expected JSON array, got: {type(metadata)}")

        return metadata

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {str(e)}")
    except Exception as e:
        raise Exception(f"Cortex call failed: {str(e)}")


def insert_metadata(
    session,
    table_name: str,
    metadata: List[Dict[str, Any]],
    columns: List[Dict[str, Any]]
) -> int:
    """
    Insert generated metadata into TABLE_DESCRIPTIONS table.

    Args:
        session: Active Snowflake Snowpark session
        table_name: Name of the table
        metadata: List of metadata dicts from Cortex
        columns: Original column information (for data_type)

    Returns:
        Number of rows inserted

    Raises:
        Exception: If insert fails
    """
    # Create lookup for data types
    data_types = {col['column_name']: col['data_type'] for col in columns}

    inserted_count = 0

    for meta in metadata:
        column_name = meta.get('column_name', '').upper()
        description = meta.get('description', '')
        synonyms = meta.get('synonyms', '')
        sample_values = meta.get('sample_values', '')

        # Get data type from original column info
        data_type = data_types.get(column_name, 'UNKNOWN')

        # Escape single quotes for SQL
        description = description.replace("'", "''")
        synonyms = synonyms.replace("'", "''")
        sample_values = sample_values.replace("'", "''")

        insert_query = f"""
            INSERT INTO ANALYTICS_COPILOT.METADATA.TABLE_DESCRIPTIONS
            (TABLE_NAME, COLUMN_NAME, DESCRIPTION, SYNONYMS, DATA_TYPE, SAMPLE_VALUES)
            VALUES (
                '{table_name}',
                '{column_name}',
                '{description}',
                '{synonyms}',
                '{data_type}',
                '{sample_values}'
            )
        """

        try:
            session.sql(insert_query).collect()
            inserted_count += 1
        except Exception as e:
            print(f"  ⚠ Warning: Failed to insert metadata for {column_name}: {str(e)}")
            continue

    return inserted_count


def clear_existing_metadata(session) -> None:
    """
    Truncate TABLE_DESCRIPTIONS table to make script idempotent.

    Args:
        session: Active Snowflake Snowpark session

    Raises:
        Exception: If truncate fails
    """
    print("\nClearing existing metadata...")

    truncate_query = "TRUNCATE TABLE ANALYTICS_COPILOT.METADATA.TABLE_DESCRIPTIONS"

    try:
        session.sql(truncate_query).collect()
        print("✓ Existing metadata cleared")
    except Exception as e:
        raise Exception(f"Failed to truncate TABLE_DESCRIPTIONS: {str(e)}")


def build_metadata_pipeline(schema_name: str = 'RAW') -> None:
    """
    Execute the complete metadata generation pipeline.

    This is the main orchestration function that coordinates all steps:
    1. Connect to Snowflake
    2. Clear existing metadata
    3. Query schema information
    4. For each table, generate metadata using Cortex
    5. Insert metadata into TABLE_DESCRIPTIONS
    6. Report summary statistics

    Args:
        schema_name: Schema to process (default: RAW)
    """
    session = None

    try:
        print_section("Analytics Copilot - Metadata Builder")
        logger.info("Starting metadata build pipeline")

        # Step 1: Connect to Snowflake
        print_section("Step 1: Connecting to Snowflake")
        session = get_session()
        logger.info("Connected to Snowflake")
        print("✓ Connection established")

        # Step 2: Clear existing metadata
        print_section("Step 2: Clearing Existing Metadata")
        clear_existing_metadata(session)

        # Step 3: Query schema information
        print_section("Step 3: Discovering Schema")
        tables = get_schema_info(session, schema_name)

        if not tables:
            print(f"\n⚠ No tables found in schema {schema_name}")
            return

        # Step 4: Generate and insert metadata for each table
        print_section("Step 4: Generating Metadata with Cortex LLM")

        total_tables = len(tables)
        total_columns_processed = 0
        total_columns_inserted = 0
        errors = []

        for idx, (table_name, columns) in enumerate(tables.items(), 1):
            logger.info("Processing table %d/%d: %s (%d columns)", idx, total_tables, table_name, len(columns))
            print(f"\n[{idx}/{total_tables}] Processing table: {table_name} ({len(columns)} columns)")

            try:
                # Generate metadata using Cortex
                print(f"  → Calling Cortex LLM...")
                metadata = generate_metadata_with_cortex(session, table_name, columns)
                print(f"  ✓ Received metadata for {len(metadata)} columns")

                # Insert into database
                print(f"  → Inserting metadata...")
                inserted_count = insert_metadata(session, table_name, metadata, columns)
                print(f"  ✓ Inserted {inserted_count} column descriptions")

                total_columns_processed += len(columns)
                total_columns_inserted += inserted_count

            except Exception as e:
                error_msg = f"{table_name}: {str(e)}"
                errors.append(error_msg)
                print(f"  ✗ Error: {str(e)}")
                print(f"  → Skipping table and continuing...")
                continue

        # Step 5: Summary
        print_section("Metadata Generation Complete")

        logger.info("Metadata build complete: %d tables, %d columns inserted, %d errors",
                    total_tables, total_columns_inserted, len(errors))
        print(f"\nSummary Statistics:")
        print(f"  Tables processed:       {total_tables}")
        print(f"  Columns discovered:     {total_columns_processed}")
        print(f"  Metadata rows inserted: {total_columns_inserted}")
        print(f"  Errors encountered:     {len(errors)}")

        if errors:
            print(f"\nErrors:")
            for error in errors:
                print(f"  - {error}")

        print("\nNext steps:")
        print("  1. Verify metadata in Snowflake:")
        print("     SELECT * FROM ANALYTICS_COPILOT.METADATA.TABLE_DESCRIPTIONS LIMIT 10;")
        print("  2. Review generated descriptions for accuracy")
        print("  3. Manually edit descriptions if needed")
        print("  4. Proceed with Cortex Search Service setup (Task 8)")

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        print(f"\n✗ Configuration error: {str(e)}")
        print("   Please check your .env file and environment variables.")
        sys.exit(1)

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        print(f"\n✗ Unexpected error: {str(e)}")
        print("   Check the error message above and Snowflake logs for details.")
        sys.exit(1)

    finally:
        # Always close the session
        if session:
            print("\nClosing Snowflake session...")
            close_session()


if __name__ == "__main__":
    build_metadata_pipeline()
