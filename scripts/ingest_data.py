"""
Data Ingestion Script for Analytics Copilot

This script automates the complete data ingestion pipeline for the Analytics Copilot project:
1. Executes Snowflake DDL scripts (setup, table creation, metadata)
2. Uploads CSV files from Olist and Superstore datasets to Snowflake stages
3. Loads data into tables using COPY INTO commands
4. Validates data ingestion by checking row counts

Required Environment Variables:
    - SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
    - SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE
    (See .env.example for details)

Directory Structure Expected:
    - data/olist/*.csv (9 CSV files from Olist dataset)
    - data/superstore/Sample - Superstore.csv
    - snowflake/01_setup.sql through 04_metadata.sql

Usage:
    python scripts/ingest_data.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports from src/
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.snowflake_conn import get_session, close_session
from src.utils.logger import get_logger

logger = get_logger("ingest_data")


# File mappings: CSV filename -> Snowflake table name
OLIST_FILE_MAPPINGS = {
    'olist_customers_dataset.csv': 'CUSTOMERS',
    'olist_orders_dataset.csv': 'ORDERS',
    'olist_order_items_dataset.csv': 'ORDER_ITEMS',
    'olist_order_payments_dataset.csv': 'ORDER_PAYMENTS',
    'olist_order_reviews_dataset.csv': 'ORDER_REVIEWS',
    'olist_products_dataset.csv': 'PRODUCTS',
    'olist_sellers_dataset.csv': 'SELLERS',
    'olist_geolocation_dataset.csv': 'GEOLOCATION',
    'product_category_name_translation.csv': 'PRODUCT_CATEGORY_TRANSLATION'
}

SUPERSTORE_FILE_MAPPINGS = {
    'Sample - Superstore.csv': 'SUPERSTORE_SALES'
}


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def execute_sql_file(session, sql_file_path: str) -> None:
    """
    Execute a SQL file in Snowflake.

    Args:
        session: Active Snowflake Snowpark session
        sql_file_path: Path to the SQL file to execute

    Raises:
        FileNotFoundError: If SQL file doesn't exist
        Exception: If SQL execution fails
    """
    if not os.path.exists(sql_file_path):
        raise FileNotFoundError(f"SQL file not found: {sql_file_path}")

    print(f"\nExecuting: {os.path.basename(sql_file_path)}")

    with open(sql_file_path, 'r') as f:
        sql_content = f.read()

    # Split into individual statements and execute
    # Note: Simple split on semicolon; assumes no semicolons in strings/comments
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

    for i, statement in enumerate(statements, 1):
        # Strip comment-only lines from the statement
        lines = [line for line in statement.splitlines() if not line.strip().startswith('--')]
        cleaned = '\n'.join(lines).strip()
        if not cleaned:
            continue

        try:
            session.sql(cleaned).collect()
            print(f"  ✓ Statement {i}/{len(statements)} executed successfully")
        except Exception as e:
            # Print statement for debugging but continue
            print(f"  ⚠ Warning on statement {i}: {str(e)}")
            print(f"    Statement preview: {statement[:100]}...")

    print(f"✓ Completed: {os.path.basename(sql_file_path)}")


def upload_files_to_stage(session, local_dir: str, stage_name: str, file_pattern: str = "*.csv") -> None:
    """
    Upload CSV files from local directory to Snowflake stage.

    Args:
        session: Active Snowflake Snowpark session
        local_dir: Local directory containing CSV files
        stage_name: Fully qualified Snowflake stage name (e.g., @SCHEMA.STAGE_NAME)
        file_pattern: File pattern to upload (default: *.csv)

    Raises:
        FileNotFoundError: If local directory doesn't exist
        Exception: If upload fails
    """
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Data directory not found: {local_dir}")

    # Get list of files
    csv_files = list(Path(local_dir).glob(file_pattern))

    if not csv_files:
        print(f"  ⚠ Warning: No files found matching {file_pattern} in {local_dir}")
        return

    print(f"\nUploading {len(csv_files)} file(s) from {local_dir} to {stage_name}")

    # Upload files one at a time for per-file progress feedback
    for csv_file in csv_files:
        file_path = str(csv_file).replace('\\', '/')
        print(f"  → Uploading {csv_file.name}...", end=" ", flush=True)
        try:
            put_command = f"PUT 'file://{file_path}' {stage_name} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
            result = session.sql(put_command).collect()
            status = result[0]['status'] if result else 'UNKNOWN'
            print(f"✓ {status}")
        except Exception as e:
            print(f"✗ {e}")
            raise Exception(f"Failed to upload {csv_file.name} to {stage_name}: {str(e)}")

    print(f"✓ Upload completed to {stage_name}")


def load_data_from_stage(session, stage_name: str, table_name: str, file_name: str) -> None:
    """
    Load data from stage into Snowflake table using COPY INTO.

    Args:
        session: Active Snowflake Snowpark session
        stage_name: Fully qualified stage name (e.g., @SCHEMA.STAGE_NAME)
        table_name: Fully qualified table name (e.g., SCHEMA.TABLE)
        file_name: Name of the file in the stage

    Raises:
        Exception: If COPY INTO fails
    """
    copy_command = f"""
        COPY INTO {table_name}
        FROM {stage_name}/{file_name}
        FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1)
        ON_ERROR = 'CONTINUE'
        PURGE = FALSE
    """

    try:
        result = session.sql(copy_command).collect()

        # Parse results
        if result:
            row = result[0]
            rows_loaded = row['rows_loaded'] if hasattr(row, 'rows_loaded') else 0
            errors_seen = row['errors_seen'] if hasattr(row, 'errors_seen') else 0

            if errors_seen > 0:
                print(f"  ⚠ {file_name} -> {table_name}: {rows_loaded} rows loaded, {errors_seen} errors")
            else:
                print(f"  ✓ {file_name} -> {table_name}: {rows_loaded} rows loaded")
        else:
            print(f"  ✓ {file_name} -> {table_name}: Loaded successfully")

    except Exception as e:
        raise Exception(f"Failed to load {file_name} into {table_name}: {str(e)}")


def validate_data_load(session, schema_name: str = "RAW") -> None:
    """
    Validate data load by checking row counts for all tables.

    Args:
        session: Active Snowflake Snowpark session
        schema_name: Schema name to check (default: RAW)
    """
    print_section("Data Validation")

    # Get all tables in the schema
    tables_query = f"""
        SELECT table_name
        FROM INFORMATION_SCHEMA.TABLES
        WHERE table_schema = '{schema_name}'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """

    try:
        tables = session.sql(tables_query).collect()

        if not tables:
            print(f"⚠ No tables found in schema {schema_name}")
            return

        print(f"\nRow counts for {len(tables)} table(s) in {schema_name} schema:\n")

        total_rows = 0
        for table in tables:
            table_name = table['TABLE_NAME']
            count_query = f"SELECT COUNT(*) as cnt FROM {schema_name}.{table_name}"

            try:
                result = session.sql(count_query).collect()
                row_count = result[0]['CNT'] if result else 0
                total_rows += row_count

                # Format with alignment
                print(f"  {table_name:40} {row_count:>10,} rows")

            except Exception as e:
                print(f"  {table_name:40} ERROR: {str(e)}")

        print(f"\n  {'TOTAL':40} {total_rows:>10,} rows")
        print("\n✓ Validation completed")

    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")


def run_ingestion_pipeline() -> None:
    """
    Execute the complete data ingestion pipeline.

    This is the main orchestration function that coordinates all steps:
    1. SQL script execution
    2. File uploads to stages
    3. Data loading via COPY INTO
    4. Validation
    """
    session = None

    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent

        print_section("Analytics Copilot - Data Ingestion Pipeline")
        logger.info("Starting data ingestion pipeline")
        print(f"\nProject root: {project_root}")

        # Step 1: Connect to Snowflake
        print_section("Step 1: Connecting to Snowflake")
        session = get_session()
        logger.info("Connected to Snowflake")
        print("✓ Connection established")

        # Step 2: Execute SQL scripts
        print_section("Step 2: Executing SQL Scripts")

        sql_files = [
            'snowflake/01_setup.sql',
            'snowflake/02_olist_tables.sql',
            'snowflake/03_superstore.sql',
            'snowflake/04_metadata.sql'
        ]

        for sql_file in sql_files:
            sql_path = project_root / sql_file
            try:
                execute_sql_file(session, str(sql_path))
            except Exception as e:
                print(f"✗ Error executing {sql_file}: {str(e)}")
                # Continue with other scripts

        # Step 3: Upload files to stages
        print_section("Step 3: Uploading CSV Files to Stages")

        # Check if data directory exists
        data_dir = project_root / 'data'

        if not data_dir.exists():
            print(f"\n⚠ WARNING: Data directory not found at {data_dir}")
            print("   Please create the data directory and add CSV files:")
            print("     - data/olist/*.csv (9 files)")
            print("     - data/superstore/Sample - Superstore.csv")
            print("\n   Skipping file upload and data loading steps...")
            return

        # Upload Olist files
        olist_dir = data_dir / 'olist'
        if olist_dir.exists():
            try:
                upload_files_to_stage(
                    session,
                    str(olist_dir),
                    '@ANALYTICS_COPILOT.RAW.OLIST_STAGE',
                    '*.csv'
                )
            except Exception as e:
                print(f"✗ Error uploading Olist files: {str(e)}")
        else:
            print(f"⚠ Olist data directory not found: {olist_dir}")

        # Upload Superstore files
        superstore_dir = data_dir / 'superstore'
        if superstore_dir.exists():
            try:
                upload_files_to_stage(
                    session,
                    str(superstore_dir),
                    '@ANALYTICS_COPILOT.RAW.SUPERSTORE_STAGE',
                    '*.csv'
                )
            except Exception as e:
                print(f"✗ Error uploading Superstore files: {str(e)}")
        else:
            print(f"⚠ Superstore data directory not found: {superstore_dir}")

        # Step 4: Load data into tables
        print_section("Step 4: Loading Data into Tables")

        # Load Olist tables
        print("\nLoading Olist tables:")
        for csv_file, table_name in OLIST_FILE_MAPPINGS.items():
            try:
                load_data_from_stage(
                    session,
                    '@ANALYTICS_COPILOT.RAW.OLIST_STAGE',
                    f'ANALYTICS_COPILOT.RAW.{table_name}',
                    csv_file
                )
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {str(e)}")

        # Load Superstore table
        print("\nLoading Superstore table:")
        for csv_file, table_name in SUPERSTORE_FILE_MAPPINGS.items():
            try:
                load_data_from_stage(
                    session,
                    '@ANALYTICS_COPILOT.RAW.SUPERSTORE_STAGE',
                    f'ANALYTICS_COPILOT.RAW.{table_name}',
                    csv_file
                )
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {str(e)}")

        # Step 5: Validate data
        validate_data_load(session, 'RAW')

        # Final summary
        logger.info("Data ingestion pipeline complete")
        print_section("Ingestion Pipeline Complete")
        print("\nNext steps:")
        print("  1. Verify row counts match expected values")
        print("  2. Check for any errors or warnings above")
        print("  3. Review data quality in Snowflake")
        print("  4. Proceed with metadata generation (Task 7)")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        print(f"\n✗ File not found: {str(e)}")
        print("   Please ensure all required files and directories exist.")
        sys.exit(1)

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
    run_ingestion_pipeline()
