"""
Smoke tests for Analytics Copilot.

These tests verify that the core modules import correctly and that
key functions behave as expected without requiring a live Snowflake
connection.
"""

import json
import sys
from pathlib import Path
from unittest import TestCase, main

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestImports(TestCase):
    """Verify all core modules can be imported."""

    def test_import_config(self):
        from src.utils.config import load_config
        config = load_config()
        self.assertIn("seed", config)
        self.assertIn("snowflake", config)
        self.assertIn("llm", config)

    def test_import_logger(self):
        from src.utils.logger import get_logger
        logger = get_logger("test")
        self.assertIsNotNone(logger)

    def test_import_schema_linker(self):
        from src.agents.schema_linker import link_schema
        self.assertTrue(callable(link_schema))

    def test_import_sql_generator(self):
        from src.agents.sql_generator import generate_sql
        self.assertTrue(callable(generate_sql))

    def test_import_validator(self):
        from src.agents.validator import validate_and_execute
        self.assertTrue(callable(validate_and_execute))

    def test_import_viz(self):
        from src.utils.viz import auto_chart
        self.assertTrue(callable(auto_chart))


class TestConfig(TestCase):
    """Verify config loading and values."""

    def test_config_has_seed(self):
        from src.utils.config import load_config
        config = load_config()
        self.assertEqual(config["seed"], 42)

    def test_config_has_llm_model(self):
        from src.utils.config import load_config
        config = load_config()
        self.assertEqual(config["llm"]["model"], "llama3.1-70b")


class TestGoldenQueries(TestCase):
    """Verify golden queries file exists and has valid structure."""

    def test_golden_queries_exists(self):
        path = PROJECT_ROOT / "data" / "golden_queries.json"
        self.assertTrue(path.exists(), "data/golden_queries.json must exist")

    def test_golden_queries_valid_json(self):
        path = PROJECT_ROOT / "data" / "golden_queries.json"
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_golden_queries_structure(self):
        path = PROJECT_ROOT / "data" / "golden_queries.json"
        with open(path) as f:
            data = json.load(f)
        first = data[0]
        self.assertIn("question", first)
        self.assertIn("sql_query", first)
        self.assertIn("difficulty", first)


class TestVisualization(TestCase):
    """Test auto_chart with synthetic data."""

    def test_auto_chart_bar(self):
        import pandas as pd
        from src.utils.viz import auto_chart

        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "value": [10, 20, 30],
        })
        chart = auto_chart(df)
        self.assertIsNotNone(chart, "Bar chart should be generated for cat+num data")

    def test_auto_chart_empty(self):
        import pandas as pd
        from src.utils.viz import auto_chart

        df = pd.DataFrame()
        chart = auto_chart(df)
        self.assertIsNone(chart, "Empty DataFrame should return None")

    def test_auto_chart_single_column(self):
        import pandas as pd
        from src.utils.viz import auto_chart

        df = pd.DataFrame({"x": [1, 2, 3]})
        chart = auto_chart(df)
        self.assertIsNone(chart, "Single column should return None")


class TestSQLExtraction(TestCase):
    """Test SQL extraction from LLM responses."""

    def test_extract_plain_sql(self):
        from src.agents.sql_generator import _extract_sql

        sql = "SELECT COUNT(*) FROM ORDERS"
        result = _extract_sql(sql)
        self.assertIn("SELECT", result)

    def test_extract_fenced_sql(self):
        from src.agents.sql_generator import _extract_sql

        text = "```sql\nSELECT * FROM ORDERS\n```"
        result = _extract_sql(text)
        self.assertIn("SELECT", result)
        self.assertNotIn("```", result)

    def test_extract_empty(self):
        from src.agents.sql_generator import _extract_sql

        result = _extract_sql("")
        self.assertEqual(result, "")


class TestSchemaLinkerInputValidation(TestCase):
    """Test schema linker handles bad input gracefully."""

    def test_empty_question_returns_empty(self):
        import warnings
        from typing import cast
        from snowflake.snowpark import Session
        from src.agents.schema_linker import link_schema

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Pass None for session since we expect early return on empty string
            result = link_schema(cast(Session, None), "", limit=5)
        self.assertEqual(result, [])


class TestPipelineTrace(TestCase):
    """Test PipelineTrace step tracking and output."""

    def test_trace_basic_flow(self):
        from src.utils.trace import PipelineTrace

        trace = PipelineTrace(question="test question")
        trace.start_step("Schema Linker")
        trace.end_step(status="success", detail="Found 3 tables")
        trace.start_step("SQL Generator")
        trace.end_step(status="success", detail="Generated SQL")
        result = trace.finish()

        self.assertEqual(result["question"], "test question")
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(result["steps"][0]["agent"], "Schema Linker")
        self.assertEqual(result["steps"][0]["status"], "success")
        self.assertEqual(result["steps"][1]["agent"], "SQL Generator")
        self.assertIn("query_id", result)
        self.assertIn("timestamp", result)
        self.assertGreaterEqual(result["total_duration_ms"], 0)

    def test_trace_timing_non_negative(self):
        from src.utils.trace import PipelineTrace

        trace = PipelineTrace(question="timing test")
        trace.start_step("Schema Linker")
        trace.end_step(status="success", detail="ok")
        result = trace.finish()

        self.assertGreaterEqual(result["steps"][0]["duration_ms"], 0)
        self.assertGreaterEqual(result["total_duration_ms"], 0)

    def test_trace_error_step(self):
        from src.utils.trace import PipelineTrace

        trace = PipelineTrace(question="error test")
        trace.start_step("Schema Linker")
        trace.end_step(status="error", detail="No tables found")
        result = trace.finish(success=False, error="No tables found")

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "No tables found")
        self.assertEqual(result["steps"][0]["status"], "error")

    def test_trace_finish_sets_sql_and_rows(self):
        from src.utils.trace import PipelineTrace

        trace = PipelineTrace(question="sql test")
        trace.start_step("Validator")
        trace.end_step(status="success", detail="42 rows")
        result = trace.finish(final_sql="SELECT 1", row_count=42)

        self.assertEqual(result["final_sql"], "SELECT 1")
        self.assertEqual(result["row_count"], 42)


if __name__ == "__main__":
    main()
