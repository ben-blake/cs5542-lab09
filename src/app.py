"""
Analytics Copilot - Streamlit Chat Application

Text-to-SQL chat interface with pipeline monitoring.
Uses Snowflake Cortex LLM for SQL generation.

Usage:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import altair as alt

from src.utils.snowflake_conn import get_session
from src.utils.config import get_config
from src.agents.schema_linker import link_schema
from src.agents.sql_generator import generate_sql
from src.agents.validator import validate_and_execute
from src.utils.viz import auto_chart
from src.utils.trace import PipelineTrace


# Page configuration
st.set_page_config(
    page_title="Analytics Copilot",
    page_icon="🤖",
    layout="wide"
)


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'query_log' not in st.session_state:
        st.session_state.query_log = []
    if 'snowflake_session' not in st.session_state:
        session = get_session()
        st.session_state.snowflake_session = session
        if session is not None:
            st.session_state.connection_status = "Connected"
        else:
            st.session_state.connection_status = "Disconnected"


def render_sidebar():
    """Render the sidebar with connection status, dataset info, and instructions."""
    with st.sidebar:
        st.title("🤖 Analytics Copilot")

        # Connection status
        st.subheader("Connection Status")
        if st.session_state.connection_status == "Connected":
            st.success("Connected to Snowflake")
        else:
            st.warning("Disconnected — demo mode")

        # Session query counter
        query_count = len(st.session_state.query_log)
        st.caption(f"Session: {query_count} queries")

        st.divider()

        # Dataset info
        st.subheader("Dataset")
        st.info("Olist Brazilian E-Commerce")

        st.divider()

        # Instructions
        st.subheader("How to Use")
        st.markdown("""
        **Ask questions about your data in natural language!**

        **Examples:**
        - "What are the top 5 product categories by revenue?"
        - "Show me monthly order trends over time"
        - "Which customers have the highest average review score?"
        - "What's the average delivery time by customer state?"

        **Features:**
        - Automatic schema detection
        - AI-powered SQL generation
        - Query validation & auto-correction
        - Smart visualizations
        - Pipeline monitoring
        """)

        st.divider()

        # Clear chat button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_log = []
            st.rerun()


def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_question(question: str):
    """Process user question through the agent pipeline with trace instrumentation."""
    session = st.session_state.snowflake_session

    if session is None:
        st.error("No active Snowflake connection.")
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        trace = PipelineTrace(question=question)

        try:
            # Step 1: Schema Linking
            status_placeholder.info("Finding relevant tables...")
            trace.start_step("Schema Linker")
            cfg = get_config()
            schema_context = link_schema(session, question)

            if not schema_context:
                trace.end_step(status="error", detail="No relevant tables found")
                trace_result = trace.finish(success=False, error="No relevant tables found")
                st.session_state.query_log.append(trace_result)
                error_msg = "Could not find relevant tables. Please try rephrasing."
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return

            trace.end_step(status="success", detail=f"Found {len(schema_context)} tables")

            # Step 2: SQL Generation
            status_placeholder.info("Generating SQL query...")
            trace.start_step("SQL Generator")
            sql_query = generate_sql(session, question, schema_context)

            if not sql_query or not sql_query.strip():
                trace.end_step(status="error", detail="Empty SQL returned")
                trace_result = trace.finish(success=False, error="Could not generate SQL")
                st.session_state.query_log.append(trace_result)
                error_msg = "Could not generate SQL query."
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return

            trace.end_step(status="success", detail="Generated SQL")

            # Step 3: Validation and Execution
            status_placeholder.info("Validating and executing query...")
            trace.start_step("Validator")
            max_retries = cfg.get("sql_generator", {}).get("max_retries", 3)
            original_sql = sql_query
            final_sql, result = validate_and_execute(
                session, sql_query, question, schema_context, max_retries=max_retries
            )

            status_placeholder.empty()

            # Check result
            if isinstance(result, str):
                # Detect retries: if final_sql differs from original, retries occurred
                retried = final_sql != original_sql
                detail = f"Failed after retries: {result[:80]}" if retried else f"Failed: {result[:80]}"
                trace.end_step(status="error", detail=detail)
                trace_result = trace.finish(success=False, error=result, final_sql=final_sql)
                st.session_state.query_log.append(trace_result)

                error_msg = f"Query execution failed:\n\n{result}"
                response_placeholder.error(error_msg)

                with st.expander("Relevant Tables Found", expanded=False):
                    for table in schema_context:
                        st.markdown(f"**{table['table_name']}** (Relevance: {table['relevance_score']:.2f})")
                        for col in table["columns"]:
                            st.text(f"  - {col['column_name']} ({col['data_type']}): {col['description']}")

                with st.expander("Failed SQL Query", expanded=True):
                    st.code(final_sql, language="sql")

                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return

            # Success
            result_df = result.to_pandas()
            row_count = len(result_df)
            retried = final_sql != original_sql
            detail = f"{row_count} rows returned"
            if retried:
                detail += " (after retries)"
            trace.end_step(status="success", detail=detail)
            trace_result = trace.finish(final_sql=final_sql, row_count=row_count)
            st.session_state.query_log.append(trace_result)

            if result_df.empty:
                success_msg = "Query executed, but no data matched."
                response_placeholder.warning(success_msg)
            else:
                success_msg = f"Found {row_count} result(s)!"
                response_placeholder.success(success_msg)

            # Pipeline trace expander
            with st.expander("Pipeline Trace", expanded=False):
                for step in trace_result["steps"]:
                    icon = "+" if step["status"] == "success" else "x"
                    st.text(f"[{icon}] {step['agent']}: {step['detail']} ({step['duration_ms']}ms)")
                st.text(f"Total: {trace_result['total_duration_ms']}ms")

            with st.expander("Relevant Tables Found", expanded=False):
                for table in schema_context:
                    st.markdown(f"**{table['table_name']}** (Relevance: {table['relevance_score']:.2f})")
                    for col in table["columns"]:
                        st.text(f"  - {col['column_name']} ({col['data_type']}): {col['description']}")

            with st.expander("Show SQL", expanded=False):
                st.code(final_sql, language="sql")

            if not result_df.empty:
                st.subheader("Results")
                display_df = result_df.head(1000)
                if len(result_df) > 1000:
                    st.caption(f"Showing first 1,000 of {len(result_df):,} rows.")
                st.dataframe(display_df, use_container_width=True)

                chart = auto_chart(result_df)
                if chart is not None:
                    st.subheader("Visualization")
                    st.altair_chart(chart, use_container_width=True)

            st.session_state.messages.append({"role": "assistant", "content": success_msg})

        except Exception as e:
            # Guard: only end_step if a step is currently in progress
            if trace._step_start is not None:
                trace.end_step(status="error", detail=str(e)[:80])
            trace_result = trace.finish(success=False, error=str(e))
            st.session_state.query_log.append(trace_result)

            error_msg = f"An unexpected error occurred: {str(e)}"
            status_placeholder.empty()
            response_placeholder.error(error_msg)
            st.exception(e)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


def render_monitor_tab():
    """Render the Monitor tab with query history, metrics, and charts."""
    query_log = st.session_state.query_log

    if not query_log:
        st.info("No queries yet. Ask a question in the Chat tab to see monitoring data.")
        return

    # Summary metrics
    total = len(query_log)
    successes = sum(1 for q in query_log if q["success"])
    avg_latency = sum(q["total_duration_ms"] for q in query_log) / total
    retry_count = sum(
        1 for q in query_log
        for s in q["steps"]
        if s["agent"] == "Validator" and "retries" in s.get("detail", "").lower()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", total)
    col2.metric("Success Rate", f"{successes / total * 100:.0f}%")
    col3.metric("Avg Latency", f"{avg_latency:.0f}ms")
    col4.metric("Queries w/ Retries", retry_count)

    st.divider()

    # Query History table
    st.subheader("Query History")
    history_data = []
    for q in query_log:
        history_data.append({
            "Time": q["timestamp"][:19].replace("T", " "),
            "Question": q["question"][:60] + ("..." if len(q["question"]) > 60 else ""),
            "Status": "Success" if q["success"] else "Error",
            "Latency (ms)": q["total_duration_ms"],
            "Rows": q.get("row_count") or 0,
        })
    st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)

    st.divider()

    # Latency chart
    st.subheader("Query Latency")
    latency_data = pd.DataFrame([
        {
            "Query": f"Q{i+1}",
            "Latency (ms)": q["total_duration_ms"],
            "Status": "Success" if q["success"] else "Error",
        }
        for i, q in enumerate(query_log)
    ])
    latency_chart = alt.Chart(latency_data).mark_bar().encode(
        x=alt.X("Query:N", sort=None),
        y=alt.Y("Latency (ms):Q"),
        color=alt.Color("Status:N", scale=alt.Scale(
            domain=["Success", "Error"],
            range=["#4CAF50", "#F44336"]
        )),
        tooltip=["Query", "Latency (ms)", "Status"],
    ).properties(height=300)
    st.altair_chart(latency_chart, use_container_width=True)

    # Agent step breakdown
    if total > 0:
        st.subheader("Agent Step Breakdown")
        step_data = []
        for i, q in enumerate(query_log):
            for s in q["steps"]:
                step_data.append({
                    "Query": f"Q{i+1}",
                    "Agent": s["agent"],
                    "Duration (ms)": s["duration_ms"],
                })
        if step_data:
            step_df = pd.DataFrame(step_data)
            step_chart = alt.Chart(step_df).mark_bar().encode(
                x=alt.X("Query:N", sort=None),
                y=alt.Y("Duration (ms):Q", stack=True),
                color=alt.Color("Agent:N", scale=alt.Scale(
                    domain=["Schema Linker", "SQL Generator", "Validator"],
                    range=["#2196F3", "#FF9800", "#9C27B0"]
                )),
                tooltip=["Query", "Agent", "Duration (ms)"],
            ).properties(height=300)
            st.altair_chart(step_chart, use_container_width=True)


def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()

    st.title("Analytics Copilot")

    tab_chat, tab_monitor = st.tabs(["Chat", "Monitor"])

    with tab_chat:
        display_chat_history()

        connected = st.session_state.snowflake_session is not None
        if not connected:
            st.info("Running in demo mode — Snowflake not connected. Configure credentials to enable queries.")

        if prompt := st.chat_input("Ask a question about your data...", disabled=not connected):
            process_user_question(prompt)

    with tab_monitor:
        render_monitor_tab()


if __name__ == "__main__":
    main()
