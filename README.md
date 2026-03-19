# Lab 9: Application and Deployment Enhancement

Analytics Copilot — a Text-to-SQL system that converts natural language questions into SQL queries against the Olist Brazilian E-Commerce dataset. Built with a three-agent pipeline on Snowflake Cortex LLM and extended with LoRA fine-tuning of CodeLlama-7B for domain adaptation.

**Deployed Application:** [https://cs5542-lab09.streamlit.app](https://cs5542-lab09.streamlit.app)

## Team Members

- **Ben Blake** (GenAI & Backend Lead) - [@ben-blake](https://github.com/ben-blake)
- **Tina Nguyen** (Data & Frontend Lead) - [@tinana2k](https://github.com/tinana2k)

## Lab 9 Enhancements

### UI/UX Improvements
- Tabbed layout with **Chat** and **Monitor** tabs
- Inline Pipeline Trace expanders showing per-agent timing and status
- Simplified to Cortex-only model (removed baseline/fine-tuned options for deployment)

### Monitoring & Evaluation
- Summary metrics: total queries, success rate, average latency, retry count
- Query History table with timestamp, status, latency, and row count
- Latency chart (per-query, color-coded by success/error)
- Agent Step Breakdown chart (stacked bar showing Schema Linker, SQL Generator, Validator timing)

### Deployment
- Deployed on **Streamlit Community Cloud**
- Snowflake credentials via `st.secrets` (Cloud) with `.env` fallback (local)
- Private key auth supported as inline secret string
- Graceful disconnected/demo mode when credentials are unavailable

## Architecture

```
User Question
  -> Schema Linker (Cortex Search RAG)
  -> SQL Generator (Cortex LLM llama3.1-70b)
  -> Validator (EXPLAIN + self-correction retries)
  -> Results + Auto Visualization
```

## Setup

```bash
# Clone and install
git clone https://github.com/ben-blake/cs5542-lab09.git
cd cs5542-lab09
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Snowflake credentials

# Run smoke tests (no Snowflake needed)
./reproduce.sh --smoke

# Run the app
streamlit run src/app.py
```

## Streamlit Cloud Deployment

1. Push repo to GitHub
2. Connect repo at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Set main file path to `src/app.py`
4. Add Snowflake credentials under Settings > Secrets (see `.streamlit/secrets.toml.example`)

## Project Structure

```
src/
  app.py                  # Streamlit app (Chat + Monitor tabs)
  agents/
    schema_linker.py      # RAG-based schema linking via Cortex Search
    sql_generator.py      # SQL generation via Cortex LLM
    validator.py          # SQL validation with self-correction
  utils/
    trace.py              # Pipeline trace system
    snowflake_conn.py     # Snowflake connection (st.secrets + .env)
    viz.py                # Auto chart generation
    config.py             # Config loader
    logger.py             # Logging setup
scripts/
  evaluate.py             # Pipeline evaluation
  fine_tune.py            # QLoRA fine-tuning
  api_server.py           # Fine-tuned model API
tests/
  test_smoke.py           # Smoke tests
.streamlit/
  config.toml             # Theme and server config
  secrets.toml.example    # Secrets template for Cloud deployment
```
