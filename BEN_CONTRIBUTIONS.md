# Individual Contribution Report — Ben

**Role:** GenAI & Backend Lead

## Personal Contributions

- Designed and implemented the three-agent pipeline (Schema Linker, SQL Generator, Validator) with self-correction retry logic
- Built Snowflake Cortex LLM integration for SQL generation using llama3.1-70b
- Created the pipeline trace system for monitoring agent step timings and status
- Implemented Snowflake connection utility with st.secrets support for Streamlit Cloud deployment
- Developed the fine-tuned model API server and LoRA evaluation pipeline
- Wrote the evaluation framework and golden query generation scripts
- Set up deployment configuration for Streamlit Cloud (secrets management, graceful fallback)
- Authored the reproducibility script (reproduce.sh)

## GitHub Commit Evidence

| File | Description | Commit |
|------|-------------|--------|
| `src/agents/schema_linker.py` | RAG-based schema linker using Cortex Search Service | [`5161f6d`](https://github.com/ben-blake/cs5542-lab09/commit/5161f6d) |
| `src/agents/sql_generator.py` | Cortex LLM SQL generation with few-shot prompting | [`dc86595`](https://github.com/ben-blake/cs5542-lab09/commit/dc86595) |
| `src/agents/validator.py` | SQL validation with EXPLAIN and self-correction retry loop | [`915ecc9`](https://github.com/ben-blake/cs5542-lab09/commit/915ecc9) |
| `src/utils/snowflake_conn.py` | Snowflake connection with st.secrets and .env support | [`b9830a0`](https://github.com/ben-blake/cs5542-lab09/commit/b9830a0) |
| `src/utils/trace.py` | Pipeline trace system for monitoring and debugging | [`38c997a`](https://github.com/ben-blake/cs5542-lab09/commit/38c997a) |
| `src/utils/config.py` | Configuration loader from config.yaml | [`a804c50`](https://github.com/ben-blake/cs5542-lab09/commit/a804c50) |
| `src/utils/logger.py` | Logging setup with file and console handlers | [`f042d94`](https://github.com/ben-blake/cs5542-lab09/commit/f042d94) |
| `scripts/api_server.py` | FastAPI server for fine-tuned LoRA model inference | [`2dfe208`](https://github.com/ben-blake/cs5542-lab09/commit/2dfe208) |
| `scripts/evaluate.py` | Pipeline evaluation framework against golden queries | [`a6cb17a`](https://github.com/ben-blake/cs5542-lab09/commit/a6cb17a) |
| `scripts/evaluate_adaptation.py` | Baseline vs fine-tuned model comparison evaluation | [`06e7a50`](https://github.com/ben-blake/cs5542-lab09/commit/06e7a50) |
| `scripts/fine_tune.py` | QLoRA 4-bit fine-tuning of CodeLlama-7B-Instruct | [`f6e9256`](https://github.com/ben-blake/cs5542-lab09/commit/f6e9256) |
| `scripts/generate_golden.py` | Golden query generation for evaluation | [`6b3c007`](https://github.com/ben-blake/cs5542-lab09/commit/6b3c007) |
| `config.yaml` | Centralized runtime configuration | [`13cd7ca`](https://github.com/ben-blake/cs5542-lab09/commit/13cd7ca) |
| `reproduce.sh` | Single-command reproducibility script | [`a9596e1`](https://github.com/ben-blake/cs5542-lab09/commit/a9596e1) |
| `.streamlit/config.toml` | Streamlit Cloud theme and server configuration | [`f6134fb`](https://github.com/ben-blake/cs5542-lab09/commit/f6134fb) |
| `.streamlit/secrets.toml.example` | Secrets template for Streamlit Cloud deployment | [`6c3a71e`](https://github.com/ben-blake/cs5542-lab09/commit/6c3a71e) |
| `tests/test_smoke.py` | Smoke tests including PipelineTrace tests | [`2e76919`](https://github.com/ben-blake/cs5542-lab09/commit/2e76919) |
| `README.md` | Project documentation and setup instructions | [`453ba99`](https://github.com/ben-blake/cs5542-lab09/commit/453ba99) |

## Tools Used

- **Claude Code (Anthropic):** Used for implementing Lab 9 enhancements — pipeline trace system, Streamlit Cloud deployment configuration, app refactoring, and test development
- **Snowflake Cortex:** LLM backend (llama3.1-70b) for SQL generation and schema search
- **Google Colab:** Fine-tuning CodeLlama-7B with QLoRA on T4 GPU
