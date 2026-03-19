# Individual Contribution Report — Tina

**Role:** Data & Frontend Lead

## Personal Contributions

- Designed and built the Streamlit chat application interface with tabbed layout (Chat + Monitor)
- Implemented the Monitor tab with summary metrics, query history table, latency charts, and agent step breakdown visualization
- Created the automatic chart generation system (auto_chart) for query result visualization
- Managed Olist dataset ingestion pipeline and Snowflake table setup
- Built the semantic metadata system with table descriptions and Cortex Search Service
- Developed the instruction dataset creation pipeline for fine-tuning
- Authored Snowflake SQL schemas for database setup, table creation, and metadata
- Created the fine-tuning Colab notebook for CodeLlama-7B LoRA adaptation

## GitHub Commit Evidence

| File | Description | Commit |
|------|-------------|--------|
| `src/app.py` | Streamlit app with Chat/Monitor tabs, trace integration, graceful fallback | [`39876dd`](https://github.com/ben-blake/cs5542-lab09/commit/39876dd) |
| `src/utils/viz.py` | Automatic chart selection (line, bar, scatter) based on DataFrame types | [`69042b8`](https://github.com/ben-blake/cs5542-lab09/commit/69042b8) |
| `src/utils/finetuned_client.py` | Client for fine-tuned model API with baseline/LoRA endpoints | [`b409be2`](https://github.com/ben-blake/cs5542-lab09/commit/b409be2) |
| `scripts/ingest_data.py` | Olist CSV data ingestion into Snowflake | [`889279c`](https://github.com/ben-blake/cs5542-lab09/commit/889279c) |
| `scripts/build_metadata.py` | Semantic metadata generation for table descriptions | [`9bc51bb`](https://github.com/ben-blake/cs5542-lab09/commit/9bc51bb) |
| `scripts/create_instruction_dataset.py` | Instruction dataset creation for fine-tuning | [`d1a3f92`](https://github.com/ben-blake/cs5542-lab09/commit/d1a3f92) |
| `notebooks/fine_tune_colab.ipynb` | Google Colab notebook for QLoRA fine-tuning | [`b308d8f`](https://github.com/ben-blake/cs5542-lab09/commit/b308d8f) |
| `data/golden_queries.json` | Evaluation golden queries dataset | [`8dc011f`](https://github.com/ben-blake/cs5542-lab09/commit/8dc011f) |
| `data/instruction_dataset.json` | Full fine-tuning instruction dataset | [`e6bd4f5`](https://github.com/ben-blake/cs5542-lab09/commit/e6bd4f5) |
| `requirements.txt` | Pinned Python dependencies | [`05184b8`](https://github.com/ben-blake/cs5542-lab09/commit/05184b8) |
| `.env.example` | Environment variable template | [`e53c929`](https://github.com/ben-blake/cs5542-lab09/commit/e53c929) |
| `.gitignore` | Git ignore rules for secrets, artifacts, venvs | [`1c8e00b`](https://github.com/ben-blake/cs5542-lab09/commit/1c8e00b) |

## Tools Used

- **Claude Code (Anthropic):** Used for implementing Lab 9 enhancements — Monitor tab design, visualization improvements, and deployment configuration
- **Snowflake:** Data warehouse for Olist dataset storage and Cortex Search Service
- **Google Colab:** Fine-tuning environment with T4 GPU for LoRA adaptation
