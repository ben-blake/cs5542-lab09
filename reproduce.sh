#!/usr/bin/env bash
# reproduce.sh - Single-command reproducibility script for Analytics Copilot
#
# Usage:
#   chmod +x reproduce.sh
#   ./reproduce.sh              # Full pipeline (requires Snowflake credentials)
#   ./reproduce.sh --smoke      # Smoke tests only (no Snowflake needed)
#   ./reproduce.sh --test       # Run tests only
#   ./reproduce.sh --eval       # Run evaluation only
#
# Prerequisites:
#   - Python 3.10+
#   - .env file with Snowflake credentials (for full pipeline)
#   - CSV data files in data/olist/ (for full pipeline)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[REPRO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ── Step 0: Parse arguments ──────────────────────────────────────────
MODE="${1:-full}"

# ── Step 1: Create virtual environment ───────────────────────────────
log "Step 1: Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log "  Created virtual environment"
fi
source venv/bin/activate
export PYTHONUNBUFFERED=1
pip install --quiet -r requirements.txt
log "  Dependencies installed (pinned versions)"

# ── Step 2: Create output directories ────────────────────────────────
mkdir -p artifacts logs
log "Step 2: Output directories ready (artifacts/, logs/)"

# ── Step 3: Verify config ───────────────────────────────────────────
if [ ! -f "config.yaml" ]; then
    fail "config.yaml not found"
fi
log "Step 3: config.yaml found"

# ── Step 4: Smoke tests ─────────────────────────────────────────────
log "Step 4: Running smoke tests..."
python -m pytest tests/test_smoke.py -v --tb=short 2>&1 | tee logs/smoke_test.log
SMOKE_EXIT=${PIPESTATUS[0]}

if [ "$SMOKE_EXIT" -ne 0 ]; then
    fail "Smoke tests failed. See logs/smoke_test.log"
fi
log "  Smoke tests passed"

if [ "$MODE" = "--smoke" ] || [ "$MODE" = "--test" ]; then
    log "Done (test-only mode)."
    exit 0
fi

# ── Step 5: Check Snowflake credentials ──────────────────────────────
if [ ! -f ".env" ]; then
    fail ".env file not found. Copy .env.example and fill in credentials."
fi
log "Step 5: .env file found"

# ── Steps 6-8: Data ingestion & metadata (skip for --eval) ───────────
if [ "$MODE" != "--eval" ]; then
    # ── Step 6: Check data files ──────────────────────────────────────
    if [ ! -d "data/olist" ] || [ -z "$(ls data/olist/*.csv 2>/dev/null)" ]; then
        fail "Olist CSV files not found in data/olist/. See README.md for download instructions."
    fi
    OLIST_COUNT=$(ls data/olist/*.csv | wc -l | tr -d ' ')
    log "Step 6: Found $OLIST_COUNT Olist CSV files"

    # ── Step 7: Run data ingestion ────────────────────────────────────
    log "Step 7: Running data ingestion pipeline..."
    python scripts/ingest_data.py 2>&1 | tee logs/ingest.log
    log "  Data ingestion complete"

    # ── Step 8: Build metadata ────────────────────────────────────────
    log "Step 8: Building semantic metadata..."
    python scripts/build_metadata.py 2>&1 | tee logs/metadata.log
    log "  Metadata generation complete"

    # ── Step 8b: Create Cortex Search Service ─────────────────────────
    log "Step 8b: Creating Cortex Search Service..."
    python -c "
import sys; sys.path.insert(0, '.')
from src.utils.snowflake_conn import get_session, close_session
session = get_session()
with open('snowflake/05_cortex_search.sql') as f:
    sql = f.read()
for stmt in sql.split(';'):
    stmt = stmt.strip()
    if stmt and not stmt.startswith('--'):
        try:
            session.sql(stmt).collect()
        except Exception as e:
            print(f'  Warning: {e}')
close_session()
print('Done')
" 2>&1 | tee -a logs/metadata.log
    log "  Cortex Search Service created"
else
    log "Steps 6-8: Skipped (eval-only mode)"
fi

# ── Step 9: Run evaluation ───────────────────────────────────────────
if [ "$MODE" = "--eval" ] || [ "$MODE" = "full" ]; then
    log "Step 9: Running evaluation..."
    python scripts/evaluate.py 2>&1 | tee logs/evaluation.log
    log "  Evaluation report saved to artifacts/"
fi

# ── Step 10: Generate instruction dataset (Lab 8) ────────────────────
log "Step 10: Creating instruction dataset for fine-tuning..."
python scripts/create_instruction_dataset.py 2>&1 | tee logs/instruction_dataset.log
log "  Instruction dataset saved to data/instruction_dataset.json"

# ── Step 11: Summary ─────────────────────────────────────────────────
log ""
log "============================================"
log "  Reproduction complete!"
log "============================================"
log "  Artifacts:  artifacts/"
log "  Logs:       logs/"
log "  Config:     config.yaml"
log "  App:        streamlit run src/app.py"
log ""
log "  Lab 8 - Fine-Tuning:"
log "    Dataset:    data/instruction_dataset.json"
log "    Fine-tune:  python scripts/fine_tune.py"
log "    Serve:      python scripts/api_server.py --model-path artifacts/fine_tuned_model"
log "    Evaluate:   python scripts/evaluate_adaptation.py"
log "============================================"
