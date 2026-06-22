#!/bin/bash
# =============================================================================
# run_signor_direct_batch.sh
#
# Runs SIGNOR evaluation using the direct mode pipeline
# (LiteLLM outer agent + bash + jupytext, no Jupyter kernel).
#
# LLM_BASE_URL must be set in the environment (done by slurm_signor_direct_job.sh).
# Falls back to localhost:8000 for manual/interactive runs.
#
# Usage:
#   bash run_signor_direct_batch.sh              # Full evaluation
#   bash run_signor_direct_batch.sh --limit 1    # Test with 1 claim
#   bash run_signor_direct_batch.sh --reps 1     # 1 repetition only
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Starting SIGNOR Evaluation (direct mode)                  "
echo "============================================================"

if [[ -z "${LLM_BASE_URL:-}" ]]; then
    export LLM_BASE_URL="http://localhost:8000/v1/"
    echo "  LLM_BASE_URL not set, using localhost:8000 fallback"
else
    echo "  Using LLM_BASE_URL=${LLM_BASE_URL}"
fi

CONFIG_NAME="${CONFIG_NAME:-signor_direct_config}"
CONFIG_FILE="experiments/configs/${CONFIG_NAME}.yaml"
echo "  Config: ${CONFIG_FILE}"
echo "============================================================"
echo ""

uv run python experiments/run_signor_eval.py \
    --config "${CONFIG_FILE}" \
    "$@"

echo ""
echo "============================================================"
echo "  Evaluation complete!                                       "
echo "============================================================"
