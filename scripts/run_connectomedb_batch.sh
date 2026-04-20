#!/bin/bash
# =============================================================================
# run_connectomedb_batch.sh
#
# One-click script to run the entire ConnectomeDB evaluation locally.
# It invokes the Python coordinator script which handles the execution of all
# 547 claims, repeated 3 times each.
#
# Usage:
#   bash run_connectomedb_batch.sh                # Run the full pipeline
#   bash run_connectomedb_batch.sh --limit 1      # Test with just the first claim
#   bash run_connectomedb_batch.sh --reps 1       # Change the number of repetitions
# =============================================================================
set -euo pipefail

# Ensure we are in the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Starting End-to-End ConnectomeDB Evaluation               "
echo "============================================================"

# Use LLM_BASE_URL from the environment (set by slurm_connectomedb_job.sh).
# Fall back to localhost:8000 for manual/interactive runs.
if [[ -z "${LLM_BASE_URL:-}" ]]; then
    export LLM_BASE_URL="http://localhost:8000/v1/"
    echo "  LLM_BASE_URL not set, using localhost:8000 fallback"
else
    echo "  Using LLM_BASE_URL=${LLM_BASE_URL}"
fi

echo "  Config: experiments/configs/connectomedb_direct_config.yaml"

echo "============================================================"
echo ""

# Execute the python batch orchestrator
uv run python experiments/run_connectomedb_eval.py \
    --config experiments/configs/connectomedb_direct_config.yaml \
    "$@"

echo ""
echo "============================================================"
echo "  Evaluation completed or paused!                           "
echo "  Check results at: results/connectomedb_eval_results.csv   "
echo "============================================================"
