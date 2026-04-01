#!/bin/bash
# =============================================================================
# run_signor_batch.sh
#
# One-click script to run the entire SIGNOR evaluation locally.
# It invokes the Python coordinator script which handles the execution of all
# 67 claims (both forward and flipped, repeated 3 times).
#
# Usage:
#   bash run_signor_batch.sh                # Run the full pipeline (402 runs)
#   bash run_signor_batch.sh --limit 1      # Test with just the first claim
#   bash run_signor_batch.sh --reps 1       # Change the number of repetitions
# =============================================================================
set -euo pipefail

# Ensure we are in the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Starting End-to-End SIGNOR Evaluation                     "
echo "============================================================"

# Use LLM_BASE_URL from the environment (set by slurm_signor_job.sh).
# Fall back to localhost:8000 for manual/interactive runs.
if [[ -z "${LLM_BASE_URL:-}" ]]; then
    export LLM_BASE_URL="http://localhost:8000/v1/"
    echo "  LLM_BASE_URL not set, using localhost:8000 fallback"
else
    echo "  Using LLM_BASE_URL=${LLM_BASE_URL}"
fi

# The LLM_BASE_URL environment variable will be used by the Python config system
# (via pydantic-settings validation_alias), so we don't need to modify the YAML file
echo "  Config will use LLM_BASE_URL environment variable (no YAML modification)"

echo "============================================================"
echo ""

# Execute the python batch orchestrator
# LLM_BASE_URL environment variable will be propagated to subprocesses by run_signor_eval.py
uv run python experiments/run_signor_eval.py \
    --config experiments/signor_eval_config.yaml \
    "$@"

echo ""
echo "============================================================"
echo "  Evaluation completed or paused!                           "
echo "  Check results at: results/signor_eval_results.csv         "
echo "============================================================"
