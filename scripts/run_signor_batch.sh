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

# Read vLLM server info directly from the shared file
if [[ -f "${PROJECT_ROOT}/.vllm_server_info" ]]; then
    VLLM_INFO=$(cat "${PROJECT_ROOT}/.vllm_server_info")
    # Extract hostname and port - hostname already includes full FQDN
    COMPUTE_HOST=$(echo "$VLLM_INFO" | cut -d: -f1)
    REMOTE_PORT=$(echo "$VLLM_INFO" | cut -d: -f2)
    # Use hostname as-is (already has .ebi.ac.uk suffix from vllm_node_setup.sh)
    export LLM_BASE_URL="http://${COMPUTE_HOST}:${REMOTE_PORT}/v1/"
    echo "  Loaded vLLM server info from .vllm_server_info"
    echo "  Located server at: $LLM_BASE_URL"
else
    # Fallback if no server info is found
    export LLM_BASE_URL="http://localhost:8000/v1/"
    echo "  No .vllm_server_info found, using localhost:8000 fallback"
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
