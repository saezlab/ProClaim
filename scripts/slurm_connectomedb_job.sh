#!/bin/bash
# =============================================================================
# slurm_connectomedb_job.sh — Self-contained SLURM job script
#
# Starts vLLM in the background on the allocated GPU, waits for it to be ready,
# runs the ConnectomeDB evaluation for this array task's row chunk, then cleans up.
#
# Environment variables (exported by submit_connectomedb_batch.sh via sbatch):
#   RUN_TAG      — timestamped run identifier (e.g. 20260330_142500)
#   EXTRA_ARGS   — additional args forwarded to run_connectomedb_eval.py (e.g. --reps 1)
#   SINGLE_MODE  — set to "true" to process all rows without chunking
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOCAL_INFO_FILE="${PROJECT_ROOT}/.vllm_server_info_${SLURM_JOB_ID}_${TASK_ID}"

echo "============================================================"
echo "  ConnectomeDB Evaluation Job"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Task ID:     ${TASK_ID}"
echo "  Run Tag:     ${RUN_TAG}"
echo "  Single mode: ${SINGLE_MODE:-false}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Row range for this array task (ignored in single mode)
# 318 rows split across 8 tasks: first 6 tasks get 40 rows, remaining 2 get 39
# ---------------------------------------------------------------------------
case "$TASK_ID" in
    0) ROW_START=0;   ROW_LIMIT=40 ;;
    1) ROW_START=40;  ROW_LIMIT=40 ;;
    2) ROW_START=80;  ROW_LIMIT=40 ;;
    3) ROW_START=120; ROW_LIMIT=40 ;;
    4) ROW_START=160; ROW_LIMIT=40 ;;
    5) ROW_START=200; ROW_LIMIT=40 ;;
    6) ROW_START=240; ROW_LIMIT=39 ;;
    7) ROW_START=279; ROW_LIMIT=39 ;;
    *) ROW_START=0;   ROW_LIMIT=0  ;;
esac

if [[ "${SINGLE_MODE:-false}" == "true" ]]; then
    ROW_START=0
    ROW_LIMIT=0
fi

# ---------------------------------------------------------------------------
# Start vLLM in background
# ---------------------------------------------------------------------------
echo "[1/4] Starting vLLM in background..."
bash "${PROJECT_ROOT}/scripts/vllm_node_setup.sh" "$LOCAL_INFO_FILE" "qwen3.5-9b" &
VLLM_PID=$!
trap "echo 'Cleaning up vLLM...'; kill ${VLLM_PID} 2>/dev/null; rm -f ${LOCAL_INFO_FILE}" EXIT

# ---------------------------------------------------------------------------
# Wait for vLLM info file (written by vllm_node_setup.sh once port is bound)
# ---------------------------------------------------------------------------
echo "[2/4] Waiting for vLLM info file..."
until [[ -f "$LOCAL_INFO_FILE" ]]; do sleep 5; done
VLLM_PORT=$(cut -d: -f2 < "$LOCAL_INFO_FILE")
echo "      vLLM port: ${VLLM_PORT}"

# ---------------------------------------------------------------------------
# Poll health endpoint until vLLM is fully loaded
# ---------------------------------------------------------------------------
echo "[3/4] Waiting for vLLM health endpoint..."
HEALTH_WAIT=0
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
    sleep 10
    HEALTH_WAIT=$((HEALTH_WAIT + 10))
    echo "      Still waiting... ${HEALTH_WAIT}s elapsed"
done
echo "      vLLM is ready at http://localhost:${VLLM_PORT}"

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo "[4/4] Running evaluation..."
export LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1/"

if [[ "${SINGLE_MODE:-false}" == "true" ]]; then
    OUTPUT_CSV="${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}/results.csv"
else
    OUTPUT_CSV="${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}/results_chunk${TASK_ID}.csv"
fi

cd "$PROJECT_ROOT"

RUN_ARGS="--run-tag ${RUN_TAG} --output-csv ${OUTPUT_CSV}"
if [[ $ROW_LIMIT -gt 0 ]]; then
    RUN_ARGS="${RUN_ARGS} --row-start ${ROW_START} --limit ${ROW_LIMIT}"
fi

# shellcheck disable=SC2086
bash "${PROJECT_ROOT}/scripts/run_connectomedb_batch.sh" ${RUN_ARGS} ${EXTRA_ARGS:-}

echo "============================================================"
echo "  Task ${TASK_ID} complete!"
echo "  Results: ${OUTPUT_CSV}"
echo "============================================================"
