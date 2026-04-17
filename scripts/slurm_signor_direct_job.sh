#!/bin/bash
# =============================================================================
# slurm_signor_direct_job.sh — SLURM job for direct-mode SIGNOR evaluation
#
# Same GPU layout as slurm_signor_job.sh:
#   - Starts vLLM in background for the subagent (qwen3.5-9b)
#   - Waits for vLLM health endpoint
#   - Runs run_signor_direct_batch.sh with the direct-mode config
#   - Outer agent uses LiteLLM → Anthropic API (reads ANTHROPIC_API_KEY from .env)
#
# Environment variables (exported by submit_signor_direct_batch.sh):
#   RUN_TAG      — timestamped run identifier (e.g. 20260330_142500)
#   EXTRA_ARGS   — forwarded to run_signor_eval.py (e.g. --reps 1)
#   SINGLE_MODE  — "true" to process all rows without chunking
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOCAL_INFO_FILE="${PROJECT_ROOT}/.vllm_server_info_${SLURM_JOB_ID}_${TASK_ID}"

echo "============================================================"
echo "  SIGNOR Direct-Mode Evaluation Job"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Task ID:     ${TASK_ID}"
echo "  Run Tag:     ${RUN_TAG}"
echo "  Single mode: ${SINGLE_MODE:-false}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Row range for this array task (ignored in single mode)
# ---------------------------------------------------------------------------
case "$TASK_ID" in
    0) ROW_START=0;  ROW_LIMIT=16 ;;
    1) ROW_START=16; ROW_LIMIT=16 ;;
    2) ROW_START=32; ROW_LIMIT=16 ;;
    3) ROW_START=48; ROW_LIMIT=16 ;;
    *) ROW_START=0;  ROW_LIMIT=0  ;;
esac

if [[ "${SINGLE_MODE:-false}" == "true" ]]; then
    ROW_START=0
    ROW_LIMIT=0
fi

# ---------------------------------------------------------------------------
# Read subagent model from config
# ---------------------------------------------------------------------------
CONFIG_FILE="${PROJECT_ROOT}/experiments/configs/signor_direct_config.yaml"
SUBAGENT_MODEL=$(uv run python3 -c "
import yaml, sys
c = yaml.safe_load(open('${CONFIG_FILE}'))
print(c.get('llm', {}).get('subagent_model', 'qwen3.5-9b'))
" 2>/dev/null || echo "qwen3.5-9b")
echo "  Subagent model: ${SUBAGENT_MODEL}"

# ---------------------------------------------------------------------------
# Detect cloud vs local subagent
# Cloud prefixes (anthropic/, openai/, gemini/, etc.) skip vLLM startup
# ---------------------------------------------------------------------------
if [[ "$SUBAGENT_MODEL" == */* ]]; then
    CLOUD_SUBAGENT=true
else
    CLOUD_SUBAGENT=false
fi

if [[ "$CLOUD_SUBAGENT" == "false" ]]; then
    # ---------------------------------------------------------------------------
    # Start vLLM in background (local subagent)
    # ---------------------------------------------------------------------------
    echo "[1/4] Starting vLLM in background..."
    bash "${PROJECT_ROOT}/scripts/vllm_node_setup.sh" "$LOCAL_INFO_FILE" "${SUBAGENT_MODEL}" &
    VLLM_PID=$!
    trap "echo 'Cleaning up vLLM...'; kill ${VLLM_PID} 2>/dev/null; rm -f ${LOCAL_INFO_FILE}" EXIT

    # ---------------------------------------------------------------------------
    # Wait for vLLM info file
    # ---------------------------------------------------------------------------
    echo "[2/4] Waiting for vLLM info file..."
    until [[ -f "$LOCAL_INFO_FILE" ]]; do sleep 5; done
    VLLM_PORT=$(cut -d: -f2 < "$LOCAL_INFO_FILE")
    echo "      vLLM port: ${VLLM_PORT}"

    # ---------------------------------------------------------------------------
    # Poll health endpoint
    # ---------------------------------------------------------------------------
    echo "[3/4] Waiting for vLLM health endpoint..."
    HEALTH_WAIT=0
    until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
        sleep 10
        HEALTH_WAIT=$((HEALTH_WAIT + 10))
        echo "      Still waiting... ${HEALTH_WAIT}s elapsed"
    done
    echo "      vLLM is ready at http://localhost:${VLLM_PORT}"
    export LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1/"
else
    echo "[1/4] Cloud subagent (${SUBAGENT_MODEL}) — skipping vLLM startup."
    echo "[2/4] Skipped."
    echo "[3/4] Skipped."
    unset LLM_BASE_URL
fi

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo "[4/4] Running direct-mode evaluation..."

if [[ "${SINGLE_MODE:-false}" == "true" ]]; then
    OUTPUT_CSV="${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}/results.csv"
else
    OUTPUT_CSV="${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}/results_chunk${TASK_ID}.csv"
fi

cd "$PROJECT_ROOT"

RUN_ARGS="--run-tag ${RUN_TAG} --output-csv ${OUTPUT_CSV}"
if [[ $ROW_LIMIT -gt 0 ]]; then
    RUN_ARGS="${RUN_ARGS} --row-start ${ROW_START} --limit ${ROW_LIMIT}"
fi

# shellcheck disable=SC2086
bash "${PROJECT_ROOT}/scripts/run_signor_direct_batch.sh" ${RUN_ARGS} ${EXTRA_ARGS:-}

echo "============================================================"
echo "  Task ${TASK_ID} complete!"
echo "  Results: ${OUTPUT_CSV}"
echo "============================================================"
