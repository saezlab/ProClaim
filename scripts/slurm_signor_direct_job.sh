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
#   NUM_TASKS    — number of array tasks used for dataset partitioning
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${GRN_LLM_CORRECT_PROJECT_ROOT:-/hps/nobackup/saezrodriguez/${USER}/workspace/grn-llm-correct}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOCAL_INFO_FILE="${PROJECT_ROOT}/.vllm_server_info_${SLURM_JOB_ID}_${TASK_ID}"
DATASET_CSV="${PROJECT_ROOT}/datasets/signor.csv"
NUM_TASKS="${NUM_TASKS:-4}"

if ! [[ "$NUM_TASKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_TASKS must be a positive integer, got '${NUM_TASKS}'"
    exit 1
fi

echo "============================================================"
echo "  SIGNOR Direct-Mode Evaluation Job"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Task ID:     ${TASK_ID}"
echo "  Run Tag:     ${RUN_TAG}"
echo "  Single mode: ${SINGLE_MODE:-false}"
echo "  Num tasks:   ${NUM_TASKS}"
echo "  vLLM seqs:   ${VLLM_MAX_NUM_SEQS:-8}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Row range for this array task (ignored in single mode)
# Compute a balanced partition from the current dataset size.
# ---------------------------------------------------------------------------
TOTAL_ROWS=$(python3 - "$DATASET_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline='') as handle:
    print(max(sum(1 for _ in csv.reader(handle)) - 1, 0))
PY
)

if (( TASK_ID >= 0 && TASK_ID < NUM_TASKS )); then
    BASE_ROWS=$((TOTAL_ROWS / NUM_TASKS))
    REMAINDER=$((TOTAL_ROWS % NUM_TASKS))

    if (( TASK_ID < REMAINDER )); then
        ROW_LIMIT=$((BASE_ROWS + 1))
        ROW_START=$((TASK_ID * ROW_LIMIT))
    else
        ROW_LIMIT=$BASE_ROWS
        ROW_START=$((REMAINDER * (BASE_ROWS + 1) + (TASK_ID - REMAINDER) * BASE_ROWS))
    fi
else
    ROW_START=0
    ROW_LIMIT=0
fi

echo "  Dataset rows: ${TOTAL_ROWS}"
echo "  Task rows:    start=${ROW_START} limit=${ROW_LIMIT}"

if [[ "${SINGLE_MODE:-false}" == "true" ]]; then
    ROW_START=0
    ROW_LIMIT=0
fi

# ---------------------------------------------------------------------------
# Read subagent model from config
# ---------------------------------------------------------------------------
CONFIG_NAME="${CONFIG_NAME:-signor_direct_config}"
CONFIG_FILE="${PROJECT_ROOT}/experiments/configs/${CONFIG_NAME}.yaml"
export CONFIG_NAME
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
    HEALTH_MAX=1800  # 30-minute cap; avoids hanging forever if vLLM crashes
    until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM process (PID ${VLLM_PID}) died before becoming healthy. Exiting."
            exit 1
        fi
        if [[ $HEALTH_WAIT -ge $HEALTH_MAX ]]; then
            echo "ERROR: vLLM did not become healthy after ${HEALTH_MAX}s. Exiting."
            exit 1
        fi
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
echo "      Eval args: ${EXTRA_ARGS:-<none>}"

if [[ "${SINGLE_MODE:-false}" == "true" ]]; then
    OUTPUT_CSV="${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}/results.csv"
else
    OUTPUT_CSV="${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}/results_chunk${TASK_ID}.csv"
fi

cd "$PROJECT_ROOT"

RUN_ARGS="--run-tag ${RUN_TAG} --output-csv ${OUTPUT_CSV}"
if [[ $ROW_START -gt 0 ]]; then
    RUN_ARGS="${RUN_ARGS} --row-start ${ROW_START}"
fi
if [[ $ROW_LIMIT -gt 0 ]]; then
    RUN_ARGS="${RUN_ARGS} --limit ${ROW_LIMIT}"
fi

# shellcheck disable=SC2086
bash "${PROJECT_ROOT}/scripts/run_signor_direct_batch.sh" ${RUN_ARGS} ${EXTRA_ARGS:-}

echo "============================================================"
echo "  Task ${TASK_ID} complete!"
echo "  Results: ${OUTPUT_CSV}"
echo "============================================================"
