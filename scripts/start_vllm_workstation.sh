#!/bin/bash
# =============================================================================
# start_vllm_workstation.sh — One-command vLLM server for a local workstation
#
# Detects GPUs, selects a free port, resolves model-family parsers, then
# launches vLLM directly (no SLURM, no SSH, no Singularity).
#
# Usage:
#   bash scripts/start_vllm_workstation.sh
#   bash scripts/start_vllm_workstation.sh --model Qwen/Qwen3-8B
#   bash scripts/start_vllm_workstation.sh --port 8001 --gpus 2
#   bash scripts/start_vllm_workstation.sh --stop
#   bash scripts/start_vllm_workstation.sh --status
#
# See --help for all options.
# =============================================================================
set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
MODEL="Qwen/Qwen3-8B"
PORT=""                      # auto-detect if empty
GPU_COUNT=1                 # auto-detect if empty
GPU_MEMORY_UTILIZATION=0.8
MAX_NUM_SEQS=8
ACTION="start"
VLLM_BIN="vllm"              # override if vllm is not on PATH

INFO_FILE="${TMPDIR:-/tmp}/.vllm_workstation_info"

# ---- Usage ------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start or manage a local vLLM server on this workstation.

Options:
  --model MODEL      Model name or path to serve       (default: qwen3-8b)
  --port PORT        Port to listen on                 (default: auto, starting 8000)
  --gpus N           Number of GPUs to use             (default: all detected)
  --gpu-mem FRAC     GPU memory utilisation fraction   (default: 0.8)
  --vllm PATH        Path to vllm binary               (default: vllm from PATH)

Actions (mutually exclusive, default is start):
  --stop             Kill the running vLLM server process
  --status           Show server status
  -h, --help         Show this help message
EOF
    exit 0
}

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";                    shift 2 ;;
        --port)     PORT="$2";                     shift 2 ;;
        --gpus)     GPU_COUNT="$2";                shift 2 ;;
        --gpu-mem)  GPU_MEMORY_UTILIZATION="$2";   shift 2 ;;
        --vllm)     VLLM_BIN="$2";                 shift 2 ;;
        --stop)     ACTION="stop";                 shift ;;
        --status)   ACTION="status";               shift ;;
        -h|--help)  usage ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- STOP -------------------------------------------------------------------
if [[ "$ACTION" == "stop" ]]; then
    if [[ ! -f "$INFO_FILE" ]]; then
        echo "No running vLLM server info found (${INFO_FILE})."
        exit 0
    fi
    PID=$(cut -d: -f1 "$INFO_FILE")
    PREV_PORT=$(cut -d: -f2 "$INFO_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID ${PID}, port ${PREV_PORT})..."
        kill "$PID"
        echo "  Done."
    else
        echo "Process ${PID} is not running (may have already stopped)."
    fi
    rm -f "$INFO_FILE"
    exit 0
fi

# ---- STATUS -----------------------------------------------------------------
if [[ "$ACTION" == "status" ]]; then
    if [[ ! -f "$INFO_FILE" ]]; then
        echo "No vLLM server info found."
        exit 0
    fi
    PID=$(cut -d: -f1 "$INFO_FILE")
    PREV_PORT=$(cut -d: -f2 "$INFO_FILE")
    PREV_MODEL=$(cut -d: -f3 "$INFO_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "vLLM server is RUNNING"
        echo "  PID:     ${PID}"
        echo "  Port:    ${PREV_PORT}"
        echo "  Model:   ${PREV_MODEL}"
        echo "  API URL: http://localhost:${PREV_PORT}/v1/"
    else
        echo "vLLM server is STOPPED (stale info file)"
        rm -f "$INFO_FILE"
    fi
    exit 0
fi

# ---- START ------------------------------------------------------------------

# -- GPU detection ------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. No NVIDIA GPUs available."
    exit 1
fi

detected_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ *//')

if [[ -z "$GPU_COUNT" ]]; then
    GPU_COUNT="$detected_gpus"
fi

# -- Port selection -----------------------------------------------------------
if [[ -z "$PORT" ]]; then
    used_ports=$(ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | rev | cut -d: -f1 | rev | sort -n | uniq)
    for candidate in $(seq 8000 30000); do
        if ! echo "$used_ports" | grep -qw "$candidate"; then
            PORT=$candidate
            break
        fi
    done
    if [[ -z "$PORT" ]]; then
        echo "ERROR: No available port found in range 8000-30000."
        exit 1
    fi
fi

# -- Resolve model path -------------------------------------------------------
if [[ "$MODEL" == /* || "$MODEL" == ./* ]]; then
    MODEL_PATH="$MODEL"
    # Handle HuggingFace snapshot paths like:
    #   ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/<hash>/
    # Extract the real model name from the models--Org--Name directory.
    _resolved=$(realpath -s "$MODEL")
    if [[ "$_resolved" == *"/models--"*"/snapshots/"* ]]; then
        _models_dir="${_resolved%%/snapshots/*}"           # .../models--Qwen--Qwen3-8B
        _hf_id=$(basename "$_models_dir" | sed 's/^models--//; s/--/\//g')  # Qwen/Qwen3-8B
        MODEL_NAME=$(basename "$_hf_id")                  # Qwen3-8B
    else
        MODEL_NAME=$(basename "$MODEL")
    fi
else
    MODEL_PATH="$MODEL"
    MODEL_NAME="$MODEL"
fi

# -- Model-family parser detection (mirrors vllm_node_setup.sh) ---------------
MODEL_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
TOOL_CALL_PARSER=""
REASONING_PARSER=""

case "$MODEL_LOWER" in
    *openai/gpt-oss* | *gpt-oss*)
        TOOL_CALL_PARSER="openai"
        REASONING_PARSER="openai_gptoss"
        ;;
    *qwen3*coder*)
        TOOL_CALL_PARSER="qwen3_xml"
        REASONING_PARSER="qwen3"
        ;;
    *qwen3* | *qwq*)
        TOOL_CALL_PARSER="hermes"
        REASONING_PARSER="qwen3"
        ;;
    *qwen2*)
        TOOL_CALL_PARSER="hermes"
        ;;
    *glm-4.7* | *glm4.7*)
        TOOL_CALL_PARSER="glm47"
        ;;
    *glm-4.5* | *glm-4.6* | *glm4.5* | *glm4.6*)
        TOOL_CALL_PARSER="glm45"
        ;;
    *deepseek*r1*)
        TOOL_CALL_PARSER="deepseek_v3"
        REASONING_PARSER="deepseek_r1"
        ;;
    *deepseek*v3*)
        TOOL_CALL_PARSER="deepseek_v3"
        REASONING_PARSER="deepseek_v3"
        ;;
    *llama-3* | *llama3*)
        TOOL_CALL_PARSER="llama3_json"
        ;;
    *llama-4* | *llama4*)
        TOOL_CALL_PARSER="llama4_pythonic"
        ;;
    *mistral*)
        TOOL_CALL_PARSER="mistral"
        ;;
    *)
        echo "WARNING: Unknown model family '${MODEL_NAME}'."
        echo "         No tool-call or reasoning parsers configured."
        ;;
esac

# -- Verify vllm binary -------------------------------------------------------
if ! command -v "$VLLM_BIN" &>/dev/null; then
    echo "ERROR: '${VLLM_BIN}' not found on PATH."
    echo "  Activate your vllm venv first:  source ~/.venvs/vllm/bin/activate"
    echo "  Or pass the binary path:        --vllm /path/to/vllm"
    exit 1
fi

# -- Summary ------------------------------------------------------------------
echo "============================================================"
echo "  Starting vLLM on workstation"
echo "============================================================"
echo "  Model:   ${MODEL_PATH}"
echo "  Port:    ${PORT}"
echo "  GPUs:    ${GPU_COUNT} of ${detected_gpus} detected (${gpu_info})"
echo "  Parsers: tool-call=${TOOL_CALL_PARSER:-none}  reasoning=${REASONING_PARSER:-none}"
echo "============================================================"
echo ""

# -- Build serve arguments ----------------------------------------------------
SERVE_ARGS=(
    "$MODEL_PATH"
    --served-model-name "${MODEL_NAME}" "./${MODEL_NAME}" "${MODEL_PATH}"
    --tensor-parallel-size "$GPU_COUNT"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --trust-remote-code
    --max-num-seqs "$MAX_NUM_SEQS"
)
if [[ -n "$TOOL_CALL_PARSER" ]]; then
    SERVE_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
    SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

# -- Launch -------------------------------------------------------------------
echo "Launching: ${VLLM_BIN} serve ${SERVE_ARGS[*]}"
echo ""
echo "  API URL: http://localhost:${PORT}/v1/"
echo "  Press Ctrl-C to stop."
echo ""

"$VLLM_BIN" serve "${SERVE_ARGS[@]}" &
VLLM_PID=$!

# Write info file so --stop / --status work
echo "${VLLM_PID}:${PORT}:${MODEL_NAME}" > "$INFO_FILE"

cleanup() {
    echo ""
    echo "Stopping vLLM server (PID ${VLLM_PID})..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    rm -f "$INFO_FILE"
    echo "  Done."
}
trap cleanup EXIT INT TERM

wait "$VLLM_PID"
