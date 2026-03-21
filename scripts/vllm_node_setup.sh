#!/bin/bash
# =============================================================================
# vllm_node_setup.sh — Runs on the compute node inside srun
#
# Detects GPUs and available port, writes connection info to a shared file,
# then starts the vLLM server. Called automatically by start_vllm_ihpc.sh.
# =============================================================================
set -euo pipefail

INFO_FILE="${1:?Usage: $0 <info_file> [model]}"
MODEL_NAME="${2:-qwen3-8b}"

if [[ "$MODEL_NAME" == /* ]]; then
    MODEL="$MODEL_NAME"
else
    MODEL="/hps/nobackup/saezrodriguez/hf_models/${MODEL_NAME}"
fi

hostname_val=$(hostname)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. No GPUs available."
    exit 1
fi

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ *//')

# ---------------------------------------------------------------------------
# Find an available port (starting from 8000)
# ---------------------------------------------------------------------------
used_ports=$(ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | rev | cut -d: -f1 | rev | sort -n | uniq)
available_port=""
for port in $(seq 8000 30000); do
    if ! echo "$used_ports" | grep -qw "$port"; then
        available_port=$port
        break
    fi
done

if [[ -z "$available_port" ]]; then
    echo "ERROR: No available port found in range 8000-30000."
    exit 1
fi

# ---------------------------------------------------------------------------
# Write connection info so the launcher script can pick it up
# ---------------------------------------------------------------------------
echo "${hostname_val}:${available_port}:${gpu_count}" > "$INFO_FILE"

echo "============================================================"
echo "  vLLM starting on compute node"
echo "============================================================"
echo "  Host:  ${hostname_val}"
echo "  Port:  ${available_port}"
echo "  GPUs:  ${gpu_count}  (${gpu_info})"
echo "  Model: ${MODEL}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Determine tool-call parser and reasoning parser based on model
# ---------------------------------------------------------------------------
MODEL_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
TOOL_CALL_PARSER=""
REASONING_PARSER=""

case "$MODEL_LOWER" in
    # --- OpenAI gpt-oss family ---
    *openai/gpt-oss* | *gpt-oss*)
        TOOL_CALL_PARSER="openai"
        REASONING_PARSER="openai_gptoss"
        ;;
    # --- Qwen3-Coder (XML-based tool calls, must match before generic Qwen3) ---
    *qwen3*coder*)
        TOOL_CALL_PARSER="qwen3_xml"
        REASONING_PARSER="qwen3"
        ;;
    # --- Qwen3 / Qwen3.5 / QwQ (thinking models, Hermes-style tool calls) ---
    *qwen3* | *qwq*)
        TOOL_CALL_PARSER="hermes"
        REASONING_PARSER="qwen3"
        ;;
    # --- Qwen2.5 / Qwen2 (non-thinking, Hermes-style tool calls) ---
    *qwen2*)
        TOOL_CALL_PARSER="hermes"
        ;;
    # --- GLM-4.7 ---
    *glm-4.7* | *glm4.7*)
        TOOL_CALL_PARSER="glm47"
        ;;
    # --- GLM-4.5 / GLM-4.6 ---
    *glm-4.5* | *glm-4.6* | *glm4.5* | *glm4.6*)
        TOOL_CALL_PARSER="glm45"
        ;;
    # --- DeepSeek R1 ---
    *deepseek*r1*)
        TOOL_CALL_PARSER="deepseek_v3"
        REASONING_PARSER="deepseek_r1"
        ;;
    # --- DeepSeek V3 ---
    *deepseek*v3*)
        TOOL_CALL_PARSER="deepseek_v3"
        REASONING_PARSER="deepseek_v3"
        ;;
    # --- Llama 3.x / 4 ---
    *llama-3* | *llama3*)
        TOOL_CALL_PARSER="llama3_json"
        ;;
    *llama-4* | *llama4*)
        TOOL_CALL_PARSER="llama4_pythonic"
        ;;
    # --- Mistral ---
    *mistral*)
        TOOL_CALL_PARSER="mistral"
        ;;
    *)
        echo "WARNING: Unknown model family '${MODEL}'."
        echo "         No tool-call or reasoning parsers configured."
        echo "         The model will serve without tool calling or reasoning support."
        ;;
esac

echo "  Parser config:"
echo "    Tool-call parser:  ${TOOL_CALL_PARSER:-none}"
echo "    Reasoning parser:  ${REASONING_PARSER:-none}"
echo ""

# ---------------------------------------------------------------------------
# Launch vLLM via Singularity
# ---------------------------------------------------------------------------
export saez_home=/hps/nobackup/saezrodriguez
export HF_HOME="$saez_home/shared_hf_home"

SINGULARITY_IMAGE="$saez_home/singularity_images/vllm-openai_latest.sif"
if [[ ! -f "$SINGULARITY_IMAGE" ]]; then
    echo "ERROR: Singularity image not found: ${SINGULARITY_IMAGE}"
    exit 1
fi

# Build serve arguments conditionally
SERVE_ARGS=(
    "$MODEL"
    --served-model-name "${MODEL_NAME}" "./${MODEL_NAME}" "${MODEL}"
    --tensor-parallel-size "$gpu_count"
    --port "$available_port"
    --gpu-memory-utilization 0.8
    --trust-remote-code
    --max-num-seqs 8
)
if [[ -n "$TOOL_CALL_PARSER" ]]; then
    SERVE_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
    SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

singularity exec --nv "$SINGULARITY_IMAGE" \
    vllm serve "${SERVE_ARGS[@]}"
