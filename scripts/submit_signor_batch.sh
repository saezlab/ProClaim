#!/bin/bash
# =============================================================================
# submit_signor_batch.sh
#
# One-click SIGNOR evaluation from your local laptop.
# Automatically starts vLLM server (if needed) and submits the batch job.
#
# Prerequisites:
#   - SSH access to EBI HPC configured
#
# Architecture Note (vLLM Background Caching):
#   The first time you run this, it will start a vLLM server in the background 
#   (--detach) and patiently wait for the model to fully load. 
#   Subsequent runs will DETECT this background server and instantly submit 
#   the new evaluation jobs to SLURM without reloading the model.
#   When you are completely finished with all evaluations, YOU MUST STOP 
#   the background server manually to release the GPU node:
#       bash scripts/start_vllm_ihpc.sh --stop
#
# Usage:
#   bash scripts/submit_signor_batch.sh                  # One-click: auto-start vLLM + submit job
#   bash scripts/submit_signor_batch.sh --limit 1        # Test with 1 claim
#   bash scripts/submit_signor_batch.sh --reps 1         # 1 repetition only
#   bash scripts/submit_signor_batch.sh --no-vllm        # Skip vLLM startup (use existing server)
#   bash scripts/submit_signor_batch.sh --status         # Check job status
#   bash scripts/submit_signor_batch.sh --cancel         # Cancel running job
#   bash scripts/submit_signor_batch.sh --logs           # View job logs
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
EBI_USER="${EBI_USER:-wuy}"
LOGIN_HOST="ihpc.ebi.ac.uk"
LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"

# SLURM job settings
JOB_NAME="signor-eval"
TIME_LIMIT="80:00:00"
CPUS=8
MEM="64G"
GPU_TYPE="a100"
GPU_COUNT=1

# vLLM settings
VLLM_TIME="80:00:00"
VLLM_GPUS=1
VLLM_MODEL="qwen3.5-9b"
SKIP_VLLM=false

# Project paths (on HPC)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct"
JOB_SCRIPT="${PROJECT_ROOT}/scripts/slurm_signor_job.sh"
JOB_INFO_FILE="${PROJECT_ROOT}/.signor_job_info"
LOG_DIR="${PROJECT_ROOT}/results/slurm_logs"

# Action
ACTION="submit"
EXTRA_ARGS=""

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)      EBI_USER="$2";    shift 2 ;;
        --status)    ACTION="status";  shift ;;
        --cancel)    ACTION="cancel";  shift ;;
        --logs)      ACTION="logs";    shift ;;
        --no-vllm)   SKIP_VLLM=true;   shift ;;
        --limit|--reps)
            EXTRA_ARGS="${EXTRA_ARGS} $1 $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "One-click execution (auto-starts vLLM if needed):"
            echo "  bash $0                # Full batch (all claims)"
            echo "  bash $0 --limit 1      # Test with 1 claim"
            echo ""
            echo "Actions:"
            echo "  --status     Check job status"
            echo "  --cancel     Cancel running job"
            echo "  --logs       View job logs"
            echo ""
            echo "Job parameters:"
            echo "  --limit N    Limit to N claims (default: all 67)"
            echo "  --reps N     Number of repetitions per claim (default: 3)"
            echo "  --no-vllm    Skip vLLM startup check (use existing server)"
            echo "  --user USER  EBI username (default: wuy)"
            echo ""
            echo "  -h, --help   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- SSH helpers ------------------------------------------------------------
ssh_login() {
    ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 "$LOGIN_NODE" "$@"
}

# ---- STATUS -----------------------------------------------------------------
if [[ "$ACTION" == "status" ]]; then
    echo "Checking job status..."
    echo ""

    # Try to read job ID from info file
    JOB_ID=$(ssh_login "cat ${JOB_INFO_FILE} 2>/dev/null || echo ''")

    if [[ -z "$JOB_ID" ]]; then
        echo "No active job found."
        echo "Submit a new job with: bash $0"
        exit 0
    fi

    echo "Job ID: $JOB_ID"
    echo ""
    ssh_login "squeue -j ${JOB_ID} -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R' || echo 'Job not in queue (might be completed)'"
    echo ""

    # Check if results CSV exists
    RESULTS_CSV="${PROJECT_ROOT}/results/signor_eval_results.csv"
    echo "Results progress:"
    ssh_login "if [[ -f ${RESULTS_CSV} ]]; then wc -l ${RESULTS_CSV}; else echo 'No results yet'; fi"

    exit 0
fi

# ---- CANCEL -----------------------------------------------------------------
if [[ "$ACTION" == "cancel" ]]; then
    echo "Cancelling all jobs named ${JOB_NAME} for user ${EBI_USER}..."
    ssh_login "scancel -u ${EBI_USER} -n ${JOB_NAME} 2>/dev/null || true"
    ssh_login "rm -f ${JOB_INFO_FILE} 2>/dev/null || true"
    echo "Jobs cancelled."
    exit 0
fi

# ---- LOGS -------------------------------------------------------------------
if [[ "$ACTION" == "logs" ]]; then
    JOB_ID=$(ssh_login "cat ${JOB_INFO_FILE} 2>/dev/null || echo ''")

    if [[ -z "$JOB_ID" ]]; then
        echo "No job info found."
        exit 0
    fi

    LOG_FILE="${LOG_DIR}/signor-eval-${JOB_ID}.out"

    echo "Following log file: ${LOG_FILE}"
    echo "Press Ctrl-C to stop following"
    echo "============================================================"
    echo ""

    ssh_login "tail -f ${LOG_FILE}"
    exit 0
fi

# ---- SUBMIT -----------------------------------------------------------------
echo "============================================================"
echo "  One-Click SIGNOR Evaluation"
echo "============================================================"
echo "  User:       ${EBI_USER}"
echo "  Host:       ${LOGIN_HOST}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  GPUs:       ${GPU_COUNT}x ${GPU_TYPE}"
echo "  CPUs:       ${CPUS}"
echo "  Memory:     ${MEM}"
echo "============================================================"
echo ""

echo "[0/2] Cleaning up any old/zombie Evaluation tasks..."
ssh_login "scancel -u ${EBI_USER} -n ${JOB_NAME} 2>/dev/null || true"
echo ""

# Step 1: Check and auto-start vLLM server if needed
if [[ "$SKIP_VLLM" == "false" ]]; then
    echo "[1/2] Checking vLLM server status..."
    VLLM_INFO=$(ssh_login "cat ${PROJECT_ROOT}/.vllm_server_info 2>/dev/null || echo ''")

    if [[ -z "$VLLM_INFO" ]]; then
        echo "  No vLLM server found. Starting vLLM server automatically..."
        echo ""

        # Auto-start vLLM using start_vllm_ihpc.sh
        if [[ -f "${SCRIPT_DIR}/start_vllm_ihpc.sh" ]]; then
            echo "  Executing: bash ${SCRIPT_DIR}/start_vllm_ihpc.sh --time ${VLLM_TIME} --gpus ${VLLM_GPUS} --model ${VLLM_MODEL}"
            echo ""
            bash "${SCRIPT_DIR}/start_vllm_ihpc.sh" \
                --user "${EBI_USER}" \
                --time "${VLLM_TIME}" \
                --gpus "${VLLM_GPUS}" \
                --model "${VLLM_MODEL}" \
                --detach

            echo ""
            echo "  vLLM startup initiated. Waiting for server to be ready..."
            echo ""

            # Wait for vLLM server to create .vllm_server_info and be fully ready
            MAX_WAIT=60  # Maximum wait time in seconds (10 seconds * 60 = 10 minutes)
            WAIT_COUNT=0
            VLLM_READY=false

            while [[ $WAIT_COUNT -lt $MAX_WAIT ]]; do
                VLLM_CHECK=$(ssh_login "cat ${PROJECT_ROOT}/.vllm_server_info 2>/dev/null || echo ''")

                if [[ -n "$VLLM_CHECK" ]]; then
                    # File exists, now verify server is actually responding
                    COMPUTE_HOST=$(echo "$VLLM_CHECK" | cut -d: -f1)
                    REMOTE_PORT=$(echo "$VLLM_CHECK" | cut -d: -f2)

                    # Test if vLLM API is responding
                    API_CHECK=$(ssh_login "curl -s --max-time 5 http://${COMPUTE_HOST}:${REMOTE_PORT}/v1/models 2>/dev/null | grep -o '\"object\":\"list\"' || echo ''")

                    if [[ -n "$API_CHECK" ]]; then
                        echo "  ✓ vLLM server is ready and responding!"
                        echo "    Node: ${COMPUTE_HOST}"
                        echo "    Port: ${REMOTE_PORT}"
                        VLLM_READY=true
                        break
                    fi
                fi

                WAIT_COUNT=$((WAIT_COUNT + 1))
                printf "  ⏳ Waiting for vLLM... (%d/%d, %ds elapsed)\r" "$WAIT_COUNT" "$MAX_WAIT" "$((WAIT_COUNT * 10))"
                sleep 10
            done

            echo ""  # New line after progress indicator

            if [[ "$VLLM_READY" == "false" ]]; then
                echo "  ✗ ERROR: vLLM server did not become ready within $((MAX_WAIT * 10)) seconds"
                echo "  Please check vLLM logs and try again, or use --no-vllm if server is already running"
                exit 1
            fi

            echo ""
            echo "  vLLM server started successfully!"
        else
            echo "  ERROR: start_vllm_ihpc.sh not found at ${SCRIPT_DIR}/start_vllm_ihpc.sh"
            echo "  Please start vLLM manually or use --no-vllm flag"
            exit 1
        fi
    else
        echo "  vLLM server is already running:"
        COMPUTE_HOST=$(echo "$VLLM_INFO" | cut -d: -f1)
        REMOTE_PORT=$(echo "$VLLM_INFO" | cut -d: -f2)
        ACTUAL_GPUS=$(echo "$VLLM_INFO" | cut -d: -f3)
        echo "    Node: ${COMPUTE_HOST}"
        echo "    Port: ${REMOTE_PORT}"
        echo "    GPUs: ${ACTUAL_GPUS}"
    fi
    echo ""
else
    echo "[1/2] Skipping vLLM check (--no-vllm flag used)"
    echo ""
fi

echo "[2/2] Submitting SIGNOR evaluation job..."
echo ""

# Verify vLLM server connectivity before submitting job
echo "Verifying vLLM server connectivity..."
VLLM_INFO=$(ssh_login "cat ${PROJECT_ROOT}/.vllm_server_info 2>/dev/null || echo ''")

if [[ -n "$VLLM_INFO" ]]; then
    COMPUTE_HOST=$(echo "$VLLM_INFO" | cut -d: -f1)
    REMOTE_PORT=$(echo "$VLLM_INFO" | cut -d: -f2)

    # Test if vLLM API is responding
    API_CHECK=$(ssh_login "curl -s --max-time 10 http://${COMPUTE_HOST}:${REMOTE_PORT}/v1/models 2>/dev/null | grep -o '\"object\":\"list\"' || echo ''")

    if [[ -z "$API_CHECK" ]]; then
        echo "  ✗ ERROR: vLLM server not responding at ${COMPUTE_HOST}:${REMOTE_PORT}"
        echo "  Please check vLLM logs or restart the server with:"
        echo "    bash scripts/start_vllm_ihpc.sh --stop"
        echo "    bash scripts/start_vllm_ihpc.sh --time ${VLLM_TIME} --gpus ${VLLM_GPUS} --model ${VLLM_MODEL} --detach"
        exit 1
    fi

    echo "  ✓ vLLM server is responding at ${COMPUTE_HOST}:${REMOTE_PORT}"
else
    echo "  ⚠ Warning: No vLLM server info found (.vllm_server_info missing)"
    echo "  If you're using an external vLLM server, make sure it's configured in example_config.yaml"
fi
echo ""

# Create log directory on HPC
ssh_login "mkdir -p ${LOG_DIR}"

# Generate SLURM job script on-the-fly and submit it
echo "Submitting batch job..."

JOB_ID=$(ssh_login "sbatch \
    --job-name=${JOB_NAME} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --gres=gpu:${GPU_TYPE}:${GPU_COUNT} \
    --output=${LOG_DIR}/signor-eval-%j.out \
    --error=${LOG_DIR}/signor-eval-%j.err \
    --wrap=\"cd ${PROJECT_ROOT} && bash scripts/run_signor_batch.sh ${EXTRA_ARGS}\" \
    | grep -oP 'Submitted batch job \K[0-9]+'"
)

if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

# Save job ID
ssh_login "echo ${JOB_ID} > ${JOB_INFO_FILE}"

echo ""
echo "============================================================"
echo "  Job submitted successfully!"
echo "============================================================"
echo "  Job ID:     ${JOB_ID}"
echo "  Log file:   ${LOG_DIR}/signor-eval-${JOB_ID}.out"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "  Monitor progress:"
echo "    bash $0 --status    # Check job status"
echo "    bash $0 --logs      # Follow logs in real-time"
echo ""
echo "  Cancel job:"
echo "    bash $0 --cancel"
echo ""
echo "  Download results when complete:"
echo "    scp ${LOGIN_NODE}:${PROJECT_ROOT}/results/signor_eval_results.csv ."
echo ""
echo "The vLLM server will keep running. Stop it when done:"
echo "  bash scripts/start_vllm_ihpc.sh --stop"
echo ""
