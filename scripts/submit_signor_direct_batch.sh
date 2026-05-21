#!/bin/bash
# =============================================================================
# submit_signor_direct_batch.sh
#
# Submit SIGNOR evaluation using the direct mode pipeline
# (LiteLLM outer agent + bash + jupytext; vLLM subagent on the same GPU).
#
# Default mode: 4-chunk parallel (--array=0-3), one GPU per task, followed by
# an auto-merge job that concatenates the chunk CSVs.
# Use --single to run a single job (all rows, one GPU).
#
# Usage:
#   bash scripts/submit_signor_direct_batch.sh                  # 4-GPU parallel (default)
#   bash scripts/submit_signor_direct_batch.sh --single         # Single-GPU mode
#   bash scripts/submit_signor_direct_batch.sh --single --limit 1 --reps 1  # test
#   bash scripts/submit_signor_direct_batch.sh --reps 1         # 1 repetition
#   bash scripts/submit_signor_direct_batch.sh --num-tasks 2    # 2-GPU parallel
#   bash scripts/submit_signor_direct_batch.sh --workers 2 --worker-shards --max-num-seqs 16
#   bash scripts/submit_signor_direct_batch.sh --status
#   bash scripts/submit_signor_direct_batch.sh --cancel
#   bash scripts/submit_signor_direct_batch.sh --logs
#   bash scripts/submit_signor_direct_batch.sh --resume               # Resume last failed run
#   bash scripts/submit_signor_direct_batch.sh --run-tag <TAG_ID> --reps 1 # Resume specific run
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
EBI_USER="${EBI_USER:-wuy}"
LOGIN_HOST="ihpc.ebi.ac.uk"
LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"

# SLURM job settings
JOB_NAME="signor-direct"
TIME_LIMIT="24:00:00"
CPUS=8
MEM="64G"
GPU_TYPE="a100"
GPU_COUNT=1  # per array task; parallel mode uses 4 GPUs total via --array=0-3 (do NOT set to 4)
NUM_TASKS=4

# Project paths (on HPC)
PROJECT_ROOT_OVERRIDE="${GRN_LLM_CORRECT_PROJECT_ROOT:-}"
PROJECT_ROOT=""
JOB_SCRIPT=""
JOB_INFO_FILE=""
LOG_DIR=""
SSH_CONTROL_DIR=""
SSH_CONTROL_PATH=""
SSH_MASTER_READY=false

ACTION="submit"
MODE="parallel"
REPS_ARG=""
LIMIT_ARG=""
WORKERS_ARG=""
WORKER_SHARDS_ARG=""
MAX_NUM_SEQS=""
RESUME_TAG=""       # set by --resume or --run-tag; empty = generate fresh tag

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)    EBI_USER="$2"; LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"; shift 2 ;;
        --status)  ACTION="status";  shift ;;
        --cancel)  ACTION="cancel";  shift ;;
        --logs)    ACTION="logs";    shift ;;
        --single)  MODE="single";    shift ;;
        --time)    TIME_LIMIT="$2"; shift 2 ;;
        --reps)    REPS_ARG="--reps $2"; shift 2 ;;
        --limit)   LIMIT_ARG="--limit $2"; shift 2 ;;
        --workers) WORKERS_ARG="--workers $2"; shift 2 ;;
        --worker-shards) WORKER_SHARDS_ARG="--worker-shards"; shift ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --num-tasks) NUM_TASKS="$2"; shift 2 ;;
        --resume)  RESUME_TAG="__auto__"; shift ;;
        --run-tag)
            RESUME_TAG="$2"
            if [[ -z "$RESUME_TAG" ]]; then
                echo "ERROR: --run-tag requires a non-empty value"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Submission modes:"
            echo "  (default)          4-chunk parallel: 4 GPUs, auto-merged results"
            echo "  --single           Single-GPU mode: all rows on one GPU"
            echo ""
            echo "Resume:"
            echo "  --resume           Reuse RUN_TAG from last run (reads .signor_direct_job_info)"
            echo "  --run-tag TAG      Reuse an explicit RUN_TAG (e.g. 20260419_135117)"
            echo "  Both modes append to existing chunk CSVs and skip already-completed cases."
            echo ""
            echo "Actions:"
            echo "  --status     Check job status"
            echo "  --cancel     Cancel running jobs"
            echo "  --logs       View job logs"
            echo ""
            echo "Job parameters:"
            echo "  --time HH:MM:SS  Slurm wall-time limit (default: 24:00:00)"
            echo "  --reps N     Number of repetitions per claim (default: 3)"
            echo "  --limit N    Limit to N rows (single mode only)"
            echo "  --num-tasks N  Number of parallel GPU tasks (default: 4; ignored with --single)"
            echo "  --workers N  Concurrent claim repetitions per GPU task (default: 1)"
            echo "  --worker-shards  Write one result CSV per worker, then merge into the task CSV"
            echo "  --max-num-seqs N  vLLM max concurrent sequences per GPU task (default: 8)"
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

if ! [[ "$NUM_TASKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --num-tasks must be a positive integer"
    exit 1
fi

if [[ -n "$PROJECT_ROOT_OVERRIDE" ]]; then
    PROJECT_ROOT="$PROJECT_ROOT_OVERRIDE"
else
    PROJECT_ROOT="/hps/nobackup/saezrodriguez/${EBI_USER}/workspace/grn-llm-correct"
fi
JOB_SCRIPT="${PROJECT_ROOT}/scripts/slurm_signor_direct_job.sh"
JOB_INFO_FILE="${PROJECT_ROOT}/.signor_direct_job_info"
LOG_DIR="${PROJECT_ROOT}/results/slurm_logs"
SSH_CONTROL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/grn-llm-correct-ssh-XXXXXX")
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/control"

cleanup_ssh_master() {
    if [[ "$SSH_MASTER_READY" == "true" ]]; then
        ssh \
            -o ControlPath="${SSH_CONTROL_PATH}" \
            -O exit \
            "$LOGIN_NODE" >/dev/null 2>&1 || true
    fi
    if [[ -n "$SSH_CONTROL_DIR" && -d "$SSH_CONTROL_DIR" ]]; then
        rm -rf "$SSH_CONTROL_DIR"
    fi
}
trap cleanup_ssh_master EXIT

# ---- SSH helper -------------------------------------------------------------
ensure_ssh_master() {
    if [[ "$SSH_MASTER_READY" == "true" ]]; then
        return
    fi

    ssh \
        -o ConnectTimeout=10 \
        -o ServerAliveInterval=30 \
        -o ControlMaster=yes \
        -o ControlPersist=600 \
        -o ControlPath="${SSH_CONTROL_PATH}" \
        -Nf \
        "$LOGIN_NODE"
    SSH_MASTER_READY=true
}

ssh_login() {
    ensure_ssh_master
    ssh \
        -o ConnectTimeout=10 \
        -o ServerAliveInterval=30 \
        -o ControlMaster=auto \
        -o ControlPersist=600 \
        -o ControlPath="${SSH_CONTROL_PATH}" \
        "$LOGIN_NODE" "$@"
}

read_job_info() {
    ssh_login "cat ${JOB_INFO_FILE} 2>/dev/null || echo ''"
}

# ---- STATUS -----------------------------------------------------------------
if [[ "$ACTION" == "status" ]]; then
    echo "Checking job status..."
    JOB_INFO=$(read_job_info)
    if [[ -z "$JOB_INFO" ]]; then
        echo "No active job found. Submit with: bash $0"
        exit 0
    fi
    ARRAY_JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
    MERGE_JOB_ID=$(echo "$JOB_INFO" | awk '{print $2}')
    RUN_TAG=$(echo "$JOB_INFO" | awk '{print $3}')
    echo "Array job ID: ${ARRAY_JOB_ID}"
    echo "Merge job ID: ${MERGE_JOB_ID}"
    echo "Run tag:      ${RUN_TAG}"
    echo ""
    ssh_login "squeue -j ${ARRAY_JOB_ID},${MERGE_JOB_ID} -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R' 2>/dev/null || echo 'Jobs not in queue (may be complete)'"
    echo ""
    RESULTS_DIR="${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}"
    echo "Results progress:"
    ssh_login "
        for f in ${RESULTS_DIR}/results*.csv; do
            [[ -f \"\$f\" ]] && echo \"\$(wc -l < \$f) rows: \$f\" || true
        done
        [[ -f ${RESULTS_DIR}/results.csv ]] || echo 'Final results.csv not yet available'
    "
    exit 0
fi

# ---- CANCEL -----------------------------------------------------------------
if [[ "$ACTION" == "cancel" ]]; then
    JOB_INFO=$(read_job_info)
    if [[ -n "$JOB_INFO" ]]; then
        ARRAY_JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
        MERGE_JOB_ID=$(echo "$JOB_INFO" | awk '{print $2}')
        echo "Cancelling jobs ${ARRAY_JOB_ID} and ${MERGE_JOB_ID}..."
        ssh_login "scancel ${ARRAY_JOB_ID} ${MERGE_JOB_ID} 2>/dev/null || true"
    fi
    ssh_login "scancel -u ${EBI_USER} -n ${JOB_NAME} 2>/dev/null || true"
    ssh_login "rm -f ${JOB_INFO_FILE}"
    echo "Jobs cancelled."
    exit 0
fi

# ---- LOGS -------------------------------------------------------------------
if [[ "$ACTION" == "logs" ]]; then
    JOB_INFO=$(read_job_info)
    if [[ -z "$JOB_INFO" ]]; then
        echo "No job info found."
        exit 0
    fi
    ARRAY_JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
    LOG_PATTERN="${LOG_DIR}/signor-direct-${ARRAY_JOB_ID}*.out"
    echo "Following logs: ${LOG_PATTERN}"
    echo "Press Ctrl-C to stop"
    echo "============================================================"
    ssh_login "tail -f ${LOG_PATTERN}"
    exit 0
fi

# ---- SUBMIT -----------------------------------------------------------------
# Resolve --resume: look up the last RUN_TAG from the job info file
if [[ "$RESUME_TAG" == "__auto__" ]]; then
    RESUME_TAG=$(ssh_login "awk '{print \$3}' ${JOB_INFO_FILE} 2>/dev/null || echo ''")
    if [[ -z "$RESUME_TAG" ]]; then
        echo "ERROR: --resume requested but ${JOB_INFO_FILE} not found or empty on ${LOGIN_HOST}"
        exit 1
    fi
fi

if [[ -n "$RESUME_TAG" ]]; then
    RUN_TAG="$RESUME_TAG"
    echo "Resuming run: ${RUN_TAG}"
    echo "  (skipping completed cases found in existing chunk CSVs)"
else
    RUN_TAG=$(date +%Y%m%d_%H%M%S)
fi

EXTRA_ARGS="${REPS_ARG}"
if [[ "$MODE" == "single" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} ${LIMIT_ARG}"
fi

if [[ "$MODE" != "single" && -n "$WORKER_SHARDS_ARG" ]]; then
    echo "ERROR: --worker-shards is only supported with --single in SIGNOR direct mode."
    exit 1
fi

EXTRA_ARGS="${EXTRA_ARGS} ${WORKERS_ARG}"
EXTRA_ARGS="${EXTRA_ARGS} ${WORKER_SHARDS_ARG}"
EXTRA_ARGS="${EXTRA_ARGS# }"

if [[ -z "$MAX_NUM_SEQS" ]]; then
    VLLM_MAX_NUM_SEQS=8
else
    VLLM_MAX_NUM_SEQS="$MAX_NUM_SEQS"
fi

echo "============================================================"
echo "  SIGNOR Direct-Mode Evaluation Submission"
echo "============================================================"
echo "  User:       ${EBI_USER}"
echo "  Host:       ${LOGIN_HOST}"
echo "  Mode:       ${MODE}"
echo "  Run tag:    ${RUN_TAG}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  GPUs:       ${GPU_COUNT}x ${GPU_TYPE} per task"
[[ "$MODE" == "parallel" ]] && echo "  Parallel tasks: ${NUM_TASKS}"
echo "  CPUs:       ${CPUS}"
echo "  Memory:     ${MEM}"
echo "  vLLM seqs:  ${VLLM_MAX_NUM_SEQS}"
[[ -n "$EXTRA_ARGS" ]] && echo "  Extra args: ${EXTRA_ARGS}"
echo "============================================================"
echo ""

echo "Cleaning up any old ${JOB_NAME} tasks..."
ssh_login "scancel -u ${EBI_USER} -n ${JOB_NAME} 2>/dev/null || true"
ssh_login "mkdir -p ${LOG_DIR}"
echo ""

if [[ "$MODE" == "parallel" ]]; then
    ARRAY_FLAG="--array=0-$((NUM_TASKS - 1))"
    SINGLE_MODE_VAL="false"
    echo "Submitting ${NUM_TASKS}-task array job..."
else
    ARRAY_FLAG=""
    SINGLE_MODE_VAL="true"
    echo "Submitting single-GPU job..."
fi

ARRAY_JOB_ID=$(ssh_login "
    export RUN_TAG='${RUN_TAG}'
    export EXTRA_ARGS='${EXTRA_ARGS}'
    export SINGLE_MODE='${SINGLE_MODE_VAL}'
    export VLLM_MAX_NUM_SEQS='${VLLM_MAX_NUM_SEQS}'
    export NUM_TASKS='${NUM_TASKS}'
    export GRN_LLM_CORRECT_PROJECT_ROOT='${PROJECT_ROOT}'
    sbatch \
        --job-name=${JOB_NAME} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --gres=gpu:${GPU_TYPE}:${GPU_COUNT} \
        --output=${LOG_DIR}/signor-direct-%j_%a.out \
        --error=${LOG_DIR}/signor-direct-%j_%a.err \
        --export=ALL \
        ${ARRAY_FLAG} \
        ${JOB_SCRIPT} \
    | grep -oP 'Submitted batch job \K[0-9]+'
")

if [[ -z "$ARRAY_JOB_ID" ]]; then
    echo "ERROR: Failed to submit array job"
    exit 1
fi
echo "  Array job ID: ${ARRAY_JOB_ID}"

MERGE_JOB_ID="none"
if [[ "$MODE" == "parallel" ]]; then
    echo "Submitting merge job (depends on array job)..."
    RESULTS_DIR="${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}"
    MERGE_JOB_ID=$(ssh_login "
        sbatch \
            --job-name=${JOB_NAME}-merge \
            --time=00:15:00 \
            --cpus-per-task=2 \
            --mem=8G \
            --dependency=afterok:${ARRAY_JOB_ID} \
            --output=${LOG_DIR}/signor-direct-merge-%j.out \
            --error=${LOG_DIR}/signor-direct-merge-%j.err \
            --wrap=\"cd ${PROJECT_ROOT} && uv run python -c \\\"
import glob, pandas as pd, sys
files = sorted(glob.glob('${RESULTS_DIR}/results_chunk*.csv'))
if not files:
    print('ERROR: no chunk CSVs found in ${RESULTS_DIR}', file=sys.stderr)
    sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('${RESULTS_DIR}/results.csv', index=False)
print(f'Merged {len(files)} chunks ({len(df)} rows) -> ${RESULTS_DIR}/results.csv')
\\\"\" \
        | grep -oP 'Submitted batch job \K[0-9]+'
    ")

    if [[ -z "$MERGE_JOB_ID" ]]; then
        echo "WARNING: Failed to submit merge job."
        MERGE_JOB_ID="none"
    else
        echo "  Merge job ID: ${MERGE_JOB_ID}"
    fi
fi

ssh_login "echo '${ARRAY_JOB_ID} ${MERGE_JOB_ID} ${RUN_TAG}' > ${JOB_INFO_FILE}"

echo ""
echo "============================================================"
echo "  Jobs submitted successfully!"
echo "============================================================"
echo "  Array job ID: ${ARRAY_JOB_ID}"
[[ "$MODE" == "parallel" ]] && echo "  Merge job ID: ${MERGE_JOB_ID}"
echo "  Run tag:      ${RUN_TAG}"
echo "  Results dir:  ${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}/"
echo "  Log dir:      ${LOG_DIR}/"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  bash $0 --status    # Check job status"
echo "  bash $0 --logs      # Follow logs in real-time"
echo ""
echo "Cancel:"
echo "  bash $0 --cancel"
echo ""
echo "Download results when complete:"
echo "  scp ${LOGIN_NODE}:${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}/results.csv ."
echo ""
