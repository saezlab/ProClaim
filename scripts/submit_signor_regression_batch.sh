#!/bin/bash
# =============================================================================
# submit_signor_regression_batch.sh
#
# Re-run only the 17 regression cases from the
# signor_direct_eval_20260427 vs 20260604 comparison (single-GPU, --single).
#
# Usage:
#   bash scripts/submit_signor_regression_batch.sh              # 3 reps (default)
#   bash scripts/submit_signor_regression_batch.sh --reps 1     # quick smoke
#   bash scripts/submit_signor_regression_batch.sh --run-tag <TAG>  # resume
#   bash scripts/submit_signor_regression_batch.sh --status
#   bash scripts/submit_signor_regression_batch.sh --cancel
#   bash scripts/submit_signor_regression_batch.sh --logs
# =============================================================================
set -euo pipefail

# ---- Regression case IDs ----------------------------------------------------
REGRESSION_IDS="SIGNOR-138451,SIGNOR-138459,SIGNOR-144163,SIGNOR-175692,SIGNOR-179390,SIGNOR-250006,SIGNOR-251047,SIGNOR-251132,SIGNOR-252589,SIGNOR-260008,SIGNOR-266406,SIGNOR-271751,SIGNOR-272078,SIGNOR-278888,SIGNOR-64676,SIGNOR-70866,SIGNOR-81674"

# ---- Configuration ----------------------------------------------------------
EBI_USER="${EBI_USER:-wuy}"
LOGIN_HOST="ihpc.ebi.ac.uk"
LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"

JOB_NAME="signor-regression"
TIME_LIMIT="12:00:00"
CPUS=8
MEM="64G"
GPU_TYPE="a100"
GPU_COUNT=4

PROJECT_ROOT_OVERRIDE="${GRN_LLM_CORRECT_PROJECT_ROOT:-}"
PROJECT_ROOT=""
JOB_SCRIPT=""
JOB_INFO_FILE=""
LOG_DIR=""
SSH_CONTROL_DIR=""
SSH_CONTROL_PATH=""
SSH_MASTER_READY=false

ACTION="submit"
REPS_ARG=""
RESUME_TAG=""

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)    EBI_USER="$2"; LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"; shift 2 ;;
        --status)  ACTION="status";  shift ;;
        --cancel)  ACTION="cancel";  shift ;;
        --logs)    ACTION="logs";    shift ;;
        --time)    TIME_LIMIT="$2"; shift 2 ;;
        --reps)    REPS_ARG="--reps $2"; shift 2 ;;
        --run-tag) RESUME_TAG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "  --reps N         Repetitions per claim (default: 3)"
            echo "  --run-tag TAG    Resume an existing run tag"
            echo "  --time HH:MM:SS  Slurm wall-time (default: 12:00:00)"
            echo "  --status / --cancel / --logs"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -n "$PROJECT_ROOT_OVERRIDE" ]]; then
    PROJECT_ROOT="$PROJECT_ROOT_OVERRIDE"
else
    PROJECT_ROOT="/hps/nobackup/saezrodriguez/${EBI_USER}/workspace/grn-llm-correct"
fi
JOB_SCRIPT="${PROJECT_ROOT}/scripts/slurm_signor_direct_job.sh"
JOB_INFO_FILE="${PROJECT_ROOT}/.signor_regression_job_info"
LOG_DIR="${PROJECT_ROOT}/results/slurm_logs"
SSH_CONTROL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/grn-llm-correct-ssh-XXXXXX")
SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/control"

cleanup_ssh_master() {
    if [[ "$SSH_MASTER_READY" == "true" ]]; then
        ssh -o ControlPath="${SSH_CONTROL_PATH}" -O exit "$LOGIN_NODE" >/dev/null 2>&1 || true
    fi
    [[ -n "$SSH_CONTROL_DIR" && -d "$SSH_CONTROL_DIR" ]] && rm -rf "$SSH_CONTROL_DIR"
}
trap cleanup_ssh_master EXIT

ensure_ssh_master() {
    [[ "$SSH_MASTER_READY" == "true" ]] && return
    ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 \
        -o ControlMaster=yes -o ControlPersist=600 \
        -o ControlPath="${SSH_CONTROL_PATH}" -Nf "$LOGIN_NODE"
    SSH_MASTER_READY=true
}

ssh_login() {
    ensure_ssh_master
    ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 \
        -o ControlMaster=auto -o ControlPersist=600 \
        -o ControlPath="${SSH_CONTROL_PATH}" \
        "$LOGIN_NODE" "$@"
}

read_job_info() { ssh_login "cat ${JOB_INFO_FILE} 2>/dev/null || echo ''"; }

# ---- STATUS -----------------------------------------------------------------
if [[ "$ACTION" == "status" ]]; then
    JOB_INFO=$(read_job_info)
    if [[ -z "$JOB_INFO" ]]; then echo "No active regression job found."; exit 0; fi
    JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
    RUN_TAG=$(echo "$JOB_INFO" | awk '{print $2}')
    echo "Job ID:  ${JOB_ID}"
    echo "Run tag: ${RUN_TAG}"
    ssh_login "squeue -j ${JOB_ID} -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R' 2>/dev/null || echo 'Job not in queue (may be complete)'"
    RESULTS_DIR="${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}"
    ssh_login "[[ -f ${RESULTS_DIR}/results.csv ]] && wc -l ${RESULTS_DIR}/results.csv || echo 'results.csv not yet available'"
    exit 0
fi

# ---- CANCEL -----------------------------------------------------------------
if [[ "$ACTION" == "cancel" ]]; then
    JOB_INFO=$(read_job_info)
    if [[ -n "$JOB_INFO" ]]; then
        JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
        ssh_login "scancel ${JOB_ID} 2>/dev/null || true"
    fi
    ssh_login "scancel -u ${EBI_USER} -n ${JOB_NAME} 2>/dev/null || true"
    ssh_login "rm -f ${JOB_INFO_FILE}"
    echo "Jobs cancelled."
    exit 0
fi

# ---- LOGS -------------------------------------------------------------------
if [[ "$ACTION" == "logs" ]]; then
    JOB_INFO=$(read_job_info)
    if [[ -z "$JOB_INFO" ]]; then echo "No job info found."; exit 0; fi
    JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
    ssh_login "tail -f ${LOG_DIR}/signor-regression-${JOB_ID}*.out"
    exit 0
fi

# ---- SUBMIT -----------------------------------------------------------------
if [[ -n "$RESUME_TAG" ]]; then
    RUN_TAG="$RESUME_TAG"
    echo "Resuming run: ${RUN_TAG}"
else
    RUN_TAG=$(date +%Y%m%d_%H%M%S)
fi

EXTRA_ARGS="--ids ${REGRESSION_IDS} ${REPS_ARG} --workers 2"
EXTRA_ARGS="${EXTRA_ARGS% }"

echo "============================================================"
echo "  SIGNOR Regression Re-run"
echo "============================================================"
echo "  User:       ${EBI_USER}"
echo "  Host:       ${LOGIN_HOST}"
echo "  Run tag:    ${RUN_TAG}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  GPU:        ${GPU_COUNT}x ${GPU_TYPE}"
echo "  IDs:        17 regression cases"
echo "  Extra args: ${EXTRA_ARGS}"
echo "============================================================"
echo ""

ssh_login "scancel -u ${EBI_USER} -n ${JOB_NAME} 2>/dev/null || true"
ssh_login "mkdir -p ${LOG_DIR}"

JOB_ID=$(ssh_login "
    export RUN_TAG='${RUN_TAG}'
    export EXTRA_ARGS='${EXTRA_ARGS}'
    export SINGLE_MODE='true'
    export VLLM_MAX_NUM_SEQS='8'
    export NUM_TASKS='1'
    export GRN_LLM_CORRECT_PROJECT_ROOT='${PROJECT_ROOT}'
    sbatch \
        --job-name=${JOB_NAME} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --gres=gpu:${GPU_TYPE}:${GPU_COUNT} \
        --output=${LOG_DIR}/signor-regression-%j.out \
        --error=${LOG_DIR}/signor-regression-%j.err \
        --export=ALL \
        ${JOB_SCRIPT} \
    | grep -oP 'Submitted batch job \K[0-9]+'
")

if [[ -z "$JOB_ID" ]]; then echo "ERROR: Failed to submit job"; exit 1; fi

ssh_login "echo '${JOB_ID} ${RUN_TAG}' > ${JOB_INFO_FILE}"

echo "============================================================"
echo "  Submitted!"
echo "  Job ID:     ${JOB_ID}"
echo "  Run tag:    ${RUN_TAG}"
echo "  Results:    ${PROJECT_ROOT}/results/signor_direct_eval_${RUN_TAG}/"
echo "============================================================"
echo ""
echo "  bash $0 --status"
echo "  bash $0 --logs"
echo "  bash $0 --cancel"
echo ""
