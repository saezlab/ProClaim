#!/bin/bash
# =============================================================================
# submit_connectomedb_batch.sh
#
# One-click ConnectomeDB evaluation from your local laptop.
# Submits a self-contained SLURM job (vLLM + eval on same GPU).
#
# Default mode: 8-chunk parallel (--array=0-7), one GPU per task, followed by
# an auto-merge job that concatenates the chunk CSVs.
# Use --single to run a single job (all rows, one GPU).
#
# Prerequisites:
#   - SSH access to EBI HPC configured
#
# Usage:
#   bash scripts/submit_connectomedb_batch.sh                  # 8-GPU parallel (default)
#   bash scripts/submit_connectomedb_batch.sh --single         # Single-GPU mode
#   bash scripts/submit_connectomedb_batch.sh --single --limit 1   # Test with 1 claim
#   bash scripts/submit_connectomedb_batch.sh --reps 1         # 1 repetition only
#   bash scripts/submit_connectomedb_batch.sh --status         # Check job status
#   bash scripts/submit_connectomedb_batch.sh --cancel         # Cancel running jobs
#   bash scripts/submit_connectomedb_batch.sh --logs           # View job logs
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
EBI_USER="${EBI_USER:-wuy}"
LOGIN_HOST="ihpc.ebi.ac.uk"
LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"

# SLURM job settings
JOB_NAME="connectomedb-eval"
TIME_LIMIT="40:00:00"
CPUS=8
MEM="64G"
GPU_TYPE="a100"
GPU_COUNT=1  # per array task; parallel mode uses 8 GPUs total via --array=0-7 (do NOT set to 8)

# Project paths (on HPC)
PROJECT_ROOT="/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct"
JOB_SCRIPT="${PROJECT_ROOT}/scripts/slurm_connectomedb_job.sh"
JOB_INFO_FILE="${PROJECT_ROOT}/.connectomedb_job_info"
LOG_DIR="${PROJECT_ROOT}/results/slurm_logs"

# Action / mode
ACTION="submit"
MODE="parallel"     # "parallel" (8-chunk array) or "single"
REPS_ARG=""
LIMIT_ARG=""

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)      EBI_USER="$2"; LOGIN_NODE="${EBI_USER}@${LOGIN_HOST}"; shift 2 ;;
        --status)    ACTION="status";  shift ;;
        --cancel)    ACTION="cancel";  shift ;;
        --logs)      ACTION="logs";    shift ;;
        --single)    MODE="single";    shift ;;
        --reps)      REPS_ARG="--reps $2";  shift 2 ;;
        --limit)     LIMIT_ARG="--limit $2"; shift 2 ;;
        -h|--help)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Submission modes:"
            echo "  (default)          8-chunk parallel: 8 GPUs, auto-merged results"
            echo "  --single           Single-GPU mode: all rows on one GPU"
            echo ""
            echo "Actions:"
            echo "  --status     Check job status"
            echo "  --cancel     Cancel running jobs"
            echo "  --logs       View job logs"
            echo ""
            echo "Job parameters:"
            echo "  --reps N     Number of repetitions per claim (default: 3)"
            echo "  --limit N    Limit to N rows (single mode only; default: all 547)"
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

# ---- SSH helper -------------------------------------------------------------
ssh_login() {
    ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 "$LOGIN_NODE" "$@"
}

# ---- Read job info file (format: "ARRAY_JOB_ID MERGE_JOB_ID RUN_TAG") ------
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

    RESULTS_DIR="${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}"
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
    LOG_PATTERN="${LOG_DIR}/connectomedb-eval-${ARRAY_JOB_ID}*.out"
    echo "Following logs: ${LOG_PATTERN}"
    echo "Press Ctrl-C to stop"
    echo "============================================================"
    ssh_login "tail -f ${LOG_PATTERN}"
    exit 0
fi

# ---- SUBMIT -----------------------------------------------------------------
RUN_TAG=$(date +%Y%m%d_%H%M%S)

# Build EXTRA_ARGS to pass through to run_connectomedb_eval.py
EXTRA_ARGS="${REPS_ARG}"
if [[ "$MODE" == "single" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} ${LIMIT_ARG}"
fi
EXTRA_ARGS="${EXTRA_ARGS# }"  # trim leading space

echo "============================================================"
echo "  ConnectomeDB Evaluation Submission"
echo "============================================================"
echo "  User:       ${EBI_USER}"
echo "  Host:       ${LOGIN_HOST}"
echo "  Mode:       ${MODE}"
echo "  Run tag:    ${RUN_TAG}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  GPUs:       ${GPU_COUNT}x ${GPU_TYPE} per task"
echo "  CPUs:       ${CPUS}"
echo "  Memory:     ${MEM}"
[[ -n "$EXTRA_ARGS" ]] && echo "  Extra args: ${EXTRA_ARGS}"
echo "============================================================"
echo ""

echo "Cleaning up any old ${JOB_NAME} tasks..."
ssh_login "scancel -u ${EBI_USER} -n ${JOB_NAME} 2>/dev/null || true"
ssh_login "mkdir -p ${LOG_DIR}"
echo ""

# ---- Build and submit the main (array) job ----------------------------------
if [[ "$MODE" == "parallel" ]]; then
    ARRAY_FLAG="--array=0-7"
    SINGLE_MODE_VAL="false"
    echo "Submitting 8-task array job..."
else
    ARRAY_FLAG=""
    SINGLE_MODE_VAL="true"
    echo "Submitting single-GPU job..."
fi

ARRAY_JOB_ID=$(ssh_login "
    export RUN_TAG='${RUN_TAG}'
    export EXTRA_ARGS='${EXTRA_ARGS}'
    export SINGLE_MODE='${SINGLE_MODE_VAL}'
    sbatch \
        --job-name=${JOB_NAME} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --gres=gpu:${GPU_TYPE}:${GPU_COUNT} \
        --output=${LOG_DIR}/connectomedb-eval-%j_%a.out \
        --error=${LOG_DIR}/connectomedb-eval-%j_%a.err \
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

# ---- Submit merge job (parallel mode only) -----------------------------------
MERGE_JOB_ID="none"
if [[ "$MODE" == "parallel" ]]; then
    echo "Submitting merge job (depends on array job)..."
    RESULTS_DIR="${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}"
    MERGE_JOB_ID=$(ssh_login "
        sbatch \
            --job-name=${JOB_NAME}-merge \
            --time=00:15:00 \
            --cpus-per-task=2 \
            --mem=8G \
            --dependency=afterok:${ARRAY_JOB_ID} \
            --output=${LOG_DIR}/connectomedb-eval-merge-%j.out \
            --error=${LOG_DIR}/connectomedb-eval-merge-%j.err \
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
        echo "WARNING: Failed to submit merge job. Merge manually after chunks complete."
        MERGE_JOB_ID="none"
    else
        echo "  Merge job ID: ${MERGE_JOB_ID}"
    fi
fi

# ---- Save job info ----------------------------------------------------------
ssh_login "echo '${ARRAY_JOB_ID} ${MERGE_JOB_ID} ${RUN_TAG}' > ${JOB_INFO_FILE}"

echo ""
echo "============================================================"
echo "  Jobs submitted successfully!"
echo "============================================================"
echo "  Array job ID: ${ARRAY_JOB_ID}"
[[ "$MODE" == "parallel" ]] && echo "  Merge job ID: ${MERGE_JOB_ID}"
echo "  Run tag:      ${RUN_TAG}"
echo "  Results dir:  ${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}/"
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
echo "  scp ${LOGIN_NODE}:${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}/results.csv ."
echo ""
