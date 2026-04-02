#!/bin/bash
# =============================================================================
# run_llm_baselines_slurm.sh
#
# Submit SLURM jobs to run LLM-only baselines (Claude Sonnet 4.6 and
# Gemini 3.1 Pro Preview) on SIGNOR and ConnectomeDB claim datasets.
#
# Usage:
#   bash scripts/run_llm_baselines_slurm.sh             # Submit all jobs
#   bash scripts/run_llm_baselines_slurm.sh --limit 5   # Limit claims per dataset
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="/hps/nobackup/saezrodriguez/shared_datasets/claims/datasets"
OUTPUT_DIR="${PROJECT_ROOT}/results/baselines"
LOG_DIR="${PROJECT_ROOT}/results/slurm_logs/llm_baselines"
GCP_CREDENTIALS="${PROJECT_ROOT}/prj-int-dev-saez-ai-pkc-734bae1cf581.json"

LIMIT=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--limit N]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# LIMIT=0 means "all claims" in run_baselines_datasets.py;
# always pass --limit so the value is embedded in the SLURM script.

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helper: submit one job and print the job ID
# ---------------------------------------------------------------------------
submit_job() {
    local job_name="$1"
    local model="$2"
    local extra_env="$3"   # additional "export VAR=VAL" lines, newline-separated

    # Write SLURM script to a temp file to avoid heredoc escaping issues
    local script_file
    script_file=$(mktemp "${LOG_DIR}/${job_name}_XXXXXX.sbatch")

    # Write shebang + SBATCH directives first (must precede any executable line)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${job_name}"
        echo "#SBATCH --time=04:00:00"
        echo "#SBATCH --cpus-per-task=2"
        echo "#SBATCH --mem=8G"
        echo "#SBATCH --output=${LOG_DIR}/${job_name}_%j.out"
        echo "#SBATCH --error=${LOG_DIR}/${job_name}_%j.err"
        echo ""
        echo "set -euo pipefail"
        echo "cd \"${PROJECT_ROOT}\""
        echo ""
        echo "# Load project .env (API keys)"
        echo "if [[ -f \"${PROJECT_ROOT}/.env\" ]]; then"
        echo "    set -a"
        echo "    source \"${PROJECT_ROOT}/.env\""
        echo "    set +a"
        echo "fi"
        echo ""
        echo "# Model-specific environment"
        echo "${extra_env}"
        echo ""
        echo "echo '========================================'"
        echo "echo '  Job: ${job_name}'"
        echo "echo '  Model: ${model}'"
        echo "echo '  Datasets: signor connectomedb'"
        echo "echo '  Limit: ${LIMIT}'"
        echo "echo \"  Start: \$(date)\""
        echo "echo '========================================'"
        echo ""
        echo "uv run python experiments/run_baselines_datasets.py \\"
        echo "    --datasets-dir \"${DATASETS_DIR}\" \\"
        echo "    --datasets signor connectomedb \\"
        echo "    --baseline llm_only \\"
        echo "    --model \"${model}\" \\"
        echo "    --repeats 1 \\"
        echo "    --limit ${LIMIT} \\"
        echo "    --output-dir \"${OUTPUT_DIR}\""
        echo ""
        echo "echo '========================================'"
        echo "echo \"  Done: \$(date)\""
        echo "echo '========================================'"
    } > "$script_file"

    local job_id
    job_id=$(sbatch --parsable "$script_file")
    echo "$job_id"
}

# ---------------------------------------------------------------------------
# 1. Claude Sonnet 4.6
# ---------------------------------------------------------------------------
MODEL_CLAUDE="anthropic/claude-sonnet-4-6"
JOB_NAME_CLAUDE="llm-baseline-claude-sonnet46"

echo "Submitting: ${MODEL_CLAUDE} ..."
JID_CLAUDE=$(submit_job "$JOB_NAME_CLAUDE" "$MODEL_CLAUDE" "# (no extra env needed — ANTHROPIC_API_KEY loaded from .env)")
echo "  -> Job ID: ${JID_CLAUDE}  logs: ${LOG_DIR}/${JOB_NAME_CLAUDE}_${JID_CLAUDE}.out"

# ---------------------------------------------------------------------------
# 2. Gemini 3.1 Pro Preview (Vertex AI)
# ---------------------------------------------------------------------------
MODEL_GEMINI="vertex_ai/gemini-3.1-pro-preview"
JOB_NAME_GEMINI="llm-baseline-gemini31"

GEMINI_ENV="export GOOGLE_APPLICATION_CREDENTIALS=\"${GCP_CREDENTIALS}\"
export VERTEXAI_LOCATION=\"global\""

echo "Submitting: ${MODEL_GEMINI} ..."
JID_GEMINI=$(submit_job "$JOB_NAME_GEMINI" "$MODEL_GEMINI" "$GEMINI_ENV")
echo "  -> Job ID: ${JID_GEMINI}  logs: ${LOG_DIR}/${JOB_NAME_GEMINI}_${JID_GEMINI}.out"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "All jobs submitted."
echo "Monitor with:  squeue -j ${JID_CLAUDE},${JID_GEMINI}"
echo "Cancel with:   scancel ${JID_CLAUDE} ${JID_GEMINI}"
