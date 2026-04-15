#!/bin/bash
# =============================================================================
# run_all_baselines_slurm.sh
#
# Submit SLURM jobs to run all 9 baselines on claim-verification datasets.
# Each baseline is submitted as a separate sbatch job.
# Re-running is safe — the evaluation harness's resume feature skips
# already-completed claims.
#
# Usage:
#   bash scripts/run_all_baselines_slurm.sh                                    # all baselines, connectomedb
#   bash scripts/run_all_baselines_slurm.sh --datasets signor                  # signor only
#   bash scripts/run_all_baselines_slurm.sh --datasets connectomedb signor     # both
#   bash scripts/run_all_baselines_slurm.sh --limit 5                          # limit claims per dataset
#   bash scripts/run_all_baselines_slurm.sh --repeats 2                        # 2 repeats per baseline
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="/hps/nobackup/saezrodriguez/shared_datasets/claims/datasets"
OUTPUT_DIR="${PROJECT_ROOT}/results/baselines"
LOG_DIR="${PROJECT_ROOT}/results/slurm_logs/baselines"
GCP_CREDENTIALS="${PROJECT_ROOT}/prj-int-dev-saez-ai-pkc-734bae1cf581.json"

LIMIT=0
REPEATS=1
TEMP=0.0
WALL_TIME="08:00:00"
DATASETS=()
BASELINES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1"); shift
            done
            ;;
        --baselines)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                BASELINES+=("$1"); shift
            done
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--baselines B1 B2 ...] [--datasets D1 D2 ...] [--limit N] [--repeats N] [--time HH:MM:SS]"
            echo ""
            echo "Available baselines: llm_only s2_retrieval react_web react_s2 ace fire open_scholar"
            echo "  (default: all)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Default to signor and connectomedb if no datasets specified
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("signor" "connectomedb")
fi
DATASETS_STR="${DATASETS[*]}"

# If no baselines specified, run all
if [[ ${#BASELINES[@]} -eq 0 ]]; then
    BASELINES=(llm_only s2_retrieval react_web react_s2 ace fire open_scholar)
fi

# Helper: check if a baseline is in the selected list
baseline_selected() {
    local needle="$1"
    for b in "${BASELINES[@]}"; do
        [[ "$b" == "$needle" ]] && return 0
    done
    return 1
}

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helper: submit one sbatch job
#   $1 = job name
#   $2 = description (for logging inside the job)
#   $3 = extra env exports (newline-separated)
#   $4... = extra args to run_baselines_datasets.py (baseline-specific flags)
# ---------------------------------------------------------------------------
submit_job() {
    local job_name="$1"; shift
    local description="$1"; shift
    local extra_env="$1"; shift
    local extra_args=("$@")

    local script_file
    script_file=$(mktemp "${LOG_DIR}/${job_name}_XXXXXX.sbatch")

    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${job_name}"
        echo "#SBATCH --time=${WALL_TIME}"
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
        echo "# Baseline-specific environment"
        echo "${extra_env}"
        echo ""
        echo "echo '========================================'"
        echo "echo '  ${description}'"
        echo "echo '  Datasets: ${DATASETS_STR}'"
        echo "echo '  Limit: ${LIMIT}  Repeats: ${REPEATS}'"
        echo "echo \"  Start: \$(date)\""
        echo "echo '========================================'"
        echo ""
        # Build the command
        echo -n "uv run python experiments/run_baselines_datasets.py"
        echo -n " --datasets-dir \"${DATASETS_DIR}\""
        echo -n " --datasets ${DATASETS_STR}"
        echo -n " --repeats ${REPEATS}"
        echo -n " --temperature ${TEMP}"
        echo -n " --limit ${LIMIT}"
        echo -n " --output-dir \"${OUTPUT_DIR}\""
        for arg in "${extra_args[@]}"; do
            echo -n " ${arg}"
        done
        echo ""
        echo ""
        echo "echo '========================================'"
        echo "echo \"  Done: \$(date)\""
        echo "echo '========================================'"
    } > "$script_file"

    local job_id
    job_id=$(sbatch --parsable "$script_file")
    echo "$job_id"
}

ALL_JOB_IDS=()

GEMINI_ENV="export GOOGLE_APPLICATION_CREDENTIALS=\"${GCP_CREDENTIALS}\"
export VERTEXAI_LOCATION=\"global\""

# ---------------------------------------------------------------------------
# 1. LLM-only: Claude Sonnet 4.6
# ---------------------------------------------------------------------------
if baseline_selected llm_only; then
echo "Submitting: LLM-only: Claude-Sonnet-4.6 ..."
JID=$(submit_job "llm-only-claude" "LLM-only: Claude-Sonnet-4.6" \
    "# ANTHROPIC_API_KEY loaded from .env" \
    --baseline llm_only --model "anthropic/claude-sonnet-4-6")
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")

# ---------------------------------------------------------------------------
# 2. LLM-only: Gemini 3.1 Pro Preview
# ---------------------------------------------------------------------------

echo "Submitting: LLM-only: Gemini-3.1-Pro-Preview ..."
JID=$(submit_job "llm-only-gemini" "LLM-only: Gemini-3.1-Pro-Preview" \
    "$GEMINI_ENV" \
    --baseline llm_only --model "vertex_ai/gemini-3.1-pro-preview")
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")
fi

# ---------------------------------------------------------------------------
# 3. S2 Retrieval: Claude Sonnet 4.6
# ---------------------------------------------------------------------------
if baseline_selected s2_retrieval; then
echo "Submitting: S2 Retrieval: Claude-Sonnet-4.6 ..."
JID=$(submit_job "s2-retrieval-claude" "S2 Retrieval: Claude-Sonnet-4.6" \
    "# ANTHROPIC_API_KEY loaded from .env" \
    --baseline s2_retrieval --s2-model "anthropic/claude-sonnet-4-6")
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")

# ---------------------------------------------------------------------------
# 4. S2 Retrieval: Gemini 3.1 Pro Preview
# ---------------------------------------------------------------------------
echo "Submitting: S2 Retrieval: Gemini-3.1-Pro-Preview ..."
JID=$(submit_job "s2-retrieval-gemini" "S2 Retrieval: Gemini-3.1-Pro-Preview" \
    "$GEMINI_ENV" \
    --baseline s2_retrieval --s2-model "vertex_ai/gemini-3.1-pro-preview")
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")
fi

# ---------------------------------------------------------------------------
# 5. ReAct + web search: Claude Sonnet 4.6
# ---------------------------------------------------------------------------
if baseline_selected react_web; then
echo "Submitting: ReAct + web: Claude-Sonnet-4.6 ..."
JID=$(submit_job "react-web-claude" "ReAct + web: Claude-Sonnet-4.6" \
    "# ANTHROPIC_API_KEY + SERPER_API_KEY loaded from .env" \
    --baseline react --react-model "anthropic/claude-sonnet-4-6" --react-search-backend web)
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")
fi

# ---------------------------------------------------------------------------
# 6. ReAct + S2: Claude Sonnet 4.6
# ---------------------------------------------------------------------------
if baseline_selected react_s2; then
echo "Submitting: ReAct + S2: Claude-Sonnet-4.6 ..."
JID=$(submit_job "react-s2-claude" "ReAct + S2: Claude-Sonnet-4.6" \
    "# ANTHROPIC_API_KEY loaded from .env" \
    --baseline react --react-model "anthropic/claude-sonnet-4-6" --react-search-backend s2)
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")
fi

# ---------------------------------------------------------------------------
# 7. ACE: Claude Sonnet 4.6
# ---------------------------------------------------------------------------
if baseline_selected ace; then
echo "Submitting: ACE: Claude-Sonnet-4.6 ..."
JID=$(submit_job "ace-claude" "ACE: Claude-Sonnet-4.6" \
    "# ANTHROPIC_API_KEY loaded from .env" \
    --baseline ace --ace-model "anthropic/claude-sonnet-4-6")
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")
fi

# ---------------------------------------------------------------------------
# 8. FIRE: Claude Sonnet 4.6
# ---------------------------------------------------------------------------
if baseline_selected fire; then
echo "Submitting: FIRE: Claude-Sonnet-4.6 ..."
JID=$(submit_job "fire-claude" "FIRE: Claude-Sonnet-4.6" \
    "# ANTHROPIC_API_KEY + SERPER_API_KEY loaded from .env" \
    --baseline fire --fire-model "anthropic/claude-sonnet-4-6")
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")
fi

# ---------------------------------------------------------------------------
# 9. OpenScholar: Claude Sonnet 4.6 (S2 retrieval, no oracle evidence)
# ---------------------------------------------------------------------------
if baseline_selected open_scholar; then
echo "Submitting: OpenScholar: Claude-Sonnet-4.6 ..."
JID=$(submit_job "openscholar-claude" "OpenScholar: Claude-Sonnet-4.6 (no oracle)" \
    "# ANTHROPIC_API_KEY loaded from .env" \
    --baseline open_scholar --os-model "claude-sonnet-4-6" --os-api "anthropic" --os-retrieval --os-top-n 10)
echo "  -> Job ID: ${JID}"
ALL_JOB_IDS+=("$JID")
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
JOB_IDS_CSV=$(IFS=,; echo "${ALL_JOB_IDS[*]}")
echo ""
echo "========================================"
echo "  All ${#ALL_JOB_IDS[@]} jobs submitted."
echo "  Datasets:  ${DATASETS_STR}"
echo "  Limit:     ${LIMIT}"
echo "  Repeats:   ${REPEATS}"
echo "  Wall time: ${WALL_TIME}"
echo "========================================"
echo "Monitor with:  squeue -j ${JOB_IDS_CSV}"
echo "Cancel all:    scancel ${JOB_IDS_CSV}"
echo "Logs in:       ${LOG_DIR}/"
