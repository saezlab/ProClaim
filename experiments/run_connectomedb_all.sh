#!/usr/bin/env bash
# Run all 9 baselines on claim-verification datasets.
# Each run uses --repeats 1 and the evaluation harness's resume feature,
# so re-running this script safely skips already-completed claims.
#
# Usage:
#   bash experiments/run_connectomedb_all.sh                  # connectomedb (default)
#   bash experiments/run_connectomedb_all.sh signor            # signor only
#   bash experiments/run_connectomedb_all.sh connectomedb signor  # both
#
# Output: results/baselines/<baseline>/<model-slug>/<dataset>_*.jsonl
# Log:    experiments/run_all_<datasets>.log
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Ensure only one instance runs at a time ──────────────────────────
# Uses flock so concurrent launches (e.g. nohup from different
# terminals) wait rather than race for the S2 rate-limit window.
LOCKFILE="/tmp/run_all_baselines.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "ERROR: Another instance is already running (lock: $LOCKFILE)."
    echo "       Wait for it to finish or remove the lock file."
    exit 1
fi

# Accept dataset names as positional args; default to connectomedb
if [ $# -gt 0 ]; then
    DATASETS="$*"
else
    DATASETS="connectomedb"
fi

DATASETS_DIR="/home/ail/workspace/claim_datasets/datasets"
REPEATS=1
TEMP=0.0
OUT="results/baselines"
LOG="experiments/run_all_${DATASETS// /_}.log"

exec > >(tee -a "$LOG") 2>&1

echo "Datasets: $DATASETS"
echo "Log file: $LOG"
echo "PID:      $$"
echo ""

run() {
    local label="$1"; shift
    echo ""
    echo "========================================"
    echo "  $label"
    echo "  $(date)"
    echo "========================================"
    uv run python experiments/run_baselines_datasets.py \
        --datasets-dir "$DATASETS_DIR" \
        --datasets $DATASETS \
        --repeats $REPEATS \
        --temperature $TEMP \
        --output-dir "$OUT" \
        "$@"
    echo "  ✓ $label finished at $(date)"
}

# 1. LLM-only: Claude Sonnet 4.6
run "LLM-only: Claude-Sonnet-4.6" \
    --baseline llm_only \
    --model "anthropic/claude-sonnet-4-6"

# 2. LLM-only: Gemini 3.1 Pro Preview
run "LLM-only: Gemini-3.1-Pro-Preview" \
    --baseline llm_only \
    --model "vertex_ai/gemini-3.1-pro-preview"

# 3. S2 Retrieval: Claude Sonnet 4.6
run "S2 Retrieval: Claude-Sonnet-4.6" \
    --baseline s2_retrieval \
    --model "anthropic/claude-sonnet-4-6"

# 4. S2 Retrieval: Gemini 3.1 Pro Preview
run "S2 Retrieval: Gemini-3.1-Pro-Preview" \
    --baseline s2_retrieval \
    --model "vertex_ai/gemini-3.1-pro-preview"

# 5. ReAct + web search: Claude Sonnet 4.6
run "ReAct + web: Claude-Sonnet-4.6" \
    --baseline react \
    --model "anthropic/claude-sonnet-4-6" \
    --search-backend web

# 6. ReAct + S2: Claude Sonnet 4.6
run "ReAct + S2: Claude-Sonnet-4.6" \
    --baseline react \
    --model "anthropic/claude-sonnet-4-6" \
    --search-backend s2

# 7. ACE: Claude Sonnet 4.6
run "ACE: Claude-Sonnet-4.6" \
    --baseline ace \
    --model "anthropic/claude-sonnet-4-6"

# 8. FIRE: Claude Sonnet 4.6
run "FIRE: Claude-Sonnet-4.6" \
    --baseline fire \
    --model "anthropic/claude-sonnet-4-6"

# 9. OpenScholar: Claude Sonnet 4.6 (S2 retrieval, no oracle evidence)
run "OpenScholar: Claude-Sonnet-4.6 (no oracle)" \
    --baseline open_scholar \
    --model "anthropic/claude-sonnet-4-6" \
    --retrieval \
    --top-k 10

echo ""
echo "========================================"
echo "  ALL RUNS COMPLETE — $(date)"
echo "========================================"
