#!/bin/bash
# =============================================================================
# run_ligand_receptor_batch.sh
#
# Runs the 12 ligand-receptor claims that were previously misclassified
# through the direct evidence programming agent (evidence_programming_direct).
#
# All claims have gold label REFUTE (these are not ligand-receptor pairs).
#
# Usage:
#   bash scripts/run_ligand_receptor_batch.sh
#   bash scripts/run_ligand_receptor_batch.sh --limit 2   # test with 2 claims
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="experiments/configs/ligand_receptor_direct_config.yaml"
RESULTS_DIR="results/ligand_receptor_eval"
RESULTS_CSV="${RESULTS_DIR}/results.csv"
LIMIT=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# Claims: "source|target|gold_label|previous_verdict"
CLAIMS=(
    "B2M|CD1A|REFUTE|SUPPORT"
    "B2M|CD1B|REFUTE|SUPPORT"
    "B2M|HLA-F|REFUTE|UNCERTAIN"
    "B2M|LILRB1|REFUTE|SUPPORT"
    "CLCF1|CRLF1|REFUTE|UNCERTAIN"
    "F2|THBD|REFUTE|SUPPORT"
    "HLA-C|LILRA3|REFUTE|UNCERTAIN"
    "HSP90AA1|CFTR|REFUTE|UNCERTAIN"
    "HSP90AA1|EGFR|REFUTE|SUPPORT"
    "LRPAP1|VLDLR|REFUTE|SUPPORT"
    "TNFSF14|TNFRSF6B|REFUTE|SUPPORT"
    "TNFSF15|TNFRSF6B|REFUTE|SUPPORT"
)

# Apply limit
if [[ "$LIMIT" -gt 0 ]]; then
    CLAIMS=("${CLAIMS[@]:0:$LIMIT}")
fi

echo "============================================================"
echo "  Ligand-Receptor Batch Evaluation (Direct Mode)"
echo "============================================================"
echo "  Claims: ${#CLAIMS[@]}"
echo "  Config: ${CONFIG}"
echo "  Output: ${RESULTS_DIR}"
echo "============================================================"
echo ""

# Write CSV header
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "source,target,claim,gold_label,previous_verdict,new_verdict,confidence,reasoning" > "$RESULTS_CSV"
fi

PASS=0
FAIL=0
TOTAL=0

for entry in "${CLAIMS[@]}"; do
    IFS='|' read -r SOURCE TARGET GOLD_LABEL PREV_VERDICT <<< "$entry"

    CLAIM="${SOURCE} as ligand directly interacts with ${TARGET} as receptor."
    SAFE_NAME="${SOURCE}_${TARGET}"
    OUTPUT_DIR="${RESULTS_DIR}/${SAFE_NAME}"

    TOTAL=$((TOTAL + 1))

    # Skip if verdict already exists
    if [[ -f "${OUTPUT_DIR}/workspace/verdict.json" ]]; then
        echo "[${TOTAL}/${#CLAIMS[@]}] SKIP (exists): ${CLAIM}"
        VERDICT=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR}/workspace/verdict.json')); print(d.get('verdict','?'))")
        CONFIDENCE=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR}/workspace/verdict.json')); print(d.get('confidence',0))")
        if [[ "$VERDICT" == "$GOLD_LABEL" ]]; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
        continue
    fi

    # Clean up incomplete previous runs (no verdict produced)
    if [[ -d "${OUTPUT_DIR}" ]]; then
        echo "[${TOTAL}/${#CLAIMS[@]}] Cleaning incomplete run: ${SAFE_NAME}"
        rm -rf "${OUTPUT_DIR}"
    fi

    echo ""
    echo "[${TOTAL}/${#CLAIMS[@]}] Running: ${CLAIM}"
    echo "  Gold: ${GOLD_LABEL}  |  Previous: ${PREV_VERDICT}"
    echo ""

    START_TIME=$(date +%s)

    # Run agent — log to file instead of piping through tail (avoids SIGPIPE kills)
    RUN_LOG="${RESULTS_DIR}/${SAFE_NAME}_run.log"
    uv run python -m proclaim.verification.evidence_programming_direct \
        --config "$CONFIG" \
        --claim "$CLAIM" \
        --output-dir "$OUTPUT_DIR" \
        > "$RUN_LOG" 2>&1 || true

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Show tail of run log
    echo "--- Last 10 lines of run log ---"
    tail -10 "$RUN_LOG" 2>/dev/null || true
    echo "---"

    # Parse verdict
    VERDICT_FILE="${OUTPUT_DIR}/workspace/verdict.json"
    if [[ -f "$VERDICT_FILE" ]]; then
        VERDICT=$(python3 -c "import json; d=json.load(open('${VERDICT_FILE}')); print(d.get('verdict','ERROR'))")
        CONFIDENCE=$(python3 -c "import json; d=json.load(open('${VERDICT_FILE}')); print(d.get('confidence',0))")
        REASONING=$(python3 -c "import json; d=json.load(open('${VERDICT_FILE}')); r=d.get('reasoning',''); print(r[:200].replace('\"','\\\"'))")
    else
        VERDICT="NO_VERDICT"
        CONFIDENCE="0"
        REASONING="No verdict file produced"
    fi

    # Record result
    echo "\"${SOURCE}\",\"${TARGET}\",\"${CLAIM}\",\"${GOLD_LABEL}\",\"${PREV_VERDICT}\",\"${VERDICT}\",\"${CONFIDENCE}\",\"${REASONING}\"" >> "$RESULTS_CSV"

    if [[ "$VERDICT" == "$GOLD_LABEL" ]]; then
        STATUS="CORRECT"
        PASS=$((PASS + 1))
    else
        STATUS="WRONG"
        FAIL=$((FAIL + 1))
    fi

    echo "  Result: ${VERDICT} (conf=${CONFIDENCE}) — ${STATUS} [${ELAPSED}s]"
    echo ""
done

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  Total:   ${TOTAL}"
echo "  Correct: ${PASS}"
echo "  Wrong:   ${FAIL}"
echo "  Accuracy: $(python3 -c "print(f'{${PASS}/${TOTAL}*100:.1f}%' if ${TOTAL} > 0 else 'N/A')")"
echo ""
echo "  Results CSV: ${RESULTS_CSV}"
echo "============================================================"
