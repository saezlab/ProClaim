#!/bin/bash
# =============================================================================
# run_ligand_receptor_original_form.sh
#
# Tests 3 ligand-receptor claims in their ORIGINAL dataset form
# (with "In the context of protein-protein interactions, " boilerplate)
# using haiku sufficiency backend.
#
# Claims: B2M-CD1A, B2M-HLA-F, TNFSF14-TNFRSF6B
# All gold=REFUTE
#
# Usage:
#   bash scripts/run_ligand_receptor_original_form.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="experiments/configs/ligand_receptor_original_config.yaml"
RESULTS_DIR="results/ligand_receptor_eval_original"
RESULTS_CSV="${RESULTS_DIR}/results.csv"

mkdir -p "$RESULTS_DIR"

# Claims in ORIGINAL dataset form: "source|target|gold_label|claim_text"
CLAIMS=(
    "B2M|CD1A|REFUTE|In the context of protein-protein interactions, B2M as ligand directly interacts with CD1A as receptor."
    "B2M|HLA-F|REFUTE|In the context of protein-protein interactions, B2M as ligand directly interacts with HLA-F as receptor."
    "TNFSF14|TNFRSF6B|REFUTE|In the context of protein-protein interactions, TNFSF14 as ligand directly interacts with TNFRSF6B as receptor."
)

echo "============================================================"
echo "  Ligand-Receptor Original Form Test (haiku sufficiency)"
echo "============================================================"
echo "  Claims: ${#CLAIMS[@]}"
echo "  Config: ${CONFIG}"
echo "  Output: ${RESULTS_DIR}"
echo "============================================================"
echo ""

# Write CSV header
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "source,target,claim,gold_label,new_verdict,confidence,reasoning" > "$RESULTS_CSV"
fi

PASS=0
FAIL=0
TOTAL=0

for entry in "${CLAIMS[@]}"; do
    IFS='|' read -r SOURCE TARGET GOLD_LABEL CLAIM <<< "$entry"

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

    # Clean up incomplete previous runs
    if [[ -d "${OUTPUT_DIR}" ]]; then
        echo "[${TOTAL}/${#CLAIMS[@]}] Cleaning incomplete run: ${SAFE_NAME}"
        rm -rf "${OUTPUT_DIR}"
    fi

    echo ""
    echo "[${TOTAL}/${#CLAIMS[@]}] Running: ${CLAIM}"
    echo "  Gold: ${GOLD_LABEL}"
    echo ""

    START_TIME=$(date +%s)

    RUN_LOG="${RESULTS_DIR}/${SAFE_NAME}_run.log"
    uv run python -m pkevolve.verification.evidence_programming_direct \
        --config "$CONFIG" \
        --claim "$CLAIM" \
        --output-dir "$OUTPUT_DIR" \
        > "$RUN_LOG" 2>&1 || true

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "--- Last 10 lines of run log ---"
    tail -10 "$RUN_LOG" 2>/dev/null || true
    echo "---"

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

    echo "\"${SOURCE}\",\"${TARGET}\",\"${CLAIM}\",\"${GOLD_LABEL}\",\"${VERDICT}\",\"${CONFIDENCE}\",\"${REASONING}\"" >> "$RESULTS_CSV"

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
