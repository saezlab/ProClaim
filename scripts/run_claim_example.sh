#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CLAIM="GNAS directly activates ADCY1."
CLAIM="$DEFAULT_CLAIM"
CONFIG="experiments/configs/smoke_config.yaml"
OUTPUT_DIR=""
DRY_RUN=0
VERBOSE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_claim_example.sh [options]

Run the single-claim ProClaim vignette.

Defaults:
    claim:   GNAS directly activates ADCY1.
    config:  experiments/configs/smoke_config.yaml

Options:
  --claim TEXT        Claim to verify.
  --config PATH       Verification config to use.
  --output-dir PATH   Output directory for this run.
  --verbose           Enable verbose logging in the Python runner.
  --dry-run           Print the resolved command without running it.
  --help              Show this help message.

Example:
    bash scripts/run_claim_example.sh --claim "SRC directly inhibits CTTN."
EOF
}

slugify() {
    local text="$1"

    text="$(printf '%s' "$text" | tr '[:upper:]' '[:lower:]')"
    text="$(printf '%s' "$text" | sed 's/[^a-z0-9]/_/g; s/_\{2,\}/_/g; s/^_//; s/_$//')"
    text="${text:0:60}"

    if [[ -z "$text" ]]; then
        text="claim"
    fi

    printf '%s' "$text"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --claim)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --claim requires a value." >&2
                exit 1
            fi
            CLAIM="$2"
            shift 2
            ;;
        --claim=*)
            CLAIM="${1#*=}"
            shift
            ;;
        --config)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --config requires a value." >&2
                exit 1
            fi
            CONFIG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG="${1#*=}"
            shift
            ;;
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --output-dir requires a value." >&2
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            CLAIM="$*"
            break
            ;;
    esac
done

cd "$ROOT_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="results/claim_example/$(slugify "$CLAIM")"
fi

CMD=(
    uv run python -m proclaim.verification.evidence_programming_direct
    --config "$CONFIG"
    --claim "$CLAIM"
    --output-dir "$OUTPUT_DIR"
)

if [[ "$VERBOSE" -eq 1 ]]; then
    CMD+=(--verbose)
fi

echo "Claim: $CLAIM"
echo "Config: $CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Resolved command:'
    for arg in "${CMD[@]}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed. Run bash scripts/setup_proclaim.sh first." >&2
    exit 1
fi

exec "${CMD[@]}"