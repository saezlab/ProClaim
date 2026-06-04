#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_proclaim.sh [--dry-run] [--help]

Set up ProClaim for the README vignette.

What this script does:
  1. Installs uv if it is missing.
  2. Runs uv sync for the main project environment.
  3. Creates .env from .env.example if needed.
  4. Builds the dedicated Python 3.10 NER environment.

Options:
  --dry-run   Print the commands without running them.
  --help      Show this help message.
EOF
}

run_shell() {
    local command="$1"
    echo "+ ${command}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        eval "$command"
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi

    if command -v curl >/dev/null 2>&1; then
        run_shell 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    elif command -v wget >/dev/null 2>&1; then
        run_shell 'wget -qO- https://astral.sh/uv/install.sh | sh'
    else
        echo "ERROR: uv is required and neither curl nor wget is available to install it." >&2
        exit 1
    fi

    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if [[ "$DRY_RUN" -eq 0 ]] && ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv was not found after installation. Add ~/.local/bin to PATH and rerun this script." >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cd "$ROOT_DIR"

ensure_uv
run_shell 'uv sync'

if [[ -f .env ]]; then
    echo "+ keeping existing .env"
else
    run_shell 'cp .env.example .env'
fi

run_shell 'bash scripts/setup_ner_venv310.sh'

cat <<'EOF'

Setup complete.

Next steps:
  1. Open .env and add ANTHROPIC_API_KEY.
  2. Run bash scripts/run_claim_example.sh
  3. To use your own claim, run:
    bash scripts/run_claim_example.sh --claim "<your claim here>"
EOF