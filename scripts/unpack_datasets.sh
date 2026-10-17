#!/usr/bin/env bash
# Unpack the ProClaim-eval CSVs from datasets/proclaim_eval.zip into datasets/.
#
# The CSVs ship only inside this password-protected archive so the claims and gold labels
# are not exposed as plain text to web crawlers. Please do not redistribute the unpacked
# files in plain text.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$ROOT_DIR/datasets/proclaim_eval.zip"
PASSWORD="${PROCLAIM_EVAL_PASSWORD:-proclaim-eval}"

python3 - "$ARCHIVE" "$ROOT_DIR/datasets" "$PASSWORD" <<'PY'
import sys, zipfile
archive, out_dir, password = sys.argv[1:]
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out_dir, pwd=password.encode())
    for name in zf.namelist():
        print(f"unpacked datasets/{name}")
PY
