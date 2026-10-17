#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required to create .venv310." >&2
    exit 1
fi

VENV_DIR=".venv310"
PYTHON_BIN="${VENV_DIR}/bin/python"
SCI_MODEL_URL="en-core-sci-sm @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "Creating ${VENV_DIR} with Python 3.10..."
uv venv --python 3.10 "$VENV_DIR"

echo "Installing PyTorch CUDA runtime for Python 3.10 NLP tooling..."
uv pip install --python "$PYTHON_BIN" \
    --index-url "$TORCH_CUDA_INDEX" \
    "torch"

echo "Installing spaCy/scispaCy and sentence-transformer dependencies..."
uv pip install --python "$PYTHON_BIN" \
    "numpy<2" \
    "spacy>=3.7.4,<3.8" \
    "sentence-transformers" \
    "$SCI_MODEL_URL"

echo "Verifying NER worker environment..."
"$PYTHON_BIN" - <<'PY'
import spacy
import torch

try:
    import sentence_transformers
    st_version = sentence_transformers.__version__
except Exception as exc:
    st_version = f"IMPORT FAILED: {exc}"

nlp = spacy.load("en_core_sci_sm")
print("python:", nlp.path)
print("spaCy:", spacy.__version__)
print("model:", nlp.meta.get("name", "en_core_sci_sm"))
print("torch:", torch.__version__)
print("sentence-transformers:", st_version)
PY

echo "Done. The Python 3.10 NLP environment now exists at ${VENV_DIR}/bin/python"