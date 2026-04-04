# OpenScholar UV Environment — 2026-04-04

**Branch:** `exp` (grn-llm-correct) / `main` (OpenScholar — untracked files, not yet committed)

## Summary

Replaced the conda-based installation instructions in `OpenScholar/` with two `uv`-managed `pyproject.toml` files — one for the main inference package and one for the retriever subpackage. The original setup required `conda create -n os_env python=3.10.0` with a flat `requirements.txt`; both are now superseded by `uv sync` with optional extras for GPU dependencies.

---

## Modified Files

| File | Changes |
|------|---------|
| `OpenScholar/pyproject.toml` | **New.** Core inference deps (`tqdm`, `spacy`, `nltk`, `transformers`, `openai`, `datasets`, `requests`, `pandas`, `jsonlines`, `beautifulsoup4`, `numpy`). Optional `[gpu]` extra adds `vllm` + `FlagEmbedding` for OS-8B local inference. `tool.uv.package = false` (script-only project). |
| `OpenScholar/retriever/pyproject.toml` | **New.** Retriever deps (`transformers`, `sentence-transformers`, `omegaconf`, `hydra-core`, `pyserini`, `datasketch`). Optional `[gpu]` extra adds `faiss-gpu` + `torch` (PyTorch Cu121 index); `[cpu]` extra adds `faiss-cpu` + `torch`. |

---

## Installation

**OS-GPT track (API only, no GPU):**
```bash
cd OpenScholar
uv venv --python 3.10
uv sync
uv run python -m spacy download en_core_web_sm
export S2_API_KEY=<key>
export OPENAI_API_KEY=<key>
```

**OS-8B track (local model, CUDA GPU):**
```bash
uv sync --extra gpu
```

**Retriever (GPU, Python 3.10 required):**
```bash
cd OpenScholar/retriever
uv venv --python 3.10    # faiss-gpu PyPI wheels cap at cp310
uv sync --extra gpu
```

---

## Constraints & Notes

- `faiss-gpu` on PyPI only provides wheels up to Python 3.10 (`cp310`). The original `retriever/environment.yml` used Python 3.11 (relying on the conda-forge build). For Python 3.11 + GPU FAISS, install faiss via conda into the uv venv: `conda install -c conda-forge faiss-gpu=1.8.0`.
- `vllm` vendors its own pinned torch; no separate `torch` pin is needed in the main `pyproject.toml`.
- `pyserini` requires Java 11+ on `PATH` — on SLURM/HPC: `module load java`.
- The main `pyproject.toml` resolved cleanly with uv 0.8.15 (`uv lock --dry-run` → 232 packages).

---

## Relation to Integration Plan

See `doc/openscholar_integration_plan.md`. The OS-GPT track (API only, core deps) is now installable via `uv sync`. The OS-8B track (GPU extras) is blocked on HPC storage (~500 GB datastore) and can be enabled with `uv sync --extra gpu` once a GPU node is available.
