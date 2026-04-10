# Workstation vLLM Setup — 2026-03-30

**Branch:** main

## Summary

Added scripts and documentation to run vLLM locally on the workstation hub (`beast`), as an alternative to the existing EBI HPC SLURM-based workflow. The workstation setup uses `uv` for Python environment management, installs vLLM into a dedicated virtual environment, downloads models via the HuggingFace CLI, and serves them directly without SLURM, SSH tunnelling, or Singularity containers.

## New Files

| File | Purpose |
|------|---------|
| `scripts/start_vllm_workstation.sh` | One-command vLLM server launcher for local workstation: auto-detects GPUs, finds a free port, selects model-family parsers (same logic as `vllm_node_setup.sh`), and manages the server process via a PID info file. Supports `--stop` and `--status` actions. |
| `scripts/test_vllm_server.py` | Smoke-test script that connects to a running vLLM server, lists available models, sends a single chat completion, and prints the response and token usage. |

## Installation Steps

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Create a dedicated vLLM virtual environment

```bash
uv venv ~/.venvs/vllm --python 3.12
source ~/.venvs/vllm/bin/activate
uv pip install vllm
```

The venv persists on disk — only activate it each session, no re-installation needed.

### 3. Configure HuggingFace model storage

The workstation has 456 GB free on `/` and 3 TB on `/data`. The default HF cache (`~/.cache/huggingface/hub/`) on `/` is sufficient for most models. If `/data` becomes accessible, redirect with:

```bash
echo 'export HF_HOME=/data/hf_models' >> ~/.bashrc
source ~/.bashrc
```

### 4. Download a model

```bash
source ~/.venvs/vllm/bin/activate
huggingface-cli download Qwen/Qwen3-8B   # ~16 GB
```

`huggingface-cli` is already available inside the vLLM venv (bundled with `huggingface_hub`).

### 5. Serve the model

```bash
bash scripts/start_vllm_workstation.sh --model Qwen/Qwen3-8B
```

The script auto-detects GPUs (2× on `beast`) and selects port 8000 by default. API is available at `http://localhost:8000/v1/`.

### 6. Test the server

```bash
# In another terminal (with project venv active):
uv run python scripts/test_vllm_server.py
uv run python scripts/test_vllm_server.py --prompt "Does MAPK1 phosphorylate H3? Answer Yes or No."
```

## Key Design Decisions

- **Dedicated venv (`~/.venvs/vllm`)**: vLLM pins specific PyTorch and CUDA versions that conflict with the project venv. Keeping it separate avoids dependency conflicts; the `--vllm PATH` flag allows overriding the binary if needed.

- **Model-family parser detection**: `start_vllm_workstation.sh` replicates the `case` statement from `vllm_node_setup.sh` to auto-select `--tool-call-parser` and `--reasoning-parser` per model family (Qwen3, GLM, DeepSeek, Llama, Mistral, etc.), so tool calling and reasoning work correctly out of the box.

- **PID-based process management**: Without SLURM or tmux, the script saves `PID:PORT:MODEL` to `$TMPDIR/.vllm_workstation_info` so `--stop` and `--status` work from a separate terminal.

- **No Singularity wrapping**: Unlike the HPC script, `vllm` is called directly from the activated venv. Singularity is only needed on EBI compute nodes where vLLM isn't installed natively.

- **HF repo ID vs local path**: vLLM (via `huggingface_hub`) resolves `Qwen/Qwen3-8B` to the local cache automatically — passing a local path must point to the `snapshots/<hash>/` directory, not the top-level cache folder.

## Notes

- **First launch is slow (5–15 min)**: vLLM compiles CUDA graphs with `torch.compile` + Inductor on first use. The compiled artifacts are cached to disk; subsequent starts are much faster.
- **`SymmMemCommunicator` warning**: Device capability 8.6 (RTX 3090 / A40) does not support SymmMem — this warning is harmless and vLLM continues normally.
- **`--enforce-eager` flag**: Skips CUDA graph compilation for fast startup at the cost of inference throughput. Useful for testing.
- **Clearing model cache**: Use `huggingface-cli delete-cache` (interactive) or `rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B` to free space before downloading a larger model.
