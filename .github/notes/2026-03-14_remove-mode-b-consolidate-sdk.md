# Remove Mode B / Consolidate SDK-only Architecture — 2026-03-14

**Branch:** `rlm-explore`

## Summary

Removed the standalone REPL orchestrator (Mode B) from the verification
subsystem, consolidating the codebase around a single execution mode: the
Claude Agent SDK + MCP notebook server (formerly "Mode A"). This eliminates
conflicting output formats between the two modes and simplifies the
architecture ahead of the NeurIPS submission. Additionally, refactored
`setup_kernel()` to read LLM config from environment variables so the
agent never sees API keys, URLs, or model names, and added
`make_subagent_llm()` on `VerificationSettings` for one-call LLM
construction.

## New Files

| File | Purpose |
|------|---------|
| `src/README.md` | Documents the verification subsystem and sufficiency classifier architectures |

## Modified Files

| File | Changes |
|------|---------|
| `src/pkevolve/verification/evidence_programming.py` | Removed `verify_claim_repl_mode()`, Mode B CLI branch, Mode B docstring/usage examples. Simplified system prompt: agent now calls `setup_kernel(claim, workspace_path)` instead of a 30-line setup block with explicit LLM config. Removed `project_root`, `subagent_model`, `llm_base_url`, `mlp_model_dir` template vars from `.format()` call. |
| `src/pkevolve/verification/config.py` | Narrowed `mode` field from `Literal["sdk", "repl"]` to `Literal["sdk"]`. Added `make_subagent_llm()` method for one-call LLM construction. Updated `build_sdk_env()` to inject `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `MLP_MODEL_DIR` into the SDK subprocess environment. |
| `src/pkevolve/verification/kernel_runner.py` | Removed `inject_prelude()` method (~60 lines) — was only used by Mode B. Updated module docstring to remove Mode B references. |
| `src/pkevolve/verification/evidence_api.py` | Added `setup_kernel(claim, workspace_path)` — reads LLM config from `os.environ` (`LLM_BASE_URL`, `LLM_API_KEY` with fallback chain, `LLM_MODEL`), creates `EvidenceState`, calls `make_llm()`, returns `(state, llm, workspace)`. |
| `src/pkevolve/verification/notebook_mcp.py` | Updated docstring to remove "shared with repl_orchestrator" reference. |
| `experiments/example_config.yaml` | Updated subagent model from `qwen3-8b` to `qwen3.5-9b`. |

## Deleted Files

| File | Reason |
|------|--------|
| `src/pkevolve/verification/repl_orchestrator.py` | Entire Mode B orchestrator (~370 lines): `SYSTEM_PROMPT`, `extract_code()`, `verify_claim_repl()`, `verify_claims_batch()`. No longer needed. |

## Architecture

```
CLI (evidence_programming.py)
 │
 ├─ VerificationSettings.from_cli()
 │   └─ build_sdk_env()  ──► LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MLP_MODEL_DIR
 │
 └─ verify_claim_notebook(cfg)
     │
     ├─ Claude Agent SDK (Anthropic API)
     │   └─ system prompt tells agent to call setup_kernel(claim, workspace_path)
     │
     ├─ MCP Notebook Server (notebook_mcp.py, subprocess)
     │   └─ KernelRunner manages Jupyter kernel lifecycle
     │
     └─ Jupyter Kernel
         ├─ setup_kernel() reads env vars → creates state + llm
         └─ agent calls evidence API via nb_execute cells
```

## Key Design Decisions

- **Single mode**: Maintaining two execution modes with incompatible outputs
  (`.ipynb` notebook vs bare `verdict.json`) added complexity without benefit.
  The SDK+notebook path is the submission target.
- **Environment-variable LLM config**: `setup_kernel()` reads `LLM_BASE_URL`,
  `LLM_API_KEY`, `LLM_MODEL` from `os.environ` instead of accepting them as
  parameters. This keeps API keys and infrastructure details hidden from the
  agent's system prompt and tool calls.
- **API key fallback chain**: `LLM_API_KEY` → `GLM_API_KEY` → `ZAI_API_KEY` →
  `OPENAI_API_KEY` → `"EMPTY"` — preserves backward compatibility with existing
  `.env` files.
- **`make_subagent_llm()` on config**: Centralises LLM construction so callers
  don't need to pass `base_url`/`api_key`/`model` individually.
- **Kept `setup_kernel` in `evidence_api.py`**: Although it was originally
  created for Mode B's prelude injection, Mode A's agent calls it as the first
  `nb_execute` cell, so it stays.
