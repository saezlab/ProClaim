# Direct Evidence Programming, Notebook Fix & Prompt Caching — 2026-04-14

**Branch:** `feature/ctx-manage`

## Summary

Implemented the Direct API evidence programming orchestrator (`evidence_programming_direct.py`) — a 749-line script that replaces the Jupyter-kernel-based architecture with bash subprocess execution + LiteLLM tool-use + jupytext-style logging. Consolidated `llm_factory.py` to use LiteLLM as the sole provider-routing backend, removing the old positional `base_url`/`api_key` parameters in favour of keyword-only arguments with auto-detection of cloud providers. Added `setup_workspace()` to `evidence_api.py` for stateless bash-mode bootstrapping. Fixed a critical notebook generation bug: the original `generate_notebook()` shelled out to `jupytext --to notebook`, which (a) leaked `# %%` and `%% [markdown]` markers into cell source, (b) left `# → …` output comments as code instead of cell outputs, and (c) split multi-line bash commands into multiple cells at blank lines. Replaced with a custom percent-format parser that produces clean `.ipynb` JSON with proper `stream` outputs.

Added Anthropic prompt caching (`cache_control: ephemeral`) to the system message and user message at each turn and context-refresh boundary. This enables Anthropic's server-side prefix caching so the system prompt (~3k tokens) and execution log are reused across the 10–17 LLM calls per claim instead of being re-processed each time.

### Prompt caching test results (1 SIGNOR claim: "GNAS directly activates ADCY1")

|  | Without caching | With caching |
|--|----------------|-------------|
| **Cost** | $0.527 | $0.501 (−5%) |
| **Wall clock** | 13m 28s | 12m 25s (−8%) |
| **LLM calls** | 12 | 17 |
| **Prompt tokens** | 151k | 272k |
| **Cache read tokens** | 0 | 157,850 (58%) |

Cost savings are modest (~5%) because (a) cache creation cost ($3.75/MTok) partially offsets read savings, and (b) unique tool-result suffixes grow per-turn and cannot be cached. The largest cost driver remains the O(n²) conversation growth pattern — each turn resends all prior turns.

### Token spend breakdown (cached run)

| Call | Cost | % | API Function |
|-----:|-----:|--:|-------------|
| 12 | $0.077 | 15% | Context-refresh → `get_evidence_summary` (full exec log resent) |
| 10 | $0.063 | 13% | `emit_verdict` (10th turn, ~25k tok context) |
| 11 | $0.059 | 12% | Last response before context refresh |
| 9 | $0.049 | 10% | `check_sufficiency` (9th turn) |
| 1–4 | $0.045 | 9% | Setup + search + extract (small, cached) |

### Wall clock breakdown

| Phase | Time | % |
|-------|-----:|--:|
| `extract_and_add_facts` (×2) | 8 min | 65% |
| `populate_paper_features + check_sufficiency` | 2.4 min | 19% |
| Search, emit verdict, other | 2 min | 16% |

## New Files

| File | Purpose |
|------|---------|
| `src/pkevolve/verification/evidence_programming_direct.py` | 749-line Direct API orchestrator: outer LLM loop with bash tool-use, per-iteration context refresh, Anthropic prompt caching, jupytext-format execution log, custom notebook generation |
| `experiments/configs/test_direct_config.yaml` | Smoke-test config using Claude Sonnet as both agent and subagent |
| `experiments/configs/test_direct_vllm_config.yaml` | Test config with Claude agent + Qwen3-8B subagent on local vLLM |
| `doc/ctx_management_improvements.md` | Design notes for context management architecture |

## Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/llm_factory.py` | Rewrote to use LiteLLM exclusively. Removed positional `base_url`/`api_key` params, made them keyword-only. Added `_CLOUD_PREFIXES` tuple to auto-detect cloud providers and skip `api_base` for them. Simplified module docstring. |
| `src/pkevolve/verification/config.py` | Updated `VerificationSettings.make_subagent_llm()` call signature to match new `make_llm(model=..., api_key=..., base_url=...)` keyword-only API. |
| `src/pkevolve/verification/evidence_api.py` | Updated `setup_kernel()` to use new `make_llm()` signature. Added `setup_workspace()` function (~100 lines) for stateless bash-mode bootstrapping — loads or creates `EvidenceState`, builds LLM callable from env vars, initialises label config. |
| `pyproject.toml` | Added `jupytext>=1.16` dependency. |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              evidence_programming_direct.py              │
│                                                         │
│  ┌──────────────┐    tool calls    ┌──────────────────┐ │
│  │  Outer LLM   │ ──────────────→  │  bash subprocess │ │
│  │  (Claude via  │                  │  python3 -c "…"  │ │
│  │   LiteLLM)   │ ←────────────── │  evidence_api.*  │ │
│  └──────────────┘    stdout/err    └──────────────────┘ │
│         │                                    │          │
│         │ append_to_jupytext_log()   setup_workspace()  │
│         ▼                                    │          │
│  ┌──────────────┐                   ┌────────▼───────┐  │
│  │  exec_log.py │                   │  Subagent LLM  │  │
│  │  (% format)  │                   │  (Qwen/vLLM    │  │
│  └──────┬───────┘                   │   via LiteLLM) │  │
│         │ generate_notebook()       └────────────────┘  │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ evidence_    │  Custom parser: no jupytext CLI        │
│  │ report.ipynb │  • Clean markdown (no %% markers)     │
│  └──────────────┘  • Code/output separation             │
│                    • Multi-line commands stay in 1 cell  │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

- **Anthropic prompt caching via `cache_control`.** System and user messages are wrapped with `cache_control: {"type": "ephemeral"}` when the agent model starts with `anthropic/`. LiteLLM passes this through to Anthropic's API. For non-Anthropic models the extra key is silently ignored. Two helpers `_cached_system_msg()` and `_cached_user_msg()` centralise the logic. Cache breakpoints are placed on: (1) the system prompt (static across all turns within a run), and (2) the user message at each context-refresh boundary (execution log, which grows but shares a long prefix with subsequent turns).

- **Custom notebook generator instead of jupytext CLI.** Jupytext 1.19.x cannot represent cell outputs in percent format and leaks boundary markers into cell content. Rather than post-processing the broken output, we parse the simple percent format directly (~60 lines) and emit proper `.ipynb` JSON with `stream` outputs from `# → …` lines.

- **LiteLLM as sole provider router.** Removed the dual `make_llm`/`make_llm_litellm` split. A single `make_llm()` function uses `litellm.completion()` internally. Cloud providers (Anthropic, OpenAI, etc.) are auto-detected via `_CLOUD_PREFIXES` — their `api_base` is not passed to LiteLLM since it handles routing natively. Local endpoints (vLLM, SGLang) pass `api_base` through.

- **`setup_workspace()` for bash-mode idempotency.** Each bash subprocess calls `setup_workspace()` which loads existing state from `evidence_state.json` or creates new state. This makes every `python3 -c "…"` invocation stateless and safe to retry — unlike the kernel-based approach which required persistent memory.

- **Keyword-only API for `make_llm()`.** Changed from `make_llm(base_url, api_key, model)` positional to `make_llm(model, *, api_key=None, base_url=None)` keyword-only. The `model` string is the primary identifier; `base_url` and `api_key` are optional overrides (inferred from environment for cloud providers).

## Bug Fixes

- **Notebook `# → …` output lines rendered as code.** Jupytext percent format has no output representation, so captured stdout lines written as `# → …` comments appeared inside code cells in the converted notebook. Fixed by parsing `# →` prefixed lines and emitting them as `{"output_type": "stream", "name": "stdout"}` entries in the notebook JSON.

- **`%% [markdown]` text visible in markdown cells.** Jupytext left the `%% [markdown]` cell-type marker as the first line of markdown cell content. The custom parser strips it and removes `# ` prefixes to recover clean markdown.

- **Multi-line bash commands split across cells.** Jupytext treated blank lines inside `python3 -c "…"` blocks as cell boundaries, scattering a single command across 3–5 code cells. The custom parser only splits on `# %%` boundary markers, keeping multi-line content intact.
