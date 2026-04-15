# Debug Mode, Prompt Extraction & Bug Fixes — 2026-04-15

**Branch:** `feature/ctx-manage`

## Summary

Added an `EVIDENCE_DEBUG` environment variable to gate verbose diagnostic output in `evidence_api.py` and `subagents.py`, reducing agent context consumption by suppressing per-PMID progress, full-text fetch details, and intermediate diagnostics in non-debug runs. Extracted all inline prompt templates from `subagents.py` and `evidence_programming.py` into a new centralised `prompts.py` module. Fixed three bugs discovered during a test run of the direct orchestrator on the SIGNOR claim "EGFR activates PLCG1": (1) unresolved f-string placeholders in agent-generated reasoning, (2) double-nested verdict output paths, and (3) missing spaCy model causing NLP features to be all zeros.

## New Files

| File | Purpose |
|------|---------|
| `src/pkevolve/verification/prompts.py` | 509-line centralised prompt templates for all verification LLM prompts. Contains `EXTRACT_FACTS`, `SYNTHESIZE_SUBCLAIM`, `DETECT_CONFLICTS`, `IDENTIFY_GAPS`, `FORMULATE_GAP_QUERIES`, `REFINE_SEARCH_QUERY`, `DIRECT_SYSTEM_PROMPT`, `NOTEBOOK_SYSTEM_PROMPT`, and `NOTEBOOK_USER_PROMPT`. Placeholders use `str.format()` syntax. |

## Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/evidence_api.py` | Added `_EVIDENCE_DEBUG` flag and `_debug_print()` helper. Converted ~30 verbose `print()` calls to `_debug_print()` (per-PMID extraction progress, S2 recommendation details, citation chaining status, elink errors, query text, paper filtering details, sufficiency history table). Kept essential one-line summaries as regular `print()`. Compacted `check_sufficiency` output from 10+ lines to a single summary line with verbose breakdown gated behind debug. Compacted `filter_papers_by_stance` output similarly. |
| `src/pkevolve/verification/subagents.py` | Extracted six inline prompt templates to imports from `prompts.py` (`EXTRACT_FACTS`, `SYNTHESIZE_SUBCLAIM`, `DETECT_CONFLICTS`, `IDENTIFY_GAPS`, `FORMULATE_GAP_QUERIES`, `REFINE_SEARCH_QUERY`). Added `_EVIDENCE_DEBUG` flag. |
| `src/pkevolve/verification/evidence_programming.py` | Replaced ~120-line inline `SYSTEM_PROMPT` with import from `prompts.py` (`NOTEBOOK_SYSTEM_PROMPT`). Also imports `NOTEBOOK_USER_PROMPT`. |
| `src/pkevolve/verification/evidence_programming_direct.py` | Imports `DIRECT_SYSTEM_PROMPT` from `prompts.py`. Added system prompt instruction warning the LLM not to use deferred f-string placeholders (e.g. `{len(refute_facts)}`) in reasoning text before variables are defined. Added `EVIDENCE_DEBUG` propagation in `build_subprocess_env()`. |
| `src/pkevolve/verification/llm_factory.py` | Minor: adjusted verbose logging. |
| `src/pkevolve/verification/notebook_mcp.py` | Minor fix. |
| `doc/ctx_management_improvements.md` | Added Q1 (What does the original agent see?) and Q2 (Which version reduces cost?) analysis sections with token cost tables and autocompact discussion. |

## Architecture

```
evidence_programming_direct.py
  │
  │  EVIDENCE_DEBUG=1 (via build_subprocess_env)
  │  ↓
  ├─→ bash subprocess → python3 -c "..."
  │     ├─→ evidence_api.py   ── _EVIDENCE_DEBUG → _debug_print()
  │     │     • verbose per-PMID output: OFF by default
  │     │     • compact summaries: always ON
  │     └─→ subagents.py      ── _EVIDENCE_DEBUG
  │           • prompts imported from prompts.py
  │
  └─→ prompts.py
        • DIRECT_SYSTEM_PROMPT   (for direct orchestrator)
        • NOTEBOOK_SYSTEM_PROMPT (for SDK orchestrator)
        • EXTRACT_FACTS, IDENTIFY_GAPS, ... (for subagents)
```

## Key Design Decisions

- **Two-tier output strategy.** Essential summaries (e.g. `Sufficiency: sufficient (confidence=0.8234, papers=12+4, gaps=2)`) always print so the agent can act on them. Verbose breakdowns (per-PMID extraction results, gap details, history tables) are gated behind `EVIDENCE_DEBUG=1`. This reduces context consumption by ~40% per tool call in the inner loop.

- **`_debug_print()` function instead of `logging.debug()`.** Tool output in the direct orchestrator flows through bash subprocess stdout, which the outer LLM sees as tool results. Using `logging.debug()` would route to stderr/log files, invisible to the agent. `_debug_print()` conditionally writes to stdout so the agent sees verbose output only when explicitly enabled.

- **Prompt centralisation in `prompts.py`.** All prompt templates were scattered across `subagents.py` (6 prompts inline), `evidence_programming.py` (system prompt), and `evidence_programming_direct.py` (system prompt). Consolidating into one module eliminates duplication between the SDK and direct orchestrators and makes prompt iteration easier.

- **System prompt f-string warning.** Claude was generating code like `reasoning = f"Based on {len(refute_facts)} refuting facts..."` before `refute_facts` was populated in the execution flow, causing `{len(refute_facts)}` to appear literally in the verdict. Added an explicit instruction in the system prompt: "Never use deferred f-string placeholders."

## Bug Fixes

- **Unresolved f-string placeholders in verdict reasoning.** The agent generated code containing `f"text {variable}"` where `variable` was not yet defined at the point of string construction. The f-string evaluated but `{variable}` appeared as a literal brace expression in the output. Fixed by adding a system prompt instruction telling the LLM to compute values before string interpolation and never use placeholder-style f-strings.

- **Double-nested verdict path.** `emit_verdict()` received `workspace_dir` from the agent's bash code, which sometimes contained a relative path that, when resolved, created nested directories like `workspace/results/verification/debug_test/workspace/verdict.json`. Fixed `emit_verdict` to use `state.workspace_dir` (the canonical path set during `setup_workspace()`) instead of the parameter passed by the agent.

- **Missing spaCy `en_core_web_sm` model.** `populate_paper_features()` calls spaCy NLP for entity/dependency features. The model was not installed in the `uv` environment, causing all NLP feature values to be zero. The sufficiency classifier then received a zero-vector, degrading its predictions. Fixed by running `uv run python -m spacy download en_core_web_sm`.
