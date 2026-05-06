# Fix LLM JSON Parsing & Auto-Generated Function Signatures — 2026-03-07

**Branch:** `RLM`

## Summary

Fixed the root cause of `identify_gaps` always falling back to generic gaps, restructured `check_sufficiency` output for agent readability, and replaced hard-coded function signatures in system prompts with auto-generated docs from `inspect.signature()`. Also simplified the `llm()` callable by removing the `/v1/completions` fallback (now redundant with vLLM's `--reasoning-parser`).

## Modified Files

| File | Changes |
|------|---------|
| `src/proclaim/verification/subagents.py` | Added `_clean_llm_json()` (strips `<think>` tags + code fences), `_extract_json_array()` (right-to-left bracket search for robust JSON extraction). Replaced duplicated parsing logic in `_parse_facts_response`, `detect_conflicts`, and `_parse_gaps_response` with the shared helpers. Added debug logging/printing on parse failure. |
| `src/proclaim/verification/evidence_api.py` | Restructured `check_sufficiency` print block: single structured output with `Label`, `Confidence` (6 decimals), `Threshold`, `Decision: PASS/FAIL`, numbered gaps with subclaim + action. Added `function_docs()` using `inspect.signature()` + `_short_sig()` regex to auto-generate function signatures. Fixed module docstring `check_sufficiency(state)` → `check_sufficiency(state, llm)`. |
| `src/proclaim/verification/evidence_programming.py` | Replaced hard-coded 16-line function signature block in `SYSTEM_PROMPT` with `{function_docs}` placeholder. Wired `function_docs()` into `.format()` call. Simplified `llm()` template: removed `_THINK_RE`, `reasoning_content` collection, and `/v1/completions` fallback — now only reads `delta.content` from streaming chat completions. |
| `src/proclaim/verification/kernel_runner.py` | Same `llm()` simplification as `evidence_programming.py` — removed completions fallback, `_THINK_RE`, and `reasoning_content`. |
| `src/proclaim/verification/repl_orchestrator.py` | Replaced hard-coded function signatures with `{function_docs}` placeholder. Fixed stale `check_sufficiency(state)` in workflow section → `check_sufficiency(state, llm)`. Wired `function_docs()` into `.format()` call. |
| `experiments/example_config.yaml` | Minor config update. |

## Key Design Decisions

- **Right-to-left bracket search in `_extract_json_array`**: When reasoning text contains stray brackets like `[the evidence]`, a left-to-right greedy match (`r"\[.*\]"`) captures everything from the first `[` in reasoning to the last `]` of the JSON → invalid JSON. Searching from the rightmost `[` backwards tries the actual JSON array first and skips reasoning noise.
- **`_clean_llm_json` strips `<think>` before bracket search**: Defense-in-depth for the `/v1/completions` path (which doesn't go through vLLM's reasoning parser). Stripping first eliminates bracket noise from thinking blocks, making extraction faster and more reliable.
- **Removed `/v1/completions` fallback from `llm()`**: With `--reasoning-parser qwen3` correctly configured in the vLLM launcher, the chat completions endpoint handles thinking models natively. The completions fallback was the source of the original reasoning-content contamination bug and is no longer needed.
- **Only read `delta.content`, ignore `reasoning_content`**: vLLM's reasoning parser splits thinking tokens into `reasoning_content` and the actual answer into `content`. The `llm()` callable only needs the answer. This is the simplest correct approach — no regex stripping needed.
- **Auto-generated function signatures via `inspect.signature()`**: Prevents system prompt signatures from drifting as the API evolves. `_short_sig()` uses regex `r"[a-z_]+(?:\.[a-z_]+)*\.([A-Z]\w*)"` to strip module paths (e.g. `proclaim.verification.evidence_state.EvidenceState` → `EvidenceState`).

## Bug Fixes

- **`identify_gaps` always returning fallback gaps**: Root cause was `llm()` concatenating `reasoning_content` + `content` (`''.join(_reasoning) + ''.join(_content)`). When vLLM's reasoning parser is active, `reasoning_content` contains clean reasoning text (no `<think>` tags), so `_THINK_RE` couldn't strip it. The reasoning text (with stray `[` chars) was prepended to the JSON, breaking all parsers. Fixed by only reading `content`.
- **All three JSON parsers in `subagents.py` lacked `<think>` stripping**: Even when `<think>` tags were present (e.g. completions path), the parsers tried to parse them as JSON. The greedy regex `r"\[.*\]"` would match from a `[` inside reasoning to the `]` at the end of the JSON → invalid. Fixed via `_clean_llm_json()` applied before extraction.
- **Stale `check_sufficiency(state)` signature in system prompts**: Both Mode A and Mode B system prompts had hard-coded `check_sufficiency(state)` but the actual signature is `check_sufficiency(state, llm)`. The agent would call it incorrectly. Fixed by auto-generating signatures with `function_docs()`.
