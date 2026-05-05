# Baseline Helper Refactor and Prompt Logging — 2026-05-01

**Branch:** `exp`

## Summary

Refactored the baseline evaluation stack to remove repeated helper logic while preserving each baseline's execution model. The runner now constructs LLM backends through a shared builder, pricing lookup is centralized, per-claim logs share one writer, repeated DuckDuckGo and Semantic Scholar formatting/search utilities were moved into shared modules, and the single-shot verdict path used by `llm_only`, `retrieval`, `single_paper`, and `s2_plus_ref` now flows through one shared helper. In parallel, per-claim logs were expanded so all baseline variants record the actual prompt or effective input payload sent to the model.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/shared/builder.py` | Builds `LLMBackend` instances from runner CLI/config arguments so LLM-backed baselines share one construction path. |
| `experiments/baselines/shared/logging_utils.py` | Writes per-claim `.log` files from structured sections, removing repeated file-open and section-formatting code. |
| `experiments/baselines/shared/search_utils.py` | Centralizes DuckDuckGo search wrappers plus shared Semantic Scholar and web result formatting helpers. |
| `experiments/baselines/shared/single_shot.py` | Implements the shared one-call verdict path for simple baselines: run LLM, parse JSON verdict, normalize labels, and return usage summary. |
| `tests/test_baseline_shared_utils.py` | Focused regression tests for the shared log writer, detailed S2 formatter, and single-shot verdict helper. |

## Modified Files

| File | Changes |
|------|---------|
| `experiments/run_baselines_datasets.py` | Switched LLM-backed baselines to the shared `build_llm_backend()` helper instead of repeating the same `LLMBackend(...)` construction block. |
| `experiments/README.md` | Documented that per-claim baseline logs now include the prompt or effective input payload. |
| `experiments/baselines/shared/cost_tracker.py` | Added `CostTracker.pricing_for()` so baselines can reuse the same model-pricing lookup instead of duplicating it locally. |
| `experiments/baselines/llm_only.py` | Replaced inline single-shot verdict parsing with the shared `run_single_shot_verdict()` helper. |
| `experiments/baselines/retrieval_baseline.py` | Reused shared DDG/S2 formatting, used the shared single-shot verdict helper, and logged system/user prompts in per-claim logs. |
| `experiments/baselines/single_paper.py` | Reused the shared single-shot verdict helper and moved per-claim log writing onto the shared logging utility. |
| `experiments/baselines/s2_plus_ref.py` | Reused the shared single-shot verdict helper, shared S2 formatting utility, and shared per-claim log writer. |
| `experiments/baselines/ace_baseline.py` | Reused shared pricing lookup and the shared per-claim log writer while preserving ACE-specific prompt and playbook logging. |
| `experiments/baselines/fire_baseline.py` | Reused shared pricing, DDG/S2 search-format helpers, and shared per-claim log writing while continuing to capture iterative FIRE prompts. |
| `experiments/baselines/safe_baseline.py` | Reused shared pricing, DDG/S2 search-format helpers, and shared per-claim log writing while continuing to capture iterative SAFE prompts. |
| `experiments/baselines/react_baseline.py` | Reused shared pricing and logging helpers, and moved the remaining private detailed S2 formatter into the shared search utility module. |
| `experiments/baselines/open_scholar_baseline.py` | Reused the shared per-claim log writer for subprocess prompt, payload, command, stdout, and stderr logging. |

## Architecture

```text
Baseline Runner and Shared Helper Flow
======================================

run_baselines_datasets.py
        |
        v
build_llm_backend(args)
        |
        +-------------------------------+
        |                               |
        v                               v
simple baselines                  iterative / subprocess baselines
llm_only                          fire / safe / react / ace / open_scholar
retrieval                         
single_paper                      
 s2_plus_ref                      
        |                               |
        | system_prompt + user_prompt   | prompts/searches/subprocess payloads
        v                               v
run_single_shot_verdict()         baseline-specific control loop
        |                               |
        +---------------+---------------+
                        |
                        v
               write_claim_log(...)
                        |
                        v
                  BaselineResult
```

## Key Design Decisions

- Share helpers by function, not through a heavyweight baseline superclass. The baselines still split cleanly into single-shot, iterative, agentic, and subprocess-backed execution models.
- Keep prompt content baseline-specific while centralizing the mechanics around pricing lookup, log writing, search formatting, and one-shot verdict parsing.
- Preserve baseline-specific evidence outputs even when sharing the single-shot verdict helper. For example, `s2_plus_ref` still returns PMID evidence assembled from the reference paper plus S2 results rather than raw model-cited evidence.
- Add regression tests only for the new shared utility layer. This gives coverage for the refactor without requiring full model-backed baseline execution in tests.
- Treat OpenScholar separately at the execution layer but reuse the shared log writer so its subprocess payload logging stays aligned with the in-process baselines.

## Bug Fixes

- Fixed missing prompt visibility in per-claim logs across baseline implementations by recording the actual system/user prompt or effective subprocess payload sent to the model.
- Fixed helper drift in the baseline runner by eliminating repeated `LLMBackend(...)` construction blocks that had to be kept manually in sync.
- Fixed duplicated model-pricing lookup logic across multiple baselines by moving it into `CostTracker.pricing_for()`.
- Fixed formatting duplication for Semantic Scholar and DuckDuckGo evidence snippets by moving shared renderers into `shared/search_utils.py`, including the last ReAct-specific detailed S2 formatter.
- Fixed duplicated single-shot verdict parsing across simple baselines by centralizing JSON parsing, label normalization, confidence extraction, and usage summary assembly in `shared/single_shot.py`.
