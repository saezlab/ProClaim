# Baseline Refactoring — 2026-05-01

**Branch:** `exp`

## Summary

Refactored the experiments baselines layer to extract duplicated logic into a set of small shared helpers, eliminating copy-paste across the eight baseline classes. The refactor also wired per-claim prompt logging into FIRE, SAFE, and all static-retrieval baselines, moved upstream repository adapters (FIRE, SAFE) into a dedicated module, added a test suite for the new shared utilities, and removed three stale SLURM batch scripts and two analysis scripts that had been superseded.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/shared/builder.py` | `build_llm_backend(args)`: constructs a `LLMBackend` from parsed CLI args; avoids repeating the same five-argument constructor in every baseline's `main()`. |
| `experiments/baselines/shared/logging_utils.py` | `write_claim_log(log_dir, claim_id, sections)`: writes a named-section `.log` file per claim when `--log-dir` is set; no-ops if `log_dir` is `None`. |
| `experiments/baselines/shared/search_utils.py` | `ddg_text_search` / `ddg_text_search_with_retry`: thread-safe DuckDuckGo wrapper extracted from `react_baseline.py`; `format_ddg_body_results` / `format_s2_detailed_results`: passage-formatting helpers shared by retrieval and SAFE baselines. |
| `experiments/baselines/shared/single_shot.py` | `run_single_shot_verdict`: runs a single LLM JSON verdict call, tracks token usage via `CostTracker`, normalises the label, and returns a `SingleShotVerdict` dataclass; replaces identical boilerplate in `llm_only.py`, `single_paper.py`, `retrieval_baseline.py`, and `s2_plus_ref.py`. |
| `experiments/baselines/shared/upstream_adapters.py` | `FireModelAdapter`, `SafeModelAdapter`, `import_upstream_module`, `do_search`: adapter layer between `LLMBackend` and the upstream FIRE / SAFE repositories, extracted from `fire_baseline.py` and `safe_baseline.py`. |
| `tests/test_baseline_shared_utils.py` | Unit tests for `write_claim_log`, `format_s2_detailed_results`, and `run_single_shot_verdict` using a `_FakeLLM` stub. |

## Modified Files

| File | Changes |
|------|---------|
| `experiments/baselines/fire_baseline.py` | Migrated `FireModelAdapter` and `import_upstream_module` to `upstream_adapters`; switched to `write_claim_log` for per-claim logging; removed ~60 lines of inline adapter code. |
| `experiments/baselines/safe_baseline.py` | Same migration: `SafeModelAdapter` moved to `upstream_adapters`; prompt logging via `write_claim_log`. |
| `experiments/baselines/react_baseline.py` | Extracted DuckDuckGo helpers to `search_utils`; simplified class body by ~40 lines. |
| `experiments/baselines/retrieval_baseline.py` | Switched verdict call to `run_single_shot_verdict`; search formatting to `format_s2_detailed_results`; removed ~80 lines of duplicate JSON-parsing code. |
| `experiments/baselines/single_paper.py` | Switched verdict call to `run_single_shot_verdict`; removed duplicate parsing boilerplate. |
| `experiments/baselines/s2_plus_ref.py` | Same as `single_paper.py`. |
| `experiments/baselines/llm_only.py` | Switched to `run_single_shot_verdict`. |
| `experiments/baselines/ace_baseline.py` | Minor: switched to `write_claim_log`. |
| `experiments/baselines/open_scholar_baseline.py` | Minor: switched to `write_claim_log`. |
| `experiments/baselines/shared/cost_tracker.py` | Added `summary()` method returning a cost dict; consumed by `run_single_shot_verdict`. |
| `experiments/baselines/shared/prompts.py` | Removed the remaining duplicate `VERIFICATION_USER_TEMPLATE` definitions (follow-up cleanup from the 2026-05-01 merge). |
| `experiments/run_baselines_datasets.py` | Simplified baseline factory to use `build_llm_backend`; removed per-baseline LLM construction boilerplate (~30 lines). |
| `scripts/analysis/plot_sandbox_vs_wild_transitions.py` | Added `draw_confusion_matrix_triple`: three-panel figure (baseline confusion matrix + two delta matrices) replacing the original alluvial-only figure for Section 2. |
| `scripts/analysis/print_baseline_metrics_table.py` | Added `safe` and `s2_plus_ref` to `BASELINE_ORDER` and `VARIANT_ORDER`. |

## Deleted Files

| File | Reason |
|------|---------|
| `experiments/configs/ligand_receptor_{direct,original,original_mlp}_config.yaml` | Superseded by unified `--datasets-dir` / `--baseline` CLI flags. |
| `scripts/run_ligand_receptor_{batch,original_form,original_mlp}.sh` | Superseded by `run_baselines_datasets.py` SLURM integration. |
| `scripts/analysis/plot_sandbox_vs_wild_gold_summary.py` | Replaced by `plot_sandbox_vs_wild_transitions.py` triple confusion-matrix panel. |
| `scripts/analysis/sandbox_vs_wild_gold_analysis.py` | Same; superseded by `find_changed_claims` inside `plot_sandbox_vs_wild_transitions.py`. |

## Architecture

```
experiments/baselines/
  shared/
    builder.py          ← build_llm_backend(args) → LLMBackend
    single_shot.py      ← run_single_shot_verdict(...) → SingleShotVerdict
    search_utils.py     ← ddg_text_search*, format_*_results
    logging_utils.py    ← write_claim_log(log_dir, claim_id, sections)
    upstream_adapters.py← FireModelAdapter, SafeModelAdapter, import_upstream_module
    llm.py              ← LLMBackend (unchanged)
    cost_tracker.py     ← CostTracker + summary()
    label_utils.py      ← normalize_label (unchanged)
    verdict.py          ← BaselineResult (unchanged)
  fire_baseline.py  ─┐
  safe_baseline.py  ─┤
  react_baseline.py ─┤── import shared helpers above
  retrieval_baseline.py ─┤
  single_paper.py   ─┤
  s2_plus_ref.py    ─┘
  llm_only.py
  ace_baseline.py
  open_scholar_baseline.py
```

## Key Design Decisions

- `run_single_shot_verdict` handles the full verdict call–parse–normalise–track cycle in one function, keeping each baseline's `verify()` method focused on evidence assembly only.
- `write_claim_log` is a no-op when `log_dir is None`, so baselines remain runnable without `--log-dir` and logging is purely opt-in.
- `upstream_adapters.py` isolates all `sys.path` manipulation and stub-injection needed for the FIRE and SAFE repositories in one place, preventing it from leaking into baseline logic.
- The DuckDuckGo lock (`_DDG_LOCK`) is kept in `search_utils` rather than inside `react_baseline.py` so SAFE can also benefit from thread-safe search without importing the ReAct baseline.
- Tests use a `_FakeLLM` stub rather than mocking at the `LLMBackend` level to keep tests independent of the LiteLLM API surface.

## Bug Fixes

- Removed the second shadowed `VERIFICATION_USER_TEMPLATE` in `shared/prompts.py` that omitted the `Output JSON.` instruction, which was causing silent JSON-parse failures in some model runs after the earlier merge.
