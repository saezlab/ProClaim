# OpenScholar Baseline — Reranker, Flip-Pair Fix, Prompt Fix, Full SIGNOR Eval — 2026-04-05

**Branch:** `exp`

## Summary

Ran the full SIGNOR evaluation (111 claims) of OpenScholar (Claude Sonnet 4.6) with live S2 adaptive retrieval, the `OpenScholar/OpenScholar_Reranker` cross-encoder, and the self-reflective feedback loop. Three bugs were discovered and fixed across two evaluation runs: a flip-pair `claim_id` collision that silently corrupted ~44 SUPPORT-class predictions; missing reranker CLI flags that cause a crash when `--ss_retriever --feedback` is active; and a wrong prompt format where the baseline was wrapping every claim as a framed question rather than passing the bare claim text expected by the `claim_verdict` task. The third bug invalidated the first full run; a corrected run is in progress.

## Modified Files

| File | Changes |
|------|---------|
| `experiments/baselines/open_scholar_baseline.py` | Added `reranker` parameter to `__init__` (default `"OpenScholar/OpenScholar_Reranker"`); `_build_cmd()` now appends `--ranking_ce --reranker <model>` whenever `use_retrieval=True` and `reranker` is set; default `subprocess_timeout` auto-scaled to 600 s when retrieval is enabled; `framed_input` changed from a question wrapper to the raw `claim` string so the input matches the `"\nClaim: "` instance header in `claim_verdict` |
| `experiments/run_baselines_datasets.py` | `build_baseline()` passes `reranker=args.os_reranker` to `OpenScholarBaseline`; added `--os-reranker` CLI flag (default `"OpenScholar/OpenScholar_Reranker"`, empty string disables); `load_claims()` disambiguates flip-pair rows by appending `_flip` to the `claim_id` when the CSV `flip` column is `True` |

## Key Design Decisions

- **`_flip` suffix for claim IDs** — The SIGNOR CSV contains flip pairs that share the same base `id`. The `EvaluationHarness` stores results in a `dict` keyed by `claim_id`; without disambiguation the flip=False result is overwritten by flip=True (or vice versa), making both rows receive the same verdict. Appending `_flip` is the minimal, non-destructive fix and is transparent to downstream metrics code.
- **Reranker always on with retrieval** — Per the OpenScholar note in `run-signor-claim.prompt.md`, omitting `--reranker` when `--ss_retriever --feedback` is active causes `AttributeError: 'NoneType'.compute_score`. The new default bundles the two together so the baseline is safe by default; passing `--os-reranker ""` still allows disabling it.
- **Empty string → `None` coercion** — `reranker=getattr(args, "os_reranker", None) or None` converts the empty-string sentinel to `None` so `_build_cmd` can use a simple `if self.reranker:` guard.
- **Truncation detection and retry** — Claims where `predicted_label == "NEI"` and `reasoning` starts with `{` are truncated JSON fragments. These were stripped from the checkpoint file and retried with `max_tokens=6000`. No architectural change was needed — the existing resume logic handled the rerun automatically.
- **Raw claim as input, not a framed question** — `open_scholar_baseline.py` was wrapping every claim as `"What does the scientific evidence say about {claim}\nIs it supported or refuted by the literature?"`. OpenScholar then prepended the `"\nClaim: "` instance header from `claim_verdict`, giving the model `"Claim: What does the scientific evidence say about …?"` — structurally a question rather than a claim. The model consequently ignored retrieved references and answered from parametric knowledge. Fix: pass `claim` directly as `framed_input`; the task instruction and instance header handle framing.

## Bug Fixes

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Flip-pair `claim_id` collision corrupting ~44 SUPPORT predictions | Both flip variants of a SIGNOR pair share the same `id` in the CSV; the `result_map` dict in `EvaluationHarness` kept only the last-seen result | `load_claims()` appends `_flip` suffix to `claim_id` when `flip == True` |
| `AttributeError: 'NoneType'.compute_score` crash with `--ss_retriever --feedback` | `--ranking_ce --reranker` flags were not passed to OpenScholar subprocess | `_build_cmd()` now includes those flags when `use_retrieval=True` and `self.reranker` is set |
| 6 claims parsed as `NEI` due to truncated JSON output | 3000-token default output limit cut the JSON mid-object | Detected via `reasoning.startswith("{")` pattern; retried with `max_tokens=6000` |
| Wrong prompt format — model answered from parametric knowledge, ignoring retrieved passages | `open_scholar_baseline.py` wrapped the claim as a question; after OpenScholar prepended `"Claim: "`, the model received `"Claim: What does the scientific evidence say about …?"` instead of a declarative claim | `framed_input = claim` — pass raw claim text; `claim_verdict`'s own system instruction and `"\nClaim: "` header provide the correct framing |

## Results

> **Note:** The metrics below are from the first completed run (with the flip-pair fix and reranker fix applied, but **before** the prompt format fix). A corrected run with `framed_input = claim` is in progress; these numbers should be considered preliminary.

**Dataset:** SIGNOR (111 claims) — seed 100, 1 repeat  
**Config:** `top_n=10`, `--ss_retriever --feedback`, `--ranking_ce --reranker OpenScholar/OpenScholar_Reranker`, no external max_tokens constraint (3000 default / 6000 for retried claims)

| Metric | Value |
|--------|-------|
| Accuracy | 0.5495 |
| Macro F1 | 0.4484 |
| Weighted FPR | 0.2122 |
| Weighted FNR | 0.4505 |
| Total cost | $1.78 USD |

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| SUPPORT | 0.875 | 0.396 | 0.546 |
| REFUTE | 0.633 | 0.731 | 0.679 |
| NEI | 0.074 | 0.333 | 0.121 |

Predicted distribution: SUPPORT=24, REFUTE=60, NEI=27 (gold: SUPPORT=53, REFUTE=52, NEI=6).
Results saved to `results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.jsonl` and `signor_metrics.json`.
