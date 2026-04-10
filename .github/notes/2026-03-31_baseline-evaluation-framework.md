# Baseline Evaluation Framework — 2026-03-31

**Branch:** exp

## Summary

Introduced a structured, multi-dataset baseline evaluation framework for claim verification. The framework supports running stochastic baselines (currently random) across multiple repeats with fixed seeds for reproducibility, computing aggregated mean ± std metrics, and saving per-claim JSONL results. A new `run_baselines_datasets.py` script targets pre-processed dataset CSVs from the connectomeDB data directory and was verified to reproduce existing random baseline results exactly (SIGNOR n=111, ConnectomeDB n=547, 10 repeats, seed=100–109).

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/__init__.py` | Package init for baselines module |
| `experiments/baselines/random_baseline.py` | Random label assignment baseline (uniform over SUPPORT/REFUTE/NEI) |
| `experiments/baselines/llm_only.py` | LLM-only (no retrieval) baseline — parametric knowledge ceiling |
| `experiments/baselines/shared/__init__.py` | Shared utilities package init |
| `experiments/baselines/shared/evaluate.py` | `EvaluationHarness`: runs a baseline, computes metrics (accuracy, macro F1, binary F1, per-class P/R/F1), saves JSONL |
| `experiments/baselines/shared/cost_tracker.py` | Token and USD cost tracking across baseline runs |
| `experiments/baselines/shared/label_utils.py` | Label normalisation (SUPPORTED→SUPPORT, REFUTES→REFUTE, etc.) |
| `experiments/baselines/shared/llm.py` | Thin LLM client wrapper for single-call baselines |
| `experiments/baselines/shared/prompts.py` | Shared prompt templates for LLM-based baselines |
| `experiments/baselines/shared/verdict.py` | Structured verdict parsing from LLM output |
| `experiments/run_baselines_datasets.py` | CLI runner: loads dataset CSVs, runs N repeats per seed, aggregates metrics, saves results |
| `experiments/run_baselines_signor.py` | Earlier SIGNOR-specific baseline runner (superseded by `run_baselines_datasets.py` for multi-dataset use) |
| `doc/evaluation_plan.md` | Full evaluation design: dataset descriptions, compared systems, metrics, implementation notes |

## Modified Files

| File | Changes |
|------|---------|
| `doc/claim_verification_datasets.md` | Expanded with ConnectomeDB and SIGNOR dataset construction details, evaluation subsets, and claim templates for CIViC-Fact |

## Architecture

```
experiments/run_baselines_datasets.py   ← CLI entry point
        │
        ├─ load_claims(csv_path)         CSV → list[claim_dict]
        │
        ├─ build_baseline(name, seed)
        │       └─ RandomBaseline        uniform sample from {SUPPORT,REFUTE,NEI}
        │
        └─ EvaluationHarness.run()       per-repeat predict + score
                └─ EvaluationHarness.metrics()
                        accuracy, macro_f1, binary_f1, per_class P/R/F1
                        → aggregate_metrics()  mean ± std over repeats
                        → save JSONL + metrics JSON
```

Results layout:
```
results/baselines/{baseline}/{dataset}_seed{seed}.jsonl   (per-claim predictions)
results/baselines/{baseline}/{dataset}_metrics.json       (aggregated stats)
```

## Key Design Decisions

- **Fixed-seed repeats**: each repeat `i` uses `seed + i`, making all runs deterministic and reproducible across machines.
- **Stochastic aggregation**: metrics are summarised as mean ± std over repeats rather than a single run, giving a stable estimate of chance-level performance.
- **CSV-driven input**: baselines consume pre-processed dataset CSVs (columns: `id`, `claim`, `label`, optional `dataset`) so the same runner works for any dataset without code changes.
- **Separate JSONL per repeat**: enables per-seed inspection and re-aggregation without re-running.
- **Binary F1 computed over SUPPORT vs. {REFUTE, NEI}**: reflects the primary verification task (supported vs. not supported) independent of the NEI distinction.

## Verification

Random baseline results reproduced exactly against stored outputs (`results/baselines/random/`):

| Dataset | accuracy | macro_f1 | binary_f1 |
|---------|----------|----------|-----------|
| SIGNOR (n=111) | 0.3261 ± 0.0447 | 0.2854 ± 0.0379 | 0.5684 ± 0.0483 |
| ConnectomeDB (n=547) | 0.3340 ± 0.0262 | 0.3260 ± 0.0265 | 0.5603 ± 0.0365 |
