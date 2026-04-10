# Baseline Evaluation Pipeline — 2026-03-28

**Branch:** `exp/init`

## Summary

Implemented the initial comparison baseline infrastructure for the evidence programming experiments, prioritising the Random and LLM-only baselines as specified in `doc/baseline_implementation_plan.md`. Shared infrastructure (label normalisation, verdict schemas, LLM backend, cost tracker, prompt templates, evaluation harness) was built first to ensure all future baselines compare fairly on identical foundations. Code was placed in `experiments/baselines/` rather than `src/` to keep the core `pkevolve` library separate from experiment-specific comparison code.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/__init__.py` | Package marker |
| `experiments/baselines/shared/__init__.py` | Package marker |
| `experiments/baselines/shared/label_utils.py` | `normalize_label()` — maps all raw dataset labels (`SUPPORTED`, `WRONG`, `CONTRADICT`, `SUPPORTS`, `REFUTED`, …) to canonical `{SUPPORT, REFUTE, NEI}` |
| `experiments/baselines/shared/verdict.py` | `Verdict` and `BaselineResult` Pydantic v2 schemas used by every baseline and the harness |
| `experiments/baselines/shared/llm.py` | `LLMBackend` — OpenAI-compatible client wrapper with retry logic; resolves `GLM_API_KEY → ZAI_API_KEY → OPENAI_API_KEY` |
| `experiments/baselines/shared/cost_tracker.py` | `CostTracker` — accumulates token counts and USD cost per claim run with per-model pricing table |
| `experiments/baselines/shared/prompts.py` | Shared verification prompt templates (same prompts across all baselines for fair comparison) |
| `experiments/baselines/shared/evaluate.py` | `EvaluationHarness` — runs any baseline on a claim list; computes accuracy, macro-F1, binary-F1 (SUPPORT vs REFUTE excluding NEI), per-class P/R/F1; saves JSONL + JSON |
| `experiments/baselines/random_baseline.py` | `RandomBaseline` — uniform random over `{SUPPORT, REFUTE, NEI}`, 0 API calls |
| `experiments/baselines/llm_only.py` | `LLMOnly` — parametric knowledge only, no retrieval, temperature=0, structured JSON verdict |
| `experiments/run_baselines_signor.py` | CLI runner: loads SIGNOR ground truth, builds claim variants (with optional flip logic mirroring `run_signor_eval.py`), runs selected baseline, saves results + metrics + summary |

## Architecture

```
experiments/
├── run_baselines_signor.py       ← CLI entry point: load claims → run baseline → save
├── run_signor_eval.py            ← existing evidence programming runner (unchanged)
└── baselines/
    ├── random_baseline.py        ← RandomBaseline (0 LLM calls)
    ├── llm_only.py               ← LLMOnly (1 LLM call per claim)
    └── shared/
        ├── label_utils.py        ← normalize_label() for all datasets
        ├── verdict.py            ← BaselineResult schema
        ├── llm.py                ← LLMBackend (OpenAI-compatible)
        ├── cost_tracker.py       ← CostTracker (tokens + USD)
        ├── prompts.py            ← shared prompt templates
        └── evaluate.py           ← EvaluationHarness (metrics + persistence)

results/baselines/
└── <baseline>_signor/
    ├── results.jsonl             ← one BaselineResult per claim
    ├── metrics.json              ← aggregated metrics
    └── summary.json             ← run metadata + metrics combined
```

## Key Design Decisions

- **Shared prompts across baselines** — all baselines that call an LLM use the same `VERIFICATION_SYSTEM_PROMPT` / `VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL`. The only variable is what evidence is provided, isolating architectural differences from prompt differences.
- **Canonical label taxonomy** — `normalize_label()` is applied to both gold and predicted labels before any metric computation, so datasets with different vocabularies (SIGNOR's `SUPPORTED`/`WRONG`, SciFact's `CONTRADICT`, CIViC's `SUPPORTS`/`REFUTES`) are all comparable.
- **Binary F1 as secondary metric** — macro-F1 is primary; binary F1 (SUPPORT vs REFUTE, excluding NEI gold rows) is reported as secondary, following the plan's recommendation for SIGNOR where only 4 UNCERTAIN edges make the NEI class unreliable.
- **`experiments/baselines/` not `src/baselines/`** — baselines are experiment-specific comparison code, not reusable library components. Keeping them in `experiments/` makes the `src/pkevolve/` boundary clear.
- **Flip logic mirrors `run_signor_eval.py`** — `construct_signor_claim()` and `get_flipped_label()` in the runner replicate the existing logic exactly to ensure consistent claim strings across the evidence programming system and the baselines.
- **`sys.path` set to `SCRIPT_DIR`** — the runner adds `experiments/` to `sys.path` so `import baselines` resolves correctly regardless of working directory.
