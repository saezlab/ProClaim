---
description: "Run the random baseline (3 repeats) on one or more datasets and report accuracy, weighted FPR, FNR, and cost."
argument-hint: "Dataset name(s), e.g. signor connectomedb"
agent: "agent"
---

Run the random baseline on `$DATASETS` with **3 repeats** (seeds 100–102) and print a summary table.

## Steps

1. Execute the baseline runner from the project root:

```bash
uv run python experiments/run_baselines_datasets.py \
    --baseline random \
    --repeats 3 \
    --seed 100 \
    --datasets $DATASETS
```

2. After the run completes, read the aggregated metrics JSON(s) produced under
   `results/baselines/random/<dataset>_metrics.json`.

3. Report results as a LaTeX table row (one row per dataset pair) ready to be
   inserted into the main results table.  Use `accuracy`, `macro_fpr`, and
   `macro_fnr` from the metrics JSON; the random baseline has zero token cost.

   Format each value as `$<mean>$` (2 decimal places). Emit a `\midrule`-separated
   row block that fits the column layout below:

   ```latex
   \midrule
   Random & $<acc_signor>$ & -- & $<fpr_signor>$ & $<fnr_signor>$ & 0.0
          & $<acc_connectomedb>$ & -- & $<fpr_connectomedb>$ & $<fnr_connectomedb>$ & 0.0 \\
   ```

   The parent table uses this column spec (for reference):
   ```
   Method | Acc↑ | Δ↑ | FPR↓ | FNR↓ | Cost↓  ||  Acc↑ | Δ↑ | FPR↓ | FNR↓ | Cost↓
   ```
   where the first five metric columns are for **SIGNOR*** and the last five are
   for **ConnectomeDB**.  If only one dataset was run, fill the missing columns
   with `--`.

   Also print the mean ± std values in a brief plain-text note below the LaTeX
   snippet for traceability.

## Notes

- If `$DATASETS` is not provided, default to `signor connectomedb`.
- Datasets must have a corresponding CSV in the connectomeDB datasets directory
  (default: `/path_to/connectomeDB_data/datasets/`).
- Each repeat `i` uses `seed + i` (100, 101, 102), making results reproducible.
- Results are saved to `results/baselines/random/` relative to the project root.
