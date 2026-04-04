---
description: "Run Claude Sonnet 4.6 and Gemini 3.1 Pro Preview LLM-only baselines (3 repeats) on one or more datasets and report accuracy, FPR, FNR, and cost as a LaTeX table matching the paper's main results format."
argument-hint: "Dataset name(s), e.g. signor connectomedb"
agent: "agent"
---

Run the `llm_only` baseline for **Claude Sonnet 4.6** (`anthropic/claude-sonnet-4-6`) and **Gemini 3.1 Pro Preview** (`vertex_ai/gemini-3.1-pro-preview`) on `$DATASETS` (default: `signor connectomedb`) and print the *LLM-only* rows of the paper's main results table.

> **Note on repeats**: `llm_only` runs at `temperature=0` (deterministic), so a single run is sufficient and `--repeats 3` has no additional effect. The script internally enforces `n_repeats=1` for this baseline. Std will be `0.0000`.

## Step 1 — Check for existing results

Before running, check whether pre-computed results already exist for **both** models. The output directory structure is:

```
results/baselines/llm_only/anthropic--claude-sonnet-4-6/<dataset>_metrics.json
results/baselines/llm_only/vertex_ai--gemini-3.1-pro-preview/<dataset>_metrics.json
```

For each dataset in `$DATASETS`, verify that both metric files above exist. If **all** expected files are present, skip Step 2 and go directly to Step 3.

## Step 2 — Run baselines (only if results are missing)

Run once per model that is missing results. Execute from the project root:

```bash
# Claude Sonnet 4.6
uv run python experiments/run_baselines_datasets.py \
    --baseline llm_only \
    --model anthropic/claude-sonnet-4-6 \
    --repeats 3 \
    --seed 100 \
    --datasets $DATASETS
```

```bash
# Gemini 3.1 Pro Preview
uv run python experiments/run_baselines_datasets.py \
    --baseline llm_only \
    --model vertex_ai/gemini-3.1-pro-preview \
    --repeats 3 \
    --seed 100 \
    --datasets $DATASETS
```

Wait for each command to finish before continuing.

## Step 3 — Read metrics

For each dataset in `$DATASETS`, read both metrics files:

- `results/baselines/llm_only/anthropic--claude-sonnet-4-6/{dataset}_metrics.json`
- `results/baselines/llm_only/vertex_ai--gemini-3.1-pro-preview/{dataset}_metrics.json`

Extract from each JSON (each field is `{"mean": float, "std": float}`):

| JSON key         | Table column |
|------------------|--------------|
| `accuracy.mean`  | Acc          |
| `macro_fpr.mean` | FPR          |
| `macro_fnr.mean` | FNR          |
| `total_cost_usd.mean` | Cost    |

Round all values to 2 decimal places. The Δ (delta vs. random) column is left as `--` (computed externally).

## Step 4 — Print LaTeX rows

Output **only the LLM-only rows** ready to paste into the paper's main results table. Use the exact format below (values to 2 decimal places, no `\pm` needed for deterministic runs):

```latex
\multicolumn{11}{@{}l}{\textit{LLM-only}} \\[2pt]
gemini-3.1-pro-preview & X.XX & -- & X.XX & X.XX & X.XX & X.XX & -- & X.XX & X.XX & X.XX \\
claude-sonnet-4.6      & X.XX & -- & X.XX & X.XX & X.XX & X.XX & -- & X.XX & X.XX & X.XX \\
```

Column order per dataset block: **Acc, Δ, FPR, FNR, Cost** — matching the full table header:

```latex
% Header reminder (do not re-emit, shown for reference only):
% \textbf{Method} | Acc↑  Δ↑  FPR↓  FNR↓  Cost↓ || Acc↑  Δ↑  FPR↓  FNR↓  Cost↓
%                    SIGNOR* (5 cols)                  ConnectomeDB* (5 cols)
```

If only one dataset was run, emit only the 6-column variant (Method + 5 metric columns) and note which dataset is missing.

Also print a plain-text summary for quick reading:

```
Model                    | SIGNOR* Acc | SIGNOR* FPR | SIGNOR* FNR | SIGNOR* Cost | CDB Acc | CDB FPR | CDB FNR | CDB Cost
gemini-3.1-pro-preview   |  X.XX       |  X.XX       |  X.XX       |  X.XX        | X.XX    | X.XX    | X.XX    | X.XX
claude-sonnet-4.6        |  X.XX       |  X.XX       |  X.XX       |  X.XX        | X.XX    | X.XX    | X.XX    | X.XX
```

## Notes

- If `$DATASETS` is not provided, default to `signor connectomedb`.
- Dataset CSVs must exist under `/home/ail/workspace/connectomeDB_data/datasets/`.
- `macro_fpr` / `macro_fnr` are macro-averaged across SUPPORT / REFUTE / NEI classes.
- Model slug directory names replace `/` with `--` (e.g. `anthropic/claude-sonnet-4-6` → `anthropic--claude-sonnet-4-6`).
- Ensure `ANTHROPIC_API_KEY` is set in `.env` for Claude and Vertex AI credentials (`VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, `VERTEX_CREDENTIALS`) are set for Gemini.
