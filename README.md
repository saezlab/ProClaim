# PKEvolve: Knowledge Graph Curation

PKEvolve is an AI tool for correcting Gene Regulatory Networks (GRNs) using Large Language Models (LLMs).

## Installation

```bash
uv sync
```

To enable full-text retrieval via INDRA (Layer 2):

```bash
uv sync --extra fulltext
```

> **Note:** Use `uv run python` — not `python` directly — to ensure the managed environment is active.

## Usage

```bash
# Run the main QA pipeline
uv run python scripts/qa_pipeline/run_qa.py --help

# Run analysis on results
uv run python scripts/analysis/analyze_qa_results.py --help

# Convert direct-eval baseline CSV output to baseline-style JSONL
uv run python scripts/analysis/convert_direct_eval_csv_to_jsonl.py --help

# Print baseline metrics tables and save a log under results/analysis/metrics
uv run python scripts/analysis/print_baseline_metrics_table.py --help
```

## Repository Structure

```
src/pkevolve/      # Core library (LLM evaluator, paper retrieval, GNN, verification)
scripts/           # CLI entry points (qa_pipeline/, analysis/, claude_sdk/, context_labeling/)
data/signor/       # Ground truth edges (true_positive_edges.csv, true_negative_edges.csv)
results/           # Output artefacts
experiments/       # Config files for eval runs
```

## Environment

Copy `.env.example` to `.env` and set:

```
OPENAI_API_KEY=...       # for cloud LLM access
UNPAYWALL_EMAIL=...      # optional, for full-text PDF retrieval
ELSEVIER_API_KEY=...     # optional, for Elsevier full-text via INDRA
```
