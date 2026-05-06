# ProClaim: In-the-Wild Scientific Claim Verification

ProClaim is a sufficiency-aware agentic verifier for open-domain scientific claim verification. It determines a claim's consensual stance from the existing literature through iterative evidence construction.

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
# Run evaluation
uv run python experiments/run_signor_eval.py --help

# Print baseline metrics tables
uv run python scripts/analysis/print_baseline_metrics_table.py --help
```

## Repository Structure

```
src/proclaim/      # Core library (verifier, retrieval, sufficiency classifier)
scripts/           # Analysis and utility scripts
results/           # Output artefacts
experiments/       # Evaluation runs and baselines
```

## Environment

Copy `.env.example` to `.env` and set:

```
OPENAI_API_KEY=...       # for cloud LLM access
UNPAYWALL_EMAIL=...      # optional, for full-text PDF retrieval
ELSEVIER_API_KEY=...     # optional, for Elsevier full-text via INDRA
```
