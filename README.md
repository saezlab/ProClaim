# ProClaim: In-the-Wild Scientific Claim Verification

ProClaim is a sufficiency-aware agentic verifier for open-domain scientific claim verification. It determines a claim's consensual stance from the existing literature through iterative evidence construction.

This repository is the supplementary code package for the NeurIPS submission. This README is the entry point for reviewers.

## What Is Included

- ProClaim-eval dataset (claims from SIGNOR and ConnectomeDB) under `datasets/`
- Core library code under `src/proclaim/`
- Evaluation entry points under `experiments/`
- Analysis and utility scripts under `scripts/`
- Configuration files under `configs/`
- Example outputs and cached artefacts under `results/` (unzip baselines.7z for baseline results)

Some experiment folders contain third-party baselines preserved in-tree for comparison. Their own licenses and README files remain in those subdirectories.

## Quick Start

1. Install dependencies:

	```bash
	uv sync
	```

2. Create an environment file:

	```bash
	cp .env.example .env
	```

3. Reproducibility demo

    The smallest runnable supplementary commands in this checkout are:

    ```bash
    # SIGNOR: run one claim from the bundled dataset
    uv run python experiments/run_signor_eval.py \
        --input-csv datasets/signor.csv \
        --config experiments/configs/signor_smoke_config.yaml \
        --limit 1 \
        --reps 1 \
        --run-tag reviewer_signor_smoke

    # ConnectomeDB: run one claim from the bundled dataset
    uv run python experiments/run_connectomedb_eval.py \
        --input-csv datasets/connectomedb.csv \
        --config experiments/configs/connectomedb_smoke_config.yaml \
        --limit 1 \
        --reps 1 \
        --run-tag reviewer_connectomedb_smoke

    # Inspect the precomputed baseline metrics already included under results/
    uv run python scripts/analysis/print_aggregated_baseline_metrics_table.py
    uv run python scripts/analysis/print_baseline_metrics_table.py
    ```

    These are minimal smoke tests, not full reruns. They use the included `datasets/` files, process a single row with `--limit 1`, and write namespaced outputs under `results/`. The smoke-test configs default to cloud-hosted Anthropic models, so no local vLLM endpoint is required.

    For full runs, use the dataset-specific direct configs in `experiments/configs/`: `signor_direct_config.yaml` for SIGNOR and `connectomedb_direct_config.yaml` for ConnectomeDB. Those configs are intended for the full evaluation workflow rather than the lightweight reviewer demo.

    If you want to use a local OpenAI-compatible endpoint for the subagent, keep the same evaluation commands but swap to the direct config and set `LLM_BASE_URL` plus the corresponding local `llm.subagent_model`.

## Repository Structure

```text
src/proclaim/      Core verifier, retrieval, and sufficiency logic
experiments/       Evaluation runners and baselines
scripts/           Analysis, batching, and helper scripts
configs/           Dataset- and baseline-specific YAML configs
results/           Output artefacts and analysis tables
```

## Environment Variables

Copy `.env.example` to `.env` and set the values required for your run:

```text
ANTHROPIC_API_KEY=...    # used by the smoke-test configs
OPENAI_API_KEY=...       # for cloud LLM access
UNPAYWALL_EMAIL=...      # optional, for full-text PDF retrieval
ELSEVIER_API_KEY=...     # optional, for Elsevier full text via INDRA
PUBMED_EMAIL=...         # optional, for Entrez usage etiquette
PUBMED_API_KEY=...       # optional, for higher Entrez rate limits
LLM_BASE_URL=...         # optional, only if using a local OpenAI-compatible subagent endpoint
```

The smoke-test configs run without local endpoints by default. For local subagent execution, set `LLM_BASE_URL` and use the dataset-specific direct config instead of the smoke config.

## Notes For Reviewers

- Commands are intended to be run from the repository root.
- Several evaluation scripts may require external APIs, model endpoints, or large cached artefacts to reproduce the full paper results.
- The included `results/` directory provides example outputs and summary tables for inspection without rerunning every experiment.
