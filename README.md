# ProClaim: In-the-Wild Scientific Claim Verification

ProClaim is a sufficiency-aware agentic verifier for open-domain scientific claim verification. It determines a claim's consensual stance from the existing literature through iterative evidence construction.

![ProClaim Overview](figs/graphic_overview-proclaim.png)

## What Is Included

- ProClaim-eval dataset (claims from SIGNOR and ConnectomeDB) under `datasets/`
- Core library code under `src/proclaim/`
- Evaluation entry points under `experiments/`
- Analysis and utility scripts under `scripts/`
- Configuration files under `configs/`
- Example outputs and cached artefacts under `results/` (unzip baselines.7z for baseline results)

Some experiment folders contain third-party baselines preserved in-tree for comparison. Their own licenses and README files remain in those subdirectories.

## Quick Start

This section is a short vignette for using ProClaim on one claim from the command line. 

All commands below are meant to be pasted from the repository root. 

1. Run the one-step setup script:

   ```bash
   bash scripts/setup_proclaim.sh
   ```

   This single script installs `uv` if needed, runs `uv sync`, creates `.env` from `.env.example` when it is missing, and builds the dedicated `.venv310` NER environment used by the biomedical feature pipeline.

2. Open `.env` and add your API key:

   ```text
   ANTHROPIC_API_KEY=your_key_here
   ```

   The default vignette config uses Anthropic's hosted API. You do not need a local Qwen or vLLM endpoint for this path. `OPENAI_API_KEY` is also supported by ProClaim, but only for configs that use an OpenAI-compatible model or proxy.

   The bundled smoke config is set to Anthropic's frontier model `anthropic/claude-opus-4-7` for the outer verification agent so the README demo prioritizes best single-claim performance. If you want a cheaper or faster run instead, switch to the commented `anthropic/claude-sonnet-4-6` line in the config.

3. Run the bundled example claim:

   ```bash
   bash scripts/run_claim_example.sh
   ```

   By default, this runs ProClaim on the example claim `GNAS directly activates ADCY1.` with `experiments/configs/smoke_config.yaml` and writes the output notebook, logs, and workspace files under `results/claim_example/`. That config keeps the stronger frontier model on the main agent path and a smaller subagent model for the search-and-tool loop.

4. Replace the claim directly from the command line:

   ```bash
   bash scripts/run_claim_example.sh --claim "<your claim here>"
   ```

   You can also choose a different output folder or config file:

   ```bash
   bash scripts/run_claim_example.sh \
       --claim "<your claim here>" \
       --config experiments/configs/my_config.yaml \
       --output-dir results/my_connectomedb_claim
   ```

Useful inspection commands:

```bash
bash scripts/setup_proclaim.sh --help
bash scripts/run_claim_example.sh --help
```

## Reproducibility

Benchmark reruns, dataset smoke tests, and baseline analysis commands now live in [experiments/README.md](experiments/README.md).

## Repository Structure

```text
src/proclaim/      Core verifier, retrieval, and sufficiency logic
experiments/       Evaluation runners and baselines
scripts/           Analysis, batching, and helper scripts
configs/           Dataset- and baseline-specific YAML configs
results/           Output artefacts and analysis tables
```

## Environment Variables

The setup script creates `.env` automatically when it is missing. These are the main variables you may want to set:

```text
ANTHROPIC_API_KEY=...    # required for the README vignette and smoke-test configs
OPENAI_API_KEY=...       # optional, for configs that use OpenAI-compatible models or proxies
S2_API_KEY=...           # optional, for higher-throughput Semantic Scholar retrieval in some baseline runs
LLM_BASE_URL=...         # optional, only if you switch to a local OpenAI-compatible subagent endpoint
```

The default vignette works without a local endpoint. For local subagent execution, set `LLM_BASE_URL` and point `bash scripts/run_claim_example.sh --config ...` at a config whose `llm.subagent_model` targets your local model.

## Notes

- Commands are intended to be run from the repository root.
- Several evaluation scripts may require external APIs, model endpoints, or large cached artefacts to reproduce the full paper results.
- The included `results/` directory provides example outputs and summary tables for inspection without rerunning every experiment.
