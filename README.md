# ProClaim: In-the-Wild Scientific Claim Verification

## Motivation: recursive knowledge construction

Many scientific agents built on large language models run a common loop - they retrieve evidence, build a record of what they have found, judge whether that record is sufficient for the task, and,
if not, use the record to direct the next retrieval. Each round's record is the next round's input, so the process is recursive. We call it *recursive knowledge construction*. 

![Recursive knowledge construction](figs/graphic_overview-recursive_knowledge_construction.png)

Two design axes decide how well this works, and both are usually assumed rather than measured, because such agents are evaluated only end to end:

1. **What the recursion builds**: the evidence state.
2. **What controls it**: the judge that decides when to stop.

ProClaim is a minimal instrument that makes both axes explicit, so each can be varied with the other held fixed. It exercises the recursive knowledge construction loop in *consensus* claim verification: establishing a claim's stance from the open literature rather than from a single given passage. ProClaim-eval (`datasets/`) provides 718 such claims with `SUPPORT` / `REFUTE` gold labels from five expert-curated databases.

## ProClaim

![ProClaim overview](figs/graphic_overview-proclaim.png)

Supplementary code and data for the submission. ProClaim is a sufficiency-aware agentic verifier
for open-domain scientific claim verification. It determines a claim's consensual stance from the
existing literature through iterative evidence construction.

## What is included

- `datasets/`: ProClaim-eval, five claim subsets (SIGNOR, ConnectomeDB, ExTRI, NUBASE, NASA
  Exoplanet Archive), 718 claims, packed in the password-protected `datasets/proclaim_eval.zip`.
  See [datasets/README.md](datasets/README.md).
- `src/proclaim/`: the ProClaim verifier (evidence state, paper search, fact extraction,
  sufficiency judge).
- `experiments/`: evaluation entry points, baselines and the YAML config of every reported run.
  See [experiments/README.md](experiments/README.md).
- `scripts/sufficiency_classifier/`: training data generation, feature extraction and training of
  the sufficiency judge, including the τ ablation.
- `results/ablation/models/tau_0.50_seed_42/`: weights of the trained sufficiency judge.

## Quick start

All commands run from the repository root.

1. Set up the environments (installs `uv` if needed, runs `uv sync`, creates `.env` from
   `.env.example`, builds the `.venv310` NER environment used by the feature pipeline, and
   unpacks the dataset CSVs into `datasets/`; `bash scripts/unpack_datasets.sh` unpacks them alone):

   ```bash
   bash scripts/setup_proclaim.sh
   ```

2. Add `ANTHROPIC_API_KEY` to `.env`. The example config uses hosted Claude models only and needs
   no local model endpoint.

3. Verify one claim:

   ```bash
   bash scripts/run_claim_example.sh --claim "GNAS directly activates ADCY1."
   ```

   This uses `experiments/configs/smoke_config.yaml` and writes the notebook, logs and evidence
   state under `results/claim_example/`. `--config` and `--output-dir` select another config or
   output folder.

## Environment variables

See `.env.example`. The paper's runs use Claude Sonnet 4.6 as the outer agent (Anthropic API or
Vertex AI) and a local Qwen3.5-9B served by vLLM at `LLM_BASE_URL` for the extraction subagent.

## Repository structure

```text
datasets/          ProClaim-eval archive (CSVs and the 40-claim samples)
src/proclaim/      Verifier, retrieval and sufficiency logic
experiments/       Evaluation runners, baselines, configs
scripts/           Setup, single-claim example, sufficiency-judge training
results/           Trained sufficiency-judge weights
figs/              README figures
```
