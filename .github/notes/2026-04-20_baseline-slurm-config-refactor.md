# Baseline SLURM Config Refactor — 2026-04-20

## Branch

`exp`

## Summary

Refactored the baseline batch-run workflow toward config-driven execution. The SLURM launcher now submits jobs by pointing `run_baselines_datasets.py` at per-baseline YAML configs instead of hard-coding dataset and output paths inside the script, and it can attach extra SBATCH directives for jobs that need additional resources. Alongside that refactor, the baseline config files were normalized to the shared dataset location and output directory, a dedicated ReAct S2 config was added, the local non-SLURM runner was restored under `scripts/`, and two baseline implementations were hardened: ACE now uses shared verdict definitions and logs its active playbook, while OpenScholar recovers more truncated verdict labels.

## New Files

| File | Purpose |
|------|---------|
| `experiments/configs/react_s2_config.yaml` | Separate config for the ReAct baseline with the Semantic Scholar backend. |
| `scripts/run_all_baselines.sh` | Local shell runner for sequential baseline execution outside SLURM. |

## Modified Files

| File | Changes |
|------|---------|
| `scripts/run_all_baselines_slurm.sh` | Switched job submission to `--config`-based runs, added support for extra SBATCH lines, logged the config path in each job, and assigned GPU/CPU/memory overrides for OpenScholar. |
| `experiments/configs/ace_config.yaml` | Updated `datasets_dir` to the shared HPS dataset path and added `output_dir`. |
| `experiments/configs/fire_config.yaml` | Updated `datasets_dir` to the shared HPS dataset path and added `output_dir`. |
| `experiments/configs/open_scholar_config.yaml` | Made the shared HPS dataset path the active default and kept the previous local path commented as an alternate. |
| `experiments/configs/react_config.yaml` | Updated `datasets_dir` to the shared HPS dataset path and added `output_dir`. |
| `experiments/configs/retrieval_config.yaml` | Updated `datasets_dir`, pinned `temperature: 0.0`, and kept `output_dir` explicit. |
| `experiments/baselines/ace_baseline.py` | Replaced inline verdict descriptions with the shared `verdict_defs_block()` text and wrote the active playbook into per-claim logs for easier debugging. |
| `experiments/baselines/open_scholar_baseline.py` | Expanded truncated-JSON label recovery to accept `SUPPORTED` and `REFUTED` in addition to the canonical labels. |
| `.gitignore` | Added `dcgm/` to ignored paths. |
| `experiments/run_all_baselines.sh` | Removed the older runner from `experiments/` after relocating the script into `scripts/`. |

## Architecture

```text
scripts/run_all_baselines_slurm.sh
    |
    +-- selects baseline + dataset set
    +-- chooses per-baseline YAML config
    +-- injects optional SBATCH resources/env overrides
    v
sbatch job script
    |
    v
uv run python experiments/run_baselines_datasets.py --config <baseline-config>
    |
    +-- CLI flags override YAML defaults
    +-- EvaluationHarness loads claims and streams JSONL results
    v
results/baselines/<baseline>/<model-slug>/...

scripts/run_all_baselines.sh
    |
    v
local sequential execution of the same evaluation entry point
```

## Key Design Decisions

- Config-driven job submission reduces duplicated argument wiring between the SLURM launcher and individual baseline runs.
- A separate `react_s2_config.yaml` keeps the two ReAct retrieval modes explicit instead of overloading one config with CLI-only backend changes.
- OpenScholar is scheduled with explicit GPU resources because its reranker path is materially different from the CPU-only baselines.
- ACE now reuses the shared verdict-definition text so label semantics stay aligned across baselines.
- Writing the ACE playbook into logs preserves the exact prompt context used for each claim and makes later result audits reproducible.
- OpenScholar’s partial-JSON recovery accepts non-canonical label variants because truncation errors were dropping otherwise recoverable predictions.

## Bug Fixes

- Fixed OpenScholar truncated-output recovery so partial JSON containing `SUPPORTED` or `REFUTED` can still be mapped back to a verdict.
- Fixed ACE prompt drift by sourcing verdict definitions from the shared label utility instead of maintaining a duplicated inline description block.
- Fixed missing run-path consistency across baseline configs by pointing them at the shared dataset directory and explicit results directory used by the current cluster runs.