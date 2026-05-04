# SAFE Baseline And Baseline Runner Updates — 2026-04-29

**Branch**

`exp`

**Summary**

Added a new SAFE baseline to the dataset evaluation pipeline by adapting the SAFE claim-rating loop to the repo's atomic scientific claim setting. The implementation registers `safe` in the shared baseline factory, adds a default YAML config, wires SAFE into the local and SLURM batch runners, and documents the new baseline in the experiments README. The same change set also tightened retrieval query preprocessing, restored explicit ReAct web config naming, serialized Semantic Scholar jobs in the SLURM launcher to reduce rate-limit contention, and temporarily added an S2-backed FIRE variant that was later removed when the upstream wrappers became web-only.

**New Files**

| File | Purpose |
|------|---------|
| `experiments/baselines/safe_baseline.py` | Implements a SAFE-style iterative search baseline that proposes queries, accumulates evidence, and emits `SUPPORT` / `REFUTE` / `UNCERTAIN` for each claim. |
| `experiments/configs/safe_config.yaml` | Default YAML config for running the SAFE baseline with web search, `top_k=3`, and `max_steps=5`. |
| `experiments/configs/react_web_config.yaml` | Dedicated config for the ReAct web-search variant after splitting web and S2 execution paths in the SLURM runner. |

**Modified Files**

| File | Changes |
|------|---------|
| `experiments/run_baselines_datasets.py` | Registered the `safe` baseline in `build_baseline()`, extended CLI choices and help text, and initially added `safe/<backend>/<model>` output routing. A later cleanup removed the backend segment because SAFE is web-only. |
| `experiments/baselines/fire_baseline.py` | Added configurable search backend support for an early FIRE experiment. A later upstream-wrapper refactor removed that configurability and fixed FIRE to the shared web-search helper. |
| `experiments/README.md` | Documented SAFE in the baseline list, config table, shared flags, prerequisites, and batch-run examples. |
| `scripts/run_all_baselines.sh` | Added SAFE to the sequential baseline launcher with the default Claude + web-search configuration, and added a second FIRE entry that runs against Semantic Scholar. |
| `scripts/run_all_baselines_slurm.sh` | Added SAFE job submission, introduced dependency-aware `submit_s2_job()`, tracked submitted job IDs explicitly, switched the ReAct web job to `react_web_config.yaml`, and added a serialized `fire_s2` job target. |
| `experiments/baselines/retrieval_baseline.py` | Hardened SIGNOR and ConnectomeDB query preprocessing with stricter regex parsing and explicit fallback logging when claims do not match the expected templates. |
| `experiments/configs/react_config.yaml` | Removed in favor of `react_web_config.yaml` so the SLURM launcher can distinguish the web-backed ReAct config from the S2-backed variant. |

**Architecture**

```text
datasets/*.csv
    |
    v
run_baselines_datasets.py
    |
    +--> build_baseline("safe")
            |
            v
      SAFEBaseline / FIREBaseline
            |
            +--> LLMBackend.complete_text()  -> next search query
            |
            +--> search backend
            |      '- web -> DuckDuckGo (ddgs)
            |
            '--> LLMBackend.complete_text()  -> final verdict
                       |
                       v
                 BaselineResult JSONL

Batch entrypoints:
  scripts/run_all_baselines.sh
  scripts/run_all_baselines_slurm.sh
```

**Key Design Decisions**

- Implemented only the SAFE rating loop, not the full long-form decomposition pipeline, because the benchmark datasets already provide one atomic claim per example.
- Mapped SAFE's original binary judgment into the repo's canonical `SUPPORT` / `REFUTE` / `UNCERTAIN` taxonomy so metrics remain comparable across all baselines.
- Reused `LLMBackend` and `BaselineResult` instead of introducing a separate SAFE-specific runtime, keeping cost accounting, output schema, and model selection consistent with the rest of the baseline framework.
- Serialized S2-consuming SLURM jobs with `afterok` dependencies to reduce concurrent Semantic Scholar traffic from retrieval, ReAct-S2, and OpenScholar jobs.

**Bug Fixes**

- Fixed an output-path issue by giving SAFE its own results subdirectory instead of reusing the ReAct path pattern.
- Restored a concrete ReAct web config path in the SLURM runner by switching from the removed `react_config.yaml` to the new `react_web_config.yaml`.
- Improved retrieval query extraction for SIGNOR and ConnectomeDB claims to avoid malformed stripped queries and to log non-matching claim formats instead of silently over-stripping.
- Reduced likely Semantic Scholar rate-limit races in the SLURM orchestration by introducing serialized submission for S2-backed jobs.
- Removed the temporary FIRE+S2 experiment path once the upstream-wrapper integration fixed FIRE to the shared web-search helper.