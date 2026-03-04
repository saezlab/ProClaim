# Centralised pydantic-settings Configuration — 2026-03-04

**Branch:** `RLM`

## Summary

Replaced ad-hoc `argparse` + `os.getenv()` configuration scattered across the verification subsystem with a centralised `pydantic-settings` module (`config.py`). Three layered settings models (`APISettings`, `LLMSettings`, `VerificationSettings`) unify environment variables, `.env` files, YAML config files, and CLI flags with a clear priority chain. The demo script (`demo_evidence_programming.py`) was refactored to use a single `VerificationSettings` object, eliminating ~80 lines of manual argument parsing and env-var resolution. YAML config support enables reproducible experiments — `save_yaml()` snapshots the full config (minus secrets and the per-run claim) for archival. SDK vs REPL mode differences were documented for academic publication guidance.

## New Files

| File | Purpose |
|------|---------|
| `src/pkevolve/verification/config.py` | Centralised pydantic-settings config: `APISettings` (API keys, Unpaywall email), `LLMSettings` (model, endpoints, temperature), `VerificationSettings` (top-level with nested sub-models). Factory methods: `from_cli()`, `from_yaml()`, `save_yaml()`, `build_sdk_env()`. Module-level `get_settings()` singleton. |
| `experiments/example_config.yaml` | Example YAML config for evidence verification experiments. Defines model, thresholds, endpoints — `claim` is intentionally absent (CLI-only). |
| `doc/sdk_vs_repl_modes.md` | Comprehensive documentation of SDK (Mode A) vs REPL (Mode B) architecture, transparency analysis, open-source status of Claude Agent SDK, and academic reproducibility recommendations. |
| `scripts/start_vllm_ihpc.sh` | One-command vLLM deployment on EBI HPC via SLURM. Handles job submission, compute node allocation, SSH port forwarding, and server lifecycle (`--start`, `--stop`, `--reconnect`, `--status`, `--logs`). |
| `scripts/vllm_node_setup.sh` | Compute-node companion script for `start_vllm_ihpc.sh`. Detects GPUs, finds an available port, writes connection info to shared filesystem, launches vLLM via Singularity. |

## Modified Files

| File | Changes |
|------|---------|
| `scripts/verification/demo_evidence_programming.py` | Replaced ~80 lines of manual `argparse` + `_build_env()` + env-var resolution with `VerificationSettings.from_cli()`. Both `verify_claim_notebook()` and `verify_claim_repl_mode()` now take a single `cfg` parameter. Inner `llm()` callable uses `subagent_model` instead of `model`. |
| `src/pkevolve/verification/repl_orchestrator.py` | Added optional `cfg: VerificationSettings` kwarg to `verify_claim_repl()`. When provided, cfg values override individual kwargs. Added `subagent_model` parameter. API key fallback uses `get_settings().api_key`. |
| `src/pkevolve/verification/orchestrator.py` | Added optional `cfg: VerificationSettings` kwarg to `verify_claim()`. Replaced inline `_build_env()` with `cfg.build_sdk_env()`. |
| `src/pkevolve/verification/full_text.py` | `_get_unpaywall_email()` now reads from `get_settings().api.unpaywall_email` with env-var fallback, instead of a hardcoded `os.getenv()`. |
| `src/pkevolve/verification/__init__.py` | Added lazy imports and `__all__` entries for `VerificationSettings`, `APISettings`, `LLMSettings`, `get_settings`. |
| `pyproject.toml` | Added `pydantic-settings>=2.0` and `pyyaml>=6.0` to dependencies. Added `[project.optional-dependencies] fulltext` section. |
| `doc/ClaudeAgentSDK/CLAUDE.md` | Added cross-reference to `doc/sdk_vs_repl_modes.md`. Updated model names table with both Anthropic and OpenAI-compatible endpoint columns. |
| `.github/copilot-instructions.md` | Updated LLM client pattern with correct Z.AI endpoints. Added `ZAI_API_KEY` fallback. Added verification subsystem module listing. |

## Architecture

```
CLI flags  ──►  ┌─────────────────────────────────────┐
                 │     VerificationSettings.from_cli()  │
                 │                                      │
YAML file  ──►  │  priority:                           │
(--config)       │    1. CLI flags                      │
                 │    2. Environment variables / .env    │
                 │    3. YAML config file                │
                 │    4. Field defaults                  │
.env file  ──►  │                                      │
                 └──────────────┬──────────────────────┘
                                │ cfg: VerificationSettings
                                │
          ┌─────────────────────┼──────────────────────────┐
          │                     │                           │
          ▼                     ▼                           ▼
   cfg.build_sdk_env()    cfg.model              cfg.save_yaml()
   → Mode A (SDK)         cfg.api_key            → reproducibility
                          cfg.openai_base_url       snapshot
                          → Mode B (REPL)
```

Settings model hierarchy:

```
VerificationSettings
├── api: APISettings
│   ├── glm_api_key        (from GLM_API_KEY env)
│   ├── zai_api_key        (from ZAI_API_KEY env)
│   ├── openai_api_key     (from OPENAI_API_KEY env)
│   ├── unpaywall_email
│   └── elsevier_api_key
├── llm: LLMSettings
│   ├── model              (outer agent model)
│   ├── subagent_model     (inner calls; defaults to model)
│   ├── openai_base_url
│   ├── anthropic_base_url
│   └── temperature
├── claim                  (CLI-only, excluded from YAML)
├── mode                   (sdk | repl)
├── max_iterations
├── sufficiency_threshold
├── max_turns
├── output_dir
├── notebook_path
└── verbose
```

## Key Design Decisions

- **pydantic-settings over Hydra/OmegaConf:** The project already uses Pydantic heavily (`data_models.py`). pydantic-settings adds env-var and `.env` merging natively without new concepts. Hydra's decorator-based approach would clash with the existing `argparse` CLI convention.
- **Three-model hierarchy (API / LLM / Verification):** Separates concerns — API keys are never mixed with model params, and both are isolated from workflow settings. Each sub-model can be tested independently.
- **`claim` excluded from YAML:** Claims vary per run (batch experiments iterate over many edges). Storing `claim` in YAML risks accidental reuse across runs. `from_yaml()` warns and strips it; `save_yaml()` omits it; `from_cli()` makes `--claim` required.
- **Backward-compatible `cfg` kwarg:** `verify_claim_repl()` and `verify_claim()` accept an optional `cfg` parameter. Old callers passing explicit kwargs still work unchanged.
- **`build_sdk_env()` centralises SDK env construction:** The 8-line `env` dict (with `ANTHROPIC_AUTH_TOKEN`, beta disabling, timeout) was duplicated in 3 files. Now it's a single method on `VerificationSettings`.
- **`get_settings()` singleton for library code:** Functions like `_get_unpaywall_email()` in `full_text.py` need config but aren't passed a `cfg` object. The `@lru_cache` singleton reads from env vars on first access.
- **YAML via built-in pydantic-settings support:** pydantic-settings 2.12+ includes `YamlConfigSettingsSource`. PyYAML was already installed (transitive dependency). `from_yaml()` and `save_yaml()` are thin wrappers.

## Bug Fixes

- **Subagent model mismatch:** The kernel prelude's `llm()` function was hardcoded to use the outer `model` (e.g. `glm-5`) for inner subagent calls. Added `subagent_model` parameter so the outer agent and inner fact-extraction calls can use different models (e.g. `glm-5` outer, `glm-4.6` inner).
- **Duplicated `_build_env()` in 3 files:** `demo_evidence_programming.py`, `orchestrator.py`, and inline in `repl_orchestrator.py` each had their own copy of the SDK environment construction logic with slight divergences. Consolidated into `VerificationSettings.build_sdk_env()`.
