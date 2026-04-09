# SIGNOR Dataset Update, Baseline Completion & Dynamic Label Configuration — 2026-04-09

**Branch:** `exp`

## Summary

Three major areas of work were completed. (1) The SIGNOR claim verification dataset was updated (101 claims: 34 SUPPORT, 63 REFUTE, 4 UNCERTAIN; 34 flipped variants) and all baseline evaluations were completed or re-run against it. (2) ACE and FIRE baselines were integrated into the evaluation harness with YAML configs and runner support. (3) The entire evidence programming system—prompts, parsers, renderers, and data models—was refactored so that stance labels (e.g. SUPPORT/REFUTE/NEUTRAL) and verdict labels (e.g. SUPPORT/REFUTE/UNCERTAIN) are fully user-configurable via YAML, replacing all hardcoded label definitions. (4) FIRE and ACE baseline prompts and label definitions were unified with the Evidence Programming agent: FIRE's original True/False/Uncertain verdicts were replaced with Support/Refute/Uncertain using the same definitions from `LabelConfig`, injected dynamically at module load. Cost tracking was added to both baselines using `CostTracker.DEFAULT_PRICING`, and existing result files were retroactively patched. FIRE was re-run on SIGNOR with the updated prompts.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/fire_baseline.py` | FIRE baseline adapter for the evaluation harness. Prompts changed from True/False/Uncertain to Support/Refute/Uncertain. Verdict label definitions dynamically injected from `LabelConfig` defaults at module load (f-string templates with `_VERDICT_OPTIONS` and `_VERDICT_DEFS`). Answer mapping uses `LabelConfig.validate_verdict()`. Cost computed from token counts via `CostTracker.DEFAULT_PRICING`. |
| `experiments/baselines/ace_baseline.py` | ACE baseline adapter for the evaluation harness. Cost computed from token counts via `CostTracker.DEFAULT_PRICING` (was hardcoded `0.0`). |
| `experiments/configs/ace_config.yaml` | YAML config for ACE baseline runs |
| `experiments/configs/fire_config.yaml` | YAML config for FIRE baseline runs |

## Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/config.py` | Added `LabelConfig` (pydantic-settings model) with `stance_labels`, `verdict_labels`, `default_stance` fields and prompt-builder helpers. Added `set_label_config()`/`get_label_config()` singleton. `VerificationSettings` now carries a `labels: LabelConfig` field. `build_sdk_env()` propagates labels via `LABEL_CONFIG_JSON` env var. |
| `src/pkevolve/verification/data_models.py` | Replaced static `Stance` enum with a dynamic factory (`_make_stance_enum()`). `Fact.stance` uses `Annotated[Any, BeforeValidator]` so Pydantic resolves the current enum at validation time—no `model_rebuild()` needed. Added `rebuild_stance_enum()` to swap the module-level enum at runtime. |
| `src/pkevolve/verification/subagents.py` | `extract_facts()` prompt now uses `label_cfg.stance_prompt_block()` and `stance_options_str()`. `_parse_facts_response()` validates via `label_cfg.validate_stance()`. `identify_gaps()` counts stances dynamically. |
| `src/pkevolve/verification/evidence_api.py` | `schema_docs()`, `add_facts_from_dicts()`, `get_evidence_summary()`, `filter_papers_by_stance()`, `_recompute_coverage()`, `search_semantic_scholar_recommendations()` all read label definitions from `get_label_config()` instead of hardcoding. `setup_kernel()` initializes label config from `LABEL_CONFIG_JSON`. |
| `src/pkevolve/verification/evidence_programming.py` | `SYSTEM_PROMPT` uses `{verdict_names}` and `{verdict_definitions}` placeholders filled from `LabelConfig`. |
| `src/pkevolve/verification/renderers.py` | Stance/verdict colors and icons generated dynamically from `get_label_config()` using colour cycles. Counting logic iterates over configured labels instead of hardcoded names. |
| `experiments/run_baselines_datasets.py` | Extended to support `ace` and `fire` baseline types alongside existing `random`, `llm_only`, `open_scholar`. |
| `experiments/configs/open_scholar_config.yaml` | Minor config update (max tokens default). |
| `experiments/configs/random_baseline_config.yaml` | Minor config update. |
| `results/baselines/open_scholar/claude-sonnet-4-6_08_04_2026/signor_seed100.jsonl` | Added 5 missing `_flip` claims (96 → 101 results). |
| `results/baselines/open_scholar/claude-sonnet-4-6_08_04_2026/signor_metrics.json` | Recomputed metrics on full 101-claim set. |
| `results/baselines/open_scholar/claude-sonnet-4-6_oracle/signor_seed100.jsonl` | Added 5 missing `_flip` claims (96 → 101 results). |
| `results/baselines/open_scholar/claude-sonnet-4-6_oracle/signor_metrics.json` | Recomputed metrics on full 101-claim set. |

## Architecture

```
YAML config (labels section)
  │
  ▼
LabelConfig (config.py)          ◄── pydantic-settings model
  │                                    stance_labels, verdict_labels,
  │                                    default_stance
  ├──► set_label_config()        ◄── module-level singleton
  │      │
  │      ├──► rebuild_stance_enum()   ◄── swaps module-level Stance enum
  │      │      in data_models.py          (dynamic str Enum factory)
  │      │
  │      └──► LABEL_CONFIG_JSON       ◄── env var propagated to SDK kernel
  │
  ├──► Subagent prompts          ◄── stance_prompt_block(), stance_options_str()
  │      (subagents.py)                injected into extract_facts / identify_gaps
  │
  ├──► System prompt             ◄── verdict_names(), verdict_prompt_block()
  │      (evidence_programming.py)     injected into SYSTEM_PROMPT
  │
  ├──► Evidence API              ◄── get_label_config() for schema_docs,
  │      (evidence_api.py)             add_facts, summary, filtering, coverage
  │
  └──► Renderers                 ◄── dynamic colour/icon cycles from label names
         (renderers.py)

Fact.stance: Annotated[Any, BeforeValidator]
  └──► reads dm.Stance at validation time (always current enum)
       ▸ no model_rebuild() needed after rebuild_stance_enum()
```

## Baseline Results (SIGNOR, 101 claims)

| Baseline | Accuracy | Macro F1 | W-FPR | W-FNR | Cost (USD) |
|----------|----------|----------|-------|-------|------------|
| Random (10 repeats) | 0.3287±0.0637 | 0.2733±0.0473 | — | — | 0.00 |
| OpenScholar S2-retrieval | 0.4059 | 0.3806 | 0.1233 | 0.5941 | 4.79 |
| OpenScholar Oracle | 0.6634 | 0.5138 | 0.1103 | 0.3366 | 1.88 |
| FIRE | 0.6436 | 0.5139 | 0.2139 | 0.3564 | 2.91 |
| ACE | 0.6733 | 0.5746 | 0.1052 | 0.3268 | 0.89 |

## Key Design Decisions

- **`Annotated[Any, BeforeValidator]` for dynamic enum**: Pydantic v2 caches compiled schemas at class creation, so `model_rebuild(force=True)` does not pick up a reassigned `Stance` type. The `BeforeValidator` reads `dm.Stance` at validation time (deferred lookup), so it always validates against the current enum without any model rebuild.
- **`str`-based Enum with `__str__`/`__format__` overrides**: `Stance` members inherit from `str` so `stance == "SUPPORT"` works. `__str__` and `__format__` return the value (not `"Stance.SUPPORT"`), making f-string interpolation and prompt construction clean.
- **Module-level singleton for `LabelConfig`**: `set_label_config()` / `get_label_config()` avoid passing config objects through every function signature. The singleton is initialised once during kernel setup.
- **Environment variable propagation**: `build_sdk_env()` serialises `LabelConfig` to JSON and passes it as `LABEL_CONFIG_JSON`. The SDK kernel's `setup_kernel()` deserialises it and calls `set_label_config()`.
- **Symlink resume pattern** (baselines): The baseline runner writes to `open_scholar/claude-sonnet-4-6/`. To resume into existing dated/oracle folders, a temporary symlink was created, then removed after completion.
- **Resume mechanism**: `run_baselines_datasets.py` loads completed claim IDs from the JSONL and skips them, only running missing claims.
- **Dynamic verdict injection in FIRE prompts**: FIRE's prompt templates are f-strings that inject `_VERDICT_OPTIONS` (e.g. `"Support" or "Refute" or "Uncertain"`) and `_VERDICT_DEFS` (full definitions) from `LabelConfig()` at module load. This ensures FIRE's LLM instructions stay in sync with the Evidence Programming agent's label taxonomy without manual duplication.
- **Cost tracking via `CostTracker.DEFAULT_PRICING`**: Both FIRE and ACE baselines compute `cost_usd` from token counts using the same pricing table as other baselines (`$3.00/$15.00 per 1M tokens for claude-sonnet-4`). Previously hardcoded to `0.0`.

## Bug Fixes

- **Pydantic v2 `model_rebuild` incompatibility**: `Fact.__annotations__["stance"] = NewStance; Fact.model_rebuild(force=True)` silently kept the old compiled schema. Replaced with `BeforeValidator` approach.
- **Sufficiency renderer case mismatch**: Original code checked `label != "INSUFFICIENT"` but `check_sufficiency()` returns lowercase `"insufficient"`. Fixed to `label.lower() == "sufficient"`.
- **`from enum import Enum` collision**: Renaming to `import enum as _enum` broke `GapType(str, Enum)` and `GapPriority(str, Enum)`. Fixed to `(str, _enum.Enum)`.
- **Missing `_flip` claims in OpenScholar results**: 5 `_flip` variants (SIGNOR-144163, 178679, 179390, 255657, 272078) were added to both S2-retrieval and oracle result sets using the resume mechanism.
- **Dangling symlink cleanup**: Temporary `claude-sonnet-4-6` symlinks removed after each baseline run.
- **FIRE/ACE `cost_usd` always `0.0`**: Both baselines tracked token counts but never computed USD cost. Fixed by importing `CostTracker.DEFAULT_PRICING` and computing `(input_tokens / 1M × in_price) + (output_tokens / 1M × out_price)`. Existing JSONL result files retroactively patched.
- **FIRE verdict label mismatch**: FIRE originally used True/False (from the original paper) which didn't match the Evidence Programming agent's SUPPORT/REFUTE/UNCERTAIN taxonomy. Updated prompts to use Support/Refute/Uncertain with full definitions from `LabelConfig`. FIRE re-run on SIGNOR with updated prompts (accuracy 0.7426 → 0.6436 after label change, reflecting the model's adaptation to the new prompt framing).
