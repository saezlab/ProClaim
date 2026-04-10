# SIGNOR Dataset Update, Baseline Completion & Centralised Label Configuration — 2026-04-09

**Branch:** `exp`

## Summary

Five areas of work were completed. (1) The SIGNOR claim verification dataset was updated (101 claims: 34 SUPPORT, 63 REFUTE, 4 UNCERTAIN; 34 flipped variants) and all baseline evaluations were completed or re-run against it. (2) ACE and FIRE baselines were integrated into the evaluation harness with YAML configs and runner support. (3) The entire evidence programming system—prompts, parsers, renderers, and data models—was refactored so that stance labels (e.g. SUPPORT/REFUTE/NEUTRAL) and verdict labels (e.g. SUPPORT/REFUTE/UNCERTAIN) are fully user-configurable via YAML, replacing all hardcoded label definitions. (4) FIRE and ACE baseline prompts and label definitions were unified with the Evidence Programming agent: FIRE's original True/False/Uncertain verdicts were replaced with SUPPORT/REFUTE/UNCERTAIN using the same definitions from `LabelConfig`. Cost tracking was added to both baselines using `CostTracker.DEFAULT_PRICING`, and existing result files were retroactively patched. FIRE was re-run on SIGNOR with the updated prompts. (5) Verdict label loading was centralised into `label_utils.py` — a single `LabelConfig` singleton and a set of helper functions (`verdict_names()`, `verdict_options_str()`, `verdict_or_str()`, `verdict_defs_block()`, `validate_verdict()`) now serve all baselines. Every baseline that previously instantiated its own `LabelConfig` or hardcoded label lists was updated to use these shared methods. `prompts.py` was refactored from static string constants to builder functions that read from the shared config. Labels are always uppercase (SUPPORT/REFUTE/UNCERTAIN) across all baselines.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/fire_baseline.py` | FIRE baseline adapter for the evaluation harness. Verdict label definitions dynamically injected from shared `label_utils` helpers at module load (f-string templates with `verdict_or_str()` and `verdict_defs_block()`). Answer mapping uses `validate_verdict()`. Cost computed from token counts via `CostTracker.DEFAULT_PRICING`. |
| `experiments/baselines/ace_baseline.py` | ACE baseline adapter for the evaluation harness. Playbook and question template use `verdict_names()` and `verdict_options_str()` from shared `label_utils`. Cost computed from token counts via `CostTracker.DEFAULT_PRICING`. |
| `experiments/baselines/s2_retrieval.py` | Fixed-k Semantic Scholar RAG baseline. Searches S2 with the raw claim, retrieves top-k abstracts, and classifies with a single LLM call. Uses shared prompt templates from `prompts.py`. |
| `experiments/baselines/react_baseline.py` | ReAct (Yao et al., ICLR 2023) agentic baseline. Thought → Action → Observation loop with web search tools. Tool schema `enum` and system prompt built from shared `label_utils` helpers (`verdict_names()`, `verdict_or_str()`, `verdict_defs_block()`). Uses native tool-use API via litellm. |
| `experiments/configs/ace_config.yaml` | YAML config for ACE baseline runs |
| `experiments/configs/fire_config.yaml` | YAML config for FIRE baseline runs |
| `experiments/configs/s2_retrieval_config.yaml` | YAML config for S2 retrieval baseline runs |
| `experiments/configs/react_config.yaml` | YAML config for ReAct baseline runs |

## Modified Files

| File | Change |
|------|--------|
| `experiments/baselines/shared/label_utils.py` | Expanded from a pure normalization module into the centralised label hub. Added `LabelConfig` singleton (`_LABEL_CONFIG`), `get_label_config()`, and verdict helpers: `verdict_names()`, `verdict_options_str()`, `verdict_or_str()`, `verdict_defs_block()`, `validate_verdict()`. All helpers always produce uppercase label names (SUPPORT/REFUTE/UNCERTAIN). |
| `experiments/baselines/shared/prompts.py` | Refactored from static string constants to builder functions: `build_verification_system_prompt(labels)` and `build_verification_system_prompt_no_retrieval(labels)`. Both read verdict definitions from `LabelConfig` (via `get_label_config()` singleton). Added `_no_retrieval_description()` to reframe retrieval-oriented definitions for the parametric-knowledge setting. Module-level constants (`VERIFICATION_SYSTEM_PROMPT`, `VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL`) preserved for backward compatibility via builder calls. |
| `experiments/baselines/random_baseline.py` | Replaced hardcoded `["SUPPORT", "REFUTE", "UNCERTAIN"]` with `verdict_names()` from shared `label_utils`. |
| `experiments/baselines/fire_baseline.py` | Removed private `LabelConfig()` instantiation and `_LABEL_CFG` / `_VERDICT_NAMES` module globals. Now imports `validate_verdict`, `verdict_defs_block`, `verdict_names`, `verdict_or_str` from shared `label_utils`. All verdict labels use consistent uppercase. |
| `experiments/baselines/react_baseline.py` | Removed private `LabelConfig()` instantiation and `_LABEL_CFG` module global. Now imports verdict helpers from shared `label_utils`. Tool schema `enum` uses `verdict_names()`, system prompt uses `verdict_or_str()` and `verdict_defs_block()`. Hardcoded label list in `_force_finish` fallback replaced with `_VERDICT_NAMES` derived from `verdict_names()`. |
| `experiments/baselines/ace_baseline.py` | Replaced hardcoded label strings in playbook (`_CLAIM_VERIFICATION_PLAYBOOK`) and question template (`_CLAIM_QUESTION_TEMPLATE`) with `verdict_names()` and `verdict_options_str()` from shared `label_utils`. |
| `experiments/run_baselines_datasets.py` | Extended to support `s2_retrieval` and `react` baseline types alongside existing `random`, `llm_only`, `open_scholar`, `fire`, `ace`. Added `--s2-model`, `--s2-top-k`, `--react-model`, `--react-max-steps` CLI flags. Model-slug subdirectory logic added for both new baselines. |
| `experiments/README.md` | Added S2 Retrieval and ReAct documentation: baseline table entries, config table entries, CLI flag tables, prerequisites sections. Added quick-start examples for `s2_retrieval_config.yaml` and `react_config.yaml`. |
| `src/pkevolve/verification/config.py` | Added `LabelConfig` (pydantic-settings model) with `stance_labels`, `verdict_labels`, `default_stance` fields and prompt-builder helpers. Added `set_label_config()`/`get_label_config()` singleton. `VerificationSettings` now carries a `labels: LabelConfig` field. `build_sdk_env()` propagates labels via `LABEL_CONFIG_JSON` env var. |
| `src/pkevolve/verification/data_models.py` | Replaced static `Stance` enum with a dynamic factory (`_make_stance_enum()`). `Fact.stance` uses `Annotated[Any, BeforeValidator]` so Pydantic resolves the current enum at validation time—no `model_rebuild()` needed. Added `rebuild_stance_enum()` to swap the module-level enum at runtime. |
| `src/pkevolve/verification/subagents.py` | `extract_facts()` prompt now uses `label_cfg.stance_prompt_block()` and `stance_options_str()`. `_parse_facts_response()` validates via `label_cfg.validate_stance()`. `identify_gaps()` counts stances dynamically. |
| `src/pkevolve/verification/evidence_api.py` | `schema_docs()`, `add_facts_from_dicts()`, `get_evidence_summary()`, `filter_papers_by_stance()`, `_recompute_coverage()`, `search_semantic_scholar_recommendations()` all read label definitions from `get_label_config()` instead of hardcoding. `setup_kernel()` initializes label config from `LABEL_CONFIG_JSON`. |
| `src/pkevolve/verification/evidence_programming.py` | `SYSTEM_PROMPT` uses `{verdict_names}` and `{verdict_definitions}` placeholders filled from `LabelConfig`. |
| `src/pkevolve/verification/renderers.py` | Stance/verdict colors and icons generated dynamically from `get_label_config()` using colour cycles. Counting logic iterates over configured labels instead of hardcoded names. |
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
  │
  ├──► set_label_config()        ◄── module-level singleton (evidence programming)
  │      │
  │      ├──► rebuild_stance_enum()   ◄── swaps module-level Stance enum
  │      │      in data_models.py          (dynamic str Enum factory)
  │      │
  │      └──► LABEL_CONFIG_JSON       ◄── env var propagated to SDK kernel
  │
  ├──► label_utils.py            ◄── single LabelConfig() singleton for baselines
  │      │                             get_label_config()
  │      │
  │      ├──► verdict_names()         → ["SUPPORT", "REFUTE", "UNCERTAIN"]
  │      ├──► verdict_options_str()   → '"SUPPORT" | "REFUTE" | "UNCERTAIN"'
  │      ├──► verdict_or_str()        → '"SUPPORT" or "REFUTE" or "UNCERTAIN"'
  │      ├──► verdict_defs_block()    → multi-line "- NAME — description" block
  │      ├──► validate_verdict()      → case-insensitive label validation
  │      └──► normalize_label()       → raw dataset label → canonical taxonomy
  │             │
  │             ├──► RandomBaseline       verdict_names() for random choice
  │             ├──► FIREBaseline         verdict_or_str(), verdict_defs_block(),
  │             │                         validate_verdict() in prompts + parsing
  │             ├──► ReActBaseline        verdict_names() in tool enum,
  │             │                         verdict_or_str(), verdict_defs_block()
  │             │                         in system prompt
  │             ├──► ACEBaseline          verdict_names() in playbook + question
  │             ├──► prompts.py           get_label_config() → builder functions
  │             │      ├──► LLMOnly       (VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL)
  │             │      └──► S2Retrieval   (VERIFICATION_SYSTEM_PROMPT)
  │             └──► EvaluationHarness    normalize_label() for gold/predicted
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
- **Centralised verdict helpers in `label_utils.py`**: A single `LabelConfig()` instance is shared across all baselines via `get_label_config()`. Helper functions (`verdict_names()`, `verdict_options_str()`, `verdict_or_str()`, `verdict_defs_block()`, `validate_verdict()`) delegate to this singleton. This eliminates duplicate `LabelConfig()` instantiations that existed in `fire_baseline.py`, `react_baseline.py`, and `prompts.py`, and replaces hardcoded label lists in `random_baseline.py` and `ace_baseline.py`.
- **Uppercase-only label formatting**: `verdict_or_str()` and `verdict_defs_block()` always produce uppercase labels (SUPPORT/REFUTE/UNCERTAIN). Earlier versions had `capitalize` parameters that title-cased names (e.g. "Support"), but this was inconsistent with the canonical taxonomy. Removed in favour of a single consistent casing.
- **`prompts.py` builder functions**: `build_verification_system_prompt(labels)` and `build_verification_system_prompt_no_retrieval(labels)` accept an optional `LabelConfig` override but default to the shared singleton. Module-level constants (`VERIFICATION_SYSTEM_PROMPT`, etc.) are preserved for backward compatibility by calling the builders at import time.
- **Symlink resume pattern** (baselines): The baseline runner writes to `open_scholar/claude-sonnet-4-6/`. To resume into existing dated/oracle folders, a temporary symlink was created, then removed after completion.
- **Resume mechanism**: `run_baselines_datasets.py` loads completed claim IDs from the JSONL and skips them, only running missing claims.
- **Cost tracking via `CostTracker.DEFAULT_PRICING`**: Both FIRE and ACE baselines compute `cost_usd` from token counts using the same pricing table as other baselines (`$3.00/$15.00 per 1M tokens for claude-sonnet-4`). Previously hardcoded to `0.0`.

## Bug Fixes

- **Pydantic v2 `model_rebuild` incompatibility**: `Fact.__annotations__["stance"] = NewStance; Fact.model_rebuild(force=True)` silently kept the old compiled schema. Replaced with `BeforeValidator` approach.
- **Sufficiency renderer case mismatch**: Original code checked `label != "INSUFFICIENT"` but `check_sufficiency()` returns lowercase `"insufficient"`. Fixed to `label.lower() == "sufficient"`.
- **`from enum import Enum` collision**: Renaming to `import enum as _enum` broke `GapType(str, Enum)` and `GapPriority(str, Enum)`. Fixed to `(str, _enum.Enum)`.
- **Missing `_flip` claims in OpenScholar results**: 5 `_flip` variants (SIGNOR-144163, 178679, 179390, 255657, 272078) were added to both S2-retrieval and oracle result sets using the resume mechanism.
- **Dangling symlink cleanup**: Temporary `claude-sonnet-4-6` symlinks removed after each baseline run.
- **FIRE/ACE `cost_usd` always `0.0`**: Both baselines tracked token counts but never computed USD cost. Fixed by importing `CostTracker.DEFAULT_PRICING` and computing `(input_tokens / 1M × in_price) + (output_tokens / 1M × out_price)`. Existing JSONL result files retroactively patched.
- **FIRE verdict label mismatch**: FIRE originally used True/False (from the original paper) which didn't match the Evidence Programming agent's SUPPORT/REFUTE/UNCERTAIN taxonomy. Updated prompts to use SUPPORT/REFUTE/UNCERTAIN with full definitions from `LabelConfig`. FIRE re-run on SIGNOR with updated prompts (accuracy 0.7426 → 0.6436 after label change, reflecting the model's adaptation to the new prompt framing).
- **Inconsistent verdict casing across baselines**: FIRE used title-cased labels ("Support"/"Refute"/"Uncertain") in prompts while other baselines used uppercase. Removed `capitalize`/`capitalize_names` parameters from `verdict_or_str()` and `verdict_defs_block()` so all baselines consistently use uppercase labels.
- **Duplicate `LabelConfig` instantiation**: `fire_baseline.py`, `react_baseline.py`, and `prompts.py` each created independent `LabelConfig()` instances. Consolidated to a single shared instance in `label_utils.py`.
