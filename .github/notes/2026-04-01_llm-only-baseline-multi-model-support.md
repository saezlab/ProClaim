# LLM-Only Baseline: Multi-Model Support — 2026-04-01

**Branch:** `exp`

## Summary

Extended the baseline evaluation pipeline to support the `llm_only` baseline in `run_baselines_datasets.py` (previously only `random` was wired up), and rewrote `LLMBackend` to use **LiteLLM** instead of the OpenAI client. LiteLLM handles provider routing, API key resolution, and endpoint selection automatically from the model name prefix — no `--base-url` flag needed.

## Modified Files

| File | Changes |
|------|---------|
| `experiments/run_baselines_datasets.py` | Added `llm_only` to `--baseline` choices; added `--model`, `--limit` args (removed `--base-url`); wired `LLMOnly(LLMBackend(...))` in `build_baseline()`; `llm_only` always uses 1 repeat (deterministic); `n_repeats` variable replaces hardcoded `args.repeats` in log lines |
| `experiments/run_baselines_signor.py` | Removed `--base-url` argparse flag; removed `base_url=` kwarg from `LLMBackend(...)` call; updated default model to `"zai/glm-4-plus"`; updated docstring examples |
| `experiments/baselines/shared/llm.py` | **Full rewrite**: replaced `openai.OpenAI` client with `litellm.completion()`; removed `_resolve_api_key()`, `_resolve_base_url()`, `base_url`/`api_key` parameters; added `load_dotenv()` at module level; LiteLLM reads API keys from env automatically based on the provider prefix in the model string; default model changed to `"zai/glm-4-plus"` |

## Architecture

```
run_baselines_datasets.py  ──► build_baseline("llm_only", args)
                                      │
                                      ▼
                              LLMBackend(model="zai/glm-4-plus")
                                      │
                               litellm.completion()
                                      │
                    ┌─────────────────┼─────────────────┐
                 zai/…             openai/…        anthropic/…
                    │                 │                 │
            ZAI_API_KEY        OPENAI_API_KEY   ANTHROPIC_API_KEY
             (from .env)          (from .env)      (from .env)
```

## Key Design Decisions

- **LiteLLM provider prefix in model name** (`zai/`, `openai/`, `anthropic/`) replaces manual endpoint + key resolution. No `--base-url` flag; no `_resolve_base_url()` helper needed.
- **`load_dotenv()` at module level** ensures `.env` keys are loaded before any LiteLLM call, regardless of how the script is invoked.
- **`llm_only` forces `n_repeats=1`** — the baseline is fully deterministic at `temperature=0`, so multiple repeats waste API budget.
- **`litellm.set_verbose = False`** suppresses per-call debug noise from LiteLLM's internal logging.
- `parse_json()` regex fallback remains for models that ignore `response_format=json_object` and return free-form text.

---

## Prompt alignment: baseline prompts aligned with evidence programming agent

Rewrote the system and user prompts in `experiments/baselines/shared/prompts.py` to match the label semantics and verification rigour of the `evidence_programming` agent.

### Changes

| Constant | Before | After |
|----------|--------|-------|
| `VERIFICATION_SYSTEM_PROMPT` | One-liner SUPPORT/REFUTE/NEI definitions; no grounding rules | Full three-way definitions (SUPPORT/REFUTE/UNCERTAIN) copied from agent; explicit rules: base verdict only on provided evidence, no-mention papers → REFUTE, cite PMIDs |
| `VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL` | One-liner definitions | Matching definitions; conservatism rule (prefer UNCERTAIN when knowledge incomplete); JSON-only response instruction |
| `VERIFICATION_USER_TEMPLATE` | "Based on the above evidence, classify the claim." | "Classify the claim based solely on the evidence passages above. Output JSON." |
| `LLM_ONLY_USER_TEMPLATE` | "Based on your scientific knowledge, classify the claim." | "Classify this claim based on your scientific knowledge. Output JSON." |

### Label change: NEI → UNCERTAIN

Labels changed from `NEI` to `UNCERTAIN` to match the evidence programming agent's vocabulary. This is **backward-compatible**: `normalize_label()` in `label_utils.py` already maps `UNCERTAIN` → `NEI` for metric computation.

### Scope

Prompts were kept **generic** (domain-agnostic scientific claim verification). A version with biomedical-specific rules was drafted but reverted at the user's request so the prompts apply equally to SIGNOR, ConnectomeDB, SciFact-Open, and CIViC-Fact datasets.

| File | Changes |
|------|---------|
| `experiments/baselines/shared/prompts.py` | Rewrote `VERIFICATION_SYSTEM_PROMPT`, `VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL`, `VERIFICATION_USER_TEMPLATE`, `LLM_ONLY_USER_TEMPLATE`; left `DECOMPOSITION_PROMPT` and `QUERY_GENERATION_PROMPT` unchanged |

---

## Documentation cleanup: Remove Z.AI / GLM provider references

Removed all Z.AI-specific endpoints and `GLM_API_KEY`/`ZAI_API_KEY` references from documentation and copilot instructions, replacing them with provider-agnostic patterns (`OPENAI_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, placeholder URLs).

| File | Changes |
|------|---------|
| `.github/copilot-instructions.md` | Rewrote LLM client pattern section: removed Z.AI endpoint URL and `GLM_API_KEY`/`ZAI_API_KEY`; replaced with generic `OPENAI_API_KEY` pattern; removed the Z.AI-specific "Important" note about `/api/openai` vs `/api/paas/v4/` |
| `README.md` | Updated `.env` example: replaced `GLM_API_KEY` with `OPENAI_API_KEY` |
| `experiments/README.md` | Replaced `glm-4-plus` example model with `<model-name>`; simplified `LLMBackend` API key description to just `OPENAI_API_KEY` |
| `doc/ClaudeAgentSDK/CLAUDE.md` | Left unchanged (reverted by user — timestamped tracker) |
| `doc/ClaudeAgentSDK/Claude_CLI_with_non_Anthropic_endpoints.md` | Removed `GLM/Z.ai` from provider lists; replaced hardcoded Z.AI env pattern in code example with env-var-driven pattern |
| `doc/baseline_implementation_plan.md` | Replaced Z.AI proxy references in model table and endpoint auto-resolution table; simplified authentication notes to `OPENAI_API_KEY` |
| `doc/evaluation_plan.md` | Replaced Z.AI API references in model tables and authentication notes |
| `doc/implementation_plan.md` | Replaced Z.AI base URL and `GLM_API_KEY` in LLM client code examples; replaced hardcoded `glm-4.6` model strings with `os.environ` lookups; updated env var export block |
| `doc/sdk_vs_repl_modes.md` | Removed `Z.AI` from endpoint examples, REPL endpoint description, CLI usage examples, and YAML config example; updated transparency table |
| `src/proclaim/verification/README.md` | Updated settings table (`agent_base_url` default, `api.openai_api_key`); updated environment variables table to use `OPENAI_API_KEY` |

---

## 2026-04-02: End-to-end model validation

Smoke-tested `run_baselines_datasets.py` (1 SIGNOR claim, `llm_only` baseline) against two new providers.

### Claude Sonnet 4.6 — Anthropic

```bash
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /hps/nobackup/saezrodriguez/shared_datasets/claims/datasets \
    --datasets signor --baseline llm_only \
    --model anthropic/claude-sonnet-4-6 \
    --limit 1 --output-dir results/baselines
```

| Claim | Gold | Prediction | Accuracy | Binary F1 | Cost |
|-------|------|------------|----------|-----------|------|
| SIGNOR-156958 | SUPPORT | SUPPORT | 1.0 | 1.0 | $0.003 |

Call time ~4–6 s. No issues.

### Gemini 3.1 Pro Preview — Vertex AI

```bash
GOOGLE_APPLICATION_CREDENTIALS=prj-int-dev-saez-ai-pkc-734bae1cf581.json \
VERTEXAI_LOCATION=global \
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /hps/nobackup/saezrodriguez/shared_datasets/claims/datasets \
    --datasets signor --baseline llm_only \
    --model vertex_ai/gemini-3.1-pro-preview \
    --limit 1 --output-dir results/baselines
```

| Claim | Gold | Prediction | Accuracy | Binary F1 | Cost |
|-------|------|------------|----------|-----------|------|
| SIGNOR-156958 | SUPPORT | SUPPORT | 1.0 | 1.0 | $0.004 |

Call time ~5 s. No issues.

**Important:** `VERTEXAI_LOCATION=us-central1` returns HTTP 404 for `gemini-3.1-pro-preview`. Must use `VERTEXAI_LOCATION=global`.

**LiteLLM warning:** setting `temperature < 1.0` with Gemini 3 models may cause infinite loops or degraded reasoning. Consider passing `temperature=1.0` for production runs.

---

## 2026-04-02: SLURM batch submission script

Created `scripts/run_llm_baselines_slurm.sh` to submit LLM-only baseline jobs (Claude Sonnet 4.6 + Gemini 3.1 Pro Preview) on SIGNOR and ConnectomeDB datasets via SLURM.

### Usage

```bash
bash scripts/run_llm_baselines_slurm.sh             # all claims, both models
bash scripts/run_llm_baselines_slurm.sh --limit 5   # 5 claims per dataset
```

### Design

- Generates a `.sbatch` temp file per job (avoids heredoc escaping pitfalls), then submits with `sbatch --parsable`
- SLURM resources: 2 CPUs, 8 GB RAM, 4 h wall-time (no GPU needed — API-only workload)
- Sources `.env` for API keys; Gemini job additionally exports `GOOGLE_APPLICATION_CREDENTIALS` and `VERTEXAI_LOCATION=global`
- `--limit 0` (default) passes through to `run_baselines_datasets.py` meaning "all claims"

### Bugs fixed during development

1. **Trailing-backslash + empty `${LIMIT_ARG}`** — original heredoc used `\\` line continuations; when `LIMIT_ARG` was empty, bash passed a whitespace-only argument → `argparse: error: unrecognized arguments:`. Fixed by always passing `--limit ${LIMIT}` (0 = all).
2. **`#SBATCH` directives ignored** — `set -euo pipefail` preceded `#SBATCH` lines in the generated script. SLURM stops parsing directives at the first non-comment executable line → `--time` was never seen → `error: You must specify a time limit`. Fixed by rewriting the generator to emit `#SBATCH` directives immediately after `#!/bin/bash`, with `set -euo pipefail` after.

### Smoke test (--limit 1)

Both models ran successfully on 1 claim from each dataset (SLURM jobs 7797828, 7797829):

| Model | Dataset | Gold | Pred | Accuracy | Macro F1 | Cost |
|-------|---------|------|------|----------|----------|------|
| Claude Sonnet 4.6 | SIGNOR | SUPPORT | SUPPORT | 1.0 | 0.333 | $0.002 |
| Claude Sonnet 4.6 | ConnectomeDB | SUPPORT | SUPPORT | 1.0 | 0.333 | $0.003 |
| Gemini 3.1 Pro | SIGNOR | SUPPORT | SUPPORT | 1.0 | 0.333 | $0.003 |
| Gemini 3.1 Pro | ConnectomeDB | SUPPORT | SUPPORT | 1.0 | 0.333 | $0.012 |

Full runs submitted as jobs 7798088 (Claude) and 7798090 (Gemini).

| File | Changes |
|------|---------|
| `scripts/run_llm_baselines_slurm.sh` | New file — SLURM batch submission for LLM-only baselines |
