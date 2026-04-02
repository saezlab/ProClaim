# experiments/

Scripts and shared infrastructure for running and evaluating baselines against SIGNOR and other claim-verification datasets.

## Top-level scripts

| Script | Purpose |
|--------|---------|
| `run_baselines_signor.py` | Run a baseline on SIGNOR ground-truth edges, with optional edge flipping |
| `run_baselines_datasets.py` | Run a baseline on pre-processed CSV datasets (SIGNOR, ConnectomeDB, SciFact-Open, CIViC-Fact); prints a summary table of Macro F1 / FPR / FNR / Cost (mean ± std over repeats) |
| `run_signor_eval.py` | End-to-end SIGNOR evaluation using the full evidence-programming pipeline |
| `qwen_test.py` | Ad-hoc test of Qwen-3 fact extraction on paper text |
| `signor_eval_config.yaml` | Config file consumed by `run_signor_eval.py` (model, mode, iteration budget) |

### Quick start

```bash
# Random baseline on SIGNOR
uv run python experiments/run_baselines_signor.py --baseline random

# LLM-only baseline, with flipped edges, limited to 20 claims
uv run python experiments/run_baselines_signor.py --baseline llm_only --flip --limit 20 --model <model-name>

# Random baseline on CSV datasets (10 repeats, seed=100)
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /hps/nobackup/saezrodriguez/shared_datasets/claims/datasets \
    --datasets signor connectomedb \
    --baseline random

# LLM-only baseline on CSV datasets
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /hps/nobackup/saezrodriguez/shared_datasets/claims/datasets \
    --datasets signor connectomedb \
    --baseline llm_only \
    --output-dir results/baselines

# Full evidence-programming evaluation on SIGNOR
uv run python experiments/run_signor_eval.py \
    --config experiments/signor_eval_config.yaml \
    --output-csv results/signor_ep.csv
```

## baselines/

Installable baseline classes. All baselines share the same infrastructure from `baselines/shared/`.

| Module | Class | Description |
|--------|-------|-------------|
| `random_baseline.py` | `RandomBaseline` | Chance floor — uniform draw from {SUPPORT, REFUTE, NEI} |
| `llm_only.py` | `LLMOnly` | Parametric knowledge ceiling — single LLM call, no retrieval |

### shared/

| Module | Key export | Purpose |
|--------|-----------|---------|
| `llm.py` | `LLMBackend` | OpenAI-compatible client with retry logic and token tracking. Resolves API key from `OPENAI_API_KEY`. |
| `evaluate.py` | `EvaluationHarness` | Runs any baseline over a list of claims; computes accuracy, macro-F1, macro-FPR, macro-FNR, binary-F1, and per-class P/R/F1/FPR/FNR with raw TP/FP/FN/TN counts. |
| `verdict.py` | `Verdict`, `BaselineResult` | Pydantic schemas shared by all baselines. |
| `label_utils.py` | `normalize_label()` | Maps dataset-specific labels to canonical `{SUPPORT, REFUTE, NEI}`. |
| `cost_tracker.py` | `CostTracker` | Records LLM calls, token counts (exact, API-reported via `resp.usage`), and latency per claim. Cost in USD is a **proxy**: `(input_tokens / 1M × in_price) + (output_tokens / 1M × out_price)`. Prices are hardcoded in `DEFAULT_PRICING` (provider prefix stripped from model name before lookup; falls back to $3/$15 per 1M if model is unknown). Token counts are the durable ground truth saved in every `BaselineResult` — USD cost can be recomputed offline. |
| `prompts.py` | `VERIFICATION_SYSTEM_PROMPT`, … | Shared prompt templates — all baselines use the same verification prompt so only the evidence varies. |

## Adding a new baseline

1. Create `baselines/my_baseline.py` with a class that exposes `verify(claim_id, claim, gold_label) -> BaselineResult`.
2. Register it in the `build_baseline()` factory in `run_baselines_signor.py` and `run_baselines_datasets.py`.
3. Use `LLMBackend` for all LLM calls and attach a `CostTracker` instance to your class.
