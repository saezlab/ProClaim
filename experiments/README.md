# experiments/

Scripts and shared infrastructure for running and evaluating baselines against SIGNOR and other claim-verification datasets.

## Top-level scripts

| Script | Purpose |
|--------|---------|
| `run_baselines_datasets.py` | Run a baseline on pre-processed CSV datasets (SIGNOR, ConnectomeDB, SciFact-Open, CIViC-Fact); prints a summary table of Macro F1 / FPR / FNR / Cost (mean ± std over repeats) |
| `run_signor_eval.py` | End-to-end SIGNOR evaluation using the full evidence-programming pipeline |
| `qwen_test.py` | Ad-hoc test of Qwen-3 fact extraction on paper text |
| `signor_eval_config.yaml` | Config file consumed by `run_signor_eval.py` (model, mode, iteration budget) |

### Quick start

```bash
# Random baseline on CSV datasets (10 repeats, seed=100)
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /path_to/claim_datasets/datasets \
    --datasets signor connectomedb \
    --baseline random

# LLM-only baseline on CSV datasets
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /path_to/claim_datasets/datasets \
    --datasets signor connectomedb \
    --baseline llm_only \
    --output-dir results/baselines

# OpenScholar baseline on SIGNOR (1 claim for smoke test)
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /path_to/claim_datasets/datasets \
    --datasets signor \
    --baseline open_scholar \
    --limit 1 \
    --output-dir results/baselines

# OpenScholar baseline on SIGNOR — full run
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /path_to/claim_datasets/datasets \
    --datasets signor \
    --baseline open_scholar \
    --output-dir results/baselines

# OpenScholar with live S2 retrieval (requires network + S2 API key or rate-limited anonymous access)
uv run python experiments/run_baselines_datasets.py \
    --datasets-dir /path_to/claim_datasets/datasets \
    --datasets signor \
    --baseline open_scholar \
    --os-retrieval \
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
| `open_scholar_baseline.py` | `OpenScholarBaseline` | RAG baseline — routes each claim through the [OpenScholar](../../OpenScholar) pipeline (Claude Sonnet 4.6 by default). When the dataset CSV provides an `evidence` snippet (e.g. SIGNOR, ConnectomeDB), it is forwarded as a retrieved passage alongside any live S2-retrieved passages. With `--os-retrieval`, OpenScholar runs its full feedback loop (keyword extraction → S2 search → re-rank → answer edit). Outputs `claim_verdict` JSON (SUPPORT / REFUTE / UNCERTAIN → normalised to SUPPORT / REFUTE / NEI). Results saved as `BaselineResult` JSONL, identical in schema to `llm_only`. |

### OpenScholar-specific CLI flags (for `run_baselines_datasets.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--os-model` | `claude-sonnet-4-6` | Model name passed to OpenScholar's `--model_name` |
| `--os-api` | `anthropic` | API provider (e.g. `anthropic`, `gemini`) |
| `--os-top-n` | `5` | Number of context passages forwarded to the generator |
| `--os-max-tokens` | `0` | Max generation tokens (0 = omit flag, OpenScholar default 3000 applies) |
| `--os-retrieval` | off | Enable S2 adaptive retrieval + self-feedback loop (`--ss_retriever --feedback`) |

### Prerequisites for `OpenScholarBaseline`

- `ANTHROPIC_API_KEY` must be set (or present in `grn-llm-correct/.env`).
- `S2_API_KEY` defaults to `""` (anonymous, rate-limited). Set it for higher throughput.
- OpenScholar must be checked out at `<workspace>/OpenScholar` with its `.venv` intact.

### shared/

| Module | Key export | Purpose |
|--------|-----------|---------|
| `llm.py` | `LLMBackend` | OpenAI-compatible client with retry logic and token tracking. Resolves API key from `OPENAI_API_KEY`. |
| `evaluate.py` | `EvaluationHarness` | Runs any baseline over a list of claims; computes accuracy, macro-F1, macro-FPR, macro-FNR, binary-F1, and per-class P/R/F1/FPR/FNR with raw TP/FP/FN/TN counts. Supports **streaming saves and resume**: pass `resume_path` to `run()` — each result is flushed to JSONL immediately and already-completed claims (matched by `claim_id`) are skipped on restart. If the baseline's `verify()` accepts a `context` keyword argument, the full claim dict (including `evidence`, `pmid`, etc.) is forwarded automatically. |
| `verdict.py` | `Verdict`, `BaselineResult` | Pydantic schemas shared by all baselines. |
| `label_utils.py` | `normalize_label()` | Maps dataset-specific labels to canonical `{SUPPORT, REFUTE, NEI}`. |
| `cost_tracker.py` | `CostTracker` | Records LLM calls, token counts (exact, API-reported via `resp.usage`), and latency per claim. Cost in USD is a **proxy**: `(input_tokens / 1M × in_price) + (output_tokens / 1M × out_price)`. Prices are hardcoded in `DEFAULT_PRICING` (provider prefix stripped from model name before lookup; falls back to $3/$15 per 1M if model is unknown). Token counts are the durable ground truth saved in every `BaselineResult` — USD cost can be recomputed offline. |
| `prompts.py` | `VERIFICATION_SYSTEM_PROMPT`, … | Shared prompt templates — all baselines use the same verification prompt so only the evidence varies. |

## Output format

All baselines write results as **JSON Lines** (one `BaselineResult` per line) to:

```
results/baselines/<baseline>/<model-slug>/<dataset>_seed<N>.jsonl
```

Aggregated metrics (mean ± std over repeats) are written to:

```
results/baselines/<baseline>/<model-slug>/<dataset>_metrics.json
```

`BaselineResult` fields:

| Field | Type | Notes |
|-------|------|-------|
| `claim_id` | str | Dataset row identifier |
| `claim` | str | Original claim text |
| `gold_label` | str | Canonical: SUPPORT / REFUTE / NEI |
| `predicted_label` | str | Canonical predicted label |
| `confidence` | float | Model confidence (0–1; 0 if not reported) |
| `reasoning` | str | Model reasoning text or JSON reasoning field |
| `evidence` | list[str] | Citation references cited by the model |
| `input_tokens` | int | Input token count (0 for OpenScholar — not tracked) |
| `output_tokens` | int | Output token count (0 for OpenScholar — not tracked) |
| `cost_usd` | float | Per-claim USD cost |
| `latency_seconds` | float | Wall-clock time for the full claim |
| `baseline_name` | str | `random` / `llm_only` / `open_scholar` |
| `model` | str | Model name |

## Adding a new baseline

1. Create `baselines/my_baseline.py` with a class that exposes `verify(claim_id, claim, gold_label, *, context=None) -> BaselineResult`.  The optional `context` kwarg receives the full claim dict (including any extra CSV columns) when present — use it to access pre-retrieved evidence without changing the harness.
2. Register it in the `build_baseline()` factory in `run_baselines_datasets.py`.
3. For LLM-backed baselines, use `LLMBackend` for all LLM calls and attach a `CostTracker` instance.  For subprocess-backed baselines (like OpenScholar), track `cost_usd` via the subprocess output and set `input_tokens=0` / `output_tokens=0`.
