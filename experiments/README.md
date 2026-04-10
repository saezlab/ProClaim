# experiments/

Scripts and shared infrastructure for running and evaluating baselines against SIGNOR and other claim-verification datasets.

## Top-level scripts

| Script | Purpose |
|--------|---------|
| `run_baselines_datasets.py` | Run a baseline on pre-processed CSV datasets (SIGNOR, ConnectomeDB, SciFact-Open, CIViC-Fact); prints a summary table of Macro F1 / FPR / FNR / Cost (mean ± std over repeats). Accepts `--config` to load baseline parameters from a YAML file. |
| `run_signor_eval.py` | End-to-end SIGNOR evaluation using the full evidence-programming pipeline (runs `pkevolve.verification.evidence_programming` as a subprocess). Config: `configs/signor_eval_config.yaml`. |

### Quick start

```bash
# Run a baseline from a YAML config (recommended)
uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/random_baseline_config.yaml

uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/llm_only_config.yaml

uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/s2_retrieval_config.yaml

uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/open_scholar_config.yaml

uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/react_config.yaml

# CLI flags override config values, e.g. smoke-test with 1 claim:
uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/open_scholar_config.yaml \
    --limit 1
```

## configs/

YAML config files for `run_baselines_datasets.py`. Load with `--config`; any CLI flag overrides the config value.

| File | Baseline | Key settings |
|------|----------|-------------|
| `random_baseline_config.yaml` | `random` | `signor` + `connectomedb`, 10 repeats, seed=100 |
| `llm_only_config.yaml` | `llm_only` | `anthropic/claude-sonnet-4-6`, 3 repeats |
| `s2_retrieval_config.yaml` | `s2_retrieval` | `anthropic/claude-sonnet-4-6`, top_k=5, 1 repeat |
| `open_scholar_config.yaml` | `open_scholar` | `claude-sonnet-4-6`, S2 adaptive retrieval on (`os_retrieval: true`), top_n=10, 1 repeat |
| `fire_config.yaml` | `fire` | `anthropic/claude-sonnet-4-6`, max_steps=5, 1 repeat |
| `ace_config.yaml` | `ace` | `anthropic/claude-sonnet-4-6`, eval_only mode, 1 repeat |
| `react_config.yaml` | `react` | `gpt-4o-mini` (OpenAI), max_steps=10, 1 repeat |
| `signor_eval_config.yaml` | *(evidence-programming)* | Config for `run_signor_eval.py` (model, mode, iteration budget) |
| `test_config.yaml` | *(evidence-programming)* | Test config for `pkevolve.verification.evidence_programming` smoke tests |

YAML keys mirror `argparse` `dest` names (e.g. `os_model`, `os_retrieval`, `datasets_dir`). All keys are optional — omitted keys fall back to the CLI defaults.

## baselines/

Installable baseline classes. All baselines share the same infrastructure from `baselines/shared/`.

| Module | Class | Description |
|--------|-------|-------------|
| `random_baseline.py` | `RandomBaseline` | Chance floor — uniform draw from {SUPPORT, REFUTE, NEI} |
| `llm_only.py` | `LLMOnly` | Parametric knowledge ceiling — single LLM call, no retrieval |
| `s2_retrieval.py` | `S2Retrieval` | Fixed-k S2 RAG — searches Semantic Scholar with the raw claim, retrieves top-k abstracts, and classifies with a single LLM call. No iteration or feedback. The "paste search results into an LLM" baseline. |
| `open_scholar_baseline.py` | `OpenScholarBaseline` | RAG baseline — routes each claim through the [OpenScholar](../../OpenScholar) pipeline (Claude Sonnet 4.6 by default). When the dataset CSV provides an `evidence` snippet (e.g. SIGNOR, ConnectomeDB), it is forwarded as a retrieved passage alongside any live S2-retrieved passages. With `--os-retrieval`, OpenScholar runs its full feedback loop (keyword extraction → S2 search → re-rank → answer edit). Outputs `claim_verdict` JSON (SUPPORT / REFUTE / UNCERTAIN → normalised to SUPPORT / REFUTE / NEI). Results saved as `BaselineResult` JSONL, identical in schema to `llm_only`. |
| `fire_baseline.py` | `FIREBaseline` | Iterative retrieval baseline — [FIRE](../../fire) dynamically decides whether to search the web or finalise a verdict at each step, integrating reasoning and retrieval. Binary output (True/False) mapped to SUPPORT/REFUTE; failures → UNCERTAIN. Runs as subprocess in FIRE's own `.venv`. |
| `ace_baseline.py` | `ACEBaseline` | Agentic context engineering baseline — [ACE](../../ace) Generator agent classifies claims using a self-improving playbook. By default runs in eval_only mode with a built-in claim-verification playbook (no training). Runs as subprocess in ACE's own `.venv`. |
| `react_baseline.py` | `ReActBaseline` | Unconstrained agentic reasoning baseline — ReAct (Yao et al., ICLR 2023) Thought → Action → Observation loop with web search tools. The agent decides when to stop with no external sufficiency signal. Uses native tool-use API. The "why not just use a ReAct agent?" baseline. |

### S2 Retrieval-specific CLI flags (for `run_baselines_datasets.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--s2-model` | `anthropic/claude-sonnet-4-6` | LiteLLM model string for the verdict LLM call |
| `--s2-top-k` | `5` | Number of Semantic Scholar abstracts to retrieve per claim |

### Prerequisites for `S2Retrieval`

- An LLM API key recognised by litellm (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
- `S2_API_KEY` defaults to `""` (anonymous, rate-limited). Set it for higher throughput.

### OpenScholar-specific CLI flags (for `run_baselines_datasets.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--os-model` | `claude-sonnet-4-6` | Model name passed to OpenScholar's `--model_name` |
| `--os-api` | `anthropic` | API provider (e.g. `anthropic`, `gemini`) |
| `--os-top-n` | `5` | Number of context passages forwarded to the generator |
| `--os-max-tokens` | `3000` | Max generation tokens (0 = no constraint, uses OpenScholar default of 3000) |
| `--os-retrieval` | off | Enable S2 adaptive retrieval + self-feedback loop (`--ss_retriever --feedback`) |
| `--os-oracle-context` | off | Inject CSV evidence as oracle context (for oracle-leakage experiments) |
| `--os-reranker` | `OpenScholar/OpenScholar_Reranker` | Reranker model for OpenScholar (`--ranking_ce --reranker`). Set to empty string to disable. |
| `--os-task-name` | `claim_verdict_question` | OpenScholar task name passed to `--task_name` |

### Prerequisites for `OpenScholarBaseline`

- `ANTHROPIC_API_KEY` must be set (or present in `grn-llm-correct/.env`).
- `S2_API_KEY` defaults to `""` (anonymous, rate-limited). Set it for higher throughput.
- OpenScholar must be checked out at `<workspace>/OpenScholar` with its `.venv` intact.

### FIRE-specific CLI flags (for `run_baselines_datasets.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--fire-model` | `openai/gpt-4o-mini` | LiteLLM model string (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`) |
| `--fire-max-steps` | `5` | Maximum iterative search steps |

### Prerequisites for `FIREBaseline`

- FIRE must be cloned at `<workspace>/fire` with `.venv` set up (`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`).
- `SERPER_API_KEY` must be set (for Google Search via SerperAPI).
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` depending on the model.

### ACE-specific CLI flags (for `run_baselines_datasets.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--ace-model` | `openai/gpt-4o-mini` | LiteLLM model string (e.g. `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-20250514`) |
| `--ace-max-tokens` | `4096` | Max generation tokens per call |
| `--ace-playbook` | (built-in) | Path to a pre-trained playbook `.txt` file |

### Prerequisites for `ACEBaseline`

- ACE must be cloned at `<workspace>/ace` (reference only; runs in-process via litellm).
- An LLM API key recognised by litellm (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### ReAct-specific CLI flags (for `run_baselines_datasets.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--react-model` | `openai/gpt-4o-mini` | LiteLLM model string (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`) |
| `--react-max-steps` | `10` | Maximum number of search steps |
| `--react-search-backend` | `web` | Search backend: `web` (Serper/DuckDuckGo) or `s2` (Semantic Scholar) |

### Prerequisites for `ReActBaseline`

- An LLM API key recognised by litellm (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- `SERPER_API_KEY` (optional; falls back to DuckDuckGo via `ddgs`).

### shared/

| Module | Key export | Purpose |
|--------|-----------|---------|
| `llm.py` | `LLMBackend` | OpenAI-compatible client with retry logic and token tracking. Resolves API key from `OPENAI_API_KEY`. |
| `evaluate.py` | `EvaluationHarness` | Runs any baseline over a list of claims; computes accuracy, macro-F1, macro-FPR, macro-FNR, binary-F1, and per-class P/R/F1/FPR/FNR with raw TP/FP/FN/TN counts. Supports **streaming saves and resume**: pass `resume_path` to `run()` — each result is flushed to JSONL immediately and already-completed claims (matched by `claim_id`) are skipped on restart. If the baseline's `verify()` accepts a `context` keyword argument, the full claim dict (including `evidence`, `pmid`, etc.) is forwarded automatically. |
| `verdict.py` | `Verdict`, `BaselineResult` | Pydantic schemas shared by all baselines. |
| `label_utils.py` | `normalize_label()` | Maps dataset-specific labels to canonical `{SUPPORT, REFUTE, NEI}`. |
| `cost_tracker.py` | `CostTracker` | Records LLM calls, token counts (exact, API-reported via `resp.usage`), and latency per claim. Cost in USD is a **proxy**: `(input_tokens / 1M × in_price) + (output_tokens / 1M × out_price)`. Prices are hardcoded in `DEFAULT_PRICING` (provider prefix stripped from model name before lookup; falls back to $3/$15 per 1M if model is unknown). Token counts are the durable ground truth saved in every `BaselineResult` — USD cost can be recomputed offline. |
| `prompts.py` | `VERIFICATION_SYSTEM_PROMPT`, … | Shared prompt templates — all baselines use the same verification prompt so only the evidence varies. |

### Common CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | `0.0` | LLM sampling temperature (0.0–1.0). Overrides each baseline's default. |
| `--limit` | `0` | Limit number of claims per dataset (0 = all). |
| `--output-dir` | `results/baselines` | Directory to save results. |

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
| `baseline_name` | str | `random` / `llm_only` / `open_scholar` / … |
| `model` | str | Model name |
| `dataset` | str | Dataset name (e.g. `signor`, `connectomedb`) |

## Adding a new baseline

1. Create `baselines/my_baseline.py` with a class that exposes `verify(claim_id, claim, gold_label, *, context=None) -> BaselineResult`.  The optional `context` kwarg receives the full claim dict (including any extra CSV columns) when present — use it to access pre-retrieved evidence without changing the harness.
2. Register it in the `build_baseline()` factory in `run_baselines_datasets.py`.
3. For LLM-backed baselines, use `LLMBackend` for all LLM calls and attach a `CostTracker` instance.  For subprocess-backed baselines (like OpenScholar), track `cost_usd` via the subprocess output and set `input_tokens=0` / `output_tokens=0`.
