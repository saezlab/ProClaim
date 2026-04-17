# experiments/

Scripts and shared infrastructure for running and evaluating baselines against SIGNOR and other claim-verification datasets.

## Top-level scripts

| Script | Purpose |
|--------|---------|
| `run_baselines_datasets.py` | Run a baseline on pre-processed CSV datasets (SIGNOR, ConnectomeDB, SciFact-Open, CIViC-Fact); prints a summary table of Macro F1 / FPR / FNR / Cost (mean ± std over repeats). Accepts `--config` to load baseline parameters from a YAML file. |
| `run_connectomedb_all.sh` | Run all 9 baselines sequentially on one or more datasets. Uses `flock` to prevent concurrent instances (avoids S2 rate-limit races). Supports resume — re-running skips already-completed claims. |
| `run_signor_eval.py` | End-to-end SIGNOR evaluation using the full evidence-programming pipeline (runs `pkevolve.verification.evidence_programming` as a subprocess). Config: `configs/signor_eval_config.yaml`. |

### Quick start

```bash
# ── Run all 9 baselines on a dataset (sequential, flock-guarded) ──
bash experiments/run_connectomedb_all.sh                       # connectomedb (default)
bash experiments/run_connectomedb_all.sh signor                # signor only
bash experiments/run_connectomedb_all.sh connectomedb signor   # both datasets

# Run in background with logging:
nohup bash experiments/run_connectomedb_all.sh connectomedb > nohup_connectomedb.out 2>&1 &

# ── Run a single baseline from a YAML config ──
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
| `llm_only_config.yaml` | `llm_only` | `anthropic/claude-sonnet-4-6`, 1 repeat |
| `s2_retrieval_config.yaml` | `retrieval` | `anthropic/claude-sonnet-4-6`, search_backend=s2, top_k=5, 1 repeat |
| `open_scholar_config.yaml` | `open_scholar` | `anthropic/claude-sonnet-4-6`, S2 adaptive retrieval on (`retrieval: true`), top_k=10, 1 repeat |
| `fire_config.yaml` | `fire` | `anthropic/claude-sonnet-4-6`, max_steps=5, 1 repeat |
| `ace_config.yaml` | `ace` | `anthropic/claude-sonnet-4-6`, max_tokens=4096, 1 repeat |
| `react_config.yaml` | `react` | `anthropic/claude-sonnet-4-6`, max_steps=10, 1 repeat |
| `signor_eval_config.yaml` | *(evidence-programming)* | Config for `run_signor_eval.py` (model, mode, iteration budget) |
| `test_config.yaml` | *(evidence-programming)* | Test config for `pkevolve.verification.evidence_programming` smoke tests |

YAML keys mirror `argparse` `dest` names (e.g. `model`, `max_steps`, `top_k`, `retrieval`). All keys are optional — omitted keys fall back to the CLI defaults.

## run_connectomedb_all.sh

Runs all 9 baselines sequentially on one or more datasets. Accepts dataset names as positional arguments (default: `connectomedb`).

### Baselines run (in order)

| # | Category | Baseline | Model |
|---|----------|----------|-------|
| 1 | LLM-only | `llm_only` | `anthropic/claude-sonnet-4-6` |
| 2 | LLM-only | `llm_only` | `vertex_ai/gemini-2.5-flash` |
| 3 | Static retrieval | `retrieval` (S2) | `anthropic/claude-sonnet-4-6` |
| 4 | Static retrieval | `retrieval` (S2) | `vertex_ai/gemini-2.5-flash` |
| 5 | Adaptive retrieval | `react` (web) | `anthropic/claude-sonnet-4-6` |
| 6 | Adaptive retrieval | `react` (S2) | `anthropic/claude-sonnet-4-6` |
| 7 | Adaptive retrieval | `ace` | `anthropic/claude-sonnet-4-6` |
| 8 | Adaptive retrieval | `fire` | `anthropic/claude-sonnet-4-6` |
| 9 | Adaptive retrieval | `open_scholar` | `claude-sonnet-4-6` (S2 retrieval, no oracle evidence) |

```
LLM-only:
Claude-Sonnet-4.6
Gemini-2.5-Flash

Static retrieval:
Claude-Sonnet-4.6 + S2 (--baseline retrieval --search-backend s2)
Gemini-2.5-Flash + S2 (--baseline retrieval --search-backend s2)

Adaptive retrieval (Claude-Sonnet-4.6 for all below):
ReAct + web search
ReAct + S2
ACE
FIRE
OpenScholar-Sonnet-4.6 (no evidence)
```

### Concurrency & resume

- **flock guard**: Uses `flock` on `/tmp/run_all_baselines.lock` so only one instance can run at a time. A second launch exits immediately with an error. This prevents S2 rate-limit races between concurrent processes.
- **Resume**: Each baseline uses the evaluation harness's resume feature — already-completed claims (matched by `claim_id` in the JSONL) are skipped. Re-running the script after a partial failure picks up exactly where it left off.

## baselines/

Installable baseline classes. All baselines share the same infrastructure from `baselines/shared/`.

| Module | Class | Description |
|--------|-------|-------------|
| `random_baseline.py` | `RandomBaseline` | Chance floor — uniform draw from {SUPPORT, REFUTE, NEI} |
| `llm_only.py` | `LLMOnly` | Parametric knowledge ceiling — single LLM call, no retrieval |
| `retrieval_baseline.py` | `RetrievalBaseline` | Fixed-k RAG — searches Semantic Scholar or the web (controlled by `--search-backend`) with the raw claim, retrieves top-k results, and classifies with a single LLM call. No iteration or feedback. The "paste search results into an LLM" baseline. |
| `open_scholar_baseline.py` | `OpenScholarBaseline` | RAG baseline — routes each claim through the [OpenScholar](../../OpenScholar) pipeline (Claude Sonnet 4.6 by default). When the dataset CSV provides an `evidence` snippet (e.g. SIGNOR, ConnectomeDB), it is forwarded as a retrieved passage alongside any live S2-retrieved passages. With `--retrieval`, OpenScholar runs its full feedback loop (keyword extraction → S2 search → re-rank → answer edit). Outputs `claim_verdict` JSON (SUPPORT / REFUTE / UNCERTAIN → normalised to SUPPORT / REFUTE / NEI). Results saved as `BaselineResult` JSONL, identical in schema to `llm_only`. |
| `fire_baseline.py` | `FIREBaseline` | Iterative retrieval baseline — [FIRE](../../fire) dynamically decides whether to search the web or finalise a verdict at each step, integrating reasoning and retrieval. Binary output (True/False) mapped to SUPPORT/REFUTE; failures → UNCERTAIN. Runs as subprocess in FIRE's own `.venv`. |
| `ace_baseline.py` | `ACEBaseline` | Agentic context engineering baseline — [ACE](../../ace) Generator agent classifies claims using a self-improving playbook. By default runs in eval_only mode with a built-in claim-verification playbook (no training). Runs as subprocess in ACE's own `.venv`. |
| `react_baseline.py` | `ReActBaseline` | Unconstrained agentic reasoning baseline — ReAct (Yao et al., ICLR 2023) Thought → Action → Observation loop with web search tools. The agent decides when to stop with no external sufficiency signal. Uses native tool-use API. The "why not just use a ReAct agent?" baseline. |

### Unified CLI flags

All baselines share a single set of core flags. Baseline-specific flags only take effect for the relevant baseline and are ignored otherwise.

#### Shared flags (all LLM-backed baselines)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-6` | LiteLLM model string (`provider/model-id`). For OpenScholar the provider prefix is split into `--model_name` / `--api` automatically. |
| `--temperature` | `0.0` | LLM sampling temperature (0.0–1.0). |
| `--max-tokens` | `4096` | Max generation tokens per LLM call. |
| `--max-steps` | `10` | Maximum iterative search/reasoning steps (fire, react). |
| `--top-k` | `5` | Number of retrieved items (S2 abstracts for retrieval; passages for open_scholar). |
| `--limit` | `0` | Limit number of claims per dataset (0 = all). |
| `--output-dir` | `results/baselines` | Directory to save results. |

#### Retrieval-specific (retrieval & react baselines)

| Flag | Default | Description |
|------|---------|-------------|
| `--search-backend` | `s2` | Search backend: `web` (DuckDuckGo) or `s2` (Semantic Scholar). Used by both `retrieval` and `react` baselines. |
| `--no-strip-query` | (stripping on) | Disable stripping dataset-specific boilerplate from claims before S2 search. |

#### OpenScholar-specific

| Flag | Default | Description |
|------|---------|-------------|
| `--retrieval` | off | Enable S2 adaptive retrieval + self-feedback loop. |
| `--oracle-context` | off | Inject CSV evidence as oracle context (for ablation only). |
| `--reranker` | `OpenScholar/OpenScholar_Reranker` | Reranker model. Set to empty string to disable. |
| `--task-name` | `claim_verdict_question` | OpenScholar task name (e.g. `claim_verdict_question`, `claim_verdict`). |

#### ACE-specific

| Flag | Default | Description |
|------|---------|-------------|
| `--playbook` | (built-in) | Path to a pre-trained playbook `.txt` file. |

### Prerequisites

- **All LLM baselines**: An LLM API key recognised by litellm (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- **Retrieval / OpenScholar**: `S2_API_KEY` (defaults to `""`, anonymous, rate-limited) when using `--search-backend s2`.
- **Retrieval (web) / ReAct (web)**: `ddgs` package (free, no API key).
- **OpenScholar**: OpenScholar checked out at `<workspace>/OpenScholar` with `.venv` intact.
- **FIRE**: `ddgs` package (free, no API key).
- **ACE**: ACE cloned at `<workspace>/ace` (reference only; runs in-process).

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
