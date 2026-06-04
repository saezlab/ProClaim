# experiments/

This directory contains the reproducibility entrypoints for dataset evaluation and baselines. The root [README.md](../README.md) is the user-facing single-claim vignette; this file is for benchmark reruns and analysis.

## Verified command surfaces

These CLI entrypoints were checked with `--help` from the repository root:

```bash
uv run python experiments/run_baselines_datasets.py --help
uv run python experiments/run_signor_eval.py --help
uv run python experiments/run_connectomedb_eval.py --help
```

Actual evaluation runs require a configured `.env` with the relevant API keys.

## Smoke-test reproducibility

These are the smallest end-to-end evaluation reruns in this checkout. They use the included datasets, process a single row with `--limit 1`, and write namespaced outputs under `results/`.

```bash
# SIGNOR: run one bundled claim
uv run python experiments/run_signor_eval.py \
    --input-csv datasets/signor.csv \
    --config experiments/configs/signor_smoke_config.yaml \
    --limit 1 \
    --reps 1 \
    --run-tag reviewer_signor_smoke

# ConnectomeDB: run one bundled claim
uv run python experiments/run_connectomedb_eval.py \
    --input-csv datasets/connectomedb.csv \
    --config experiments/configs/connectomedb_smoke_config.yaml \
    --limit 1 \
    --reps 1 \
    --run-tag reviewer_connectomedb_smoke
```

The smoke configs default to hosted Anthropic models, so no local vLLM endpoint is required.

## Full evaluation configs

Use these configs when you want the full direct evaluation workflow instead of the one-row smoke tests:

- `experiments/configs/signor_direct_config.yaml`
- `experiments/configs/connectomedb_direct_config.yaml`

Keep the same commands as above, swap the config path, and remove `--limit 1` when you want the full dataset run.

If you want a local OpenAI-compatible endpoint for the subagent, set `LLM_BASE_URL` and use a config whose `llm.subagent_model` points at your local model.

## Baseline runners

`run_baselines_datasets.py` runs the baseline harness over prepared CSV datasets. Start from one of the YAML configs in `experiments/configs/` and override specific flags from the CLI when needed.

Examples:

```bash
uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/random_baseline_config.yaml

uv run python experiments/run_baselines_datasets.py \
    --config experiments/configs/open_scholar_config.yaml \
    --limit 1
```

## Included analysis commands

The repository already includes result summaries under `results/`. These two commands inspect the bundled baseline metrics without rerunning the experiments:

```bash
uv run python scripts/analysis/print_aggregated_baseline_metrics_table.py
uv run python scripts/analysis/print_baseline_metrics_table.py
```

## configs/

YAML keys mirror the corresponding `argparse` destination names. CLI flags override config values.

| File | Baseline or workflow | Key settings |
|------|----------------------|-------------|
| `random_baseline_config.yaml` | `random` | `signor` + `connectomedb`, 10 repeats, seed=100 |
| `llm_only_config.yaml` | `llm_only` | `anthropic/claude-sonnet-4-6`, 1 repeat |
| `retrieval_config.yaml` | `retrieval` | `anthropic/claude-sonnet-4-6`, search_backend=`s2`, top_k=5, 1 repeat |
| `retrieval_web_config.yaml` | `retrieval` | `anthropic/claude-sonnet-4-6`, search_backend=`web`, top_k=5, 1 repeat |
| `open_scholar_config.yaml` | `open_scholar` | adaptive S2 retrieval on, top_k=10, 1 repeat |
| `fire_config.yaml` | `fire` | `anthropic/claude-sonnet-4-6`, max_steps=5, 1 repeat |
| `safe_config.yaml` | `safe` | `anthropic/claude-sonnet-4-6`, top_k=3, max_steps=5, 1 repeat |
| `ace_config.yaml` | `ace` | `anthropic/claude-sonnet-4-6`, max_tokens=4096, 1 repeat |
| `react_web_config.yaml` | `react` | `anthropic/claude-sonnet-4-6`, search_backend=`web`, max_steps=10, 1 repeat |
| `react_s2_config.yaml` | `react` | `anthropic/claude-sonnet-4-6`, search_backend=`s2`, max_steps=10, 1 repeat |
| `signor_smoke_config.yaml` | direct SIGNOR smoke test | hosted Anthropic smoke setup |
| `connectomedb_smoke_config.yaml` | direct ConnectomeDB smoke test | hosted Anthropic smoke setup |
| `signor_direct_config.yaml` | direct SIGNOR full run | full direct evaluation settings |
| `connectomedb_direct_config.yaml` | direct ConnectomeDB full run | full direct evaluation settings |

## baselines/

Installable baseline classes. All baselines share the same infrastructure from `baselines/shared/`.

| Module | Class | Description |
|--------|-------|-------------|
| `random_baseline.py` | `RandomBaseline` | Chance floor — uniform draw from {SUPPORT, REFUTE, NEI} |
| `llm_only.py` | `LLMOnly` | Parametric knowledge ceiling — single LLM call, no retrieval |
| `retrieval_baseline.py` | `RetrievalBaseline` | Fixed-k RAG — searches Semantic Scholar or the web (controlled by `--search-backend`) with the raw claim, retrieves top-k results, and classifies with a single LLM call. No iteration or feedback. The "paste search results into an LLM" baseline. |
| `open_scholar_baseline.py` | `OpenScholarBaseline` | RAG baseline — routes each claim through the upstream [OpenScholar](OpenScholar) pipeline under `experiments/OpenScholar` (Claude Sonnet 4.6 by default). When the dataset CSV provides an `evidence` snippet (e.g. SIGNOR, ConnectomeDB), it is forwarded as a retrieved passage alongside any live S2-retrieved passages. With `--retrieval`, OpenScholar runs its full feedback loop (keyword extraction → S2 search → re-rank → answer edit). Outputs `claim_verdict` JSON (SUPPORT / REFUTE / UNCERTAIN → normalised to SUPPORT / REFUTE / NEI). Results saved as `BaselineResult` JSONL, identical in schema to `llm_only`. |
| `fire_baseline.py` | `FIREBaseline` | Iterative retrieval baseline — routes each claim through the upstream [FIRE](fire) verification loop in-process. The original FIRE search step is patched to use the repo's shared `do_search` web helper, while the upstream control flow and prompts remain intact. Binary output (True/False) is mapped to SUPPORT / REFUTE; failures default to UNCERTAIN. Upstream FIRE is web-only in this integration, and the SLURM launcher requests a GPU for FIRE jobs because the upstream sentence-similarity path may use CUDA. |
| `ace_baseline.py` | `ACEBaseline` | Agentic context engineering baseline — [ACE](../../ace) Generator agent classifies claims using a self-improving playbook. By default runs in eval_only mode with a built-in claim-verification playbook (no training). Runs as subprocess in ACE's own `.venv`. |
| `react_baseline.py` | `ReActBaseline` | Unconstrained agentic reasoning baseline — ReAct (Yao et al., ICLR 2023) Thought → Action → Observation loop with web search tools. The agent decides when to stop with no external sufficiency signal. Uses native tool-use API. The "why not just use a ReAct agent?" baseline. |
| `safe_baseline.py` | `SAFEBaseline` | SAFE baseline — routes each atomic claim through the upstream SAFE `rate_atomic_fact` loop in-process. The original SAFE search step is patched to use the repo's shared `do_search` web helper, and the SAFE final-verdict prompt is rewritten to reuse the shared SUPPORT / REFUTE / UNCERTAIN definitions used by the other baselines. Failures default to UNCERTAIN. Upstream SAFE is web-only in this integration. |

### Unified CLI flags

All baselines share a single set of core flags. Baseline-specific flags only take effect for the relevant baseline and are ignored otherwise.

#### Shared flags (all LLM-backed baselines)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-6` | LiteLLM model string (`provider/model-id`). For OpenScholar the provider prefix is split into `--model_name` / `--api` automatically. |
| `--temperature` | `0.0` | LLM sampling temperature (0.0–1.0). |
| `--max-tokens` | `4096` | Max generation tokens per LLM call. |
| `--max-steps` | `10` | Maximum iterative search/reasoning steps (fire, react, safe). |
| `--top-k` | `5` | Number of retrieved items (S2 abstracts for retrieval; passages for open_scholar). |
| `--limit` | `0` | Limit number of claims per dataset (0 = all). |
| `--output-dir` | `results/baselines` | Directory to save results. |

#### Retrieval-specific (retrieval and react baselines)

| Flag | Default | Description |
|------|---------|-------------|
| `--search-backend` | `s2` | Search backend: `web` (DuckDuckGo) or `s2` (Semantic Scholar). Used by `retrieval` and `react`. `fire` and `safe` ignore this flag and always use the shared web-search helper. |
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
- **OpenScholar**: OpenScholar checked out at `experiments/OpenScholar` with `.venv` intact.
- **FIRE (web)**: `ddgs` package (free, no API key).
- **SAFE (web)**: `ddgs` package (free, no API key).
- **ACE**: ACE cloned at `<workspace>/ace` (reference only; runs in-process).

## Output format

All baselines write results as **JSON Lines** (one `BaselineResult` per line) to:

```
results/baselines/<baseline>/<model-slug>/<dataset>_seed<N>.jsonl
```

Retrieval and ReAct runs add their backend slug to keep web and S2 outputs separate:

```
results/baselines/retrieval/<backend>/<model-slug>/top<K>/<dataset>_seed<N>.jsonl
results/baselines/react/<backend>/<model-slug>/<dataset>_seed<N>.jsonl
```

Aggregated metrics (mean ± std over repeats) are written to:

```
results/baselines/<baseline>/<model-slug>/<dataset>_metrics.json
```

Baselines that emit per-claim `.log` files now include the prompt text sent to the model. For subprocess-backed OpenScholar runs, the log records the effective task input payload and command invocation.

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
