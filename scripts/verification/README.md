# Verification Scripts

## demo_evidence_programming.py

Standalone demo of the **evidence programming loop** for GRN edge verification.
It calls MCP tool functions directly as Python -- no MCP server or Claude Agent SDK needed.

### How it works

The loop iterates through these steps:

1. **Progressive PubMed search** -- queries are tried from most specific (exact gene pair) to broader bridge queries (each gene + shared biological terms). This handles cases where the two genes never co-occur in the literature.
2. **Fact extraction (LLM)** -- for each paper, the LLM extracts stance-labeled facts (SUPPORT / REFUTE / NEUTRAL) with confidence scores.
3. **Sufficiency check** -- a deterministic classifier evaluates whether enough evidence has been gathered (source diversity, stance coverage, conflicts).
4. **Gap-targeted retrieval (LLM)** -- if insufficient, the classifier reports gap types and the LLM formulates new PubMed queries. Failed queries are tracked to avoid repetition.
5. **Citation graph fallback** -- when keyword searches find no new papers, `find_related_articles` explores citation neighborhoods (rotating seed PMIDs across iterations).
6. **Verdict (LLM)** -- once sufficient or after max iterations, the LLM synthesises a final SUPPORT / REFUTE / INSUFFICIENT verdict.

The LLM is only used for three steps (fact extraction, gap query formulation, verdict). Everything else (search, sufficiency, compression) is deterministic.

### Usage

```bash
# Using a preset endpoint
uv run python scripts/verification/demo_evidence_programming.py \
    --preset local_glm_5 \
    --claim "MAPK1 directly activates H3-3A."

# Custom endpoint
uv run python scripts/verification/demo_evidence_programming.py \
    --base-url http://my-server:8000/v1 \
    --api-key EMPTY \
    --model my-model \
    --claim "Does p53 activate BAX?"
```

### Available presets

| Preset | Endpoint | Model |
|--------|----------|-------|
| `local_oss_120b` | `http://localhost:8000/v1` | `gpt-oss-120b` |
| `local_glm_5` | `http://codon-gpu-001.ebi.ac.uk:8000/v1/` | `glm-5-fp8` |

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--claim` | *(required)* | Scientific claim to verify |
| `--preset` | — | Endpoint preset name |
| `--base-url` | — | OpenAI-compatible base URL (alternative to preset) |
| `--api-key` | `EMPTY` | API key |
| `--model` | `default` | Model identifier |
| `--max-iterations` | `5` | Max evidence loop iterations |
| `--threshold` | `0.80` | Sufficiency confidence threshold |
| `--output-dir` | auto temp dir | Workspace directory for evidence state |
| `-v` | off | Verbose / debug logging |

### Output

The script prints a step-by-step trace to stdout and writes structured JSON files to the workspace directory:

- `evidence_state.json` -- papers, facts, sufficiency history, trace log
- `verdict.json` -- final verdict with reasoning

## demo_evidence_programming.py (notebook mode)

Notebook-enabled evidence verification — runs the Claude Agent SDK orchestrator with **two MCP servers** and produces a rich Jupyter notebook artifact that documents the entire evidence programming workflow.

Connects directly to GLM's native Anthropic-compatible endpoint at `api.z.ai`.

### How it works

The script launches a Claude Agent SDK agent with two MCP tool servers:

1. **evidence-tools** (`pkevolve.verification.mcp_tools`) — search, classify, extract, compress, verdict
2. **notebook-tools** (`pkevolve.verification.notebook_mcp`) — write results into a live Jupyter notebook

The agent follows this workflow:

1. **Initialize** — creates the notebook (`nb_init`) and decomposes the claim into subclaims.
2. **Progressive PubMed search** — `search_pubmed_progressive` automatically broadens from strict gene-pair queries to bridge queries. Papers are documented via `nb_render_papers`.
3. **Fact extraction (LLM)** — for each paper, tries `get_full_text_article` first, falls back to `get_paper_text`. Facts are extracted via `add_facts` and displayed with `nb_render_facts`.
4. **Sufficiency check** — `check_sufficiency` is the agent's primary feedback signal (deterministic, no LLM cost). Results are visualized via `nb_render_sufficiency`.
5. **Gap-targeted retrieval** — if insufficient, the classifier reports gap types and the agent formulates targeted queries via `search_for_gap`.
6. **Citation graph fallback** — `find_related_articles` explores citation neighborhoods when keyword searches return no new papers.
7. **Synthesis** — `update_synthesis` consolidates evidence per subclaim; `add_conflict` records conflicting facts.
8. **Compression** — when `token_estimate` exceeds 40,000, `compress_evidence` is called.
9. **Verdict (LLM)** — `emit_verdict` produces a final SUPPORT / REFUTE / INSUFFICIENT verdict, displayed via `nb_render_verdict`.

The agent uses `nb_markdown` between steps to explain its reasoning. Steps 4–8 repeat until confidence ≥ threshold or max iterations (8) are reached.

### Usage

```bash
# Default (GLM endpoint, glm-4.6 model)
uv run python scripts/verification/demo_evidence_programming.py \
    --claim "SRC directly down-regulates CTTN"

# Custom model
uv run python scripts/verification/demo_evidence_programming.py \
    --claim "Does p53 activate BAX?" \
    --model glm-5

# Custom output paths
uv run python scripts/verification/demo_evidence_programming.py \
    --claim "MAPK1 directly activates H3-3A." \
    --output-dir results/verification/mapk1_h3 \
    --notebook-path results/verification/mapk1_h3/report.ipynb
```

### Prerequisites

- `GLM_API_KEY` must be set in `.env` at the project root.
- The `claude_agent_sdk` package must be installed.
- The `pkevolve` package must be importable (install via `uv pip install -e .`).

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--claim` | *(required)* | Scientific claim to verify |
| `--model` | `glm-4.6` | Model identifier |
| `--threshold` | `0.80` | Sufficiency confidence for early stopping |
| `--max-iterations` | `8` | Maximum verification iterations |
| `--output-dir` | `results/verification/notebook_demo` | Output workspace directory |
| `--notebook-path` | `<output-dir>/evidence_report.ipynb` | Path for the output notebook |
| `-v` | off | Enable debug logging |

### Output

The script produces:

- **`evidence_report.ipynb`** — Jupyter notebook with rendered papers, facts, sufficiency checks, and verdict cards. Open in JupyterLab or VS Code to view the HTML widgets.
- **`workspace/evidence_state.json`** — full evidence state (papers, facts, sufficiency history, synthesis, coverage, token estimate)
- **`workspace/verdict.json`** — final structured verdict with reasoning, key evidence, and remaining gaps
- **`evidence_report.log`** — agent trace log (tool calls, agent messages)

### Environment variables

| Variable | Description |
|----------|-------------|
| `GLM_API_KEY` | API key for the GLM endpoint (required) |
| `ANTHROPIC_BASE_URL` | Override the default GLM endpoint (`https://api.z.ai/api/anthropic`) |
| `API_TIMEOUT_MS` | Request timeout in milliseconds (default: `3000000`) |

## run_verification.py

Orchestrator entry point that uses the Claude Agent SDK with the full MCP tool server. See `src/pkevolve/verification/orchestrator.py` for details.
