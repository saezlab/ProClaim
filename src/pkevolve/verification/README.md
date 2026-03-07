# Verification Subsystem

Evidence programming loop for GRN edge verification. Verifies scientific
claims (e.g. "MAPK1 directly activates H3-3A") by iteratively gathering,
analysing, and reasoning over published papers.

The LLM is used surgically for three steps (fact extraction, gap query
formulation, verdict). Everything else — search, sufficiency checking,
compression, state management — is deterministic.

## Architecture

```
evidence_programming.py   ← unified CLI entry point (--mode sdk | repl)
├── Mode A (sdk)          ← Claude Agent SDK + nb_execute (notebook audit trail)
│   ├── notebook_mcp.py   ← MCP tool server: nb_init, nb_execute, nb_render_*
│   └── kernel_runner.py  ← persistent Jupyter kernel management
├── Mode B (repl)         ← standalone REPL, any OpenAI-compatible endpoint
│   ├── repl_orchestrator.py  ← LLM generates Python code blocks
│   └── kernel_runner.py      ← same shared kernel
├── evidence_api.py       ← pure Python API: search, extract, check, verdict
├── subagents.py          ← LLM subagent functions (facts, synthesis, conflicts, gaps)
├── evidence_state.py     ← EvidenceState: mutable container with auto-save
├── data_models.py        ← Pydantic v2 models: PaperRecord, Fact, Stance, Gap, …
├── full_text.py          ← layered full-text retrieval (PMC → INDRA → Unpaywall)
├── compressor.py         ← L1 deduplication (lossless)
├── config.py             ← VerificationSettings (pydantic-settings + YAML + CLI)
├── adapters.py           ← dataset adapters (SIGNOR, SciFact → common Claim format)
└── renderers.py          ← HTML/notebook rendering helpers
```

## How it works

The loop iterates through these steps:

1. **Progressive PubMed search** — queries are tried from most specific (exact gene pair + mechanism) to broader bridge queries (each gene + shared biological terms). This handles cases where the two genes never co-occur in the literature.
2. **Full-text retrieval** — layered fallback: PMC XML → Europe PMC REST → INDRA → Unpaywall PDF. Retrieved text is validated against the paper title.
3. **Fact extraction (LLM)** — for each paper, the LLM extracts stance-labeled facts (SUPPORT / REFUTE / NEUTRAL) with confidence scores and subclaim mappings.
4. **Sufficiency check** — a deterministic classifier evaluates source diversity, stance coverage, conflicts, and subclaim coverage. This is the agent's primary feedback signal (no LLM cost).
5. **Gap-targeted retrieval (LLM)** — if insufficient, the classifier reports gap types and the LLM formulates new PubMed queries. Failed queries are tracked to avoid repetition.
6. **Citation graph fallback** — when keyword searches find no new papers, `find_related_articles` explores citation neighborhoods (rotating seed PMIDs across iterations).
7. **Compression** — when `token_estimate` exceeds 40,000, `compress_evidence` deduplicates facts (L1 lossless).
8. **Verdict (LLM)** — once sufficient or after max iterations, the LLM synthesises a final SUPPORT / REFUTE / INSUFFICIENT verdict.

Steps 1–7 repeat until confidence ≥ threshold or max iterations are reached.

## Two execution modes

Both modes share the same evidence_api, subagents, and kernel_runner.

### Mode A: Claude Agent SDK (`--mode sdk`)

- The outer agent loop is Claude Agent SDK (Anthropic-compatible endpoint)
- **One MCP server**: `notebook-tools` (`pkevolve.verification.notebook_mcp`) — provides `nb_init`, `nb_execute`, `nb_render_*`, `nb_save`
- `nb_execute` is the primary tool — the agent writes Python code that calls evidence_api functions directly in a persistent Jupyter kernel
- The notebook serves as an **audit trail** — every search, extraction, and decision is a cell
- Requires `GLM_API_KEY` in `.env` and the `claude_agent_sdk` package

### Mode B: Standalone REPL (`--mode repl`)

- No Claude SDK needed — uses any OpenAI-compatible endpoint (vLLM, Z.AI, local)
- The LLM generates Python code blocks; the orchestrator extracts and executes them in a Jupyter kernel
- Kernel stdout/stderr is fed back as the next user message
- Hard ceiling of 30 LLM turns; conversation log saved to `conversation.json`
- Supports batch processing via `verify_claims_batch()`

## Configuration

Configuration uses `pydantic-settings` with layered resolution (highest priority first):

1. CLI flags (`--model`, `--claim`, etc.)
2. Environment variables (from `.env` at project root)
3. YAML config file (`--config path/to/config.yaml`)
4. Field defaults

### YAML config files

Store experiment parameters in YAML for reproducibility (see `experiments/example_config.yaml`):

```yaml
mode: repl
max_iterations: 8
sufficiency_threshold: 0.80

llm:
  model: glm-5
  subagent_model: Qwen/Qwen3-8B
  subagent_base_url: "http://localhost:8000/v1/"
  agent_base_url: "https://api.z.ai/api/anthropic"
  temperature: 0.7

output_dir: results/verification/my_experiment
verbose: true
```

The `claim` field is intentionally excluded from YAML — it varies per run and must be provided via `--claim`.

Save a snapshot of the current run config: `cfg.save_yaml("experiments/run_snapshot.yaml")`

## Usage

```bash
# Start a vLLM endpoint (for subagent LLM calls)
bash /hps/nobackup/saezrodriguez/ail/workspace/start_vllm_ihpc.sh \
    --user <username> --gpu-type a100 --gpus 1 --model Qwen/Qwen3-8B

# Mode B (REPL) — any OpenAI-compatible endpoint
uv run python -m pkevolve.verification.evidence_programming \
    --mode repl \
    --model Qwen/Qwen3-8B \
    --subagent-base-url http://localhost:8000/v1/ \
    --claim "MAPK1 directly activates H3-3A."

# Mode A (SDK) — requires GLM_API_KEY in .env
uv run python -m pkevolve.verification.evidence_programming \
    --mode sdk \
    --claim "SRC directly down-regulates CTTN." \
    --output-dir results/verification/src_cttn

# Using a YAML config file
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/example_config.yaml \
    --claim "Does p53 activate BAX?"

# Override config file values with CLI flags
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/example_config.yaml \
    --claim "EGFR activates MAPK1 via phosphorylation." \
    --model gpt-4o --max-iterations 12
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--claim` | *(required)* | Scientific claim to verify |
| `--config`, `-c` | — | Path to a YAML config file |
| `--mode` | `sdk` | Orchestration mode: `sdk` or `repl` |
| `--model` | `glm-5` | Model identifier for the outer/main agent |
| `--subagent-model` | same as `--model` | Model for inner subagent LLM calls |
| `--subagent-base-url` | `http://localhost:8000/v1/` | OpenAI-compatible base URL for subagent |
| `--agent-base-url` | `https://api.z.ai/api/anthropic` | Anthropic-compatible base URL (Mode A) |
| `--threshold` | `0.80` | Sufficiency confidence threshold |
| `--max-iterations` | `8` | Max sufficiency-check iterations |
| `--output-dir` | `results/verification/notebook_demo` | Output workspace directory |
| `--notebook-path` | `<output-dir>/evidence_report.ipynb` | Notebook path (Mode A only) |
| `-v` | off | Enable debug logging |

## Output

Both modes produce structured output in the workspace directory:

- **`workspace/evidence_state.json`** — full evidence state: papers, facts, conflicts, sufficiency history, synthesis, coverage, token estimate
- **`workspace/verdict.json`** — final structured verdict with reasoning, key evidence, and remaining gaps
- **`run.log`** — execution log

Mode A additionally produces:
- **`evidence_report.ipynb`** — Jupyter notebook with rendered papers, facts, sufficiency checks, and verdict cards

Mode B additionally produces:
- **`workspace/conversation.json`** — full LLM conversation log
- **`workspace/trace.json`** — operation trace with timestamps

## Environment variables

| Variable | Description |
|----------|-------------|
| `GLM_API_KEY` | API key for Z.AI / GLM models (primary) |
| `ZAI_API_KEY` | Alias for `GLM_API_KEY` (fallback) |
| `OPENAI_API_KEY` | Fallback API key |
| `ANTHROPIC_BASE_URL` | Override the Anthropic-compatible endpoint for Mode A |
| `LLM_BASE_URL` | Override the OpenAI-compatible endpoint for subagent calls |
| `API_TIMEOUT_MS` | Request timeout in milliseconds (default: `3000000`) |
| `UNPAYWALL_EMAIL` | Email for Unpaywall API (full-text PDF retrieval) |
| `ELSEVIER_API_KEY` | Elsevier API key for INDRA full-text via ScienceDirect |

## Prerequisites

- Python ≥ 3.10, managed with `uv`
- The `pkevolve` package must be importable (`uv pip install -e .`)
- For Mode A: `claude_agent_sdk` + `GLM_API_KEY` in `.env`
- For Mode B: any OpenAI-compatible endpoint (vLLM, Z.AI OpenAI endpoint, etc.)

## Module reference

| File | Purpose |
|------|---------|
| `evidence_programming.py` | Unified CLI entry point; Mode A orchestrator |
| `repl_orchestrator.py` | Mode B standalone REPL orchestrator |
| `evidence_api.py` | Pure Python API: search, extract, check sufficiency, verdict |
| `subagents.py` | LLM subagent functions: fact extraction, synthesis, conflict detection, gap identification |
| `evidence_state.py` | `EvidenceState`: mutable Pydantic container with auto-save to JSON |
| `data_models.py` | Pydantic v2 models: `PaperRecord`, `Fact`, `Stance`, `Gap`, `SufficiencyResult`, `VerificationVerdict` |
| `full_text.py` | Layered full-text retrieval: PMC → Europe PMC → INDRA → Unpaywall PDF |
| `compressor.py` | `SufficiencyPreservingCompressor`: L1 deduplication (lossless) |
| `config.py` | `VerificationSettings`: unified config via pydantic-settings + YAML + CLI |
| `adapters.py` | Dataset adapters: `SignorAdapter`, `SciFact` → common `Claim` format |
| `notebook_mcp.py` | MCP tool server for Jupyter notebook management (Mode A) |
| `kernel_runner.py` | Jupyter kernel lifecycle management (shared by both modes) |
| `renderers.py` | HTML/notebook rendering helpers |
