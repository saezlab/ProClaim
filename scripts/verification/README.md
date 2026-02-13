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

## run_verification.py

Orchestrator entry point that uses the Claude Agent SDK with the full MCP tool server. See `src/pkevolve/verification/orchestrator.py` for details.
