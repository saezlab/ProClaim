# PKEvolve – Copilot Instructions

## Project Overview

PKEvolve is a bioinformatics research tool that validates Gene Regulatory Network (GRN) edges using LLMs. Given a (source gene, target gene, interaction type) triple from the SIGNOR database, it asks an LLM whether the interaction is supported by scientific evidence and compares the answer to ground truth.

## Architecture

- **`src/pkevolve/`** – installable library (`hatchling` build, imported as `pkevolve`)
  - `llm/evaluator.py` – `GeneInteractionEvaluator`: core LLM judge that constructs structured Yes/No/None prompts for edge validation
  - `llm/rater.py` – `PaperRater`: pairwise paper comparison via LLM to rank evidence quality
  - `search/` – paper retrieval: PubMed (`custom_pubmed.py`), web search (`main.py` `WebSearchAssistant`), LangChain-based agent (`paper_search_agent.py`)
  - `gnn/` – GNN link-prediction baseline (`GATv2Conv` model in `train_gnn_signor.py`)
  - `utils/signor_utils.py` – shared helpers: `load_signor_data()`, `construct_signor_question()` (maps interaction types to natural-language questions)
- **`scripts/`** – CLI entry points, always run via `uv run python scripts/...`
  - `qa_pipeline/run_qa.py` – main QA pipeline (modes: `nosearch`, `specify_paper`, `search`)
  - `analysis/` – metrics computation (F1, accuracy) on QA results
  - `claude_sdk/` – Anthropic SDK scripts with PubMed plugin + `edge_curation` skill
  - `context_labeling/` – dataset generation for model distillation
- **`langgraph/`** – LangGraph agent prototype: `edge_correction_agent.py` defines a `StateGraph` workflow with LLM → confidence → relevance → retry/search nodes
- **`data/signor/`** – ground truth CSVs (`true_positive_edges.csv`, `true_negative_edges.csv`) with SIGNOR schema columns
- **`prompts/`** – prompt templates (`.txt`, `.md`) for edge removal/recovery experiments

## Key Conventions

### Running scripts
Always use `uv run python` from the project root:
```bash
uv run python scripts/qa_pipeline/run_qa.py --mode nosearch --model gpt-oss-120b
uv run python scripts/analysis/analyze_qa_results.py --mode nosearch --model gpt-oss-120b
```

### LLM client pattern
All LLM calls use the OpenAI-compatible client. Local models default to `http://localhost:8000/v1` with `api_key="EMPTY"`. Cloud models (GLM, Claude) read keys from a `.env` file via `python-dotenv`. Follow this existing pattern:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
```

### Path resolution
Scripts resolve PROJECT_ROOT relative to their own location:
```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
```
Keep this pattern; do not assume the working directory.

### SIGNOR data model
Edges are `(ENTITYA, ENTITYB, EFFECT)` triples. The `EFFECT` column uses values like `up-regulates activity`, `down-regulates quantity by destabilization`, `form complex`. Always convert interactions to natural-language questions via `construct_signor_question()` — never construct question strings manually.

### Structured LLM output
The evaluator expects LLM responses in a strict format ending with `Answer: Yes`, `Answer: No`, or `Answer: None`. The `StructuredOutcome` Pydantic model captures `reasoning`, `answer_text`, `answer` (bool/None), and `usage` stats. Preserve this schema when adding new evaluation modes.

### Results directory layout
```
results/qa_unified/{mode}/{model}/[scenario/ordering/]{true_positive_edges,true_negative_edges}/
```
Each edge result is a JSON file named `{SOURCE}_{TARGET}_{INTERACTION}.json`.

### Environment
- Python ≥ 3.10, managed with `uv`
- Key deps: `langchain-core`, `langchain-openai`, `langgraph`, `torch`, `torch_geometric`, `paper-search-mcp`
- API keys in `.env` at project root: `GLM_API_KEY`, `ANTHROPIC_AUTH_TOKEN` (never commit)

## When Adding New Features

1. Reusable logic goes in `src/pkevolve/`; one-off experiments go in `scripts/` with `argparse` CLI
2. New evaluator modes should extend `GeneInteractionEvaluator` or follow its prompt structure
3. Paper search integrations should implement rate-limiting (see `WebSearchAssistant._rate_limit()`)
4. Always include `--help` docstrings in argparse scripts; follow the existing multi-mode pattern in `run_qa.py`
