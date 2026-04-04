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
  - `claude_sdk/` – Anthropic SDK scripts: `run_signor_qa_claude.py`, `run_signor_qa_glm.py`
  - `context_labeling/` – dataset generation for model distillation
  - `sufficiency_classifier/` – MLP training pipeline: `resolve_pmids.py` → `fetch_full_texts.py` → `generate_classifier_data.py` → `train_mlp_classifier.py`; includes `tau_ablation/` sub-package
- **`experiments/`** – baseline comparison experiments, run via `uv run python experiments/...`
  - `run_signor_eval.py` – end-to-end SIGNOR evaluation with evidence-programming pipeline
  - `run_baselines_datasets.py` – multi-dataset baseline runner; outputs Macro F1/FPR/FNR/Cost tables
  - `baselines/` – installable baseline classes (`RandomBaseline`, `LLMOnly`) plus `shared/` framework
- **`langgraph/`** – LangGraph agent prototype: `edge_correction_agent.py` defines a `StateGraph` workflow with LLM → confidence → relevance → retry/search nodes
- **`data/signor/`** – ground truth CSVs (`true_positive_edges.csv`, `true_negative_edges.csv`) with SIGNOR schema columns
- **`prompts/`** – prompt templates (`.txt`, `.md`) for edge removal/recovery experiments
- **`moon_denoising/`** – MOON network-denoising integration scripts (`correct_pkn_for_moon.py`, `graph_repair.py`, `llm_to_pkn.py`)

## Key Conventions

### Running scripts
Always use `uv run python` from the project root:
```bash
uv run python scripts/qa_pipeline/run_qa.py --mode nosearch --model gpt-oss-120b
uv run python scripts/analysis/analyze_qa_results.py --mode nosearch --model gpt-oss-120b
```

### LLM client pattern
Core library (`src/pkevolve/`) uses the OpenAI-compatible client directly. Local models default to `http://localhost:8000/v1` with `api_key="EMPTY"`. Cloud models read keys from `.env` via `python-dotenv`:
```python
from openai import OpenAI
# Cloud models — OpenAI-compatible endpoint:
client = OpenAI(base_url="<cloud_base_url>", api_key=os.getenv("OPENAI_API_KEY"))
# Local models:
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
```

The `experiments/baselines/shared/llm.py` layer uses **LiteLLM** (`LLMBackend`) instead, which is provider-agnostic (OpenAI, Anthropic, Gemini, GLM-4). Use `LLMBackend` for new baseline scripts; use the direct OpenAI client for core library code.

API key env var: `OPENAI_API_KEY`.

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
- Key deps: `langchain-core`, `langchain-openai`, `langgraph`, `torch`, `torch_geometric`, `paper-search-mcp`, `litellm`, `spacy`, `sentence-transformers`, `pymupdf`, `pydantic-settings`
- API keys in `.env` at project root: `OPENAI_API_KEY`, `ANTHROPIC_AUTH_TOKEN` (never commit)
- Optional: `UNPAYWALL_EMAIL` for full-text PDF retrieval via Unpaywall, `ELSEVIER_API_KEY` for Elsevier full text via INDRA, `PUBMED_EMAIL` / `PUBMED_API_KEY` for NCBI Entrez

### Verification subsystem (`src/pkevolve/verification/`)
- `evidence_api.py` – Pure Python evidence API (search, extract, check sufficiency, emit verdict)
- `evidence_programming.py` – Evidence programming orchestration loop
- `full_text.py` – Layered full-text retrieval: PMC → INDRA → Unpaywall+PDF
- `subagents.py` – LLM subagent functions (fact extraction, synthesis, conflict detection)
- `kernel_runner.py` – Jupyter kernel management for the evidence REPL
- `data_models.py` – Pydantic models: PaperRecord, Fact, Stance, SufficiencyResult, etc.
- `evidence_state.py` – EvidenceState: central state object with auto-save
- `config.py` – Configuration management (Pydantic Settings)
- `adapters.py`, `llm_factory.py`, `model_registry.py` – LLM interface layer

### Baselines framework (`experiments/baselines/shared/`)
- `llm.py` – `LLMBackend`: LiteLLM wrapper with retry logic, returns `(text, input_tokens, output_tokens)`
- `evaluate.py` – `EvaluationHarness`: runs any baseline over claims; computes Accuracy, Macro F1, FPR, FNR, binary F1
- `verdict.py` – `Verdict`, `BaselineResult` Pydantic schemas
- `label_utils.py` – `normalize_label()`: maps dataset-specific labels to canonical `{SUPPORT, REFUTE, NEI}`
- `cost_tracker.py` – `CostTracker`: records token counts and USD cost per claim (proxy pricing hardcoded)
- `prompts.py` – `VERIFICATION_SYSTEM_PROMPT`, `VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL`, query generation templates

## When Adding New Features

1. Reusable logic goes in `src/pkevolve/`; one-off experiments go in `scripts/` with `argparse` CLI
2. New evaluator modes should extend `GeneInteractionEvaluator` or follow its prompt structure
3. Paper search integrations should implement rate-limiting (see `WebSearchAssistant._rate_limit()`)
4. Always include `--help` docstrings in argparse scripts; follow the existing multi-mode pattern in `run_qa.py`
5. New baseline scripts go in `experiments/`; use `LLMBackend` and `EvaluationHarness` from `experiments/baselines/shared/`
