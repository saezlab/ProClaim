# Verification System — Agent Reference

## Purpose

The `pkevolve.verification` package implements a **Metacognitive Evidence Verification** system. Given a scientific claim (e.g., a gene regulatory interaction), it searches PubMed for supporting/refuting papers, extracts grounded facts, computes NLP/metadata features, runs a trained MLP sufficiency classifier, and emits a structured verdict.

## Architecture Overview

```
                  ┌──────────────────────────────┐
                  │  evidence_programming.py      │  ← Entry point (Claude Agent SDK)
                  │  (notebook orchestrator)       │
                  └──────────┬───────────────────┘
                             │ calls MCP tools
                  ┌──────────▼───────────────────┐
                  │  notebook_mcp.py              │  ← MCP tool server (nb_execute, etc.)
                  │  kernel_runner.py             │  ← Jupyter kernel lifecycle
                  └──────────┬───────────────────┘
                             │ executes Python in kernel
          ┌──────────────────▼──────────────────────────┐
          │              evidence_api.py                  │  ← Core API (all functions)
          │  search → extract → populate features →      │
          │  check_sufficiency → emit_verdict             │
          └──┬──────┬──────────┬──────────┬─────────────┘
             │      │          │          │
    ┌────────▼┐ ┌───▼────┐ ┌──▼───────┐ ┌▼──────────────┐
    │full_text│ │subagents│ │feature_  │ │model_registry │
    │.py      │ │.py      │ │tools.py  │ │.py            │
    │(4-layer │ │(LLM     │ │(NLP +    │ │(singleton     │
    │ fetch)  │ │ calls)  │ │ metadata)│ │ model cache)  │
    └─────────┘ └─────────┘ └──────────┘ └───────────────┘
             │                    │               │
    ┌────────▼────────────────────▼───────────────▼─────┐
    │  data_models.py  │  evidence_state.py  │ config.py │
    │  (Pydantic v2)   │  (mutable state +   │ (settings │
    │                   │   JSON persistence) │  + YAML)  │
    └───────────────────┴────────────────────┴──────────┘
```

## Module Reference

### Data Layer

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `data_models.py` | Pydantic v2 schemas for all evidence structures | `PaperRecord`, `Fact`, `Stance`, `Conflict`, `Gap`, `SufficiencyResult`, `VerificationVerdict`, `PaperFeatureVector`, `NLPFeatureVector` |
| `evidence_state.py` | Central mutable state container with auto-save to JSON | `EvidenceState`, `TraceLog` |
| `config.py` | Layered settings (CLI > YAML > env > defaults) via pydantic-settings | `VerificationSettings`, `APISettings`, `LLMSettings`, `get_settings()` |

### Core API

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `evidence_api.py` | Pure Python evidence manipulation API — **the main interface** | `search_pubmed()`, `search_pubmed_progressive()`, `search_pubmed_llm()`, `get_full_text_article()`, `extract_and_add_facts()`, `extract_and_add_facts_batch()`, `populate_paper_features()`, `filter_papers_by_stance()`, `check_sufficiency()`, `emit_verdict()` |
| `subagents.py` | LLM subagent prompts for fact extraction, synthesis, conflict detection, gap queries | `extract_facts()`, `synthesize_subclaim()`, `detect_conflicts()`, `identify_gaps()`, `formulate_gap_queries()` |
| `llm_factory.py` | Factory for thread-safe `llm(prompt) -> str` callables with retry + streaming | `make_llm()` |

### Retrieval

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `full_text.py` | 4-tier full-text fetching: PMC → Europe PMC → INDRA → Unpaywall+PDF | `fetch_full_text(pmid, *, doi=, title=, max_chars=)` |
| `adapters.py` | Dataset-agnostic claim adapters (SIGNOR edges, SciFact claims) | `SignorAdapter`, `SciFactAdapter`, `Claim` dataclass |

### Feature Extraction & Classification

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `feature_tools.py` | NLP features (scispaCy NER, SBERT similarity, DeBERTa NLI) + metadata (OpenAlex) | `SemanticSimilarityComputer`, `NLIEntailmentComputer`, `PaperFeatureExtractor`, `compute_entity_coverage()` |
| `model_registry.py` | Global singleton cache for heavy ML models (SBERT, NLI, MLP classifier) | `get_mlp_classifier()`, `get_semantic_similarity_computer()`, `get_nli_entailment_computer()`, `get_feature_aggregator()`, `get_metadata_extractor()`, `prewarm_all_models()`, `get_cache_status()` |
| `compressor.py` | L1 lossless deduplication of facts by (text, stance) | `SufficiencyPreservingCompressor` |

### Orchestration & Rendering

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `evidence_programming.py` | Main entry point — Claude Agent SDK orchestrator with notebook as audit trail | `async verify_claim_notebook(cfg)`, `main()` |
| `notebook_mcp.py` | MCP tool server exposing 9 notebook tools (`nb_init`, `nb_execute`, `nb_render_*`, `nb_read_output`, `nb_save`) | FastMCP server |
| `kernel_runner.py` | Jupyter kernel lifecycle (start, execute, shutdown) | `KernelRunner` |
| `renderers.py` | HTML renderers for evidence state (papers, facts, sufficiency, verdict) displayed in notebook cells | `render_papers()`, `render_facts()`, `render_sufficiency()`, `render_verdict()` |

## Data Flow — Verification Workflow

```
1. CLAIM SETUP
   Subclaims are set on EvidenceState (populated externally or by the orchestrator).

2. SEARCH (per iteration, up to MAX_ITERATIONS=8)
   subclaims → search_pubmed_progressive() → PMIDs[]
   PMIDs → get_full_text_article() → full text or abstract

3. FACT EXTRACTION
   (claim, paper text) → extract_and_add_facts(llm, pmid, state) → Fact[]
   Each fact: {text, stance: SUPPORT|REFUTE|NEUTRAL, source_pmid, confidence}

4. FEATURE POPULATION
   papers[] → populate_paper_features(state) → PaperFeatureVector + NLPFeatureVector per paper
     - Entity coverage (scispaCy NER recall)
     - Semantic similarity (SBERT cosine, max-pooled over chunks)
     - NLI scores (DeBERTa cross-encoder: entailment/contradiction/neutral)
     - Metadata (publication year, impact factor, citations, h-index via OpenAlex)

5. SUFFICIENCY CHECK
   aggregated features → MLP classifier → SufficiencyResult {label, confidence, gaps[]}
   If insufficient → loop back to step 2 with gap-targeted queries

6. VERDICT
   state → emit_verdict(verdict, confidence, reasoning, key_evidence, gaps_remaining, state) → VerificationVerdict
```

## Key Data Structures

### EvidenceState (central mutable state)

```python
EvidenceState(
    claim="Does MAPK1 phosphorylate H3?",
    subclaims=["MAPK1 has kinase activity", "H3 is a MAPK1 substrate", ...],
    papers={pmid: PaperRecord(pmid, title, abstract, full_text, metadata, nlp)},
    facts=[Fact(text, stance, source_pmid, confidence)],
    conflicts=[Conflict(fact_a_id, fact_b_id, description, severity)],
    sufficiency_history=[SufficiencyResult(label, confidence, gaps)],
    iteration=0,
)
```

Auto-saves to `{workspace}/evidence_state.json` after every mutation.

### Feature Vectors (per paper)

```python
PaperFeatureVector(pmid, publication_year, log_impact_factor, normalized_citation_count, author_h_index_max)
NLPFeatureVector(claim_entity_coverage, semantic_similarity, nli_entailment, nli_contradiction, nli_neutral)
```

### 24-Feature Aggregated Schema (MLP input)

Aggregated across all papers in the evidence pool by `FeatureAggregator`:

| Category | Features |
|----------|----------|
| **Metadata (11)** | `num_papers`, `num_papers_with_metadata`, `num_full_text`, `max_log_IF`, `mean_log_IF`, `max_h_index`, `avg_max_h_index`, `max_norm_citation`, `mean_norm_citation`, `latest_year_age`, `year_span` |
| **NLP (7)** | `entailment_ratio`, `contradiction_ratio`, `controversy_index`, `max_entity_coverage`, `mean_entity_coverage`, `max_similarity`, `mean_similarity` |
| **Cross (6)** | `weighted_entailment_IF`, `weighted_contradiction_IF`, `weighted_entailment_citation`, `weighted_contradiction_citation`, `weighted_entailment_temporal`, `weighted_contradiction_temporal` |

## Configuration

### From YAML (recommended)

```python
from pkevolve.verification.config import VerificationSettings
cfg = VerificationSettings.from_yaml("experiments/signor_eval_config.yaml")
```

### From CLI

```bash
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/signor_eval_config.yaml \
    --claim "Does MAPK1 phosphorylate H3?"
```

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.model` | — | Main LLM model identifier |
| `llm.subagent_model` | (falls back to `model`) | Inner subagent model |
| `llm.agent_base_url` | `https://api.z.ai/api/anthropic` | Anthropic-compatible endpoint |
| `llm.subagent_base_url` | `http://localhost:8000/v1` | OpenAI-compatible (vLLM) |
| `max_iterations` | 8 | Search/extract loop limit |
| `sufficiency_threshold` | 0.80 | MLP confidence threshold |
| `api.glm_api_key` | env `GLM_API_KEY` | API key (priority: GLM > ZAI > OpenAI) |

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GLM_API_KEY` / `ZAI_API_KEY` | Yes (one of) | LLM API authentication |
| `UNPAYWALL_EMAIL` | Optional | Full-text PDF retrieval via Unpaywall |
| `ELSEVIER_API_KEY` | Optional | Full-text via INDRA/Elsevier |
| `MLP_MODEL_DIR` | Optional | Override MLP classifier directory (default: `results/models/classifier_best/`) |
| `NB_MAX_OUTPUT_CHARS` | Optional | Notebook output truncation limit (default: 12000) |

## Common Agent Tasks

### Run a single verification

```python
from pkevolve.verification.config import VerificationSettings
from pkevolve.verification.evidence_programming import verify_claim_notebook

cfg = VerificationSettings.from_yaml("experiments/config.yaml", claim="Does X regulate Y?")
result_path = await verify_claim_notebook(cfg)  # async — use asyncio.run() if not in async context
```

### Use the evidence API directly (without notebook)

```python
from pkevolve.verification.evidence_state import EvidenceState
from pkevolve.verification.evidence_api import (
    search_pubmed_progressive, extract_and_add_facts,
    populate_paper_features, check_sufficiency, emit_verdict
)
from pkevolve.verification.llm_factory import make_llm

llm = make_llm(base_url="http://localhost:8000/v1", api_key="EMPTY", model="my-model")
state = EvidenceState.init_new(claim="...", subclaims=["..."], workspace=Path("workspace/"))

# Search → Extract → Features → Check → Verdict
pmids = search_pubmed_progressive(claim, state)
for pmid in pmids:
    extract_and_add_facts(llm, pmid, state)
populate_paper_features(state)
result = check_sufficiency(state, llm)
verdict = emit_verdict(
    verdict=result.label, confidence=result.confidence,
    reasoning="...", key_evidence=["..."], gaps_remaining=[],
    state=state,
)
```

### Load and inspect an existing state

```python
from pkevolve.verification.evidence_state import EvidenceState
state = EvidenceState.load(Path("workspace/evidence_state.json"))
print(f"Papers: {len(state.papers)}, Facts: {len(state.facts)}")
print(f"Last sufficiency: {state.sufficiency_history[-1].label}")
```

### Prewarm ML models (avoid cold-start latency)

```python
from pkevolve.verification.model_registry import prewarm_all_models
timings = prewarm_all_models()  # loads SBERT, NLI, MLP into memory
```
