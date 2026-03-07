# Potential Extensions: Open-Source Libraries for Evidence Programming

This document catalogs open-source packages that can enhance the Evidence Programming framework using a **glue development approach** — integrating mature, well-tested libraries rather than reimplementing functionality.

---

## 1. Literature Retrieval & Full-Text Access

### paper-qa
**Repository**: https://github.com/Future-House/paper-qa  
**Purpose**: Full RAG pipeline for scientific papers with automatic citation, chunking, and retrieval.

**Integration Point**: Replace or supplement `get_full_text_article` and fact extraction in `mcp_tools.py`. Already handles PDF parsing, embedding, and QA in a production-ready pipeline.

**Current Gap Addressed**: The framework currently uses raw PMC XML parsing for full text. paper-qa provides:
- Automatic PDF download and parsing
- Semantic chunking with overlap
- Dense retrieval over chunks
- Citation-aware answer generation

### semanticscholar
**Repository**: https://pypi.org/project/semanticscholar/  
**Purpose**: Python client for Semantic Scholar API with citation graphs and paper recommendations.

**Integration Point**: Add as fallback to PubMed in `search_pubmed_progressive`. The data model already has `source: "semantic_scholar"` field but no S2 integration exists.

**Current Gap Addressed**: PubMed-only search misses preprints and papers without MeSH annotations. S2 provides:
- Broader coverage (200M+ papers)
- Citation graph traversal
- Author disambiguation
- Paper recommendations

### pyalex
**Repository**: https://github.com/J535D165/pyalex  
**Purpose**: OpenAlex API client for bibliometric data.

**Integration Point**: Formalize existing ad-hoc OpenAlex queries (see `src/pkevolve/search/inspect_openalex.py`) into proper journal impact metrics and citation counts.

**Current Gap Addressed**: The codebase already queries OpenAlex for journal info but without a proper client. pyalex provides:
- Clean API access
- Rate limiting
- Caching
- Full entity resolution

### unpaywall-py
**Repository**: https://unpaywall.org/data  
**Purpose**: Legal open-access PDF discovery.

**Integration Point**: Add to `get_full_text_article` in `mcp_tools.py` when PMC OA fails.

**Current Gap Addressed**: The framework only checks PMC for full text. Unpaywall indexes:
- Repository copies (institutional, subject)
- Bronze OA (free-to-read on publisher sites)
- Green OA versions

---

## 2. Biomedical NER & Knowledge Extraction

### scispacy
**Repository**: https://allenai.github.io/scispacy/  
**Purpose**: SpaCy models trained on biomedical text (CRAFT, NCBI Disease, BC5CDR).

**Integration Point**: Use in `formulate_pubmed_query` (mcp_tools.py:85-118) to extract gene/protein entities more robustly than the current regex pattern.

**Current Gap Addressed**: The regex `r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b"` misses:
- Lowercase gene names (e.g., "p53")
- Multi-word entities ("epidermal growth factor receptor")
- Disease/chemical mentions that provide query context

scispacy provides:
- Pre-trained NER models (en_ner_craft_md, en_ner_bionlp13cg_md)
- Entity linking to UMLS/GO/HPO
- Dependency parsing for relation extraction

### INDRA (Integrated Network and Dynamical Reasoning Assembler)
**Repository**: https://github.com/sorgerlab/indra  
**Purpose**: Assembles mechanistic models from literature; already handles SIGNOR, REACH, and pathway databases.

**Integration Point**: Could replace much of `llm_extract_facts` — INDRA statements are stance-labeled and grounded by design.

**Current Gap Addressed**: The LLM-based fact extraction in `evidence_programming.py` is:
- Expensive (LLM call per paper)
- Inconsistent (depends on prompt engineering)
- Ungrounded (raw text, no entity IDs)

INDRA provides:
- Pre-assembled statements from SIGNOR, Reactome, PathwayCommons
- Reader outputs from REACH, Sparser, TRIPS
- Belief scoring (analogous to your confidence field)
- Automatic deduplication and contradiction detection

**High Priority**: INDRA is directly relevant to this project's domain (GRN curation).

### PyOBO
**Repository**: https://github.com/biopragmatics/pyobo  
**Purpose**: Unified access to biomedical ontologies (GO, ChEBI, HP, etc.).

**Integration Point**: Normalize gene/protein names before PubMed queries in `formulate_pubmed_query`.

**Current Gap Addressed**: Gene symbols are ambiguous:
- "H3-3A" vs "H3F3A" vs "HIST3H3"
- "MAPK1" vs "ERK2" vs "p42MAPK"

PyOBO provides:
- Synonym lookup
- Cross-reference mapping (UniProt, Entrez, HGNC)
- Ontology hierarchy traversal

### gilda
**Repository**: https://github.com/indralab/gilda  
**Purpose**: Biomedical entity grounding and disambiguation.

**Integration Point**: Resolve gene symbols before searching in `formulate_pubmed_query`.

**Current Gap Addressed**: The current implementation trusts gene symbols from claims verbatim. gilda provides:
- Context-aware disambiguation
- Grounding to standard identifiers (HGNC, UniProt, GO)
- Confidence scores for groundings

---

## 3. LLM Orchestration & Prompt Engineering

### DSPy
**Repository**: https://github.com/stanfordnlp/dspy  
**Purpose**: Optimizable prompt pipelines with automatic few-shot selection and prompt optimization.

**Integration Point**: Replace manual prompts in `llm_extract_facts`, `llm_formulate_gap_query`, `llm_verdict` with learnable modules.

**Current Gap Addressed**: Current prompts in `evidence_programming.py` are:
- Hand-crafted and static
- Not optimized for specific models
- No automatic few-shot selection

DSPy provides:
- `dspy.Predict`, `dspy.ChainOfThought` modules
- Automatic prompt optimization via `dspy.teleprompt`
- Cross-model portability

### Instructor
**Repository**: https://github.com/jxnl/instructor  
**Purpose**: Structured output extraction via Pydantic models.

**Integration Point**: Replace manual JSON parsing in `_strip_code_fences` and the try/except JSON handling throughout the codebase.

**Current Gap Addressed**: The current implementation:
```python
def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    # ... manual parsing
```

Instructor provides:
- Automatic retry on validation failure
- Type-safe extraction with Pydantic
- Support for all major LLM providers

**Example refactor**:
```python
import instructor
from pydantic import BaseModel

class FactList(BaseModel):
    facts: list[Fact]

client = instructor.from_openai(OpenAI(...))
result = client.chat.completions.create(
    model="...",
    response_model=FactList,
    messages=[...]
)
```

### LangGraph (already installed)
**Repository**: https://github.com/langchain-ai/langgraph  
**Purpose**: Stateful, multi-actor agent workflows with checkpointing.

**Integration Point**: Currently underutilized. The evidence programming loop in `evidence_programming.py` is a manual `for` loop; this could be a proper `StateGraph`.

**Current Gap Addressed**: The existing `langgraph/edge_correction_agent.py` shows partial adoption, but the verification framework doesn't use it.

**Benefits of full adoption**:
- Built-in checkpointing (resume failed runs)
- Visualization of agent state
- Proper retry semantics with backoff
- Streaming support

---

## 4. Claim Verification & Evaluation

### SciFact Baseline
**Repository**: https://github.com/allenai/scifact  
**Purpose**: Reference implementations for scientific claim verification + evaluation scripts.

**Integration Point**: Use for apples-to-apples comparison; the `SciFactAdapter` in `adapters.py` already exists.

**Current Gap Addressed**: Currently no standardized baseline comparison. SciFact provides:
- Pre-trained verifier models
- Standard evaluation metrics
- Training data for the sufficiency classifier

### FEVER Scorer
**Repository**: https://github.com/sheffieldnlp/fever-scorer  
**Purpose**: Standard verification metrics (FEVER score, label accuracy).

**Integration Point**: Add to `scripts/analysis/` alongside `analyze_qa_results.py`.

**Current Gap Addressed**: Current metrics are ad-hoc (F1, accuracy). FEVER provides:
- Standardized scoring protocol
- Evidence F1 (not just label accuracy)
- Multi-hop reasoning metrics

---

## 5. Knowledge Graphs & Pathway Databases

### OmniPath / pypath
**Repository**: https://github.com/saezlab/pypath  
**Purpose**: Unified signaling pathway database aggregating SIGNOR, Reactome, KEGG, etc.

**Integration Point**: Cross-reference SIGNOR edges against multiple databases for ground-truth enrichment.

**Current Gap Addressed**: The framework only uses SIGNOR ground truth. OmniPath provides:
- 100+ integrated resources
- Literature references per interaction
- Confidence scoring
- Directed/signed interactions

**Note**: This is from the same lab (saezlab) as the project.

### NDEx Python
**Repository**: https://github.com/ndexbio/ndex2-client  
**Purpose**: Network Data Exchange client for sharing biological networks.

**Integration Point**: Export verification results as annotated CX2 networks.

**Current Gap Addressed**: Results are currently JSON files. NDEx enables:
- Visual network exploration
- Community sharing
- Provenance tracking

---

## Integration Priority Matrix

| Library | Effort | Impact | Priority |
|---------|--------|--------|----------|
| **Instructor** | Low (~10 lines) | Medium | **P0** |
| **INDRA** | Medium | High | **P0** |
| **scispacy** | Low | Medium | **P1** |
| **paper-qa** | Medium | High | **P1** |
| **gilda** | Low | Medium | **P1** |
| **semanticscholar** | Low | Medium | **P2** |
| **DSPy** | High | High | **P2** |
| **pypath/OmniPath** | Medium | Medium | **P2** |
| **pyalex** | Low | Low | **P3** |
| **unpaywall-py** | Low | Low | **P3** |

---

## Recommended Integration Path

```
Phase 1: Low-hanging fruit (1-2 days)
├── Instructor: Replace JSON parsing
├── gilda: Entity grounding before search
└── scispacy: Better NER in formulate_pubmed_query

Phase 2: Core enhancement (1-2 weeks)
├── INDRA: Pre-assembled statements for fact extraction
├── paper-qa: Full-text RAG pipeline
└── semanticscholar: Broader paper coverage

Phase 3: Optimization (ongoing)
├── DSPy: Optimized prompts
├── LangGraph: Proper state machine for verification loop
└── OmniPath: Multi-database ground truth
```

---

## Installation Notes

Most libraries install via pip:
```bash
uv add scispacy
uv add indra
uv add instructor
uv add paper-qa
uv add gilda
uv add semanticscholar
uv add pyalex
```

scispacy requires model download:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

INDRA requires Java for some readers (REACH, Sparser).
