# ProClaim – Source Package

The `src/proclaim/` package contains the core library for GRN edge
verification. This document is the **top-level map**; detailed API
reference lives in each subsystem's own README.

## Package Layout

```
src/proclaim/
├── verification/       ← evidence programming loop  → see verification/README.md
├── llm/                ← LLM evaluator & paper rater
├── search/             ← paper retrieval (PubMed, web search, LangChain agent)
├── gnn/                ← GNN link-prediction baseline
└── utils/              ← shared helpers (SIGNOR data loading, graph utilities)

scripts/sufficiency_classifier/   ← MLP training pipeline  → see its README.md
```

---

## Verification Subsystem (`proclaim/verification/`)

An evidence programming loop that verifies scientific claims (e.g. "MAPK1
directly activates H3-3A") by iteratively searching PubMed, extracting
facts from papers, and deciding whether the gathered evidence is sufficient
to render a verdict.

The LLM is used for **only three steps**: fact extraction, gap query
formulation, and the final verdict. Everything else—search, sufficiency
checking, compression, state management—is deterministic.

### The Loop

```
1. Progressive PubMed search  (most-specific → broadest queries)
       ↓
2. Per paper:
   a) Retrieve full text  (PMC → Europe PMC → INDRA → Unpaywall PDF)
   b) Extract stance-labelled facts via LLM → add to state
       ↓
3. Populate paper features  (NLP + bibliometric metadata)
       ↓
4. Check sufficiency  (MLP classifier — no LLM cost)
   ├─ Sufficient  → emit verdict (LLM)
   └─ Insufficient → identify gaps → formulate new queries (LLM) → go to 1
       ↓
5. After max iterations or sufficient confidence
   → final SUPPORT / REFUTE / UNCERTAIN verdict
```

Steps 1–4 repeat until confidence ≥ threshold or the iteration cap is
reached.

> **Full module reference, data models, configuration, and code examples:**
> see [`proclaim/verification/README.md`](proclaim/verification/README.md).

---

## Sufficiency Classifier (`scripts/sufficiency_classifier/`)

A binary MLP classifier that predicts whether the current evidence pool is
**sufficient** to verify a claim. This model is the "free" (no LLM cost)
feedback signal that drives the verification loop's search-or-stop
decision.

### Training Pipeline

```
resolve_pmids.py → fetch_full_texts.py → generate_classifier_data.py → train_mlp_classifier.py
```

1. **`resolve_pmids.py`** — maps SciFact-Open doc IDs to PubMed PMIDs.
2. **`fetch_full_texts.py`** — retrieves full-text papers from PMC.
3. **`generate_classifier_data.py`** — constructs balanced
   (claim, evidence_pool, label) triples, computes 24 features per sample.
4. **`train_mlp_classifier.py`** — trains the MLP:
   `Linear(24→64) → BN → ReLU → Dropout → Linear(64→32) → BN → ReLU → Dropout → Linear(32→1)`.
   Loss: `BCEWithLogitsLoss` with `pos_weight` for class imbalance.

Saved weights: `results/models/classifier/mlp_classifier_weights.pth`.

> **Feature schema, training data construction, τ-ablation study, and
> script reference:** see
> [`scripts/sufficiency_classifier/README.md`](../../scripts/sufficiency_classifier/README.md).

---

## How They Connect

```
Sufficiency Classifier  (trained once, offline)
         │
         │  saved weights: mlp_classifier_weights.pth
         ▼
Verification Loop  (runs per claim)
  ├─ populate_paper_features(state)   ← computes NLP + metadata
  ├─ check_sufficiency(state, llm)    ← loads MLP, predicts confidence + gaps
  │    ├─ SUFFICIENT  → emit_verdict()
  │    └─ INSUFFICIENT → formulate_gap_queries() → search more papers
  └─ repeat
```

The classifier replaces an LLM "do we have enough evidence?" call each
iteration with a deterministic, zero-cost forward pass. The LLM is
reserved for the creative tasks: understanding papers, identifying gaps,
and synthesising the final verdict.

### Critical call order inside the loop

```python
search_pubmed_progressive(claim, state)

for pmid in new_pmids:
    extract_and_add_facts(llm, pmid, state)

populate_paper_features(state)   # must be called before check_sufficiency
result = check_sufficiency(state, llm)
```

`populate_paper_features()` is idempotent — it skips papers whose features
are already populated.
