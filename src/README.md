# PKEvolve – Source Package

The `src/pkevolve/` package contains the core library for GRN edge
verification. This document covers the two main subsystems: the
**verification loop** and the **sufficiency classifier** that powers its
feedback signal.

## Package Layout

```
src/pkevolve/
├── verification/       ← evidence programming loop (iterative claim verification)
├── llm/                ← LLM evaluator & paper rater
├── search/             ← paper retrieval (PubMed, web search, LangChain agent)
├── gnn/                ← GNN link-prediction baseline
└── utils/              ← shared helpers (SIGNOR data loading, graph utilities)

scripts/sufficiency_classifier/   ← training pipeline for the MLP used by verification
```

---

## Verification Subsystem (`pkevolve/verification/`)

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
   → final SUPPORT / REFUTE / INSUFFICIENT verdict
```

Steps 1–4 repeat until confidence ≥ threshold or the iteration cap is
reached.

### Modules

| File | Purpose |
|------|---------|
| `evidence_programming.py` | Unified CLI entry point (`--mode sdk \| repl`) |
| `evidence_api.py` | Pure Python API: search, extract, check sufficiency, emit verdict |
| `subagents.py` | LLM subagent functions: fact extraction, synthesis, conflict detection, gap query formulation |
| `evidence_state.py` | `EvidenceState` — mutable Pydantic container with auto-save to JSON |
| `data_models.py` | Pydantic v2 models: `PaperRecord`, `Fact`, `Stance`, `Gap`, `SufficiencyResult`, `VerificationVerdict` |
| `full_text.py` | Layered full-text retrieval: PMC → Europe PMC → INDRA → Unpaywall PDF |
| `feature_tools.py` | NLP + metadata feature computation (same tools used in classifier training) |
| `compressor.py` | L1 lossless deduplication of duplicate facts |
| `config.py` | `VerificationSettings` — pydantic-settings: CLI → YAML → env vars → defaults |
| `adapters.py` | Dataset adapters: SIGNOR, SciFact → common `Claim` format |
| `llm_factory.py` | `make_llm()` — factory for OpenAI-compatible LLM callables |
| `kernel_runner.py` | Persistent Jupyter kernel lifecycle management (shared by both modes) |
| `notebook_mcp.py` | MCP tool server for Jupyter notebook management (Mode A only) |
| `repl_orchestrator.py` | Mode B standalone REPL orchestrator |
| `renderers.py` | HTML / notebook rendering helpers |

### Key Data Models

**EvidenceState** tracks the full evidence collection per claim:

- `papers: dict[str, PaperRecord]` — keyed by PMID
- `facts: list[Fact]` — stance-labelled extractions (SUPPORT / REFUTE / NEUTRAL)
- `conflicts: list[Conflict]`
- `coverage: dict[str, float]` — per-subclaim coverage
- `sufficiency_history: list[SufficiencyResult]`
- Auto-saves to `evidence_state.json` after every mutation.

**SufficiencyResult** (returned by `check_sufficiency()`):

- `label`: `SUFFICIENT_SUPPORT | SUFFICIENT_REFUTE | INSUFFICIENT`
- `confidence`: 0–1 (MLP probability)
- `gaps`: list of `Gap` objects identifying what is missing

### Two Execution Modes

| | Mode A (`--mode sdk`) | Mode B (`--mode repl`) |
|-|----------------------|----------------------|
| **Outer loop** | Claude Agent SDK (Anthropic-compatible endpoint) | Any OpenAI-compatible LLM |
| **MCP server** | `notebook_mcp.py` — `nb_init`, `nb_execute`, etc. | None |
| **Audit trail** | Jupyter notebook (`evidence_report.ipynb`) | `conversation.json` + `trace.json` |
| **Requirements** | `claude_agent_sdk` + `GLM_API_KEY` | Any vLLM / Z.AI / local endpoint |

Both modes share: `evidence_api`, `subagents`, `kernel_runner`,
`evidence_state`, `data_models`, `full_text`.

### Usage

```bash
# Mode B (REPL) — any OpenAI-compatible endpoint
uv run python -m pkevolve.verification.evidence_programming \
    --mode repl \
    --model Qwen/Qwen3-8B \
    --subagent-base-url http://localhost:8000/v1/ \
    --claim "MAPK1 directly activates H3-3A."

# Mode A (SDK) — requires GLM_API_KEY
uv run python -m pkevolve.verification.evidence_programming \
    --mode sdk \
    --claim "SRC directly down-regulates CTTN." \
    --output-dir results/verification/src_cttn

# With a YAML config file
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/example_config.yaml \
    --claim "Does p53 activate BAX?"
```

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

### Feature Schema (24 features)

| Category | Features |
|----------|----------|
| **Metadata** (11) | `num_papers`, `num_papers_with_metadata`, `num_full_text`, `max_log_IF`, `mean_log_IF`, `max_h_index`, `avg_max_h_index`, `max_norm_citation`, `mean_norm_citation`, `latest_year_age`, `year_span` |
| **NLP** (7) | `max_similarity`, `mean_similarity` (SBERT cosine), `entailment_ratio`, `contradiction_ratio`, `controversy_index` (DeBERTa NLI), `max_entity_coverage`, `mean_entity_coverage` (scispacy) |
| **Cross** (6) | `weighted_entailment_IF`, `weighted_contradiction_IF`, `weighted_entailment_citation`, `weighted_contradiction_citation`, `weighted_entailment_temporal`, `weighted_contradiction_temporal` |

Top discriminators (by Cohen's d on one training run): `mean_similarity`,
`num_full_text`, `year_span`.

### Feature Extraction Modules

| File | Purpose |
|------|---------|
| `feature_extractor.py` | `PaperFeatureExtractor` — bibliometric metadata via OpenAlex + PubMed |
| `extract_features_scifact.py` | `SemanticSimilarityComputer` (SBERT), `NLIEntailmentComputer` (DeBERTa), `BiomedicalEntityExtractor` (scispacy) |
| `feature_aggregation.py` | `FeatureAggregator` — pools per-paper features into a single 24-dim vector |
| `nlp_tools.py` | NLP utility functions |

### Training Data Construction

From SciFact-Open claims, builds balanced samples (roughly 50/50 split):

| Pool type | Label | Strategy |
|-----------|-------|----------|
| `positive_support` | 1 | Papers unanimously supporting a claim |
| `positive_contradict` | 1 | Papers unanimously contradicting a claim |
| `negative_conflict` | 0 | Mixed support + contradict → ambiguous = insufficient |
| `negative_noise` | 0 | NEI claims + random irrelevant papers = insufficient |

### τ-Ablation Study (`tau_ablation/`)

Studies how threshold-based relabeling of borderline "conflict" samples
affects model performance. The minority-stance ratio is:

```
r_minority = min(N_sup, N_con) / (N_sup + N_con)
```

If `r_minority < τ`, the sample is relabeled from "insufficient" to
"sufficient" (the majority stance dominates). 14 threshold values
(0.00–1.00) are tested with multi-seed training for robustness.

| File | Purpose |
|------|---------|
| `tau_config.py` | Hardcoded threshold list |
| `generate_ablation_datasets.py` | Generate all 13 dataset variants |
| `relabel_by_threshold.py` | Implement relabeling logic |
| `train_ablation_models.py` | Train one model per τ |
| `train_ablation_multiseed.py` | Train with multiple random seeds |
| `analyze_ablation.py` | Compare performance across thresholds |
| `create_unambiguous_test_set.py` | Build a clean held-out test set |

### Other Classifier Scripts

| File | Purpose |
|------|---------|
| `analyze_classifier_features.py` | Feature importance analysis (Cohen's d, correlations) |
| `analyze_conflict_ratios.py` | Explore minority-stance ratio distribution |
| `model_selection.py` | Compare MLP vs logistic regression vs random forest |
| `test_mlp_classifier.py` | Evaluation script (accuracy, F1, ROC-AUC) |

---

## How They Connect

```
Sufficiency Classifier  (trained once, offline)
         │
         │  saved weights: mlp_classifier_weights.pth
         ▼
Verification Loop  (runs per claim)
  ├─ populate_paper_features(state)   ← computes NLP + metadata
  ├─ check_sufficiency(state)         ← loads MLP, predicts confidence + gaps
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
