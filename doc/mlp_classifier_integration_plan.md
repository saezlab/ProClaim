# MLP Sufficiency Classifier — Integration Plan

> **Date**: 2026-03-06  
> **Branch**: RLM  
> **Status**: Planning  
> **Depends on**: feature-classifier branch (merged), colleague's trained checkpoints

## 1. Overview

Replace the heuristic rule-based `SufficiencyClassifier` with a trained MLP
(`SufficiencyMLP(10→64→32→1)`) that predicts P(sufficient) from aggregated
per-paper metadata and NLP features. The MLP was trained on 726 SciFact claim–
evidence pairs across 4 pool types by the colleague's `feature-classifier`
branch.

The integration adds three new capabilities to the verification pipeline:

1. **SBERT semantic similarity** between claim and evidence (per paper)
2. **NLI entailment/contradiction scores** via cross-encoder (per paper)
3. **Feature aggregation** from per-paper vectors to a single 10-dim global
   vector that the MLP classifies

The existing heuristic classifier remains as a fallback (default when MLP is
not configured or when metadata extraction fails).

### System Context (from architecture diagram)

```
LLM Agent (REPL loop)
  │
  ├── subagent.retrieval(claim)   → PaperRecords (PMID, full text, summary)
  ├── subagent.summarise(paper)   → summary text
  │
  ├── sufficiency_classifier.classify(claim, papers)
  │     ├── Per-paper NLP extraction (SBERT + NLI + entity coverage)
  │     ├── Per-paper metadata extraction (year, IF, citations)
  │     ├── Aggregation → 10-dim vector
  │     ├── StandardScaler → z-scores
  │     └── MLP forward pass → P(sufficient)
  │           ├── Sufficient to support
  │           ├── Sufficient to refute
  │           └── Insufficient → subagent.find_gaps(papers)
  │
  └── emit_verdict() → final verdict
```

---

## 2. Feature Mapping: What the MLP Expects vs What We Have

The MLP takes exactly **10 features** (from `mlp_config.json`):

| # | Feature | Source | Current Status |
|---|---------|--------|----------------|
| 1 | `mean_similarity` | SBERT cosine sim (claim vs evidence), mean across papers | **MISSING** — need `SemanticSimilarityComputer` |
| 2 | `num_full_text` | Count of papers with `full_text` | **TRIVIAL** — `sum(1 for p in state.papers.values() if p.full_text)` |
| 3 | `year_span` | max(year) − min(year) across papers | **PARTIAL** — `PaperFeatureExtractor` gets year, need aggregation |
| 4 | `mean_entity_coverage` | Mean entity recall (claim ∩ evidence / claim), across papers | **PARTIAL** — `compute_entity_overlap()` exists in `feature_extractor.py` |
| 5 | `controversy_index` | Shannon entropy of argmax NLI stance distribution | **MISSING** — need NLI scores per paper |
| 6 | `weighted_entailment_temporal` | Σ(e^{−λ·age} · P_entailment) | **MISSING** — needs NLI + pub year |
| 7 | `mean_norm_citation` | Mean of (citations / age) | **PARTIAL** — `PaperFeatureExtractor` computes per-paper |
| 8 | `mean_log_IF` | Mean of log(1 + IF) | **PARTIAL** — same |
| 9 | `contradiction_ratio` | Fraction of papers where argmax(NLI) = Contradiction | **MISSING** — needs NLI |
| 10 | `num_papers` | Number of papers | **TRIVIAL** — `len(state.papers)` |

**Summary**: 2 trivial, 3 partial (metadata extractable, need aggregation), 5 blocked on SBERT + NLI models.

---

## 3. Data Flow

```
check_sufficiency(state: EvidenceState)
    │
    ▼
MLPSufficiencyClassifier.__call__(state)
    │
    ├─ 1. Per-paper NLP extraction (lazy, cached on PaperRecord)
    │     ├── SemanticSimilarityComputer.compute(claim, text) → cosine sim
    │     ├── NLIEntailmentComputer.compute(claim, text)      → {ent, con, neu}
    │     └── compute_entity_coverage(claim, text)            → coverage float
    │
    ├─ 2. Per-paper metadata extraction (PaperFeatureExtractor, disk-cached)
    │     └── extract_metadata(pmid) → {year, log_IF, norm_citation}
    │
    ├─ 3. Aggregation (FeatureAggregator → 10-dim global vector)
    │     ├── Metadata: mean_log_IF, mean_norm_citation, year_span, num_papers, num_full_text
    │     ├── NLP: mean_similarity, mean_entity_coverage
    │     └── Cross: controversy_index, contradiction_ratio, weighted_entailment_temporal
    │
    ├─ 4. StandardScaler normalization (embedded mean/scale from mlp_config.json)
    │
    └─ 5. MLP forward pass → sigmoid → P(sufficient)
           ├── P ≥ 0.5 → SUFFICIENT_{SUPPORT|REFUTE} (determined by fact majority)
           └── P < 0.5 → INSUFFICIENT (with heuristic gap identification)
```

---

## 4. Files to Create / Modify

### 4A. CREATE: `src/pkevolve/verification/nlp_tools.py`

Port from colleague's `nlp_tools.py` + `extract_features_scifact.py`.

**Classes**:

- **`SemanticSimilarityComputer`** — wraps `sentence-transformers` `all-MiniLM-L6-v2`
  - `compute(claim: str, evidence: str) -> float` — max cosine sim over
    256-word overlapping chunks (stride = 256 − 64 = 192 words)
  - Colleague's implementation: `nlp_tools.py` lines 88–131
  - Returns `float(similarities.max().item())`

- **`NLIEntailmentComputer`** — wraps `cross-encoder/nli-deberta-v3-large`
  - `compute(claim: str, evidence: str) -> dict[str, float]` with keys
    `nli_entailment`, `nli_contradiction`, `nli_neutral`
  - Same chunking strategy; uses `id2label` from model config for correct
    entailment/contradiction/neutral index mapping
  - Colleague's implementation: `extract_features_scifact.py` lines 150–222
  - Default label indices: entailment=1, contradiction=0, neutral=2

Both use **lazy singleton** loading (model loaded on first `.compute()` call).

**NOT ported**: `BiomedicalEntityExtractor` (MCP-based, requires separate
Python 3.10 env). Entity coverage uses our existing scispaCy-based
`compute_entity_overlap()` in `feature_extractor.py` instead.

### 4B. CREATE: `src/pkevolve/verification/feature_aggregation.py`

Port `FeatureAggregator` from colleague's `scripts/sufficiency_classifier/feature_aggregation.py` (467 lines), **stripped down** to only the functions that produce the 10 MLP features.

**Class**: `FeatureAggregator`

```python
class FeatureAggregator:
    """Aggregate per-paper features into the 10-dim MLP input vector."""

    def __init__(self, lambda_decay: float = 0.1, current_year: int | None = None): ...

    def aggregate(self, state: EvidenceState, claim: str) -> dict[str, float]:
        """Return dict of all 10 aggregated features."""
        ...

    def build_feature_vector(self, agg: dict, feature_names: list[str]) -> list[float]:
        """Assemble features in the exact order the MLP expects.
        Raises ValueError if any feature is missing."""
        ...
```

**Aggregation formulas** (from colleague's code, Section 2 of `classifier_plan.md`):

| Feature | Formula |
|---------|---------|
| `num_papers` | `len(papers)` |
| `num_full_text` | `sum(1 for p if p.full_text)` |
| `year_span` | `max(years) - min(years)` |
| `mean_log_IF` | `mean([log(1+IF_i)])` |
| `mean_norm_citation` | `mean([cit_i / age_i])` |
| `mean_similarity` | `mean([sbert_sim_i])` |
| `mean_entity_coverage` | `mean([coverage_i])` |
| `controversy_index` | `-Σ p_k · log₂(p_k)` where `p_k` = fraction of papers with argmax stance k ∈ {ent, con, neu} |
| `contradiction_ratio` | `count(argmax=contradiction) / N` |
| `weighted_entailment_temporal` | `Σ e^{-0.1·age_i} · P_entailment_i` |

**Input change**: Colleague's code takes `list[dict]` with `metadata_features` and `nlp_features` sub-dicts. Our version takes `EvidenceState` directly and reads cached features from `PaperRecord.metadata_features` / `PaperRecord.nlp_features`.

### 4C. MODIFY: `src/pkevolve/verification/data_models.py`

Extend `NLPFeatureVector` with SBERT and NLI fields:

```python
class NLPFeatureVector(BaseModel):
    entity_overlap_ratio: Optional[float] = None
    claim_entity_coverage: Optional[float] = None
    claim_entities: list[str] = Field(default_factory=list)
    evidence_entities: list[str] = Field(default_factory=list)
    # --- NEW: SBERT & NLI ---
    semantic_similarity: Optional[float] = None       # max cosine sim over chunks
    nli_entailment: Optional[float] = None            # max P(entailment)
    nli_contradiction: Optional[float] = None         # max P(contradiction)
    nli_neutral: Optional[float] = None               # max P(neutral)
```

Add per-paper feature cache fields to `PaperRecord`:

```python
class PaperRecord(BaseModel):
    ...
    metadata_features: Optional[PaperFeatureVector] = None  # cached per-paper metadata
    nlp_features: Optional[NLPFeatureVector] = None         # cached per-paper NLP
```

### 4D. MODIFY: `src/pkevolve/verification/classifier.py`

Add `MLPSufficiencyClassifier` alongside the existing heuristic. Existing
`SufficiencyClassifier` stays unchanged.

**MLP architecture** (from colleague's training code):

```python
class SufficiencyMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),       # 10 → 64
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), # 64 → 32
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),          # 32 → 1
        )
    def forward(self, x):
        return self.net(x)
```

**Classifier class**:

```python
class MLPSufficiencyClassifier:
    """MLP-based sufficiency classifier.

    Loads a trained checkpoint and runs inference on aggregated
    evidence features. Same __call__ interface as the heuristic.
    """

    def __init__(self, config_path: Path):
        # Load mlp_config.json → feature_names, scaler params, architecture
        # Load best_model.pth from same directory
        # Build SufficiencyMLP(input_dim, hidden_dim) and load state dict
        # Set to eval mode
        ...

    def __call__(self, state: EvidenceState) -> SufficiencyResult:
        # 1. Extract per-paper features (metadata + NLP) with caching
        # 2. Aggregate → dict of 10 features
        # 3. Build ordered vector, scale with embedded mean/scale
        # 4. Forward pass → sigmoid → P(sufficient)
        # 5. Threshold → SUFFICIENT_{SUPPORT|REFUTE} or INSUFFICIENT
        # 6. If INSUFFICIENT: delegate gap identification to heuristic
        ...
```

**Gap identification on INSUFFICIENT**: Reuse `SufficiencyClassifier._identify_gaps()` as a static/standalone function. The MLP only predicts sufficient/insufficient — gap types still use heuristic logic.

### 4E. MODIFY: `src/pkevolve/verification/config.py`

Add classifier settings to `VerificationSettings`:

```python
class ClassifierSettings(BaseSettings):
    use_mlp: bool = Field(
        default=False,
        description="Use trained MLP classifier instead of heuristic."
    )
    mlp_config_path: Optional[Path] = Field(
        default=None,
        description="Path to mlp_config.json (architecture, scaler, feature names)."
    )
    sufficiency_threshold: float = Field(
        default=0.5,
        description="P(sufficient) threshold for SUFFICIENT verdict."
    )
```

Add to `VerificationSettings`:
```python
class VerificationSettings(BaseSettings):
    ...
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
```

### 4F. MODIFY: `src/pkevolve/verification/evidence_api.py`

Change the module-level singleton to respect config:

```python
_classifier = None  # lazy init

def _get_classifier():
    from pkevolve.verification.config import get_settings
    cfg = get_settings()
    if cfg.classifier.use_mlp and cfg.classifier.mlp_config_path:
        from pkevolve.verification.classifier import MLPSufficiencyClassifier
        return MLPSufficiencyClassifier(cfg.classifier.mlp_config_path)
    return SufficiencyClassifier()

def check_sufficiency(state: EvidenceState) -> SufficiencyResult:
    global _classifier
    if _classifier is None:
        _classifier = _get_classifier()
    ...
```

### 4G. ADD DEPENDENCIES: `pyproject.toml`

```toml
[project.optional-dependencies]
classifier = [
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
]
```

(`torch` is already a dependency via `torch_geometric`; `sentence-transformers`
adds SBERT + cross-encoder support.)

---

## 5. Checkpoint Decision

| Checkpoint | Location | Accuracy | F1 | Training Set | Notes |
|---|---|---|---|---|---|
| `classifier_best/` | `rain/.../results/models/classifier_best/` | 91.8% | 92.0% | Balanced (4 pool types) | **Recommended** — better generalisation |
| `tau_1.00_seed_1011/` | `rain/.../results/ablation/models/tau_1.00_seed_1011/` | 97.9% | 98.6% | 3:1 imbalanced (tau=1.0) | Higher metrics, may overfit to "sufficient" |

**Recommendation**: Ship with `classifier_best/` as default. Path is configurable
via `mlp_config_path` so any checkpoint can be swapped in.

**Files to copy** (from colleague's workspace):
- `mlp_config.json` (feature names, scaler params, architecture)  
- `best_model.pth` (18.9 KB state dict)

**Destination**: `data/models/mlp_classifier/` in our repo.

---

## 6. Implementation Phases

| Phase | What | Files | Estimated Risk |
|---|---|---|---|
| **Phase 1** | Copy checkpoint files to repo | `data/models/mlp_classifier/` | None |
| **Phase 2** | Port NLP tools (SBERT + NLI) | Create `nlp_tools.py`, extend `data_models.py` | Medium — GPU/model loading latency |
| **Phase 3** | Port feature aggregation | Create `feature_aggregation.py` | Low — pure NumPy math |
| **Phase 4** | Build MLP classifier class | Modify `classifier.py` | Low — straightforward PyTorch |
| **Phase 5** | Wire config + evidence_api | Modify `config.py`, `evidence_api.py` | Low — plumbing |
| **Phase 6** | End-to-end integration test | Test script or notebook | Medium — feature order must match training |

---

## 7. Risks & Mitigations

### 7.1 Model Loading Latency

SBERT (`all-MiniLM-L6-v2`, ~90 MB) + NLI DeBERTa-v3-large (~1.3 GB) on first call.

**Mitigation**: Lazy singleton loading. Pre-warm in `evidence_programming.py`
before the REPL loop starts. On GPU nodes the load is ~3s; on CPU ~8s.

### 7.2 Feature Order Mismatch

The MLP expects features in the exact order of `expected_features` from
`mlp_config.json`. A transposition would silently produce wrong predictions.

**Mitigation**: `build_feature_vector()` reads `expected_features` from config
and assembles the vector in that order. An assertion checks that all 10 features
are present and non-None before inference.

### 7.3 Missing Metadata for Some Papers

API failures (OpenAlex, PubMed) → `None` values in `PaperFeatureVector`.

**Mitigation**: The aggregator filters for papers with valid features. If fewer
than 2 papers have metadata, fall back to the heuristic classifier for that
iteration and log a warning.

### 7.4 Entity Extractor Differences

Colleague's training used MCP-based scispaCy BioNER via a separate Python 3.10
env (`BiomedicalEntityExtractor` in `nlp_tools.py`). Our `feature_extractor.py`
uses direct scispaCy `en_core_sci_sm` via `compute_entity_overlap()`.

**Mitigation**: Both produce biomedical NER entity sets with the same recall
calculation. Acceptable distributional shift — entity coverage is not the
strongest signal (SBERT similarity has Cohen's d=1.27, entity coverage d=0.74).

### 7.5 No GPU on Login Nodes

SBERT and NLI inference on CPU may be slow for many papers.

**Mitigation**: Chunk-level max-pooling keeps computation bounded. Typical
evidence pool is 5–15 papers. On CPU, per-paper NLI takes ~2s (DeBERTa) and
SBERT ~0.5s. For a 10-paper pool: ~25s total, acceptable for an iterative loop.

### 7.6 Distributional Shift (SciFact → SIGNOR)

The MLP was trained on SciFact claims (biomedical fact verification). SIGNOR
edges are (gene, gene, interaction type) triples, which may have different
evidence characteristics.

**Mitigation**: The 10 features are domain-agnostic (similarity, citation
metrics, NLI scores). Monitor calibration on first SIGNOR runs. If accuracy
degrades, fine-tune on SIGNOR-specific data using the same pipeline.

---

## 8. Source File Reference (Colleague's Workspace)

All files in `/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct/`:

| File | Lines | Key Contents |
|---|---|---|
| `scripts/sufficiency_classifier/feature_aggregation.py` | 467 | `FeatureAggregator` class — metadata/NLP/cross-feature aggregation |
| `scripts/sufficiency_classifier/extract_features_scifact.py` | ~650 | End-to-end SciFact extraction pipeline; `SemanticSimilarityComputer`, `NLIEntailmentComputer`, Claude SDK PubMed retrieval |
| `src/pkevolve/verification/nlp_tools.py` | ~135 | Library-quality SBERT + `BiomedicalEntityExtractor` (MCP-based) |
| `src/pkevolve/verification/feature_extractor.py` | ~340 | Same as ours + OpenAlex author h-index helpers |
| `results/models/classifier_best/mlp_config.json` | — | 10 features, scaler params, architecture |
| `results/models/classifier_best/best_model.pth` | 18.9 KB | Trained state dict |
| `results/ablation/models/tau_1.00_seed_1011/` | — | Latest ablation checkpoint (97.9% acc) |
