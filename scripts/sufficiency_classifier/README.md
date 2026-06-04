# Classifier Training Pipeline

Scripts for training a Sufficiency MLP Classifier that predicts whether a set of evidence papers is sufficient to verify a scientific claim.

## Pipeline Overview

```
resolve_pmids.py → fetch_full_texts.py → generate_classifier_data.py → train_mlp_classifier.py
```

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `resolve_pmids.py` | Map SciFact doc_ids → PubMed PMIDs (cached) |
| 2 | `fetch_full_texts.py` | Fetch full text via MCP (no Claude SDK needed) |
| 3 | `generate_classifier_data.py` | Build evidence pools → extract features → aggregate → output JSON |
| 4 | `train_mlp_classifier.py` | Train MLP binary classifier on aggregated features |

## Quick Start

```bash
# 1. Resolve PMIDs (one-time, uses NCBI Entrez API)
uv run scripts/sufficiency_classifier/resolve_pmids.py --all-corpus

# 2. Fetch full texts from PubMed Central
uv run scripts/sufficiency_classifier/fetch_full_texts.py --skip-existing

# 3. Generate training data
uv run scripts/sufficiency_classifier/generate_classifier_data.py --output data/classifier_train_data.json

# 4. Train classifier
uv run scripts/sufficiency_classifier/train_mlp_classifier.py --data_path data/classifier_train_data.json
```

## Evidence Pool Construction Logic

`generate_classifier_data.py` builds training samples from SciFact-Open claims:

```
                        Claim
                          │
                 ┌────────┴────────┐
            Has evidence?       No evidence (NEI)
                 │                    │
           Check labels         negative_noise (y=0)
                 │              (random papers)
      ┌──────────┼──────────┐
      │          │          │
 Unanimous    Conflicting   Evidence docs
 SUPPORT/     (S + C mix)   not in corpus
 CONTRADICT       │              │
      │      negative_conflict (skip)
 positive     (y=0)
 (y=1)
 sub-pools 1..N
```

> [!NOTE]
> SciFact-Open has only SUPPORT/CONTRADICT labels. The 73 NEI claims are those
> with **no evidence field at all**.

### Pool Types

| Pool Type | Label (y) | Content | Rationale |
|-----------|-----------|---------|-----------|
| `positive_support` | 1 | Unanimous SUPPORT papers (sub-pools size 1..N) | Sufficient evidence |
| `positive_contradict` | 1 | Unanimous CONTRADICT papers (sub-pools size 1..N) | Sufficient evidence |
| `negative_conflict` | 0 | Mixed SUPPORT + CONTRADICT | Conflicting → insufficient |
| `negative_noise` | 0 | NEI claims + random irrelevant papers | Unrelated → insufficient |

### `num_noise` Parameter

Default: `2`. Controls how many random papers are added per noise pool.
- `negative_noise`: exactly `num_noise` random papers
- `negative_mixed`: 1 real paper + `num_noise` random papers

## Full Text Strategy

Evidence text is selected with a fallback strategy:

1. **Full text** from `results/sufficiency_classifier/scifact_feature_extraction/doc_{id}/full_text.txt` (if > 1000 chars)
2. **Abstract** from SciFact corpus (fallback)

Full text generally produces stronger NLI signals (entailment/contradiction) compared to abstracts, which tend to be classified as neutral due to their general nature.

### `num_full_text` Feature

Counts how many papers in the pool had full text available. This is informative because:
- Full text → stronger NLI features → more reliable verification
- Abstract-only → weaker signals → less certain

## Feature Schema (24 features)

### Metadata (11)
| Feature | Description |
|---------|-------------|
| `num_papers` | Total papers in pool |
| `num_papers_with_metadata` | Papers with resolved PubMed metadata |
| `num_full_text` | Papers using full text (vs abstract) |
| `max_log_IF` / `mean_log_IF` | Log impact factor stats |
| `max_h_index` / `avg_max_h_index` | Author h-index stats |
| `max_norm_citation` / `mean_norm_citation` | Normalized citation count stats |
| `latest_year_age` | Age of newest paper (years) |
| `year_span` | Time span between oldest and newest paper |

### NLP (7)
| Feature | Description |
|---------|-------------|
| `entailment_ratio` | Proportion of papers where argmax(NLI) = entailment |
| `contradiction_ratio` | Proportion where argmax(NLI) = contradiction |
| `controversy_index` | Shannon entropy of stance distribution |
| `max_entity_coverage` / `mean_entity_coverage` | Biomedical entity overlap (claim vs evidence) |
| `max_similarity` / `mean_similarity` | SBERT cosine similarity stats |

### Cross-Features (6)
| Feature | Description |
|---------|-------------|
| `weighted_entailment_IF` / `weighted_contradiction_IF` | NLI × log impact factor |
| `weighted_entailment_citation` / `weighted_contradiction_citation` | NLI × normalized citations |
| `weighted_entailment_temporal` / `weighted_contradiction_temporal` | NLI × recency |

## Supporting Scripts

| Script | Purpose |
|--------|---------|
| `extract_features_scifact.py` | Original feature extraction using Claude SDK + MCP (for exploration) |
| `feature_aggregation.py` | Contains `FeatureAggregator` class used by `generate_classifier_data.py` |

## Environment Variables

Set in `.env` at project root:

```
PUBMED_EMAIL=your@email.com
PUBMED_API_KEY=your_key    # optional, increases rate limit
```

See [README_PUBMED.md](README_PUBMED.md) for MCP server details.


---

## Sufficiency Classifier (MLP)

> **Design Doc:** [`doc/classifier_plan.md`](doc/classifier_plan.md)

A binary MLP classifier that predicts whether the current evidence pool is **sufficient** (y=1) to verify a scientific claim, or if more retrieval is needed (y=0). Used as an **LLM Agent Tool** during iterative retrieval.

### Pipeline

```
1. Feature Extraction     → scripts/claude_sdk/extract_features_scifact.py
2. Data Generation        → scripts/claude_sdk/generate_classifier_data.py
3. Model Training         → scripts/claude_sdk/train_mlp_classifier.py
4. Inference / Testing    → scripts/claude_sdk/test_mlp_classifier.py
```

### Data (`data/classifier_train_data.json`)

| Pool Type | target_y | Description |
|-----------|----------|-------------|
| `positive_support` | 1 | All evidence unanimously SUPPORTS the claim |
| `positive_contradict` | 1 | All evidence unanimously CONTRADICTS the claim |
| `negative_conflict` | 0 | Mixed SUPPORT + CONTRADICT evidence (insufficient) |
| `negative_noise` | 0 | Injected irrelevant/random papers (insufficient) |

**Current stats:** 727 samples, 52% y=1 / 48% y=0 (well-balanced).

### Model Architecture

- **Input:** 24 aggregated features (metadata + NLP + cross-features), StandardScaler normalized
- **Network:** `Linear(24→64) → BN → ReLU → Dropout(0.2) → Linear(64→32) → BN → ReLU → Dropout(0.2) → Linear(32→1)`
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` (no Sigmoid in model — logits output)
- **Optimizer:** AdamW (weight_decay=1e-4) + ReduceLROnPlateau
- **Output:** `results/models/classifier/mlp_classifier_weights.pth` + `mlp_config.json`

### Feature List (24 features)

| Category | Features |
|----------|----------|
| **Metadata** (11) | `num_papers`, `num_papers_with_metadata`, `num_full_text`, `max_log_IF`, `mean_log_IF`, `max_h_index`, `avg_max_h_index`, `max_norm_citation`, `mean_norm_citation`, `latest_year_age`, `year_span` |
| **NLP** (7) | `entailment_ratio`, `contradiction_ratio`, `controversy_index`, `max_entity_coverage`, `mean_entity_coverage`, `max_similarity`, `mean_similarity` |
| **Cross** (6) | `weighted_entailment_IF`, `weighted_contradiction_IF`, `weighted_entailment_citation`, `weighted_contradiction_citation`, `weighted_entailment_temporal`, `weighted_contradiction_temporal` |

**Most discriminative features** (by Cohen's d): `mean_similarity` (d=1.27), `max_similarity` (d=1.00), `num_full_text` (d=0.84).

### Training Commands

```bash
# Generate training data
uv run scripts/claude_sdk/generate_classifier_data.py

# Train the model
uv run scripts/claude_sdk/train_mlp_classifier.py \
  --data_path data/classifier_train_data.json \
  --output_dir results/models/classifier \
  --epochs 100 --batch_size 32 --lr 1e-3

# Test on individual claims
uv run scripts/claude_sdk/test_mlp_classifier.py
```

### Important Notes

> ⚠️ **Feature format:** `classifier_train_data.json` stores features as **flat** keys under `"features"` (not nested under `metadata_aggregation/nlp_aggregation/cross_features`). The training script auto-detects both formats.

> ⚠️ **Logits output:** The model outputs raw logits (no Sigmoid). Use threshold `> 0.0` for binary prediction, or apply `torch.sigmoid()` on outputs to get probability scores.

### Model Selection Results

The following table summarizes the performance of different model architectures and feature subsets.

| Model | Feature Set | Number of Features | Accuracy | Precision | Recall | F1 Score |
|-------|-------------|--------------------|----------|-----------|--------|----------|
| Logistic Regression | All | 24 | 86.36% | 87.50% | 85.96% | 86.73% |
| MLP Small | All | 24 | 90.91% | 89.83% | 92.98% | 91.38% |
| MLP Large | All | 24 | 89.09% | 90.91% | 87.72% | 89.29% |
| Logistic Regression | Important | 10 | 71.82% | 74.07% | 70.18% | 72.07% |
| MLP Small | Important | 10 | 88.18% | 90.74% | 85.96% | 88.29% |
| **MLP Large (Best)**| **Important** | **10** | **91.82%** | **92.86%** | **91.23%** | **92.04%** |
| Logistic Regression | Minimal | 5 | 73.64% | 74.14% | 75.44% | 74.78% |
| MLP Small | Minimal | 5 | 82.73% | 82.76% | 84.21% | 83.48% |
| MLP Large | Minimal | 5 | 84.55% | 83.33% | 87.72% | 85.47% |