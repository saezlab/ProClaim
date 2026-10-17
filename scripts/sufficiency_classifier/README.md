# Sufficiency judge: training pipeline

Trains the MLP that predicts whether a pool of evidence papers is sufficient to verify a claim
(y=1) or more retrieval is needed (y=0). Training data come from SciFact-Open; place
`claims.jsonl` and `corpus.jsonl` under `data/scifact-open/`.

```
resolve_pmids.py → fetch_full_texts.py → generate_classifier_data.py → train_mlp_classifier.py
```

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `resolve_pmids.py` | Map SciFact doc_ids to PubMed PMIDs (cached) |
| 2 | `fetch_full_texts.py` | Fetch full texts from PubMed Central |
| 3 | `generate_classifier_data.py` | Build evidence pools, extract features, aggregate, write JSON |
| 4 | `train_mlp_classifier.py` | Train the MLP on the aggregated features |

```bash
uv run scripts/sufficiency_classifier/resolve_pmids.py --all-corpus
uv run scripts/sufficiency_classifier/fetch_full_texts.py --skip-existing
uv run scripts/sufficiency_classifier/generate_classifier_data.py --output data/classifier_train_data.json
uv run scripts/sufficiency_classifier/train_mlp_classifier.py --data_path data/classifier_train_data.json
```

Steps 1–2 need `PUBMED_EMAIL` in `.env` (`PUBMED_API_KEY` optional). Step 3 uses the scispaCy NER
environment built by `scripts/setup_proclaim.sh`. The τ ablation is in [tau_ablation/](tau_ablation/README.md).

Supporting modules: `feature_extractor.py` (paper metadata), `nlp_tools.py` (similarity, NLI,
entities), `feature_aggregation.py` (pool-level features, also used at inference), `inference.py`
(model loading, used by the verifier).

## Evidence pools

| Pool type | y | Content |
|-----------|---|---------|
| `positive_support` | 1 | Unanimous SUPPORT papers (sub-pools of size 1..N) |
| `positive_contradict` | 1 | Unanimous CONTRADICT papers (sub-pools of size 1..N) |
| `negative_conflict` | 0 | Mixed SUPPORT + CONTRADICT papers |
| `negative_noise` | 0 | Claims without evidence, plus random unrelated papers |

Evidence text is the paper's full text when available (over 1,000 characters), else its abstract.

## Model

Input: the 10 aggregated features listed in `results/ablation/models/tau_0.50_seed_42/mlp_config.json`
(`mean_similarity`, `num_full_text`, `year_span`, `mean_entity_coverage`, `controversy_index`,
`weighted_entailment_temporal`, `mean_norm_citation`, `mean_log_IF`, `contradiction_ratio`,
`num_papers`), standardised. The network outputs a logit; the verifier applies a sigmoid and
compares it with the sufficiency threshold in the config.
