# `tau_ablation/` — τ-Threshold Ablation Pipeline

Scripts for the τ (tau) threshold ablation study on the sufficiency classifier.

## Pipeline Overview

Run these steps in order, from the **project root**:

### Step 0 — Generate base training data (with labels)

```bash
uv run scripts/sufficiency_classifier/generate_classifier_data.py \
    --output data/classifier_train_data.json \
    --seed 42 --max_combos_per_claim 30 --noise_paper_counts 1 2 3
```

### Step 1 — Generate all τ dataset variants

```bash
uv run scripts/sufficiency_classifier/tau_ablation/generate_ablation_datasets.py \
    --input  data/classifier_train_data.json \
    --output_dir data/ablation \
    --thresholds 0.00 0.11 0.13 0.14 0.17 0.20 0.25 0.29 0.33 0.38 0.40 0.43 0.50
```

This calls `relabel_by_threshold.py` internally for each τ value.

### Step 2 — Create shared unambiguous test set

```bash
uv run scripts/sufficiency_classifier/tau_ablation/create_unambiguous_test_set.py \
    --ablation_dir data/ablation \
    --output_dir data/ablation/splits
```

### Step 3 — Train models (multi-seed, recommended)

```bash
uv run scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models \
    --seeds 42 123 456 789 1011
```

### Step 4 — Analyze results

```bash
uv run scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py \
    --models_dir results/ablation/models \
    --splits_dir data/ablation/splits \
    --test_data  data/ablation/splits/tau_0.00_test.json \
    --output_dir results/ablation/analysis
```

## Files

| Script | Role |
|---|---|
| `relabel_by_threshold.py` | Relabel a dataset for a given τ |
| `generate_ablation_datasets.py` | Batch-generate all τ variants |
| `create_unambiguous_test_set.py` | Build shared test split |
| `train_ablation_models.py` | Single-seed training (legacy) |
| `train_ablation_multiseed.py` | Multi-seed training (recommended) |
| `analyze_ablation.py` | Evaluate models and generate report |
