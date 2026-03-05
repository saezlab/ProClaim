# τ-Threshold Ablation Study: Implementation Guide

**Status:** ✅ Phases 1-4 Complete (Ready for Training)
**Date:** 2026-03-03
**Author:** Claude Code

---

## Overview

This document describes the complete implementation of the τ-threshold ablation study for the sufficiency classifier. The study investigates how different conflict ratio thresholds (τ) for relabeling ambiguous evidence pools affect model performance.

## Research Question

**Does relabeling conflicting evidence pools based on their support/contradict ratio improve classifier performance?**

- **τ = 0.0:** Keep ALL conflicts as negative (most conservative - trusts all conflict labels)
- **τ = 0.5:** Relabel most conflicts to positive, keep only balanced (1S:1C) as negative (most aggressive relabeling)

## Implementation Status

### ✅ Phase 1: Dataset Generation

**1.1 Relabeling Function** ([relabel_by_threshold.py](../scripts/sufficiency_classifier/tau_ablation/relabel_by_threshold.py))
- Implements the relabeling logic using ground truth SciFact labels
- For each `negative_conflict` sample:
  - Compute `r_minority = min(N_support, N_contradict) / (N_support + N_contradict)`
  - If `r_minority >= τ`: keep as `negative_conflict` (label 0)
  - If `r_minority < τ`: relabel to `positive_support` or `positive_contradict` (label 1)
- Handles edge cases (missing labels, equal support/contradict)

**Usage:**
```bash
python scripts/sufficiency_classifier/tau_ablation/relabel_by_threshold.py \
    --input data/classifier_train_data_with_labels.json \
    --output data/ablation/tau_0.30.json \
    --threshold 0.30 \
    --verbose
```

**1.2 Dataset Generation Script** ([generate_ablation_datasets.py](../scripts/sufficiency_classifier/tau_ablation/generate_ablation_datasets.py))
- Generates 6 dataset variants for τ ∈ {0.00, 0.16, 0.20, 0.25, 0.33, 0.50}
- Calls `relabel_by_threshold.py` for each threshold
- Produces summary statistics and CSV report

**Usage:**
```bash
python scripts/sufficiency_classifier/tau_ablation/generate_ablation_datasets.py \
    --input data/classifier_train_data_with_labels.json \
    --output_dir data/ablation
```

**Output:**
```
data/ablation/
├── tau_0.00.json  (727 samples: 378 pos, 349 neg, ratio 1.08)
├── tau_0.16.json  (727 samples: 415 pos, 312 neg, ratio 1.33)
├── tau_0.20.json  (727 samples: 422 pos, 305 neg, ratio 1.38)
├── tau_0.25.json  (727 samples: 430 pos, 297 neg, ratio 1.45)
├── tau_0.33.json  (727 samples: 452 pos, 275 neg, ratio 1.64)
├── tau_0.50.json  (727 samples: 502 pos, 225 neg, ratio 2.23)
├── dataset_summary.json
└── dataset_summary.csv
```

**Key Finding:** As τ increases, more `negative_conflict` samples are relabeled as positive, creating increasing class imbalance.

---

### ✅ Phase 2: Training Setup

**2.1 Training Script** ([train_mlp_classifier.py](../scripts/sufficiency_classifier/train_mlp_classifier.py))
- Already supports necessary CLI arguments:
  - `--data_path`: Input training data
  - `--output_dir`: Model save directory
  - `--features`: Feature preset (all/important/minimal)
  - `--epochs`, `--batch_size`, `--lr`, `--hidden_dim`: Hyperparameters

**2.2 Unambiguous Test Set** ([create_unambiguous_test_set.py](../scripts/sufficiency_classifier/tau_ablation/create_unambiguous_test_set.py))
- Identifies 603 unambiguous samples (82.9%) whose labels don't change across τ values
- Creates stratified train/test splits:
  - **Test:** 108 samples (14.9%) - all unambiguous
  - **Train:** 619 samples (85.1%) - mix of 495 unambiguous + 124 ambiguous
- Test set is IDENTICAL across all τ values for fair comparison

**Usage:**
```bash
python scripts/sufficiency_classifier/tau_ablation/create_unambiguous_test_set.py \
    --ablation_dir data/ablation \
    --output_dir data/ablation/splits \
    --test_size 0.1808 \
    --seed 42
```

**Output:**
```
data/ablation/splits/
├── tau_0.00_train.json  (619 samples)
├── tau_0.00_test.json   (108 samples)
├── tau_0.16_train.json  (619 samples)
├── tau_0.16_test.json   (108 samples)
├── tau_0.20_train.json  (619 samples)
├── tau_0.20_test.json   (108 samples)
├── tau_0.25_train.json  (619 samples)
├── tau_0.25_test.json   (108 samples)
├── tau_0.33_train.json  (619 samples)
├── tau_0.33_test.json   (108 samples)
├── tau_0.50_train.json  (619 samples)
├── tau_0.50_test.json   (108 samples)
└── test_set_info.json   (metadata)
```

**Train Set Composition:**
```
τ=0.00: 310 pos (50.1%) | 309 neg (49.9%) - balanced
τ=0.16: 347 pos (56.1%) | 272 neg (43.9%)
τ=0.20: 354 pos (57.2%) | 265 neg (42.8%)
τ=0.25: 362 pos (58.5%) | 257 neg (41.5%)
τ=0.33: 384 pos (62.0%) | 235 neg (38.0%)
τ=0.50: 434 pos (70.1%) | 185 neg (29.9%) - imbalanced
```

**2.3 Training Automation** ([train_ablation_models.py](../scripts/sufficiency_classifier/tau_ablation/train_ablation_models.py))
- Trains all 6 models with identical hyperparameters
- Logs training metrics and generates summary report
- Supports background execution with logging

**Usage:**
```bash
# Interactive
python scripts/sufficiency_classifier/tau_ablation/train_ablation_models.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models \
    --features important

# Background (recommended)
nohup python scripts/sufficiency_classifier/tau_ablation/train_ablation_models.py \
    > results/ablation/training.log 2>&1 &
```

---

### ✅ Phase 3: Analysis

**3.1 Analysis Script** ([analyze_ablation.py](../scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py))
- Loads trained models and evaluates on shared test set
- Computes detailed metrics (accuracy, F1, precision, recall)
- Generates comprehensive markdown report
- Identifies best-performing threshold

**Usage:**
```bash
python scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py \
    --models_dir results/ablation/models \
    --test_data data/ablation/splits/tau_0.00_test.json \
    --output_dir results/ablation/analysis
```

**Output:**
```
results/ablation/analysis/
├── analysis_report.md      (comprehensive markdown report)
├── metrics_detailed.json   (detailed metrics)
└── confusion_matrices.json (confusion matrices)
```

---

## How to Run the Complete Study

### Step 1: Generate Datasets (✅ Complete)

```bash
# Already done - datasets are in data/ablation/
ls -lh data/ablation/*.json
```

### Step 2: Create Train/Test Splits (✅ Complete)

```bash
# Already done - splits are in data/ablation/splits/
ls -lh data/ablation/splits/
```

### Step 3: Train Models (⏳ USER ACTION REQUIRED)

```bash
# Run training (recommended: multi-seed, 18 models total)
python scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models \
    --seeds 42 123 456 \
    --features important \
    --epochs 100

# Or run in background:
nohup python scripts/sufficiency_classifier/tau_ablation/train_ablation_models.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models \
    --features important \
    --epochs 100 \
    > results/ablation/training.log 2>&1 &

# Monitor progress:
tail -f results/ablation/training.log
```

### Step 4: Analyze Results (⏳ After Training)

```bash
# Run analysis after training completes
python scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py \
    --models_dir results/ablation/models \
    --test_data data/ablation/splits/tau_0.00_test.json \
    --output_dir results/ablation/analysis

# View the report:
cat results/ablation/analysis/analysis_report.md
```

---

## Key Design Decisions

### 1. Ground Truth Labels vs NLI Predictions
- **Decision:** Use SciFact ground truth labels (SUPPORT/CONTRADICT) for computing conflict ratios
- **Rationale:** NLI predictions (`entailment_ratio`, `contradiction_ratio`) are model outputs and may be noisy
- **Implementation:** Labels are now embedded directly by `generate_classifier_data.py` (the former `augment_existing_data.py` is deprecated and removed)

### 2. Unambiguous Test Set
- **Decision:** Test only on samples whose labels don't change across τ values
- **Rationale:** Ensures fair comparison - performance differences due to training, not evaluation
- **Trade-off:** Smaller test set (89 vs 109 samples), but higher confidence in results

### 3. Stratified Splitting
- **Decision:** Use stratified sampling to maintain class balance in test set
- **Rationale:** Prevents bias toward majority class, especially important as τ increases
- **Implementation:** 62.9% positive, 37.1% negative in test set

### 4. Identical Hyperparameters
- **Decision:** Train all models with same architecture and hyperparameters
- **Rationale:** Isolates effect of τ from confounding factors (architecture, learning rate, etc.)
- **Parameters:** 10 features (important preset), 64 hidden dim, AdamW optimizer, early stopping

---

## Expected Outcomes

### Hypotheses

**H1:** Higher τ values (more aggressive relabeling) will improve test performance
- **Rationale:** Conflicting evidence may represent labeling noise or edge cases
- **Prediction:** τ=0.25 or τ=0.33 will perform best

**H2:** Class imbalance will hurt performance at very high τ values
- **Rationale:** τ=0.5 creates 70/30 imbalance, may bias model toward positive class
- **Prediction:** τ=0.20 or τ=0.25 will balance noise removal and class balance

**H3:** Precision and recall will trade off across τ values
- **Rationale:** More positive training samples → higher recall, lower precision
- **Prediction:** Low τ → high precision, low recall; High τ → low precision, high recall

### Metrics to Compare

1. **Test Accuracy:** Overall correctness on unambiguous test set
2. **F1 Score:** Harmonic mean of precision and recall (primary metric)
3. **Precision:** Fraction of predicted positives that are correct
4. **Recall:** Fraction of actual positives that are detected
5. **Training Stability:** Convergence speed, early stopping epoch

---

## Files Created

### Scripts (in `tau_ablation/` subfolder)
- [scripts/sufficiency_classifier/tau_ablation/relabel_by_threshold.py](../scripts/sufficiency_classifier/tau_ablation/relabel_by_threshold.py)
- [scripts/sufficiency_classifier/tau_ablation/generate_ablation_datasets.py](../scripts/sufficiency_classifier/tau_ablation/generate_ablation_datasets.py)
- [scripts/sufficiency_classifier/tau_ablation/create_unambiguous_test_set.py](../scripts/sufficiency_classifier/tau_ablation/create_unambiguous_test_set.py)
- [scripts/sufficiency_classifier/tau_ablation/train_ablation_models.py](../scripts/sufficiency_classifier/tau_ablation/train_ablation_models.py)
- [scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py](../scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py)

### Data
- `data/ablation/tau_{0.00,0.16,0.20,0.25,0.33,0.50}.json` (6 dataset variants)
- `data/ablation/dataset_summary.{json,csv}` (dataset statistics)
- `data/ablation/splits/tau_{tau}_{train,test}.json` (12 files, 638+89 samples each)
- `data/ablation/splits/test_set_info.json` (metadata)

### Documentation
- [doc/tau_ablation_action_plan.md](tau_ablation_action_plan.md) (original plan)
- [doc/tau_ablation_implementation.md](tau_ablation_implementation.md) (this file)

---

## Next Steps (Phase 4 - User Actions)

### 1. Train Models (Required)
```bash
python scripts/sufficiency_classifier/tau_ablation/train_ablation_models.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models \
    --features important \
    --epochs 100
```

**Expected Output:**
- 6 trained models in `results/ablation/models/tau_{tau}/`
- Training summary in `results/ablation/models/training_summary.{json,csv}`
- Training logs showing convergence

### 2. Analyze Results (After Training)
```bash
python scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py \
    --models_dir results/ablation/models \
    --test_data data/ablation/splits/tau_0.00_test.json \
    --output_dir results/ablation/analysis
```

**Expected Output:**
- Comprehensive markdown report: `results/ablation/analysis/analysis_report.md`
- Detailed metrics: `results/ablation/analysis/metrics_detailed.json`

### 3. Interpret Results
- Identify best-performing τ value
- Analyze trade-offs between τ values
- Check if hypotheses are supported
- Decide on production threshold

### 4. Iterate (Optional)
- Try additional τ values (e.g., 0.1, 0.35, 0.45)
- Experiment with different feature sets
- Test with different model architectures
- Cross-validate with multiple random seeds

---

## Troubleshooting

### Issue: "Training data not found"
**Solution:** Ensure you've run [generate_ablation_datasets.py](../scripts/sufficiency_classifier/tau_ablation/generate_ablation_datasets.py) first

### Issue: "No SciFact labels available"
**Solution:** Use `data/classifier_train_data_with_labels.json` as input (not `classifier_train_data.json`)

### Issue: "Module not found: torch"
**Solution:** Use `uv run python` instead of `python` to activate the virtual environment

### Issue: Training hangs or runs out of memory
**Solution:** Reduce `--batch_size` or run on a GPU node

---

## Summary

This implementation provides a complete, reproducible framework for studying the effect of τ-threshold relabeling on classifier performance. The key innovation is the **unambiguous test set**, which ensures fair comparison across all τ values.

**Current Status:** Phases 1-3 complete, ready for training (Phase 4).

**Next Action:** Run [train_ablation_multiseed.py](../scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py) to train all 18 models (6 τ × 3 seeds).

**Estimated Time:** 30-60 minutes for training, 5 minutes for analysis.
