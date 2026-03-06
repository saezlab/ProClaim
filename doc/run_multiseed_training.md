# Multi-Seed Training Guide

## Overview

To distinguish **signal** (consistent effects of τ) from **noise** (random variation), we train models with 3 different random seeds for each τ value.

**Total models:** 6 τ values × 3 seeds = **18 models**

## Why Multiple Seeds?

Without multiple seeds, we can't tell if performance differences are due to:
- ✅ **Signal:** True effect of τ on model performance
- ❌ **Noise:** Random initialization, data shuffling, or other stochasticity

With 3 seeds, we can:
1. Compute **mean ± std** for each τ
2. Check if differences are **statistically meaningful**
3. Identify **robust trends** vs random fluctuations

## Quick Start

### Option 1: Run All 18 Models (Recommended)

```bash
# Run in background (takes ~2-3 hours on CPU, ~30-45 min on GPU)
nohup python scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models_multiseed \
    --seeds 42 123 456 \
    --features important \
    --epochs 100 \
    > results/ablation/training_multiseed.log 2>&1 &

# Monitor progress
tail -f results/ablation/training_multiseed.log
```

### Option 2: Test with Fewer Models First

```bash
# Quick test: 2 τ values × 2 seeds = 4 models (~15-20 min)
python scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models_test \
    --thresholds 0.0 0.5 \
    --seeds 42 123 \
    --features important \
    --epochs 50
```

## Expected Output

### Directory Structure

```
results/ablation/models_multiseed/
├── tau_0.00_seed_42/
│   ├── mlp_classifier_weights.pth
│   └── mlp_config.json
├── tau_0.00_seed_123/
│   ├── mlp_classifier_weights.pth
│   └── mlp_config.json
├── tau_0.00_seed_456/
│   └── ...
├── tau_0.20_seed_42/
│   └── ...
├── ... (18 model directories total)
├── training_metrics_all_seeds.json    (raw metrics for all models)
├── training_statistics.json           (mean ± std for each τ)
└── training_summary_multiseed.csv     (summary table)
```

### Summary Report

The script will print a table like:

```
================================================================================
MULTI-SEED ABLATION SUMMARY (Mean ± Std across seeds)
================================================================================
     τ | Seeds |     Accuracy       |      F1 Score      |     Precision      |       Recall
--------------------------------------------------------------------------------
  0.00 |     3 | 0.7500 ± 0.0123 | 0.7800 ± 0.0098 | 0.7600 ± 0.0150 | 0.8000 ± 0.0100
  0.16 |     3 | 0.7600 ± 0.0100 | 0.7900 ± 0.0080 | 0.7700 ± 0.0130 | 0.8050 ± 0.0090
  0.20 |     3 | 0.7650 ± 0.0089 | 0.7950 ± 0.0067 | 0.7800 ± 0.0120 | 0.8100 ± 0.0080
  0.25 |     3 | 0.7700 ± 0.0060 | 0.8000 ± 0.0050 | 0.7900 ± 0.0100 | 0.8200 ± 0.0070
  0.33 |     3 | 0.7800 ± 0.0045 | 0.8100 ± 0.0034 | 0.7950 ± 0.0090 | 0.8250 ± 0.0060
  0.50 |     3 | 0.7600 ± 0.0134 | 0.7900 ± 0.0112 | 0.7700 ± 0.0160 | 0.8100 ± 0.0105
================================================================================

✓ Best τ by mean F1: 0.30 (F1 = 0.8100 ± 0.0034)
```

## Interpreting Results

### 1. Check Standard Deviations

**Small std (< 0.01):** Robust result, performance is consistent across seeds
```
τ=0.33: 0.8100 ± 0.0034  ← Very stable!
```

**Large std (> 0.02):** Noisy result, high variability
```
τ=0.20: 0.7800 ± 0.0234  ← Unstable, hard to trust
```

### 2. Compare Effect Size vs Noise

Check if difference between τ values is larger than std:

```
τ=0.30: 0.8100 ± 0.0034
τ=0.40: 0.7750 ± 0.0110

Difference: 0.8100 - 0.7750 = 0.0350
Combined std: sqrt(0.0034² + 0.0110²) ≈ 0.0115

Effect size / noise: 0.0350 / 0.0115 ≈ 3.0
```

**Rule of thumb:**
- Ratio > 2: Likely a real effect (signal)
- Ratio < 1: Probably just noise

### 3. Look for Consistent Trends

✅ **Good (Signal):** All 3 seeds show same ranking
```
Seed 42:  τ=0.30 (0.81) > τ=0.20 (0.78) > τ=0.50 (0.76)
Seed 123: τ=0.30 (0.81) > τ=0.20 (0.79) > τ=0.50 (0.75)
Seed 456: τ=0.30 (0.82) > τ=0.20 (0.78) > τ=0.50 (0.77)
→ Consistent! τ=0.30 is genuinely better
```

❌ **Bad (Noise):** Ranking changes between seeds
```
Seed 42:  τ=0.30 (0.81) > τ=0.20 (0.78) > τ=0.50 (0.76)
Seed 123: τ=0.50 (0.80) > τ=0.30 (0.79) > τ=0.20 (0.77)
Seed 456: τ=0.20 (0.82) > τ=0.50 (0.78) > τ=0.30 (0.75)
→ Inconsistent! No clear winner, just noise
```

## Time Estimates

| Setup | Models | Epochs | CPU Time | GPU Time |
|-------|--------|--------|----------|----------|
| Quick test | 4 | 50 | ~20 min | ~5 min |
| Half run | 9 | 100 | ~90 min | ~20 min |
| **Full run** | **18** | **100** | **~2.5-3.5 hours** | **~35-55 min** |

## Troubleshooting

### Training is too slow
```bash
# Reduce epochs (trades off final performance)
--epochs 50

# Use fewer seeds for initial exploration
--seeds 42 123

# Train fewer τ values first
--thresholds 0.0 0.3 0.5
```

### Out of memory
```bash
# Reduce batch size
--batch_size 16

# Or train sequentially (not in parallel)
# The script already trains sequentially by default
```

### Check progress
```bash
# View log in real-time
tail -f results/ablation/training_multiseed.log

# Count completed models
ls results/ablation/models_multiseed/ | grep "tau_" | wc -l

# Check if any failed
grep -i "failed\|error" results/ablation/training_multiseed.log
```

## What's Next?

After training completes:

1. **Review the summary table** - Which τ has best mean F1?
2. **Check std values** - Are results stable across seeds?
3. **Look at individual seeds** - Do they agree on ranking?
4. **Run analysis script** (coming soon) - Generate detailed report with:
   - Statistical tests (t-tests, ANOVA)
   - Confidence intervals
   - Visualization plots
   - Recommendations

## Files Created

1. **[train_ablation_multiseed.py](../scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py)** - Main training script
2. **[train_mlp_classifier.py](../scripts/sufficiency_classifier/train_mlp_classifier.py)** - Modified to support `--seed` parameter
3. This guide

## Example Command (Copy-Paste Ready)

```bash
# Create output directory
mkdir -p results/ablation

# Run training (full 18 models)
nohup python scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py \
    --splits_dir data/ablation/splits \
    --output_dir results/ablation/models_multiseed \
    --seeds 42 123 456 \
    --thresholds 0.0 0.16 0.20 0.25 0.33 0.5 \
    --features important \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --hidden_dim 64 \
    > results/ablation/training_multiseed.log 2>&1 &

# Get the process ID
echo $!

# Monitor progress
tail -f results/ablation/training_multiseed.log

# When done, check results
cat results/ablation/models_multiseed/training_summary_multiseed.csv
```

---

**Ready to run!** 🚀

The script is fully implemented and tested. Just execute the command above to start training all 18 models.
