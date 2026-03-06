# Negative_Conflict Sample SUPPORT vs CONTRADICT Ratio Analysis

**Date**: 2026-02-27
**Dataset**: SciFact-Open negative_conflict samples
**Objective**: Understand the distribution of SUPPORT vs CONTRADICT papers in negative_conflict samples

---

## Executive Summary

The generated `classifier_train_data.json` currently contains 157 `negative_conflict` training samples, utilizing a **stratified sampling** approach across conflict ratios. The dataset exhibits the following characteristics:

- **Overall Ratio**: **63.2% SUPPORT vs 36.8% CONTRADICT** (Approx 1.7:1)
- **Most Common Patterns**: 1S:1C (25.5%), 2S:1C (13.4%), 1S:2C (12.1%)
- **Data Tail**: The distribution successfully captures complex blended cases (e.g., 3S:2C, 4S:2C, 5S:3C) effectively leveraging the stratification bucket system.
- **Top 3 Sizes**: Size 2 (25%), Size 3 (25%), Size 4 (16%)

---

## 1. Data Source and Methodology

### 1.1 Dataset Overview

- **Total Claims**: 279
- **Generated negative_conflict Combination Samples**: 157

### 1.2 Generation Logic

Negative_conflict samples are generated through the following process (see [`generate_classifier_data.py:377-394`](../scripts/sufficiency_classifier/generate_classifier_data.py#L377-L394)):

1. **Filtering**: Only claims containing both SUPPORT and CONTRADICT labels qualify.
2. **Combination Expansion Phase**: For a claim with N papers, generate *all* combinations containing at least 1 SUPPORT AND 1 CONTRADICT paper.
3. **Stratification & Capping**: Group combinations by their specific S:C ratio (e.g., 2S:1C, 3S:2C). Evenly allocate slots to each S:C bucket up to a maximum of 30 combinations per claim.
4. **Result**: A structured blending that ensures diverse representation across pure symmetric (e.g., 1S:1C), blended (e.g., 3S:2C), and highly asymmetric (e.g., 8S:1C) cases.

### 1.3 Analysis Tools

Analysis script (merged):
- [`analyze_conflict_ratios.py`](../scripts/sufficiency_classifier/analyze_conflict_ratios.py): Claim-level analysis by default; add `--detailed` for combination-level analysis

---
## 2. Key Findings (Generated Dataset)

Analysis of the 157 `negative_conflict` training samples:

### 2.1 Overall Ratio Statistics

| Metric | Value |
|--------|-------|
| Total SUPPORT papers (across all 157 combinations) | 395 |
| Total CONTRADICT papers (across all 157 combinations) | 230 |
| **Overall Ratio** | **395:230 ≈ 1.7:1** |
| **SUPPORT Percentage** | **63.2%** |
| **CONTRADICT Percentage** | **36.8%** |

*Note: This 1.7:1 ratio is driven by the fact that the underlying SciFact-Open dataset inherently contains more supporting than contradicting papers.*

### 2.2 Combination Size Distribution (number of papers per pool)

| Size | Samples | Percentage |
|------|---------|------------|
| 2 papers | 40 | 25.5% |
| 3 papers | 40 | 25.5% |
| 4 papers | 25 | 15.9% |
| 5 papers | 17 | 10.8% |
| 6 papers | 16 | 10.2% |
| 7 papers | 10 | 6.4% |
| 8 papers | 7 | 4.5% |
| 9 papers | 2 | 1.3% |

### 2.3 Detailed SUPPORT:CONTRADICT Ratio Patterns

Below is the complete distribution of all 21 ratio combinations present in the 157 `negative_conflict` training samples:

| Ratio | Samples | Percentage | Tau Ratio ($r_{minority}$) |
|-------|---------|------------|-----------------------------|
| **1S:1C** | 40 | 25.5% | **50.0%** (0.50) |
| **2S:1C** | 21 | 13.4% | **33.3%** (0.33) |
| **1S:2C** | 19 | 12.1% | **33.3%** (0.33) |
| **3S:1C** | 13 | 8.3% | **25.0%** (0.25) |
| **5S:1C** | 12 | 7.6% | **16.7%** (0.17) |
| **1S:3C** | 11 | 7.0% | **25.0%** (0.25) |
| **4S:1C** | 11 | 7.0% | **20.0%** (0.20) |
| **6S:1C** | 6 | 3.8% | **14.3%** (0.14) |
| **7S:1C** | 4 | 2.5% | **12.5%** (0.13) |
| 1S:4C | 2 | 1.3% | **20.0%** (0.20) |
| 3S:2C | 2 | 1.3% | **40.0%** (0.40) |
| 3S:3C | 2 | 1.3% | **50.0%** (0.50) |
| 4S:2C | 2 | 1.3% | **33.3%** (0.33) |
| 5S:2C | 2 | 1.3% | **28.6%** (0.29) |
| 2S:3C | 2 | 1.3% | **40.0%** (0.40) |
| 5S:3C | 2 | 1.3% | **37.5%** (0.38) |
| 4S:3C | 2 | 1.3% | **42.9%** (0.43) |
| 8S:1C | 1 | 0.6% | **11.1%** (0.11) |
| 6S:3C | 1 | 0.6% | **33.3%** (0.33) |
| 6S:2C | 1 | 0.6% | **25.0%** (0.25) |
| 2S:2C | 1 | 0.6% | **50.0%** (0.50) |

**Observations**:
- The perfectly balanced **1S:1C** remains the plurality pattern (25.5%).
- The top 3 core patterns (1S:1C, 2S:1C, 1S:2C) cover 51% of all negative conflict samples.
- The use of stratification explicitly captures complex blended ratios (like `3S:2C`, `5S:3C`) into the long tail of the distribution.

### 2.4 Tau Threshold Mapping (Sorted)

For ablation studies sweeping the threshold $\tau$, here is the sorted list of exact cutoffs present in this dataset. The value represents the exact **$\tau = r_{minority} = \min(N_S, N_C) / (N_S + N_C)$**.

- **0.11 (11.1%)**: `8S:1C`
- **0.13 (12.5%)**: `7S:1C`
- **0.14 (14.3%)**: `6S:1C`
- **0.17 (16.7%)**: `5S:1C`
- **0.20 (20.0%)**: `4S:1C`, `1S:4C`
- **0.25 (25.0%)**: `3S:1C`, `1S:3C`, `6S:2C`
- **0.29 (28.6%)**: `5S:2C`
- **0.33 (33.3%)**: `2S:1C`, `1S:2C`, `4S:2C`, `6S:3C`
- **0.38 (37.5%)**: `5S:3C`
- **0.40 (40.0%)**: `3S:2C`, `2S:3C`
- **0.43 (42.9%)**: `4S:3C`
- **0.50 (50.0%)**: `1S:1C`, `2S:2C`, `3S:3C` *(Perfectly Balanced)*

## 3. Overall Training Dataset Distribution

| Pool Type | Count | Percentage |
|-----------|-------|------------|
| **positive_support** | 194 | 26.7% |
| **positive_contradict** | 184 | 25.3% |
| **negative_noise** | 191 | 26.3% |
| **negative_conflict** | 157 | 21.6% |
| **Total** | **726** | **100%** |

**Observations**:
- The dataset remains perfectly balanced across all four pool types (roughly 25% each).

---

## 4. Conclusions & Ablation Setup

1. **Strategic Baseline**: The new logic bounds large asymmetric claims (e.g. 23S:1C) from overflowing the conflict sample pool. The 157 combinations represent a highly structured spectrum of S:C pairs.
2. **Ideal for Tau-Ablation**: Because the dataset contains distinct ratios spanning from 0.11 (`8S:1C`) up to 0.80 (`1S:4C`), and includes heavily mixed blends like `3S:2C`, sweeping `tau_threshold` from `0.10` to `0.50` will cleanly capture and flip these samples at predictable moments.

```bash
# Claim-level analysis (fast)
uv run scripts/sufficiency_classifier/analyze_conflict_ratios.py

# Claim + combination analysis (slower)
uv run scripts/sufficiency_classifier/analyze_conflict_ratios.py \
  --detailed \
  --max_combos 30 \
  --save_claims data/conflict_claims.json \
  --save_combos data/conflict_combos.json

# Check training data distribution
jq '[.[] | .pool_type] | group_by(.) | map({type: .[0], count: length})' \
  data/classifier_train_data.json
```

---

**Report Completed**
**Analyst**: Claude Code
**Data Version**: SciFact-Open (279 claims, 500K corpus)
