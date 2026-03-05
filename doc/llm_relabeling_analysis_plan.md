# LLM-Based Relabeling: Analysis & Refinement Plan

**Purpose**: Guide for AI agent to analyze results and refine the relabeling approach
**Date**: 2026-03-05

---

## Overview

After implementing the basic pipeline, analyze results to identify issues and refine the approach.

**This plan should be executed AFTER**: `llm_relabeling_implementation_plan.md` is complete.

---

## Analysis Phase 1: Pilot Study Review

**Goal**: Determine if pilot results are good enough to proceed with full relabeling.

### Inputs
- `data/llm_relabeling/pilot_results.json`
- `data/llm_relabeling/pilot_analysis.md`

### Analysis Tasks

1. **Vote Consistency Check**
   - Calculate % of samples with 3/3 agreement
   - Calculate % of samples with 2/3 agreement
   - Calculate % of samples with 1/3 disagreement
   - **Decision criteria**:
     - If (3/3 + 2/3) ≥ 70% → proceed to full relabeling
     - If (3/3 + 2/3) < 70% → prompt needs adjustment

2. **Reasoning Quality Review**
   - Manually read 10 sample reasonings from pilot_analysis.md
   - Check for:
     - Does LLM reference paper quality (journal, IF)?
     - Does LLM reference paper count/ratio?
     - Is reasoning logical and consistent?
     - Are there hallucinations or nonsensical statements?
   - **Decision criteria**:
     - If >8/10 reasonings are good → proceed
     - If ≤8/10 → prompt needs adjustment

3. **Relabeling Rate Check**
   - Calculate: % of pilot samples relabeled to y=1
   - Break down by ratio group (1:1, 2:1, ≥3:1)
   - **Decision criteria**:
     - Record the rate for reference
     - No hard cutoff (just for awareness)

4. **Context Sufficiency Check**
   - Check if LLM ever says "need more context" in reasonings
   - **Decision criteria**:
     - If >5 samples mention needing more info → consider adding full texts
     - Otherwise → abstracts are sufficient

### Outputs
- Analysis report summarizing the 4 checks above
- Decision: proceed to Phase 3 or adjust prompt

### If Prompt Adjustment Needed
1. Identify issues from reasoning review
2. User provides updated prompt template
3. Re-run Phase 2 with new prompt
4. Repeat analysis

---

## Analysis Phase 2: Full Relabeling Review

**Goal**: Validate full relabeling results before dataset reconstruction.

### Inputs
- `data/llm_relabeling/relabeling_results.json` (all 157 samples)
- `data/llm_relabeling/relabeling_summary.json`

### Analysis Tasks

1. **Sanity Checks**
   - Verify all 157 samples processed (no missing samples)
   - Check for API failures or parsing errors
   - Verify vote_agreement distribution (3/3, 2/3, 1/3)

2. **Relabeling Pattern Analysis**
   - For each ratio group (1:1, 2:1, 3:1, ≥4:1):
     - Calculate relabeling rate
     - Expected pattern: higher ratio → higher relabeling rate
   - **Flag anomalies**: e.g., if 1:1 cases get relabeled more than 3:1 cases

3. **Reasoning Consistency Spot Check**
   - Randomly sample 10 relabeled cases (y=0 → y=1)
   - Randomly sample 10 kept cases (stayed y=0)
   - Manually review reasonings for logical consistency

4. **Edge Case Review**
   - Identify samples with 1-1-1 vote disagreement
   - Manually review these cases
   - Check if it makes sense to label them as 0

### Outputs
- Analysis report with findings
- List of any anomalies or concerns
- Decision: proceed to Phase 4 or investigate issues

---

## Analysis Phase 3: Dataset Impact Assessment

**Goal**: Understand how relabeling changed the dataset.

### Inputs
- `data/llm_relabeling/dataset_comparison.json`
- `data/classifier_train_data_llm_relabeled.json`

### Analysis Tasks

1. **Class Balance Check**
   - Calculate final y=1 vs y=0 ratio
   - Compare with original (was 52% vs 48%)
   - **Concern threshold**: If y=1 > 70% or y=0 > 70% → severe imbalance

2. **Pool Type Distribution**
   - Check how many conflicts were relabeled to each type:
     - `negative_conflict` → `positive_support`
     - `negative_conflict` → `positive_contradict`
     - `negative_conflict` → stayed `negative_conflict`
   - Verify this makes sense based on n_support vs n_contradict

3. **Relabeling by Conflict Ratio**
   - Group original conflicts by r_minority
   - For each group, calculate:
     - % relabeled to y=1
     - % kept as y=0
   - Visualize as a plot (r_minority vs relabeling rate)

### Outputs
- Dataset impact report
- Recommendation: proceed to training or address imbalance

### If Severe Imbalance Detected
**Options**:
1. Accept imbalance and use class weights in training (already implemented)
2. Generate additional `negative_noise` samples to balance
3. Selectively relabel only strong majority cases (requires re-running Phase 3)

**Decision**: Discuss with user before proceeding

---

## Analysis Phase 4: Model Performance Comparison

**Goal**: Compare LLM-relabeled model with baseline.

### Inputs
- `results/models/classifier_llm_relabeled/` (new model)
- `results/models/classifier_best/` or tau_0.00 model (baseline)
- Test set (same for both models)

### Analysis Tasks

1. **Test Set Performance**
   - Evaluate both models on same test set
   - Compare:
     - Accuracy
     - F1 score
     - Precision
     - Recall
   - **Expected**: LLM-relabeled model should achieve F1 ≥ baseline

2. **Predictions on Conflict Cases**
   - Load all 157 original conflict samples
   - Get predictions from both models:
     - Baseline model predictions
     - LLM-relabeled model predictions
   - Compare:
     - Do predictions differ systematically?
     - Does LLM-relabeled model predict more y=1 for conflicts?

3. **Feature Importance Analysis**
   - Extract feature importance from both models
   - Compare top 5 features for each
   - Check if LLM-relabeled model:
     - Uses metadata features (IF, H-index) more?
     - Uses cross-features differently?

4. **Prediction Confidence Analysis**
   - For conflict cases, compare prediction probabilities:
     - Baseline: P(y=1) distribution
     - LLM-relabeled: P(y=1) distribution
   - Check if LLM-relabeled model is more confident

### Outputs
- Comprehensive comparison report
- Plots: feature importance, prediction distributions
- Recommendation: use LLM-relabeled model or revert to baseline

---

## Refinement Scenarios

### Scenario 1: Pilot consistency < 70%

**Root causes**:
- Prompt is ambiguous
- LLM is confused by metadata format
- Temperature too high (0.7 → try 0.3)

**Actions**:
1. Review sample reasonings to identify confusion
2. User provides clearer prompt
3. Re-run pilot with adjusted prompt/temperature
4. Repeat analysis

---

### Scenario 2: Severe class imbalance (>70% y=1)

**Root causes**:
- LLM is too lenient (labels most conflicts as sufficient)
- Prompt emphasizes "majority wins" too strongly

**Options**:
1. **Accept and use class weights** (simplest)
   - Already implemented in training
   - May work if imbalance < 75%

2. **Generate more negative_noise samples**
   - Add 100-200 noise samples
   - Rebalance to ~50-50

3. **Re-run with stricter prompt**
   - User provides more conservative prompt
   - Re-run Phase 3 with new prompt

**Decision**: Depends on severity and user preference

---

### Scenario 3: Model performance worse than baseline

**Root causes**:
- LLM labels are noisy or inconsistent
- Relabeling introduced errors
- Test set doesn't match new label distribution

**Actions**:
1. **Investigate low-confidence predictions**
   - Find samples where model is uncertain (P ≈ 0.5)
   - Check if these are relabeled conflicts
   - Review LLM reasonings for these samples

2. **Check for label flips that don't make sense**
   - Find samples relabeled from y=0 → y=1
   - But model predicts y=0 with high confidence
   - Manually review these cases

3. **If LLM labels are clearly wrong**:
   - Identify patterns in bad labels
   - Adjust prompt to fix these patterns
   - Re-run Phase 3

4. **If issue is fundamental**:
   - Consider reverting to baseline
   - Or try different LLM model

---

### Scenario 4: Relabeling patterns don't make sense

**Example**: 1:1 cases get relabeled more than 5:1 cases

**Actions**:
1. Manually review reasonings for anomalous cases
2. Check if prompt is being misinterpreted
3. Check if metadata is incorrect (wrong IF values, etc.)
4. If issue found:
   - Fix data preparation (Phase 1)
   - Or adjust prompt
   - Re-run from affected phase

---

## Decision Tree

```
Phase 1 Complete
    ↓
Pilot Analysis (Analysis Phase 1)
    ↓
    ├─ Consistency < 70%? → Adjust prompt → Re-run pilot
    ├─ Reasoning poor? → Adjust prompt → Re-run pilot
    └─ Good? → Proceed to Phase 3
        ↓
Full Relabeling Complete
    ↓
Full Relabeling Review (Analysis Phase 2)
    ↓
    ├─ Anomalies found? → Investigate → Fix → Re-run
    └─ Good? → Proceed to Phase 4
        ↓
Dataset Reconstruction Complete
    ↓
Dataset Impact Assessment (Analysis Phase 3)
    ↓
    ├─ Severe imbalance? → Discuss options → Take action
    └─ Acceptable? → Proceed to Phase 5
        ↓
Training Complete
    ↓
Model Performance Comparison (Analysis Phase 4)
    ↓
    ├─ Performance worse? → Investigate → Refine or revert
    └─ Performance good? → Deploy LLM-relabeled model ✓
```

---

## Key Metrics to Track

### Pilot Phase
- Vote consistency rate
- Reasoning quality score (subjective, /10)
- Relabeling rate by ratio group
- Context sufficiency (% needing more info)

### Full Relabeling Phase
- Total samples processed
- Vote agreement distribution
- Relabeling rate by ratio group
- Number of 1-1-1 disagreements

### Dataset Phase
- Final class balance (y=1 / y=0)
- Pool type distribution changes
- Number of conflicts relabeled vs kept

### Model Phase
- Test F1 (baseline vs LLM-relabeled)
- Feature importance differences
- Prediction confidence on conflicts

---

## Output Files for Analysis

All analysis scripts should generate markdown reports in:
```
results/llm_relabeling_analysis/
├── pilot_analysis_report.md
├── full_relabeling_review.md
├── dataset_impact_assessment.md
└── model_comparison_report.md
```

Each report should include:
- Summary of findings
- Tables and statistics
- Plots (if applicable)
- Recommendations for next steps

---

## Next Steps

1. After implementation completes, run Analysis Phase 1
2. Based on pilot results, decide to proceed or adjust
3. After full relabeling, run Analysis Phases 2-3
4. After training, run Analysis Phase 4
5. Make final decision: deploy, refine, or revert
