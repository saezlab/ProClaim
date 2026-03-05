# Action Plan: τ-Threshold Ablation Study for Negative_Conflict Relabeling

**Date**: 2026-03-03  
**Objective**: Implement and validate ratio-based threshold relabeling for negative_conflict samples, with controlled experiments to disentangle label quality from class balance effects.

---

## Phase 1: Data Infrastructure

### 1.1 Implement Relabeling Function

Create `scripts/sufficiency_classifier/tau_ablation/relabel_by_threshold.py`:

- **Input**: `data/classifier_train_data.json`, threshold τ
- **Core logic**:
  - For each negative_conflict sample, compute `r_minority = min(N_sup, N_con) / (N_sup + N_con)`
  - If `r_minority >= τ` → keep as `negative_conflict` (label 0)
  - If `r_minority < τ` → relabel to `positive_support` or `positive_contradict` (label 1), based on majority stance
- **Output**: New JSON with updated `pool_type` and `label` fields
- **Logging**: Print per-sample relabeling decisions for traceability

### 1.2 Generate All Dataset Variants

Write a runner script `scripts/sufficiency_classifier/tau_ablation/generate_ablation_datasets.py`:

- τ values: `[0.10, 0.15, 0.20, 0.25, 0.50]`
- For each τ, produce one relabeled dataset
- Save to `data/ablation/tau_{value}.json`
- Also save a summary CSV: τ, negative_conflict count, positive count, total count, class ratio

---

## Phase 2: Training Pipeline

### 2.1 Training Script Modifications

Modify `scripts/sufficiency_classifier/train_classifier.py` (or equivalent):

- Accept `--data_path` argument (to swap in different ablation datasets)
- Accept `--run_name` argument (e.g., `tau_0.20_natural`)
- Log all metrics to a structured output (JSON or CSV), not just stdout
- Fix random seeds across all runs: data split, model init, training order

### 2.2 Test Set Construction

**Principle: Test set contains only unambiguous samples whose labels are invariant across all τ values.**

This avoids the circular problem of the test set implicitly assuming a specific τ.

**Unambiguous samples (include in test set):**

- `positive_support` → always label 1
- `positive_contradict` → always label 1
- `negative_noise` → always label 0
- `negative_conflict` with `r_minority >= 0.50` (i.e., 1S:1C) → always label 0

**Ambiguous samples (exclude from test set, training only):**

- `negative_conflict` with `r_minority < 0.50` (i.e., 2S:1C, 3S:1C, ... 23S:1C)
- These are exactly the samples whose label changes depending on τ

**Why this works:**

- The test set is neutral — it does not favor any τ
- What we measure is: "Does changing the training labels for ambiguous cases improve the model's ability to classify clear-cut cases?"
- If yes → the relabeling is genuinely helping the model learn better representations
- The test set still covers all four original pool types, so it tests the full classification space

**Implementation (order matters):**

1. Separate all data into **unambiguous** vs **ambiguous** pools
2. From unambiguous pool, allocate test set: ~15% of **total data** (~109 samples), stratified by pool type
3. Save test set → **FROZEN, never touch again**
4. Remaining data (unambiguous remainder + ALL ambiguous) = working pool (~618 samples)
5. Split working pool into train/val (e.g., 85/15, seed=42)
6. For each τ: apply relabeling to ambiguous samples in **both** train and val

```
All Data (727 samples)
│
├── Unambiguous (~534 samples)
│   │
│   ├── Test (~109) ← FROZEN, unambiguous only, ~15% of total
│   └── Remaining (~425) → working pool
│
└── Ambiguous (~193 samples)
    └── 100% → working pool

Working pool (~618 samples)
├── Train (~525, ~85%) — labels for ambiguous samples depend on τ
└── Val   (~93,  ~15%) — labels for ambiguous samples depend on τ
```

Only the test set needs to be unambiguous and frozen. Val mirrors the training distribution for each τ, which is correct for early stopping and hyperparameter selection.

### 2.3 Run All Training Jobs

```bash
for tau in 0.00 0.16 0.20 0.25 0.33 0.50; do
  python scripts/sufficiency_classifier/tau_ablation/train_ablation_multiseed.py \
    --data_path data/ablation/tau_${tau}.json \
    --run_name tau_${tau} \
    --seed 42 \
    --output_dir results/ablation/
done
```

---

## Phase 3: Evaluation & Analysis

### 3.1 Metrics to Collect (Per Run)

For each of the 5 runs (5 τ values), record:

| Metric | Description |
|--------|-------------|
| Overall Accuracy | Standard |
| Macro F1 | Primary comparison metric |
| Per-class Precision | Especially positive class (to detect pollution from bad relabels) |
| Per-class Recall | Especially negative_conflict (to detect loss of conflict detection) |
| Per-class F1 | For completeness |

### 3.2 Stratified Analysis on Test Set

Beyond overall metrics, break down performance by **conflict ratio bucket** on the test set:

| Bucket | r_minority Range | Example Patterns | What to Observe |
|--------|-----------------|------------------|-----------------|
| Balanced | 0.40 - 0.50 | 1S:1C, 2S:2C | Should be easy for all τ |
| Moderate | 0.20 - 0.39 | 2S:1C, 3S:2C | Transition zone |
| Imbalanced | 0.05 - 0.19 | 5S:1C, 8S:1C | Key differentiator |
| Extreme | < 0.05 | 23S:1C | Should these be "conflict"? |

For each bucket, report accuracy and confusion matrix. This directly answers: "Does the model's behavior on edge cases change with τ?"

### 3.3 Analysis Script

Create `scripts/sufficiency_classifier/tau_ablation/analyze_ablation.py`:

- Read all results from `results/ablation/`
- Generate comparison tables and plots
- Output to `results/ablation/ablation_report.md`

---

## Phase 4: Visualization for Paper

### 4.1 Main Figure

**Plot: τ vs. Performance Metrics**

- X-axis: τ
- Y-axis: Macro F1, negative_conflict Recall, positive Precision
- Expected pattern: inverted-U for Macro F1, monotonic decrease for recall, monotonic increase for precision
- This is the core result figure

### 4.2 Dataset Composition Table (Appendix)

| τ | neg_conflict | relabeled_to_pos | pos_total | neg_total | ratio |
|---|---|---|---|---|---|
| 0.50 | 158 | 0 | 378 | 349 | 1.08 |
| 0.25 | ~110 | ~48 | ~426 | ~301 | ~1.42 |
| ... | ... | ... | ... | ... | ... |

---

## Phase 5: Writing the Paper Section

### 5.1 Where This Goes

- **Section 3 (Methodology)**: Define τ, the relabeling rule, and the rationale (1 paragraph)
- **Section 4 (Experiments)**: Report main ablation results with the natural variant
- **Section 5 (Discussion)**: Interpret τ's domain meaning — what constitutes "genuine conflict" in scientific literature
- **Appendix**: Controlled experiment, full tables, dataset composition

### 5.2 Key Narrative

Frame τ as a principled design choice with domain interpretation:

> "The threshold τ operationalizes the question: at what point does minority dissent constitute genuine scientific conflict versus an outlier position against established consensus? Our ablation shows that τ ≈ 0.20 yields the best downstream performance, suggesting that evidence bodies with < 20% dissent are better characterized as consensus with noise."

---

## Implementation Timeline

| Step | Task | Dependency |
|------|------|------------|
| 1 | `tau_ablation/relabel_by_threshold.py` | None |
| 2 | `tau_ablation/generate_ablation_datasets.py` | Step 1 |
| 3 | Modify training script for CLI args | None |
| 4 | Construct unambiguous test set | Step 2 |
| 5 | Run all 5 training jobs (× 3 seeds) | Steps 2, 3, 4 |
| 6 | `analyze_ablation.py` | Step 5 |
| 7 | Generate plots | Step 6 |
| 8 | Write paper sections | Step 7 |

Steps 1 and 3 can be done in parallel.

---

## Potential Issues & Mitigations

**Issue: Test set contamination from relabeling**  
Mitigation: Test set only contains unambiguous samples (labels invariant across all τ). Ambiguous negative_conflict samples are training-only.

**Issue: Test set doesn't cover extreme cases (14S:1C etc.)**  
Response: This is by design. Extreme cases have no ground-truth label — including them would bias the evaluation toward a specific τ. The test measures whether relabeling ambiguous training data improves generalization on clear-cut cases.

**Issue: Reviewer asks about class balance confound**  
Response: Acknowledge that τ changes both labels and class distribution. Argue these are two symptoms of the same root cause (over-inclusive conflict definition). If reviewer insists, can add a controlled experiment with resampling as follow-up.

**Issue: Small sample sizes after filtering**  
Mitigation: Track sample counts per pool type. If any pool has < 5 test samples, use cross-validation instead of a single split.

**Issue: Unstable results across seeds**  
Mitigation: Run each configuration with 3 seeds (42, 123, 456), report mean ± std. If variance is high relative to differences between τ values, the effect is not robust.

**Issue: Reviewer asks "why not learn τ end-to-end?"**  
Response: τ defines the dataset, not the model. Learning it would require meta-learning over dataset construction, which is out of scope. The ablation provides empirical guidance for selection.