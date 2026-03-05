# Classifier Training Data Augmentation with SciFact Labels

## Overview

The classifier training data has been augmented to include ground truth labels from the original SciFact dataset. This enhancement allows for better analysis of the classifier's behavior and understanding of which types of evidence pools are being correctly classified.

## What Was Done

### 1. Modified `generate_classifier_data.py`

The script was updated to:

- **Track document IDs**: Added `doc_ids` list to track which papers are in each evidence pool
- **Include ground truth labels**: Added `scifact_labels` list containing the original SciFact labels ("SUPPORT", "CONTRADICT", "NEI")
- **Count labels**: Added `n_support`, `n_contradict`, `n_nei` fields with label counts
- **Fix combinatorial explosion**: Added reservoir sampling for claims with many evidence documents to prevent hanging on large combination sets

#### Key Changes:

**Changed function signature:**
```python
def build_evidence_pools(...) -> List[Tuple[Dict, List[Dict], int, str, List[str], List[str]]]:
    # Returns: (claim, papers, target_y, pool_type, doc_ids, labels)
```

**Added reservoir sampling:**
For claims with >10 evidence documents that have conflicting labels, the script now uses reservoir sampling to efficiently sample combinations instead of generating all possible combinations first.

**Updated output structure:**
Each sample now includes:
```json
{
  "claim_id": 145,
  "claim_text": "...",
  "target_y": 0,
  "pool_type": "negative_conflict",
  "num_papers": 2,
  "doc_ids": ["10582939", "1092058"],
  "scifact_labels": ["CONTRADICT", "SUPPORT"],
  "n_support": 1,
  "n_contradict": 1,
  "n_nei": 0,
  "features": {...}
}
```

## Label Distribution

From the augmented data (`classifier_train_data_with_labels.json`):

### Pool Type Distribution:
- `positive_support`: 194 samples (100% with labels)
- `positive_contradict`: 184 samples (100% with labels)
- `negative_conflict`: 158 samples (77% with labels)
- `negative_noise`: 191 samples (100% with labels)

### Ground Truth Label Counts:
- SUPPORT: 690
- CONTRADICT: 821
- NEI: 383

### Samples with Conflicting Evidence:
- 122 samples have both SUPPORT and CONTRADICT labels

## Key Insights

### Why Some Samples Are Unmatched

36 samples (5%) couldn't be matched to ground truth labels. These are all from `negative_conflict` pools for the following reasons:

1. **Claim 207, 599**: Have 9 evidence documents with both SUPPORT and CONTRADICT labels
   - Generate 502 possible combinations each
   - Random sampling of 30 combinations means different draws won't match

2. **Claim 872**: Has 24 evidence documents with both SUPPORT and CONTRADICT labels
   - Would generate 16+ million combinations
   - Reservoir sampling with early stopping after 30,000 combinations
   - Impossible to reproduce exact same combinations without storing the original pools

### Solution for Future Data Generation

To have 100% label coverage, use the modified `generate_classifier_data.py` to regenerate the full dataset:

```bash
uv run scripts/sufficiency_classifier/generate_classifier_data.py \
    --output data/classifier_train_data.json
```

This will create training data with labels built-in from the start.

## Performance Improvements

### Reservoir Sampling Optimization

The reservoir sampling fix provides significant speedup:

- **Before**: Claim 872 would hang trying to generate 16+ million combinations
- **After**: Completes in <100ms using reservoir sampling with early stopping

**Algorithm:**
1. For claims with >10 evidence docs: estimate total combinations
2. If estimate > 3000 (30 * 100), use reservoir sampling
3. Fill first 30 slots normally
4. For additional combinations, use reservoir sampling (replace with probability k/seen)
5. Early stop after seeing 30,000 valid combinations

## Files

### Modified:
- `/scripts/sufficiency_classifier/generate_classifier_data.py`
  - Updated to track doc_ids and labels
  - Added reservoir sampling for large claims

### Created:
- `/doc/classifier_data_augmentation.md` (this file)

### Generated Data:
- `/data/classifier_train_data_with_labels.json`
  - 727 samples, 691 with ground truth labels (95%)

## Usage Examples

### Analyze Label Distribution in Training Data

```python
import json

with open('data/classifier_train_data_with_labels.json') as f:
    data = json.load(f)

# Find samples with high conflict
conflict_samples = [s for s in data if s['n_support'] > 2 and s['n_contradict'] > 2]
print(f"High conflict samples: {len(conflict_samples)}")

# Check classifier performance on pure SUPPORT vs pure CONTRADICT
pure_support = [s for s in data if s['n_support'] > 0 and s['n_contradict'] == 0]
pure_contradict = [s for s in data if s['n_contradict'] > 0 and s['n_support'] == 0]
```

### Regenerate Full Dataset with Labels

```bash
# This will take ~30-60 minutes due to feature extraction
uv run scripts/sufficiency_classifier/generate_classifier_data.py \
    --output data/classifier_train_data.json \
    --seed 42 \
    --max_combos_per_claim 30 \
    --noise_paper_counts 1 2 3
```

## Future Work

Potential enhancements:

1. **Store original pool indices**: Save which specific combination was selected to enable perfect matching
2. **Stratified sampling**: Sample combinations to ensure diverse label distributions
3. **Label-aware features**: Create features that use ground truth labels for analysis
4. **Confusion matrix analysis**: Compare classifier predictions vs ground truth labels

## Validation

All changes have been tested and validated:

- ✅ Pool generation with labels works correctly
- ✅ Positive samples have unanimous labels
- ✅ Negative conflict samples have both SUPPORT and CONTRADICT
- ✅ Negative noise samples are all NEI
- ✅ Reservoir sampling handles large claims efficiently
- ✅ 95% of existing samples successfully matched to ground truth

## References

- Original SciFact data: `/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/`
  - `claims.jsonl`: Ground truth claims and evidence
  - `corpus.jsonl`: Paper abstracts and metadata
