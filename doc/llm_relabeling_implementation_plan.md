# LLM-Based Relabeling: Implementation Plan

**Purpose**: Guide for AI agent to implement basic LLM relabeling pipeline
**Date**: 2026-03-05

---

## Overview

Implement a pipeline to relabel 157 `negative_conflict` samples using LLM reasoning.

**Scope**: Only relabel `negative_conflict` cases. Keep `positive_support`, `positive_contradict`, and `negative_noise` unchanged.

---

## Phase 1: Data Preparation

**Script**: `scripts/sufficiency_classifier/prepare_llm_relabeling_data.py`

### Inputs
- `data/ablation/tau_0.00.json` (contains 157 negative_conflict samples)
- SciFact-Open corpus: `/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl`
- Metadata cache: `data/metadata_cache.json` (from existing pipeline)

### Tasks
1. Load tau_0.00 dataset and filter for `pool_type == "negative_conflict"` (should be 157 samples)

2. For each conflict sample:
   - Extract: `claim_id`, `claim_text`, `doc_ids`, `scifact_labels`, `n_support`, `n_contradict`

3. For each doc_id in the evidence pool:
   - **From SciFact corpus**:
     - Lookup doc_id in corpus.jsonl
     - Extract: `title`, `abstract`, `structured` (full text if available)

   - **From metadata cache**:
     - Extract Impact Factor (IF)
     - Extract author max H-index
     - Extract normalized citation count
     - Extract publication year
     - Extract journal name

4. Generate statistics:
   - Total samples, total papers
   - Papers with abstract vs full text availability
   - Papers with metadata (IF, H-index) availability
   - Average token counts (abstract, full text)
   - Max context window needed if using abstracts vs full texts

5. Save to JSON

### Output Format
```json
[
  {
    "sample_id": "conflict_001",
    "claim_id": 145,
    "claim_text": "...",
    "evidence_pool": [
      {
        "doc_id": "10582939",
        "scifact_label": "CONTRADICT",
        "title": "...",
        "abstract": "...",
        "full_text": "...",  // may be null if unavailable
        "metadata": {
          "journal": "Nature Medicine",
          "pub_year": 2018,
          "impact_factor": 53.4,
          "author_max_h_index": 82,
          "norm_citation_count": 45.2
        }
      }
    ],
    "conflict_stats": {
      "n_support": 1,
      "n_contradict": 1,
      "n_papers": 2,
      "r_minority": 0.5
    }
  }
]
```

### Outputs
- `data/llm_relabeling/conflict_samples_for_labeling.json` (157 samples)
- `data/llm_relabeling/preparation_stats.json`:
  ```json
  {
    "total_samples": 157,
    "total_papers": 450,
    "papers_with_abstract": 420,
    "papers_with_full_text": 380,
    "papers_with_metadata": 398,
    "avg_abstract_length_tokens": 250,
    "avg_full_text_length_tokens": 8500,
    "max_context_with_abstracts": 1500,
    "max_context_with_full_texts": 35000
  }
  ```

---

## Phase 2: Pilot Study

**Script**: `scripts/sufficiency_classifier/run_llm_labeling_pilot.py`

### Inputs
- `data/llm_relabeling/conflict_samples_for_labeling.json`
- Prompt template file: `prompts/llm_sufficiency_judgment.txt` (to be provided by user)

### Tasks
1. Stratified sampling (20-30 samples):
   - Group by r_minority ratio
   - Sample ~10 balanced (r_minority == 0.5)
   - Sample ~10 weak majority (0.33 <= r_minority < 0.5)
   - Sample ~10 strong majority (r_minority < 0.33)

2. For each sample:
   - Load prompt template from file
   - Fill in template with sample data (claim, evidence pool)
   - Call LLM API 3 times with temperature=0.7
   - Parse JSON response: `{"label": 0 or 1, "reasoning": "..."}`
   - Record all 3 votes

3. Compute statistics:
   - Vote agreement counts (3/3, 2/3, 1/3)
   - Majority vote for each sample
   - Relabeling rate overall and by ratio group
   - Average reasoning length

4. Save results

### Outputs
- `data/llm_relabeling/pilot_results.json` (raw votes for all pilot samples)
- `data/llm_relabeling/pilot_analysis.md` (summary stats + sample reasonings for manual review)

### LLM Configuration
- Model: `glm-4-plus` (via ZhipuAI API) or `claude-3-5-sonnet-20241022` or `gpt-4o`
- Temperature: 0.7
- Max tokens: 800

---

## Phase 3: Full Relabeling

**Script**: `scripts/sufficiency_classifier/run_llm_relabeling.py`

### Inputs
- `data/llm_relabeling/conflict_samples_for_labeling.json` (all 157)
- Prompt template file: `prompts/llm_sufficiency_judgment.txt`

### Tasks
1. For each of 157 samples:
   - Load and fill prompt template
   - Call LLM 3 times (temperature=0.7)
   - Record votes
   - Compute majority vote:
     - If 3/3 or 2/3 agreement → use majority
     - If 1-1-1 disagreement → label as 0 (conservative)

2. Save results with provenance

### Output Format
```json
[
  {
    "sample_id": "conflict_001",
    "claim_id": 145,
    "original_label": 0,
    "llm_votes": [
      {"label": 1, "reasoning": "..."},
      {"label": 1, "reasoning": "..."},
      {"label": 0, "reasoning": "..."}
    ],
    "majority_vote": 1,
    "vote_agreement": "2/3",
    "final_label": 1
  }
]
```

### Outputs
- `data/llm_relabeling/relabeling_results.json` (all 157 samples with votes)
- `data/llm_relabeling/relabeling_summary.json`:
  ```json
  {
    "total_processed": 157,
    "relabeled_to_y1": 95,
    "kept_as_y0": 62,
    "relabeling_rate": 0.605,
    "vote_agreement_3of3": 78,
    "vote_agreement_2of3": 71,
    "vote_disagreement_1of3": 8
  }
  ```

---

## Phase 4: Dataset Reconstruction

**Script**: `scripts/sufficiency_classifier/apply_llm_relabeling.py`

### Inputs
- `data/classifier_train_data_with_labels.json` (original 727 samples)
- `data/llm_relabeling/relabeling_results.json` (LLM decisions)

### Tasks
1. Load original dataset (727 samples)

2. Create mapping: claim_id → LLM relabeling decision

3. For each sample in original dataset:
   - If `pool_type != "negative_conflict"`:
     - Keep unchanged

   - If `pool_type == "negative_conflict"`:
     - Lookup LLM decision by claim_id

     - If `final_label == 1`:
       - Update `target_y = 1`
       - Update `pool_type`:
         - If `n_support > n_contradict` → `positive_support`
         - Else → `positive_contradict`
       - Add provenance metadata:
         ```json
         "llm_relabeling": {
           "original_pool_type": "negative_conflict",
           "vote_agreement": "2/3",
           "relabeling_timestamp": "2026-03-05T10:30:00Z"
         }
         ```

     - If `final_label == 0`:
       - Keep `target_y = 0`
       - Keep `pool_type = "negative_conflict"`

4. Validate:
   - Check total sample count (should still be 727)
   - Count pool_type distribution
   - Count y=1 vs y=0

5. Generate comparison stats

### Outputs
- `data/classifier_train_data_llm_relabeled.json` (new dataset, same 727 samples)
- `data/llm_relabeling/dataset_comparison.json`:
  ```json
  {
    "original": {
      "total": 727,
      "y=1": 378,
      "y=0": 349,
      "pool_types": {
        "positive_support": 194,
        "positive_contradict": 184,
        "negative_conflict": 157,
        "negative_noise": 191
      }
    },
    "llm_relabeled": {
      "total": 727,
      "y=1": 473,
      "y=0": 254,
      "pool_types": {
        "positive_support": 260,
        "positive_contradict": 213,
        "negative_conflict": 62,
        "negative_noise": 191
      },
      "changes": {
        "conflicts_relabeled": 95,
        "conflicts_kept": 62
      }
    }
  }
  ```

---

## Phase 5: Retrain Classifier

**Script**: Use existing `scripts/sufficiency_classifier/train_mlp_classifier.py`

### Tasks
Train MLP on new LLM-relabeled dataset:

```bash
python scripts/sufficiency_classifier/train_mlp_classifier.py \
  --data_path data/classifier_train_data_llm_relabeled.json \
  --output_dir results/models/classifier_llm_relabeled \
  --features important \
  --epochs 100
```

### Outputs
- `results/models/classifier_llm_relabeled/mlp_weights.pth`
- `results/models/classifier_llm_relabeled/mlp_config.json`
- Training logs with metrics

---

## Implementation Notes

### Dependencies
- Reuse existing metadata cache from `data/metadata_cache.json`
- Reuse existing `PaperFeatureExtractor` class for metadata lookup
- SciFact corpus parser for abstracts and full texts

### Error Handling
- **Missing abstracts**: Use title + metadata only, log warning
- **Missing metadata**: Use available fields, set others to None
- **API rate limits**: Retry with exponential backoff (max 3 retries)
- **JSON parsing errors**: Log error, skip sample, flag for manual review
- **API errors**: Log and continue, report failed samples at end

### Logging
- Phase 1: Log paper lookup progress, metadata cache hits/misses
- Phase 2: Log pilot progress, vote agreements
- Phase 3: Log API calls, progress (X/157), vote disagreements
- Phase 4: Log relabeling decisions, dataset changes
- Phase 5: Use existing training script logging

### File Organization
```
data/llm_relabeling/
├── conflict_samples_for_labeling.json    (Phase 1 output)
├── preparation_stats.json                 (Phase 1 output)
├── pilot_results.json                     (Phase 2 output)
├── pilot_analysis.md                      (Phase 2 output)
├── relabeling_results.json                (Phase 3 output)
├── relabeling_summary.json                (Phase 3 output)
└── dataset_comparison.json                (Phase 4 output)

prompts/
└── llm_sufficiency_judgment.txt           (User-provided prompt template)

scripts/sufficiency_classifier/
├── prepare_llm_relabeling_data.py         (Phase 1)
├── run_llm_labeling_pilot.py              (Phase 2)
├── run_llm_relabeling.py                  (Phase 3)
└── apply_llm_relabeling.py                (Phase 4)
```

---

## Success Criteria

**Phase 1**:
- ✅ 157 conflict samples extracted
- ✅ Metadata (IF, H-index) available for >80% of papers
- ✅ Statistics generated successfully

**Phase 2**:
- ✅ 20-30 pilot samples labeled
- ✅ Vote consistency ≥ 70% (2/3 or 3/3 agreement)
- ✅ Reasonings are interpretable (manual check)

**Phase 3**:
- ✅ All 157 samples relabeled (or flagged if failed)
- ✅ Cost within budget (~$2-4 USD)
- ✅ Summary statistics generated

**Phase 4**:
- ✅ Dataset reconstructed successfully (727 samples)
- ✅ No data loss or corruption
- ✅ Comparison stats show expected changes

**Phase 5**:
- ✅ Model trains without errors
- ✅ Can generate predictions on test set

---

## Next Steps After Implementation

1. Execute Phase 1 → review preparation stats
2. Execute Phase 2 → manually review pilot_analysis.md
3. If pilot looks good → execute Phase 3
4. Execute Phase 4 → review dataset_comparison.json
5. Execute Phase 5 → training completes
6. Proceed to analysis phase (see separate analysis plan)
