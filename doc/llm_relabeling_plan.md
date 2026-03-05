# LLM-Based Relabeling for Negative Conflict Cases

**Date**: 2026-03-05
**Purpose**: Use LLM reasoning to establish ground truth labels for `negative_conflict` samples, where the correct sufficiency judgment is currently unknown.

---

## 1. Motivation

### The Core Problem: Unknown Ground Truth

For `negative_conflict` samples (mixed SUPPORT + CONTRADICT evidence):
- We don't know if they should be labeled **sufficient (y=1)** or **insufficient (y=0)**
- Examples:
  - **5S:1C** with high-IF SUPPORT papers → Should this be sufficient?
  - **1S:1C** with equal-quality papers → Probably insufficient (true conflict)
  - **3S:1C** where the 1C is a Nature paper → Depends on reasoning

### Why Use LLM for Labeling?

**LLM can**:
- Read paper texts and evaluate argument strength
- Consider metadata (Impact Factor, citations, publication year)
- Provide reasoning for each judgment
- Scale to 157 samples

**Purpose of MLP Classifier**:
- **Fast approximation** of LLM reasoning at runtime (~10ms vs 1-2s)
- Agent needs fast sufficiency checks after each paper retrieval
- Train MLP to mimic LLM judgments

---

## 2. Scope

### What We Relabel
- **Only `negative_conflict` cases** (157 samples from tau_0.00 dataset)

### What We Keep Unchanged
- `positive_support` (y=1) — all papers unanimously SUPPORT
- `positive_contradict` (y=1) — all papers unanimously CONTRADICT
- `negative_noise` (y=0) — irrelevant papers

### LLM Output
For each `negative_conflict` sample:
- **Binary label**: 0 (Insufficient) or 1 (Sufficient)
- **Reasoning**: 2-3 sentences explaining the decision

---

## 3. Key Challenges & Solutions

### Challenge 1: Context Window and Full Text Length

**Question**: Do we need full texts, or are abstracts sufficient?

**Possible approaches**:

**Option A: Abstracts Only (Simple)**
- Use only abstracts + metadata
- Fast, cheap, simple
- Risk: May miss nuanced arguments in full text

**Option B: Full Text with Chunking**
- Include full texts, chunk if too long
- More complete information
- Risk: Context overflow, higher cost

**Option C: Agentic Approach (Complex)**
- LLM first reads abstracts
- Agent decides what additional information is needed
- Retrieves specific sections from full text
- Risk: Much more complex, harder to debug

**Decision**: Start with **Option A (abstracts only)**, then:
1. Check how many papers have abstracts vs full texts
2. Run pilot to see if abstracts are sufficient
3. If LLM frequently says "need more context", upgrade to Option B

---

### Challenge 2: LLM Consistency

**Problem**: LLM may give different answers for the same input.

**Solution**:
- Query LLM **3 times** per sample with temperature=0.7
- Take **majority vote** (2/3 or 3/3) for final label
- If votes are 1-1-1 (complete disagreement):
  - Keep as y=0 (conservative: insufficient if ambiguous)

---

## 4. Implementation Plan

### Phase 1: Data Preparation

**Script**: `scripts/sufficiency_classifier/prepare_llm_relabeling_data.py`

**Inputs**:
- `data/ablation/tau_0.00.json` (157 negative_conflict samples)
- SciFact-Open corpus: `/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl`

**Tasks**:
1. Load all 157 `negative_conflict` samples
2. For each sample:
   - Extract claim text, doc_ids, scifact_labels
   - Lookup each doc_id in SciFact corpus
   - Extract abstract (check availability)
   - Extract full_text (check availability)
   - Extract metadata (journal, year, IF if available)
3. **Generate statistics**:
   - How many papers have abstracts?
   - How many have full texts?
   - Average abstract length, full text length
   - Context window requirements if using abstracts vs full texts

**Output format**:
```json
{
  "sample_id": "conflict_001",
  "claim_id": 145,
  "claim_text": "Autologous transplantation...",
  "evidence_pool": [
    {
      "doc_id": "10582939",
      "scifact_label": "CONTRADICT",
      "title": "...",
      "abstract": "...",
      "full_text": "...",  // May be null
      "metadata": {
        "journal": "Nature Medicine",
        "pub_year": 2018,
        "doi": "..."
      }
    },
    {
      "doc_id": "1092058",
      "scifact_label": "SUPPORT",
      "title": "...",
      "abstract": "...",
      "full_text": null,
      "metadata": {...}
    }
  ],
  "conflict_stats": {
    "n_support": 1,
    "n_contradict": 1,
    "n_papers": 2,
    "r_minority": 0.5
  }
}
```

**Outputs**:
- `data/llm_relabeling/conflict_samples_for_labeling.json` (157 samples)
- `data/llm_relabeling/preparation_stats.json`:
  ```json
  {
    "total_samples": 157,
    "total_papers": 450,
    "papers_with_abstract": 420,
    "papers_with_full_text": 380,
    "avg_abstract_length_tokens": 250,
    "avg_full_text_length_tokens": 8500,
    "max_context_with_abstracts": 1500,
    "max_context_with_full_texts": 35000
  }
  ```

**Decision point after Phase 1**:
- If most papers have abstracts + max context < 10k → use abstracts
- If many papers missing abstracts or abstracts seem too short → consider full texts

**Time estimate**: 1-2 hours

---

### Phase 2: Pilot Study

**Script**: `scripts/sufficiency_classifier/run_llm_labeling_pilot.py`

**Inputs**:
- `data/llm_relabeling/conflict_samples_for_labeling.json`

**Tasks**:
1. **Stratified sampling** (20-30 samples):
   - Balanced (1:1): 8-10 samples
   - Weak majority (2:1): 8-10 samples
   - Strong majority (≥3:1): 4-10 samples

2. For each sample:
   - Call LLM API **3 times** (temperature=0.7)
   - Parse response: label (0/1), reasoning
   - Record all 3 votes

3. **Analysis**:
   - Vote agreement: count of 3/3, 2/3, 1/3 agreements
   - Relabeling rate: % that get majority y=1 (by ratio group)
   - Manual review: Read 5-10 reasoning examples, check quality
   - Context sufficiency: Did LLM ever say "need more context"?

**LLM Setup**:
- Model: `claude-3-5-sonnet-20241022` or `gpt-4o` (TBD)
- Temperature: 0.7
- Max tokens: 800

**Outputs**:
- `data/llm_relabeling/pilot_results.json` (raw votes for 20-30 samples)
- `data/llm_relabeling/pilot_analysis.md` (summary report with):
  - Overall relabeling rate
  - Vote consistency statistics
  - Example reasonings (good and bad)
  - Recommendation: proceed with full relabeling or adjust approach

**Key metric**: Vote consistency and reasoning quality
- If consistency ≥ 70% and reasoning looks good → proceed to Phase 3
- If issues found → adjust prompt and re-run pilot

**Time estimate**: 2-3 hours

---

### Phase 3: Full Relabeling

**Script**: `scripts/sufficiency_classifier/run_llm_relabeling.py`

**Inputs**:
- `data/llm_relabeling/conflict_samples_for_labeling.json` (all 157)
- `data/llm_relabeling/pilot_analysis.md` (estimated relabeling rate)

**Tasks**:
1. Process all 157 samples
2. For each sample:
   - Call LLM **3 times**
   - Record all votes
   - Compute majority vote (2/3 or 3/3)
   - If 1-1-1 disagreement → label as 0 (conservative)

**Output format**:
```json
{
  "sample_id": "conflict_001",
  "claim_id": 145,
  "original_label": 0,
  "llm_votes": [
    {
      "label": 1,
      "reasoning": "The 5 supporting papers are from high-impact journals..."
    },
    {
      "label": 1,
      "reasoning": "Overwhelming evidence supports the claim..."
    },
    {
      "label": 0,
      "reasoning": "The contradicting paper raises significant concerns..."
    }
  ],
  "majority_vote": 1,
  "vote_agreement": "2/3",
  "final_label": 1,
  "relabeling_decision": "RELABELED_TO_SUFFICIENT"
}
```

**Outputs**:
- `data/llm_relabeling/relabeling_results.json` (all 157 samples)
- `data/llm_relabeling/relabeling_summary.json`:
  ```json
  {
    "total_processed": 157,
    "relabeled_to_y1": 95,
    "kept_as_y0": 62,
    "relabeling_rate": 0.605,
    "vote_agreement_3of3": 78,
    "vote_agreement_2of3": 71,
    "vote_disagreement_1of3": 8,
    "avg_reasoning_length": 145
  }
  ```

**Cost estimate**:
- 157 samples × 3 votes = 471 API calls
- ~1000 tokens per call (500 in + 500 out)
- Total: ~471k tokens
- Claude Sonnet: $2-3 USD
- GPT-4o: ~$2 USD
- GPT-4o-mini: ~$0.20 USD

**Time estimate**: 4-6 hours (mostly API time, can run overnight)

---

### Phase 4: Dataset Reconstruction

**Script**: `scripts/sufficiency_classifier/apply_llm_relabeling.py`

**Inputs**:
- `data/classifier_train_data_with_labels.json` (original 727 samples)
- `data/llm_relabeling/relabeling_results.json` (LLM decisions for 157 conflicts)

**Tasks**:
1. Load original dataset
2. For each sample:
   - If `pool_type == "negative_conflict"`:
     - Lookup LLM relabeling decision
     - If `final_label == 1`:
       - Update `target_y = 1`
       - Update `pool_type`:
         - If n_support > n_contradict → `positive_support`
         - If n_contradict > n_support → `positive_contradict`
       - Add provenance field:
         ```json
         "llm_relabeling": {
           "original_pool_type": "negative_conflict",
           "vote_agreement": "2/3",
           "relabeling_timestamp": "2026-03-05"
         }
         ```
     - Else keep as `negative_conflict` with y=0

3. **Validate final dataset**:
   - Check class balance
   - Check no duplicate samples
   - Check all required fields present

**Outputs**:
- `data/classifier_train_data_llm_relabeled.json` (new training dataset)
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

**Time estimate**: 1 hour

---

### Phase 5: Retrain & Evaluate

**Script**: Use existing `scripts/sufficiency_classifier/train_mlp_classifier.py`

**Inputs**:
- `data/classifier_train_data_llm_relabeled.json`

**Tasks**:
1. Create stratified train/test split (85/15)
2. Train MLP with `important` features (10 features)
3. Evaluate on test set
4. **Compare with baseline** (tau_0.00 model):
   - Test metrics: accuracy, F1, precision, recall
   - Feature importance
   - Predictions on conflict cases

**Additional analysis**:
- Load all conflict cases (both relabeled and kept)
- Compare:
  - Baseline model predictions on conflicts
  - LLM-relabeled model predictions on conflicts
  - Do they differ in systematic ways?

**Outputs**:
- `results/models/classifier_llm_relabeled/mlp_weights.pth`
- `results/models/classifier_llm_relabeled/mlp_config.json`
- `results/models/classifier_llm_relabeled/evaluation_report.md`
- `results/models/classifier_llm_relabeled/comparison_with_baseline.md`

**Time estimate**: 1-2 hours

---

## 5. Prompt Design

### Sufficiency Judgment Prompt (v1)

```
You are a scientific expert evaluating whether a set of research papers provides SUFFICIENT evidence for making a confident judgment about a scientific claim.

# CLAIM
{claim_text}

# EVIDENCE POOL
{for each paper:}
Paper {i}: [{scifact_label}]
Title: {title}
Journal: {journal}, {pub_year}
Abstract: {abstract}

# TASK
Judge whether this evidence pool is SUFFICIENT for an AI research agent to confidently judge the claim, or INSUFFICIENT (requiring more evidence).

# DEFINITIONS

SUFFICIENT (label = 1):
- Evidence strongly points in one direction (support OR refutation)
- Quality and consistency allow confident judgment
- Minor dissent doesn't undermine overall conclusion
- Examples:
  * 5 high-quality papers SUPPORT, 1 low-quality CONTRADICT → sufficient
  * 3 methodologically sound papers CONTRADICT, 0 SUPPORT → sufficient

INSUFFICIENT (label = 0):
- Evidence is genuinely conflicting with no clear winner
- High-quality evidence on both sides
- Ambiguous or requires deeper investigation
- Examples:
  * 2 high-impact SUPPORT, 2 high-impact CONTRADICT → insufficient
  * 1 SUPPORT, 1 CONTRADICT, equal quality → insufficient

# GUIDELINES
- Consider journal quality (Nature > low-impact journal)
- Consider publication year (newer may supersede older)
- Consider consistency of findings across papers
- A strong majority can outweigh weak minority dissent
- True high-quality conflicts should remain insufficient

# OUTPUT FORMAT (JSON only, no extra text)
{
  "label": 0 or 1,
  "reasoning": "Two to three sentences explaining your judgment, explicitly referencing paper quality, consistency, and any important considerations."
}
```

**Note**: Prompt may need adjustment after pilot study based on LLM behavior.

---

## 6. Expected Outcomes

### Quality Improvements

1. **Principled labels**: Based on LLM reasoning, not arbitrary thresholds
2. **Interpretable**: Each decision has reasoning attached
3. **Testable**: Can validate LLM reasoning quality manually
4. **Flexible**: Can retrain with different LLM judgments if needed

---

## 7. Validation Strategy

### Option 1: Human Agreement Study
- You manually label 30 conflict cases (stratified)
- Compare with LLM majority votes
- Measure agreement rate
- If >80% → trust LLM labels

### Option 2: Reasoning Quality Review
- Manually read 20-30 LLM reasonings
- Assess quality:
  - Does reasoning make sense?
  - Does it consider paper quality?
  - Does it handle conflicts appropriately?
- If high quality → trust labels

### Option 3: Model Performance
- If MLP trained on LLM labels achieves F1 ≥ 0.90
- And predictions are interpretable
- Then labels are likely good quality

---

## 8. Timeline

| Phase | Duration | Blocking? | Cost |
|-------|----------|-----------|------|
| 1. Data prep | 1-2 hours | Yes | $0 |
| 2. Pilot (20-30 samples) | 2-3 hours | Yes | $0.30 |
| **Decision point** | 30 min | Yes | $0 |
| 3. Full relabeling (157) | 4-6 hours | No* | $2-3 |
| 4. Dataset reconstruction | 1 hour | Yes | $0 |
| 5. Retrain & evaluate | 1-2 hours | No* | $0 |
| **Total** | **10-13 hours** | ~5 hours hands-on | **$2-4** |

*Can run in background/overnight

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Low vote consistency (<70%) | Medium | Pilot will catch this, adjust prompt |
| Missing abstracts | Low | Stats in Phase 1 will show, use what's available |
| Context overflow | Low | Start with abstracts, upgrade if needed |
| LLM reasoning poor quality | Medium | Manual review in pilot, adjust prompt |

---

## 10. Success Criteria

1. ✅ LLM inter-vote consistency ≥ 70% (2/3 or 3/3 votes)
2. ✅ MLP F1 ≥ 0.90 on test set (comparable to baseline)
3. ✅ LLM reasonings are interpretable and sensible
4. ✅ Total cost < $5 USD

---

## 11. Open Questions for Discussion

1. **Text input**: Abstracts only, or include full texts? (Will decide after Phase 1 stats)

2. **LLM model**:
   - Claude 3.5 Sonnet (~$2.50 total)
   - GPT-4o (~$2 total)
   - GPT-4o-mini (~$0.20 total, but may be lower quality)

3. **Human validation**: Should you manually label a subset before full LLM relabeling, or trust pilot study?

4. **Agentic approach**: Is it worth exploring LLM agent that requests specific paper sections, or keep it simple (abstracts only)?

---

## 12. Next Steps

1. ✅ **Review this plan** — get your feedback on the simplified approach
2. **Implement Phase 1** — check paper text availability, generate stats
3. **Run pilot** — 20-30 samples to validate feasibility
4. **Decision** — proceed with full relabeling or adjust
5. **Execute** — complete all phases
6. **Document** — write up results and comparison with tau-based approach

---

**Key Design Principles**:
- ✅ Simple majority vote for consistency
- ✅ LLM decides all labels based on reasoning
- ✅ Start with abstracts, upgrade only if needed
- ✅ Pilot study to validate approach before full run
