# Gemini vs Claude LLM-Only Baseline Analysis — 2026-04-15

## Branch

`exp`

## Summary

Investigated why Gemini models (2.5 Pro, 3.1 Pro Preview) substantially outperform Claude Sonnet 4.6 on both SIGNOR and ConnectomeDB benchmarks in the `llm_only` baseline. The performance gap is driven by **calibration to the label distribution**, not superior biological knowledge. Claude massively over-predicts UNCERTAIN (~10× the gold rate), while Gemini's prediction distribution closely mirrors the gold distribution. Both models demonstrate comparable domain knowledge in their reasoning.

## Metrics Comparison

### SIGNOR (n=101)

| Metric | Gemini 2.5 Pro | Gemini 3.1 Pro Preview | Claude Sonnet 4.6 |
|--------|---------------|----------------------|-------------------|
| Accuracy | 0.78 | 0.74 | 0.505 |
| Macro F1 | 0.52 | 0.53 | 0.44 |
| SUPPORT F1 | 0.72 | 0.75 | 0.65 |
| REFUTE F1 | 0.72* | 0.83 | 0.57 |
| UNCERTAIN F1 | — | 0.00 | 0.10 |

*Gemini 2.5 Pro was run under different label definitions (`llm_only/` vs `llm_only_long_label_def/`).

### ConnectomeDB (n=318)

| Metric | Gemini 3.1 Pro Preview | Claude Sonnet 4.6 |
|--------|----------------------|-------------------|
| Accuracy | 0.654 | 0.415 |
| Macro F1 | 0.497 | 0.341 |
| SUPPORT F1 | 0.745 | 0.643 |
| REFUTE F1 | 0.584 | 0.210 |
| UNCERTAIN F1 | 0.162 | 0.170 |

## Prediction Distribution Analysis

### SIGNOR

| Label | Gold | Gemini 2.5 Pro | Gemini 3.1 Pro | Claude Sonnet |
|-------|------|---------------|---------------|---------------|
| SUPPORT | 34 | 41 | 41 | 34 |
| REFUTE | 63 | 60 | 50 | 32 |
| UNCERTAIN | 4 | **0** | 10 | **35** |

### ConnectomeDB

| Label | Gold | Gemini 3.1 Pro | Claude Sonnet |
|-------|------|---------------|---------------|
| SUPPORT | 184 | 189 | 130 |
| REFUTE | 121 | 105 | 60 |
| UNCERTAIN | 13 | 24 | **128** |

Claude predicts UNCERTAIN ~10× the gold rate on both datasets.

## Disagreement Analysis

### SIGNOR (Gemini 2.5 Pro vs Claude)

- Both correct: 46
- Both wrong: 17
- Gemini right, Claude wrong: **33**
- Claude right, Gemini wrong: **5**

### ConnectomeDB (Gemini 3.1 Pro vs Claude)

- Both correct: 108
- Both wrong: 86
- Gemini right, Claude wrong: **100**
- Claude right, Gemini wrong: **24**

#### Where Gemini wins (ConnectomeDB, 100 cases):

- 53× gold=REFUTE → Claude said UNCERTAIN (49) or SUPPORT (4)
- 47× gold=SUPPORT → Claude said UNCERTAIN (29) or REFUTE (18)

#### Where Claude wins (ConnectomeDB, 24 cases):

- 9× gold=UNCERTAIN → Gemini said REFUTE (8): Gemini's decisiveness backfires on genuinely uncertain claims
- 9× gold=SUPPORT → Gemini said REFUTE (6) or UNCERTAIN (3)
- 6× gold=REFUTE → Gemini said SUPPORT (6)

## Key Findings

### 1. Claude over-hedges with UNCERTAIN

Both models identify the same absence of evidence, but interpret it differently:

- **Claude**: "no evidence" → "insufficient to determine" → UNCERTAIN
- **Gemini**: "no evidence" → "not substantiated" → REFUTE

Example — *"CDK2 directly inhibits UBE2R2"* (gold=REFUTE):
- Claude: *"a direct inhibitory relationship has not been clearly documented… making this claim unverifiable"* → UNCERTAIN
- Gemini: *"there is no established evidence that CDK2 directly inhibits UBE2R2"* → REFUTE

### 2. Claude confuses interaction direction (3 cases on SIGNOR)

Example — *"ATR directly inhibits CHEK1"* (gold=REFUTE, ATR **activates** CHEK1):
- Claude describes ATR phosphorylating CHEK1 (activation) but concludes SUPPORT for an inhibition claim
- Gemini correctly notices the claim says "inhibits" but evidence shows "activates" → REFUTE

### 3. Claude applies an overly strict "canonical interaction" bar (ConnectomeDB)

Example — *"CD40LG as ligand directly interacts with ITGA2B as receptor"* (gold=SUPPORT):
- Gemini cites the platelet GPIIb/IIIa literature → SUPPORT
- Claude knows the same biology but rejects it: *"CD40LG's canonical receptor is CD40, not ITGA2B"* → REFUTE

### 4. Gemini over-commits on rare UNCERTAIN cases

Example — *"CHUK directly inhibits NFKBIA"* (gold=SUPPORT):
- Gemini argues degradation ≠ inhibition (semantically wrong) → REFUTE
- Claude correctly maps phosphorylation→degradation as inhibition → SUPPORT

### 5. Dataset labeling convention favors decisiveness

Both SIGNOR and ConnectomeDB label fabricated/flipped/unsubstantiated edges as REFUTE rather than UNCERTAIN. UNCERTAIN is reserved for genuinely ambiguous biology (~4% SIGNOR, ~4% ConnectomeDB). Gemini's strategy of almost never predicting UNCERTAIN aligns with this convention.

## Implications

- The gap is largely a **calibration** issue, not a knowledge issue
- Post-processing Claude's UNCERTAIN→REFUTE would close most of the gap
- Alternatively, prompt engineering to discourage UNCERTAIN unless truly conflicting evidence exists could help
- For fair comparison, consider reporting results both with and without UNCERTAIN collapse
