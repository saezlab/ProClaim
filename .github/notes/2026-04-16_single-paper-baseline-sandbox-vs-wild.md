# Single-Paper Baseline: Sandbox vs. In-the-Wild Analysis

**Date:** 2026-04-16  
**Branch:** `exp` (baseline code) · `main` (paper)  
**Commits:** `8f3a99e` ("in sandbox vs. in the wild test") · `9269b6e` ("section 2")

## Summary

Implemented a new `single_paper` baseline that verifies claims using only the source paper's abstract (the paper originally cited when the SIGNOR edge was curated). This creates a controlled "in-sandbox" setting (Setting 1) that is directly comparable to the existing S2 retrieval baseline (Setting 2, "in-the-wild"). Both use the same LLM (`claude-sonnet-4-6`), same prompt template (`VERIFICATION_SYSTEM_PROMPT` + `VERIFICATION_USER_TEMPLATE`), and same label normalization — the only variable is the evidence source. The experiment ran on all 101 SIGNOR-Fact claims and produced quantitative evidence for the paper's Section 2 argument.

The results directly informed a rewrite of Section 2 of `ProClaim.tex`: the empirical paragraph was changed to focus on **verdict transitions between settings** (not accuracy vs. ground truth), the in-sandbox/in-the-wild definitions were sharpened to emphasise what a verdict is grounded in, and a new figure (`graphic_overview-motivation.pdf`) was added to visually contrast the two settings.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/single_paper.py` | "In-sandbox" baseline: fetches source paper abstract via PubMed E-utilities efetch, formats as a single evidence passage, single LLM call |
| `results/baselines/single_paper/anthropic--claude-sonnet-4-6/signor_seed100.jsonl` | Per-claim results (101 claims) |
| `results/baselines/single_paper/anthropic--claude-sonnet-4-6/signor_metrics.json` | Aggregated metrics |

## Modified Files

| File | Changes |
|------|---------|
| `experiments/run_baselines_datasets.py` | Registered `single_paper` in `build_baseline()` factory, added to CLI `--baseline` choices, added output directory logic for model slug, added `total_input_tokens` / `total_output_tokens` to `aggregate_metrics()` |
| `-NeurIPS26-Evidence-Programming/ProClaim.tex` | Section 2 rewritten: added motivation figure, sharpened sandbox/wild definitions, rewrote empirical paragraph to focus on verdict transitions, tied in-the-wild framing to knowledge construction problem |

## Architecture

```
                  ┌──────────────┐
                  │  signor.csv  │
                  │  (101 claims │
                  │  with PMIDs) │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
    ┌─────────▼─────────┐ ┌────────▼─────────┐
    │  Setting 1:       │ │  Setting 2:      │
    │  single_paper     │ │  s2_retrieval    │
    │                   │ │                  │
    │ PubMed efetch     │ │ Semantic Scholar │
    │ → 1 source        │ │ → top-5 papers   │
    │   abstract        │ │   abstracts      │
    └─────────┬─────────┘ └────────┬─────────┘
              │                     │
              │  Same LLM + prompt  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  VERIFICATION_      │
              │  SYSTEM_PROMPT +    │
              │  USER_TEMPLATE      │
              │  (claude-sonnet-4-6)│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ BaselineResult      │
              │ → SUPPORT / REFUTE  │
              │   / UNCERTAIN       │
              └─────────────────────┘
```

## Key Design Decisions

- **Same prompt for both settings**: The `single_paper` baseline reuses `VERIFICATION_SYSTEM_PROMPT` and `VERIFICATION_USER_TEMPLATE` (not the `LLM_ONLY` variants) so the only experimental variable is the evidence source, not the prompt framing.
- **PubMed efetch over S2 API**: Source paper abstracts are fetched directly via NCBI E-utilities (free, no API key required, reliable for known PMIDs) rather than looking them up on Semantic Scholar.
- **Context-forwarding via `EvaluationHarness`**: The harness auto-detects that `SinglePaper.verify()` accepts a `context` kwarg and passes the full claim dict (including `pmid`). No harness modifications needed.
- **Graceful fallback**: Claims without a valid PMID or with a failed efetch default to UNCERTAIN with explanatory reasoning, rather than crashing the run.
- **Structured abstract handling**: PubMed structured abstracts (with labels like BACKGROUND, METHODS, RESULTS) are concatenated with their section labels preserved.
- **Verdict transitions as the paper's argument**: The empirical paragraph in Section 2 was deliberately rewritten to report verdict *changes between settings* rather than accuracy vs. gold labels. This frames the result as evidence that additional retrieved literature *conflicts* with the source paper's perspective, motivating agentic evidence construction — not as a head-to-head accuracy comparison.
- **Definition precision**: In-sandbox is defined as a verdict grounded in a *claim and a specific retrieval passage* (limited but unambiguous); in-the-wild as a verdict against *the entire literature* (conflicting perspectives, verdict emerges from the aggregate). The sandbox definition avoids claiming the verdict depends on a single passage, since sandbox systems still do retrieval — they just do it in one turn over a fixed set.

## Results

**Overall metrics (n=101, claude-sonnet-4-6, seed 100):**

| Metric | Single Paper | S2 Top-5 |
|--------|-------------|----------|
| Accuracy | 0.644 | 0.653 |
| Macro F1 | **0.491** | 0.439 |
| Weighted FPR | **0.175** | 0.444 |
| Cost (USD) | 0.39 | 0.89 |

**Per-class F1:**

| Class | Single Paper | S2 Top-5 |
|-------|-------------|----------|
| SUPPORT | **0.610** | 0.356 |
| REFUTE | 0.786 | **0.781** |
| UNCERTAIN | 0.077 | **0.182** |

**Disagreement analysis (50/101 claims disagree):**

| Pattern | Count |
|---------|-------|
| Both agree & correct | 44 |
| Both agree & wrong | 7 |
| SP correct, S2 wrong | 21 |
| S2 correct, SP wrong | 22 |
| Both disagree & wrong | 7 |

**Key disagreement patterns:**
- 12× gold=REFUTE: SP says UNCERTAIN, S2 correctly says REFUTE — source paper doesn't contradict itself; retrieval finds contradicting evidence
- 11× gold=SUPPORT: SP correctly says SUPPORT, S2 wrongly says REFUTE — retrieved papers override the source and flip the verdict
- 5× gold=REFUTE: SP wrongly says SUPPORT, S2 correctly says REFUTE — source paper actively supports its own (now-refuted) claim

**Takeaway:** Both settings achieve ~65% accuracy but fail in complementary ways. The source paper exhibits confirmation bias (over-supports), while naive retrieval exhibits contradictory-evidence flooding (over-refutes). Neither handles UNCERTAIN well (F1 ≤ 0.18). This motivates agentic evidence construction that must actively balance perspectives.
