# Section 2 Motivation Analysis Plan

## Goal

Section 2 must clearly demonstrate two claims:

1. **The in-the-wild setting is necessary** for establishing scientific consensus (a single paper's perspective ≠ consensus).
2. **Knowledge construction is required** in the wild setting — naive bulk retrieval is insufficient, iterative structured evidence assembly is needed.

The current empirical illustration (101 SIGNOR-Fact claims, sandbox vs. top-5 S2) is descriptive: it shows verdicts change but does not show whether the change is correct or why naive retrieval fails.

---

## Proposed Analyses

### Analysis A — Sandbox is insufficient for consensus (Point 1)

**What it shows:** A single source paper does not reflect the scientific consensus.

**Method:** Compare both settings against gold labels.
- Setting 1 (source paper abstract only) accuracy vs. gold
- Setting 2 (top-5 S2 abstracts) accuracy vs. gold

**Key outputs:**
- Accuracy, macro F1, per-class recall for both settings against gold
- Confusion matrix: sandbox verdict × gold verdict — shows which sandbox errors get corrected by broader retrieval and which new errors appear
- Highlight: proportion of claims where the source paper verdict disagrees with gold

**Data needed:** Already available (101 claims with gold labels, both setting outputs).

**Findings (from `scripts/analysis/sandbox_vs_wild_gold_analysis.py`):**

Gold distribution: 34 SUPPORT, 63 REFUTE, 4 UNCERTAIN (n=101).

| Setting | Accuracy | Macro F1 | Macro FPR | Macro FNR |
|---------|----------|----------|-----------|-----------|
| S1 (source paper) | 0.644 | 0.491 | 0.177 | 0.497 |
| S2 (top-5 S2) | 0.653 | 0.439 | 0.264 | 0.537 |
| S3 (top-5 + ref) | 0.703 | 0.538 | 0.154 | 0.438 |

Per-class recall (the key metric):

| Setting | SUPPORT recall | REFUTE recall | UNCERTAIN recall |
|---------|---------------|--------------|-----------------|
| S1 | 0.529 | 0.730 | 0.250 |
| S2 | 0.235 | 0.905 | 0.250 |
| S3 | 0.706 | 0.730 | 0.250 |

**Key takeaway for Point 1:** S1 accuracy of 0.644 means 36% of verdicts from the source paper alone disagree with the scientific consensus. Sandbox is insufficient.

**Key takeaway for Point 2 setup:** S2 accuracy barely improves (+0.009) despite adding 5 retrieved papers. Macro F1 actually *drops* (0.491 → 0.439). SUPPORT recall collapses from 0.529 to 0.235 (56% relative drop). 31% of S2's REFUTE predictions are wrong (26/83). Naive retrieval trades one type of error for another without net improvement.

**S3 finding (oracle context):** Including the reference paper alongside the 5 S2 papers achieves the best accuracy (0.703) and recovers SUPPORT recall to 0.706. This shows that the reference paper contains critical supporting evidence that generic retrieval misses — further motivating targeted, gap-driven retrieval.

---

### Analysis B — Naive retrieval is necessary but insufficient (bridge to Point 2)

**What it shows:** More evidence helps but introduces systematic errors without structured reconciliation.

**Method:** Break down the 50 verdict transitions by correctness against gold.
- Of the 17 SUPPORT→REFUTE transitions: how many are correct corrections vs. false corrections?
- Of the 19 UNCERTAIN→REFUTE transitions: how many collapse genuine ambiguity?
- Net accuracy change from Setting 1 to Setting 2

**Findings (from `scripts/analysis/sandbox_vs_wild_gold_analysis.py`):**

50/101 verdicts change from S1→S2. Of these 50 changed verdicts:
- **22 fixed** (was wrong, now correct)
- **21 broke** (was correct, now wrong)
- **7 stayed wrong** (changed to a different wrong label)
- **Net accuracy change: +1 claim** — essentially a wash

Breakdown of the two dominant transitions:

| Transition | Count | Broke | Fixed | Stayed wrong |
|-----------|-------|-------|-------|-------------|
| SUPPORT → REFUTE | 17 | **11** | 5 | 1 |
| UNCERTAIN → REFUTE | 19 | 1 | **12** | 6 |
| SUPPORT → UNCERTAIN | 4 | 3 | 1 | 0 |
| REFUTE → SUPPORT | 4 | 3 | 1 | 0 |
| REFUTE → UNCERTAIN | 3 | 3 | 0 | 0 |
| UNCERTAIN → SUPPORT | 3 | 0 | 3 | 0 |

**Key insight:** Naive retrieval creates a *destructive trade-off*:
- It correctly resolves UNCERTAIN → REFUTE (12 fixes) — broader evidence does surface real refutations.
- But it catastrophically over-refutes SUPPORT claims (11 breaks in SUPPORT → REFUTE alone) — the system misinterprets "retrieved papers don't mention this specific interaction" as "the interaction is false."
- The net effect is nearly zero improvement (+1 claim), but the composition of errors shifts dramatically. The system loses the ability to recognise supported claims (recall: 0.529 → 0.235).

---

### Analysis C — Error taxonomy showing construction is needed (Point 2)

**What it shows:** Why naive retrieval fails — categorised failure modes that motivate ProClaim's design.

**Method:** Manually annotate ~10-15 representative failure cases from Setting 2 (top-5 S2) into categories.

**Failure mode categories:**

1. **Conflict flooding:** Multiple retrieved papers disagree; the LLM defaults to REFUTE instead of weighing evidence quality. *Motivates: structured evidence state with typed facts and quality weighting.* This is the dominant failure mode: 11 of 17 SUPPORT→REFUTE transitions are incorrect, and 26/83 REFUTE predictions in S2 are false positives.

2. **Missing targeted evidence:** The claim involves a specific mechanism/context that generic queries don't surface. A follow-up query informed by what's already been found would succeed. *Motivates: iterative, gap-driven retrieval.* The S3 result (accuracy 0.703 vs. S2's 0.653) shows that including the reference paper — which contains the *right* supporting evidence — recovers SUPPORT recall from 0.235 to 0.706. The issue is not that evidence doesn't exist, but that a single generic query fails to find it.

3. **Ambiguity collapse:** Genuine UNCERTAIN cases classified as REFUTE because the system cannot distinguish "evidence conflicts" from "evidence is absent." *Motivates: sufficiency classifier and explicit conflict tracking.* 3 of 4 gold-UNCERTAIN claims are predicted as REFUTE in S2.

**Key outputs:**
- 2-3 worked examples per category (claim text, retrieved papers, verdict, gold label, explanation of failure)
- Distribution of failure modes across all incorrect Setting 2 predictions

**Data needed:** Setting 2 outputs + gold labels + manual inspection of ~15 cases.

**Expected finding:** Each failure mode maps directly to a ProClaim design decision (evidence state, adaptive retrieval, sufficiency classifier), creating a tight motivation→solution narrative.

---

### Analysis D — Claim complexity varies (Optional, Point 2)

**What it shows:** A fixed retrieval budget is fundamentally mismatched to variable claim difficulty.

**Method:** Use ProClaim iteration counts from run logs.
- Histogram of iterations to convergence
- Correlate iteration count with claim properties: number of entities, specificity, gold label
- Show that simple claims converge in 1 iteration, complex/ambiguous claims need 3+

**Key outputs:**
- Iteration histogram
- Scatter/box plot: iterations vs. claim complexity proxy
- Examples of 1-iteration vs. 3+-iteration claims

**Data needed:** ProClaim run logs with per-claim iteration counts (should be available from evidence state history).

---

## Proposed Logical Flow for Section 2

| Step | Demonstrates | Data required | Status |
|------|-------------|---------------|--------|
| Sandbox vs. gold accuracy | Single-paper ≠ consensus | 101 claims + gold | **Done** (S1 acc=0.644) |
| Top-5 vs. gold accuracy | More papers help but over-refute | Same | **Done** (S2 acc=0.653, SUPPORT recall 0.235) |
| Transition correctness | Naive retrieval introduces new errors | Same + transition analysis | **Done** (22 fixes vs 21 breaks = net +1) |
| Error taxonomy | *Why* naive retrieval fails → design motivation | Manual annotation ~15 cases | TODO |
| Iteration histogram | Claims need variable effort | ProClaim run logs | Check availability |

## Script

The analysis script is at `scripts/analysis/sandbox_vs_wild_gold_analysis.py`.

Preferred visualization script: `scripts/analysis/plot_sandbox_vs_wild_transitions.py`.

Alternative summary figure script: `scripts/analysis/plot_sandbox_vs_wild_gold_summary.py`.

Generated figure:
- `results/analysis/sandbox_vs_wild_gold_summary.pdf`
- `results/analysis/sandbox_vs_wild_gold_summary.png`

Run with:
```bash
uv run python scripts/analysis/sandbox_vs_wild_gold_analysis.py --include-s3
uv run python scripts/analysis/plot_sandbox_vs_wild_transitions.py
uv run python scripts/analysis/plot_sandbox_vs_wild_gold_summary.py
```

## Figure takeaway

The figure works best as a two-step motivation argument. Panel A shows that the sandbox setting is not enough for consensus building: using only the source paper reaches 64.4% accuracy against gold, so more than one third of claims still disagree with the curated consensus. But Panel A also shows that simply adding more retrieved papers does not solve the problem: moving from S1 to S2 changes 50 of 101 verdicts while improving accuracy by only one claim overall (0.644 to 0.653), and it sharply reduces SUPPORT recall from 0.529 to 0.235. Panel B explains why: naive retrieval produces almost as many newly broken predictions as fixes (21 breaks vs. 22 fixes), with the largest damage concentrated in SUPPORT→REFUTE transitions, where 11 of 17 changes are incorrect. The oracle-like S3 condition, which restores the reference paper alongside retrieved evidence, recovers both accuracy (0.703) and SUPPORT recall (0.706), indicating that the main bottleneck is not the absence of evidence but the failure to retrieve, preserve, and reconcile the right evidence. This is the motivation for knowledge construction rather than one-shot evidence pooling.

## Narrative for the Paper

**Paragraph 1 (Point 1 — sandbox is insufficient):** When the LLM sees only the source paper, it agrees with the gold consensus in just 64.4% of cases (S1). SUPPORT claims are the most affected: only 52.9% of truly supported claims are correctly identified, because the source paper often provides ambiguous or incomplete evidence for the specific interaction. This demonstrates that a single paper's perspective is insufficient for establishing scientific consensus.

**Paragraph 2 (Point 2 — naive retrieval doesn't solve it):** Naively retrieving 5 additional papers achieves almost identical accuracy (65.3%, Δ=+0.9pp). But this near-identical top-line masks a destructive trade-off: 50/101 verdicts change, with 22 fixes offset by 21 breaks. The dominant pattern is over-refutation — SUPPORT recall collapses from 53% to 24% as the LLM defaults to REFUTE when retrieved papers don't explicitly mention the claimed interaction (11 of 17 SUPPORT→REFUTE transitions are incorrect). Simultaneously, genuine UNCERTAIN cases are flattened to REFUTE (3/4 gold-UNCERTAIN → predicted REFUTE). The system resolves some ambiguity correctly (12 UNCERTAIN→REFUTE fixes) but at the cost of systematic false negatives. The REFUTE FPR balloons from 21% to 68%.

**Paragraph 3 (bridge to Evidence Programming):** Including the original reference paper alongside retrieved papers (S3) recovers accuracy to 70.3% and SUPPORT recall to 70.6%, showing that the critical supporting evidence *exists* but generic retrieval misses it. This gap between what *could* be found (S3) and what *is* found (S2) motivates iterative, gap-driven retrieval. The over-refutation pattern motivates structured evidence management with quality-weighted stance aggregation and an explicit sufficiency signal that distinguishes "no evidence found yet" from "evidence refutes the claim."

**Draft paragraph:**

Figure X makes the motivation concrete in two steps. First, the sandbox setting is insufficient for consensus building: when the model sees only the source paper, it matches the gold claim label on just 64.4% of cases, so more than one third of verdicts still disagree with the curated scientific consensus. Second, simply broadening retrieval does not solve this problem. Moving from the source paper to five retrieved abstracts changes 50 of 101 verdicts, but improves accuracy by only one claim overall, while collapsing SUPPORT recall from 0.529 to 0.235. The transition analysis shows why: naive retrieval introduces almost as many newly broken predictions as fixes, with the largest damage concentrated in SUPPORT-to-REFUTE flips, where 11 of 17 changes are wrong. By contrast, restoring the reference paper alongside the retrieved evidence raises accuracy to 0.703 and SUPPORT recall to 0.706, indicating that the core problem is not the absence of evidence, but the failure to retrieve, preserve, and reconcile the right evidence. This is exactly why in-the-wild claim verification requires knowledge construction rather than one-shot evidence pooling.