# NeurIPS Paper Review & Benchmark Analysis
**Date:** 2026-03-30  
**Branch:** exp/init

---

## Summary

Conducted an impartial NeurIPS reviewer assessment of the Evidence Programming paper draft, followed by iterative analysis of how to strengthen the experimental design. The review identified critical gaps (missing baselines, incomplete benchmarks, placeholder text) and revised scores upward conditionally on completion of key items. A systematic search was conducted for non-biomedical, open-retrieval KG databases that could serve as transferability benchmarks. Three candidate datasets were evaluated in detail: Matter-of-Fact, CLIMATE-FEVER, and CIViC-Fact.

---

## Review Scores

### Initial Assessment (draft as submitted)

| Criterion | Score | Rationale |
|-----------|:-----:|-----------|
| Novelty | 6/10 | Conceptually close to FIRE; code-generation interface and persistent state are real but not yet empirically validated |
| Clarity | 4/10 | Placeholder text (`[X]`, dangling headings, undefined "Evidence Programming"), no system diagram |
| Soundness | 3/10 | Only 14 SIGNOR edges evaluated; no baselines run; connectomeDB not evaluated |
| Impact | 5/10 | Biology-only; complex stack limits reproducibility |
| **Overall** | **4/10** | Pre-submission stage; would receive reject at NeurIPS |

### Revised Assessment (assuming critical items completed + significant performance gaps)

| Criterion | Score | Notes |
|-----------|:-----:|-------|
| Novelty | 7/10 | Ablations showing each component is load-bearing would push to 7.5 |
| Clarity | 6–7/10 | Ceiling depends on system diagram, crisp EP definition, FIRE distinction table |
| Soundness | 7/10 | Capped by missing SciFact-Open (withdrawn—wrong task), backbone fairness risk |
| Impact | 7/10 | Datasets become reusable artifact; classifier transferable |
| **Overall** | **6–7/10** | Borderline accept / weak accept |

**Path to 8+:** SciFact-Open withdrawn as inappropriate (constrained corpus, passage-level). Instead: cost-accuracy frontier plot, clean ablation table (remove classifier / gap search / structured state each independently), one transferability experiment.

---

## Critical Actions (Must-Do Before Submission)

1. **Run connectomeDB\* at scale** (~450 claims) — this is the primary benchmark; the paper cannot stand without it
2. **Implement 4 baselines minimum:** LLM-only, Fixed-10 RAG, ReAct (most important), SAFE or FIRE
3. **Fill all placeholders** — remove `[X]`, `"Sufficiency classifier -"`, `"[dependent on final implementation]"`
4. **Add datasets as an explicit 4th contribution** — currently absent from the contributions list

---

## SciFact-Open: Withdrawn as Benchmark

**Decision:** Do not run SciFact-Open.  
**Rationale:** SciFact-Open tests constrained retrieval over a fixed 500K-abstract corpus with passage-level labels. Evidence Programming's core contribution is open-retrieval evidence *construction* from unbounded literature. Running SciFact-Open would evaluate a neutered version of the system (no iterative gap-driven search, no sufficiency classifier driving termination, no open PubMed queries). Reviewer pushback on absence can be addressed by explaining the paradigm mismatch directly.

---

## Non-Biomedical Benchmark Search: Findings

### Why the space is sparse

Expert-curated KGs with literature-linked relational claims and verified positive/negative labels are concentrated in biomedicine (NIH/EBI/EMBL funding infrastructure). Other sciences store knowledge as measurements, crystal structures, or spectra—not as verifiable relational claims with gold labels.

### Candidate Datasets Evaluated

#### 1. Matter-of-Fact (Jansen et al., EMNLP 2025) ✅ Recommended
- **Domain:** Materials science (superconductors, semiconductors, batteries, aerospace)
- **Size:** 8.4k claims, 4.4k test; binary TRUE/FALSE (feasible/infeasible); 50/50 balanced
- **Labels:** LLM-extracted from full-text arXiv papers; domain expert audit on 100 claims: κ=0.86, 99% agreement after resolution
- **Retrieval:** Open (Semantic Scholar API); temporally filtered by knowledge cutoff date
- **Best baseline:** 72% accuracy (Claude Sonnet 3.7 + ICL + RAG). Chance = 50%
- **Key finding in paper:** Multi-round naive retrieval (≤5 rounds) did *not* improve performance — precisely the gap structured evidence construction is designed to fill
- **Task alignment:** Feasibility assessment (can this be true given prior knowledge?) vs. claim verification. TRUE → SUPPORT, FALSE → REFUTE mapping is valid *with temporal filter ON*; this is the hard regime (72% ceiling). Without temporal filter, RAG achieves 87–100% (too easy).
- **Adaptations needed:** Swap PubMed → Semantic Scholar API; add knowledge-cutoff date filtering; map SUPPORT/REFUTE/NEI to feasible/infeasible
- **Framing:** Position as transferability experiment, not primary benchmark. If system beats 72%, it demonstrates structured iteration generalizes across domains and task formulations.

#### 2. CLIMATE-FEVER (Diggelmann et al., 2021) ❌ Rejected
- **Domain:** Climate science; 1,535 claims from internet sources
- **Labels:** Derived from majority vote across ≤5 Wikipedia sentence annotations per claim
- **Critical problems:**
  - Average inter-annotator agreement: Krippendorff α = 0.334 ("fair"); slices 3 and 5 have α = 0.106 / 0.091 (near-random)
  - Many slices had only **2 voters** — insufficient for reliable majority voting
  - Fragile aggregation: a single SUPPORTS micro-verdict against 4 NOT_ENOUGH_INFO is sufficient to set claim label to SUPPORTS
  - Labels conditioned on Wikipedia BM25 retrieval, not scientific literature — your system searching PubMed would be systematically penalized for correct answers not findable in Wikipedia
  - DISPUTED claims (9.97%) excluded from evaluation, removing cases where conflict detection is most valuable
  - FEVER-trained baseline achieves only 38.78% accuracy on CLIMATE-FEVER (barely above chance), confirming the Wikipedia retrieval mismatch

#### 3. CIViC-Fact (Reisle et al., 2025) ❌ Rejected
- **Domain:** Cancer variant interpretation ("BRAF V600E predicts sensitivity to vemurafenib")
- **Size:** 1,434 claim/evidence pairs (1,119 distinct claims); expert-curated
- **Labels:** SUPPORTS / REFUTES / NEI — but these are **claim/evidence pair-level stance labels**, not claim-level verdicts
- **Critical problems:**
  - No claim-level ground truth: the dataset labels whether a *specific pre-linked passage* supports/refutes a claim, not whether the claim is true or false in the literature overall. Evidence Programming emits a claim-level verdict; there is no gold label to evaluate it against.
  - Evidence is pre-selected per CIViC entry (curators manually annotate sentence spans via hypothes.is) — not open retrieval. Running EP on open PubMed search produces a fundamentally different retrieval set, making comparison invalid.
  - The task is stance classification (retrieval already done), not evidence construction + verdict — the same paradigm mismatch that disqualifies SciFact-Open.
  - IAA of 0.65 (Fleiss' κ) on pair-level stance is reasonable, but irrelevant if the unit of evaluation is wrong.

---

## Recommended Benchmark Portfolio

| Benchmark | Domain | Claim Type | Size | Open Retrieval | Role |
|-----------|--------|-----------|------|:--------------:|------|
| **connectomeDB\*** | Ligand-receptor biology | Physical binding | ~450 | ✅ | Primary (largest, cleanest labels) |
| **SIGNOR\*** | Kinase signaling | Regulatory mechanism | 67 | ✅ | Secondary (molecular mechanisms) |
| ~~CIViC-Fact~~ | ~~Cancer genomics~~ | ~~Clinical evidence~~ | ~~1,119~~ | ❌ | ~~Rejected — no claim-level labels, pre-linked evidence~~ |
| **Matter-of-Fact** | Materials science | Feasibility (TRUE→SUPPORT) | 4.4k test | ✅ | Transferability experiment (non-biomedical) |

**Framing for paper:** Two biology benchmarks test qualitatively different *reasoning patterns* (physical interaction, enzymatic mechanism). Matter-of-Fact adds a genuine non-biomedical domain. The third biology-domain slot (claim-type diversity) is currently vacant — CIViC-Fact was rejected; a replacement with claim-level labels and open retrieval is needed if reviewers push on biology-domain coverage.

---

## Key Design Decisions

- **SciFact-Open excluded by design**, not omission — paradigm mismatch is the justification; preempt reviewer concern with one explicit sentence in §4
- **Matter-of-Fact temporal filter must be ON** — without it, baselines reach 87–100% and there is no room for improvement; with it, the 72% ceiling creates the experimental headroom Evidence Programming needs
- **CLIMATE-FEVER rejected** on label reliability grounds, not domain grounds — the problem is Wikipedia-conditioned labels with α=0.334 IAA, not climate science as a domain
- **CIViC-Fact rejected** on structural grounds — no claim-level labels (only pair-level stance) and pre-linked evidence (not open retrieval). The same paradigm mismatch as SciFact-Open: the dataset evaluates stance classification given retrieved evidence, while Evidence Programming evaluates open-retrieval evidence construction + claim verdict. No gold label exists to compare against EP's output.
- **Claim-type diversity argument** is stronger than domain diversity for an evidence construction system — use this framing in §1/§3 to preempt "biology-only" reviewer concern
- **Transferability statement should be architectural, not just empirical:** sufficiency classifier features (semantic similarity, NLI ratios, source diversity, bibliometrics) are domain-independent by construction; only the claim template is domain-specific

---

## Remaining Risks for NeurIPS Acceptance

| Risk | Mitigation |
|------|-----------|
| Reviewer demands SciFact-Open | One-sentence paradigm mismatch explanation in §4 |
| Backbone fairness (Claude Sonnet vs. weaker LLM for baselines) | Use same backbone for all non-specialised baselines; report backbone-controlled ablations |
| Sufficiency classifier circularity (trained on agent-labeled data) | Explicitly document label provenance and run cross-validation; report AUC |
| Matter-of-Fact task mismatch (feasibility ≠ verification) | Acknowledge in one sentence: "feasibility is a weaker condition than direct evidential support, making this a conservative test" |
| connectomeDB evaluation not run | **Blocking** — must be completed before submission |
| Third biology-domain benchmark vacant (CIViC-Fact rejected) | Find a replacement with claim-level labels + open retrieval, or strengthen claim-type diversity argument for two benchmarks only |
