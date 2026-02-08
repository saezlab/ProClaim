# Revised Implementation Plan: Unified Metacognitive Evidence Verification

## What Changed and Why

This plan merges the former RES and NPES documents into a single framework. The restructuring directly addresses the six priority recommendations from the NeurIPS review.

| Review Priority | What Was Wrong | What Changes |
|---|---|---|
| **P1: Unify into one contribution** | Two loosely-coupled systems dilute novelty | Single framework with one Algorithm box; notebook/parallelism become implementation details |
| **P2: Formalize sufficiency-preserving compression** | Strongest novel claim had no theoretical backing | New Section 3 with formal invariant, proof sketch, and dedicated ablation |
| **P3: Update baselines and evaluation** | Missing 2025 competitors; SciFact too small | Add Sufficient Context, SAFE, FIRE, Stop-RAG; evaluate on SciFact-Open + SciClaimHunt |
| **P4: Address "consistently wrong" failure** | Self-consistency ≠ correctness unaddressed | New Section 4 with failure mode taxonomy and adversarial experiments |
| **P5: Generalization beyond scientific claims** | Application paper framing limits significance | Add one non-biomedical domain (FEVER or contract analysis) |
| **P6: Ablation design** | No ablations specified | Six targeted ablations isolating each component |

---

## 1. Unified Framework: The Metacognitive Control Loop

### 1.1 Contribution Hierarchy (for paper narrative)

**Primary contribution:** A metacognitive control loop where the system maintains explicit awareness of its own evidence state — what it knows, what it doesn't know, and whether it knows enough — and uses this awareness to autonomously drive claim verification to completion.

**Key technical novelty:** Sufficiency-preserving compression — a compression scheme with the formal invariant that compression must not decrease the system's ability to make correct verification decisions, as measured by a trained classifier.

**Supporting contributions (implementation, not novelty claims):**
- External mutable evidence state with concurrent agent access
- Gap-directed retrieval targeting identified evidence deficiencies
- The verification trace as an auditable, reproducible report artifact

### 1.2 Unified Algorithm

This replaces both the RES recursive loop AND the NPES orchestrator with a single specification. The old `SufficiencyOrchestrator.run()` and `recursive_evidence_collection()` collapse into one algorithm.

```
Algorithm 1: Metacognitive Evidence Verification

Input: claim c, retriever R, LLM M, sufficiency classifier φ, 
       compression budget B, threshold τ, max iterations T
Output: verdict v ∈ {SUPPORT, REFUTE, INSUFFICIENT}, confidence p, report

1.  subclaims ← DECOMPOSE(c, M)
2.  S ← INIT_STATE(c, subclaims)
3.  S ← RETRIEVE(S, R, query=c, k=5)          // seed evidence
4.  S ← PROCESS(S, M)                          // summarize + extract facts (parallel)

5.  for t = 1 to T do
6.      // ---- METACOGNITIVE ASSESSMENT ----
7.      σ ← EXTRACT_SIGNALS(S, c, M)           // confidence signals
8.      (p, label, gaps) ← φ(S, σ)             // sufficiency classifier (cheap, no LLM)
9.
10.     if p ≥ τ then
11.         v ← MAP_VERDICT(label)
12.         report ← GENERATE_REPORT(S)
13.         return (v, p, report)
14.
15.     if gaps = ∅ and t > T/2 then
16.         return (INSUFFICIENT, p, GENERATE_REPORT(S))
17.
18.     // ---- GAP-DIRECTED ACTION ----
19.     for g ∈ TOP_K(gaps, 2) do                // top-2 gaps
20.         q ← FORMULATE_QUERY(c, g, M)
21.         S ← RETRIEVE(S, R, query=q, k=3)
22.     S ← PROCESS(S, M)                        // parallel: summarize, extract, detect conflicts
23.
24.     // ---- SUFFICIENCY-PRESERVING COMPRESSION ----
25.     if TOKEN_COUNT(S) > B then
26.         S' ← COMPRESS(S, c, M)
27.         (p', _, _) ← φ(S', EXTRACT_SIGNALS(S', c, M))
28.         if p' ≥ p - ε then                   // compression invariant
29.             S ← S'
30.         else
31.             S ← COMPRESS_CONSERVATIVE(S, c, M)   // less aggressive fallback
32.
33. return (INSUFFICIENT, p, GENERATE_REPORT(S))
```

**Key design choices visible in the algorithm:**
- Line 8: Sufficiency check is a trained classifier, not an LLM call (cost: ~0)
- Lines 19-22: Retrieval is *directed by identified gaps*, not broad re-querying
- Lines 26-31: Compression has a formal guard — it only proceeds if the invariant holds
- Line 22: PROCESS is parallelizable (the old NPES agents run here, but as an implementation detail)

### 1.3 Where the Old Components Live

| Old Component | New Location | Role |
|---|---|---|
| RES recursive loop | Algorithm 1, lines 5-32 | The main loop IS the metacognitive control |
| RES sufficiency classifier | Algorithm 1, line 8 | φ — one component of the control loop |
| RES hierarchical state | Section 2.1 (Evidence State) | Implementation of S |
| RES compression | Section 3 (elevated to primary technical novelty) | Lines 24-31 |
| NPES EvidenceState class | Section 2.1 | Direct reuse as the state container |
| NPES parallel agents | Section 2.3 (PROCESS subroutine) | Implementation detail of line 22 |
| NPES notebook output | Section 2.4 (GENERATE_REPORT) | Report generation, not a contribution |
| NPES orchestrator | Absorbed into Algorithm 1 | No longer a separate system |

---

## 2. Evidence State and Operations

### 2.1 Evidence State Container

Retain the `EvidenceState` class from NPES essentially unchanged, but reframe it as a generic external state container rather than a "notebook." The key properties are:

- Thread-safe atomic operations (add_paper, add_fact, update_coverage, etc.)
- Hierarchical representation: Level 0 (raw papers, stored not loaded), Level 1 (summaries + facts), Level 2 (per-subclaim synthesis)
- Consistent snapshots for the classifier via `get_snapshot()`
- Audit log for report generation

**Implementation change:** Remove all Jupyter notebook framing from the code. Replace with a plain Python class called `EvidenceState`. The "notebook" language appears only when discussing the output report, not the state container.

```python
# Revised: evidence_state.py (was notebook_state.py)
# Same class structure as NPES EvidenceState, with these changes:

class EvidenceState:
    """
    External mutable evidence state with thread-safe atomic operations.
    Supports hierarchical representation for bounded-context LLM interaction.
    """
    # ... (same atomic operations as before)
    
    # NEW: Method required for compression invariant checking
    def clone(self) -> 'EvidenceState':
        """Deep copy for compression invariant testing."""
        with self._lock:
            new_state = EvidenceState(self.claim, list(self.subclaims))
            new_state.papers = dict(self.papers)
            new_state.facts = list(self.facts)
            new_state.conflicts = list(self.conflicts)
            new_state.coverage = dict(self.coverage)
            new_state.subclaim_evidence = {k: list(v) for k, v in self.subclaim_evidence.items()}
            new_state.synthesis = dict(self.synthesis)
            new_state.gap_analysis = self.gap_analysis
            return new_state
    
    # NEW: Context generation with budget awareness
    def get_context(self, budget_tokens: int) -> str:
        """
        Generate bounded context for LLM consumption.
        Always includes Level 2 synthesis; selectively includes Level 1 
        for uncovered subclaims if budget allows.
        """
        # (Absorb the HierarchicalEvidenceState.get_context() logic from RES)
        ...
    
    # NEW: Token accounting
    def token_count(self) -> int:
        """Total tokens across Level 1 + Level 2 representations."""
        ...
```

### 2.2 Tool Definitions

Retain the `NotebookTools` class from NPES, renamed to `VerificationTools`. Same methods:

- `retrieve_papers(query, top_k)` → adds to state
- `retrieve_for_gap(gap_description)` → LLM-formulated query, then retrieve
- `summarize_paper(pmid)` → atomic update to paper.summary
- `extract_facts(pmid)` → atomic adds to state.facts
- `compute_coverage(subclaim)` → updates state.coverage
- `detect_conflicts()` → adds to state.conflicts
- `synthesize_subclaim(subclaim)` → updates state.synthesis
- `analyze_gaps()` → updates state.gap_analysis

No structural changes needed here. These are the "tools" the system invokes.

### 2.3 PROCESS Subroutine (Parallel Execution)

The old NPES agent pool becomes the implementation of the PROCESS step in Algorithm 1. This is where parallelism lives — but it's now framed as an efficiency optimization, not a contribution.

```python
async def process_evidence(state: EvidenceState, tools: VerificationTools) -> None:
    """
    Process all unprocessed evidence in state.
    Runs summarization, fact extraction, coverage, conflict detection,
    synthesis, and gap analysis concurrently where possible.
    
    This is the parallel execution step from Algorithm 1, line 22.
    """
    # Phase 1 (parallel): Summarize unsummarized papers
    unsummarized = [pmid for pmid, p in state.papers.items() if p.summary is None]
    await asyncio.gather(*[tools.summarize_paper(pmid) for pmid in unsummarized])
    
    # Phase 2 (parallel): Extract facts from summarized papers not yet processed
    unextracted = [pmid for pmid, p in state.papers.items() 
                   if p.summary and pmid not in state._extracted_pmids]
    await asyncio.gather(*[tools.extract_facts(pmid) for pmid in unextracted])
    
    # Phase 3 (parallel): Update coverage + detect conflicts + synthesize + gap analysis
    await asyncio.gather(
        *[tools.compute_coverage(sc) for sc in state.subclaims],
        tools.detect_conflicts(),
        *[tools.synthesize_subclaim(sc) for sc in state.subclaims],
        tools.analyze_gaps()
    )
```

**What changed from NPES:** The seven named agent classes (RetrievalAgent, SummarizationAgent, etc.) are replaced by direct async function calls. The agent abstraction added complexity without research value. The `should_act()` / `act()` pattern is removed — the process function simply runs everything that needs running. This is simpler, clearer, and removes the need to justify "agents" as a contribution.

### 2.4 Report Generation

The GENERATE_REPORT function from Algorithm 1 produces the final verification report. This is essentially the `_generate_notebook()` method from the old NPES orchestrator. Keep the output format (markdown with subclaim analysis, evidence summary, key evidence, conflicts, gaps, references, execution log).

**Framing change:** Call it a "verification report" or "evidence trace," not a "notebook." The report is a property of the system (auditability, reproducibility) rather than a contribution.

---

## 3. Sufficiency-Preserving Compression (Primary Technical Novelty)

This section is entirely new. It elevates the compression mechanism from an implementation detail to the paper's key formal contribution.

### 3.1 Formal Definition

**Definition (Sufficiency-Preserving Compression).** Given evidence state S, claim c, and trained sufficiency classifier φ, a compression function COMPRESS is *ε-sufficiency-preserving* if:

```
P(φ(COMPRESS(S)) = φ(S)) ≥ 1 - ε
```

That is, compression changes the classifier's decision with probability at most ε.

A stronger variant, which we target in practice:

```
|confidence(φ(COMPRESS(S))) - confidence(φ(S))| ≤ ε
```

### 3.2 Compression Procedure

The compression has three levels of aggressiveness, tried in order:

**Level 1 — Fact deduplication (lossless):**
- Remove duplicate facts (same text, same source)
- Merge facts with identical content from different sources into a single fact with multiple citations
- This never reduces information content

**Level 2 — Synthesis refresh (lossy, usually safe):**
- For each subclaim, regenerate the Level 2 synthesis from current Level 1 facts
- Discard individual Level 1 facts for subclaims where coverage ≥ 0.8
- Retain Level 1 facts for uncovered subclaims (these are still needed for gap-filling)
- Check invariant: run φ on compressed state, verify confidence drop ≤ ε

**Level 3 — Aggressive synthesis (lossy, guarded):**
- Merge all Level 1 content into Level 2 synthesis
- Retain only paper metadata (PMID, title) at Level 0
- Level 1 becomes empty
- Check invariant: if violated, fall back to Level 2

```python
class SufficiencyPreservingCompressor:
    """
    Compresses evidence state while preserving the sufficiency classifier's
    ability to make correct decisions.
    
    Key invariant: |φ(S') - φ(S)| ≤ ε after compression.
    """
    
    def __init__(self, llm, classifier, epsilon: float = 0.05):
        self.llm = llm
        self.classifier = classifier
        self.epsilon = epsilon
    
    async def compress(self, state: EvidenceState, claim: str) -> EvidenceState:
        """
        Compress with invariant guarantee.
        Tries levels 1 → 2 → 3 in order, checking invariant after each.
        """
        # Get baseline sufficiency
        baseline_signals = await extract_signals(state, claim, self.llm)
        baseline_score = self.classifier(state, baseline_signals)['confidence'].item()
        
        # Level 1: Lossless deduplication
        compressed = self._deduplicate(state.clone())
        
        # Check if that was enough
        if compressed.token_count() <= state.context_budget:
            return compressed
        
        # Level 2: Synthesis refresh with selective fact retention
        compressed = await self._synthesis_refresh(compressed, claim)
        if self._check_invariant(compressed, claim, baseline_score):
            if compressed.token_count() <= state.context_budget:
                return compressed
        
        # Level 3: Aggressive synthesis (guard with invariant)
        aggressive = await self._aggressive_synthesis(compressed, claim)
        if self._check_invariant(aggressive, claim, baseline_score):
            return aggressive
        
        # Fallback: return Level 2 result even if over budget
        return compressed
    
    def _check_invariant(self, state: EvidenceState, claim: str, 
                         baseline_score: float) -> bool:
        """Verify compression preserved sufficiency within ε."""
        signals = extract_signals_sync(state, claim)  # cached signals, cheap
        new_score = self.classifier(state, signals)['confidence'].item()
        return abs(new_score - baseline_score) <= self.epsilon
    
    async def _synthesis_refresh(self, state: EvidenceState, claim: str) -> EvidenceState:
        """Level 2: Regenerate synthesis, selectively discard facts."""
        for subclaim in state.subclaims:
            facts = state.get_facts_for_subclaim(subclaim)
            
            if not facts:
                continue
            
            # Generate new synthesis
            facts_text = "\n".join(f"- {f.text} [{f.stance}] (PMID:{f.source_pmid})"
                                   for f in facts[:8])
            
            prompt = f"""Synthesize evidence for: "{subclaim}"
Facts:
{facts_text}
Write a 2-3 sentence synthesis preserving all decision-relevant information
(stance, key findings, source PMIDs):"""
            
            synthesis = await self.llm.generate_async(prompt, max_tokens=200)
            state.update_synthesis(subclaim, synthesis)
            
            # Discard Level 1 facts for well-covered subclaims
            if state.coverage.get(subclaim, 0) >= 0.8:
                state.facts = [f for f in state.facts 
                              if subclaim not in f.relevant_subclaims]
        
        return state
    
    async def _aggressive_synthesis(self, state: EvidenceState, claim: str) -> EvidenceState:
        """Level 3: Merge everything into synthesis, keep only metadata."""
        # Generate a single comprehensive synthesis per subclaim
        for subclaim in state.subclaims:
            all_evidence = []
            # Gather from both existing synthesis and remaining facts
            if state.synthesis.get(subclaim):
                all_evidence.append(f"Previous synthesis: {state.synthesis[subclaim]}")
            facts = state.get_facts_for_subclaim(subclaim)
            for f in facts:
                all_evidence.append(f"- {f.text} [{f.stance}] (PMID:{f.source_pmid})")
            
            if all_evidence:
                prompt = f"""Create a comprehensive evidence summary for: "{subclaim}"
All available evidence:
{chr(10).join(all_evidence)}
Summary (preserve all PMIDs, stances, and key quantitative findings):"""
                
                synthesis = await self.llm.generate_async(prompt, max_tokens=250)
                state.update_synthesis(subclaim, synthesis)
        
        # Clear all Level 1 facts
        state.facts = []
        
        return state
    
    def _deduplicate(self, state: EvidenceState) -> EvidenceState:
        """Level 1: Remove duplicate facts."""
        seen = set()
        unique_facts = []
        for fact in state.facts:
            key = (fact.text.strip().lower(), fact.stance)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        state.facts = unique_facts
        return state
```

### 3.3 Experiments Required for This Section

These experiments are critical — the review identified this as the strongest novel claim.

| Experiment | What It Shows | Comparison |
|---|---|---|
| **Compression ablation** | Sufficiency-preserving vs. standard summarization | Our compression vs. RECOMP abstractive vs. simple truncation vs. RAPTOR-style clustering |
| **Invariant violation rate** | How often does compression break decisions? | Measure across all claims: % where φ(S') ≠ φ(S) |
| **ε sensitivity** | Effect of tolerance parameter | Sweep ε ∈ {0.01, 0.02, 0.05, 0.10} — accuracy vs. compression ratio |
| **Compression ratio vs. accuracy** | Pareto frontier | Plot tokens-retained vs. verification accuracy for each method |
| **Qualitative analysis** | What information does compression discard vs. retain? | Show examples where standard summarization loses decision-relevant detail but ours doesn't |

---

## 4. Sufficiency Classifier (Supporting Component)

### 4.1 Training

Retain the self-consistency training approach from RES, with these changes:

**Training data generation** — same as before:
- For each claim, retrieve evidence at k ∈ {1, 3, 5, 10, 15, 20}
- At each k, sample 10 LLM decisions at temperature 0.7
- Label as SUFFICIENT_SUPPORT, SUFFICIENT_REFUTE, or INSUFFICIENT based on consistency + correctness

**Classifier architecture** — same lightweight MLP from RES, but with expanded features:

```python
class SufficiencyClassifier(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sufficiency_head = nn.Linear(hidden_dim, 3)   # SUFF_SUP, SUFF_REF, INSUFF
        self.confidence_head = nn.Linear(hidden_dim, 1)    # calibrated confidence
        self.gap_head = nn.Linear(hidden_dim, 8)           # gap type prediction
    
    def extract_features(self, state: EvidenceState, signals: Dict) -> torch.Tensor:
        """
        Fixed-size feature vector from state + signals.
        NO LLM calls — uses pre-computed values only.
        """
        features = [
            # Coverage features (from evidence state)
            min(state.coverage.values()) if state.coverage else 0,
            sum(state.coverage.values()) / max(len(state.coverage), 1),
            sum(1 for v in state.coverage.values() if v >= 0.7) / max(len(state.coverage), 1),
            
            # Quantity features (normalized)
            min(len(state.papers) / 20, 1.0),
            min(len(state.facts) / 50, 1.0),
            
            # Conflict features
            len(state.conflicts),
            max((c.severity for c in state.conflicts), default=0),
            
            # Confidence signals (from pre-extraction)
            signals.get('verbalized', 50) / 100,
            signals.get('consistency', 0),
            signals.get('entropy', 1.0),
            min(signals.get('num_gaps', 5) / 5, 1.0),
            
            # Evidence quality
            sum(1 for f in state.facts if f.stance == 'SUPPORT') / max(len(state.facts), 1),
            sum(1 for f in state.facts if f.stance == 'REFUTE') / max(len(state.facts), 1),
            len(set(f.source_pmid for f in state.facts)) / max(len(state.papers), 1),
            
            # Embedding similarity (if available)
            signals.get('max_similarity', 0),
            signals.get('mean_similarity', 0),
        ]
        return torch.tensor(features, dtype=torch.float32)
```

### 4.2 Addressing the "Consistently Wrong" Failure Mode

**NEW SECTION — directly addresses Review Priority 4.**

The core risk: self-consistency can be high when the LLM is confidently wrong (e.g., strong priors override weak evidence, or misleading evidence is retrieved).

**Failure mode taxonomy:**

| Mode | Description | Detection Strategy |
|---|---|---|
| **Prior-dominated** | LLM's training knowledge overrides retrieved evidence | Check if decision is same with and without evidence |
| **Misleading evidence** | Retrieved papers are topically relevant but don't address the claim | Check if supporting facts are actually about the claim's specific mechanism |
| **Echo chamber** | Multiple papers from same research group all agree | Check source diversity (unique author groups) |
| **Partial evidence** | Evidence supports a related but different claim | Check subclaim decomposition quality |

**Experiments for this section:**

1. **Prior override test:** For each claim, compare classifier decision with evidence vs. classifier decision with empty evidence state. If they agree, the evidence didn't matter — flag as potentially prior-dominated. Measure what fraction of "sufficient" decisions are actually prior-dominated.

2. **Adversarial evidence injection:** Take claims where the system reaches the correct verdict, then inject 1-3 papers with opposing conclusions. Measure: (a) does the classifier's sufficiency score drop appropriately? (b) does the system request more evidence to resolve the conflict?

3. **Source diversity analysis:** For claims labeled SUFFICIENT, measure the number of unique research groups contributing evidence. Correlate with correctness: do diverse-source decisions have higher accuracy than single-source decisions?

4. **Calibration curve:** Plot predicted confidence vs. actual accuracy across all claims. A well-calibrated classifier should have predicted confidence ≈ actual accuracy. Report Expected Calibration Error (ECE).

```python
def test_prior_override(claim, state, classifier, llm):
    """Test whether the classifier is relying on LLM priors rather than evidence."""
    # Decision with evidence
    signals_with = extract_signals(state, claim, llm)
    decision_with = classifier(state, signals_with)
    
    # Decision with empty state
    empty_state = EvidenceState(claim, state.subclaims)
    signals_without = extract_signals(empty_state, claim, llm)
    decision_without = classifier(empty_state, signals_without)
    
    # If decisions agree, evidence didn't matter
    prior_dominated = (decision_with['sufficiency'].argmax() == 
                       decision_without['sufficiency'].argmax())
    
    return {
        'prior_dominated': prior_dominated,
        'confidence_with': decision_with['confidence'].item(),
        'confidence_without': decision_without['confidence'].item(),
        'confidence_delta': (decision_with['confidence'] - decision_without['confidence']).item()
    }
```

---

## 5. Evaluation Plan

### 5.1 Datasets

| Dataset | Size | Scope | Why Include |
|---|---|---|---|
| **SciFact** (Wadden et al., 2020) | ~1,400 claims, 5K abstracts | Biomedical, abstract-level | Standard benchmark; enables comparison with all prior work |
| **SciFact-Open** (Wadden et al., 2022) | ~1,400 claims, 500K abstracts | Biomedical, open-domain | Tests retrieval at scale; stresses gap-directed retrieval and compression |
| **SciClaimHunt** (Bose et al., 2025) | Larger, full-text claims | Multi-domain scientific | Tests full-text evidence, moves beyond abstract limitation |
| **FEVER** (Thorne et al., 2018) | 185K claims, Wikipedia | General factual | Tests generalization beyond scientific domain (Review P5) |

**SciFact-Open is the primary evaluation dataset.** It has 500K abstracts, multiple evidence documents per claim, and variable evidence difficulty — exactly the conditions where our system's advantages (gap-directed retrieval, compression, iterative collection) should matter most.

### 5.2 Baselines

**Tier 1 — Must include (directly compete on the same claims):**

| Baseline | Why Critical | Source |
|---|---|---|
| **"Sufficient Context"** (Joren et al., ICLR 2025) | Directly solves sufficiency classification; the paper that most threatens our novelty | Must show our lightweight trained classifier matches or beats their prompted Gemini |
| **SAFE** (Wei et al., NeurIPS 2024) | SOTA for agentic fact verification; single-agent baseline | Must show multi-step metacognitive control outperforms SAFE's single-agent pipeline |
| **Self-RAG** (Asai et al., ICLR 2024) | Strongest learned retrieval decision baseline | Must show our approach outperforms reflection tokens |
| **MultiVerS** (Wadden et al., 2022) | SOTA on SciFact leaderboard (F1 ~0.73) | Must beat the fine-tuned specialist |

**Tier 2 — Should include (cover the landscape):**

| Baseline | Why Relevant |
|---|---|
| **Stop-RAG** (2025) | Q-learning for retrieval stopping — different stopping mechanism |
| **FIRE** (Xie et al., NAACL 2025) | Iterative retrieval-and-verification with adaptive query generation |
| **RAPTOR** (Sarthi et al., ICLR 2024) | Hierarchical compression baseline |
| **Fixed-k RAG** (k=5, k=10, k=20) | Non-adaptive baselines showing the value of learned stopping |

**Tier 3 — Nice to have:**

| Baseline | Why |
|---|---|
| **RECOMP** (Xu et al., ICLR 2024) | Compression-specific baseline for the compression ablation |
| **IRCoT** (Trivedi et al., 2023) | Interleaved retrieval + CoT, fixed iterations |

### 5.3 Metrics

| Metric | What It Measures |
|---|---|
| **Label F1** | Verification accuracy (SUPPORT/REFUTE/NEI) |
| **Label+Rationale F1** | Accuracy with correct evidence sentences |
| **Abstract-level retrieval F1** | Finding the right papers |
| **Cost (tokens/claim)** | Computational efficiency |
| **LLM calls/claim** | API cost proxy |
| **Sufficiency rounds** | How many iterations before stopping |
| **Compression ratio** | Tokens retained after compression |
| **ECE** | Calibration of confidence scores |

**Primary comparison:** Label F1 vs. tokens/claim (cost-accuracy Pareto frontier). This follows HippoRAG's NeurIPS 2024 precedent of showing efficiency alongside accuracy.

### 5.4 Ablation Studies

Six ablations, each removing one component to isolate its contribution:

| Ablation | What's Removed | What It Tests |
|---|---|---|
| **A1: No sufficiency classifier** | Replace φ with fixed 5 iterations | Value of learned stopping |
| **A2: No gap-directed retrieval** | Replace gap queries with re-queries of the original claim | Value of targeted evidence gathering |
| **A3: No compression** | Let context grow unbounded (truncate if over limit) | Value of sufficiency-preserving compression |
| **A4: No parallel processing** | Run all operations sequentially | Speed vs. quality tradeoff of parallelism |
| **A5: No conflict detection** | Remove conflict detection agent | Value of explicit conflict awareness |
| **A6: Standard compression** | Replace sufficiency-preserving compression with RECOMP-style abstractive compression | Value of the sufficiency invariant specifically |

**A6 is the most important ablation.** It directly tests whether sufficiency-aware compression is better than standard compression — this is the paper's strongest novel claim.

---

## 6. Implementation Stages

### Stage 1: Evidence State + Tools (Week 1-2)

**Goal:** Working evidence state container with all tool operations.

**Tasks:**
- [ ] Implement `EvidenceState` class (from NPES, renamed)
- [ ] Implement `VerificationTools` class (from NPES, renamed)
- [ ] Implement claim decomposition (LLM prompt → subclaims)
- [ ] Set up PubMed retrieval (API integration)
- [ ] Set up LLM integration (async, batched)
- [ ] Unit tests for atomic operations and thread safety

**Deliverable:** Can retrieve papers, extract facts, compute coverage for a claim. No control loop yet.

### Stage 2: Sufficiency Classifier Training (Week 3-4)

**Goal:** Trained classifier that predicts sufficiency from evidence state features.

**Tasks:**
- [ ] Generate training data on SciFact: for each claim × each k ∈ {1,3,5,10,15,20}, run 10 LLM samples, record consistency + correctness
- [ ] Implement feature extraction (16 features from state + signals)
- [ ] Train MLP classifier (multi-task: sufficiency + confidence + gaps)
- [ ] Evaluate classifier accuracy on held-out claims
- [ ] Calibration analysis (ECE, reliability diagram)
- [ ] Prior override test (Section 4.2)

**Deliverable:** Trained classifier checkpoint. Classifier evaluation report showing accuracy, calibration, and failure mode analysis.

### Stage 3: Metacognitive Control Loop (Week 5-6)

**Goal:** Full Algorithm 1 running end-to-end.

**Tasks:**
- [ ] Implement the main loop (Algorithm 1)
- [ ] Implement PROCESS subroutine (parallel execution)
- [ ] Implement gap-directed retrieval (LLM query formulation → targeted search)
- [ ] Implement report generation
- [ ] End-to-end test on 50 SciFact claims
- [ ] Tune hyperparameters: threshold τ, max iterations T, context budget B

**Deliverable:** Working system that takes a claim and produces a verdict + confidence + report.

### Stage 4: Sufficiency-Preserving Compression (Week 7-8)

**Goal:** Compression with formal invariant, plus ablation evidence.

**Tasks:**
- [ ] Implement three-level compression (dedup → synthesis refresh → aggressive)
- [ ] Implement invariant checking (φ(S') vs φ(S))
- [ ] Run compression ablation: our method vs. RECOMP-style vs. truncation vs. RAPTOR-style
- [ ] Measure invariant violation rate across all SciFact claims
- [ ] ε sensitivity sweep
- [ ] Compression ratio vs. accuracy Pareto plot

**Deliverable:** Compression module with empirical invariant validation. Ablation results showing advantage over standard compression.

### Stage 5: Full Evaluation (Week 9-11)

**Goal:** Complete experimental results on all datasets and baselines.

**Tasks:**
- [ ] Run on SciFact (standard leaderboard comparison)
- [ ] Run on SciFact-Open (500K abstracts, open-domain)
- [ ] Run on SciClaimHunt (full-text claims)
- [ ] Run on FEVER subset (generalization test)
- [ ] Implement and run all Tier 1 baselines
- [ ] Implement and run all Tier 2 baselines (as time allows)
- [ ] Run all 6 ablations
- [ ] Cost-accuracy Pareto analysis
- [ ] Adversarial evidence injection experiments
- [ ] Source diversity analysis
- [ ] Qualitative examples (what does the system do well? where does it fail?)

**Deliverable:** Complete results tables, Pareto plots, ablation tables, qualitative analysis.

### Stage 6: Writing (Week 12-13)

**Goal:** Complete paper draft.

**Paper structure:**

1. **Introduction** — Frame the problem as "agentic systems need metacognitive control to know when they have enough evidence." Motivate with the scientific verification use case. State the primary contribution (metacognitive control loop) and key technical novelty (sufficiency-preserving compression).

2. **Related Work** — Organize by: (a) Adaptive retrieval and learned stopping (Self-RAG, Stop-RAG, Sufficient Context, RASC, SEARAG), (b) Multi-agent verification (SAFE, LoCal, MA-RAG), (c) Evidence compression (RAPTOR, RECOMP, xRAG). Clearly articulate what our framework adds that none of these have.

3. **Method** — Algorithm 1 as the centerpiece. Subsections: evidence state, sufficiency classifier, gap-directed retrieval, sufficiency-preserving compression (with formal definition).

4. **Experiments** — Main results, ablations, compression analysis, failure mode analysis, cost-accuracy Pareto.

5. **Analysis** — Qualitative examples, when does the system fail, calibration analysis, generalization to FEVER.

6. **Conclusion** — The metacognitive control paradigm for agentic evidence gathering.

---

## 7. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| "Sufficient Context" comparison is unfavorable | High | Their method uses Gemini 1.5 Pro (expensive). Our advantage is cost: lightweight classifier vs. full LLM call. Frame as efficiency contribution if accuracy is similar. |
| SciFact too small to show compression advantage | Medium | SciFact-Open (500K abstracts) is the primary dataset. If compression advantage doesn't show on small SciFact, it should show when retrieval returns many papers on SciFact-Open. |
| Self-consistency fails on misleading evidence | Medium | Section 4 failure mode analysis. If failure rate is high, report it honestly and propose mitigations (source diversity weighting, prior override detection). |
| Parallel processing doesn't improve quality, only speed | Low | Fine — frame parallelism as an implementation efficiency, not a quality contribution. The paper's claims don't depend on parallelism improving accuracy. |
| Compression invariant is violated frequently | High | If ε=0.05 is violated >10% of the time, increase ε or improve the conservative fallback. Report violation rates transparently. |
| FEVER generalization is weak | Medium | Scientific claims and Wikipedia claims are structurally different. If FEVER results are mediocre, present as "the framework is domain-specific but the control loop architecture is general" and call for future work on domain adaptation. |

---

## 8. Differences from Original Plans

### Removed
- "Notebook" as a claimed contribution (now just an implementation choice)
- Seven named agent classes (replaced with direct async functions)
- NPES as a separate system
- RES as a separate system
- Claims about parallelism being novel

### Added
- Formal compression invariant (Definition + experiments)
- Failure mode taxonomy and adversarial experiments
- Expanded baselines (Sufficient Context, SAFE, FIRE, Stop-RAG)
- Expanded datasets (SciFact-Open, SciClaimHunt, FEVER)
- Six structured ablations
- Cost-accuracy Pareto analysis
- Calibration analysis (ECE)
- Prior override detection
- Clear contribution hierarchy (primary → supporting)

### Restructured
- Algorithm 1 unifies both old systems into one specification
- Sufficiency-preserving compression elevated to primary technical novelty
- Metacognitive control loop framed as the architectural contribution
- Parallel execution demoted to implementation detail
- Report generation demoted to system property
