# Core Functionalities Implementation Plan

This document details the **minimal backbone** implementation for the Metacognitive Evidence Verification system. The goal is to get a working end-to-end pipeline as quickly as possible, with clean extension points for later refinement. It is derived from the overall [implementation_plan.md](implementation_plan.md).

**Guiding principle:** Implement the simplest version of each component that (a) has the correct interface for later substitution and (b) lets the next component in the chain be developed and tested. Defer complexity (thread safety, parallelism, trained models, multi-level compression) until the backbone works.

---

## Overview

The dependency chain remains:

```
1. EvidenceState  →  2. VerificationTools  →  3. Sufficiency Classifier  →  4. Control Loop  →  5. Compression
```

For the backbone, each component starts at its **minimum viable form**:

| Core | Minimal Form | Deferred to Extensions |
|---|---|---|
| 1. EvidenceState | Plain dataclass container, no locking | Thread safety, audit log, budget-aware `get_context` |
| 2. VerificationTools | 3 essential tools (retrieve, summarize, extract_facts) | 5 advanced tools (coverage, conflicts, synthesis, gaps, gap-retrieval) |
| 3. Classifier | Heuristic function matching the trained-model interface | MLP training, signal extraction, calibration |
| 4. Control Loop | Fixed-iteration loop with heuristic stopping | Gap-directed retrieval, adaptive stopping |
| 5. Compression | L1 deduplication only | L2/L3 lossy compression, invariant checking |

---

## 1. EvidenceState — Central Data Structure

**Target file:** `src/pkevolve/verification/evidence_state.py`

### Minimal Implementation

A plain Python container. No locking — the backbone runs sequentially. The interface is designed so locking can be added later without changing callers.

### Data Model

```python
@dataclass
class PaperRecord:
    pmid: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    summary: Optional[str] = None          # L1

@dataclass
class Fact:
    text: str
    stance: str                             # SUPPORT | REFUTE | NEUTRAL
    source_pmid: str
    relevant_subclaims: List[str] = field(default_factory=list)
```

### Required Interface (backbone)

| Method | Description | Notes |
|---|---|---|
| `__init__(claim, subclaims)` | Initialize with claim and subclaims | Subclaims can be `[claim]` initially (no decomposition) |
| `add_paper(paper: PaperRecord)` | Append to `self.papers` dict (keyed by PMID) | No locking needed yet |
| `add_fact(fact: Fact)` | Append to `self.facts` list | |
| `clone() -> EvidenceState` | Deep copy via `copy.deepcopy` | Needed by compression even in backbone |
| `token_count() -> int` | Sum `len(text)` across all summaries + facts (char-based approximation) | tiktoken encoding deferred |
| `get_context() -> str` | Concatenate all summaries and facts into a single string | Budget-aware version deferred |

### Deferred to Extensions

- Thread safety (`threading.RLock` on all mutations)
- `Conflict` dataclass and `add_conflict()`
- `update_coverage()`, `update_synthesis()`
- `get_snapshot()` (frozen dict for classifier)
- `get_context(budget_tokens)` with L2-first priority
- Audit log (`self._audit_log`)
- tiktoken-based `token_count()`

### Acceptance Criteria

- [ ] Can store papers and facts, retrieve them, and produce a text context
- [ ] `clone()` produces an independent copy

---

## 2. VerificationTools — State-Mutating Operations

**Target file:** `src/pkevolve/verification/tools.py`
**Depends on:** EvidenceState

### Minimal Implementation — 3 Essential Tools

The backbone needs only the tools required to populate L0 and L1 of the evidence state. The 5 advanced tools (coverage, conflicts, synthesis, gaps, gap-retrieval) are stubs that can be filled in later.

| Tool | Backbone | Status |
|---|---|---|
| `retrieve_papers(query, top_k)` | Wraps `RelevancePubMedSearcher`, converts to `PaperRecord` | **Implement** |
| `summarize_paper(pmid)` | LLM call to summarize abstract/full-text → `paper.summary` | **Implement** |
| `extract_facts(pmid)` | LLM call to extract stance-labeled facts → `state.add_fact()` | **Implement** |
| `retrieve_for_gap(gap)` | Calls `retrieve_papers(gap, top_k)` directly (no LLM query reformulation) | **Stub** — pass-through |
| `compute_coverage(subclaim)` | Returns 0.5 (neutral) | **Stub** |
| `detect_conflicts()` | No-op | **Stub** |
| `synthesize_subclaim(subclaim)` | Concatenates facts for that subclaim | **Stub** |
| `analyze_gaps()` | Returns empty list | **Stub** |

### Constructor

```python
class VerificationTools:
    def __init__(self, state: EvidenceState, llm_client: OpenAI, model: str,
                 pubmed_searcher: RelevancePubMedSearcher):
        self.state = state
        self.client = llm_client
        self.model = model
        self.searcher = pubmed_searcher
```

Follows the project's existing LLM client pattern (`scripts/qa_pipeline/run_qa.py` → `setup_client_and_evaluator()`). All calls use `client.chat.completions.create()`.

### Reuse of Existing Code

| Existing Component | Location | Reuse |
|---|---|---|
| `RelevancePubMedSearcher` | `src/pkevolve/search/custom_pubmed.py` | Direct — `retrieve_papers()` wraps it |
| OpenAI client pattern | `src/pkevolve/llm/evaluator.py` | LLM calls follow same `client.chat.completions.create()` pattern |
| `paper_search_agent.py` | `src/pkevolve/search/paper_search_agent.py` | Full-text retrieval functions (deferred — PubMed abstracts sufficient for backbone) |

### Deferred to Extensions

- LLM-formulated gap queries in `retrieve_for_gap()`
- LLM-based coverage scoring in `compute_coverage()`
- LLM-based conflict detection in `detect_conflicts()`
- LLM-based synthesis in `synthesize_subclaim()`
- LLM-based gap analysis in `analyze_gaps()`
- Async versions of all tools
- Full-text retrieval (PDF download, PMC)
- Rate limiting (reuse `WebSearchAssistant._rate_limit()` pattern)

### Acceptance Criteria

- [ ] `retrieve_papers("p53 apoptosis", 5)` adds papers to state
- [ ] `summarize_paper(pmid)` sets `paper.summary` for a paper in state
- [ ] `extract_facts(pmid)` adds facts with SUPPORT/REFUTE/NEUTRAL stances
- [ ] Stub tools exist and are callable (return defaults without errors)

---

## 3. Sufficiency Classifier φ — Learned Stopping Criterion

**Target file:** `src/pkevolve/verification/classifier.py`
**Depends on:** EvidenceState

### Minimal Implementation — Heuristic Classifier

The backbone uses a **rule-based heuristic** that matches the interface of the future trained MLP. The control loop calls `classifier(state) -> dict` regardless of whether it's a heuristic or a neural network.

```python
class SufficiencyClassifier:
    """
    Heuristic classifier (backbone).
    Drop-in replacement for the trained MLP — same interface.
    """
    def __call__(self, state: EvidenceState) -> dict:
        n_papers = len(state.papers)
        n_facts = len(state.facts)
        support = sum(1 for f in state.facts if f.stance == 'SUPPORT')
        refute = sum(1 for f in state.facts if f.stance == 'REFUTE')

        # Simple heuristic: sufficient if we have ≥3 facts and clear majority
        total = max(n_facts, 1)
        if n_facts >= 3 and (support / total >= 0.7):
            return {'label': 'SUFFICIENT_SUPPORT', 'confidence': support / total, 'gaps': []}
        elif n_facts >= 3 and (refute / total >= 0.7):
            return {'label': 'SUFFICIENT_REFUTE', 'confidence': refute / total, 'gaps': []}
        else:
            return {'label': 'INSUFFICIENT', 'confidence': max(support, refute) / total, 'gaps': ['need_more_evidence']}
```

**Key:** The return dict has keys `label`, `confidence`, `gaps` — the same keys the control loop reads. When the trained MLP replaces this, no controller code changes.

### Deferred to Extensions

The full classifier from implementation_plan.md Section 4.1:
- `SufficiencyClassifier(nn.Module)` with 16-feature input, 3 output heads
- Signal extraction (`EXTRACT_SIGNALS`) with LLM calls for verbalized confidence, self-consistency, entropy
- Training data generation (k ∈ {1,3,5,10,15,20}, 10 samples per k)
- Calibration analysis (ECE)
- Prior override test (Section 4.2)
- Gap type prediction (8-class multi-label)

All 16 features from the full plan are documented here for reference when extending:

| # | Feature | Backbone Status |
|---|---|---|
| 1–3 | Coverage features | Deferred (requires `compute_coverage` tool) |
| 4–5 | Paper/fact count | **Used** in heuristic |
| 6–7 | Conflict features | Deferred (requires `detect_conflicts` tool) |
| 8–11 | LLM signals (confidence, consistency, entropy, gaps) | Deferred (requires `extract_signals`) |
| 12–13 | SUPPORT/REFUTE fractions | **Used** in heuristic |
| 14 | Source diversity | Deferred |
| 15–16 | Embedding similarity | Deferred |

### Acceptance Criteria

- [ ] `classifier(state)` returns dict with `label`, `confidence`, `gaps` keys
- [ ] Returns `SUFFICIENT_*` when facts are plentiful and consistent
- [ ] Returns `INSUFFICIENT` when evidence is sparse or conflicting
- [ ] Interface is identical to what the trained MLP will expose

---

## 4. Metacognitive Control Loop — Algorithm 1

**Target file:** `src/pkevolve/verification/controller.py`
**Depends on:** EvidenceState, VerificationTools, Classifier

### Minimal Implementation — Fixed-Iteration Loop

The backbone implements Algorithm 1's structure but uses the heuristic classifier and skips gap-directed retrieval (re-queries with the original claim instead). Compression uses L1 deduplication only.

```python
class MetacognitiveController:
    def __init__(self, llm_client: OpenAI, model: str,
                 classifier: SufficiencyClassifier,
                 compressor: SufficiencyPreservingCompressor,
                 pubmed_searcher: RelevancePubMedSearcher,
                 threshold: float = 0.7, max_iterations: int = 3,
                 context_budget: int = 50000):
        self.llm_client = llm_client
        self.model = model
        self.classifier = classifier
        self.compressor = compressor
        self.searcher = pubmed_searcher
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.context_budget = context_budget

    def verify(self, claim: str) -> Tuple[str, float, str]:
        """
        Backbone entry point. Synchronous — no async needed yet.
        Returns (verdict, confidence, report_text).
        """
        # Phase 1: Init (no subclaim decomposition — use claim directly)
        state = EvidenceState(claim, subclaims=[claim])
        tools = VerificationTools(state, self.llm_client, self.model, self.searcher)
        tools.retrieve_papers(claim, top_k=5)
        self._process(state, tools)

        # Phase 2: Loop
        for t in range(1, self.max_iterations + 1):
            result = self.classifier(state)
            confidence = result['confidence']

            if confidence >= self.threshold:
                return result['label'], confidence, self._report(state)

            # Re-query with original claim (gap-directed retrieval deferred)
            tools.retrieve_papers(claim, top_k=3)
            self._process(state, tools)

            # Compression
            if state.token_count() > self.context_budget:
                state = self.compressor.compress(state, claim)

        # Phase 3: Timeout
        result = self.classifier(state)
        return result['label'], result['confidence'], self._report(state)

    def _process(self, state: EvidenceState, tools: VerificationTools):
        """Sequential processing — summarize and extract facts for new papers."""
        for pmid, paper in state.papers.items():
            if paper.summary is None:
                tools.summarize_paper(pmid)
                tools.extract_facts(pmid)

    def _report(self, state: EvidenceState) -> str:
        """Minimal report — claim, paper count, fact count, verdict."""
        lines = [f"Claim: {state.claim}",
                 f"Papers: {len(state.papers)}",
                 f"Facts: {len(state.facts)}"]
        for f in state.facts:
            lines.append(f"  [{f.stance}] {f.text} (PMID:{f.source_pmid})")
        return "\n".join(lines)
```

### What This Backbone Tests

Even with a heuristic classifier and no gap-directed retrieval, this loop validates:
- The data flow: claim → retrieval → LLM processing → state → classifier → decision
- The interface contracts between all 5 core components
- That the loop terminates (not infinite, not always-1-iteration)
- That compression triggers correctly

### Deferred to Extensions

- Subclaim decomposition (`_decompose()` via LLM)
- Async execution (`async def verify`)
- Parallel processing (`asyncio.gather` in `_process`)
- Gap-directed retrieval (LLM query formulation from gaps)
- `INSUFFICIENT` early-exit when no gaps and past halfway
- Full report generation (subclaim analysis, conflicts, evidence summary)
- Hyperparameter tuning (τ, T, B ranges from implementation_plan.md)

### Acceptance Criteria

- [ ] `controller.verify("Does p53 activate BAX?")` returns a verdict, confidence, and report
- [ ] The loop runs 1–3 iterations (not degenerate)
- [ ] Each iteration adds new evidence to the state
- [ ] Report is a readable text string summarizing the evidence

---

## 5. Sufficiency-Preserving Compression — Primary Technical Novelty

**Target file:** `src/pkevolve/verification/compressor.py`
**Depends on:** EvidenceState (for `clone()`, `token_count()`)

### Minimal Implementation — L1 Deduplication Only

The backbone implements only the lossless compression level. The interface matches the full compressor so L2/L3 can be added without changing the controller.

```python
class SufficiencyPreservingCompressor:
    """
    Backbone: L1 deduplication only.
    Same interface as the full compressor — controller doesn't know the difference.
    """
    def __init__(self, llm_client=None, classifier=None, epsilon: float = 0.05):
        # llm_client and classifier unused in backbone — stored for interface compatibility
        self.llm_client = llm_client
        self.classifier = classifier
        self.epsilon = epsilon

    def compress(self, state: EvidenceState, claim: str) -> EvidenceState:
        """L1: Remove duplicate facts."""
        compressed = state.clone()
        seen = set()
        unique_facts = []
        for fact in compressed.facts:
            key = (fact.text.strip().lower(), fact.stance)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        compressed.facts = unique_facts
        return compressed
```

### Deferred to Extensions

The full compressor from implementation_plan.md Section 3:
- L2: Synthesis refresh (LLM-based, lossy, coverage-gated)
- L3: Aggressive synthesis (LLM-based, lossy, guarded)
- `_check_invariant()`: classifier comparison `|φ(S') - φ(S)| ≤ ε`
- `async compress()` (async for L2/L3 LLM calls)
- Invariant violation rate tracking
- ε sensitivity sweep

### Acceptance Criteria

- [ ] `compressor.compress(state, claim)` returns a new state with fewer facts when duplicates exist
- [ ] Original state is not mutated
- [ ] No-op when there are no duplicates

---

## Timeline

The backbone is a single sprint. No parallelism needed — each step takes about a day:

```
Day 1:  1. EvidenceState + data_models.py
Day 2:  2. VerificationTools (3 essential tools + stubs)
Day 3:  3. Heuristic classifier
Day 3:  5. L1-only compressor
Day 4:  4. Control loop (wires everything together)
Day 5:  End-to-end test on 5–10 SIGNOR edges or SciFact claims
```

After the backbone works end-to-end, extensions can proceed independently:
- Tools (add coverage, conflicts, synthesis, gaps) — no loop changes
- Classifier (train MLP, add signal extraction) — swap in, same interface
- Compression (add L2, L3, invariant checking) — swap in, same interface
- Loop (add subclaim decomposition, async, gap-directed retrieval) — incremental

---

## File Structure

Backbone files — small and focused:

```
src/pkevolve/verification/
├── __init__.py
├── data_models.py           # PaperRecord, Fact dataclasses
├── evidence_state.py        # EvidenceState (no locking)
├── tools.py                 # VerificationTools (3 real + 5 stubs)
├── classifier.py            # Heuristic classifier (same interface as future MLP)
├── compressor.py            # L1 deduplication only
└── controller.py            # MetacognitiveController (sync, fixed-iteration)
```

CLI entry point:

```
scripts/verification/
└── run_verification.py      # CLI: claim → verdict + confidence + report
```

Files added during extension (not created in backbone):

```
src/pkevolve/verification/
├── signals.py               # extract_signals() — added with trained classifier
└── report.py                # Full report generation — added with subclaim decomposition

scripts/verification/
├── generate_training_data.py   # Added in classifier extension
└── train_classifier.py         # Added in classifier extension
```

This follows the project convention: reusable logic in `src/pkevolve/`, CLI scripts in `scripts/`.
