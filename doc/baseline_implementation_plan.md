# Baseline Implementation Plan: Evidence Programming Experiments

## Shared Infrastructure (Build First)

Before implementing any baseline, build these shared components that every system needs:

### Retrieval Backend
All systems except "LLM-only" and "Random" need access to the same corpus. Build a unified retrieval module:

```python
# src/baselines/shared/retrieval.py
class RetrievalBackend:
    """Shared retrieval for all baselines. Ensures fair comparison."""
    
    def bm25_search(self, query: str, corpus: str, k: int) -> list[dict]:
        """BM25 over pre-indexed corpus. Used by Fixed-k RAG, ReAct, FIRE, etc."""
        # corpus = "scifact_open" | "civic_fact" | "signor"
        # Returns: [{pmid, title, abstract, score}, ...]
    
    def semantic_search(self, query: str, corpus: str, k: int) -> list[dict]:
        """Dense retrieval (Contriever/BioLinkBERT) for comparison."""
    
    def pubmed_api_search(self, query: str, k: int) -> list[dict]:
        """Live PubMed API. Used by agentic systems (ReAct, SAFE, FIRE, OpenScholar)."""
    
    def semantic_scholar_search(self, query: str, k: int) -> list[dict]:
        """Semantic Scholar API. Used by OpenScholar."""
```

### Evaluation Harness
```python
# src/baselines/shared/evaluate.py
class EvaluationHarness:
    """Runs any baseline on any dataset, collects metrics."""
    
    def run(self, baseline, dataset, n_runs=1) -> Results:
        # For each claim: run baseline, collect verdict + cost metrics
        # Returns: all metrics below
        # For temperature=0 (deterministic): single run suffices
        # For temperature>0 or agentic systems: 3 runs with varied claim order
    
    def compute_f1(self, predictions, gold_labels) -> dict:
        # Macro F1 over {SUPPORT, REFUTE, NEI}
    
    def compute_per_class(self, predictions, gold_labels) -> dict:
        # Per-class precision, recall, F1 for each label
    
    def compute_calibration(self, predictions, gold_labels) -> dict:
        # Expected Calibration Error (ECE) on confidence scores
    
    def compute_evidence_recall(self, predictions, gold_evidence) -> dict:
        # For claims with gold PMIDs: did the system retrieve them?
        # Evidence recall@k, evidence precision
    
    def compute_cost(self, trace) -> dict:
        # Total tokens (input+output), LLM calls, wall-clock time
    
    def efficiency_frontier(self, all_results) -> dict:
        # F1 vs. cost (tokens/claim) across all systems
        # Key selling point: better F1 at lower cost
```

### Verdict Schema
All baselines must output the same structured format:
```python
class Verdict(BaseModel):
    label: str          # SUPPORT | REFUTE | NEI
    confidence: float   # 0-1
    evidence: list[str] # PMIDs or text snippets used
    reasoning: str      # Free-text explanation
```

### Label Taxonomy & Normalization

**Critical issue:** The four evaluation datasets use inconsistent label vocabularies. Every dataset adapter and every baseline must normalize labels to the canonical `{SUPPORT, REFUTE, NEI}` taxonomy before evaluation.

| Dataset | Raw Labels | Canonical Mapping |
|---------|-----------|-------------------|
| **SciFact-Open** | `SUPPORT`, `CONTRADICT`, ∅ (no evidence) | `SUPPORT` → `SUPPORT`, `CONTRADICT` → `REFUTE`, no evidence → `NEI` |
| **SIGNOR\*** | `SUPPORTED`, `WRONG`, `UNCERTAIN` | `SUPPORTED` → `SUPPORT`, `WRONG` → `REFUTE`, `UNCERTAIN` → `NEI` |
| **CIViC-Fact** | `SUPPORTS`, `REFUTES`, `NEI` | `SUPPORTS` → `SUPPORT`, `REFUTES` → `REFUTE`, `NEI` → `NEI` |
| **ConnectomeDB** | `SUPPORTED`, `REFUTED`, `NEI` | `SUPPORTED` → `SUPPORT`, `REFUTED` → `REFUTE`, `NEI` → `NEI` |
| **Evidence Programming (our system)** | `SUPPORT`, `REFUTE`, `UNCERTAIN` | `UNCERTAIN` → `NEI` |

Implement this in the evaluation harness as a `normalize_label()` function applied to both predictions and gold labels before any metric computation. For CIViC-Fact (3-class), report macro F1; binary F1 (SUPPORTS vs. REFUTES) as a secondary metric.

```python
# src/baselines/shared/label_utils.py
LABEL_MAP = {
    # SciFact-Open (evidence dict labels)
    "CONTRADICT": "REFUTE",
    # SIGNOR* (ground_truth.csv Label column)
    "SUPPORTED": "SUPPORT", "WRONG": "REFUTE", "UNCERTAIN": "NEI",
    # CIViC-Fact (gold_label_name field: SUPPORTS / REFUTES / NEI)
    "SUPPORTS": "SUPPORT", "REFUTES": "REFUTE",
    # ConnectomeDB (eval files: SUPPORTED from positives, REFUTED/NEI from negatives)
    "REFUTED": "REFUTE",
    # Identity (already canonical)
    "SUPPORT": "SUPPORT", "REFUTE": "REFUTE", "NEI": "NEI",
}

def normalize_label(label: str) -> str:
    return LABEL_MAP.get(label.upper().strip(), "NEI")
```

### LLM Backend
Use the same backbone LLM across all baselines. This isolates the architectural comparison.

```python
# src/baselines/shared/llm.py
class LLMBackend:
    def __init__(self, model: str = "claude-sonnet-4-5-20250929",
                 tracker: CostTracker | None = None):
        self.client = Anthropic()
        self.model = model
        self.tracker = tracker or CostTracker()
    
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Single-turn completion. Tracks cost automatically."""
        response = self.client.messages.create(
            model=self.model, max_tokens=2048, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}]
        )
        self.tracker.record_llm_call(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.content[0].text
    
    def complete_with_tools(self, system: str, messages: list,
                            tools: list, temperature: float = 0.0):
        """Multi-turn completion with tool use. For ReAct, FIRE, SAFE."""
        response = self.client.messages.create(
            model=self.model, max_tokens=2048, temperature=temperature,
            system=system, messages=messages, tools=tools
        )
        self.tracker.record_llm_call(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response
```

### Cost Tracker

Wraps every LLM call and search call to automatically track cost. Attach one `CostTracker` per baseline run; reset between claims.

```python
# src/baselines/shared/cost_tracker.py
import time
from dataclasses import dataclass, field

@dataclass
class TraceEntry:
    step: int
    action: str          # "llm_call" | "bm25_search" | "pubmed_search" | "read_abstract"
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_seconds: float = 0.0
    details: dict = field(default_factory=dict)

class CostTracker:
    def __init__(self):
        self.trace: list[TraceEntry] = []
        self._step = 0
        self._t0 = time.time()
    
    def record_llm_call(self, input_tokens: int, output_tokens: int, details: dict = {}) -> None:
        self.trace.append(TraceEntry(
            step=self._step, action="llm_call",
            input_tokens=input_tokens, output_tokens=output_tokens,
            wall_clock_seconds=time.time() - self._t0, details=details
        ))
        self._step += 1
    
    def record_search_call(self, query: str, n_results: int, source: str = "pubmed") -> None:
        self.trace.append(TraceEntry(
            step=self._step, action=f"{source}_search",
            wall_clock_seconds=time.time() - self._t0,
            details={"query": query, "n_results": n_results}
        ))
        self._step += 1
    
    @property
    def total_llm_calls(self) -> int:
        return sum(1 for e in self.trace if e.action == "llm_call")
    
    @property
    def total_tokens(self) -> int:
        return sum(e.input_tokens + e.output_tokens for e in self.trace)
    
    @property
    def total_search_calls(self) -> int:
        return sum(1 for e in self.trace if e.action != "llm_call")
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._t0
```

### Baseline Result Schema

Every baseline returns a `BaselineResult` for each claim; the eval harness aggregates these.

```python
# src/baselines/shared/verdict.py
from pydantic import BaseModel

class Verdict(BaseModel):
    label: str          # SUPPORT | REFUTE | NEI
    confidence: float   # 0-1
    evidence: list[str] # PMIDs or text snippets used
    reasoning: str      # Free-text explanation
    
    @classmethod
    def from_llm_response(cls, text: str) -> "Verdict":
        """Parse LLM JSON response. Falls back to regex; defaults to NEI."""
        ...

class BaselineResult(BaseModel):
    claim_id: str
    claim_text: str
    gold_label: str
    verdict: Verdict
    trace: list = []           # TraceEntry list from CostTracker
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_search_calls: int = 0
    latency_seconds: float = 0
```

### Shared Prompts

All baselines that make a final verdict call **must** use the same verification prompt. This is critical for fairness — the only variable across baselines is what evidence is provided.

```python
# src/baselines/shared/prompts.py

VERIFICATION_SYSTEM_PROMPT = """You are a scientific claim verification expert.

Given a claim and retrieved evidence, determine whether the claim is:
- SUPPORT: The evidence supports the claim
- REFUTE: The evidence contradicts the claim
- NEI: There is not enough information to determine

Respond in JSON format:
{
    "label": "SUPPORT" | "REFUTE" | "NEI",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation citing specific evidence",
    "evidence": ["PMID1", "PMID2", ...]
}"""

VERIFICATION_USER_TEMPLATE = """Claim: {claim}

Retrieved Evidence:
{evidence}

Based on the above evidence, classify the claim."""

DECOMPOSITION_PROMPT = """Decompose the following scientific claim into
independently verifiable atomic facts. Each fact should be a single
statement that can be checked against scientific literature.

Output as a JSON list of strings.

Claim: {claim}"""

QUERY_GENERATION_PROMPT = """Generate a PubMed search query to find
evidence about the following. Output only the query string.

Topic: {topic}"""
```

### Async & Testing

The `EvaluationHarness` uses `asyncio` with a semaphore (`max_concurrent=5`) to run claims concurrently without overwhelming APIs. Phase completion is verified with `pytest`: `test_shared.py` (infrastructure), `test_datasets.py` (loaders + BM25 index round-trip), `test_baselines.py` (5 claims end-to-end per baseline).

---

## Datasets & Corpus Specification

All experiments run on shared datasets located at `/hps/nobackup/saezrodriguez/shared_datasets/`.

### Dataset Overview

| Dataset | Eval Subset | Classes | Corpus | Source |
|---------|-------------|---------|--------|--------|
| **SciFact-Open** | ~41 (20% test split of 206 annotated) | SUPPORT, CONTRADICT | 500K S2ORC abstracts (`data/corpus.jsonl`) | Wadden et al. 2022 |
| **SIGNOR\*** | 66 edges / **110 variants** (incl. flipped) | SUPPORTED (34), WRONG (28), UNCERTAIN (4) | *No pre-built corpus — see below* | Custom annotation |
| **CIViC-Fact** | 2,055 (test partition, `flagged != True`) | SUPPORTS, REFUTES, NEI | *No pre-built corpus — see below* | CIViC database |
| **ConnectomeDB** | 184 positive + 363 negative/NEI | SUPPORTED, REFUTED, NEI | *No pre-built corpus — see below* | Liu et al. 2025, [doi:10.1093/nar/gkaf1108](https://doi.org/10.1093/nar/gkaf1108) |

### Corpus Strategy per Dataset

**SciFact-Open:** Corpus is provided (500K abstracts from S2ORC). Use directly for BM25 indexing. Agentic baselines use PubMed API but the 500K corpus serves as the controlled retrieval pool for Fixed-k RAG.

**Evaluation subset:** Use the **20% held-out test split** of the 206 annotated claims (those with `evidence != {}`), stratified by consensus label with `random_state=42`. This produces ~41 claims not used in sufficiency classifier training. Claim string: `claim["claim"]` directly; evidence text from `corpus.jsonl` via `doc_id`.

```python
from sklearn.model_selection import train_test_split
annotated = [c for c in all_claims if c.get("evidence")]
labels = [consensus_label(c) for c in annotated]  # SUPPORT or CONTRADICT
_, test_claims = train_test_split(annotated, test_size=0.2, stratify=labels, random_state=42)
# ~41 claims: ~20 SUPPORT, ~21 CONTRADICT
```

**SIGNOR\*:** No pre-built corpus. Strategy:
- Path: `/hps/nobackup/saezrodriguez/shared_datasets/signor*/ground_truth.csv`
- For BM25 baselines: build a corpus from (a) PubMed abstracts of the PMIDs in the `PMID` column of ground_truth.csv, plus (b) 1-hop citation neighbors of those PMIDs. This gives a ~5K–10K abstract pool.
- For agentic baselines: use live PubMed API search (same as our evidence programming system).
- **Corrected class distribution:** 66 edges — SUPPORTED (34), WRONG (28), UNCERTAIN (4). Only 4 UNCERTAIN edges make the 3-class macro F1 unreliable for that class. Recommended: report 2-class macro F1 (SUPPORTED vs. WRONG) as the primary metric; treat UNCERTAIN edges as a secondary analysis.
- **Flip logic — 110 variants:** Evaluate all 66 forward claims **plus** negated variants for the 44 `up-regulates*` edges. Only `EFFECT ∈ {up-regulates, up-regulates activity, up-regulates quantity, up-regulates quantity by expression}` is flipped (activation → inhibition direction). Down-regulates and non-directional effects are left as-is. Label inversion on flip: SUPPORTED → WRONG, WRONG → SUPPORTED, UNCERTAIN → UNCERTAIN. **Always use `construct_signor_claim()` from `experiments/run_signor_eval.py`** — do not construct claim strings manually.

```python
from experiments.run_signor_eval import construct_signor_claim, get_flipped_label
claims = []
for _, row in df.iterrows():
    for flip in [False, True]:
        claim_str = construct_signor_claim(row["ENTITYA"], row["ENTITYB"], row["EFFECT"], flip=flip)
        if claim_str is None:  # non-flippable effect — skip duplicate
            continue
        label = get_flipped_label(row["Label"], flip=flip)
        claims.append({"id": row["SIGNOR_ID"], "flip": flip, "claim": claim_str, "label": label})
# 110 variants total: 66 forward + 44 flipped
```

**CIViC-Fact:** No pre-built abstract corpus. Strategy:
- Path: `/hps/nobackup/saezrodriguez/shared_datasets/civicfact/data_builder/builds/civicfact-2025.03.25/data.jsonl.gz`
- **Evaluation subset:** `partition == "test"` and `flagged != True` → **2,055 rows** (SUPPORTS: 689, REFUTES: 666, NEI: 700). `train` + `dev` partitions reserved for fine-tuning or few-shot sampling.
- **3-class labels:** `gold_label_name` field contains `SUPPORTS` / `REFUTES` / `NEI`. NEI has ~34% prevalence — it is a real class, not an artifact. Report 3-class macro F1; binary F1 (SUPPORTS vs. REFUTES, excluding NEI) as secondary metric.
- **Claim string:** `row["claim.flat"]` directly — already a natural-language claim; no verbalization needed. Evidence text: `row["evidence.flat"]`. Source PMID: `row["document.pmid"]`.
- For BM25 baselines: fetch abstracts of the referenced PMIDs via PubMed E-utilities to build the corpus.
- For agentic baselines: use live PubMed API.
- **Subsampling:** ~2K test claims is manageable for static baselines; subsample to ~500 stratified claims for expensive agentic systems.

**ConnectomeDB:** Curated ligand–receptor interaction database (Liu et al. 2025, [doi:10.1093/nar/gkaf1108](https://doi.org/10.1093/nar/gkaf1108)). Data already available at `/hps/nobackup/saezrodriguez/shared_datasets/connectomedb/` — **no download or synthetic negative generation needed**.

- **Evaluation files (use directly):**
  - `cdb25_direct_multipub_unique.csv` (184 rows) — positives: CDB25 Direct pairs with ≥2 publications, deduplicated. Key columns: `LR Pair`, `Ligand Symbols`, `Receptor Symbols`, `AI summary` (Perplexity URL embedding PMIDs), `Species`.
  - `ConnectomeDB2020_rejected_labeled.csv` (363 rows) — negatives/NEI: CDB2020 pairs rejected by CDB25 curators. Key columns: `LR_pair`, `Ligand`, `Receptor`, `Label` (`REFUTED` / `NEI`), `Rejection_reason`, `Curator comments`.
- **Claim string:** form from LR pair columns, e.g. `"{Ligand} directly binds to and activates {Receptor} as a ligand–receptor pair."` for positives; equivalently negated for REFUTED entries.
- **Gold evidence:** the `AI summary` field in the positive file embeds supporting PMIDs (Perplexity URL format). Extract PMIDs for evidence recall evaluation.
- **Corpus for BM25:** fetch abstracts of the PMIDs extracted from `AI summary` via PubMed E-utilities.
- **No subsampling needed:** total 547 claims (184 + 363) is manageable for all baseline types.
- **Key value:** complements SIGNOR\* — larger sample, gold PMIDs, pre-curated negatives from a real rejection process (not synthetic corruption).

### Primary vs. Secondary Datasets

| Role | Datasets | Rationale |
|------|----------|-----------|
| **Primary** | SciFact-Open, CIViC-Fact | Established benchmarks, sufficient size, 3-class |
| **Primary** | ConnectomeDB | Rigorously curated molecular interactions, gold PMIDs, pre-curated negatives |
| **Secondary** | SIGNOR\* | Domain-specific (GRN edges), small sample (66 edges / 110 variants), qualitative |

---

## 0. Evidence Programming (Our System)

**What it tests:** The full evidence programming pipeline — gap-directed retrieval, MLP sufficiency classifier, shared evidence state across subclaims, structured fact extraction.

**Canonical mode:** Mode B (REPL orchestrator) with the same backbone LLM as all other baselines.

**Integration with eval harness:** The evidence programming system emits `VerificationVerdict` (from `data_models.py`). Adapt to the shared `Verdict` schema:
```python
# src/baselines/shared/adapt_evidence_programming.py
def verdict_from_verification(vv: VerificationVerdict) -> Verdict:
    return Verdict(
        label=normalize_label(vv.verdict),  # UNCERTAIN → NEI
        confidence=vv.confidence,
        evidence=vv.key_evidence,
        reasoning=vv.reasoning,
    )
```

**Cost tracking:** The REPL orchestrator already tracks LLM calls and token usage. Expose these via the same cost interface used by baselines.

---

## 1. Random Baseline

**What it tests:** Chance-level performance. Establishes the floor.

**Implementation:**
```python
# src/baselines/random_baseline.py
import random

class RandomBaseline:
    def verify(self, claim: str) -> Verdict:
        label = random.choice(["SUPPORT", "REFUTE", "NEI"])
        return Verdict(
            label=label,
            confidence=random.uniform(0, 1),
            evidence=[],
            reasoning="Random assignment"
        )
```

**Cost:** 0 tokens, 0 LLM calls, ~0 latency.

**Expected result:** ~33% F1 for balanced 3-class; lower for imbalanced datasets.

**Implementation time:** 30 minutes.

---

## 2. LLM-only (No Retrieval)

**What it tests:** How much the LLM knows from pretraining alone. Measures parametric knowledge ceiling and prior-dominated failure rate.

**Implementation:**
```python
# src/baselines/llm_only.py

class LLMOnly:
    def __init__(self, llm: LLMBackend):
        self.llm = llm
    
    def verify(self, claim: str) -> Verdict:
        response = self.llm.complete(
            system="""You are a scientific claim verification expert.
Given a claim, determine whether it is SUPPORTED, REFUTED, or 
there is NOT ENOUGH INFORMATION (NEI) based on your knowledge.

Respond in JSON: {"label": "...", "confidence": 0.0-1.0, "reasoning": "..."}""",
            user=f"Claim: {claim}"
        )
        return parse_verdict(response)
```

**Prompt design notes:**
- Do NOT mention retrieval or evidence — the LLM should answer purely from parametric knowledge.
- Use temperature=0 for deterministic output (variance comes from the 3 runs with different random seeds if using sampling).
- For biomedical claims (SciFact-Open, CIViC-Fact), this tests whether the LLM has seen the relevant papers during pretraining.

**Cost:** 1 LLM call per claim, ~500-1000 tokens per claim.

**Implementation time:** 1 hour.

---

## 3. LLM + 5 Search Results

**What it tests:** Minimal augmentation — giving the LLM raw search snippets without any retrieval pipeline engineering. This is what a researcher would get by pasting claim + Google Scholar results into ChatGPT.

**Implementation:**
```python
# src/baselines/llm_plus_search.py

class LLMPlusSearch:
    """Simulate 'paste search results into LLM' workflow."""
    
    def __init__(self, llm: LLMBackend, retrieval: RetrievalBackend):
        self.llm = llm
        self.retrieval = retrieval
    
    def verify(self, claim: str, corpus: str) -> Verdict:
        # Step 1: Use the claim directly as the search query
        results = self.retrieval.pubmed_api_search(query=claim, k=5)
        
        # Step 2: Format results as raw search snippets (title + snippet)
        # NOT full abstracts — simulates search engine snippets
        context = "\n\n".join([
            f"Result {i+1}: {r['title']}\n{r['snippet']}"
            for i, r in enumerate(results)
        ])
        
        # Step 3: Single LLM call with raw context
        response = self.llm.complete(
            system="""You are a scientific claim verification expert.
Given a claim and search results, determine if the claim is 
SUPPORTED, REFUTED, or NEI (Not Enough Information).
Respond in JSON: {"label": "...", "confidence": ..., "reasoning": "..."}""",
            user=f"Claim: {claim}\n\nSearch Results:\n{context}"
        )
        return parse_verdict(response)
```

**Key distinction from Fixed-k RAG:** This uses the raw claim as query (no BM25 over a corpus) and uses search snippets (not full abstracts). It represents the zero-engineering baseline.

**Cost:** 1 API search call + 1 LLM call per claim.

**Implementation time:** 1-2 hours.

---

## 4. Fixed-5 RAG (BM25) and Fixed-10 RAG (BM25)

**What it tests:** Standard retrieve-then-verify pipeline with a fixed budget. The primary lower bound for adaptive approaches. Running at k=5 and k=10 shows diminishing returns of blindly adding more documents.

**Implementation:**
```python
# src/baselines/fixed_k_rag.py

class FixedKRAG:
    """Standard BM25 retrieve-then-verify pipeline."""
    
    def __init__(self, llm: LLMBackend, retrieval: RetrievalBackend, k: int):
        self.llm = llm
        self.retrieval = retrieval
        self.k = k
    
    def verify(self, claim: str, corpus: str) -> Verdict:
        # Step 1: BM25 retrieval over the dataset corpus
        papers = self.retrieval.bm25_search(query=claim, corpus=corpus, k=self.k)
        
        # Step 2: Concatenate full abstracts into context
        context = "\n\n".join([
            f"[Paper {i+1}] PMID: {p['pmid']}\n"
            f"Title: {p['title']}\n"
            f"Abstract: {p['abstract']}"
            for i, p in enumerate(papers)
        ])
        
        # Step 3: Single LLM call for verdict
        response = self.llm.complete(
            system=VERIFICATION_SYSTEM_PROMPT,  # shared prompt
            user=f"Claim: {claim}\n\nRetrieved Evidence:\n{context}\n\n"
                 f"Based on the above evidence, classify the claim."
        )
        return parse_verdict(response)
```

**Corpus indexing (one-time setup):**
```python
# scripts/index_corpus.py
import rank_bm25

def build_bm25_index(corpus_path: str) -> BM25Okapi:
    """Pre-index each dataset's corpus for BM25 retrieval."""
    # SciFact-Open: 500K abstracts from S2ORC
    # CIViC-Fact: 554 full-text publications
    # SIGNOR*: PubMed abstracts linked to SIGNOR interactions
    papers = load_corpus(corpus_path)
    tokenized = [tokenize(p['abstract']) for p in papers]
    return BM25Okapi(tokenized)
```

**Important details:**
- Use the SAME verification prompt across Fixed-5, Fixed-10, LLM+Search, and the verification step of agentic baselines. The only variable should be what evidence is provided.
- For SciFact-Open: index the 500K S2ORC abstracts.
- For CIViC-Fact: index the 554 publications (abstracts or full text depending on availability).
- For SIGNOR*: index relevant PubMed abstracts.

**Cost:** 1 LLM call per claim. BM25 is local (~0 cost). Token cost scales with k.

**Implementation time:** 2-3 hours (including corpus indexing).

---

## 5. ReAct (Same Tools as Evidence Programming)

**What it tests:** Whether structured evidence programming adds value over letting an agent freely reason-and-act with the same tools. This is your most fundamental architectural comparison.

**Implementation:**

ReAct interleaves Thought → Action → Observation steps. The agent has access to the same retrieval tools as your system but NO structured evidence state, NO sufficiency classifier, and NO gap-directed retrieval.

```python
# src/baselines/react_baseline.py

REACT_TOOLS = [
    {
        "name": "search_pubmed",
        "description": "Search PubMed for papers. Input: query string. Output: list of papers with titles and abstracts.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5}}}
    },
    {
        "name": "read_abstract",
        "description": "Read the full abstract of a paper by PMID.",
        "input_schema": {"type": "object", "properties": {"pmid": {"type": "string"}}}
    },
]

REACT_SYSTEM_PROMPT = """You are a scientific claim verification agent using the ReAct framework.

For each claim, you should:
1. Think about what evidence you need
2. Use tools to search for and read relevant papers
3. Think about whether you have enough evidence
4. Repeat if needed
5. Provide a final verdict: SUPPORT, REFUTE, or NEI

You have a budget of {max_steps} reasoning steps. Use them wisely.

After each tool call, you will see the observation. Reason about it before 
deciding your next action.

When ready, output your final verdict in JSON:
{{"label": "...", "confidence": ..., "evidence": [...], "reasoning": "..."}}"""


class ReActBaseline:
    def __init__(self, llm: LLMBackend, retrieval: RetrievalBackend, 
                 max_steps: int = 10):
        self.llm = llm
        self.retrieval = retrieval
        self.max_steps = max_steps
    
    def verify(self, claim: str, corpus: str) -> Verdict:
        messages = [{"role": "user", "content": f"Verify this claim: {claim}"}]
        
        for step in range(self.max_steps):
            # LLM generates thought + action (or final answer)
            response = self.llm.complete_with_tools(
                system=REACT_SYSTEM_PROMPT.format(max_steps=self.max_steps),
                messages=messages,
                tools=REACT_TOOLS
            )
            
            # Check if the agent produced a final verdict
            if response.stop_reason == "end_turn":
                return parse_verdict(response.content)
            
            # Execute tool call
            tool_name = response.tool_use.name
            tool_input = response.tool_use.input
            
            if tool_name == "search_pubmed":
                observation = self.retrieval.pubmed_api_search(
                    query=tool_input["query"], k=tool_input.get("k", 5)
                )
            elif tool_name == "read_abstract":
                observation = self.retrieval.get_abstract(tool_input["pmid"])
            
            # Append tool result to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "content": format_observation(observation)}
            ]})
        
        # Max steps reached — force verdict
        return self._force_verdict(messages)
```

**Critical implementation details:**
- `max_steps=10` matches a reasonable agent budget. Tune this so ReAct's median token usage is comparable to your system's — otherwise you're comparing different compute budgets.
- The agent has NO memory between claims (fresh conversation each time).
- The agent must decide on its own when to stop — there is no sufficiency signal. This is the key difference from your system.
- Use the same LLM backbone as all other baselines.

**What to track:** How many search queries does ReAct issue? How many papers does it read? Does it stop too early or too late compared to your system?

**Cost:** Variable (5-15 LLM calls per claim typical for ReAct).

**Implementation time:** 3-4 hours.

---

## 6. SAFE (Search Augmented Factuality Evaluator)

**What it tests:** Per-fact independent verification vs. your shared evidence state across subclaims. SAFE decomposes claims into atomic facts and verifies each independently — no cross-fact evidence aggregation.

**Public framework:** Use the official SAFE codebase from Google DeepMind.
- Repository: https://github.com/google-deepmind/long-form-factuality
- The repo contains the full SAFE pipeline: decomposition → per-fact search → rating
- Search backend uses Serper API (Google Search) by default

**Implementation strategy:** Fork their pipeline, replacing only the search backend (Serper → PubMed API) and the final aggregation (per-fact supported/not-supported → our 3-class Verdict). Keep their prompt templates and decomposition logic intact.

```python
# src/baselines/safe_baseline.py
# Wraps the official SAFE pipeline from:
#   https://github.com/google-deepmind/long-form-factuality

class SAFEBaseline:
    """Uses the official SAFE pipeline with PubMed search backend.
    
    Adaptation from original:
    - Search: Serper (Google) → PubMed API
    - Task: long-form factuality → scientific claim verification
    - Output: per-fact supported/not_supported → aggregated 3-class Verdict
    """
    
    def __init__(self, llm: LLMBackend, retrieval: RetrievalBackend,
                 max_search_rounds: int = 3):
        self.llm = llm
        self.retrieval = retrieval
        self.max_search_rounds = max_search_rounds
    
    def verify(self, claim: str) -> Verdict:
        # Stage 1: Decompose claim into atomic facts (SAFE's prompt)
        atomic_facts = self._decompose(claim)
        
        # Stage 2: Verify each fact independently (SAFE's core loop)
        fact_verdicts = []
        for fact in atomic_facts:
            verdict = self._verify_single_fact(fact)
            fact_verdicts.append(verdict)
        
        # Stage 3: Aggregate into claim-level verdict
        return self._aggregate(fact_verdicts)
    
    def _verify_single_fact(self, fact: str) -> dict:
        """SAFE's per-fact iterative search loop."""
        search_history = []
        for round in range(self.max_search_rounds):
            query = self._generate_query(fact, search_history)
            search_history.append(query)
            # PubMed instead of Serper
            results = self.retrieval.pubmed_api_search(query=query, k=5)
            assessment = self._rate_fact(fact, results)
            if assessment["verdict"] != "NEED_MORE_SEARCH":
                return assessment
        return {"verdict": "NOT_SUPPORTED", "confidence": 0.3}
    
    def _aggregate(self, fact_verdicts: list[dict]) -> Verdict:
        """Aggregate per-fact verdicts into claim-level verdict."""
        support_count = sum(1 for v in fact_verdicts if v["verdict"] == "SUPPORTED")
        total = len(fact_verdicts)
        if support_count / total >= 0.8:
            label = "SUPPORT"
        elif support_count / total <= 0.2:
            label = "REFUTE"
        else:
            label = "NEI"
        avg_conf = sum(v["confidence"] for v in fact_verdicts) / total
        return Verdict(label=label, confidence=avg_conf, ...)
```

**Setup:**
1. Clone https://github.com/google-deepmind/long-form-factuality into `external/safe/`
2. Replace `search_utils.py` Serper calls with our `RetrievalBackend.pubmed_api_search`
3. Keep their prompt templates for decomposition and per-fact rating
4. Add a 3-class aggregation layer on top (their binary supported/not_supported → our SUPPORT/REFUTE/NEI)

**Critical comparison point:** SAFE has no cross-fact evidence sharing. If paper X is relevant to both atomic fact 1 and fact 3, SAFE might retrieve it twice (or miss it for fact 3). Our system's shared evidence state avoids this.

**Cost:** (N_atomic_facts × max_search_rounds × ~3) LLM calls per claim. For a claim with 3 subclaims and 2 search rounds each: ~18 LLM calls.

**Implementation time:** 3-4 hours (mostly search backend swap).

---

## 7. FIRE (Fact-checking with Iterative Retrieval and Verification)

**What it tests:** Our learned MLP sufficiency classifier vs. FIRE's LLM-internal confidence gating. FIRE uses the LLM's own confidence to decide whether to search or stop.

**Public framework:** Check for an official FIRE implementation. As of writing, the FIRE paper (Findings of NAACL 2025) may not have an open-source release. If no public code is available, implement from the paper description — the algorithm is straightforward (confidence-gated retrieval loop). Document the implementation faithfully.

**Implementation:**

FIRE's key mechanism: the LLM first attempts to answer, then self-assesses confidence. If confident → emit verdict. If uncertain → generate a query, retrieve, incorporate results, and try again.

```python
# src/baselines/fire_baseline.py

class FIREBaseline:
    """Adapted from FIRE (Findings of NAACL 2025)."""
    
    def __init__(self, llm: LLMBackend, retrieval: RetrievalBackend,
                 confidence_threshold: float = 0.8,
                 max_iterations: int = 5):
        self.llm = llm
        self.retrieval = retrieval
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
    
    def verify(self, claim: str) -> Verdict:
        evidence_pool = []
        previous_queries = []
        
        for iteration in range(self.max_iterations):
            # Step 1: Attempt verification with current evidence
            assessment = self._assess_with_confidence(claim, evidence_pool)
            
            # Step 2: Check internal confidence
            if assessment["confidence"] >= self.confidence_threshold:
                return Verdict(
                    label=assessment["verdict"],
                    confidence=assessment["confidence"],
                    evidence=[p["pmid"] for p in evidence_pool],
                    reasoning=assessment["reasoning"]
                )
            
            # Step 3: Confidence insufficient — generate retrieval query
            query = self._generate_query(claim, evidence_pool, previous_queries)
            
            # Prevent repetitive queries (FIRE's dedup mechanism)
            if self._is_duplicate_query(query, previous_queries):
                break
            previous_queries.append(query)
            
            # Step 4: Retrieve and add to evidence pool
            results = self.retrieval.pubmed_api_search(query=query, k=5)
            evidence_pool.extend(results)
        
        # Max iterations — return best assessment so far
        return Verdict(
            label=assessment["verdict"],
            confidence=assessment["confidence"],
            evidence=[p["pmid"] for p in evidence_pool],
            reasoning=assessment["reasoning"]
        )
    
    def _assess_with_confidence(self, claim: str, evidence: list) -> dict:
        """Core FIRE mechanism: LLM self-assesses confidence."""
        evidence_text = format_evidence(evidence) if evidence else "No evidence retrieved yet."
        
        response = self.llm.complete(
            system="""You are a scientific claim verification system.
Assess the claim against the provided evidence. 

Output JSON with:
- "verdict": "SUPPORT" | "REFUTE" | "NEI"
- "confidence": 0.0-1.0 (how confident you are in this verdict given 
  the available evidence; be honest — low confidence means you need 
  more evidence)
- "reasoning": brief explanation
- "uncertainty_reason": if confidence < 0.8, explain what additional 
  evidence would increase your confidence""",
            user=f"Claim: {claim}\n\nEvidence:\n{evidence_text}"
        )
        return json.loads(response)
    
    def _generate_query(self, claim, evidence, previous_queries) -> str:
        """Generate a targeted retrieval query based on uncertainty."""
        response = self.llm.complete(
            system="""Generate a PubMed search query to find evidence that would 
help verify this claim. The query should target the specific uncertainty 
identified in the previous assessment. Output only the query.""",
            user=f"Claim: {claim}\n"
                 f"Current evidence count: {len(evidence)}\n"
                 f"Previous queries: {previous_queries}\n"
                 f"Uncertainty: {evidence[-1] if evidence else 'No evidence yet'}"
        )
        return response.strip()
    
    def _is_duplicate_query(self, query, previous) -> bool:
        """FIRE's repetitive query prevention."""
        # Simple: exact or high-overlap check
        for prev in previous:
            if fuzz.ratio(query.lower(), prev.lower()) > 85:
                return True
        return False
```

**Key adaptation decisions:**
- Original FIRE uses Google Search → adapt to PubMed API
- Original targets general factuality → adapt prompt to scientific claim verification
- Preserve the core mechanism: LLM-internal confidence as the stopping signal
- Match `confidence_threshold=0.8` to your system's sufficiency threshold for fair comparison

**What to report:** Compare the distribution of stopping iterations between FIRE and your system. If FIRE stops earlier (fewer iterations) but with lower accuracy, that shows the LLM's self-assessed confidence is miscalibrated.

**Cost:** 2-3 LLM calls per iteration × 1-5 iterations = ~5-15 LLM calls per claim.

**Implementation time:** 4-5 hours.

---

## 8. OpenScholar

**What it tests:** Whether a specialised scientific literature synthesis system — with a curated datastore of 45M papers, trained retrievers, and self-feedback — outperforms our evidence programming approach on claim verification (a task OpenScholar was not specifically designed for).

**Public framework:** Use the official OpenScholar codebase and models.
- Repository: https://github.com/AkariAsai/OpenScholar
- Models: OpenScholar-8B (fine-tuned Llama 3.1 8B) on HuggingFace
- Datastore: 45M papers from Semantic Scholar (peS2o) with passage embeddings
- The repo includes end-to-end inference scripts with retrieval, reranking, and self-feedback

**Recommended approach:** Run their pipeline as-is, adapting only input format (claims → questions) and output parsing (synthesis → verdict).

```python
# src/baselines/openscholar_baseline.py
# Wraps the official OpenScholar pipeline from:
#   https://github.com/AkariAsai/OpenScholar

class OpenScholarBaseline:
    """Uses the official OpenScholar pipeline.
    
    Two variants:
    - OS-8B: Their fine-tuned Llama 3.1 8B (local GPU)
    - OS-GPT4o: Their pipeline with GPT-4o backbone (API)
    """
    
    def __init__(self, model_name: str = "OpenScholar/OpenScholar-8B",
                 use_feedback: bool = True, top_n: int = 10):
        self.model_name = model_name
        self.use_feedback = use_feedback
        self.top_n = top_n
    
    def verify(self, claim: str) -> Verdict:
        # Frame claim as a question for OpenScholar
        query = (f"What does the current scientific evidence say about the "
                 f"following claim? Is it supported, refuted, or is there "
                 f"insufficient evidence? Claim: {claim}")
        
        # Run their pipeline (retrieval → reranking → generation → feedback)
        result = self._run_openscholar_pipeline(query)
        
        # Parse synthesis response into verdict
        return self._parse_to_verdict(result, claim)
```

**Setup:**
1. Clone https://github.com/AkariAsai/OpenScholar into `external/openscholar/`
2. Download OpenScholar-DataStore (passage embeddings)
3. Download reranker model
4. Run via their scripts:
```bash
# OS-8B (local)
python run.py --input_file claims.jsonl --model_name "OpenScholar/OpenScholar-8B" \
  --use_contexts --output_file results.jsonl --top_n 10 --feedback --ranking_ce --zero_shot

# OS-GPT4o (API)
python run.py --input_file claims.jsonl --model_name "gpt-4o" --api "openai" \
  --api_key_fp key.txt --use_contexts --output_file results.jsonl \
  --top_n 10 --feedback --ranking_ce --zero_shot
```

**Key adaptation challenge:** OpenScholar is designed for open-ended literature synthesis, NOT claim verification. The framing mismatch (synthesis → verification) should be acknowledged in the paper — it strengthens our argument if a task-specific system outperforms a general-purpose tool.

**What to report:** OpenScholar provides citation accuracy for free. Report whether it cites relevant papers even if verdict parsing is imperfect.

**Model note:** Report as "OpenScholar-8B" or "OS-GPT4o" — different model family than our backbone. The OS-GPT4o variant uses GPT-4o, closer to our backbone.

**Priority:** Deprioritize to "nice-to-have" unless reviewers explicitly request it. The ReAct + FIRE + SAFE baselines already cover the agentic comparison space well. The datastore download (~100GB+) and GPU requirements make this the highest-setup-cost baseline.

**Cost:** ~5-10 LLM-equivalent calls per claim (OS-8B local; OS-GPT4o API).

**Implementation time:** 6-8 hours (mainly setup: downloading datastore, models, environment).

---

## Implementation Priority and Timeline

### Phase 1: Foundation (get eval harness working end-to-end)

| System | Effort | Datasets | Notes |
|--------|--------|----------|-------|
| Shared infrastructure (retrieval, eval harness, LLM backend, label normalization) | 2-3 days | All | Build & test first |
| Dataset adapters (SciFact-Open, CIViC-Fact verbalization, ConnectomeDB2025, SIGNOR\*) | 1-2 days | All | Includes corpus indexing & negative generation |
| Random + LLM-only | 0.5 day | SciFact-Open, SIGNOR\* | Establish floor & parametric ceiling |
| Fixed-k RAG (k=5, k=10) + LLM + 5 Search | 1 day | SciFact-Open, SIGNOR\* | Static retrieval baselines |
| **Evidence Programming (our system)** | 1 day | SciFact-Open, SIGNOR\* | Adapt to shared harness |

### Phase 2: Core Agentic Comparisons (key ablation dimensions)

| System | Effort | Datasets | Key Comparison |
|--------|--------|----------|----------------|
| ReAct (same tools, no sufficiency signal) | 1-2 days | SciFact-Open, SIGNOR\* | Architecture: free-form vs. structured |
| FIRE (LLM confidence vs. MLP classifier) | 1-2 days | SciFact-Open, SIGNOR\* | Stopping signal: self-assessed vs. learned |

### Phase 3: Extended Comparisons

| System | Effort | Datasets | Key Comparison |
|--------|--------|----------|----------------|
| SAFE (per-fact independent verification) | 2 days | SciFact-Open, CIViC-Fact | Evidence sharing: none vs. shared state |
| ConnectomeDB2025 eval (all core systems) | 1-2 days | ConnectomeDB2025 | Molecular interactions with gold PMIDs; evidence recall |
| Full eval on CIViC-Fact (all systems) | 1-2 days | CIViC-Fact | Scale test; requires claim verbalization |

### Deferred (only if reviewers request)

| System | Effort | Notes |
|--------|--------|-------|
| OpenScholar (OS-8B or OS-GPT4o) | 3-4 days | Heavy setup (datastore download); synthesis→verification framing mismatch |

---

## Fairness Checklist

Before running experiments, verify these consistency conditions:

- [ ] **Label normalization** applied to both predictions and gold labels via `normalize_label()` before any metric computation (see Label Taxonomy section above)
- [ ] **Same LLM backbone** for all non-specialised baselines (except OpenScholar-8B — report model size)
- [ ] **Same verification prompt** for verdict extraction across Fixed-k RAG, LLM+Search, and final-step verdict in agentic systems
- [ ] **Same corpus** for BM25 retrieval (Fixed-k) and PubMed API search (agentic systems operate on the same underlying literature)
- [ ] **Same output schema** (Verdict dataclass) parsed the same way for all systems, including our evidence programming system via `verdict_from_verification()` adapter
- [ ] **Same evaluation harness** computing F1, per-class metrics, ECE, and evidence recall identically across all systems
- [ ] **Same max compute budget** — ensure no system gets dramatically more tokens than others without this being visible in the Cost column
- [ ] **Deterministic runs:** At temperature=0, a single run suffices (repeated runs are identical). For agentic systems (ReAct, FIRE, our system) where PubMed API results may vary, run 3 times with **varied claim order** (not just random seeds) to capture variance from API result ordering and rate limits
- [ ] **Cost tracking** captures ALL LLM calls including intermediate reasoning, query generation, relevance judgments — not just the final verdict call
- [ ] **CIViC-Fact label field** confirmed: use `gold_label_name` (SUPPORTS/REFUTES/NEI); filter `flagged == True` rows; use test partition only
- [ ] **ConnectomeDB eval files** confirmed: load from curated CSVs at `/hps/nobackup/saezrodriguez/shared_datasets/connectomedb/`; verify `Label` column in rejected file contains only `REFUTED`/`NEI`
- [ ] **SIGNOR flip logic** validated: 110 variants generated via `construct_signor_claim()` + `get_flipped_label()` from `experiments/run_signor_eval.py`; no manual claim strings
- [ ] **Public framework fidelity**: for systems using official codebases (SAFE, OpenScholar), document all modifications from the original (search backend swap, prompt changes, output parsing) in a reproducibility appendix
