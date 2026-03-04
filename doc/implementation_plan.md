# Framework Implementation Plan: Evidence Programming via RLM REPL

## Architecture Overview

The evidence programming framework implements the Recursive Language Model
(RLM) paradigm: the LLM generates Python code that runs in a persistent
Jupyter kernel. Evidence state lives as a Python variable in the kernel
rather than being serialized to disk on every operation. The kernel is the
"working memory"; the LLM's context window holds only the conversation and
the latest kernel output.

Two orchestration modes share a common pure-Python evidence library:

```
                 ┌──────────────────────────────────────────┐
                 │         Pure Python Library               │
                 │  evidence_api · subagents · classifier    │
                 │  compressor · data_models · renderers     │
                 └────────┬──────────────────┬──────────────┘
                          │                  │
             ┌────────────┴───┐    ┌─────────┴─────────────┐
             │  Mode A (SDK)  │    │  Mode B (Standalone)   │
             │                │    │                        │
             │  Claude Agent  │    │  repl_orchestrator.py  │
             │  SDK + nb_exe  │    │  OpenAI-compatible     │
             │  cute as tool  │    │  client + code-fence   │
             │                │    │  parsing               │
             └──────┬─────┬──┘    └────┬────────────┬──────┘
                    │     │            │            │
              ┌─────┘  ┌──┘       ┌────┘       ┌────┘
              ▼        ▼          ▼            ▼
        ┌──────────┐ ┌─────────────────────────────────┐
        │ Notebook  │ │       Jupyter Kernel             │
        │ MCP tools │ │  (KernelRunner)                  │
        │           │ │                                  │
        │ nb_init   │ │  state = EvidenceState(...)      │
        │ nb_execute│ │  search_pubmed(q, state)         │
        │ nb_render │ │  check_sufficiency(state)        │
        │ nb_save   │ │  emit_verdict(...)               │
        └───────┬──┘ └──────────────────────────────────┘
                │
          ┌─────┘
          ▼
   ┌───────────────┐
   │  .ipynb file   │
   │  (audit trail) │
   └───────────────┘
```

The RLM "give the LLM a REPL" principle means:
- Evidence state lives as a Python variable in the kernel
- The sufficiency classifier is called as a Python function
- Paper retrieval functions are called directly from evidence_api
- Subagent logic runs as Python functions with injected LLM callables
- The LLM generates code; the kernel output is the feedback signal

---

## 1. Project Structure (Actual)

```
src/pkevolve/verification/
├── __init__.py                  # Lazy imports facade
├── data_models.py               # Pydantic models: Fact, PaperRecord, Stance, Gap, etc.
├── evidence_state.py            # EvidenceState container, MAX_ITERATIONS, trace
├── evidence_api.py              # Pure Python library: all evidence functions (NEW)
├── subagents.py                 # Subagent logic as Python functions (NEW)
├── kernel_runner.py             # Jupyter kernel lifecycle management (NEW)
├── repl_orchestrator.py         # Mode B: standalone REPL loop (NEW)
├── orchestrator.py              # Mode A: Claude Agent SDK orchestrator (existing)
├── mcp_tools.py                 # Thin MCP wrappers over evidence_api (refactored)
├── notebook_mcp.py              # Notebook MCP tools, uses KernelRunner (refactored)
├── classifier.py                # Heuristic SufficiencyClassifier
├── compressor.py                # L1 deduplication compressor
├── renderers.py                 # HTML renderers for Jupyter
└── adapters.py                  # Dataset adapters

scripts/verification/
├── demo_evidence_programming.py # CLI entry point (--mode sdk|repl)
└── README.md
```

---

## 2. Evidence State as Files on Disk

The SDK's design principle is "the folder and file structure becomes a form of context engineering." Evidence state is persisted as JSON that the agent reads, writes, and queries through custom MCP tools. This is the "variable" the agent programs on.

### 2.1 Pydantic Schema

```python
# src/evidence_programming/state/schema.py

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Stance(str, Enum):
    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    NEUTRAL = "NEUTRAL"

class Paper(BaseModel):
    pmid: str
    title: str
    authors: list[str] = []
    abstract: str = ""              # defaults to empty (allows synthetic records)
    full_text: Optional[str] = None
    summary: Optional[str] = None
    source: str = "pubmed"  # pubmed | semantic_scholar

class Fact(BaseModel):
    id: str
    text: str
    stance: Stance
    source_pmid: str
    relevant_subclaims: list[str]
    confidence: float = 0.0

class Conflict(BaseModel):
    id: str
    fact_a_id: str
    fact_b_id: str
    description: str
    severity: float  # 0-1

class GapType(str, Enum):
    MISSING_SUBCLAIM = "missing_subclaim_evidence"
    CONTRADICTORY = "contradictory_evidence"
    LOW_DIVERSITY = "low_source_diversity"
    WEAK_STANCE = "weak_stance_evidence"
    MISSING_MECHANISM = "missing_mechanism"
    MISSING_QUANTITATIVE = "missing_quantitative"
    MISSING_TEMPORAL = "missing_temporal"
    MISSING_POPULATION = "missing_population"

class Gap(BaseModel):
    subclaim: str
    gap_type: GapType
    description: str
    priority: float  # 0-1

class SufficiencyResult(BaseModel):
    label: str        # SUFFICIENT_SUPPORT | SUFFICIENT_REFUTE | INSUFFICIENT
    confidence: float  # 0-1, calibrated
    gaps: list[Gap]

class EvidenceState(BaseModel):
    claim: str
    subclaims: list[str]
    papers: dict[str, Paper] = {}
    facts: list[Fact] = []
    conflicts: list[Conflict] = []
    coverage: dict[str, float] = {}        # subclaim -> coverage score
    synthesis: dict[str, str] = {}         # subclaim -> synthesis text
    extracted_pmids: list[str] = []        # PMIDs already processed for fact extraction
    sufficiency_history: list[SufficiencyResult] = []
    iteration: int = 0                     # 1-indexed: incremented AFTER each sufficiency check
    token_estimate: int = 0

class VerificationVerdict(BaseModel):
    """Structured output schema for the orchestrator's final answer."""
    verdict: str  # SUPPORT | REFUTE | INSUFFICIENT
    confidence: float
    reasoning: str
    key_evidence: list[str]
    gaps_remaining: list[str]
```

### 2.2 State Manager

```python
# src/evidence_programming/state/manager.py

import json
from pathlib import Path
from .schema import EvidenceState

class StateManager:
    """Handles persistence of evidence state to disk."""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.state_path = workspace / "evidence_state.json"
        self.trace_path = workspace / "trace.json"
        self.papers_dir = workspace / "papers"
        self.papers_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> EvidenceState:
        if self.state_path.exists():
            return EvidenceState.model_validate_json(self.state_path.read_text())
        raise FileNotFoundError(f"No state at {self.state_path}")
    
    def save(self, state: EvidenceState) -> None:
        self.state_path.write_text(state.model_dump_json(indent=2))
    
    def init(self, claim: str, subclaims: list[str]) -> EvidenceState:
        state = EvidenceState(claim=claim, subclaims=subclaims)
        self.save(state)
        return state
    
    def save_paper_text(self, pmid: str, text: str) -> Path:
        path = self.papers_dir / f"{pmid}.txt"
        path.write_text(text)
        return path
    
    def append_trace(self, operation: str, details: dict) -> None:
        trace = []
        if self.trace_path.exists():
            trace = json.loads(self.trace_path.read_text())
        trace.append({"operation": operation, **details})
        self.trace_path.write_text(json.dumps(trace, indent=2))
```

---

## 3. Evidence API — Pure Python Library

All evidence manipulation logic lives in `evidence_api.py` as ordinary
Python functions.  These are called directly in REPL mode (Mode B) and
via `nb_execute` code cells in SDK mode (Mode A).  The MCP tools in
`mcp_tools.py` are now thin wrappers that load state from disk, call the
corresponding `evidence_api` function, and save state back.

### 3.1 Function Signatures

```python
# src/pkevolve/verification/evidence_api.py

def formulate_pubmed_query(claim: str) -> str: ...
def search_pubmed(query: str, state: EvidenceState, max_results: int = 5) -> list[str]: ...
def search_pubmed_progressive(claim: str, state: EvidenceState, max_results_per_tier: int = 5) -> list[str]: ...
def search_for_gap(gap_description: str, state: EvidenceState, max_results: int = 3) -> list[str]: ...
def find_related_articles(pmid: str, state: EvidenceState, max_results: int = 5) -> list[str]: ...
def get_full_text_article(pmid: str, state: EvidenceState) -> str: ...
def get_paper_text(pmid: str, state: EvidenceState) -> str: ...
def add_facts_from_dicts(facts_data: list[dict], state: EvidenceState) -> int: ...
def update_synthesis(subclaim: str, synthesis_text: str, state: EvidenceState) -> None: ...
def add_conflict(fact_a_id: str, fact_b_id: str, description: str, severity: float, state: EvidenceState) -> str: ...
def get_evidence_summary(state: EvidenceState) -> str: ...
def check_sufficiency(state: EvidenceState) -> SufficiencyResult: ...
def compress_evidence(state: EvidenceState, target_tokens: int = 40000) -> EvidenceState: ...
def emit_verdict(verdict: str, confidence: float, reasoning: str,
                 key_evidence: list, gaps_remaining: list,
                 state: EvidenceState, workspace: Path) -> VerificationVerdict: ...
```

### 3.2 Design Principles

- **State by reference**: All functions take `state: EvidenceState` and
  mutate it in-place.  No disk I/O inside evidence_api — that responsibility
  lives in the MCP wrappers (for tool mode) or checkpoint_save (for REPL).
- **Print for feedback**: Functions use `print()` for output.  In REPL mode
  the kernel captures stdout and feeds it back to the LLM.
- **No MCP dependency**: evidence_api imports only data_models, classifier,
  compressor, and standard library.  It is testable without MCP.
- **Shared query logic**: `_generate_tiered_queries`, `_extract_symbol_subtokens`,
  `_BIO_VERB_TO_NOUN`, `_STOP_WORDS` are defined in evidence_api (moved from
  the former mcp_tools.py).

### 3.3 MCP Wrappers (Backward Compatibility)

`mcp_tools.py` retains the same 13 `@mcp.tool()` signatures but each body
is now 3-5 lines:

```python
@mcp.tool()
def search_pubmed(query: str, workspace: str, max_results: int = 5) -> str:
    state = _load_state(workspace)
    added_pmids = api.search_pubmed(query, state, max_results)
    _save_state(workspace, state)
    _trace(workspace, "search_pubmed", {"query": query, "added": len(added_pmids)})
    return f"Added {len(added_pmids)} new papers. PMIDs: {', '.join(added_pmids) or 'none'}"
```

This preserves compatibility with the original orchestrator.py (Mode A
without REPL) while ensuring all logic is in one place.

---

## 4. Subagents as Python Functions

In the RLM paradigm, subagent logic is implemented as plain Python
functions with dependency-injected LLM callables.  This replaces the
Claude Agent SDK's `Task` subagents with testable, MCP-free code.

### 4.1 Type Signature

```python
# src/pkevolve/verification/subagents.py

LLMCallable = Callable[[str], str]  # prompt → response text

def extract_facts(llm: LLMCallable, paper_text: str, claim: str,
                  subclaims: list[str], source_pmid: str) -> list[Fact]: ...

def synthesize_subclaim(llm: LLMCallable, facts: list[Fact],
                        subclaim: str) -> str: ...

def detect_conflicts(llm: LLMCallable, facts: list[Fact]) -> list[dict]: ...

def formulate_gap_queries(llm: LLMCallable, gaps: list[Gap]) -> list[str]: ...
```

### 4.2 Usage in REPL

In Mode B, the LLM callable is wired up in the kernel prelude or by the
REPL orchestrator.  The callable includes retry logic with exponential
backoff and guards against `None`/empty API responses:

```python
import time, os
from openai import OpenAI
client = OpenAI(base_url="https://api.z.ai/api/paas/v4/", api_key=os.environ["GLM_API_KEY"])

def llm(prompt: str, _retries: int = 3) -> str:
    for attempt in range(_retries):
        try:
            resp = client.chat.completions.create(
                model="glm-4.6",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
        except Exception as e:
            print(f"llm(): error on attempt {attempt+1}: {e}")
        if attempt < _retries - 1:
            time.sleep(2 ** attempt)
    return ""  # graceful degradation

facts = extract_facts(llm, paper_text, state.claim, state.subclaims, pmid)
add_facts_from_dicts([f.model_dump() for f in facts], state)
```

### 4.3 Advantages over SDK Task Subagents

- **Testable**: Each function can be unit tested with a mock `llm` callable
- **Portable**: No dependency on Claude Agent SDK
- **Transparent**: Prompts and parsing logic are visible Python code
- **Composable**: Functions can be chained in a single code cell

---

## 5. Dual-Mode Orchestrator

### 5.1 Mode A — Claude Agent SDK + nb_execute

The agent uses the Claude Agent SDK's `query()` loop with a single MCP
server (notebook-tools). The `nb_execute` tool is the REPL gateway: the
LLM generates Python code that calls evidence_api functions directly in
a persistent Jupyter kernel.

```python
# scripts/verification/demo_evidence_programming.py  (simplified)

options = ClaudeAgentOptions(
    model="glm-4.6",
    system_prompt=SYSTEM_PROMPT,  # instructs LLM to use nb_execute
    allowed_tools=[
        "Task", "Read",
        "mcp__notebook-tools__nb_init",
        "mcp__notebook-tools__nb_execute",   # ← primary REPL tool
        "mcp__notebook-tools__nb_markdown",
        "mcp__notebook-tools__nb_render_*",
        "mcp__notebook-tools__nb_save",
    ],
    mcp_servers={
        "notebook-tools": {
            "command": sys.executable,
            "args": ["-m", "pkevolve.verification.notebook_mcp"],
        },
    },
)

async for message in query(prompt=prompt, options=options):
    ...  # stream messages
```

**Key difference from the original design:** no evidence-tools MCP server.
All 13 evidence tools are replaced by a single `nb_execute` call per step.
The notebook is both the REPL and the audit trail.

### 5.2 Mode B — Standalone REPL Orchestrator

A pure Python loop that uses the OpenAI-compatible chat API. No Claude
Agent SDK required.

```python
# src/pkevolve/verification/repl_orchestrator.py  (simplified)

runner = KernelRunner("repl-session")
runner.start()
runner.inject_prelude(claim=claim, workspace=str(workspace))

messages = [{"role": "system", "content": SYSTEM_PROMPT}, ...]

for turn in range(MAX_TURNS):
    response = client.chat.completions.create(model=model, messages=messages)
    code = extract_code(response.choices[0].message.content)
    outputs = runner.execute(code)
    output_text = outputs_to_text(outputs)
    messages.append({"role": "user", "content": f"Kernel output:\n{output_text}"})

    if (workspace / "verdict.json").exists():
        break
```

### 5.3 Kernel Runner (Shared Infrastructure)

Both modes share `KernelRunner` for Jupyter kernel lifecycle:

- `start()` — start a Python 3 kernel
- `execute(code)` — run code, return structured outputs
- `inject_prelude(claim, workspace)` — import evidence_api, init state
- `shutdown()` — clean up

The prelude injects all imports and initializes `state = EvidenceState.init_new(...)`.

### 5.4 Guard Rails

Guard rails are now enforced in the evidence API and state machine:

- **Iteration limit**: `EvidenceState.MAX_ITERATIONS = 8`. Both
  `check_sufficiency()` and the MCP wrapper enforce this.
- **Token budget**: `compress_evidence()` is called when `state.token_count() > 40000`.
- **Verdict requirement**: The loop terminates only when `verdict.json` exists.
- **Trace log**: `state.append_trace()` records every operation for audit.

---

## 6. Sufficiency Classifier (Training & Inference)

### 6.1 Feature Extraction

```python
# src/evidence_programming/classifier/features.py

import numpy as np
from ..state.schema import EvidenceState

def extract_features(state: EvidenceState) -> np.ndarray:
    """
    Extract 16 features from evidence state for the MLP classifier.
    No LLM calls — uses only pre-computed values from state.
    
    CANONICAL FEATURE SET: This is the authoritative feature definition.
    The paper plan's EvidenceStateSnapshot must derive from these same fields.
    Features like verbalized_confidence, consistency_score, entropy, and
    embedding similarities (max_similarity, mean_similarity) from the paper
    plan's snapshot are NOT used — they would require LLM calls, breaking
    the "~0 cost" property of the feedback instrument.
    """
    n_subclaims = max(len(state.subclaims), 1)
    coverages = [state.coverage.get(sc, 0.0) for sc in state.subclaims]
    n_facts = len(state.facts)
    n_papers = len(state.papers)
    
    support_facts = sum(1 for f in state.facts if f.stance == "SUPPORT")
    refute_facts = sum(1 for f in state.facts if f.stance == "REFUTE")
    unique_sources = len(set(f.source_pmid for f in state.facts)) if state.facts else 0
    
    features = [
        # Coverage features (3)
        min(coverages) if coverages else 0.0,
        sum(coverages) / n_subclaims,
        sum(1 for c in coverages if c >= 0.7) / n_subclaims,
        
        # Quantity features (2)
        min(n_papers / 20, 1.0),
        min(n_facts / 50, 1.0),
        
        # Conflict features (2)
        len(state.conflicts),
        max((c.severity for c in state.conflicts), default=0.0),
        
        # Synthesis coverage (1)
        sum(1 for sc in state.subclaims if sc in state.synthesis) / n_subclaims,
        
        # Confidence signals from history (3)
        state.sufficiency_history[-1].confidence if state.sufficiency_history else 0.0,
        len(state.sufficiency_history) / 8.0,  # iteration progress
        len(state.sufficiency_history[-1].gaps) / 5.0 if state.sufficiency_history else 1.0,
        
        # Evidence balance (3)
        support_facts / max(n_facts, 1),
        refute_facts / max(n_facts, 1),
        unique_sources / max(n_papers, 1),
        
        # Fact confidence stats (2)
        np.mean([f.confidence for f in state.facts]) if state.facts else 0.0,
        np.std([f.confidence for f in state.facts]) if len(state.facts) > 1 else 0.0,
    ]
    
    return np.array(features, dtype=np.float32)
```

### 6.2 Model

```python
# src/evidence_programming/classifier/model.py

import torch
import torch.nn as nn

class SufficiencyClassifier(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64, n_gap_types=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sufficiency_head = nn.Linear(hidden_dim, 3)  # SUP, REF, INSUFF
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.gap_head = nn.Sequential(
            nn.Linear(hidden_dim, n_gap_types),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.encoder(x)
        return {
            "sufficiency_logits": self.sufficiency_head(h),
            "confidence": self.confidence_head(h),
            "gap_probs": self.gap_head(h),
        }
```

### 6.3 Training Data Generation (Self-Consistency)

```python
# src/scripts/generate_training_data.py

"""
Generate training data for the sufficiency classifier using self-consistency.

For each claim in SciFact:
1. Retrieve evidence at k ∈ {1, 3, 5, 10, 15, 20} papers
2. At each k, sample 10 LLM verdicts at temperature 0.7
3. Compute agreement rate
4. Label based on agreement + gold label match
"""

import asyncio
import json
from pathlib import Path
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def sample_verdicts(claim: str, evidence: str, n_samples: int = 10) -> list[str]:
    """Sample n verdicts from the LLM at temperature 0.7."""
    verdicts = []
    tasks = []
    for _ in range(n_samples):
        tasks.append(client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": (
                    f"Based on the following evidence, is this claim SUPPORTED, "
                    f"REFUTED, or is there INSUFFICIENT evidence?\n\n"
                    f"Claim: {claim}\n\nEvidence:\n{evidence}\n\n"
                    f"Answer with exactly one word: SUPPORTED, REFUTED, or INSUFFICIENT"
                )
            }]
        ))
    
    responses = await asyncio.gather(*tasks)
    for resp in responses:
        text = resp.content[0].text.strip().upper()
        if "SUPPORT" in text:
            verdicts.append("SUPPORT")
        elif "REFUT" in text:
            verdicts.append("REFUTE")
        else:
            verdicts.append("INSUFFICIENT")
    
    return verdicts


def compute_label(verdicts: list[str], gold_label: str) -> tuple[str, str]:
    """Compute training label from self-consistency signal.
    
    Returns:
        (training_label, diagnostic_label) where:
        - training_label is one of: SUFFICIENT_SUPPORT, SUFFICIENT_REFUTE, INSUFFICIENT
          (maps to the classifier's 3-class head)
        - diagnostic_label adds CONFIDENT_WRONG for failure mode analysis
          (tracked separately, not used in classifier training)
    """
    from collections import Counter
    counts = Counter(verdicts)
    majority = counts.most_common(1)[0]
    agreement = majority[1] / len(verdicts)
    majority_verdict = majority[0]
    
    if agreement >= 0.8 and majority_verdict == gold_label:
        return f"SUFFICIENT_{gold_label}", f"SUFFICIENT_{gold_label}"
    elif agreement >= 0.8 and majority_verdict != gold_label:
        # CONFIDENT_WRONG: high agreement but wrong answer.
        # For classifier training: map to INSUFFICIENT (conservative — don't
        # let the classifier call this "sufficient" since the verdict is wrong).
        # For failure mode analysis: track separately as CONFIDENT_WRONG.
        return "INSUFFICIENT", "CONFIDENT_WRONG"
    else:
        return "INSUFFICIENT", "INSUFFICIENT"


async def generate_training_data(
    claims_path: Path,
    corpus_path: Path,
    output_path: Path,
    k_values: list[int] = [1, 3, 5, 10, 15, 20]
):
    """Generate training dataset for the sufficiency classifier."""
    # Load SciFact claims and corpus
    claims = json.loads(claims_path.read_text())
    # ... retrieval and sampling logic ...
    # Each sample: (features, label) where features come from the evidence state
    # at retrieval depth k, and label comes from self-consistency + gold
    pass
```

### 6.3.1 Heuristic Gap Label Generation

The sufficiency head is trained with self-consistency labels, but the gap head needs
labels identifying *what type* of evidence is missing. These are derived heuristically
from the evidence state — no LLM calls required during label generation.

```python
# src/evidence_programming/classifier/gap_labels.py

"""
Generate gap type training labels from evidence state features.

The MLP gap head predicts which of 8 gap types are present. Training labels
are derived heuristically from evidence state, not from LLM analysis.
This keeps the entire classifier pipeline LLM-free.
"""

from ..state.schema import EvidenceState, GapType
import numpy as np

def generate_gap_labels(state: EvidenceState) -> dict[str, float]:
    """
    Produce a binary label (0 or 1) for each of the 8 gap types.
    These become the multi-label targets for the gap_head during training.
    
    Heuristics are intentionally simple — the MLP learns to refine them.
    """
    n_subclaims = max(len(state.subclaims), 1)
    labels = {}
    
    # 1. MISSING_SUBCLAIM: any subclaim with fewer than 2 relevant SUPPORT/REFUTE facts
    undercovered = sum(
        1 for sc in state.subclaims
        if len([f for f in state.facts
                if sc in f.relevant_subclaims and f.stance != "NEUTRAL"]) < 2
    )
    labels[GapType.MISSING_SUBCLAIM.value] = 1.0 if undercovered > 0 else 0.0
    
    # 2. CONTRADICTORY: unresolved conflicts with severity > 0.5
    severe_conflicts = [c for c in state.conflicts if c.severity > 0.5]
    labels[GapType.CONTRADICTORY.value] = 1.0 if len(severe_conflicts) > 0 else 0.0
    
    # 3. LOW_DIVERSITY: fewer unique source papers than subclaims
    unique_sources = len(set(f.source_pmid for f in state.facts)) if state.facts else 0
    labels[GapType.LOW_DIVERSITY.value] = 1.0 if unique_sources < n_subclaims else 0.0
    
    # 4. WEAK_STANCE: >50% of facts are NEUTRAL
    if state.facts:
        neutral_ratio = sum(1 for f in state.facts if f.stance == "NEUTRAL") / len(state.facts)
        labels[GapType.WEAK_STANCE.value] = 1.0 if neutral_ratio > 0.5 else 0.0
    else:
        labels[GapType.WEAK_STANCE.value] = 1.0
    
    # 5-8. Semantic gap types: derived from keyword absence in extracted facts.
    # These are coarser signals — the MLP learns the boundary.
    fact_text = " ".join(f.text.lower() for f in state.facts)
    
    mechanism_keywords = {"mechanism", "pathway", "causes", "mediates", "via", "through"}
    labels[GapType.MISSING_MECHANISM.value] = (
        0.0 if any(kw in fact_text for kw in mechanism_keywords) else 1.0
    )
    
    quant_keywords = {"mg", "dose", "%", "fold", "ci ", "p=", "p<", "ratio", "odds"}
    labels[GapType.MISSING_QUANTITATIVE.value] = (
        0.0 if any(kw in fact_text for kw in quant_keywords) else 1.0
    )
    
    temporal_keywords = {"weeks", "months", "years", "longitudinal", "follow-up", "duration"}
    labels[GapType.MISSING_TEMPORAL.value] = (
        0.0 if any(kw in fact_text for kw in temporal_keywords) else 1.0
    )
    
    population_keywords = {"patients", "subjects", "cohort", "participants", "population", "n="}
    labels[GapType.MISSING_POPULATION.value] = (
        0.0 if any(kw in fact_text for kw in population_keywords) else 1.0
    )
    
    return labels
```

### 6.4 CLI Inference Entry Point

```python
# src/evidence_programming/classifier/inference.py

"""
CLI entry point for sufficiency classification.
Called by the check_sufficiency MCP tool.

Usage: python -m evidence_programming.classifier.inference evidence_state.json
Outputs: JSON with label, confidence, gaps
"""

import sys
import json
import torch
from pathlib import Path
from ..state.schema import EvidenceState, Gap, GapType
from .features import extract_features
from .model import SufficiencyClassifier

MODEL_PATH = Path(__file__).parent.parent.parent.parent / "data" / "classifier" / "best_model.pt"

GAP_TYPES = list(GapType)

def predict(state: EvidenceState) -> dict:
    model = SufficiencyClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    features = extract_features(state)
    x = torch.tensor(features).unsqueeze(0)
    
    with torch.no_grad():
        output = model(x)
    
    # Sufficiency label
    labels = ["SUFFICIENT_SUPPORT", "SUFFICIENT_REFUTE", "INSUFFICIENT"]
    label_idx = output["sufficiency_logits"].argmax(dim=1).item()
    label = labels[label_idx]
    
    # Calibrated confidence
    confidence = output["confidence"].item()
    
    # Gap predictions
    gap_probs = output["gap_probs"].squeeze().numpy()
    gaps = []
    for i, prob in enumerate(gap_probs):
        if prob > 0.3:  # threshold for reporting a gap
            # Map gap to the lowest-coverage subclaim of that type
            subclaim = min(
                state.subclaims,
                key=lambda sc: state.coverage.get(sc, 0.0)
            )
            gaps.append(Gap(
                subclaim=subclaim,
                gap_type=GAP_TYPES[i],
                description=f"{GAP_TYPES[i].value} for subclaim: {subclaim}",
                priority=float(prob)
            ))
    
    gaps.sort(key=lambda g: g.priority, reverse=True)
    
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "gaps": [g.model_dump() for g in gaps]
    }


if __name__ == "__main__":
    state_path = Path(sys.argv[1])
    state = EvidenceState.model_validate_json(state_path.read_text())
    result = predict(state)
    print(json.dumps(result))
```

---

## 7. Evaluation Runner

```python
# src/evidence_programming/evaluation/runner.py

import asyncio
import json
from pathlib import Path
from ..orchestrator import verify_claim
from ..state.manager import StateManager
from .metrics import compute_metrics

async def evaluate_dataset(
    claims_path: Path,
    output_dir: Path,
    model: str = "claude-sonnet-4-5-20250929",
    max_workers: int = 5
):
    """Run evidence programming on a dataset of claims."""
    claims = json.loads(claims_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_claim(claim_data):
        async with semaphore:
            claim_id = claim_data["id"]
            claim_text = claim_data["claim"]
            workspace = output_dir / f"claim_{claim_id}"
            
            # Initialize state
            manager = StateManager(workspace)
            manager.init(claim_text, subclaims=[])  # orchestrator decomposes
            
            # Run evidence programming
            verdict = await verify_claim(
                claim=claim_text,
                workspace=workspace,
                model=model
            )
            
            # Save result
            result = {
                "claim_id": claim_id,
                "claim": claim_text,
                "gold_label": claim_data.get("label"),
                "predicted": verdict.model_dump(),
            }
            (workspace / "result.json").write_text(json.dumps(result, indent=2))
            
            return result
    
    tasks = [process_claim(c) for c in claims]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors
    valid_results = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]
    
    # Compute metrics
    metrics = compute_metrics(valid_results)
    
    summary = {
        "total": len(claims),
        "completed": len(valid_results),
        "errors": len(errors),
        "metrics": metrics
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    
    return summary
```

---

## 8. Implementation Stages (Revised for RLM REPL)

### Stage 1: Pure Python Library (DONE)

**Goal:** All evidence logic callable without MCP.

Files created/modified:
- `evidence_api.py` — 13 evidence functions as plain Python
- `subagents.py` — 4 subagent functions with injected LLM callable
- `kernel_runner.py` — Jupyter kernel lifecycle management
- `evidence_state.py` — added MAX_ITERATIONS, trace, checkpoint_save
- `mcp_tools.py` — refactored to thin wrappers over evidence_api
- `notebook_mcp.py` — refactored to use KernelRunner

**Verification:** `from pkevolve.verification.evidence_api import *` works.

### Stage 2: Standalone REPL Orchestrator (DONE)

**Goal:** Mode B working end-to-end without Claude Agent SDK.

Files created:
- `repl_orchestrator.py` — OpenAI client loop with code-fence parsing

**Verification:** `verify_claim_repl(claim, workspace)` produces verdict.json.

### Stage 3: SDK REPL Mode (DONE)

**Goal:** Mode A using nb_execute as primary tool.

Files modified:
- `demo_evidence_programming.py` — --mode sdk|repl, single notebook-tools MCP server

**Verification:** `--mode sdk` runs with only notebook-tools MCP server.

### Stage 3.5: Post-Integration Hardening (DONE)

**Goal:** Fix all runtime errors discovered during end-to-end demo runs.

Four rounds of diagnose → fix → re-run identified and resolved the
following issues across 6 files:

**Round 1 — Structural wiring (6 fixes)**
- `evidence_state.py`: auto-save on mutation via `checkpoint_save()`
- `evidence_api.py`: field alias normalization (`statement` → `text`)
- `evidence_api.py`: fact deduplication in `add_facts_from_dicts()`
- `evidence_api.py`: `schema_docs()` added — auto-generated from Pydantic models
- `kernel_runner.py`: `llm()` callable injected in kernel prelude
- `repl_orchestrator.py`: verdict quality gate + anti-fabrication prompt

**Round 2 — Extraction quality (4 fixes)**
- `evidence_api.py`: `extract_and_add_facts()` tries `get_full_text_article()` before abstract
- `evidence_api.py`: `add_facts_from_dicts()` validates `source_pmid ∈ state.papers`
- `evidence_api.py`: `schema_docs()` enriched with EvidenceState field listing,
  type annotations, and "Common pitfalls" section
- `repl_orchestrator.py` + `demo_evidence_programming.py`: "Grounded Evidence Only"
  anti-fabrication rules added to both system prompts

**Round 3 — PMC full-text retrieval (3 fixes)**
- `evidence_api.py`: elink `LinkName` filter — only accept `pubmed_pmc`, not `pubmed_pmc_refs`
- `evidence_api.py`: title cross-validation — discard PMC text with < 25% word overlap
- `evidence_api.py`: `add_facts_from_dicts()` now rejects (not just warns) unknown PMIDs

**Round 4 — LLM callable robustness (3 fixes)**
- `kernel_runner.py` + `demo_evidence_programming.py`: `llm()` callable now retries
  3× with exponential backoff and guards against `None`/empty `resp.choices`
- `subagents.py`: `extract_facts()` returns `[]` if `llm()` returns empty string
- `evidence_api.py`: `extract_and_add_facts()` appends to `state.extracted_pmids`

**Round 5 — Schema visibility (2 fixes)**
- `data_models.py`: `PaperRecord.abstract` default changed from required to `""`
- `evidence_api.py`: `schema_docs()` now includes `PaperRecord` field listing with
  required/default annotations; "Common pitfalls" warns `authors` must be `list[str]`

**Verification:** All fixes validated with import tests, assertion checks, and
live PubMed API calls.  Demo runs exit cleanly with valid `verdict.json`.

### Stage 4: Sufficiency Classifier (existing)

**Goal:** Trained classifier returning SufficiencyResult.

Already implemented in `classifier.py`.  Heuristic version is working.
Future: train MLP on self-consistency labels.

### Stage 5: Evaluation + Notebook Renderers

**Goal:** Renderers that work from in-memory state (not disk).

Tasks:
- Add `render_papers_from_state(state)` etc. variants to `renderers.py`
- Batch evaluation runner using `verify_claims_batch()`
- Cost/accuracy metrics

### Stage 6: Writing + Polish

**Goal:** Paper draft with RLM framing.

Tasks:
- Motivating example: side-by-side MCP vs REPL comparison
- Ablation: tool-call overhead vs REPL efficiency
- Token usage analysis

---

## 9. Dependencies and Environment

```toml
# pyproject.toml (relevant subset)
[project]
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "openai>=1.0",        # Mode B REPL orchestrator
    "jupyter-client>=8.0", # kernel management
    "ipykernel",           # Python 3 kernel
    "nbformat>=5.0",       # notebook I/O
    "mcp",                 # MCP server (for backward-compat tool wrappers)
    "python-dotenv",
]

[project.optional-dependencies]
sdk = ["claude-agent-sdk>=0.1.20"]  # Mode A only
```

Environment variables:
```bash
export GLM_API_KEY=...              # Required for both modes
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic  # Mode A
```

---

## 10. Key Design Decisions and Rationale

### Why REPL instead of MCP tools?

The RLM paradigm replaces 13 tool-call round-trips per iteration with a
single code execution.  Each tool call in the MCP approach requires:
serialization → IPC → deserialization → disk I/O → response serialization.
In REPL mode, `search_pubmed(q, state)` is a direct function call in the
kernel — zero serialization overhead, and the state stays in memory.

### Why keep MCP wrappers?

Backward compatibility.  The original orchestrator.py (Mode A without REPL)
still works with the thin MCP wrappers.  The wrappers delegate to
evidence_api, so logic is not duplicated.

### Why Jupyter kernel instead of exec()?

1. **Process isolation**: kernel crashes don't kill the orchestrator.
2. **Persistent state**: variables survive across code cells.
3. **Rich output**: display_data messages support HTML (renderers).
4. **Notebook artifact**: every code cell is recorded in the .ipynb file.

### Why subagents as Python functions with injected LLM callable?

- **Testable**: `extract_facts(mock_llm, ...)` can be unit tested.
- **Portable**: No Claude Agent SDK dependency.
- **Transparent**: Prompts are visible Python strings, parsing is explicit.
- **Composable**: Can be chained in a single REPL cell.

### Why dual mode (SDK + standalone)?

Mode A (SDK) is useful when the Claude Agent SDK is available and provides
built-in features like `Task` subagents, `Read` file access, and telemetry.
Mode B (standalone) requires only `openai` and `jupyter-client`, making it
runnable against any OpenAI-compatible endpoint without installing the SDK.
Both modes share the same evidence_api, subagents, and kernel infrastructure.
