# Framework Implementation Plan: Evidence Programming via Claude Agent SDK

## Architecture Overview

The evidence programming framework maps directly onto the Claude Agent SDK's architecture. The SDK gives agents a computer — bash, file I/O, subagents, custom MCP tools — which is exactly the REPL environment evidence programming requires. The agent programs on evidence by invoking custom tools that manipulate an evidence state persisted to disk, with the sufficiency classifier running as a local Python process invoked via bash.

```
┌─────────────────────────────────────────────────────────┐
│                  Orchestrator Agent                       │
│  (Claude Agent SDK main loop)                            │
│                                                          │
│  System prompt: evidence programming instructions        │
│  Tools: all custom MCP tools + Bash + Read + Write       │
│  Structured output: VerificationVerdict schema            │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │  Retrieval  │  │ Extraction │  │  Sufficiency     │   │
│  │  Tools      │  │ Subagent   │  │  Classifier      │   │
│  │  (MCP)      │  │ (Task)     │  │  (Bash/MCP)      │   │
│  └──────┬─────┘  └──────┬─────┘  └────────┬─────────┘   │
│         │               │                  │              │
│         └───────────┬───┘──────────────────┘              │
│                     ▼                                     │
│           ┌─────────────────┐                             │
│           │  Evidence State  │                             │
│           │  (JSON on disk)  │                             │
│           └─────────────────┘                             │
└─────────────────────────────────────────────────────────┘
```

The SDK's "give your agent a computer" principle means:
- Evidence state lives as JSON files the agent reads and writes
- The sufficiency classifier is a Python script invoked via Bash
- Paper retrieval is a custom MCP tool wrapping PubMed/Semantic Scholar APIs
- Fact extraction and synthesis use subagents with isolated context
- The orchestrator agent decides what to do based on sufficiency feedback

---

## 1. Project Structure

```
evidence-programming/
├── pyproject.toml
├── CLAUDE.md                          # Memory/instructions for the orchestrator
├── .claude/
│   ├── agents/
│   │   ├── fact-extractor.md          # Subagent: extracts facts from papers
│   │   ├── synthesizer.md            # Subagent: synthesizes evidence per subclaim
│   │   ├── conflict-detector.md      # Subagent: finds contradictions
│   │   └── gap-query-formulator.md   # Subagent: translates MLP gap types into search queries
│   └── commands/
│       └── verify.md                  # Slash command: /verify <claim>
├── src/
│   ├── evidence_programming/
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # Main entry: runs the evidence programming loop
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py          # MCP tools: search_pubmed, search_semantic_scholar
│   │   │   ├── evidence_state.py     # MCP tools: read/write/query evidence state
│   │   │   ├── sufficiency.py        # MCP tool: check_sufficiency (wraps classifier)
│   │   │   └── compression.py        # MCP tool: compress_evidence
│   │   ├── classifier/
│   │   │   ├── __init__.py
│   │   │   ├── model.py              # SufficiencyClassifier (PyTorch MLP)
│   │   │   ├── features.py           # 16-feature extraction from evidence state
│   │   │   ├── train.py              # Training loop with self-consistency labels
│   │   │   └── inference.py          # CLI entry: python -m classifier.inference state.json
│   │   │   └── gap_labels.py          # Heuristic gap label generation for training
│   │   ├── state/
│   │   │   ├── __init__.py
│   │   │   ├── schema.py             # Pydantic models: EvidenceState, Fact, Paper, etc.
│   │   │   └── manager.py            # State persistence: load/save/snapshot
│   │   ├── compression/
│   │   │   ├── __init__.py
│   │   │   └── compress.py            # Three-level compression with sufficiency invariant
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── runner.py             # Batch evaluation across datasets
│   │       ├── baselines.py          # Baseline implementations
│   │       └── metrics.py            # F1, ECE, cost tracking
│   └── scripts/
│       ├── train_classifier.py       # Train sufficiency classifier
│       ├── generate_training_data.py # Self-consistency sampling
│       └── run_evaluation.py         # Full evaluation pipeline
├── data/
│   ├── scifact/                      # SciFact dataset
│   ├── scifact_open/                 # SciFact-Open (500K abstracts)
│   └── classifier/                   # Trained classifier checkpoints
├── workspaces/                       # Per-claim working directories
│   └── claim_<id>/
│       ├── evidence_state.json       # Current evidence state
│       ├── papers/                   # Retrieved paper texts
│       └── trace.json                # Audit log
└── tests/
    ├── test_tools.py
    ├── test_classifier.py
    ├── test_state.py
    └── test_integration.py
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
    abstract: str
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

## 3. Custom MCP Tools (The Evidence Programming Instruction Set)

Each tool is a Python function registered via `@tool` decorator and served as an in-process MCP server. The orchestrator agent invokes these to manipulate evidence state — this is how it "programs on evidence."

### 3.1 Retrieval Tools

```python
# src/evidence_programming/tools/retrieval.py

from claude_agent_sdk import tool
from ..state.manager import StateManager
from ..state.schema import Paper
import httpx

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

@tool(
    "search_pubmed",
    "Search PubMed for papers relevant to a query. Returns list of PMIDs added to evidence state.",
    {"query": str, "max_results": int}
)
async def search_pubmed(args: dict) -> dict:
    query = args["query"]
    max_results = args.get("max_results", 5)
    workspace = args.get("_workspace", ".")
    
    manager = StateManager(Path(workspace))
    state = manager.load()
    
    async with httpx.AsyncClient() as client:
        # Step 1: Search for PMIDs
        search_resp = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json"
        })
        pmids = search_resp.json()["esearchresult"]["idlist"]
        
        # Step 2: Fetch abstracts
        if pmids:
            fetch_resp = await client.get(f"{PUBMED_BASE}/efetch.fcgi", params={
                "db": "pubmed", "id": ",".join(pmids),
                "rettype": "abstract", "retmode": "xml"
            })
            papers = _parse_pubmed_xml(fetch_resp.text)
            
            added = []
            for paper in papers:
                if paper.pmid not in state.papers:
                    state.papers[paper.pmid] = paper
                    manager.save_paper_text(paper.pmid, paper.abstract)
                    added.append(paper.pmid)
            
            state.token_estimate += sum(len(p.abstract.split()) for p in papers)
            manager.save(state)
            manager.append_trace("search_pubmed", {
                "query": query, "found": len(pmids), "added": len(added)
            })
            
            return {"content": [{"type": "text", "text": 
                f"Found {len(pmids)} papers, added {len(added)} new. "
                f"PMIDs: {', '.join(added)}"
            }]}
    
    return {"content": [{"type": "text", "text": "No results found."}]}


@tool(
    "search_semantic_scholar",
    "Search Semantic Scholar for papers. Useful for broader coverage beyond PubMed.",
    {"query": str, "max_results": int}
)
async def search_semantic_scholar(args: dict) -> dict:
    # Similar structure, uses Semantic Scholar API
    ...
```

### 3.2 Evidence State Tools

```python
# src/evidence_programming/tools/evidence_state.py

from claude_agent_sdk import tool
from pathlib import Path
from ..state.manager import StateManager
from ..state.schema import Fact, Conflict, Stance

@tool(
    "get_evidence_summary",
    "Get a summary of the current evidence state: paper count, fact count, "
    "coverage per subclaim, conflicts, and latest sufficiency result.",
    {}
)
async def get_evidence_summary(args: dict) -> dict:
    manager = StateManager(Path(args.get("_workspace", ".")))
    state = manager.load()
    
    summary_lines = [
        f"Claim: {state.claim}",
        f"Iteration: {state.iteration}",
        f"Papers: {len(state.papers)}",
        f"Facts: {len(state.facts)} (support: {sum(1 for f in state.facts if f.stance == 'SUPPORT')}, "
        f"refute: {sum(1 for f in state.facts if f.stance == 'REFUTE')})",
        f"Conflicts: {len(state.conflicts)}",
        f"Token estimate: ~{state.token_estimate}",
        "",
        "Coverage per subclaim:"
    ]
    for sc in state.subclaims:
        cov = state.coverage.get(sc, 0.0)
        synth = "✓" if sc in state.synthesis else "✗"
        summary_lines.append(f"  [{cov:.1%}] [synth:{synth}] {sc}")
    
    if state.sufficiency_history:
        latest = state.sufficiency_history[-1]
        summary_lines.extend([
            "",
            f"Latest sufficiency: {latest.label} (confidence: {latest.confidence:.2f})",
            f"Gaps: {len(latest.gaps)}"
        ])
        for gap in latest.gaps[:5]:
            summary_lines.append(f"  - [{gap.gap_type}] {gap.description}")
    
    return {"content": [{"type": "text", "text": "\n".join(summary_lines)}]}


@tool(
    "add_facts",
    "Add extracted facts to the evidence state. Each fact has text, stance "
    "(SUPPORT/REFUTE/NEUTRAL), source PMID, and relevant subclaims.",
    {"facts": list}
)
async def add_facts(args: dict) -> dict:
    manager = StateManager(Path(args.get("_workspace", ".")))
    state = manager.load()
    
    added = 0
    for f in args["facts"]:
        fact = Fact(
            id=f"fact_{len(state.facts) + added}",
            text=f["text"],
            stance=Stance(f["stance"]),
            source_pmid=f["source_pmid"],
            relevant_subclaims=f.get("relevant_subclaims", []),
            confidence=f.get("confidence", 0.5)
        )
        state.facts.append(fact)
        added += 1
    
    # Recompute coverage (only SUPPORT/REFUTE count — NEUTRAL facts don't
    # provide evidence for or against the subclaim)
    for sc in state.subclaims:
        relevant = [f for f in state.facts 
                    if sc in f.relevant_subclaims and f.stance != Stance.NEUTRAL]
        state.coverage[sc] = min(1.0, len(relevant) / 3.0)
    
    manager.save(state)
    manager.append_trace("add_facts", {"count": added})
    return {"content": [{"type": "text", "text": f"Added {added} facts. Coverage updated."}]}


@tool(
    "update_synthesis",
    "Update the evidence synthesis for a specific subclaim.",
    {"subclaim": str, "synthesis": str}
)
async def update_synthesis(args: dict) -> dict:
    manager = StateManager(Path(args.get("_workspace", ".")))
    state = manager.load()
    state.synthesis[args["subclaim"]] = args["synthesis"]
    manager.save(state)
    return {"content": [{"type": "text", "text": f"Synthesis updated for: {args['subclaim']}"}]}


@tool(
    "add_conflict",
    "Record a detected conflict between two facts.",
    {"fact_a_id": str, "fact_b_id": str, "description": str, "severity": float}
)
async def add_conflict(args: dict) -> dict:
    manager = StateManager(Path(args.get("_workspace", ".")))
    state = manager.load()
    conflict = Conflict(
        id=f"conflict_{len(state.conflicts)}",
        fact_a_id=args["fact_a_id"],
        fact_b_id=args["fact_b_id"],
        description=args["description"],
        severity=args["severity"]
    )
    state.conflicts.append(conflict)
    manager.save(state)
    return {"content": [{"type": "text", "text": f"Conflict recorded: {conflict.id}"}]}


@tool(
    "get_paper_text",
    "Retrieve the full abstract/text for a specific paper by PMID.",
    {"pmid": str}
)
async def get_paper_text(args: dict) -> dict:
    manager = StateManager(Path(args.get("_workspace", ".")))
    state = manager.load()
    paper = state.papers.get(args["pmid"])
    if not paper:
        return {"content": [{"type": "text", "text": f"Paper {args['pmid']} not found."}]}
    
    return {"content": [{"type": "text", "text":
        f"PMID: {paper.pmid}\nTitle: {paper.title}\n"
        f"Authors: {', '.join(paper.authors[:5])}\n\n{paper.abstract}"
    }]}


@tool(
    "get_facts_for_subclaim",
    "Retrieve all extracted facts relevant to a specific subclaim.",
    {"subclaim": str}
)
async def get_facts_for_subclaim(args: dict) -> dict:
    manager = StateManager(Path(args.get("_workspace", ".")))
    state = manager.load()
    relevant = [f for f in state.facts if args["subclaim"] in f.relevant_subclaims]
    
    if not relevant:
        return {"content": [{"type": "text", "text": "No facts found for this subclaim."}]}
    
    lines = []
    for f in relevant:
        lines.append(f"[{f.stance}] {f.text} (from {f.source_pmid}, conf: {f.confidence:.2f})")
    
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
```

### 3.3 Sufficiency Classifier Tool

This is the feedback instrument — the equivalent of running `pytest` in code programming.

```python
# src/evidence_programming/tools/sufficiency.py

from claude_agent_sdk import tool
from pathlib import Path
import subprocess
import json
from ..state.manager import StateManager

@tool(
    "check_sufficiency",
    "Run the sufficiency classifier on the current evidence state. "
    "Returns: sufficiency label, confidence score, and identified gaps. "
    "This is cheap (no LLM call) — call it frequently to guide your next action. "
    "Think of this as running your test suite.",
    {}
)
async def check_sufficiency(args: dict) -> dict:
    workspace = Path(args.get("_workspace", "."))
    manager = StateManager(workspace)
    state = manager.load()
    
    # Run classifier as subprocess (keeps it decoupled)
    result = subprocess.run(
        ["python", "-m", "evidence_programming.classifier.inference",
         str(workspace / "evidence_state.json")],
        capture_output=True, text=True, timeout=10
    )
    
    if result.returncode != 0:
        return {"content": [{"type": "text", "text":
            f"Classifier error: {result.stderr}"}]}
    
    output = json.loads(result.stdout)
    sufficiency = SufficiencyResult(**output)
    
    # Persist to state history
    state.sufficiency_history.append(sufficiency)
    state.iteration += 1  # 1-indexed: first check → iteration 1, max allowed → iteration 8
    manager.save(state)
    manager.append_trace("check_sufficiency", output)
    
    # Format feedback for the agent
    lines = [
        f"═══ SUFFICIENCY CHECK (iteration {state.iteration}) ═══",
        f"Label: {sufficiency.label}",
        f"Confidence: {sufficiency.confidence:.3f}",
        f"Threshold: 0.80",
        f"Status: {'✓ SUFFICIENT' if sufficiency.confidence >= 0.80 else '✗ INSUFFICIENT'}",
    ]
    if sufficiency.gaps:
        lines.append(f"\nIdentified gaps ({len(sufficiency.gaps)}):")
        for gap in sufficiency.gaps:
            lines.append(f"  [{gap.priority:.1f}] {gap.gap_type}: {gap.description}")
            lines.append(f"         Subclaim: {gap.subclaim}")
    else:
        lines.append("\nNo specific gaps identified.")
    
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
```

### 3.4 Compression Tool

```python
# src/evidence_programming/tools/compression.py

from claude_agent_sdk import tool
from pathlib import Path
import subprocess
import json
from ..state.manager import StateManager

@tool(
    "compress_evidence",
    "Compress the evidence state to fit within token budget while preserving "
    "sufficiency. Tries: (1) deduplication, (2) synthesis refresh, (3) aggressive "
    "compression. Rejects any compression that changes the sufficiency score by "
    "more than epsilon=0.05. Use when token_estimate exceeds budget.",
    {"target_tokens": int}
)
async def compress_evidence(args: dict) -> dict:
    workspace = Path(args.get("_workspace", "."))
    target = args["target_tokens"]
    
    # Run compression as subprocess (invokes compression/compress.py module,
    # which internally loads the classifier and checks the sufficiency invariant)
    result = subprocess.run(
        ["python", "-m", "evidence_programming.compression.compress",
         str(workspace / "evidence_state.json"),
         "--target-tokens", str(target),
         "--epsilon", "0.05"],
        capture_output=True, text=True, timeout=60
    )
    
    if result.returncode != 0:
        return {"content": [{"type": "text", "text":
            f"Compression failed: {result.stderr}"}]}
    
    output = json.loads(result.stdout)
    return {"content": [{"type": "text", "text":
        f"Compression complete. Level used: {output['level']}. "
        f"Tokens: {output['before']} → {output['after']}. "
        f"Sufficiency preserved: {output['invariant_held']}"
    }]}
```

---

## 4. Subagents (Parallel Evidence Processing)

The SDK's subagent architecture maps to evidence programming's need for parallel processing. Each subagent has isolated context, so it can process a full paper without polluting the orchestrator's context window.

### 4.1 Programmatic Subagent Definitions

```python
# Used in orchestrator.py when constructing ClaudeAgentOptions

from claude_agent_sdk import AgentDefinition

EVIDENCE_SUBAGENTS = {
    "fact-extractor": AgentDefinition(
        description=(
            "Extracts atomic, verifiable facts from a scientific paper. "
            "Each fact gets a stance label (SUPPORT/REFUTE/NEUTRAL) relative "
            "to the claim being verified, plus relevance mapping to subclaims. "
            "Use this agent when new papers have been retrieved."
        ),
        prompt="""You are a scientific fact extraction specialist.

Given a paper and a claim with subclaims, extract every atomic fact relevant to the claim.

For each fact, provide:
- text: the factual statement (one sentence, self-contained)
- stance: SUPPORT if it supports the claim, REFUTE if it contradicts, NEUTRAL if relevant but neither
- source_pmid: the paper's PMID
- relevant_subclaims: which subclaims this fact speaks to
- confidence: 0-1, how clearly the paper states this fact

Be precise. Do not infer beyond what the paper states. 
If a paper doesn't address a subclaim, don't manufacture facts about it.

Output your facts as a JSON array.""",
        tools=["Read", "Bash"]
    ),
    
    "synthesizer": AgentDefinition(
        description=(
            "Synthesizes evidence for a specific subclaim from extracted facts. "
            "Produces a concise evidence summary weighing support vs refutation. "
            "Use when facts have been extracted and a subclaim needs synthesis."
        ),
        prompt="""You are an evidence synthesis specialist.

Given a subclaim and the facts relevant to it, produce a synthesis that:
1. States the weight of evidence (mostly supporting, mostly refuting, mixed, insufficient)
2. Summarizes the key supporting facts with citations
3. Summarizes any contradicting facts with citations
4. Notes the quality and diversity of sources
5. Identifies what additional evidence would be needed

Keep the synthesis under 200 words. Be precise about what the evidence does and does not show.

Output as plain text.""",
        tools=["Read"]
    ),
    
    "conflict-detector": AgentDefinition(
        description=(
            "Detects contradictions and inconsistencies among extracted facts. "
            "Use after fact extraction to identify conflicts that need resolution."
        ),
        prompt="""You are a scientific conflict detection specialist.

Given a list of extracted facts, identify pairs that contradict each other.
For each conflict:
- Identify the two conflicting facts by ID
- Describe the nature of the contradiction
- Rate severity 0-1 (0 = minor methodological difference, 1 = direct contradiction)
- Suggest what additional evidence might resolve the conflict

Output as JSON array of conflict objects.""",
        tools=["Read"]
    ),
    
    "gap-query-formulator": AgentDefinition(
        description=(
            "Formulates targeted retrieval queries from gap predictions. "
            "The MLP sufficiency classifier identifies coarse gap types (free, no LLM); "
            "this subagent translates those gap types into specific PubMed queries (1 LLM call). "
            "Use after check_sufficiency reports gaps with confidence < threshold."
        ),
        prompt="""You are an evidence retrieval query specialist.

You receive gap predictions from the sufficiency classifier. Each gap has:
- A subclaim that needs more evidence
- A gap type (one of 8 categories)
- A priority score

Your job is to formulate targeted PubMed search queries that will close these gaps.
Do NOT re-analyze the evidence state for gaps — the classifier has already done that.

Gap types and what they mean for query formulation:
- missing_subclaim_evidence: search directly for the subclaim topic
- contradictory_evidence: search for meta-analyses or reviews that resolve the conflict
- low_source_diversity: search with different terminology or in adjacent fields
- weak_stance_evidence: search for studies with stronger methodology (RCTs, large cohorts)
- missing_mechanism: search for mechanistic or pathway studies
- missing_quantitative: search for dose-response, effect size, or quantitative studies
- missing_temporal: search for longitudinal or time-course studies
- missing_population: search for studies in the specific population mentioned in the claim

For each gap, provide:
- The original gap type and subclaim
- 1-2 targeted PubMed search queries (short, specific, 3-8 words)
- Brief rationale for why this query should close the gap

Output as JSON array.""",
        tools=["Read"]
    ),
}
```

### 4.2 Filesystem-Based Subagent Definitions (Alternative)

For simpler deployment, subagents can also be defined as markdown files:

```markdown
# .claude/agents/fact-extractor.md
---
name: fact-extractor
description: Extracts atomic, verifiable facts from scientific papers with stance labels
tools: Read, Bash
---

You are a scientific fact extraction specialist.
[... same prompt as above ...]
```

---

## 5. The Orchestrator (Evidence Programming Loop)

The orchestrator is the main agent that runs the evidence programming loop. It uses the SDK's `query()` function with structured outputs to produce a final `VerificationVerdict`.

### 5.1 Orchestrator Implementation

```python
# src/evidence_programming/orchestrator.py

import asyncio
import json
from pathlib import Path
from claude_agent_sdk import (
    query, ClaudeAgentOptions, ClaudeSDKClient, AgentDefinition,
    create_sdk_mcp_server, AssistantMessage, ResultMessage, TextBlock
)

from .tools.retrieval import search_pubmed, search_semantic_scholar
from .tools.evidence_state import (
    get_evidence_summary, add_facts, update_synthesis,
    add_conflict, get_paper_text, get_facts_for_subclaim
)
from .tools.sufficiency import check_sufficiency
from .tools.compression import compress_evidence
from .state.schema import VerificationVerdict
from .hooks import EVIDENCE_HOOKS

# Collect all custom tools into an MCP server
evidence_tools_server = create_sdk_mcp_server(
    name="evidence-tools",
    version="1.0.0",
    tools=[
        search_pubmed,
        search_semantic_scholar,
        get_evidence_summary,
        add_facts,
        update_synthesis,
        add_conflict,
        get_paper_text,
        get_facts_for_subclaim,
        check_sufficiency,
        compress_evidence,
    ]
)

SYSTEM_PROMPT = """You are an evidence programming agent for scientific claim verification.

You program on evidence the way a coding agent programs on code. Your tools let you
retrieve, extract, synthesize, and evaluate scientific evidence. The sufficiency
classifier (check_sufficiency) is your test suite — call it frequently to know whether
your evidence gathering is working.

## Your workflow:

1. DECOMPOSE the claim into verifiable subclaims
2. RETRIEVE initial evidence with search_pubmed
3. EXTRACT facts from papers (delegate to fact-extractor subagent)
4. CHECK SUFFICIENCY — this is your primary feedback signal
5. If INSUFFICIENT: read the gap types from the classifier, then delegate to
   gap-query-formulator subagent to translate gaps into targeted retrieval queries
6. SYNTHESIZE evidence per subclaim (delegate to synthesizer subagent)
7. CHECK SUFFICIENCY again
8. COMPRESS if token budget exceeded (compress_evidence tool)
9. Repeat 5-8 until sufficient or max iterations reached

## Key principles:

- Call check_sufficiency after EVERY round of retrieval/extraction. It's free.
- The gaps it reports tell you WHAT TYPE of evidence is missing (coarse MLP prediction).
- Delegate to gap-query-formulator to get specific search queries for each gap type.
- Use subagents for extraction and synthesis — they have isolated context.
- When the classifier reports confidence ≥ 0.80, you have enough evidence.
- If you hit max iterations without reaching sufficiency, verdict is INSUFFICIENT.
- Always produce a structured verdict at the end.

## Token budget: 50,000 tokens. Compress when token_estimate > 40,000.
## Max iterations: 8 sufficiency checks.
"""

# Pydantic -> JSON Schema for structured output
VERDICT_SCHEMA = VerificationVerdict.model_json_schema()


async def verify_claim(
    claim: str,
    workspace: Path,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
    model: str = "claude-sonnet-4-5-20250929"
) -> VerificationVerdict:
    """
    Run the evidence programming loop for a single claim.
    Returns a structured VerificationVerdict.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        cwd=str(workspace),
        allowed_tools=[
            "Read", "Write", "Bash", "Glob", "Task",
            # All custom MCP tools
            "mcp__evidence-tools__search_pubmed",
            "mcp__evidence-tools__search_semantic_scholar",
            "mcp__evidence-tools__get_evidence_summary",
            "mcp__evidence-tools__add_facts",
            "mcp__evidence-tools__update_synthesis",
            "mcp__evidence-tools__add_conflict",
            "mcp__evidence-tools__get_paper_text",
            "mcp__evidence-tools__get_facts_for_subclaim",
            "mcp__evidence-tools__check_sufficiency",
            "mcp__evidence-tools__compress_evidence",
        ],
        mcp_servers={"evidence-tools": evidence_tools_server},
        agents=EVIDENCE_SUBAGENTS,
        hooks=EVIDENCE_HOOKS,
        permission_mode="acceptEdits",
        max_turns=50,  # generous; the agent self-limits via sufficiency
        output_format={
            "type": "json_schema",
            "schema": VERDICT_SCHEMA
        }
    )
    
    prompt = (
        f"Verify the following scientific claim using evidence programming.\n\n"
        f"Claim: {claim}\n\n"
        f"The evidence state has been initialized at {workspace}/evidence_state.json.\n"
        f"Begin by decomposing the claim into subclaims, then follow the "
        f"evidence programming workflow. Call check_sufficiency after each "
        f"round of evidence gathering. Stop when confidence ≥ {sufficiency_threshold} "
        f"or after {max_iterations} iterations."
    )
    
    verdict = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            if hasattr(message, 'structured_output') and message.structured_output:
                verdict = VerificationVerdict(**message.structured_output)
            elif hasattr(message, 'result'):
                # Fallback: parse from result text
                try:
                    verdict = VerificationVerdict.model_validate_json(message.result)
                except Exception:
                    verdict = VerificationVerdict(
                        verdict="INSUFFICIENT",
                        confidence=0.0,
                        reasoning=str(message.result),
                        key_evidence=[],
                        gaps_remaining=["Failed to produce structured verdict"]
                    )
    
    if verdict is None:
        verdict = VerificationVerdict(
            verdict="INSUFFICIENT",
            confidence=0.0,
            reasoning="Agent did not produce a verdict",
            key_evidence=[],
            gaps_remaining=["No verdict produced"]
        )
    
    return verdict


async def verify_claim_interactive(claim: str, workspace: Path) -> VerificationVerdict:
    """
    Interactive version using ClaudeSDKClient for multi-turn control.
    Useful for debugging and development.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        cwd=str(workspace),
        allowed_tools=[
            "Read", "Write", "Bash", "Glob", "Task",
            "mcp__evidence-tools__*",  # all evidence tools
        ],
        mcp_servers={"evidence-tools": evidence_tools_server},
        agents=EVIDENCE_SUBAGENTS,
        permission_mode="acceptEdits",
    )
    
    async with ClaudeSDKClient(options=options) as client:
        # Initial prompt
        await client.query(
            f"Initialize evidence state for claim: {claim}\n"
            f"Decompose into subclaims and begin evidence programming."
        )
        
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text)
        
        # The agent runs autonomously from here — the SDK handles
        # tool calls, subagent invocations, and context management.
        # We could add manual checkpoints here if needed.
    
    # Load final state and construct verdict
    state = StateManager(workspace).load()
    if state.sufficiency_history:
        latest = state.sufficiency_history[-1]
        return VerificationVerdict(
            verdict=latest.label.replace("SUFFICIENT_", ""),
            confidence=latest.confidence,
            reasoning=json.dumps(state.synthesis),
            key_evidence=[f.text for f in state.facts[:10]],
            gaps_remaining=[g.description for g in latest.gaps]
        )
    
    return VerificationVerdict(
        verdict="INSUFFICIENT", confidence=0.0,
        reasoning="No sufficiency checks completed",
        key_evidence=[], gaps_remaining=[]
    )
```

### 5.2 Hooks for Monitoring and Cost Control

```python
# src/evidence_programming/hooks.py

from claude_agent_sdk import HookMatcher

async def log_tool_use(input_data, tool_use_id, context):
    """PostToolUse hook: log every tool invocation for the audit trail."""
    tool_name = input_data.get("tool_name", "unknown")
    print(f"  [TRACE] Tool used: {tool_name}")
    return {}

async def enforce_iteration_limit(input_data, tool_use_id, context):
    """PreToolUse hook on check_sufficiency: enforce max iterations."""
    # Read current iteration from state
    import json
    from pathlib import Path
    state_path = Path(".") / "evidence_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("iteration", 0) >= 8:  # After 8 checks, deny further calls
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": 
                        "Max iterations (8) reached. Produce final verdict now.",
                }
            }
    return {}

async def block_dangerous_bash(input_data, tool_use_id, context):
    """PreToolUse hook: prevent destructive bash commands."""
    if input_data.get("tool_name") != "Bash":
        return {}
    command = input_data.get("tool_input", {}).get("command", "")
    dangerous = ["rm -rf", "sudo", "curl | bash", "wget"]
    for pattern in dangerous:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked dangerous command: {pattern}",
                }
            }
    return {}

# Hook configuration for the orchestrator
EVIDENCE_HOOKS = {
    "PostToolUse": [
        HookMatcher(hooks=[log_tool_use]),
    ],
    "PreToolUse": [
        HookMatcher(
            matcher="mcp__evidence-tools__check_sufficiency",
            hooks=[enforce_iteration_limit]
        ),
        HookMatcher(matcher="Bash", hooks=[block_dangerous_bash]),
    ],
}
```

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

## 8. Implementation Stages (Revised for SDK)

### Stage 1: Foundation (Week 1-2)

**Goal:** Working state management + retrieval tools that the agent can call.

Tasks:
- Set up project structure with `pyproject.toml` (dependencies: `claude-agent-sdk`, `httpx`, `pydantic`, `torch`)
- Implement `state/schema.py` — all Pydantic models
- Implement `state/manager.py` — JSON persistence
- Implement `tools/retrieval.py` — PubMed search as MCP tool
- Implement `tools/evidence_state.py` — state read/write MCP tools
- Write `tests/test_state.py` and `tests/test_tools.py`
- **Verification:** Agent can search PubMed and add papers to state via tool calls

```bash
# Smoke test
python -c "
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, create_sdk_mcp_server
from evidence_programming.tools.retrieval import search_pubmed
from evidence_programming.tools.evidence_state import get_evidence_summary

server = create_sdk_mcp_server('ev', tools=[search_pubmed, get_evidence_summary])
opts = ClaudeAgentOptions(
    mcp_servers={'ev': server},
    allowed_tools=['mcp__ev__*'],
    permission_mode='acceptEdits',
    max_turns=5,
)

async def main():
    async for msg in query(
        prompt='Search PubMed for metformin diabetes and show me the evidence summary',
        options=opts
    ):
        print(msg)

asyncio.run(main())
"
```

### Stage 2: Sufficiency Classifier (Week 3-4)

**Goal:** Trained classifier that can be called as a tool.

Tasks:
- Implement `scripts/generate_training_data.py` — self-consistency sampling on SciFact
- Run sampling: ~1,400 claims × 6 k-values × 10 samples = ~84K API calls (~$50-100 on Sonnet)
- Implement `classifier/features.py`, `classifier/model.py`
- Train MLP classifier with multi-task loss (sufficiency + confidence + gaps)
- Implement `classifier/inference.py` — CLI entry point
- Implement `tools/sufficiency.py` — MCP tool wrapping the CLI
- Calibration analysis: plot predicted confidence vs actual accuracy
- **Verification:** `check_sufficiency` tool returns meaningful feedback on real evidence states

### Stage 3: Subagents + Evidence Programming Loop (Week 5-6)

**Goal:** Full orchestrator running the evidence programming loop with subagents.

Tasks:
- Define subagents: `fact-extractor`, `synthesizer`, `conflict-detector`, `gap-query-formulator`
- Implement `orchestrator.py` with system prompt and all tool registrations
- Implement `hooks.py` — iteration limits, logging, safety guards
- Tune system prompt: ensure agent calls `check_sufficiency` after each retrieval round
- Tune `max_turns`, token budget, sufficiency threshold
- Test on 10 SciFact claims end-to-end
- **Verification:** Agent produces structured verdicts, respects iteration limits, uses gap feedback to direct retrieval

### Stage 4: Compression + Budget Management (Week 7-8)

**Goal:** Agent can handle long evidence chains without exceeding context.

Tasks:
- Implement three-level compression in `tools/compression.py`
- Implement sufficiency invariant check (run classifier before/after compression)
- Test compression on claims that require 15+ papers
- ε sensitivity sweep: {0.01, 0.02, 0.05, 0.10}
- Compare against truncation and RECOMP baselines
- **Verification:** Agent can verify claims requiring extensive evidence without context overflow

### Stage 5: Full Evaluation (Week 9-11)

**Goal:** Complete experimental results on all datasets.

Tasks:
- Implement `evaluation/runner.py` and `evaluation/metrics.py`
- Run on SciFact (1,400 claims), SciFact-Open (1,400 claims, 500K corpus)
- Run on SciClaimHunt and FEVER subsets
- Implement baselines: Fixed-k RAG, Self-RAG-style, SAFE-style
- Run all 6 ablations (A1-A6 from updated plan)
- Generate cost-accuracy Pareto curves
- Adversarial experiments (Section 4 of updated plan)
- **Verification:** Complete results tables for the paper

### Stage 6: Writing + Polish (Week 12-13)

**Goal:** Paper draft.

Tasks:
- Write motivating example showing the agent programming on evidence
- Draft all sections per updated plan structure
- Generate figures: Pareto curves, sufficiency trajectories, calibration plots
- Qualitative examples of gap-directed vs undirected retrieval
- Internal review and revision

---

## 9. Dependencies and Environment

```toml
# pyproject.toml
[project]
name = "evidence-programming"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "claude-agent-sdk>=0.1.20",
    "anthropic>=0.40.0",
    "pydantic>=2.0",
    "httpx>=0.27",
    "torch>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",  # for calibration metrics
    "lxml",               # for PubMed XML parsing
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

Environment variables:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export EVIDENCE_PROGRAMMING_DATA=./data
export EVIDENCE_PROGRAMMING_MODEL=claude-sonnet-4-5-20250929
```

---

## 10. Key Design Decisions and Rationale

### Why MCP tools instead of raw Bash scripts?

MCP tools give the agent typed interfaces with clear descriptions. The agent sees `search_pubmed(query, max_results)` rather than having to construct bash commands. This makes the programming analogy concrete — the agent has a well-defined instruction set.

### Why subagents for extraction/synthesis?

Paper processing produces large amounts of intermediate text (full abstracts, extracted fact lists). Running this in the main agent's context would quickly fill the context window. Subagents process in isolation and return only the structured results, keeping the orchestrator's context focused on the programming loop — the feedback/decide/act cycle.

### Why persist state as JSON files?

The SDK's built-in tools (Read, Write, Bash) can interact with files natively. The agent can `Read evidence_state.json` to inspect state directly, or use structured MCP tools for specific operations. This dual access means the agent has both programmatic tools and raw inspection capability — exactly like a coding agent has both API calls and the ability to read source files.

### Why run the classifier via Bash subprocess?

Decoupling. The classifier is a separate Python process with its own PyTorch model. The MCP tool invokes it, parses the JSON output, and returns structured feedback. This mirrors how a coding agent runs `python -m pytest` as a subprocess and interprets the output. The classifier doesn't share memory or state with the SDK process.

### Why structured outputs for the final verdict?

The SDK's structured output feature guarantees a valid `VerificationVerdict` JSON matching our Pydantic schema. This means the evaluation runner can parse verdicts reliably without worrying about free-text parsing. Every claim produces a typed result with verdict, confidence, reasoning, evidence, and remaining gaps.

### Why hooks for iteration limits?

The `PreToolUse` hook on `check_sufficiency` enforces the max iteration limit deterministically, independent of whether the agent's system prompt compliance is perfect. This is a safety rail — the agent can't accidentally run forever even if it ignores its instructions. The hook denies the tool call and forces the agent to produce a final verdict.
