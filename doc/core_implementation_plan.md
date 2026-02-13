# Core Implementation Plan: Evidence Programming on Claude Agent SDK

## Status: Implemented

All core components have been built and tested. This document describes the
architecture, dependency chain, and what has been deferred to extensions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Orchestrator Agent                      │
│  (orchestrator.py → Claude Agent SDK query() loop)      │
│                                                         │
│  System prompt: evidence programming workflow            │
│  Tools: 10 MCP tools + Read + Task                      │
│  Structured output: VerificationVerdict via emit_verdict │
│                                                         │
│  ┌──────────────┐  ┌────────────┐  ┌────────────────┐  │
│  │  Retrieval   │  │ Subagents  │  │  Sufficiency   │  │
│  │  Tools (MCP) │  │ (Task)     │  │  Classifier    │  │
│  │  search_     │  │ fact-      │  │  (heuristic)   │  │
│  │  pubmed      │  │ extractor  │  │  check_        │  │
│  │  search_     │  │ synthesizer│  │  sufficiency   │  │
│  │  for_gap     │  │ conflict-  │  │                │  │
│  │              │  │ detector   │  │                │  │
│  │              │  │ gap-query- │  │                │  │
│  │              │  │ formulator │  │                │  │
│  └──────┬───────┘  └──────┬─────┘  └───────┬────────┘  │
│         │                │                 │            │
│         └────────────┬───┘─────────────────┘            │
│                      ▼                                  │
│           ┌─────────────────┐                           │
│           │  Evidence State  │                           │
│           │  (JSON on disk)  │                           │
│           │  + TraceLog      │                           │
│           └─────────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

---

## Dependency Chain

```
Phase 1 (Foundation):
  data_models.py     → Pydantic schemas (Fact, PaperRecord, Gap, etc.)
  evidence_state.py  → Pydantic state with JSON persistence + TraceLog
  classifier.py      → Returns SufficiencyResult with Gap objects
  compressor.py      → L1 deduplication (lossless)
  __init__.py        → Export new types via lazy imports
  pyproject.toml     → Add pydantic, mcp dependencies

Phase 2 (MCP + Adapters):
  mcp_tools.py       → FastMCP server with 10 tools
  adapters.py        → SIGNOR + SciFact dataset adapters

Phase 3 (Agent Integration):
  .claude/agents/    → 4 subagent markdown definitions
  orchestrator.py    → Claude Agent SDK query() loop
  run_verification.py → CLI entry point
```

---

## Component Summary

| Component | File | Minimal Form | Deferred to Extensions |
|-----------|------|-------------|----------------------|
| Data models | `data_models.py` | Pydantic v2 BaseModels: Stance, GapType, PaperRecord, Fact, Conflict, Gap, SufficiencyResult, VerificationVerdict | Training data schemas |
| Evidence state | `evidence_state.py` | Pydantic BaseModel with JSON save/load, TraceLog | Thread-safe mutations, tiktoken-based token counting |
| Classifier | `classifier.py` | Heuristic rule-based returning SufficiencyResult with Gap objects | Trained 16-feature MLP, self-consistency training data generation, calibration |
| Compressor | `compressor.py` | L1 deduplication only | L2 synthesis refresh, L3 aggressive synthesis, sufficiency invariant checking |
| MCP tools | `mcp_tools.py` | 10 tools via FastMCP: search_pubmed, search_for_gap, get_evidence_summary, add_facts, get_paper_text, update_synthesis, add_conflict, check_sufficiency, compress_evidence, emit_verdict | Semantic Scholar search, async tool execution |
| Adapters | `adapters.py` | SIGNOR + SciFact adapters with common Claim dataclass | FEVER, SciClaimHunt adapters |
| Subagents | `.claude/agents/*.md` | 4 markdown definitions: fact-extractor, synthesizer, conflict-detector, gap-query-formulator | Programmatic AgentDefinition objects |
| Orchestrator | `orchestrator.py` | async verify_claim() + verify_claim_batch() via claude_agent_sdk.query() | Interactive mode, structured output schema validation |
| CLI | `run_verification.py` | --claim / --dataset {signor,scifact} | Results comparison dashboard |

---

## Files

### Core modules (7 files)
| File | Purpose |
|------|---------|
| `src/pkevolve/verification/data_models.py` | Pydantic v2 schemas (Stance, GapType, Fact, Gap, etc.) |
| `src/pkevolve/verification/evidence_state.py` | Pydantic BaseModel state with JSON persistence + TraceLog |
| `src/pkevolve/verification/classifier.py` | Heuristic sufficiency classifier returning SufficiencyResult |
| `src/pkevolve/verification/compressor.py` | L1 deduplication compressor |
| `src/pkevolve/verification/mcp_tools.py` | FastMCP server with 10 evidence programming tools |
| `src/pkevolve/verification/adapters.py` | Dataset adapters (SIGNOR, SciFact) |
| `src/pkevolve/verification/orchestrator.py` | Claude Agent SDK integration |

### Supporting files (5 files)
| File | Purpose |
|------|---------|
| `src/pkevolve/verification/__init__.py` | Lazy imports for public API |
| `.claude/agents/fact-extractor.md` | Subagent: extract atomic facts from papers |
| `.claude/agents/synthesizer.md` | Subagent: synthesize evidence per subclaim |
| `.claude/agents/conflict-detector.md` | Subagent: find contradictions among facts |
| `.claude/agents/gap-query-formulator.md` | Subagent: translate gap types into PubMed queries |

### Entry point
| File | Purpose |
|------|---------|
| `scripts/verification/run_verification.py` | CLI for single and batch verification |

---

## MCP Tool Reference

All tools take `workspace: str` as parameter. State is loaded from
`{workspace}/evidence_state.json`, mutated, and saved back.

| Tool | Category | Description |
|------|----------|-------------|
| `search_pubmed` | Retrieval | Search PubMed, add papers to state |
| `search_for_gap` | Retrieval | Gap-targeted retrieval using gap_type context |
| `get_evidence_summary` | State | Paper/fact counts, coverage, latest sufficiency |
| `add_facts` | State | Persist extracted facts (JSON array input) |
| `get_paper_text` | State | Retrieve paper abstract by PMID |
| `update_synthesis` | State | Store synthesis text for a subclaim |
| `add_conflict` | State | Record conflict between two facts |
| `check_sufficiency` | Feedback | Run heuristic classifier, return gaps (free, no LLM) |
| `compress_evidence` | Compression | L1 dedup, report before/after tokens |
| `emit_verdict` | Output | Write VerificationVerdict to verdict.json |

---

## Usage

```bash
# Single claim
python scripts/verification/run_verification.py \
    --claim "Does GNAS directly activate ADCY1?" \
    --model claude-sonnet-4-5-20250929

# Batch SIGNOR edges
python scripts/verification/run_verification.py \
    --dataset signor \
    --label true_positive \
    --max-edges 5

# Batch SciFact claims
python scripts/verification/run_verification.py \
    --dataset scifact \
    --scifact-path data/scifact/claims.json \
    --max-edges 10
```

---

## Test Results

All smoke tests pass in the pkevolve conda environment (Python 3.12):

- Schema round-trip (Pydantic save/load): PASS
- Classifier returns SufficiencyResult with Gap objects: PASS
- Compressor: PASS
- Clone independence: PASS
- MCP tool registration (10 tools): PASS
- MCP tools functional test (add_facts, check_sufficiency, emit_verdict): PASS
- PubMed retrieval (real query): PASS
- SIGNOR adapter loads data: PASS
- Iteration limit enforcement: PASS
- MCP server starts as standalone process: PASS
- Orchestrator import: SKIPPED (claude_agent_sdk only available in Claude Code runtime)

---

## What Is Deferred (Not in Core)

- **Trained MLP classifier**: Self-consistency data generation, training loop, calibration
- **L2/L3 compression**: LLM-based synthesis refresh, aggressive compression
- **Sufficiency invariant**: |phi(S') - phi(S)| <= epsilon checking
- **Evaluation runner**: Batch evaluation with F1, ECE, cost tracking metrics
- **Baselines**: Fixed-k RAG, Self-RAG, SAFE implementations
- **Semantic Scholar**: Second retrieval source
- **CLAUDE.md**: Project-level instruction file for the orchestrator
