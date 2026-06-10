# ProClaim Architecture: Full Workflow Reference

> **Last updated:** 2026-06-08  
> **Based on:** `src/proclaim/verification/` source analysis

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Data Flow Diagram](#2-high-level-data-flow-diagram)
3. [LLMs in the System](#3-llms-in-the-system)
4. [The Main Loop: Orchestrator](#4-the-main-loop-orchestrator)
5. [The Workbook: Agent's Planning Surface](#5-the-workbook-agents-planning-surface)
6. [Tool Calling: What the Planner Can Do](#6-tool-calling-what-the-planner-can-do)
7. [Evidence API: The Callable Library](#7-evidence-api-the-callable-library)
8. [Subagent Layer: LLM-Powered Operations](#8-subagent-layer-llm-powered-operations)
9. [Reflection Module](#9-reflection-module)
10. [Sufficiency Gate](#10-sufficiency-gate)
11. [Curation Queue](#11-curation-queue)
12. [Prompt Templates & Conflicts](#12-prompt-templates--conflicts)
13. [Durable State & Disk Artifacts](#13-durable-state--disk-artifacts)
14. [Guardrails (GR1–GR9)](#14-guardrails-gr1gr9)
15. [Forced Verdict Path](#15-forced-verdict-path)
16. [Configuration & Label System](#16-configuration--label-system)
17. [Known Tensions & Design Trade-offs](#17-known-tensions--design-trade-offs)

---

## 1. System Overview

ProClaim is a **sufficiency-aware agentic claim verifier**. Given a scientific claim (e.g. "SRC directly inhibits CTTN"), it:

1. Searches PubMed and Semantic Scholar for relevant papers.
2. Extracts stance-labeled facts from retrieved papers via an LLM subagent.
3. Populates NLP + metadata features for a trained MLP classifier.
4. Runs a **sufficiency gate** that decides if the current evidence is enough to emit a verdict.
5. If insufficient, identifies gaps and searches for more targeted evidence (iterative loop).
6. Emits a configured final verdict label (defaults: `SUPPORT`, `REFUTE`, or `UNCERTAIN`).

There are **two execution modes**, sharing the same `evidence_api.py` and `EvidenceState`:

| Mode (`config.mode`) | File | LLM Client | Kernel |
|----------------------|------|-----------|--------|
| **`direct`** | `evidence_programming_direct.py` | LiteLLM | No Jupyter; subprocess per `python`/`bash` tool call |
| **`sdk`** | `evidence_programming.py` | Claude Agent SDK | Persistent Jupyter kernel |

This document focuses on **Direct mode**. The evaluation runners select Direct
or SDK mode from the experiment configuration; Direct mode is not hardcoded as
the evaluation mode.

---

## 2. High-Level Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OUTER LOOP (orchestrator)                        │
│                   evidence_programming_direct.py                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Per-turn sequence                                          │        │
│  │                                                             │        │
│  │  1. run_auto_curation(workspace)   ← refresh curation queue │        │
│  │  2. detect_stall_signals(workspace)← check ledger patterns  │        │
│  │  3. run_reflection(...)            ← REFLECT LLM call       │        │
│  │     └─ writes reflection.json                               │        │
│  │  4. write_workbook(EvidenceState)  ← regenerate workbook.md │        │
│  │  5. litellm.completion(...)        ← PLANNER LLM call       │        │
│  │     └─ [system_prompt + workbook + instruction]             │        │
│  │  6. dispatch_tool(tool_call)       ← execute first call     │        │
│  │     └─ python / bash / read_file / web_search               │        │
│  │  7. record_action(ledger)          ← append to JSONL        │        │
│  │  8. Check verdict.json → stop if present                    │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                         │
│  Planner loop until: verdict.json exists OR max_calls                  │
│                      (= max_iterations × max_turns planner calls)      │
│  If no verdict: _force_verdict()  ← SUBAGENT LLM makes the call        │
└─────────────────────────────────────────────────────────────────────────┘

                     ┌──────────────────────────────────┐
                     │        EvidenceState (disk)       │
                     │  evidence_state.json              │
                     │  ├─ claim, subclaims              │
                     │  ├─ papers{} (PaperRecord)        │
                     │  ├─ facts[] (Fact)                │
                     │  ├─ conflicts[]                   │
                     │  ├─ extraction_context[]          │
                     │  ├─ extracted_pmids[]             │
                     │  ├─ sufficiency_history[]         │
                     │  ├─ sufficiency_candidate_pmids[] │
                     │  ├─ stance_filter_*               │
                     │  ├─ coverage{}                    │
                     │  ├─ synthesis{}                   │
                     │  ├─ papers_per_iteration[]        │
                     │  ├─ token_estimate (int)          │
                     │  └─ iteration (int)               │
                     └──────────────────────────────────┘

       ┌──────────────────────┐    ┌──────────────────────────────┐
       │  action_ledger.jsonl │    │  workbook.md                 │
       │  (append-only JSONL) │    │  §1 Claim Frame (stable)     │
       │  ActionRecord per    │    │  §2 Guardrails (stable)      │
       │  tool call:          │    │  ─── stable│volatile ───     │
       │  • target            │    │  §3 Header                   │
       │  • observation       │    │  §4 State Snapshot           │
       │  • delta             │    │  §4a Evidence Digest         │
       │  • diagnosis         │    │  §5 Action Loop Record       │
       │  • artifact_handle   │    │  §6 Recuration Queue         │
       │  • raw_size_chars    │    │  §7 Verdict Readiness        │
       └──────────────────────┘    │  §8 Artifact Index           │
                                   │  §9 Next-Turn Guidance       │
                                   └──────────────────────────────┘
```

### Detailed Interaction Flow Inside One Turn

```
Orchestrator
    │
    ├─1─ run_auto_curation(workspace)
    │       adds latest gaps/conflicts/zero-fact papers to curation_queue.json
    │       without clearing older entries
    │
    ├─2─ detect_stall_signals(workspace)
    │       reads action_ledger.jsonl + evidence_state.json
    │       returns: StallSignals {search_drought, extraction_drought,
    │                              sufficiency_stall, action_repetition}
    │
    ├─3─ run_reflection(workspace, volatile_tail, stall_signals, turn)
    │       ┌────────────────────────────────────┐
    │       │         REFLECT LLM                │
    │       │  model: same as planner            │
    │       │  system: REFLECTION_SYSTEM_PROMPT  │
    │       │  user: REFLECTION_USER_PROMPT      │
    │       │         {stall_signals, workbook_volatile}
    │       │  tool: submit_reflection(          │
    │       │    diagnosis, classification,      │
    │       │    proposed_next_family,           │
    │       │    override_invoked)               │
    │       └────────────────────────────────────┘
    │           └─ writes reflection.json
    │           └─ recorded in action_ledger as "reflect" entry
    │
    ├─4─ write_workbook(EvidenceState, ...)
    │       also writes verdict_packet.md (bounded fact-centered view)
    │       returns workbook.md path
    │
    ├─5─ litellm.completion(model=agent_model, messages=[...], tools=[...])
    │       ┌────────────────────────────────────────────┐
    │       │            PLANNER LLM                     │
    │       │  system: DIRECT_SYSTEM_PROMPT (cached)     │
    │       │  user: workbook_text + turn_instruction    │
    │       │         (stable prefix cached,             │
    │       │          volatile tail sent fresh)         │
    │       │  tools: [python, bash, read_file,          │
    │       │          web_search (optional)]            │
    │       │  max_tokens: 16384                         │
    │       └────────────────────────────────────────────┘
    │           └─ returns tool_calls[] (or text response)
    │
    ├─6─ GR6 enforcement: dispatch only tool_calls[0], drop rest
    │
    ├─7─ dispatch_tool(name, args, workspace, log_path, env)
    │       ┌─────────────────────────────────────────────┐
    │       │ Tool: python(code=...)                      │
    │       │   subprocess.run(["python3", "-c", code])   │
    │       │   env: LLM_MODEL=subagent_model,            │
    │       │        SUFFICIENCY_BACKEND=..., etc.        │
    │       │   working dir: workspace                    │
    │       │   The code imports from evidence_api and    │
    │       │   calls functions that may in turn call     │
    │       │   the SUBAGENT LLM (extraction, gaps, etc.) │
    │       └─────────────────────────────────────────────┘
    │           └─ stdout/stderr returned as string
    │           └─ appended to execution_log.py (jupytext)
    │
    ├─8─ record_action(workspace, turn, tool_name, args, raw_output,
    │                   before_snap, after_snap)
    │       builds delta, observation, diagnosis
    │       spills large returned outputs to artifacts/
    │       appends ActionRecord to action_ledger.jsonl
    │
    └─9─ check verdict.json → if exists, stop
```

---

## 3. LLMs in the System

There are **four distinct LLM roles**, each with its own prompt, model, and calling convention:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LLM Role          │ Who calls it  │ Model         │ When               │
├────────────────────┼───────────────┼───────────────┼────────────────────┤
│ 1. PLANNER         │ Orchestrator  │ cfg.llm.model │ Every turn          │
│    (outer agent)   │ via litellm   │ (e.g. Sonnet) │                     │
├────────────────────┼───────────────┼───────────────┼────────────────────┤
│ 2. SUBAGENT        │ Evidence API  │ cfg.subagent_ │ Inside python tool  │
│    (inner worker)  │ via env vars  │ model         │ calls: extraction,  │
│                    │ + LiteLLM     │ (e.g. Haiku)  │ gap ID, queries     │
├────────────────────┼───────────────┼───────────────┼────────────────────┤
│ 3. REFLECT LLM     │ Orchestrator  │ same as       │ Every turn, before  │
│                    │ via litellm   │ planner       │ planner call        │
├────────────────────┼───────────────┼───────────────┼────────────────────┤
│ 4. FORCED VERDICT  │ Orchestrator  │ subagent_     │ Once, at end, only  │
│    LLM             │ via subprocess│ model         │ if no verdict yet   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Planner LLM

- **Model:** `cfg.llm.model` (e.g. `anthropic/claude-sonnet-4-20250514`)
- **Called by:** `litellm.completion()` in `verify_claim_direct()`
- **Input:** `[system_msg, user_msg]` — stateless two-message context each turn
- **Prompt caching:** System prompt + stable workbook prefix cached as `ephemeral` (5 min TTL) when model starts with `anthropic/`; disabled when thinking is active
- **Extended thinking:** Optional, set via `cfg.llm.thinking_budget_tokens`; forces `temperature=1`
- **Output:** `tool_calls[]` (one dispatched per GR6) OR text reasoning
- **Controls:** Action selection, reasoning trace, tool call arguments

### 3.2 Subagent LLM

- **Model:** `cfg.subagent_model` (e.g. `anthropic/claude-haiku-4-5`)
- **Called by:** Evidence API functions inside the `python` subprocess
- **Mechanism:** The subagent model config is injected into the subprocess via environment variables (`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, etc.); inside the subprocess, `setup_workspace()` creates a `llm` callable from these env vars
- **Concurrency:** The LLM callable is invoked from a `ThreadPoolExecutor` during
  parallel extraction. However, `EvidenceState` has no internal lock: workers
  concurrently mutate shared lists and may write the same JSON file. The current
  implementation therefore does not provide a hard thread-safety guarantee.
- **Functions that use the subagent:**
  - `extract_and_add_facts()` → `extract_facts()` (EXTRACT_FACTS prompt)
  - `check_sufficiency()` with `backend=llm` or `backend=haiku` → `identify_gaps()` (IDENTIFY_GAPS prompt)
  - `search_pubmed_llm()` → `generate_search_query()` (QUERY_GENERATION prompt)
  - `search_semantic_scholar_dual()` → `generate_search_query_s2()` (QUERY_GENERATION_S2 prompt)
  - `refine_search_for_failed_papers()` → `refine_search_query()` (REFINE_SEARCH_QUERY prompt)
  - `formulate_gap_queries()` → `formulate_gap_queries()` subagent (FORMULATE_GAP_QUERIES prompt)
  - `synthesize_subclaim()` → (SYNTHESIZE_SUBCLAIM prompt)
  - `detect_conflicts()` → (DETECT_CONFLICTS prompt)

### 3.3 Reflect LLM

- **Model:** Same as planner (`agent_model`) — same reasoning capacity, different role
- **Called by:** Orchestrator in `verify_claim_direct()`, every turn, **before** the planner call
- **Input:** `REFLECTION_SYSTEM_PROMPT` + `REFLECTION_USER_PROMPT` with stall signals + volatile workbook tail
- **Tool:** `submit_reflection` (one call required) with 4 fields: `diagnosis`, `classification`, `proposed_next_family`, `override_invoked`
- **Output:** `ReflectionRecord` stored in `reflection.json`; rendered in workbook Section 9
- **Role:** Meta-judge that reads the current state and recommends the next action family. On routine turns: points to next workflow step. On stalled turns: diagnoses root cause and proposes corrective action.

### 3.4 Forced Verdict LLM

- **Model:** Subagent model (from `sub_env["LLM_MODEL"]`)
- **Called by:** `_force_verdict()` at the end of the main loop if no verdict was emitted
- **Input:** `FORCE_VERDICT_PROMPT` with a bounded, control-history-free
  `verdict_packet` (facts grouped by stance/PMID plus sufficiency caveats)
- **Output:** Structured text: `VERDICT:`, `CONFIDENCE:`, `REASONING:`, `KEY_EVIDENCE:`
- **Role:** Makes the final call from a fact-centered verdict packet. The packet
  also includes the latest sufficiency gaps as explicitly labeled caveats; the
  prompt tells the model not to treat those caveats as evidence.

---

## 4. The Main Loop: Orchestrator

```
verify_claim_direct(cfg)
│
│  max_calls = max_iterations × max_turns   (planner-call kill-switch;
│                                              excludes reflect/subagent calls)
│  planning_budget = max_turns              (per-iteration turn budget)
│  call_count = 0, iteration_turn = 0
│
while call_count < max_calls:
│
│  ── iteration boundary detection ──────────────────────────────────────
│  Load EvidenceState.iteration from disk
│  If iteration changed since last turn → reset iteration_turn to 0
│
│  iter_turns_remaining = planning_budget - iteration_turn
│  is_last_iteration = (state.iteration >= max_iterations - 1)
│
│  ── Phase 3: auto-curation ─────────────────────────────────────────────
│  run_auto_curation(workspace)
│
│  ── Phase 4: per-turn reflection ───────────────────────────────────────
│  if evidence_state.json exists:
│      signals = detect_stall_signals(workspace)
│      volatile_tail = build_workbook_parts(...)[1]
│      reflect_record, reflect_usage = run_reflection(...)
│      record_action(workspace, "reflect", ...)
│
│  ── Context refresh ────────────────────────────────────────────────────
│  current_state = EvidenceState.load(...)
│  workbook_path = write_workbook(current_state, ...)
│  workbook_text = workbook_path.read_text()
│
│  ── Build user message ──────────────────────────────────────────────────
│  if call_count == 0:
│      user_text = "Verify claim... workbook... start with setup code"
│  elif is_last_iteration and iter_turns_remaining <= 2:
│      user_text = "URGENT — call check_sufficiency then emit_verdict NOW"
│  elif iter_turns_remaining <= 2:
│      user_text = "Only N turns left — call check_sufficiency to close iteration"
│  else:
│      user_text = "Continue verification... make ONE tool call"
│
│  ── Planner LLM call ────────────────────────────────────────────────────
│  response = litellm.completion(model, [system_msg, user_msg], tools, ...)
│  call_count += 1; iteration_turn += 1
│
│  ── GR6 enforcement ─────────────────────────────────────────────────────
│  dispatched = tool_calls[:1]   ← only first tool call
│  dropped    = tool_calls[1:]   ← rest dropped + ledger note
│
│  ── Tool dispatch ───────────────────────────────────────────────────────
│  for tc in dispatched:
│      before_snap = snapshot_state(EvidenceState.load(...))
│      result = dispatch_tool(tc.name, tc.args, workspace, log_path, env)
│      after_snap  = snapshot_state(EvidenceState.load(...))
│      record_action(workspace, turn, tc.name, tc.args, result,
│                    before_snap, after_snap)
│
│  ── Stop check ──────────────────────────────────────────────────────────
│  if verdict.json exists → break
│
else:  (max_calls exhausted)
│  logger.info("Reached max calls")
│
── Forced verdict ────────────────────────────────────────────────────────
if not verdict.json:
    _force_verdict(workspace, claim, sub_env)
│
── Finalize ──────────────────────────────────────────────────────────────
save token_usage.json, token_usage_trace.json
generate_notebook(log_path, notebook_path)
print verdict
```

---

## 5. The Workbook: Agent's Planning Surface

The workbook is the **primary context** the planner sees each turn. It is
regenerated from `EvidenceState` before every planner LLM call (stateless
planner context refresh — no conversation accumulation).

```
workbook.md
├── § 1  Claim Frame            [STABLE PREFIX — intended to be invariant]
│        - claim, verdict labels, subclaims
│        - stopping rule (confidence threshold, max iterations)
│        - consensus notes (prefer multiple independent sources)
│        - changes if the planner later mutates subclaims
│
├── § 2  Guardrails GR1–GR9    [STABLE]
│        - read-only rules for the planner
│        - each has: id, source, rule, override condition
│
│   <!-- ---- workbook: stable | volatile ---- -->   ← cache breakpoint
│
├── § 3  Header                 [VOLATILE — changes each turn]
│        - current iteration, turns remaining THIS iteration
│        - status: collecting-evidence / ready-for-verdict / 
│                  forced-verdict-imminent / check-sufficiency-imminent
│
├── § 4  State Snapshot         [VOLATILE]
│        - papers retrieved, full text, extracted, zero-fact count
│        - total facts, by-stance counts, unique sources
│        - sufficiency trend history
│        - open gaps from last sufficiency check
│        - active extraction context note
│
├── § 4a Known Evidence Digest  [VOLATILE]
│        - facts grouped by stance/PMID (capped at 8 per stance)
│        - pointer to verdict_packet.md (broader but still capped:
│          20 facts per stance; 10 non-directional papers)
│
├── § 5  Action Loop Record     [VOLATILE]
│        - last 5 ledger entries (one bullet per tool call)
│        - format: #id target — observation [; diagnosis] [artifact]
│
├── § 6  Recuration Queue       [VOLATILE]
│        - to-do buckets: re_rank, re_extract, filtered_to_revisit,
│                         zero_fact, gaps, contradictions
│
├── § 7  Verdict Readiness      [VOLATILE]
│        - current sufficiency label + confidence
│        - is verdict ready? (yes/no + reason)
│        - blockers, minimum additional evidence needed
│        - forced-verdict risk level
│
├── § 8  Raw Artifact Index     [VOLATILE]
│        - artifact://artifacts/ handles for spilled ledger outputs
│        - read_file does not resolve artifact:// URIs; use the corresponding
│          workspace path, e.g. /workspace/artifacts/<file>
│
└── § 9  Next-Turn Guidance     [VOLATILE]
         - reflect LLM's recommendation for this turn
         - classification, proposed_next_family, override_invoked
         - NOTE: "When emitting verdict, base reasoning on §4a — not action history"
```

**Prompt caching:** Sections 1–2 are treated as the stable prefix and cached
with `cache_control: ephemeral` (5 min TTL). Sections 3–9 are re-sent fresh each
turn. This is split at the
`<!-- ---- workbook: stable | volatile ---- -->` marker. Section 1 includes
`state.subclaims`, so changing subclaims changes the nominally stable prefix and
prevents reuse of the previous prefix value.

---

## 6. Tool Calling: What the Planner Can Do

The planner has **4 tools** (3 always active, 1 optional):

```
┌──────────────────────────────────────────────────────────────────┐
│  Tool         │ Description                  │ Executes as       │
├───────────────┼──────────────────────────────┼───────────────────┤
│ python        │ Run Python code in workspace  │ subprocess.run    │
│  (code=...)   │ via evidence API              │ ["python3","-c",…]│
│               │ State does NOT persist across │ timeout: 600s     │
│               │ calls — reload each time      │                   │
├───────────────┼──────────────────────────────┼───────────────────┤
│ bash          │ Shell operations only         │ subprocess.run    │
│  (command=…)  │ (ls, cat, find)               │ ["bash","-c",…]   │
│               │ NOT for Python                │ timeout: 600s     │
├───────────────┼──────────────────────────────┼───────────────────┤
│ read_file     │ Read a file at given path     │ Path.read_text()  │
│  (path=…)     │                               │ (in-process)      │
├───────────────┼──────────────────────────────┼───────────────────┤
│ web_search    │ Web search via Serper API     │ HTTP request      │
│  (query=…,    │ with DuckDuckGo fallback      │ (Serper) or       │
│   num_results)│ Disabled if cfg.disable_      │ DDGS library      │
│               │ web_search=True               │                   │
└──────────────────────────────────────────────────────────────────┘
```

**GR6 enforcement:** The orchestrator explicitly drops all tool calls after the first in any response. Only `tool_calls[0]` is dispatched. This is enforced in code (not just the prompt), logged in the action ledger as `gr6_enforcement`.

**Output truncation:** The value used by `_smart_truncate()` is read from
`NB_MAX_OUTPUT_CHARS` when `evidence_programming_direct.py` is imported
(default 12,000). `cfg.max_output_chars` is injected into child subprocesses,
but does not update that already-imported parent-module constant. `python` and
`bash` write their full output to `execution_log.py` before returning a
truncated string; `read_file` and `web_search` log the truncated string.

The action ledger receives the already-truncated tool result. Consequently,
ledger artifacts and `raw_size_chars` describe that returned/truncated value,
not necessarily the original complete stdout/stderr. Artifact handles use an
`artifact://artifacts/...` display form, but `read_file` accepts filesystem
paths only; the planner must translate the handle to
`<workspace>/artifacts/...`.

**Output logged to:** `execution_log.py` (jupytext percent-format) → converted to `evidence_report.ipynb` at the end.

---

## 7. Evidence API: The Callable Library

All functions are importable from `proclaim.verification.evidence_api`. Calls inside the `python` tool are executed in a subprocess that loads state from `evidence_state.json`.

### 7.1 Mandatory First Call (Every Python Cell)

```python
from proclaim.verification.evidence_api import setup_workspace
state, llm, workspace = setup_workspace(
    claim="...",
    workspace_path="/path/to/workspace",
)
```

This is idempotent — loads existing `evidence_state.json` or creates a new one. Returns:
- `state`: `EvidenceState` (mutable; selected API mutation methods auto-save,
  while direct assignments such as `state.subclaims = ...` require an explicit
  `state._auto_save()` or `state.save()`)
- `llm`: `Callable[[str], str]` — the subagent LLM endpoint
- `workspace`: `Path`

### 7.2 Search Functions

```
search_pubmed(query, state, max_results=5)
    └─ RelevancePubMedSearcher → NCBI E-utilities → PaperRecord → state.papers

search_pubmed_llm(claim, state, llm, max_results=10)
    └─ generate_search_query(claim, llm)  [QUERY_GENERATION prompt]
    └─ generate_search_query(claim, llm, subclaims)  [subclaim-enriched]
    └─ two queries, deduplicated → state.papers

search_semantic_scholar(query, state, max_results=10)
    └─ S2Client.search() → _add_s2_records() → state.papers

search_semantic_scholar_dual(claim, state, llm, max_results=10)
    └─ generate_search_query_s2(claim, llm)  [QUERY_GENERATION_S2]
    └─ generate_search_query_s2(claim, llm, subclaims)
    └─ two queries, deduplicated → state.papers

search_semantic_scholar_recommendations(state, seed_pmids=None, max_results=20)
    └─ auto-derives positive seeds from the first configured stance
       (SUPPORT by default)
    └─ auto-derives negative seeds from papers containing only the second
       configured stance (REFUTE by default)
    └─ S2Client.recommendations(pos_ids, neg_ids) → state.papers

search_for_gap(gap_description, state, max_results=3)
    └─ delegates to search_pubmed()

find_related_articles(pmid, state, max_results=5)
    └─ NCBI elink (citation co-occurrence) → metadata fetch → state.papers

expand_via_citations(state, max_per_paper=5)
    └─ reference_dois from full-text JATS XML → S2Client.lookup_doi() → state.papers

refine_search_for_failed_papers(failed_pmids, state, llm, max_new_papers=5)
    └─ refine_search_query(llm, ...) [REFINE_SEARCH_QUERY]
    └─ → new queries → search_pubmed() × N
```

### 7.3 Full-Text Retrieval

```
get_full_text_article(pmid, state)
    └─ layered fallback chain (full_text.fetch_full_text):
         Layer 1  → PMC Open Access XML
         Layer 1b → Europe PMC REST API
         Layer 1.5→ Semantic Scholar OA PDF
         Layer 2  → INDRA literature
         Layer 3  → Unpaywall + PDF
         Layer 4  → PubMed or Semantic Scholar abstract
                    (may still return no text if remote lookup and stored
                     abstract are both unavailable)
    └─ updates paper.full_text, paper.reference_dois in state

get_paper_text(pmid, state)
    └─ returns formatted title + authors + abstract (no fetch)
```

### 7.4 Fact Extraction (Parallel)

```
extract_and_add_facts(llm, pmids, state, max_workers=8)
    └─ ThreadPoolExecutor(max_workers=8)
    └─ for each pmid not in state.extracted_pmids:
         _extract_and_add_facts_single(llm, pmid, state)
         └─ get_full_text_article → get_paper_text fallback
         └─ extract_facts(llm, paper_text, claim, subclaims, ...)
            └─ EXTRACT_FACTS prompt → SUBAGENT LLM → JSON array
         └─ add_facts_from_dicts(facts_dicts, state)
            └─ rejects a non-empty source_pmid not in state.papers
               (an empty/missing source_pmid currently passes validation)
            └─ deduplicates by (text, source_pmid)
            └─ state.add_fact() → auto-save
         └─ marks pmid as extracted → state.extracted_pmids
    └─ returns {pmid: fact_count} dict
```

### 7.5 Feature Population (NLP + Metadata)

```
populate_paper_features(state, compute_nli=True, max_text_length=10000)
    └─ for each paper not already featuring nlp + metadata:
         NLP features (via model_registry):
           ├─ compute_entity_coverage(claim, text)  [scispaCy NER in .venv310]
           ├─ get_semantic_similarity_computer().compute(claim, text)  [SBERT]
           └─ get_nli_entailment_computer().compute(claim, text)       [DeBERTa]
               returns: nli_entailment, nli_contradiction, nli_neutral, nli_best_chunk_text
         Metadata features (via model_registry):
           └─ get_metadata_extractor().extract_metadata(pmid)
               returns: publication_year, log_impact_factor,
                        normalized_citation_count, author_h_index_max
    └─ state.save() (batch save at end, not per-paper)

CRITICAL: Must be called BEFORE check_sufficiency().
CRITICAL: Never parallelized (PyTorch GPU models not thread-safe).
```

### 7.6 Filtering

```
filter_papers_by_stance(state)
    └─ non-destructive: keeps papers in state.papers
    └─ records which PMIDs have ≥1 non-default-stance fact:
         default keep_stances = all configured stances except default_stance
         state.sufficiency_candidate_pmids = [...PMIDs with a kept-stance fact...]
         state.stance_filter_removed_pmids = [...PMIDs excluded from classifier view...]
         state.stance_filter_keep_stances = [...configured kept stances...]
    └─ auto-saves

CRITICAL: Must be called BEFORE check_sufficiency().
```

### 7.7 Sufficiency

```
check_sufficiency(state, llm, threshold=0.5, min_total_papers=3)
    └─ build_sufficiency_view(state)  ← workspace-less clone of state
         if filter_papers_by_stance() has run:
             restricted to sufficiency_candidate_pmids
         otherwise:
             retains the full paper/fact corpus
    └─ backend dispatch (SUFFICIENCY_BACKEND env var):
         'mlp'   → _check_sufficiency_mlp(view, ...)
                     ├─ FeatureAggregator → feature vector
                     └─ SufficiencyMLP.predict_proba() → label + confidence
         'llm'   → _check_sufficiency_llm(view, ...)
                     └─ LLM_SUFFICIENCY_PROMPT → SUBAGENT LLM → score
         'haiku' → _check_sufficiency_haiku(view, ...)
                     └─ get_haiku_llm() → LLM_SUFFICIENCY_PROMPT → score
         (All backends) if insufficient:
             identify_gaps(llm, ...) [IDENTIFY_GAPS prompt → SUBAGENT LLM]
             → SufficiencyResult.gaps
    └─ _low_diversity_gate(label, current_paper_count, min_total_papers, ...)
         soft gate (default): keeps SUFFICIENT, adds LOW_DIVERSITY caveat gap
         hard gate: flips to INSUFFICIENT (forces more retrieval)
    └─ syncs view.sufficiency_history, view.iteration → state
    └─ state.iteration += 1
    └─ auto-saves
```

### 7.8 Verdict

```
emit_verdict(verdict, confidence, reasoning, key_evidence, gaps_remaining, state, workspace)
    └─ Structural quality gate (before writing):
         no facts extracted      → confidence capped at 0.10
         no papers retrieved     → confidence capped at 0.10
         zero subclaim coverage  → confidence capped at 0.30
         (verdict is still emitted even if quality is low)
    └─ creates VerificationVerdict (Pydantic)
    └─ writes verdict.json → signals the orchestrator to stop
    └─ state.checkpoint_save(workspace)
```

### 7.9 Curation Helpers

```
add_extraction_context_note(state, note)
    └─ appends note to state.extraction_context
    └─ CLEARS state.extracted_pmids (forces re-extraction)
    └─ auto-enqueues cleared PMIDs into curation re-extract bucket
    └─ auto-saves

enqueue_curation(state, bucket, *, pmid=None, reason=None, description=None,
                 priority="medium", subclaim=None)
    └─ adds item to curation_queue.json
    └─ buckets: re_rank, re_extract, filtered_to_revisit, zero_fact, gaps, contradictions
```

---

## 8. Subagent Layer: LLM-Powered Operations

Subagents are Python functions in `subagents.py` that take `(llm, ...)` and return structured data. They are called inside the `python` subprocess tool, using the subagent LLM.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     subagents.py                                    │
│                                                                     │
│  extract_facts(llm, paper_text, claim, subclaims, source_pmid,      │
│                extraction_context=None)                             │
│    Prompt: EXTRACT_FACTS                                            │
│    Returns: list[Fact]  (JSON array → validated Fact objects)       │
│    Input truncated: paper_text[:50000]                              │
│    Stance labels: from LabelConfig (defaults:                     │
│                   SUPPORT/REFUTE/NEUTRAL)                           │
│                                                                     │
│  synthesize_subclaim(llm, facts, subclaim)                          │
│    Prompt: SYNTHESIZE_SUBCLAIM                                      │
│    Returns: str (200-word synthesis)                                │
│                                                                     │
│  detect_conflicts(llm, facts)                                       │
│    Prompt: DETECT_CONFLICTS                                         │
│    Returns: list[dict] with fact_a_id, fact_b_id, description,      │
│             severity                                                │
│                                                                     │
│  identify_gaps(llm, claim, subclaims, facts)                        │
│    Prompt: IDENTIFY_GAPS                                            │
│    Returns: list[Gap] (1–4 gaps, sorted by priority)               │
│    Called by: check_sufficiency() when label="insufficient"         │
│                                                                     │
│  formulate_gap_queries(llm, gaps)                                   │
│    Prompt: FORMULATE_GAP_QUERIES                                    │
│    Returns: list[str] (3–8 word PubMed queries)                     │
│                                                                     │
│  refine_search_query(llm, claim, subclaims, failed_papers)          │
│    Prompt: REFINE_SEARCH_QUERY                                      │
│    Returns: list[str] (refined PubMed query strings)               │
└─────────────────────────────────────────────────────────────────────┘
```

**JSON parsing:** `extract_facts()`, `detect_conflicts()`, and
`identify_gaps()` use `_extract_json_array()` with direct parsing, heuristic
bracket extraction, bracket-pair matching, Unicode normalization, and
`<think>` stripping. `formulate_gap_queries()` and `refine_search_query()` use
separate, simpler `json.loads()` plus line-based fallbacks; they do not use the
common normalization/`<think>`-stripping path.

---

## 9. Reflection Module

The reflection module is a **meta-controller** that runs once per turn, before the planner call, and writes a structured recommendation to `reflection.json`.

```
Orchestrator every turn:
    │
    ├─ detect_stall_signals(workspace)  ← reads action_ledger.jsonl
    │    StallSignals:
    │      search_drought:     last 2 searches → 0 new papers
    │      extraction_drought: last 2 extractions → 0 new facts
    │      sufficiency_stall:  confidence did not improve between last 2 checks
    │      action_repetition:  same target head called 3 times in a row
    │
    └─ run_reflection(workspace, workbook_volatile, stall_signals, turn)
         │
         ├─ Build user prompt: REFLECTION_USER_PROMPT
         │    {stall_signals: "; "-joined reasons or "(none)"}
         │    {workbook_volatile: sections 3–9 of current workbook}
         │
         └─ litellm.completion(model=agent_model,  ← SAME model as planner
                               system=REFLECTION_SYSTEM_PROMPT,
                               user=REFLECTION_USER_PROMPT,
                               tools=[submit_reflection])
              │
              └─ submit_reflection tool call:
                   diagnosis:             short free-text root cause
                   classification:        retrieval | extraction | framing |
                                          budget | other
                   proposed_next_family:  search | extract | re_extract |
                                          feature_populate | filter |
                                          check_sufficiency | emit_verdict |
                                          curate | none
                   override_invoked:      string | null
                                          (prompt examples use GR1 / GR3;
                                           schema does not restrict the string)
              │
              └─ writes reflection.json
              └─ token usage billed to tracker as "reflect_llm"
```

**Reflection is rendered in Workbook Section 9** and consumed by the planner in
the same outer-loop turn: reflection runs first, then `write_workbook()` loads
the newly written `reflection.json`, and that workbook is sent to the planner.
Because `EvidenceState.init_new()` runs before entering the loop, this also
happens before the first planner call unless the reflect call fails.

**Stagnation escape:** When the reflection sees directional facts AND flat/declining sufficiency confidence, it proposes `emit_verdict` — stopping further retrieval that won't move the classifier.

---

## 10. Sufficiency Gate

The sufficiency gate is the key decision point controlling how many iterations run.

```
                    ┌───────────────────────────────────────┐
                    │         check_sufficiency()           │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │      build_sufficiency_view()         │
                    │   Clones state. If stance filtering    │
                    │   has run, restricts papers/facts to   │
                    │   sufficiency_candidate_pmids;         │
                    │   otherwise keeps the full corpus.     │
                    │   _workspace=None → no auto-save      │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │         Backend dispatch              │
                    │                                       │
                    │  SUFFICIENCY_BACKEND=mlp (default):   │
                    │    FeatureAggregator → feature vector  │
                    │    SufficiencyMLP.predict_proba()      │
                    │    → P(sufficient) → threshold gate   │
                    │                                       │
                    │  SUFFICIENCY_BACKEND=llm:             │
                    │    LLM_SUFFICIENCY_PROMPT              │
                    │    → subagent LLM → 0.0–1.0 score    │
                    │                                       │
                    │  SUFFICIENCY_BACKEND=haiku:           │
                    │    get_haiku_llm() → same prompt      │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │      Low-diversity gate               │
                    │                                       │
                    │  if SUFFICIENT but <3 candidate papers │
                    │    soft (default): keep SUFFICIENT +   │
                    │                   add LOW_DIVERSITY gap│
                    │    hard:          flip to INSUFFICIENT │
                    └───────────────────┬───────────────────┘
                                        │
                         label=INSUFFICIENT?
                              │
                    ┌─────────▼─────────────────────────────┐
                    │   identify_gaps(llm, claim,           │
                    │                subclaims, facts)      │
                    │   IDENTIFY_GAPS prompt → gaps list    │
                    │   gap types: missing_subclaim,        │
                    │   contradictory, low_source_diversity,│
                    │   weak_stance, missing_mechanism, etc.│
                    └───────────────────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │  Sync view → real state               │
                    │  state.sufficiency_history.append()   │
                    │  state.iteration += 1                 │
                    │  state._auto_save()                   │
                    └───────────────────────────────────────┘
```

**Verdict Readiness signals** (computed in workbook Section 7):

| Signal | Condition |
|--------|-----------|
| `sufficiency met threshold` | label=SUFFICIENT AND confidence >= threshold |
| `soft low-diversity caveat only` | SUFFICIENT with only LOW_DIVERSITY gaps |
| `stagnation escape` | directional facts exist AND confidence flat/declining over 3 checks |
| `final iteration / budget low` | last iteration OR turns_remaining ≤ 2 AND directional facts exist |

---

## 11. Curation Queue

The curation queue (`curation_queue.json`) is a to-do list for evidence that needs revisiting, organized into buckets:

```
curation_queue.json
├── re_rank:              papers to re-rank under a different framing
├── re_extract:           papers to re-extract (auto-populated when
│                         extraction_context changes)
├── filtered_to_revisit:  papers excluded from sufficiency view by
│                         filter_papers_by_stance (auto-populated)
├── zero_fact:            papers extracted but yielding 0 facts
├── gaps:                 targeted gaps to chase (from sufficiency gaps)
└── contradictions:       fact pairs needing resolution
```

**Auto-population:**
- When `add_extraction_context_note()` is called: cleared PMIDs → `re_extract` bucket
- On each orchestrator turn, `run_auto_curation()` adds latest sufficiency gaps,
  conflicts, and zero-fact papers.
- Existing entries are deduplicated but not automatically removed when they stop
  appearing in the latest state; the queue is cumulative until explicitly edited.
- The queue is rendered in Workbook Section 6 so the planner can prioritize.

**Manual enqueueing (via planner):**
```python
from proclaim.verification.evidence_api import enqueue_curation
# paper-keyed buckets (re_rank, re_extract, filtered_to_revisit, zero_fact)
enqueue_curation(state, "re_rank", pmid="12345", reason="reframed as off-claim")
# gaps bucket
enqueue_curation(state, "gaps", description="missing dose-response data",
                 priority="high", subclaim=state.subclaims[0])
# contradictions bucket
enqueue_curation(state, "contradictions", conflict_id="conflict_0",
                 reason="direction disagreement between two RCTs")
```

---

## 12. Prompt Templates & Conflicts

All prompts are defined in `prompts.py`. This is the full inventory:

### 12.1 Planner Prompts

| Constant | Used by | Key placeholders |
|----------|---------|------------------|
| `DIRECT_SYSTEM_PROMPT` | Direct mode orchestrator (system msg) | `verdict_names`, `verdict_definitions`, `claim`, `workspace`, `function_docs`, `schemas`, `max_iterations`, `sufficiency_threshold`, `subclaim_examples`, `web_search_step` |
| `NOTEBOOK_SYSTEM_PROMPT` | Notebook mode orchestrator (system msg) | similar + `notebook_path` |
| `NOTEBOOK_USER_PROMPT` | Notebook mode (initial user msg) | `claim`, `notebook_path`, `sufficiency_threshold`, `max_iterations` |

### 12.2 Reflection Prompts

| Constant | Used by | Key placeholders |
|----------|---------|------------------|
| `REFLECTION_SYSTEM_PROMPT` | Reflect LLM (system msg) | *(none — no format placeholders)* |
| `REFLECTION_USER_PROMPT` | Reflect LLM (user msg) | `stall_signals`, `workbook_volatile` |

### 12.3 Subagent Prompts

| Constant | Used by | Key placeholders |
|----------|---------|------------------|
| `EXTRACT_FACTS` | `subagents.extract_facts` | `stance_options`, `stance_block`, `claim`, `subclaims_str`, `context_block`, `source_pmid`, `paper_text` |
| `SYNTHESIZE_SUBCLAIM` | `subagents.synthesize_subclaim` | `subclaim`, `facts_str` |
| `DETECT_CONFLICTS` | `subagents.detect_conflicts` | `facts_str` |
| `IDENTIFY_GAPS` | `subagents.identify_gaps` | `claim`, `subclaims_str`, `num_facts`, `counts_str`, `unique_sources`, `facts_str` |
| `FORMULATE_GAP_QUERIES` | `subagents.formulate_gap_queries` | `gaps_str` |
| `REFINE_SEARCH_QUERY` | `subagents.refine_search_query` | `claim`, `subclaims_str`, `papers_str` |

### 12.4 Search Query Generation Prompts

| Constant | Used by | Key placeholders |
|----------|---------|------------------|
| `QUERY_GENERATION` | `search/llm_query_generator.generate_search_query` | `claim` |
| `QUERY_GENERATION_S2` | `search/llm_query_generator.generate_search_query_s2` | `claim`, `subclaims_section` |
| `GENERATE_GAP_QUERY` | `search/llm_query_generator.generate_gap_query` | `claim`, `gap_description` |

### 12.5 Sufficiency and Verdict Prompts

| Constant | Used by | Key placeholders | Notes |
|----------|---------|------------------|-------|
| `LLM_SUFFICIENCY_PROMPT` | `llm_sufficiency.check_sufficiency_llm` | `claim`, `reference` | Uses `str.replace()` — NOT `.format()` (contains literal JSON `{}`) |
| `FORCE_VERDICT_PROMPT` | `evidence_programming_direct._force_verdict` | `verdict_packet`, `verdict_names` | Called inside forced-verdict subprocess |

### 12.6 Conflict: LLM_SUFFICIENCY_PROMPT Uses str.replace

Most prompts use `str.format()`. `LLM_SUFFICIENCY_PROMPT` is the exception — it uses `str.replace("{claim}", ...)` because the template body contains literal JSON `{}` braces that would break `.format()`. This is a known deviation documented in a comment in the source.

### 12.7 Label Injection into Prompts

The main label definition blocks are dynamically injected via `LabelConfig`:

```python
# In extract_facts():
stance_block = label_cfg.stance_prompt_block()   # builds human-readable text
stance_options = label_cfg.stance_options_str()  # e.g. '"SUPPORT" | "REFUTE" | "NEUTRAL"'

# In DIRECT_SYSTEM_PROMPT format():
verdict_names = ", ".join(label_cfg.verdict_names())
verdict_definitions = label_cfg.verdict_prompt_block()
```

This supports configurable label names and descriptions, but the implementation
is not fully taxonomy-neutral. Several prompt examples and workflow instructions
still explicitly mention `SUPPORT`, `REFUTE`, and neutral facts. Semantic Scholar
recommendations also interpret the first configured stance as positive and the
second as negative. Custom taxonomies must preserve those ordering/semantic
assumptions or update the affected code and prompts.

---

## 13. Durable State & Disk Artifacts

Run state and artifacts are persisted under `workspace/`. The workspace path is
`{output_dir}/workspace/`, where `output_dir` is
`VerificationSettings.resolved_output_dir` (an explicit `output_dir` config
value, or an auto-generated timestamped path under `results/verification/`).
The experiments runner (`run_signor_eval.py`) computes its own `output_dir` per
claim using `run_tag`, but that is evaluation harness logic, not part of
`VerificationSettings`.

```
workspace/
├── evidence_state.json       ← EvidenceState (Pydantic JSON)
│                                auto-saved by selected mutation methods;
│                                direct/nested assignments need explicit save
│                                contains: claim, subclaims, papers,
│                                facts, conflicts, extraction_context,
│                                extracted_pmids, sufficiency_history,
│                                coverage, synthesis,
│                                papers_per_iteration, token_estimate,
│                                iteration, stance_filter_*
│
├── workbook.md               ← regenerated/overwritten each turn; latest only
├── verdict_packet.md         ← bounded fact-centered view, also overwritten
│                                 each turn; latest only
├── execution_log.py          ← jupytext percent-format audit log
│                                grows each turn (one cell per tool call)
├── verdict.json              ← VerificationVerdict (termination signal)
│                                verdict, confidence, reasoning,
│                                key_evidence, gaps_remaining
│                                (no claim or timestamp fields)
│
├── action_ledger.jsonl       ← append-only ActionRecord log
│                                one row per tool call (+ reflect entries)
│                                fields: action_id, turn, timestamp,
│                                target, observation, delta,
│                                diagnosis, artifact_handle,
│                                raw_size_chars
│
├── reflection.json           ← latest ReflectionRecord (overwritten each turn)
│                                fields: turn, timestamp, diagnosis,
│                                classification, proposed_next_family,
│                                override_invoked
│
├── curation_queue.json       ← CurationQueue (re_rank, re_extract, gaps, ...)
│
└── artifacts/                ← spilled ledger outputs (>4000 chars or errors);
                                  these may already be truncated by dispatch
    ├── action_0001_search_pubmed.txt
    ├── action_0004_extract_and_add_facts.txt
    └── ...

output_dir/                   ← (one level up from workspace)
├── token_usage.json          ← aggregated cost/token summary
├── token_usage_trace.json    ← per-call token trace (llm_call + reflect_llm)
├── evidence_report.ipynb     ← notebook generated from execution_log.py
└── run.log                   ← Python logging output
```

---

## 14. Guardrails (GR1–GR9)

Guardrails are written into the workbook (Section 2) and enforced by a combination of the LLM prompt, the orchestrator code, and the reflection module. Some are **code-enforced** (the orchestrator implements them regardless of what the planner says), others are **prompt-enforced only**.

| ID | Rule | Enforcement | Override Condition |
|----|------|-------------|-------------------|
| **GR1** | Do not rerun the same search query unless reflection records a changed rationale | Prompt only | Reflection records `GR1` override with different rationale |
| **GR2** | Do not re-extract the same PMID under the same extraction-context version | **Code**: `extracted_pmids` cache in `extract_and_add_facts()` | `add_extraction_context_note()` called (clears cache) |
| **GR3** | Do not broad-search after extraction failure unless reflection classifies problem as retrieval | Prompt only | Reflection classifies `retrieval` and records `GR3` |
| **GR4** | Do not repeat a full status narration when no material state change occurred | Prompt only | State delta is non-empty |
| **GR5** | Do not consume final turns on retrieval if verdict readiness is already high | Prompt + workbook §7 signal | turns_remaining > 2 AND still insufficient |
| **GR6** | Propose one next action only | **Code**: orchestrator drops `tool_calls[1:]` | Never (hard contract) |
| **GR7** | Always call `populate_paper_features()` before `check_sufficiency()` | Prompt only | Never (MLP requires features) |
| **GR8** | Always call `filter_papers_by_stance()` before `check_sufficiency()` | Prompt only | Never (required for stable signal) |
| **GR9** | Never fabricate facts; every fact must come from a retrieved paper | **Partial code check**: rejects unknown non-empty `source_pmid`, but currently accepts an empty/missing `source_pmid` | Prompt says never; code check is not a complete invariant |

**Override mechanism:** Reflection's `override_invoked` field names a specific guardrail (e.g. `"GR3"`). This is rendered in Section 9. The planner is told that an override is valid **only when Section 9 names it** — not self-granted by the planner.

---

## 15. Forced Verdict Path

When the main loop exits without a `verdict.json`, `_force_verdict()` is called:

```
_force_verdict(workspace, claim, sub_env)
    │
    ├─ Writes a Python script to a temp file (tempfile.NamedTemporaryFile)
    │
    └─ subprocess.run(["python3", tmp_path], env=sub_env, timeout=300)
         │
         ├─ setup_workspace(claim, workspace_path)
         ├─ populate_paper_features(state)
         ├─ if state.sufficiency_history: reuse last result
         │  else: check_sufficiency(state, llm)
         ├─ build_verdict_packet(state, label_cfg)
         │    → fact-centered, bounded evidence view:
         │       claim, label definitions, facts grouped by stance/PMID
         │       (max 20 facts per stance), up to 10 non-directional papers,
         │       sufficiency metadata and latest gaps as labeled caveats
         │
         └─ SUBAGENT LLM(FORCE_VERDICT_PROMPT.format(verdict_packet=..., verdict_names=...))
              │
              └─ Parses structured response:
                   VERDICT: one configured verdict label
                   CONFIDENCE: 0.0–1.0
                   REASONING: one paragraph
                   KEY_EVIDENCE: bullet 1 | bullet 2 | bullet 3
              │
              └─ emit_verdict(verdict, confidence, reasoning, key_evidence,
                              gaps[:5], state, workspace)
                   └─ writes verdict.json
```

**Key design choice:** The forced verdict prompt explicitly instructs the LLM
to base reasoning on the **known facts** in the verdict packet. Reflection and
retrieval history are excluded from the packet; latest gaps remain present as
labeled caveats that the prompt says not to treat as evidence. This reduces, but
does not structurally eliminate, missing-evidence anchoring.

---

## 16. Configuration & Label System

```
VerificationSettings (pydantic-settings)
├── claim: str
├── mode: "direct" | "sdk"        ("sdk" = Claude Agent SDK + Jupyter kernel)
├── max_iterations: int = 8       (sufficiency check limit)
├── max_turns: int = 30           (Direct: per-iteration planning budget;
│                                    current SDK orchestrator does not use it)
├── Direct hard planner-call ceiling = max_iterations × max_turns
├── sufficiency_threshold: float = 0.80
├── sufficiency_backend: "mlp" | "llm" | "haiku"
├── mlp_model_dir: str
├── disable_web_search: bool = False
├── include_subclaim_examples: bool = True
├── output_dir: Path
├── verbose: bool
├── llm:
│   ├── model: str = "glm-5"
│   ├── subagent_model: Optional[str] = None  (falls back to model)
│   ├── subagent_base_url: str = "http://localhost:8000/v1/"
│   ├── temperature: float = 0.2
│   ├── thinking_budget_tokens: int = 0
│   └── disable_thinking: bool = True
│
├── labels: LabelConfig
│   ├── stance_labels: dict[str, str]   (SUPPORT, REFUTE, NEUTRAL + descriptions)
│   ├── verdict_labels: dict[str, str]  (SUPPORT, REFUTE, UNCERTAIN + descriptions)
│   └── default_stance: str = "NEUTRAL"
│
└── api_key: str  (property: resolves ANTHROPIC_API_KEY → GLM_API_KEY → ZAI_API_KEY → OPENAI_API_KEY)

Effective resolution depends on the construction path:
  1. Explicit programmatic/CLI overrides
  2. Values supplied by YAML (passed as BaseSettings constructor arguments)
  3. Environment/.env values for fields not supplied above
  4. Field defaults

Because pydantic-settings gives constructor arguments priority over environment
sources, a YAML value loaded by `from_yaml()` / `from_cli()` overrides the same
environment variable rather than sitting below it.
```

**Subprocess env injection:** The orchestrator passes these to the `python`/`bash` subprocess via `build_subprocess_env()`:

```
LLM_BASE_URL       → cfg.llm.subagent_base_url
LLM_API_KEY        → cfg.api_key
LLM_MODEL          → cfg.subagent_model
LLM_TEMPERATURE    → cfg.llm.temperature
LLM_DISABLE_THINKING → cfg.llm.disable_thinking
MLP_MODEL_DIR      → cfg.mlp_model_dir
SUFFICIENCY_BACKEND → cfg.sufficiency_backend
MAX_ITERATIONS     → cfg.max_iterations
LABEL_CONFIG_JSON  → cfg.labels.model_dump_json()
NB_MAX_OUTPUT_CHARS → cfg.max_output_chars in child processes
                      → parent Direct dispatcher already captured its limit at import
EVIDENCE_DEBUG     → cfg.verbose
PYTHONPATH         → includes src/
```

---

## 17. Known Tensions & Design Trade-offs

### 17.1 Planner State Is Stateless, But Subagent State Is Not

The planner LLM gets a fresh 2-message context every turn (stateless). But the `python` subprocess calls read/write `evidence_state.json`, which accumulates across turns. This is intentional: the workbook (derived from `EvidenceState`) is the "memory" — not the conversation.

**Tension:** If the workbook generation fails (e.g., `EvidenceState` is corrupted), the planner falls back to a minimal stub. This can cause the planner to re-run actions already done.

### 17.2 Reflect LLM vs. Planner: Redundant Reasoning?

Both the reflect LLM and the planner LLM use the same model. The reflect LLM runs first and produces `proposed_next_family`. The planner then reads this in Section 9 and is expected to follow it. But the planner's full prompt also contains the full workflow description, so it might ignore or contradict the reflection.

**Mitigation:** Workbook Section 9 explicitly says "Read Section 9 first" and "If Section 9 says emit_verdict, call emit_verdict." However, the planner is not hard-constrained from disagreeing — this is prompt-level guidance only.

### 17.3 GR6 (One Tool Call) vs. Multi-Step Cells

GR6 limits the planner to one tool call per turn. But a single `python` cell can contain multiple Python statements (e.g., `setup_workspace` + `search_pubmed_llm` + `extract_and_add_facts`). GR6 applies to the number of *tool calls* (OpenAI function calls), not to the number of statements inside one cell. This is explicitly stated in the comments.

**Tension:** A multi-step cell can succeed partially and fail at the last step, leaving state in an intermediate position with no way to tell from the ledger which step failed.

### 17.4 Sufficiency View vs. Full Corpus

After `filter_papers_by_stance()` has run, `check_sufficiency()` scores only
the candidate papers/facts selected by that filter. If the filter has not run,
`build_sufficiency_view()` keeps the full corpus. `emit_verdict()` always receives
the full `state`. The intended design is that the classifier decides *when to
stop*, while verdict reasoning can use all extracted facts.

**Tension:** The planner prompt and Section 9 sometimes anchor on "insufficient" gap descriptions from the MLP/sufficiency check, causing verdicts that emphasize missing evidence over the actual directional facts extracted. Section 9's emit_verdict guidance explicitly addresses this: "base reasoning on Section 4a, not on action history."

### 17.5 Token Cost: Reflect LLM Runs Every Turn

The reflect LLM call runs on every turn (not only on stall), using the same planner model. This doubles the effective LLM cost per turn when the planner model is expensive (e.g., Sonnet). Token usage is tracked separately as `reflect_llm` in the cost trace.

### 17.6 Prompt Cache Disabled with Extended Thinking

Anthropic's API does not support prompt caching when `thinking` is enabled. When `thinking_budget_tokens > 0`, both the system-prompt cache and the workbook stable-prefix cache are disabled, significantly increasing token costs for extended-thinking runs.

### 17.7 LLM_SUFFICIENCY_PROMPT Uses str.replace

This is the one prompt in the codebase that cannot use `.format()` (because the template body contains literal `{"sufficiency_score": ...}` JSON). If someone tries to `.format()` it, Python will raise `KeyError`. The comment in `prompts.py` documents this, but it is easy to break if someone edits the prompt.

### 17.8 formulate_gap_queries: Workflow Text Contradicts Function Signature

**This is a live bug in the prompts.**

In both `DIRECT_SYSTEM_PROMPT` (lines 379, 519) and `NOTEBOOK_SYSTEM_PROMPT` (line 379), the `## Workflow` section says:

```
c. formulate_gap_queries(llm, state) — LLM-generated gap queries
```

But the actual function in `subagents.py` is:

```python
def formulate_gap_queries(llm: LLMCallable, gaps: list[Gap]) -> list[str]:
```

The second argument is `gaps` (a `list[Gap]` from `state.sufficiency_history[-1].gaps`), **not** `state`. The same system prompt's `## Available functions` block is auto-generated by `function_docs()` and shows the correct signature — so there is a direct contradiction within the same prompt. A planner following the `## Workflow` text literally will receive a `TypeError` at runtime. The correct call is:

```python
from proclaim.verification.subagents import formulate_gap_queries
queries = formulate_gap_queries(llm, state.sufficiency_history[-1].gaps)
```

### 17.9 reflection.py Comment Claims Conditional Firing; Code Always Fires

The module-level comment in `reflection.py` (lines 266–267) says:

> "It runs only when the orchestrator's stall-signal detector fires, so cost is bounded."

In reality, `evidence_programming_direct.py` calls `run_reflection()` before
every planner call. `EvidenceState.init_new()` creates `evidence_state.json`
before the loop, so the existence check is already true on the first turn. Stall
signals are passed as context into the reflect prompt, but do not gate whether
the call happens. The comment is stale from an earlier design where reflection
was intended to be conditional.

---

## Appendix A: Standard Workflow Sequence

```
The reflection call precedes each planner call. A typical planner-action
sequence is:

Call 1:  python(setup_workspace + set subclaims + explicit state save)
Call 2:  web_search (if enabled)
Call 3:  python(PubMed/Semantic Scholar paper search)
Call 4:  python(extract_and_add_facts(llm, all_pmids, state, max_workers=8))
Call 5:  python(populate_paper_features(state))
Call 6:  python(filter_papers_by_stance(state))
Call 7:  python(check_sufficiency(state, llm))
         ├── SUFFICIENT → emit_verdict on a later planner call
         └── INSUFFICIENT → read state.sufficiency_history[-1].gaps
Later calls, one tool call at a time:
         ├── search_for_gap(gap.description, state)
         ├── search_semantic_scholar_recommendations(state)
         └── formulate_gap_queries(
                 llm, state.sufficiency_history[-1].gaps
             )
         Then repeat extraction → features → filtering → sufficiency.
Final:   emit_verdict(verdict, confidence, reasoning, key_evidence, gaps, ...)
```

## Appendix B: File → Role Map

| File | Role |
|------|------|
| `evidence_programming_direct.py` | Main orchestrator (direct mode) |
| `evidence_programming.py` | Main orchestrator (notebook mode) |
| `evidence_api.py` | Evidence API library (all callable functions) |
| `subagents.py` | LLM-powered subagent functions |
| `prompts.py` | All prompt templates |
| `workbook.py` | Workbook renderer (9 sections) |
| `action_ledger.py` | Action ledger (JSONL append-only) |
| `reflection.py` | Stall detection + reflect LLM |
| `curation.py` | Curation queue management |
| `config.py` | `VerificationSettings` + `LabelConfig` |
| `evidence_state.py` | `EvidenceState` (central data structure) |
| `data_models.py` | Pydantic schemas: `Fact`, `PaperRecord`, `SufficiencyResult`, `VerificationVerdict`, etc. |
| `feature_tools.py` | NLP feature computation (entity coverage) |
| `llm_sufficiency.py` | LLM/Haiku sufficiency backend |
| `full_text.py` | 4-layer full-text retrieval |
| `compressor.py` | `SufficiencyPreservingCompressor` |
| `model_registry.py` | ML model singleton management (SBERT, NLI, MLP, metadata) |
| `cost_tracker.py` | Token/cost tracking |
| `verdict_packet.py` | Verdict packet builder (clean fact view) |
| `search/custom_pubmed.py` | PubMed search client |
| `search/semantic_scholar.py` | Semantic Scholar API client |
| `search/llm_query_generator.py` | LLM-based query generation |
