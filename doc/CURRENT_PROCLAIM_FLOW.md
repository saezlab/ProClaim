# Current ProClaim Flow

This document describes the current `direct` mode implemented by
`evidence_programming_direct.py`. It distinguishes:

- what the orchestrator reads;
- what each LLM or classifier actually sees;
- which files are written;
- where decisions are made.

## 1. End-to-End Control Flow

```mermaid
flowchart TD
    A["CLI / YAML / environment<br/>claim, models, thresholds, budgets"] --> B["verify_claim_direct"]
    B --> C["Create output directory and workspace"]
    C --> D["Initialize evidence_state.json<br/>Initialize execution_log.py"]
    D --> E{"call_count &lt; max_iterations x max_turns?"}

    E -- No --> FV["Force verdict path"]
    E -- Yes --> F["Load evidence_state.json<br/>detect iteration rollover and remaining turns"]

    F --> G["Auto-curation<br/>read state and queue<br/>refresh gaps, conflicts, zero-fact papers"]
    G --> H["Detect stall signals<br/>read action ledger and sufficiency history"]
    H --> I["Build volatile workbook tail"]
    I --> J["Reflection LLM<br/>diagnose current stage or stall<br/>recommend next action family"]
    J --> K["Write reflection.json<br/>append reflect row to action_ledger.jsonl"]

    K --> L["Regenerate verdict_packet.md<br/>Regenerate workbook.md from durable state"]
    L --> M{"Prompt branch"}
    M -- "first planner call" --> M1["Start / setup instruction"]
    M -- "last 2 turns, non-final iteration" --> M2["Require check_sufficiency"]
    M -- "last 2 turns, final iteration" --> M3["Require check_sufficiency then verdict"]
    M -- otherwise --> M4["Continue with one next action"]

    M1 --> N["Planner LLM sees<br/>system prompt + current workbook + instruction"]
    M2 --> N
    M3 --> N
    M4 --> N

    N --> O{"Planner returned a tool call?"}
    O -- No --> P{"verdict.json exists?"}
    P -- Yes --> W
    P -- No --> P2{"latest confidence >= threshold?"}
    P2 -- Yes --> FV
    P2 -- No --> E

    O -- Yes --> Q["GR6 enforcement<br/>execute only the first top-level tool call"]
    Q --> R{"Selected tool"}
    R -- python --> R1["Run one Python subprocess<br/>usually setup_workspace + one evidence API action"]
    R -- bash --> R2["Run shell command in workspace"]
    R -- read_file --> R3["Read an explicitly selected file"]
    R -- web_search --> R4["Serper or DuckDuckGo snippets"]

    R1 --> S["Append command/output to execution_log.py"]
    R2 --> S
    R3 --> S
    R4 --> S

    S --> T["Reload evidence_state.json<br/>compute before/after delta"]
    T --> U["Append action_ledger.jsonl<br/>spill long/error output to artifacts/"]
    U --> V{"verdict.json exists?"}
    V -- Yes --> W["Finish run"]
    V -- No --> E

    FV --> FV1["Load full evidence state"]
    FV1 --> FV2["Populate paper features"]
    FV2 --> FV3["Reuse latest sufficiency result<br/>or run check_sufficiency"]
    FV3 --> FV4["Forced-verdict LLM sees only clean verdict packet"]
    FV4 --> FV5["Parse label, confidence, reasoning, key evidence"]
    FV5 --> FV6["emit_verdict writes verdict.json"]
    FV6 --> W

    W --> X["Write token usage files<br/>Convert execution_log.py to evidence_report.ipynb"]
```

## 2. What Each Decision Maker Sees

```mermaid
flowchart LR
    subgraph Durable["Durable state and control files"]
        ES["evidence_state.json<br/>papers, full text, facts, features,<br/>conflicts, sufficiency history"]
        AL["action_ledger.jsonl<br/>recent actions and state deltas"]
        CQ["curation_queue.json<br/>re-extract, gaps, conflicts, zero-fact"]
        RF["reflection.json<br/>latest diagnosis and next family"]
        VP["verdict_packet.md<br/>full-corpus fact-centered view"]
        AR["artifacts/*.txt<br/>large or error outputs"]
        EL["execution_log.py<br/>complete operational audit"]
    end

    ES --> WB["workbook.md"]
    AL --> WB
    CQ --> WB
    RF --> WB
    VP -.->|filename reference| WB
    AR -.->|handles only| WB

    WB --> PL["Planner LLM"]
    SP["DIRECT_SYSTEM_PROMPT<br/>workflow, tools, guardrails"] --> PL
    TI["Turn-specific instruction<br/>normal, urgent sufficiency, or final"] --> PL

    ES --> VT["Volatile workbook tail"]
    AL --> VT
    CQ --> VT
    RF --> VT
    VT --> RL["Reflection LLM"]
    SS["Stall signals"] --> RL
    RP["REFLECTION_SYSTEM_PROMPT"] --> RL

    ES --> SV["Sufficiency candidate clone<br/>only filtered candidate PMIDs and facts"]
    SV --> CL{"Configured sufficiency backend"}
    CL --> MLP["MLP classifier"]
    CL --> SLLM["LLM / Haiku sufficiency evaluator"]

    ES --> FVP["Clean full-corpus verdict packet"]
    FVP --> FLLM["Forced-verdict LLM"]

    EL -.->|not automatically injected| PL
    AR -.->|only after explicit read_file| PL
```

### Actual visibility

| Component | Sees | Does not automatically see | Decision |
| --- | --- | --- | --- |
| Planner LLM | `DIRECT_SYSTEM_PROMPT`, full current `workbook.md`, turn instruction | previous chat turns, raw `execution_log.py`, full artifacts, raw `action_ledger.jsonl`, raw `evidence_state.json` | Selects the next concrete action and arguments |
| Reflection LLM | stall signals and the volatile workbook tail | raw paper text, raw execution log, full stable workbook prefix | Classifies retrieval/extraction/framing/budget state and recommends an action family |
| Extraction subagent | claim, subclaims, latest extraction context, one paper's full text or abstract | workbook control text and action history | Extracts grounded facts and stance labels |
| Sufficiency backend | a cloned candidate-only state after stance filtering | excluded papers and facts in the full corpus | Produces sufficient/insufficient, confidence, and gaps |
| Forced-verdict LLM | clean full-corpus verdict packet | action history, reflection diagnosis, guardrails, recuration queue | Chooses final verdict, confidence, reasoning, and key evidence |

The current implementation builds the reflection input with
`build_workbook_parts()`. Its volatile tail includes Sections 3, 4, 4a, 5, 6,
7, 8, and 9, so the reflection LLM can also see the previous turn's Section 9
guidance.

## 3. Evidence Action Flow

```mermaid
flowchart TD
    A["Planner chooses one action family"] --> B{"Action family"}

    B -- search --> S1["PubMed / Semantic Scholar search<br/>or web orientation snippets"]
    S1 --> S2{"Structured paper search?"}
    S2 -- Yes --> S3["Add PaperRecord objects to state.papers"]
    S2 -- "No, web_search" --> S4["Keep snippets in audit/ledger only<br/>not verdict evidence"]

    B -- extract --> E1["For each new paper ID"]
    E1 --> E2{"paper.full_text already stored?"}
    E2 -- Yes --> E5["Use stored full text"]
    E2 -- No --> E3["Try PMC, Europe PMC, S2 OA PDF,<br/>INDRA, Unpaywall/PDF, PubMed abstract"]
    E3 --> E4["Store recovered text in PaperRecord.full_text"]
    E4 --> E5
    E5 --> E6["Extraction LLM returns atomic grounded facts"]
    E6 --> E7["Validate source paper, deduplicate,<br/>update facts, coverage, extracted_pmids"]

    B -- feature_populate --> F1["Compute NLP and metadata features for papers"]

    B -- filter --> P1["Find papers with non-default stance facts"]
    P1 --> P2["Persist candidate and excluded PMID lists<br/>do not delete papers or facts"]

    B -- check_sufficiency --> C1["Clone full state"]
    C1 --> C2["Restrict clone to candidate PMIDs and facts"]
    C2 --> C3["Run MLP, LLM, or Haiku backend"]
    C3 --> C4["Apply soft low-diversity caveat by default"]
    C4 --> C5["Append result to full state's sufficiency history<br/>increment iteration"]

    B -- emit_verdict --> V1["Planner supplies label, confidence,<br/>reasoning, key evidence, and gaps"]
    V1 --> V2["emit_verdict checks the full state<br/>and applies quality confidence caps"]
    V2 --> V3["Write verdict.json and checkpoint state/trace"]

    S3 --> Z["Auto-save evidence_state.json"]
    E7 --> Z
    F1 --> Z
    P2 --> Z
    C5 --> Z
    V3 --> Z
```

## 4. Workspace File Map

```mermaid
flowchart LR
    ORCH["Direct orchestrator"] -->|initialize / reload| ES["workspace/evidence_state.json"]
    ORCH -->|append audit cells| EL["workspace/execution_log.py"]
    ORCH -->|append action delta| AL["workspace/action_ledger.jsonl"]
    ORCH -->|write latest reflection| RF["workspace/reflection.json"]
    ORCH -->|refresh queue| CQ["workspace/curation_queue.json"]
    ORCH -->|regenerate every turn| WB["workspace/workbook.md"]
    ORCH -->|regenerate every turn| VP["workspace/verdict_packet.md"]
    ORCH -->|spill large/error output| AR["workspace/artifacts/*.txt"]
    ORCH -->|final result| V["workspace/verdict.json"]
    ORCH -->|state operation trace| TR["workspace/trace.json"]

    EL -->|post-run conversion| NB["output/evidence_report.ipynb"]
    ORCH --> TU["output/token_usage.json"]
    ORCH --> TT["output/token_usage_trace.json"]
    ORCH --> LG["output/run.log"]

    ES --> WB
    AL --> WB
    RF --> WB
    CQ --> WB
    VP -->|filename reference only| WB
```

### File purpose

| File | Read by current loop | Written by current loop | Role |
| --- | --- | --- | --- |
| `evidence_state.json` | orchestrator, workbook, curation, tools, sufficiency | evidence API mutations | Source of truth for papers, facts, features, filtering, and sufficiency |
| `workbook.md` | planner; indirectly reflection through an in-memory rebuild | workbook renderer every turn | Compact planning context |
| `action_ledger.jsonl` | workbook, stall detector | orchestrator after reflection/tool actions | Anti-repetition memory and state deltas |
| `reflection.json` | workbook; next reflection input through Section 9 | reflection LLM path | Latest diagnosis and recommended action family |
| `curation_queue.json` | workbook, auto-curation | curation helpers | Explicit retry/gap/contradiction queue |
| `verdict_packet.md` | explicit file read only; workbook references its filename, while forced verdict rebuilds the same packet in memory | workbook renderer | Clean fact-centered final reasoning view |
| `execution_log.py` | notebook generator; explicit `read_file` only if requested | every dispatched tool action and planner narration | Full audit, not primary planner context |
| `artifacts/*.txt` | only after explicit file read | action ledger spill logic | Raw large/error outputs |
| `verdict.json` | stop checks and final display | `emit_verdict` | Terminal result |
| `trace.json` | audit/debugging | state checkpoints and traced operations | Detailed state-operation audit |

## 5. Decision Points

1. **Iteration rollover:** reset the per-iteration turn counter after
   `check_sufficiency` increments `state.iteration`.
2. **Auto-curation:** derive queue items from latest gaps, conflicts, and
   extracted papers that produced zero facts.
3. **Reflection:** choose the recommended family:
   `search`, `extract`, `re_extract`, `feature_populate`, `filter`,
   `check_sufficiency`, `emit_verdict`, `curate`, or `none`.
4. **Turn urgency:** with two turns left, close a non-final iteration with
   sufficiency; in the final iteration, stop retrieval and move to verdict.
5. **Planner action:** choose one concrete top-level tool call. GR6 drops every
   additional top-level call.
6. **Search result handling:** PubMed/S2 results become papers; generic web
   snippets remain orientation/audit text and are not inserted as facts.
7. **Extraction cache:** skip PMIDs in `extracted_pmids`; a new terminology
   context note clears the cache and queues re-extraction.
8. **Stance filter:** retain the full corpus but select only papers with
   non-default stance facts for sufficiency.
9. **Sufficiency:** evaluate the candidate-only clone, then copy only
   sufficiency bookkeeping back to the full state.
10. **Verdict readiness:** recommend `emit_verdict` on threshold, a soft
    low-diversity-only caveat, stagnation with directional facts, or
    final-budget pressure. The loop itself terminates only after a verdict, or
    exits to the forced-verdict path.
11. **Forced verdict:** if no `verdict.json` exists when the loop exits, run the
    clean packet-based verdict path.

## 6. Source Files Inspected

- `src/proclaim/verification/evidence_programming_direct.py`
- `src/proclaim/verification/workbook.py`
- `src/proclaim/verification/action_ledger.py`
- `src/proclaim/verification/reflection.py`
- `src/proclaim/verification/curation.py`
- `src/proclaim/verification/evidence_state.py`
- `src/proclaim/verification/evidence_api.py`
- `src/proclaim/verification/verdict_packet.py`
- `src/proclaim/verification/prompts.py`
- `src/proclaim/verification/full_text.py`
- `src/proclaim/verification/config.py`
