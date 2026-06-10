# ProClaim Improvement Plan

## Objective

Refactor ProClaim's direct scientific claim verification loop so it targets literature consensus with lower context cost, less duplicate work, and clearer recovery when retrieval or extraction fails.

The core change is to replace `execution_log.py` replay as the planning surface with a compact markdown workbook that is regenerated each turn from structured state. The existing execution log and rendered notebook should remain as audit artifacts, not as the main context fed back to the planner.

## Current Constraints

- The direct orchestrator currently rebuilds every turn from `workspace/execution_log.py`.
- Tool results flow through bash output appended to the log, which mixes planning state with verbose narration and raw stderr.
- `evidence_state.json` already contains high-signal durable state, but it is not surfaced in a planner-friendly format.
- The agent has no compact memory of failed search, extraction, or curation strategies.
- Reflection and re-curation are currently implicit recovery behaviors rather than explicit workflow stages.

## Design Goal

Keep the current execution model that works well:

- one outer planner call at a time
- tool execution through bash or native tools
- durable evidence state on disk
- notebook and trace artifacts for auditability

Change the planning interface:

- planner reads a bounded markdown workbook instead of a long execution transcript
- planner chooses one next action only
- orchestrator executes it
- orchestrator summarizes the observation
- orchestrator updates reflection and curation state before the next turn

This yields a bounded ReAct-style loop:

1. `Act`: execute one tool action, usually via bash.
2. `Observe`: summarize what changed in state and what failed.
3. `Reflect`: explain why progress did or did not happen.
4. `Curate`: update the evidence pool, retry queue, and next-action constraints.

## Planning And Audit Split

### Planning artifacts

- `workspace/workbook.md`: primary planner context, regenerated every turn.
- `workspace/action_ledger.jsonl`: append-only machine-readable action/outcome log.
- `workspace/reflection.json`: latest structured reflection state.
- `workspace/curation_queue.json`: papers, gaps, and retries selected for re-curation.

### Audit artifacts

- `workspace/execution_log.py`: full jupytext percent-format operational trace.
- `output_dir/evidence_report.ipynb`: human-readable rendered notebook.
- `workspace/artifacts/`: raw outputs fetched on demand, such as failed extraction payloads, stderr, abstract dumps, or search result snapshots.
- `output_dir/token_usage.json` and `output_dir/token_usage_trace.json`: token accounting.

Rule: the planner should receive `workbook.md` by default, not `execution_log.py`. Raw detail is only pulled into context when a current decision depends on it.

## Exact Workbook Structure

The workbook should be a markdown-first notebook. In the rendered notebook, each top-level section below should map to a markdown cell. Code cells should not live inside the workbook - executed bash commands or short structured dumps are explicitly referenced by the workbook and stay in the execution log.

The workbook should remain stable across turns so the planner can compare state deltas reliably.

### Section 1. Claim Frame

Purpose: keep the claim decomposition, interpretation, and consensus criteria explicit and stable.

Required fields:

- input claim
- verdict label set and definitions
- target question: whether the literature overall supports, refutes, or leaves the claim uncertain
- entities and relation
- subclaims
- special interpretation notes
- consensus notes: what counts as adequate source diversity, contradiction handling, and claim-specific relevance constraints
- stopping rule summary: sufficiency threshold, remaining gaps tolerance, verdict emission condition
- extraction-context notes currently in force

Examples of interpretation notes:

- scope constraints that determine what counts as on-claim evidence
- contextual qualifiers that change whether evidence is relevant or out of scope
- alternative formulations or mechanism classes that should be treated as satisfying the same high-level claim

This section should prevent the planner from optimizing for single-paper support instead of consensus-level verification.

### Section 2. Guardrails

Purpose: make workflow constraints explicit to the agent without hardcoding most of them into orchestrator branches.

This section should absorb operational guidance that would otherwise live as scattered prompt requirements. These guardrails should be predetermined by the user or migrated during implementation from the current ProClaim prompt and workflow code. Once written into the workbook, they are read-only to the agent for the duration of a run.

Typical rules to keep here:

- do not rerun the same search query unless reflection records a changed rationale
- do not re-extract the same paper ID under the same claim frame and extraction-context version
- do not perform broad search immediately after an extraction failure unless reflection classifies the problem as retrieval
- do not repeat a full status narration when no material state change occurred
- do not consume the final turns on retrieval if verdict readiness is already high enough for `check_sufficiency` plus `emit_verdict`
- propose one next action only

These guardrails should be visible to the agent and enforced as workbook read-only policy, but the agent should not edit, relax, or rewrite them. Reflection may explain when an override condition has been met; it should not mutate the rule itself.

### Section 3. Header

Purpose: identify the run and make the planning frame explicit.

Required fields:

- claim
- run id / workspace path
- iteration number
- turns remaining
- current status derived from sufficiency state and turn budget, e.g. collecting-evidence (`sufficiency=insufficient`), ready-for-verdict (`sufficiency=sufficient`), forced-verdict (`turn budget low and sufficiency still insufficient`), complete (`verdict emitted`)
- workbook schema version

### Section 4. State Snapshot

Purpose: compact, high-signal state derived from `EvidenceState`.

Required fields:

- paper counts that are directly derivable today: total retrieved papers, papers with full text, and processed paper IDs from the extraction cache
- fact counts by stance
- source diversity summary only if explicitly computed; this is not currently a first-class field in `EvidenceState`
- conflict counts
- sufficiency history with confidence trend
- open gaps from the most recent sufficiency result

Future workbook bookkeeping can add richer paper lifecycle buckets such as zero-fact, filtered, retained, or re-curation candidates, but those are not currently persisted as explicit per-paper statuses in the implementation.

This section should be mostly tables and short bullet lists, not prose.

### Section 5. Action Loop Record

Purpose: capture the most recent local decision cycle in one compact place.

This section replaces separate ledger, observation, reflection, and next-action sections.

Keep only the last 3 to 5 meaningful action records. Each record should fit on a single compact block or table row.

Required fields:

- action id
- action purpose: search, extract, re-extract, feature-populate, sufficiency-check, reflect, curate, verdict
- target: query, paper ID, subclaim, or artifact handle
- short observation: what happened
- minimal state delta
- short diagnosis when progress was poor
- next step or retry rule
- raw artifact handle only when needed

Do not include full expectations, long rationales, or repeated status text. This is the main anti-repetition memory and local control surface for the next turn.

### Section 6. Recuration Queue

Purpose: keep stale evidence handling explicit.

Required buckets:

- papers to re-rank
- papers to re-extract
- filtered papers to revisit
- zero-fact papers needing claim-frame adjustment
- gaps and new subclaims needing targeted search
- contradictions needing resolution evidence

Each item should record why it is in the queue and what new framing or rationale justifies touching it again.

### Section 7. Verdict Readiness

Purpose: state clearly whether the system should continue searching or stop.

Required fields:

- current sufficiency label and confidence
- verdict readiness: yes/no
- blockers to verdict
- minimum additional evidence needed if not ready
- note on forced-verdict risk when turn budget is nearly exhausted

### Section 8. Raw Artifact Index

Purpose: point to detail without inlining it.

Examples:

- `artifact://search/query_003.json`
- `artifact://extract/pmid_12345678_failure.txt`
- `artifact://stderr/call_011.txt`
- `artifact://abstract/pmid_87654321.txt`

The planner may request one of these handles when it needs more detail.

## Workbook Skeleton

The workbook should follow a stable template such as:

```markdown
# ProClaim Workbook

## 1. Claim Frame
- Verdict labels: SUPPORT | REFUTE | UNCERTAIN
- Target question: whether the literature overall supports, refutes, or leaves the claim uncertain
- Subclaims:
- Consensus notes:
- Interpretation notes:
- Stopping rule:
- Extraction context in force:

## 2. Guardrails
- GR1:
	- source: migrated legacy rule
	- rule: do not rerun the same search query unless reflection records a changed rationale
	- override: reflection records a changed rationale
	- status: active

<!-- ---- workbook: stable | volatile ---- -->

## 3. Header
- Claim: 
- Workspace: 
- Iteration: 
- Turns remaining: 
- Status: 
- Schema version: workbook.v1

## 4. State Snapshot
| Metric | Value |
| --- | ---: |
| Papers retrieved |  |
| Papers with full text |  |
| Papers extracted |  |
| Zero-fact papers |  |
| Support facts |  |
| Refute facts |  |
| Uncertain facts |  |
| Conflicts |  |

- Sufficiency trend:
- Open gaps:

## 5. Action Loop Record
| Id | Type | Target | Observed | Delta | Diagnosis | Next | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |

## 6. Recuration Queue
- Re-extract:
- Search:

## 7. Verdict Readiness
- Sufficiency:
- Ready for verdict:
- Missing:

## 8. Raw Artifact Index
- artifact:
```

## ReAct-Style Orchestration Loop

The overall loop should keep the current bash-centric execution model but implement the same bounded cycle defined above: `Act -> Observe -> Reflect -> Curate`.

### Act step

This step includes both planning the next action and executing it.

Planner inputs:

- system prompt
- `workbook.md`
- optional requested raw artifact content

Planner output:

- one action only, consistent with the active workbook guardrails
- explicit expected observation
- explicit failure trigger for reflection or re-curation

Executor behavior:

- run the selected action through bash or a native tool
- write full raw output to audit artifacts
- summarize only the high-signal outcome into the workbook

### Observe step

- compare pre/post `EvidenceState`
- compute deltas for papers, facts, conflicts, sufficiency, and queue state
- update the action loop record and any affected queue or verdict-readiness sections

### Reflect step

Trigger reflection when:

- search returns no new papers worth retention
- extraction yields zero facts for relevant-looking papers
- sufficiency confidence stalls or drops
- the same action family is attempted repeatedly without gain
- contradictory evidence accumulates without narrowing the verdict

Reflection should produce a structured diagnosis and either:

- authorize a changed retry
- redirect to a different action family
- declare verdict readiness or forced-verdict posture

### Curate step

Run re-curation after new evidence, contradictions, or reflection.

Re-curation should:

- re-rank papers against the normalized claim
- re-open false-negative extractions when framing changed
- retire stale extraction-context notes
- revisit filtered papers when the interpretation changed
- prioritize evidence that improves consensus judgment rather than raw fact count

## Implementation Plan

### Phase 1. Introduce Workbook As Planning Surface (Done)

- add a workbook serializer derived primarily from `EvidenceState`
- add stable markdown section writers for the workbook schema
- stop feeding the full execution log back into the planner by default
- keep `execution_log.py` generation unchanged for audit

### Phase 2. Add Action Ledger And Observation Summaries (Done)

- persist concise action records in `action_ledger.jsonl`
- compute state deltas after each action
- update the compact `Action Loop Record` after every tool execution
- add a dedicated workbook `Guardrails` section initialized from user-specified policy or migrated legacy prompt/code rules
- add raw artifact handles instead of inlining large outputs

### Phase 3. Add Reflection And Recuration (Done)

- define structured reflection triggers and output schema
- define curation queue schema
- update the workbook after failed progress before the next planner turn
- allow retries only when a fixed guardrail's stated override condition is satisfied, such as reflection recording a changed rationale

### Phase 4. Tighten Planner Contract (Done)

- require one next action only
- require explicit expected observation and abort condition
- add verdict-readiness gating near the turn limit
- preserve forced-verdict fallback for exhausted budgets

### Phase 5. Evaluate (Done)

Implemented in `scripts/analysis/workbook_metrics.py` (tests in
`tests/test_workbook_metrics.py`). Reads the per-rep run artifacts
(`token_usage.json`, `workspace/action_ledger.jsonl`, `workspace/evidence_state.json`,
`workspace/verdict.json`, with a `results.csv` verdict fallback for pre-workbook
runs) and reports, per run and as a before/after comparison:

- compare token usage before and after workbook adoption
- measure reduction in repeated setup and repeated status narration
- measure action duplication rate
- track fact yield, source diversity, contradiction resolution, and verdict stability
- inspect hard claims for improved recovery from false-negative extraction
  (per-rep CSV export feeds claim-level inspection alongside `compare_runs.py`)

Usage:

```bash
# Single run
uv run python scripts/analysis/workbook_metrics.py \
    --run-dir results/signor_direct_eval_<TS>

# Before/after workbook adoption
uv run python scripts/analysis/workbook_metrics.py \
    --baseline-dir results/signor_direct_eval_<OLD_TS> \
    --workbook-dir results/signor_direct_eval_<NEW_TS>
```

## Success Criteria

- smaller per-turn planning context windows
- fewer repeated turns with no state change
- fewer redundant `setup_workspace(...)` calls in rendered notebooks
- clearer diagnosis of whether failures are due to retrieval, extraction, framing, or orchestration
- better recovery from zero-fact and false-negative extraction cases
- more deliberate consensus-oriented evidence curation
- more stable verdicts with clearer justification for `SUPPORT`, `REFUTE`, and `UNCERTAIN`