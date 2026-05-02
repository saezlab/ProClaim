# Direct Mode Performance Findings — 2026-04-22

**Branch:** `motivation`

## Summary

Reviewed the latest direct-mode SIGNOR evaluation at `results/baselines/ProClaim_ours/signor_direct_eval_20260419_135117` and traced the main cost drivers back to `evidence_programming_direct.py` and nearby evidence API surfaces. The current direct orchestrator is functionally working, but performance is dominated by repeated prompt replay, repeated `setup_workspace()` bootstrapping inside `bash` calls, and long-tail runs that keep iterating after progress has stalled. The largest improvements are control-flow and context-shaping changes rather than model swaps.

## New Files

| File | Purpose |
|------|---------|
| `.github/notes/2026-04-22_direct-mode-performance-findings.md` | Records direct-mode performance issues, quantitative evidence from the latest evaluation, and concrete improvement suggestions. |

## Architecture

```text
Current direct-mode hot path
============================

outer LLM call
   |
   v
serialize execution_log.py
   |
   v
send [system prompt + full refreshed user log]
   |
   v
tool call: bash python3 -c "... setup_workspace(...) ..."
   |
   v
reload evidence_state.json + rebuild LLM + run API function
   |
   v
append stdout/stderr back into execution_log.py
   |
   v
repeat until verdict.json exists or sufficiency threshold is met

Observed bottlenecks:
- prompt replay grows with the execution log
- Anthropic cache writes are large because the user message changes every turn
- each bash call pays Python startup + setup_workspace() overhead
- stalled runs can consume most of the max_iterations * max_turns budget
```

## Issues Found

| Area | Evidence | Impact |
|------|----------|--------|
| Prompt replay remains expensive | Evaluation summary shows 51.1M non-cache input tokens and 40.5M cache-write tokens across 100 claims. 13 runs exceeded 1M prompt tokens and 38 exceeded 500k. | High token cost and slower outer-loop latency. |
| Cache writes dominate cache reads | Direct mode marks both the system prompt and the refreshed user message as cacheable in `evidence_programming_direct.py`. Because the user message embeds a changing execution log, cache creation cost is repeatedly paid on large prefixes. | Cache pricing provides little benefit and was slightly worse than no-cache in the evaluated run. |
| Repeated workspace bootstrap in every bash call | Representative logs show `setup_workspace()` being re-run on nearly every turn. One cheap successful run did this 9 times; an expensive outlier did it 61 times. | Avoidable fixed overhead on every tool invocation. |
| Stagnation is not stopped early | `SIGNOR-175304/flip_False/rep_1` ran 73 outer calls and 74 tool calls before stopping. Large stretches of the run had 35 to 45 papers and 0 facts while the agent kept searching and re-checking. | Long-tail runtime and token burn without evidence gain. |
| Execution log is too raw for model context | The orchestrator serializes `execution_log.py` back into the next user message and also logs assistant free text and full tool output into the same log. | The model re-reads planning chatter and bulky tool output instead of a compact state summary. |
| Tool ergonomics encourage debugging loops | In `evidence_api.py`, `get_paper_text()` returns metadata plus abstract even when full text is already present on the paper record. One expensive run spent many turns rediscovering this mismatch while trying to diagnose failed extraction. | More outer-agent turns spent inspecting plumbing instead of progressing the claim. |
| Feature/sufficiency passes are not cheap on tail cases | One `populate_paper_features + check_sufficiency` command in the expensive outlier hit the 600 second subprocess timeout before later succeeding. | Tail latency spikes and rerun amplification. |
| Outer loop budget is generous relative to typical successful runs | Direct mode uses `max_calls = max_iterations * max_turns`. Defaults in config are `max_iterations=8` and `max_turns=30`, which allows up to 240 calls. Typical successful claims finished in roughly 8 to 11 calls. | Pathological runs have room to spend far more budget than normal runs need. |

## Key Design Decisions

- Prefer targeted control-flow changes over model replacement. The evaluation already shows that successful claims can finish cheaply with the current models; the main waste comes from how direct mode loops and replays context.
- Treat the execution log as an audit artifact, not as the primary model context. Keeping the full jupytext log is useful for reproducibility, but the outer model should receive a compact state summary rather than the raw transcript.
- Separate static and dynamic cache regions. The system prompt is a good cache candidate; the refreshed user message is not, because it changes every turn and forces large cache writes.
- Add explicit stagnation detection. Claims that add no papers, no facts, and no better sufficiency signal for consecutive rounds should terminate early or switch strategy instead of consuming the full call budget.
- Reduce subprocess reinitialization. A persistent Python worker or narrower first-class tools would remove repeated interpreter startup and `setup_workspace()` overhead while keeping direct mode independent of notebook MCP.

## Suggestions

1. Stop caching the changing user message. In src/pkevolve/verification/evidence_programming_direct.py#L422 to src/pkevolve/verification/evidence_programming_direct.py#L441, both the system prompt and the per-turn user message are marked cacheable. That user message contains the refreshed execution log, so every turn creates a huge new cache segment. For this workload, cache writes dominate reads and barely help cost. Cache only the static system prompt.

2. Replace raw execution-log replay with a compact state summary. The direct loop rebuilds context from the serialized log every call at src/pkevolve/verification/evidence_programming_direct.py#L238 and src/pkevolve/verification/evidence_programming_direct.py#L467. That gives the model a large, noisy transcript instead of the few state variables it actually needs: papers found, facts found, last sufficiency score, unresolved gaps, last tool results, and whether the search frontier changed. A structured summary would cut prompt tokens substantially and make stagnation easier to detect.

3. Do not append assistant free text and full bash output into the replay log. The code logs up to 2000 characters of assistant prose back into the notebook log at src/pkevolve/verification/evidence_programming_direct.py#L533, and bash stores the full command output in the log at src/pkevolve/verification/evidence_programming_direct.py#L300 before only truncating what is returned to the model. That means the next turn replays repetitive planning text and bulky tool output. Keep the full notebook artifact for auditability, but feed the model only a distilled summary of tool results.

4. Add stagnation-based early exit or strategy switching. The CRTC2→AKT1 outlier spent dozens of turns with 35 to 45 papers and 0 facts before finally recovering. The runner currently stops only on verdict existence or sufficiency score at src/pkevolve/verification/evidence_programming_direct.py#L322, while the loop budget is max_iterations × max_turns at src/pkevolve/verification/evidence_programming_direct.py#L462. Add a controller rule like: if two consecutive search or extraction rounds add no papers, no facts, and no higher sufficiency score, either terminate as uncertain or switch to a narrower fallback workflow. Most successful runs are finishing in 8 to 11 calls, so the tails are not pulling their weight.

5. Remove repeated setup_workspace calls by using a persistent worker or narrower tools. The current bash tool pattern re-runs Python startup and setup_workspace on nearly every turn. In the outlier that happened 61 times; even the cheap run did it 9 times. A long-lived Python worker, or direct first-class tools for search, extract, feature-populate, sufficiency-check, and verdict emit, would remove several seconds of fixed overhead per turn and reduce prompt verbosity because the agent would stop re-describing setup each time.

6. Tighten stopping immediately after sufficiency is reached. In the outlier log, once facts were finally added, the run still kept spending turns even after the classifier reported confidence 1.0. Given the existing post-tool stop check at src/pkevolve/verification/evidence_programming_direct.py#L569, that suggests either stale persisted state or a workflow path where the sufficient result is not reliably written before the next call. This is worth fixing because it directly burns tokens after the decision is already made.

6. Fix the paper-text tool ergonomics so the agent does not debug retrieval mid-run. In src/pkevolve/verification/evidence_api.py#L932, get_paper_text always returns metadata plus abstract, even if full text has already been fetched. The expensive CRTC2→AKT1 run spent many turns discovering exactly that mismatch. Even if extraction uses a different path internally, this tool shape encourages unproductive debugging loops. Make get_paper_text prefer paper.full_text when present, or expose a get_best_paper_text tool that always returns the richest available text.

8. Make feature population and sufficiency checking incremental. In the same outlier, one populate_paper_features plus check_sufficiency command hit the 600 second timeout. That is a strong sign those steps are recomputing too much state. Cache per-paper features and only recompute for new or changed papers and facts.
