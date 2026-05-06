# Per-Call Context Refresh — 2026-04-15

**Branch:** `feature/ctx-manage`

## Summary

Replaced the nested outer/inner loop architecture in `evidence_programming_direct.py` with a single flat `while` loop that rebuilds LLM messages from the execution log before every API call. Instead of accumulating conversation history across turns (O(n²) token growth), each call receives only `[system_prompt, execution_log]`. Tool results flow through the jupytext execution log — the single source of truth — and messages are discarded after each call. Also fixed `dispatch_tool()` to log `read_file` results to the execution log (previously lost between calls). Validated with a test run on the SIGNOR claim "GNAS directly activates ADCY1": same verdict (REFUTE, 0.80 confidence) at 79% lower cost and 44% faster wall clock vs the per-iteration baseline.

## Modified Files

| File | Change |
|------|--------|
| `src/proclaim/verification/evidence_programming_direct.py` | (1) Updated module docstring from "per-iteration" to "per-call" context refresh. (2) Rewrote `verify_claim_direct()` main loop from nested `for iteration / while tool_call_count` to single flat `while call_count < max_calls`. First call gets initial prompt; subsequent calls wrap execution log in `<execution_log>` tags. No `messages.append()` — context rebuilt fresh each call. `for/else` pattern logs "Reached max calls" on budget exhaustion. (3) Fixed `dispatch_tool()` `read_file` handler to append results to execution log via `append_to_jupytext_log(log_path, f"cat {fpath}", truncated)` — previously returned content directly, losing it on the next per-call refresh. Renamed `path` to `fpath` to avoid shadowing the stdlib. |

## Architecture

```
                        Per-call context refresh
                        ========================

  while call_count < max_calls:
      ┌────────────────────────────────────────────────────┐
      │  1. Serialize execution log from disk              │
      │  2. Build messages = [system_prompt, exec_log]     │
      │  3. litellm.completion(messages, tools)            │
      │  4. Dispatch tool calls → bash subprocess          │
      │  5. Append tool results to execution log           │
      │  6. Discard messages (no accumulation)             │
      │  7. Check stopping condition                       │
      └────────────────────────────────────────────────────┘

  Token flow per call:

  ┌──────────────┐   cache hit    ┌──────────────────┐
  │ System prompt │ ─────────────→│  Anthropic API   │
  │   (~4.2k tok) │  (ephemeral)  │                  │
  └──────────────┘                │  Only new suffix │
  ┌──────────────┐   cache create │  of exec log is  │
  │ Execution log│ ─────────────→│  billed at full  │
  │  (growing)   │  (ephemeral)  │  rate             │
  └──────────────┘                └──────────────────┘

  Before (per-iteration):         After (per-call):
  ┌─────────────────────┐         ┌─────────────────────┐
  │ outer loop (iters)  │         │ while call < max:   │
  │  ┌────────────────┐ │         │   log = read(disk)  │
  │  │ inner loop     │ │         │   msgs = [sys, log] │
  │  │ messages.append│ │    →    │   resp = llm(msgs)  │
  │  │ (O(n²) tokens) │ │         │   dispatch(tools)   │
  │  └────────────────┘ │         │   # msgs discarded  │
  │  context refresh    │         └─────────────────────┘
  └─────────────────────┘         Total: O(n) tokens
```

## Key Design Decisions

- **Execution log as single source of truth.** All tool results must be written to the jupytext execution log, since messages are discarded after each call. The `read_file` handler was missing this — content was returned to the LLM but never logged, so it would vanish on the next call. Fixed by adding `append_to_jupytext_log()` in the `read_file` branch of `dispatch_tool()`.

- **Flat loop instead of nested loops.** The original architecture had an outer `for iteration` loop with context refresh and an inner `while tool_call_count` loop that accumulated messages. The per-call refresh makes the inner loop unnecessary — every call is identical: read log, build messages, call LLM, dispatch, check stop. A single `while call_count < max_calls` with `max_calls = max_iterations × max_turns` as safety budget.

- **First call vs subsequent calls.** Call 0 sends the initial verification prompt (no log yet). All subsequent calls send the execution log wrapped in `<execution_log>` tags with a continuation instruction. This is a simple `if call_count == 0` branch.

- **Cache-creation dominates cache-read.** With per-call refresh, the system prompt (4.2k tokens) is consistently cache-read, but the execution log grows each call so its new suffix requires cache-creation. Cache-read rate dropped from 58% to 26%, but total cost still dropped 79% because total tokens across all calls is dramatically lower (no duplicated conversation history).

## Test Results

### GNAS → ADCY1 claim comparison

| Metric | Baseline (per-iteration) | Per-call refresh | Δ |
|--------|------------------------:|------------------:|---|
| Verdict | REFUTE (0.80) | REFUTE (0.80) | Same |
| Cost (agent LLM) | $0.501 | $0.106 | **−79%** |
| Wall clock | 12m 25s | 6m 55s | **−44%** |
| API calls | 17 | 13 | −4 |
| Total tokens | — | 203,277 | — |
| Prompt tokens | 272k | 195,553 | −28% |
| Cache read % | 58% | 25.8% | — |
| Cache creation % | — | 74.2% | — |

### Token usage breakdown (per-call run)

| Bucket | Tokens | % of prompt |
|--------|-------:|------------:|
| Cache read | 50,412 | 25.8% |
| Cache creation | 145,115 | 74.2% |
| Uncached input | 26 | ~0% |

### Per-call cost progression

| Call | Cum. cost | Agent action |
|-----:|----------:|-------------|
| 1 | $0.005 | Setup workspace |
| 2 | $0.008 | Define 5 subclaims |
| 3 | $0.012 | PubMed + S2 search (15 papers) |
| 4 | $0.015 | Extract facts (44 facts from 15 papers) |
| 5 | $0.031 | Features + filter + sufficiency (conf=0.025) |
| 6 | $0.044 | Gap search + S2 recommendations (8 papers) |
| 7 | $0.048 | Extract from 8 new papers (7 facts) |
| 8 | $0.063 | Sufficiency check (still 0.025) |
| 9 | $0.071 | Final targeted search (16 papers) |
| 10 | $0.076 | Extract from 10 new papers (39 facts) |
| 11 | $0.086 | Final sufficiency (conf=0.004) |
| 12 | $0.097 | Final targeted search iteration 4 (0 new) |
| 13 | $0.106 | `emit_verdict(REFUTE, 0.80)` |

## Bug Fixes

- **`read_file` results lost between calls.** In per-call refresh, messages are discarded after each call. The `read_file` handler in `dispatch_tool()` returned file content to the LLM but did not write it to the execution log. On the next call, the log would not contain the file content, making it invisible. Fixed by adding `append_to_jupytext_log(log_path, f"cat {fpath}", truncated)` and logging errors similarly.
