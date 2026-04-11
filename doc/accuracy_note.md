# Accuracy Note — companion to `cost_attribution_report.md`

**Context.** The cost attribution report analyses a single-claim run
(`SIGNOR-156958`, rep 1) costing $1.51. That report is framed around
dollars. This note re-reads the same evidence with **accuracy** as the
objective, and sets the guiding principles for the fix branch that will
be cut from commit `a4e2312` (the last stable single-`query()`
architecture, before the per-iteration refresh experiments).

The fix branch inherits this note and the cost report together. Other
docs in the current working tree that discuss per-iteration query
refresh, notebook serialisation, or paper-notes amnesia mitigation are
**not** to be carried over — they address problems that only exist in
the per-iteration architecture.

---

## TL;DR

The cost problems in the report are not a separate concern from
accuracy. They are the same problem viewed on a different axis: the
orchestrator's context is being filled with **low-signal noise** while
the high-signal evidence is competing with it for attention.

Sources of noise identified in the cost report:

- vLLM 404 retry loop output (caused by the `./qwen3.5-9b` vs
  `qwen3.5-9b` model-id mismatch),
- per-paper progress prints from `extract_and_add_facts`,
- spaCy / HTTP / retry instrumentation in `print()` stream,
- **whole cells where the agent manually re-derives signals that the
  sufficiency classifier already computed** — this is the single
  biggest accuracy-side symptom and is a direct consequence of the next
  principle.

Shrinking the noise and stopping the agent from re-deriving downstream
state improves accuracy and cost at the same time. There is no
trade-off here; the two fixes are the same fix.

---

## Core principle: do not patch downstream, fix upstream

The single most important behavioural rule for the orchestrator — and
the one that must be enforced in the system prompt:

> When `check_sufficiency` returns low confidence, or when
> `extract_and_add_facts` returns zero facts on a paper you expected to
> help, the correct response is **never** to debug the classifier or
> the extractor. The correct response is to assume that the search
> missed the right papers, or that the subclaim is phrased with the
> wrong vocabulary, wrong synonyms, or wrong level of granularity.

Downstream components (MLP sufficiency classifier, LLM fact extractor,
NLP feature pipeline) are the **measurement layer**. If the measurement
says "insufficient", the fix belongs to what is being measured — the
corpus and the subclaim phrasing — not to the instrument. An
orchestrator that debugs its own instruments loses calibration and
inflates context with noise.

### Symptoms → correct responses

| Symptom | Wrong response (observed in the cost-debug run) | Right response |
|---|---|---|
| `check_sufficiency` returns `prob < threshold` | Print NLP features, inspect top-20 best chunks, compute NLI scores manually, dump titles+abstracts | Re-decompose subclaims; try synonyms; broaden/narrow granularity; re-search |
| `extract_and_add_facts` returns 0 for a paper | Call `get_paper_text` on the same paper, inspect the text manually, try to spot a fact by eye | Record that this paper does not carry the evidence; search elsewhere |
| Entity coverage low for a subclaim | Manually print entity lists, guess synonyms in agent code | Reformulate the subclaim (synonym expansion, granularity shift) and search with the new wording |
| Semantic similarity low across the pool | Hand-pick papers by title; read abstracts in a loop | Generate a reformulated PubMed query with alternative entity names / MeSH terms |

The common pattern: the *right* response always loops back to **search
reformulation**, never forward into the classifier internals.

### Why this must live in the system prompt

The agent's behaviour in iteration 3 of the cost-debug run
(§4 of the report, turns 9–16) shows it was not being lazy or
careless — it was being *rational* given the information it had. The
classifier returned `prob=0.6682` with a raw feature dump; the agent
had no actionable gap description; the only tools it could reach for
were notebook cells that print and inspect. Telling a future agent
"stop debugging the classifier" without giving it an actionable
alternative will not work. The system prompt must:

1. State the principle above (upstream, not downstream).
2. Name the concrete upstream knobs it *should* turn: subclaim
   re-decomposition, synonym expansion, granularity shift,
   reformulated search queries.
3. Forbid, by name, the downstream knobs it should *not* turn:
   inspecting MLP features, inspecting NLI scores, inspecting best
   chunks, reading paper full text in order to second-guess the
   extractor.

---

## Concrete evidence from the cost report

### Iter 3's 16.9K output tokens are the agent debugging its instruments

From cost report §2, per-iteration output tokens are
`3.5K / 5.5K / 6.2K / 16.9K`. Iteration 3 alone produces nearly as many
output tokens as iterations 0–2 combined. From §4, the iter-3 cells
responsible are:

- "Print titles and abstracts for the most semantically similar papers" — 8,895 chars
- "Check NLP feature structure" — 7,606 chars
- "Check GNAS AND ADCY1 papers specifically (from targeted search)" — 6,466 chars
- "Look at key papers specifically about type I adenylyl cyclase" — 5,508 chars
- "Inspect best-chunk texts and NLI scores" — 3,500 chars

Every one of these is a downstream-patching cell. The trigger is
always the same: sufficiency prob below threshold, no actionable gap
in the classifier's return value, so the agent improvises by
recomputing the classifier's inputs in a notebook cell and trying to
interpret them by eye.

The fix is structural, not behavioural. Giving `check_sufficiency` an
actionable return shape — for example

```
SufficiencyResult(
    prob=0.67,
    label="insufficient",
    diagnosis="Subclaim 2 ('GNAS activates type I adenylyl cyclase') "
              "has entity coverage 0.12. The retrieved corpus mentions "
              "GNAS broadly but few papers specifically link it to ADCY1.",
    suggested_action="reformulate_search",
    candidate_reformulations=[
        "Try 'Gs-alpha subunit' in place of 'GNAS'",
        "Try 'ADCY1' in place of 'type I adenylyl cyclase'",
    ],
)
```

— collapses all of the above cells into one turn: read the diagnosis,
call `search_for_gap` or reformulate the subclaim, re-run the loop.
Zero manual feature inspection. Zero accuracy risk from the agent
reading abstracts and guessing.

### Tool-result noise is half of the orchestrator's context

Cost report §4's top three tool results are ~12K chars each, all from
`extract_and_add_facts`. They contain:

- `extract_and_add_facts: processing N papers with 8 workers`
- Per-paper `no full text, using abstract` lines
- Per-paper `Layer 1b retrieved N chars` lines
- vLLM 404 retry loops (see next section)

This is instrumentation output. It belongs in `logger.info` (captured
by `run.log` for post-hoc debugging), not in `print()` (captured by
`nb_execute` tool results, which get replayed on every subsequent
model turn within the query). A single 3K-token verbose tool result in
a 50-turn run costs ~50× its own size in attention footprint — and
every one of those replays is attention the agent is *not* spending on
the facts that decide the verdict.

The verbosity reduction is therefore an accuracy fix before it is a
cost fix.

### The model-id bug is the biggest accuracy hazard, not a footnote

Cost report §6 flags `./qwen3.5-9b` vs `qwen3.5-9b` as a "data-quality
caveat" with low cost impact. From the accuracy perspective it is the
most dangerous finding in the report:

- Iteration 0 fact extraction returned 0 on most papers (every
  extraction call hit `404: model './qwen3.5-9b' does not exist`).
- Iterations 1 and 2 partially recovered via retries with corrected
  model ids in later cells.
- The 15 facts that ultimately grounded the SUPPORT verdict mostly
  came from iteration 3.

This run reached SUPPORT at 0.95 confidence, but that outcome was
lucky. Any run where iter 3 does not happen to hit productive papers
(because the search was mis-phrased, or because the agent ran out of
iterations) will produce UNCERTAIN on a claim that should have been
SUPPORT — not because the orchestrator is bad, but because the
instrument it depends on was broken for the first half of the run.

**Nothing else in this note should be measured until the model-id
mismatch is fixed.** Any accuracy or cost experiment run against a
broken fact extractor is contaminated by the retry loops and the
partial recovery path.

---

## Prioritised changes for the fix branch

In the fix branch that will be cut from `a4e2312`, apply these changes
in order. Priorities are set by accuracy impact, not cost.

### P0 — block experiments until done (done)

1. **Fix the `./qwen3.5-9b` model-id mismatch.** The YAML config and
   the vLLM `--served-model-name` must agree. Until they do, every
   accuracy or cost measurement is contaminated by 404 retry loops in
   the fact extractor. This is non-negotiable and must land first.

2. **Verbosity reduction in the evidence API.** Move per-item progress,
   HTTP request logs, retry messages, and spaCy warnings from
   `print()` to `logger`. Keep `print()` only for the single-line
   result summary of each top-level call (e.g. `extract_and_add_facts:
   done. facts=N papers=M`). Files to sweep:

   - `evidence_api.py` (search, extract, features, sufficiency)
   - `subagents.py`
   - `feature_tools.py`
   - `full_text.py`

   The signal-to-noise problem dissolves most of the apparent
   "context bloat" identified in the cost report, and it does so
   without throwing away any information that the agent actually needs
   — everything moved to `logger` is still in `run.log` for humans.

### P1 — structural accuracy fix

3. **Upgrade `identify_gaps` to emit actionable guidance, not
   `check_sufficiency` to emit feature scores.** The orchestrator is
   not supposed to interpret MLP features directly — that is exactly
   the downstream-patching pattern this note is trying to kill.

   Keep `check_sufficiency` as-is: a calibrated numeric measurement
   (probability + MLP feature values). It is the instrument, and the
   instrument should not change its dial.

   Move the "translation to action" layer into `identify_gaps` (the
   existing gap-identification subagent). When `prob < threshold`,
   the orchestrator calls `identify_gaps(state)` and receives, for
   example:

   - `"PMIDs [27492469, 9651336] have semantic similarity > 0.5 but
      no facts extracted yet — prioritise extract_and_add_facts on
      these before searching for more papers."`
   - `"Subclaim 2 ('GNAS activates type I adenylyl cyclase') has
      entity coverage 0.12 across the pool. Try reformulating with
      'Gs-alpha subunit' and 'ADCY1' as alternative entity names."`
   - `"No papers in the pool link GNAS to ADCY1 directly — only
      indirect mechanism papers. Search for the direct interaction
      with a narrower query."`

   Each of these is a concrete next action the orchestrator can
   execute without inspecting any MLP feature, NLI score, or best
   chunk itself. The subagent is allowed to read features; the
   orchestrator is not. This is the clean upstream/downstream split
   the core principle demands.

   The current `identify_gaps` output is the root cause of the
   iteration-3 debugging spiral documented above: it returned gap
   descriptions too vague to act on, the orchestrator saw only a raw
   MLP probability, so it improvised. Rich, PMID-grounded gap
   descriptions collapse that spiral into a single turn.

4. **Encode the upstream-fix principle in the system prompt.** Add a
   CRITICAL section that states the principle (see "Core principle"
   above), names the upstream knobs the agent *should* turn, and
   names the downstream inspections it must *not* perform. Reference
   the concrete cells from the cost-debug run as examples of what not
   to do, if space permits.

5. **Early exit on sufficiency plateau.** If confidence changes by
   less than 0.05 across two consecutive iterations, stop and emit
   the verdict with the evidence on hand. Continuing to loop after
   the search space is exhausted at the current phrasing only
   accumulates noise — the agent has no new signal to work with and
   will fall back to downstream patching (per the iter-3 pattern).

6. **Remove downstream-patching attractors from the orchestrator's
   toolset.** System-prompt rules (P1 item 4) tell the agent *not*
   to do something; removing the tool *makes it impossible*. The
   tool cannot be called if it does not exist. This is the
   structural enforcement of the core principle and should be
   preferred over prompt-level enforcement wherever the tool has no
   legitimate orchestrator use case.

   An audit of the 27 functions currently listed in `function_docs`
   (22 in `evidence_api`, 5 in `subagents`) identified **10 tools
   that should be removed from orchestrator exposure**. They fall
   into three categories.

   #### Category A — dead code (remove entirely)

   These have **zero callers anywhere in the codebase**. They were
   designed as orchestrator-facing tools and never wired into any
   internal flow. Exposing them is pure attractor surface — the
   orchestrator is the only thing that can call them, and the only
   reason the orchestrator *would* call them is to patch downstream
   state by hand.

   | Tool | What it does | Why it is an attractor |
   |---|---|---|
   | `update_synthesis` | Writes a free-form synthesis narrative per subclaim into state | Synthesis should come from a subagent operating over `state.facts`, not from the orchestrator writing prose by hand. |
   | `add_conflict` | Declares that two specific facts are in conflict | Conflict detection should be automatic on fact addition, not a manual orchestrator judgment. |
   | `synthesize_subclaim` (subagent) | LLM call to synthesise a subclaim summary from facts | Never called. If it is needed, it should run automatically when facts are added, not be triggered by the orchestrator. |
   | `detect_conflicts` (subagent) | LLM call to find conflicts between facts | Same pattern — never called, exposing it invites the orchestrator to "go find conflicts" by hand. |

   Action: delete from `function_docs`. Keep or delete the function
   definitions themselves based on whether a future automatic flow
   will use them — but they are not orchestrator tools.

   #### Category B — has legitimate internal caller, just hide from orchestrator

   These functions are used by the extractor / sufficiency path
   internally and must stay in the module. They should become
   private helpers (underscore prefix, or simply dropped from the
   `function_docs` export list). The orchestrator gets the batched /
   automatic version; the single-paper / raw-access version stays
   internal.

   | Tool | Internal caller | What changes |
   |---|---|---|
   | `get_paper_text` | `_extract_and_add_facts_single` fallback | Rename to `_get_paper_text`; drop from `function_docs`. Orchestrator that wants to "see" a paper reads `state.papers[pmid].abstract` directly. |
   | `get_full_text_article` | `_extract_and_add_facts_single` primary path | Rename to `_get_full_text_article`; drop from `function_docs`. Orchestrator never needs raw full text — the extractor consumes it. |
   | `add_facts_from_dicts` | `extract_and_add_facts` at [evidence_api.py:1180](src/pkevolve/verification/evidence_api.py#L1180) | Rename to `_add_facts_from_dicts`; drop from `function_docs`. The system prompt already says "NEVER call this with manually written text" — the existence of the tool is the only reason that rule has to exist. |
   | `extract_facts` (subagent, single-paper) | `_extract_and_add_facts_single` in `evidence_api.py:1157` | Drop from `function_docs`. The batched `extract_and_add_facts` is the only extraction entry point the orchestrator should see. Exposing the single-paper version contradicts the system prompt's "never loop over PMIDs" rule and makes that rule prompt-level only. |

   Action: for each, drop from `_api_funcs` / `_sub_funcs` in
   `function_docs`. Optionally rename with underscore prefix to make
   the private-helper status explicit in the code.

   #### Category C — architecturally mis-placed

   These two are not strictly downstream-patching but are still
   wrong to expose to the orchestrator.

   - **`compress_evidence`** — lets the orchestrator decide when to
     compress its own working memory. Memory management is infra,
     not orchestrator judgment. If and when compression is needed,
     it should be triggered automatically by a token-budget check in
     the loop, not by an agent deciding "I feel full". An
     orchestrator that calls `compress_evidence` is effectively
     throwing away state it cannot interpret, based on its own
     estimate of its own confusion — a textbook case of downstream
     patching applied to its own context.

   - **`populate_paper_features_parallel`** — duplicate of
     `populate_paper_features` exposed separately. Only one feature
     entry point should be visible. Merge by having
     `populate_paper_features` auto-switch to the parallel path when
     the paper count crosses a threshold (it already does, based on
     the cost-debug run's "Auto-switching to parallel feature
     computation" log line).

   Action: remove `compress_evidence` from `function_docs` and move
   its trigger into the loop infrastructure. Remove
   `populate_paper_features_parallel` from `function_docs` and rely
   on the auto-switch inside `populate_paper_features`.

   #### Net effect

   `function_docs` shrinks from 27 tools to 17. The 17 that remain
   are all upstream (search / reformulation), measurement
   (`check_sufficiency`), read-only (`get_evidence_summary`,
   `get_sufficiency_history`), or required workflow (`emit_verdict`,
   `populate_paper_features`, `filter_papers_by_stance`,
   `extract_and_add_facts` batched).

   No orchestrator-visible path remains for:
   - reading raw paper text or abstract content,
   - writing facts, synthesis, or conflicts by hand,
   - manually inspecting or recomputing classifier inputs,
   - triggering memory compression on a hunch.

   Every one of those actions was observed in the iteration-3
   debugging spiral. Removing the tools makes the spiral
   structurally impossible instead of merely discouraged.

   #### Special case: `identify_gaps` is correctly unexposed

   [`subagents.py:314`](src/pkevolve/verification/subagents.py#L314)
   defines `identify_gaps` but it is **not** in the exposed
   `_sub_funcs` list — correctly. It is called internally by
   `check_sufficiency` (at lines 1777, 1856, 1939 of `evidence_api.py`).
   This is the right shape: the orchestrator calls
   `check_sufficiency`, which calls `identify_gaps` under the hood,
   which returns gap objects embedded in the sufficiency result.

   The problem is not the wiring — the problem is that
   `identify_gaps` currently receives only `(llm, claim, subclaims,
   facts)`. It cannot see the paper pool, the NLP features, or the
   MLP feature values, so it cannot produce PMID-grounded actionable
   gaps like "PMIDs [X, Y] have semantic similarity > 0.5 but no
   facts extracted yet — prioritise these." It can only say "subclaim
   2 needs more evidence", which is too vague for the orchestrator
   to act on without its own inspection — and that inspection is
   exactly the iteration-3 debugging spiral.

   This is the concrete landing point for P1 item 3. Extend
   `identify_gaps`'s signature to include the paper pool and NLP
   feature pointers, and enrich its prompt to emit PMID-specific and
   reformulation-specific gap descriptions. No orchestrator change
   is needed — the orchestrator already receives `identify_gaps`'s
   output via `check_sufficiency`.

### P2 — defer unless P0/P1 fall short of the accuracy target

6. Trim duplicated verdict definitions and redundant workflow rules
   from the system prompt (low risk, modest attention savings).
7. Collapse superseded `nb_render_*` HTML tables in the cached
   conversation so old tables don't occupy attention alongside the
   fresh ones.

---

## Out of scope for this note and this branch

The following fixes are explicitly **not** recommended, and should not
be bundled into the same branch:

- **Per-iteration `query()` refresh with compact state summary.** This
  is the architecture we are stepping back from. Compact summaries
  throw away continuity signal that a single-`query()` architecture
  gets for free — the conversation history *is* the continuity.

- **"One tool call per turn, merge search+extract+features+sufficiency
  into a single cell."** Merging steps breaks incremental error
  recovery (any one sub-step failing takes down the whole cell), which
  matters more under accuracy than the turn-count savings matter under
  cost. Incremental notebook cells are a feature, not a bug.

- **Hard-capping `max_iterations` from 4 → 2.** The cost-debug run's
  first two iterations were crippled by the model-id bug and
  contributed almost no grounding. Accuracy would collapse on any
  run whose productive iteration happens to be the third or fourth.
  Prefer the plateau-based early exit (P1 item 5) instead.

### Deferred — flagged for a future branch, not for this one

- **Selective notebook serialisation** — only relevant if a future
  branch ever revisits per-iteration `query()` refresh. If it does,
  the serialisation strategy should be **surgical, not wholesale**:
  keep every cell's code (the logic path) and its final return value
  (the outcome), drop intermediate progress logs, HTTP retry output,
  spaCy warnings, and any stdout list that runs beyond ~10 lines.
  The agent needs to remember "I ran this query and got these 5
  facts", not "there were 3 HTTP retries during the search".

  This is strictly better than the "replace serialisation with a
  compact state summary" fix suggested in the cost report — it keeps
  continuity signal while removing noise. But it only applies in the
  per-iteration architecture, so it is out of scope for the fix
  branch cut from `a4e2312`. Flagged here so that if the
  per-iteration architecture is ever revisited, this is the entry
  point, not the cost report's suggestion.

Any of the above can be revisited after the P0/P1 changes land and a
clean accuracy measurement exists. None of them are safe to adopt on
faith based on the cost report alone.

---

## Cross-reference

- `doc/cost_attribution_report.md` — the source cost analysis this note
  reinterprets. Carry it over to the fix branch alongside this note.
