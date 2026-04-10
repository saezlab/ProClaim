# Notebook-as-Context: Cost Reduction via Context Refresh

Design for replacing cumulative conversation history with notebook
injection at each evidence iteration, inspired by the ACE (Agentic
Context Engineering) paradigm.

Supersedes the hybrid recommendation in `repl_vs_directory_ace_analysis.md`
and the passive-notebook position in `REPL_issues.md`.

---

## Problem

The current evidence programming agent uses the Claude Agent SDK, which
accumulates the full conversation history in context: every `nb_execute`
call, its code input, its output, every `nb_markdown` cell, and every
agent reasoning turn.  Over 4–8 iterations of search → extract →
features → sufficiency, this grows to 80–120K tokens.

Cost is proportional to context size × iterations.  Each new agent turn
re-reads the *entire* history.

---

## Proposal

At the start of each evidence iteration, **reset the agent context** and
inject two components:

1. **Cached prefix** — system prompt, tool definitions, instructions,
   claim.  Static across iterations.  Eligible for prompt caching
   (Anthropic cache read pricing ≈ free).

2. **Current notebook content** — all code cells (with their stdout) and
   markdown cells, serialized as text.  This is the agent's working
   memory.

The agent does **not** see its own prior conversation turns.  It sees
the notebook it has been building, which contains richer grounding:
actual Python calls, printed outputs, and its own markdown reasoning
notes.

---

## Cost Model

| Segment | Tokens | Cached? | Per-iteration cost |
|---------|--------|---------|-------------------|
| System prompt + tool defs + instructions + claim | ~4–6K | Yes (after first call) | ≈ 0 (cache read) |
| Notebook content (grows per iteration) | ~5–25K | Partially (prefix-stable) | Input rate on delta |
| Agent output (new cells + reasoning) | ~2–4K | No | Output rate |

**Current (cumulative history):** iteration N pays for all N prior
turns.  Tokens grow quadratically with iterations.

**Proposed (notebook injection):** iteration N pays for the notebook
size at iteration N.  Tokens grow linearly with iterations.

Estimated 4–5× context reduction at iteration 4.

---

## Why the Notebook Is Sufficient Context

### Search history

The notebook cell:

```python
pmids = search_pubmed_llm(state.claim, state, llm)
# stdout: Generated query: "MAPK1 AND phosphorylation AND H3" → 8 new papers
```

is a richer search log than any structured `search_log` field.  It shows
the function called, the query generated, the result count.  The agent
reads this cell at the start of the next iteration and knows not to
re-run it.

### Gap search attempts

Similarly, gap-targeted searches appear as cells:

```python
search_for_gap("kinase substrate specificity for H3-3A", state)
# stdout: 3 new papers added
```

No separate `gap_search_attempts` state field needed.

### Reasoning continuity

The agent writes `nb_markdown` cells explaining its reasoning between
steps.  These survive context refresh and serve as the reasoning trace.

### Sufficiency trajectory

`get_sufficiency_history(state)` prints the confidence trend and returns
`{"trend": "improving" | "flat" | "declining"}`.  Together with
`get_evidence_summary(state)`, the agent can reconstruct the full
status from `EvidenceState` alone.

---

## Safety: Why Re-execution Is Not a Problem

`EvidenceState` has dedup guards at every mutation point:

| Operation | Guard |
|-----------|-------|
| Paper addition | Dedup by PMID and DOI (`_s2_paper_to_record`, `search_pubmed`) |
| Fact extraction | `extracted_pmids` list skips already-processed papers |
| Fact addition | Dedup by `(text.lower(), source_pmid)` |
| Feature population | `populate_paper_features` is idempotent (skips papers with existing features) |
| Sufficiency check | Appends to `sufficiency_history` — safe to re-run (returns new result) |

Even if the agent *does* re-issue a call, the data layer is protected.
The only waste is LLM cost for `search_pubmed_llm` (1 LLM call to
generate a query) — but since the agent can see its prior search cell in
the notebook, it should not re-issue it.

---

## ACE Analogy

ACE's Generator receives:

    playbook (evolving artifact) + current question → answer

Our proposed agent receives:

    notebook (evolving artifact) + continuation prompt → next iteration

Key parallel: the agent does **not** need its prior conversation turns.
It needs the *artifact it has been building*.

### Difference from ACE

ACE is multi-episode: the playbook improves across tasks.  Our notebook
is single-episode: it tracks one claim's verification.  But within a
single verification, the dynamics are the same — state accumulates in a
persistent artifact, and the LLM context resets with that artifact
injected.

ACE is also stateful due to the Reflector: each round's reflection feeds
back into the Curator, which updates the playbook.  Our sufficiency
classifier is the analogous Reflector — it produces structured feedback
(gaps, confidence, trend) that the agent uses to decide the next search
action.

---

## Notebook Growth Management

A typical 4-iteration verification produces ~15–20 cells (~15–25K tokens).
At the 8-iteration cap, this may reach ~40–50K tokens.

### Mitigation (if needed)

1. **Collapse old render outputs.**  `nb_render_*` cells produce large
   HTML tables.  After iteration N+2, replace old render cell outputs
   with a one-line summary: `"[Papers table: 12 papers at iteration 1]"`.
   The *code* (what was called) is preserved; only verbose output is
   trimmed.

2. **Truncate early cell stdout.**  Keep the code (the search log) but
   trim stdout for cells older than N iterations.  The code line
   `search_pubmed_llm(state.claim, state, llm)` tells the agent what
   was done; the full paper listing is less useful 3 iterations later.

3. **Sliding window + summary.**  Inject `get_evidence_summary()` output
   as a preamble, then only the last N iterations' cells verbatim.

Option 1 is the natural starting point.  Options 2–3 are fallbacks if
high-iteration claims hit context limits.

---

## Implementation Sketch

### Changes to `evidence_programming.py`

Replace the single `async for message in query(prompt, options)` loop
with an iteration-level loop:

```python
for iteration in range(max_iterations):
    notebook_text = serialize_notebook_as_text(notebook_path)
    messages = [
        {"role": "user", "content": f"{notebook_text}\n\n{continuation_prompt}"}
    ]
    async for message in query(prompt=messages, options=options):
        # ... process tool calls as before
    if verdict_emitted(workspace):
        break
```

The `options` object (system prompt, tool defs, MCP servers) is
unchanged and cacheable.

### New utility: `serialize_notebook_as_text()`

Read the `.ipynb`, emit each cell as:

```
## [Cell N: markdown]
<cell content>

## [Cell N: code]
```python
<code>
```
### Output:
<stdout, truncated to limit>
```

Plain text — no JSON overhead, no HTML rendering artifacts.

### Changes to `notebook_mcp.py`

None.  The MCP tools continue to work identically.

### Changes to `EvidenceState`

None.  No new fields needed.

---

## What This Does NOT Change

- **Subagent costs** — fact extraction (`extract_and_add_facts`), gap
  identification (`identify_gaps`), query formulation
  (`formulate_gap_queries`) use the subagent LLM.  These costs are
  per-paper/per-gap, independent of outer context size.

- **MLP classifier** — runs locally, no token cost.

- **Search APIs** — PubMed, Semantic Scholar.  No token cost.

- **Notebook as audit trail** — the notebook continues to be saved as
  `.ipynb` for human review.  The serialization for context injection is
  a read-only view.

---

## References

- `doc/repl_vs_directory_ace_analysis.md` — ACE alignment analysis
- `doc/REPL_issues.md` — prior analysis of notebook as passive view layer
- `doc/sdk_vs_repl_modes.md` — SDK vs REPL architecture
- Zhang et al. "Agentic Context Engineering." ICLR 2026.
