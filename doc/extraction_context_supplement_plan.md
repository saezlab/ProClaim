# Extraction Context Supplement — Implementation Plan

## Problem

`extract_and_add_facts` uses a fixed `claim` + `subclaims` baked into the prompt at
step 2 of the workflow. Information discovered later (synonyms, entity disambiguation,
negative scope clarifications) has no path into the extraction prompt, causing relevant
papers to return 0 facts.

Concrete example from SIGNOR-175304 / test_direct_1:

- Step 2: subclaims reference "CRTC2 directly activates AKT1"
- Step 10: web search reveals "TORC2" in AKT literature = mTOR Complex 2, **not** CRTC2
- The TORC2/mTORC2 disambiguation is never injected into the extraction prompt
- Papers about AKT phosphorylation by mTORC2 are silently discarded as 0 facts

**Constraint**: `claim` and `subclaims` must stay immutable — changing their wording
(e.g. "directly interact" → "interact") alters the logical strength of the assertion
and would corrupt the final verdict.

---

## Proposed Solution: `extraction_context` field on `EvidenceState`

Add a new field `extraction_context: list[str]` to `EvidenceState`.  
This is a free-form list of short notes the agent can append at any point during the
workflow. The content is injected into the `EXTRACT_FACTS` prompt as a supplementary
block, **after** the subclaims and **before** the paper text.

The agent decides when and what to write — it is not triggered automatically.

### What goes in `extraction_context`

- Synonym / alias mappings: `"CRTC2 is also known as TORC2 (transducer of regulated CREB activity); this is distinct from mTOR Complex 2 (mTORC2), which also appears in literature as TORC2."`
- Disambiguation notes: `"Papers mentioning 'TORC2 phosphorylates AKT' are describing mTOR Complex 2, not CRTC2, and are unlikely to contain relevant facts unless they also explicitly mention CRTC2."`
- Scope clarifications (narrowing): `"'Direct activation' requires a demonstrated biochemical mechanism (PTM, physical binding, or transcriptional regulation of the gene). Correlational knockdown studies alone are insufficient."`
- Any other free-text the agent judges useful for the LLM reader

The field is **additive only** — the agent appends notes, never deletes them, so
the audit trail is preserved.

---

## Re-extraction Requirement

When `extraction_context` is updated, all previously extracted papers must be re-run.
Adding a disambiguation note can reverse the direction of a previously extracted fact —
a paper tagged SUPPORT under the old prompt may become NEUTRAL or REFUTE once the model
understands the entities correctly. Therefore `extracted_pmids` is always cleared in full.

When the agent appends to `extraction_context`:
1. `extracted_pmids` is cleared.
2. The agent then calls `extract_and_add_facts` again for the relevant PMIDs.

---

## Files to Change

### 1. `evidence_state.py` — add field and helper to `EvidenceState`

```python
extraction_context: list[str] = Field(default_factory=list)
```

Expose a helper for convenience and to handle invalidation:

```python
def add_extraction_context(self, note: str) -> None:
    """Append a context note and clear the extraction cache for full re-extraction."""
    self.extraction_context.append(note)
    self.extracted_pmids.clear()
    self._auto_save()
```

### 2. `prompts.py` — add optional block to `EXTRACT_FACTS`

Current prompt ends with:
```
Subclaims:
{subclaims_str}

Paper (PMID: {source_pmid}):
{paper_text}
```

New prompt adds a conditional block between subclaims and paper:

```
Subclaims:
{subclaims_str}
{context_block}
Paper (PMID: {source_pmid}):
{paper_text}
```

Where `{context_block}` is either empty string (when no context) or:

```
Supplementary extraction context (use to interpret the paper correctly):
- <note 1>
- <note 2>
...

```

This block is rendered by a helper function rather than inline string formatting, so
the prompt template stays clean.

### 3. `subagents.py` — thread `extraction_context` through `extract_facts`

```python
def extract_facts(
    llm,
    paper_text: str,
    claim: str,
    subclaims: list[str],
    source_pmid: str,
    extraction_context: list[str] | None = None,   # NEW
) -> list[Fact]:
```

Inside the function:
- Render `context_block` from `extraction_context` (empty string if None or empty list)
- Pass it to `EXTRACT_FACTS.format(..., context_block=context_block)`

### 4. `evidence_api.py` — pass context through `_extract_and_add_facts_single`

```python
facts = extract_facts(
    llm=llm,
    paper_text=paper_text,
    claim=state.claim,
    subclaims=state.subclaims,
    source_pmid=pmid,
    extraction_context=state.extraction_context,   # NEW
)
```

No other changes needed — `extract_and_add_facts` already skips `extracted_pmids`,
so after `add_extraction_context` clears them it will automatically re-run all papers
on the next call.

### 5. `evidence_api.py` — expose `add_extraction_context` in the public API

Add to the `__all__` / docstring injection block so the agent REPL can call:

```python
from pkevolve.verification.evidence_api import add_extraction_context_note

add_extraction_context_note(
    state,
    "CRTC2 (also called TORC2) is a CREB transcription coactivator. "
    "mTOR Complex 2 (mTORC2) is a distinct kinase complex that also appears "
    "in literature as 'TORC2'. Papers discussing mTORC2 phosphorylating AKT "
    "are NOT about CRTC2 unless they explicitly name CRTC2.",
)
# → Added extraction context note. Cleared all extracted_pmids.
# → Re-run extract_and_add_facts to process all papers with the updated prompt.
```

This wrapper function handles the `_auto_save` and prints a summary for the agent.

### 6. `subagents.py` — thread `extraction_context` through search functions

`search_pubmed_llm` and `search_semantic_scholar_dual` use `state.subclaims` to
generate queries. Pass `extraction_context` so the query-generation prompt can
incorporate synonym/disambiguation notes when constructing search terms.

```python
def search_pubmed_llm(
    llm,
    claim: str,
    subclaims: list[str],
    extraction_context: list[str] | None = None,   # NEW
    ...
) -> list[str]:
```

The same `{context_block}` rendering helper used in `EXTRACT_FACTS` is reused here,
injected into the search query generation prompt after the subclaims.

Update `evidence_api.py` to pass `state.extraction_context` when calling these
functions, mirroring the pattern in step 4 above.

---

## Agent Workflow with the New Feature

```
Step 1-2   set up claim + subclaims (immutable from here)
Step 3     search PubMed + S2
Step 4     extract_and_add_facts  →  facts=0 for all (common case)
Step 5     inspect papers / web_search
             → discover: "TORC2 = mTORC2, not CRTC2"
Step 5.5   add_extraction_context_note(state, "CRTC2 ≠ mTORC2 ...")
             → all extracted_pmids cleared
Step 6     extract_and_add_facts  →  now uses enriched prompt
             → papers about TORC2+AKT now correctly yield 0 facts (or NEUTRAL/REFUTE)
             → papers about CRTC2+AKT (if any) more likely to surface SUPPORT facts
Step 7     check_sufficiency ...
```

The agent can call `add_extraction_context_note` multiple times at any point.

---

## What This Does NOT Change

- `claim` — immutable throughout
- `subclaims` — immutable throughout
- `sufficiency_history` — unaffected
- Any other field of `EvidenceState`

---

## Backward Compatibility

- `extraction_context` defaults to `[]` — no change in behavior for existing runs
- The `{context_block}` in the prompt renders as empty string when context is empty,
  so the prompt is identical to today's prompt for existing runs
- `evidence_state.json` files from old runs load cleanly (Pydantic ignores unknown
  fields by default; missing fields get defaults)

