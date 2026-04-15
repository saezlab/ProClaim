# P1 Implementation Plan — Structural Accuracy Fix

Branch base: `refactor/cost-accuracy`. Execute items in order: 3 → 5 → 6 → 4.
Sub-tasks within an item can be parallelised.

---

## Item 3 — Enrich `identify_gaps` with paper-pool context

`identify_gaps` currently sees only facts. Without paper-pool context it produces vague
gap descriptions with no vocabulary guidance, causing the orchestrator to read paper
texts by hand instead of reformulating searches.

### 3a. Add `candidate_reformulations` to `Gap`

**File:** `src/pkevolve/verification/data_models.py`

Add one field to `Gap` after `priority`:

```python
candidate_reformulations: list[str] = Field(default_factory=list)
```

Synonym/MeSH suggestions derived by `identify_gaps` from claim/subclaim vocabulary
analysis. Used directly as input to the next `search_pubmed_llm` or `search_for_gap`
call. No migration needed (optional field, defaults on load).

### 3b. Extend `identify_gaps` to receive paper pool

**File:** `src/pkevolve/verification/subagents.py`, `identify_gaps` at line 324.

New signature:

```python
def identify_gaps(
    llm: LLMCallable,
    claim: str,
    subclaims: list[str],
    facts: list[Fact],
    papers: dict | None = None,
    extracted_pmids: list[str] | None = None,
    mlp_feature_vec: dict | None = None,
) -> list[Gap]:
```

Replace the existing `prompt = f"""..."""` with the following additions:

**1. Build `paper_pool_str`** from `papers` and `extracted_pmids` (when both provided).
Split into two groups — omit papers that already have facts:

- `pmid not in extracted_pmids` → unextracted
- `pmid in extracted_pmids`, no fact with `source_pmid == pmid` → extracted, 0 facts

Limit to 20 papers across both groups. Format:
```
Papers not yet extracted (call extract_and_add_facts on these):
  PMID 12345678: Title here
  ...

Papers extracted but 0 facts found (vocabulary signal — do NOT re-extract):
  PMID 27492469: Title here
  ...
```

**2. Build `feature_summary_str`** from `mlp_feature_vec` (when provided), two features only:
```
MLP snapshot (background only): mean_entity_coverage={v:.2f}, mean_similarity={v:.2f}
```

**3. Insert both blocks** between the `Extracted facts` block and the `Gap types` block.

**4. Extend the per-gap JSON schema** to include `candidate_reformulations`. Instruct
the LLM to derive reformulations from claim/subclaim wording vs. extracted fact
vocabulary and titles of 0-fact papers — not from NLP scores:

```json
{
  "subclaim": "...",
  "gap_type": "...",
  "description": "...",
  "priority": "high | medium | low",
  "candidate_reformulations": [
    "Try 'Gs-alpha subunit' in place of 'GNAS'",
    "Try 'ADCY1' in place of 'type I adenylyl cyclase'"
  ]
}
```

**5. Update `_parse_gaps_response`** (line ~402) to read `.get("candidate_reformulations", [])`.

### 3c. Thread new args into all three `check_sufficiency` backends

**File:** `src/pkevolve/verification/evidence_api.py`

| Function | Line | Change |
|---|---|---|
| `_check_sufficiency_mlp` | ~1814 | add `papers=state.papers, extracted_pmids=state.extracted_pmids, mlp_feature_vec=flat_feats` |
| `_check_sufficiency_llm` | ~1901 | add `papers=state.papers, extracted_pmids=state.extracted_pmids, mlp_feature_vec=None` |
| `_check_sufficiency_haiku` | ~1992 | add `papers=state.papers, extracted_pmids=state.extracted_pmids, mlp_feature_vec=None` |

`flat_feats` is already in scope in the MLP backend (line ~1758).

### 3d. Surface reformulations in the `check_sufficiency` print line

In each backend extend the `print(...)`:

```python
reformulation_hint = ""
if gaps and gaps[0].candidate_reformulations:
    reformulation_hint = f" reformulations={gaps[0].candidate_reformulations[:2]}"
print(f"check_sufficiency[mlp]: label={label} prob={prob:.3f} papers={current_paper_count} gaps={len(gaps)}{reformulation_hint}")
```

---

## Item 5 — Early exit on sufficiency plateau

### 5a. Add `early_exit_recommended` to `SufficiencyResult`

**File:** `src/pkevolve/verification/data_models.py`

```python
early_exit_recommended: bool = False
```

### 5b. Compute plateau flag in all three backends

**File:** `src/pkevolve/verification/evidence_api.py`

After `result = SufficiencyResult(...)` and before `state.sufficiency_history.append(result)`
in each backend:

```python
if len(state.sufficiency_history) >= 2:
    delta = abs(state.sufficiency_history[-1].confidence - state.sufficiency_history[-2].confidence)
    if delta < 0.05:
        result = result.model_copy(update={"early_exit_recommended": True})
        logger.info("check_sufficiency: plateau detected (delta=%.4f)", delta)
```

Use `prob` (MLP) or `score` (llm/haiku) as appropriate.

### 5c. Surface the flag in the print line

```python
exit_hint = " [EARLY EXIT — call emit_verdict now]" if result.early_exit_recommended else ""
print(f"check_sufficiency[mlp]: label={label} prob={prob:.3f} papers={current_paper_count} gaps={len(gaps)}{exit_hint}")
```

### 5d. Update system prompt

**File:** `src/pkevolve/verification/evidence_programming.py`

In workflow step 11a, replace:
```
If trend shows 'declining' or 'flat' for multiple iterations, consider whether to emit verdict.
```
With:
```
If check_sufficiency() returns early_exit_recommended=True, call emit_verdict() immediately.
Only continue if candidate_reformulations contains a query not yet tried.
```

---

## Item 6 — Remove downstream-patching tools from `function_docs`

**File:** `src/pkevolve/verification/evidence_api.py`, function `function_docs` (~line 190).

### Removals

**Category A — dead code (no callers anywhere):** remove from `_api_funcs` and `_sub_funcs`:
- `update_synthesis`, `add_conflict` (api)
- `synthesize_subclaim`, `detect_conflicts` (subagents) + their imports

**Category B — internal helpers:** remove from `_api_funcs` / `_sub_funcs`:
- `get_full_text_article`, `get_paper_text`, `add_facts_from_dicts` (api)
- `extract_facts` (subagents) + its import

Do not rename or delete function definitions — only drop from the exposure lists.

**Category C — architecturally mis-placed:** remove from `_api_funcs`:
- `compress_evidence` — trigger belongs in loop infra, not orchestrator judgment
- `populate_paper_features_parallel` — `populate_paper_features` already auto-switches to parallel

### Final counts

- `_api_funcs`: 15
- `_sub_funcs`: 2 (`formulate_gap_queries`, `refine_search_query`)
- **Total: 17**

Add a guard at the end of `function_docs`:
```python
assert len(_api_funcs) == 15 and len(_sub_funcs) == 2
```

---

## Item 4 — Encode the upstream-fix principle in the system prompt

Written after Item 6 so the WRONG responses list only covers behaviours not already
blocked structurally by tool removal.

**File:** `src/pkevolve/verification/evidence_programming.py`

Insert immediately before `## CRITICAL: Grounded Evidence Only` (line 126):

```
## CRITICAL: Upstream Fixes Only

When check_sufficiency() returns label="insufficient", or when
extract_and_add_facts() returns 0 for papers you expected to yield evidence,
the cause is ALWAYS one of:
  (a) the search missed the right papers, or
  (b) the subclaim uses the wrong vocabulary, synonyms, or granularity level.

The correct response is ALWAYS to loop back to search reformulation:
  - Re-decompose subclaims using synonyms, MeSH terms, or broader/narrower concepts.
  - Re-run search_pubmed_llm or search_for_gap with the reformulated query.
  - Use candidate_reformulations from the gap objects returned by check_sufficiency.

The WRONG responses (NEVER do these):
  - Do NOT inspect NLP features, claim_entity_coverage, or semantic_similarity by hand.
  - Do NOT loop over PMIDs printing titles or abstracts to find evidence by eye.
  - Do NOT manually compute NLI scores or inspect best-chunk texts.

The sufficiency classifier and fact extractor are measurement instruments. When they
report "insufficient", fix what is being measured — not the instruments.
```

---

## Acceptance criteria

| Item | Test |
|---|---|
| 3 | Run `check_sufficiency` with some papers in `extracted_pmids` but 0 facts; assert `gaps[0].candidate_reformulations` is non-empty. |
| 5 | Build `sufficiency_history` with two entries where `abs(delta) < 0.05`; call a backend; assert `result.early_exit_recommended is True`. |
| 6 | Call `function_docs()`; assert all 10 removed tools are absent; assert total count = 17. |
| 4 | Assert the CRITICAL section is present in `SYSTEM_PROMPT` and no tool removed in Item 6 is mentioned by name. |

---

## Files touched

| File | Items |
|---|---|
| `src/pkevolve/verification/data_models.py` | 3a, 5a |
| `src/pkevolve/verification/subagents.py` | 3b |
| `src/pkevolve/verification/evidence_api.py` | 3c, 3d, 5b, 5c, 6 |
| `src/pkevolve/verification/evidence_programming.py` | 5d, 4 |
