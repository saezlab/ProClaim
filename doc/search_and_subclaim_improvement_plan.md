# Search Coverage & Subclaim Quality Improvement Plan

## Motivation

Analysis of two runs on SIGNOR-156958 (GNAS→ADCY1) revealed that the final verdict
(SUPPORT vs REFUTE) depended entirely on whether PMID 1692962 was retrieved. The
difference traced back to two root causes:

1. **Search query quality**: `search_semantic_scholar` receives a query written
   inline by the orchestrator. The orchestrator's query in the claude-4 run lacked
   biological aliases ("Gs-alpha", "cAMP"), so the critical paper was never found.
   `search_pubmed_llm` already uses an internal LLM sub-call to generate the query,
   but only produces one query string; the orchestrator never contributes a second
   perspective.

2. **Subclaim framing**: The orchestrator decided subclaims before searching. Because
   the subclaims in the claude-4 run used generic phrasing ("forms a physical complex")
   rather than domain-specific aliases ("Gs-alpha", "cAMP production"), the vocabulary
   of subsequent search queries was weaker. There is currently no prompt guidance for
   subclaim decomposition — the workflow only says "do it".

---

## Proposed Changes

### 1. Dual-Query Search Strategy

**Goal**: Maximize paper retrieval coverage by always running two independent query
sources per search engine, then deduplicating.

#### Current flow (iteration 0)

```
search_pubmed_llm(claim, state, llm)
    └── generate_search_query(claim, llm)   ← sub-agent LLM call, query A
        └── PubMed search A

search_semantic_scholar(query, state)
    └── query written inline by orchestrator
        └── S2 search B
```

#### Proposed flow

```
search_pubmed_llm(claim, state, llm)
    └── generate_search_query(claim, llm)          ← sub-agent query A  (unchanged)
    └── generate_search_query(claim, llm, subclaims)  ← second sub-agent call using subclaims
        └── PubMed searches A + B, deduplicated internally

search_semantic_scholar_dual(claim, state, llm)    ← new wrapper
    └── generate_search_query_s2(claim, llm, subclaims)  ← sub-agent S2 query
    └── generate_search_query_s2(claim, llm)              ← claim-only S2 query
        └── S2 searches C + D, deduplicated internally
```

The orchestrator only calls two functions (same as now); the dual-query logic is
**encapsulated inside the API functions**, not delegated to the orchestrator. This keeps
the workflow prompt simple and avoids burdening the orchestrator with extra steps.

Both queries run independently; `_search_and_add` already deduplicates by PMID.

#### Implementation notes

- **`generate_search_query`** gains an optional `subclaims: list[str] | None` param.
  When provided, the prompt includes the subclaims so the sub-agent can pick up aliases
  the orchestrator introduced during decomposition.
- **`search_pubmed_llm`** calls `generate_search_query` twice: once without subclaims
  (current behaviour) and once with `state.subclaims`, then issues both searches.
- Add **`generate_search_query_s2(claim, llm, subclaims=None) -> str`** in
  `src/pkevolve/search/llm_query_generator.py` with a prompt tuned for Semantic Scholar.
- Add **`search_semantic_scholar_dual(claim, state, llm) -> list[str]`** in
  `evidence_api.py` that calls `generate_search_query_s2` twice (with and without
  subclaims) and issues both searches.
- Update `DIRECT_SYSTEM_PROMPT` step 3 to say:

  ```
  3. Search (iteration 0):
     a. search_pubmed_llm(state.claim, state, llm)
     b. search_semantic_scholar_dual(state.claim, state, llm)
  ```

#### Trade-offs

| | Pro | Con |
|---|---|---|
| Dual query inside API | Orchestrator stays simple; no extra workflow steps | Slightly harder to debug which query found which papers |
| Subclaims passed to sub-agent | Sub-agent uses richer vocabulary from decomposition | Subclaims must be set before calling search (already required by workflow) |
| Dedup by PMID | No redundant fact extraction | Already implemented |

---

### 2. Guided Subclaim Decomposition Prompt

**Goal**: Improve subclaim quality without overfitting to a specific domain
(e.g., protein–protein interactions). Keep guidance at the level of "biomedical
reasoning" rather than "biology of proteins".

#### Why subclaims matter

Subclaims affect:
- The orchestrator's vocabulary when writing search queries
- `extract_and_add_facts`: facts are linked to subclaims; zero-coverage subclaims trigger gaps
- `check_sufficiency`: gap descriptions reference subclaim text
- `refine_search_for_failed_papers`: subclaim list is passed to the sub-agent to generate
  better queries

#### Current prompt (line 429, prompts.py)

```
2. Decompose the claim: `state.subclaims = ["subclaim A", ...]` then `state._auto_save()`.
```

No guidance on how to decompose.

#### Proposed replacement

```
2. Decompose the claim into 1–5 atomic subclaims, each a single independently
   verifiable assertion. Use 1 (the original claim) if the claim is already simple
   enough to search directly. Include alternative names or aliases for key entities.
   Set subclaims before searching: `state.subclaims = [...]` then `state._auto_save()`.
```

#### Design rationale

- **1–5 range**: 1 is valid for simple or narrow claims; no artificial minimum forces
  spurious decomposition. Upper bound of 5 prevents over-decomposition that dilutes gap
  detection.
- **"atomic … single independently verifiable assertion"**: echoes the existing language
  in the `EXTRACT_FACTS` prompt ("extract every atomic fact"), keeping terminology
  consistent across the system.
- **"alternative names or aliases"**: domain-neutral — applies to gene symbols, drug
  names, disease terms, pathway names — without hard-coding biology-specific rules.
- Deliberately short: the model should exercise judgment; over-specified instructions
  tend to produce rigid subclaims that miss the claim's actual intent.

#### Example (GNAS→ADCY1 claim)

With the proposed guidance, expected subclaims:
- "Gs alpha (GNAS) stimulates adenylyl cyclase activity to increase cAMP"
- "GNAS forms a direct physical complex with ADCY1"
- "GNAS activates ADCY1 through GTP-dependent conformational change"
- "Loss-of-function mutations in GNAS reduce ADCY1-mediated cAMP production"

Without the guidance, the claude-4 model generated:
- "GNAS forms a physical complex with ADCY1" (no alias)
- "GNAS activates ADCY1 through post-translational modification" (mechanism mismatch)

---

## Files to Change

| File | Change |
|------|--------|
| `src/pkevolve/verification/prompts.py` | Update step 2 (subclaim guidance) and step 3 (dual search API calls) in `DIRECT_SYSTEM_PROMPT` |
| `src/pkevolve/search/llm_query_generator.py` | Add optional `subclaims` param to `generate_search_query`; add `generate_search_query_s2()` |
| `src/pkevolve/verification/evidence_api.py` | Update `search_pubmed_llm` to run two queries; add `search_semantic_scholar_dual` |
