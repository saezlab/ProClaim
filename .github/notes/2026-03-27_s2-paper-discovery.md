# Semantic Scholar Paper Discovery — 2026-03-27

**Branch:** `fix/pid_search`

## Summary

Added three new paper discovery methods to the evidence API based on the Semantic Scholar Graph API: keyword search, graph-based recommendations, and backward citation chaining. These complement existing PubMed-only search by covering bioRxiv preprints, non-MEDLINE journals, and by finding seminal cited works that keyword searches systematically miss. A new standalone `S2Client` module handles all S2 HTTP calls with rate-limiting, and three public functions in `evidence_api.py` wire it into the verification pipeline.

## New Files

| File | Purpose |
|------|---------|
| `src/pkevolve/search/semantic_scholar.py` | `S2Client` — thin wrapper around S2 Graph API with `search()`, `recommendations()`, `lookup_doi()` methods; 1 req/s rate limiting without API key |
| `doc/s2_paper_discovery_plan.md` | Implementation plan covering architecture, API contracts, seed selection logic, and recommended call order |

## Modified Files

| File | Changes |
|------|---------|
| `src/pkevolve/verification/evidence_api.py` | Added `_DOI_RE`, `_s2_paper_to_record()`, `_add_s2_records()`, `search_semantic_scholar()`, `search_semantic_scholar_recommendations()`, `expand_via_citations()` (~265 lines) |
| `src/pkevolve/verification/README.md` | Updated Core API table to list new functions; expanded search method comparison table with Corpus and Best Call Point columns; added prose descriptions for the three new methods |

## Architecture

```
evidence_api.py (public surface)            semantic_scholar.py (HTTP layer)
┌─────────────────────────────────┐        ┌────────────────────────────────┐
│ search_semantic_scholar()       │───────→│ S2Client.search()              │
│ search_semantic_scholar_recs()  │───────→│ S2Client.recommendations()    │
│ expand_via_citations()          │───────→│ S2Client.lookup_doi()          │
│                                 │        │                                │
│   _s2_paper_to_record()         │        │   _rate_limit()  (1 req/s)     │
│   _add_s2_records()             │        │   _get() / _post()             │
└──────────┬──────────────────────┘        └────────────────────────────────┘
           │
           ▼
  EvidenceState.add_paper()
  (dedup by PMID + cross-source DOI match)
```

### Recommended Call Order

```
ITERATION 0  (initial discovery)
  search_pubmed_llm(claim, state, llm)
  search_semantic_scholar(query, state)          ← S2 keyword search
  get_full_text_article() + extract_and_add_facts()
  populate_paper_features() → check_sufficiency()

ITERATION 1+  (gap filling + expansion)
  search_for_gap(gap, state)                     ← existing PubMed gap search
  search_semantic_scholar_recommendations(state)  ← S2 graph expansion
  expand_via_citations(state)                     ← backward citation chaining
  get_full_text_article() + extract_and_add_facts()
  populate_paper_features() → check_sufficiency()
```

## Key Design Decisions

- **Separate `S2Client` module in `src/pkevolve/search/`:** Mirrors the existing `custom_pubmed.py` / `evidence_api.py` separation. HTTP concerns are isolated from evidence state manipulation, and the client is reusable outside the verification pipeline.

- **Cross-source deduplication by DOI:** The same paper can arrive via PubMed (numeric PMID) and S2 (`S2:<hash>`). `_s2_paper_to_record()` performs a linear scan of `state.papers.values()` comparing `.doi` to prevent duplicates. Acceptable for <200 papers per run; a `doi → pmid` index can be added if this becomes a bottleneck.

- **Synthetic PMID format `S2:<paperId>`:** Papers without a PubMed ID (preprints, non-indexed venues) use `S2:` prefix so they integrate into the existing PMID-keyed `state.papers` dict without collision. The `get_full_text_article` fallback chain handles S2-origin papers via DOI resolution.

- **Auto-derived seeds for recommendations:** Positive seeds are papers with ≥1 SUPPORT fact; negative seeds are papers with only REFUTE facts. This makes graph expansion targeted rather than random. Manual seed override via `seed_pmids` parameter is supported.

- **Recommendations vs. citation chaining are complementary, not redundant:** They find different paper sets via different mechanisms.

  | | `search_semantic_scholar_recommendations` | `expand_via_citations` |
  |---|---|---|
  | **Discovery direction** | Lateral — papers *similar to* seeds in S2's embedding space | Backward — papers *cited by* existing full texts |
  | **Typical paper age** | Recent (ML similarity favours newer literature) | Older seminal works that established the biology |
  | **Dependency** | ≥ 1 SUPPORT fact extracted | Full text retrieved for ≥ 1 paper |
  | **What it misses** | Classic papers not semantically similar to the modern framing | Papers similar to seeds but not directly cited |

  The only scenario where one is clearly preferable is when full texts are sparse (citation chaining has no input). Otherwise, calling both adds papers from distinct lineages at zero LLM cost.

- **Citation chaining via DOI regex, not LLM:** `_DOI_RE` applied to `paper.full_text` has near-100% recall on structured reference sections. Using an LLM for citation extraction would add latency and cost with no accuracy benefit.

- **Rate limiting without API key:** `S2Client` enforces 1 req/s via `time.sleep` when `S2_API_KEY` is not set. The unauthenticated S2 tier allows ~5,000 req/day — sufficient for a verification run of ≤8 iterations.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `S2_API_KEY` | Optional | Semantic Scholar API key; removes 1 req/s rate limit |
