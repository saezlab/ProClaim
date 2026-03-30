# Semantic Scholar Paper Discovery — Implementation Plan

Covers three new discovery methods motivated by the 4-source pipeline analysis:

1. **S2 Bulk Search** — parallel keyword search via Semantic Scholar
2. **S2 Recommendations** — graph-expansion from confirmed-relevant seed papers
3. **Citation Chaining** — DOI extraction from full text + S2 lookup

---

## Architecture Overview

All S2 HTTP calls are isolated in a new module:

```
src/pkevolve/search/semantic_scholar.py   ← S2Client + _s2_paper_to_record()
src/pkevolve/verification/evidence_api.py ← 3 new public functions
```

This mirrors the existing `custom_pubmed.py` / `evidence_api.py` separation.
`PaperRecord.source` already accepts `"semantic_scholar"` with no model changes needed.

### S2Client (semantic_scholar.py)

```python
_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_REC_URL    = "https://api.semanticscholar.org/recommendations/v1/papers/"
_S2_PAPER_URL  = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
_S2_FIELDS     = "paperId,externalIds,title,abstract,authors,year,openAccessPdf"

class S2Client:
    def search(self, query: str, limit: int) -> list[dict]: ...
    def recommendations(self, positive_ids: list[str],
                        negative_ids: list[str], limit: int) -> list[dict]: ...
    def lookup_doi(self, doi: str) -> dict | None: ...
```

Reads `S2_API_KEY` from environment (unauthenticated fallback: 1 req/s enforced with `time.sleep`).

### Shared helper: `_s2_paper_to_record`

Private function in `evidence_api.py`. Converts an S2 API response dict to a
`PaperRecord`, handling the two ID cases:

- If `externalIds.PubMed` exists → `pmid = externalIds["PubMed"]`
- Else → `pmid = f"S2:{data['paperId']}"`

If `openAccessPdf.url` is present, store it in a future `pdf_url` field so
`get_full_text_article` can retrieve it without Unpaywall.

### Cross-source deduplication

Current `_search_and_add` deduplicates only by `pmid`. After adding S2,
the same paper can appear with a PubMed PMID and an `S2:` ID. Before adding,
`_s2_paper_to_record` must check whether the paper's DOI already exists in
`state.papers`. Implementation: linear scan over `state.papers.values()` comparing
`.doi` (acceptable for <200 papers). If a future bottleneck, add a
`doi → pmid` index to `EvidenceState`.

### Recommendations vs. Citation Chaining

Methods 2 and 3 are complementary — they find different paper sets via different
mechanisms and should both be called at iteration ≥ 1:

| | `search_semantic_scholar_recommendations` | `expand_via_citations` |
|---|---|---|
| **Discovery direction** | Lateral — papers *similar to* seeds in S2's embedding space | Backward — papers *cited by* existing full texts |
| **Typical paper age** | Recent (ML similarity favours newer literature) | Older seminal works that established the biology |
| **Dependency** | ≥ 1 SUPPORT fact extracted | Full text retrieved for ≥ 1 paper |
| **What it misses** | Classic papers not semantically similar to the modern framing | Papers similar to seeds but not directly cited |

A keyword search finds papers by vocabulary. Recommendations finds papers by
semantic neighbourhood. Citation chaining finds the foundational references
that established the interaction — exactly the papers SIGNOR curators cited
and that rarely surface in keyword or recommendation searches.

The only scenario where one is clearly preferable over the other is when full
texts are sparse (citation chaining has no input) — in which case
recommendations alone covers the gap-filling iteration.  Otherwise, calling both
adds papers from distinct lineages at zero LLM cost.

---

## 1. S2 Bulk Search

### Purpose
Cover papers that PubMed/MEDLINE does not index: bioRxiv preprints, CS/ML
venues, non-MEDLINE journals.  Best used in parallel with `search_pubmed_llm`
at iteration 0.

### API
```
GET https://api.semanticscholar.org/graph/v1/paper/search
    ?query=<url-encoded query>
    &fields=paperId,externalIds,title,abstract,authors,year,openAccessPdf
    &limit=<max_results>
```

### Implementation

**Signature** (added to `evidence_api.py`):
```python
def search_semantic_scholar(
    query: str,
    state: EvidenceState,
    max_results: int = 10,
) -> list[str]:
    """Search Semantic Scholar and add papers to state.

    Complements search_pubmed_llm by covering preprints and non-MEDLINE
    sources. Use the same query string as the PubMed search.

    Returns:
        List of added paper IDs (PMID or S2:{id}).
    """
```

**Steps:**
1. Instantiate `S2Client`, call `client.search(query, limit=max_results)`.
2. For each result dict, call `_s2_paper_to_record(data)`.
3. Skip if `pmid` already in `state.papers` or DOI matches existing paper.
4. Call `state.add_paper(record)`.
5. Print summary line: `f"S2 Search: found {found}, added {added} new."`
6. Return list of added IDs.

### When to call
```python
# Iteration 0 — run in sequence (HTTP calls can't be parallelised in kernel)
pubmed_pmids = search_pubmed_llm(claim, state, llm, max_results=10)
s2_pmids = search_semantic_scholar(query, state, max_results=10)
```

---

## 2. S2 Recommendations

### Purpose
Graph-expansion starting from papers that have already yielded supporting/refuting
evidence. S2's recommendation model returns up to 500 semantically similar
papers given positive/negative seed IDs.  This is the highest-value addition
because seeds are selected from confirmed-relevant papers, making discovery
targeted rather than keyword-driven.

### API
```
POST https://api.semanticscholar.org/recommendations/v1/papers/
     ?fields=paperId,externalIds,title,abstract,authors,year,openAccessPdf
     &limit=<max_results>

Body: {
  "positivePaperIds": ["PMID:12345678", ...],
  "negativePaperIds": []
}
```

S2 accepts `PMID:<id>` format directly in recommendation seeds.

### Seed Selection

Auto-derived from state when `seed_pmids=None`:

```python
# Positive seeds: papers that produced ≥1 SUPPORT fact
positive = [
    pmid for pmid in state.papers
    if any(f.source_pmid == pmid and f.stance == Stance.SUPPORT
           for f in state.facts)
]
# Negative seeds (user overwrite - should be mandatory): papers with only REFUTE facts
negative = [
    pmid for pmid in state.papers
    if all(f.stance == Stance.REFUTE
           for f in state.facts if f.source_pmid == pmid)
    and pmid not in positive
]
```

If fewer than 1 seed exists, skip and return `[]` with a warning —
the recommendations model requires at least one seed.

### Implementation

**Signature** (added to `evidence_api.py`):
```python
def search_semantic_scholar_recommendations(
    state: EvidenceState,
    seed_pmids: list[str] | None = None,
    max_results: int = 20,
) -> list[str]:
    """Expand paper pool via S2 Recommendations using confirmed-relevant seeds.

    Auto-derives seeds from papers with SUPPORT facts if seed_pmids is None.
    Requires at least one seed paper; returns [] if none are available.

    Args:
        state: EvidenceState with papers and facts populated.
        seed_pmids: Override automatic seed selection (PMID strings).
        max_results: Maximum papers to add.

    Returns:
        List of added paper IDs.
    """
```

**Steps:**
1. Derive positive/negative seeds (see above) or use `seed_pmids`.
2. Format seeds as `f"PMID:{pmid}"` for non-S2 IDs; `S2:` IDs are passed as-is.
3. Call `client.recommendations(positive_ids, negative_ids, limit=max_results)`.
4. For each result: `_s2_paper_to_record` → dedup → `state.add_paper`.
5. Print: `f"S2 Recommendations: {len(positive)} seeds → added {added} papers."`

### When to call
```python
# Iteration ≥ 1, after facts have been extracted from initial papers
new_pmids = search_semantic_scholar_recommendations(state, max_results=20)
```

---

## 3. Citation Chaining

### Purpose
Backward snowballing: mine the reference sections of full-text papers already
in state to find seminal works that keyword searches systematically miss.  No
LLM call needed — DOI regex extraction is near-perfect on structured reference lists.

### DOI Extraction

```python
_DOI_RE = re.compile(r'\b10\.\d{4,}/[^\s,;)\]"\'>]+')
```

Applied to `paper.full_text` for all papers in `state.papers` where
`full_text is not None`. Collected DOIs are normalised (lowercase, trailing
punctuation stripped) and deduplicated against DOIs already in state.

### API Lookup

For each candidate DOI:
```
GET https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}
    ?fields=paperId,externalIds,title,abstract,authors,year,openAccessPdf
```

Returns single paper dict → `_s2_paper_to_record` → dedup → add.

### Implementation

**Signature** (added to `evidence_api.py`):
```python
def expand_via_citations(
    state: EvidenceState,
    max_per_paper: int = 5,
) -> list[str]:
    """Expand evidence pool by mining DOIs from full-text reference sections.

    For each paper in state with full_text, extracts cited DOIs via regex,
    looks them up via Semantic Scholar, and adds new papers to state.
    No LLM call required.

    Args:
        state: EvidenceState; only papers with full_text are processed.
        max_per_paper: Maximum new papers to add per source paper.

    Returns:
        List of added paper IDs.
    """
```

**Steps:**
1. Collect all papers where `paper.full_text is not None`.
2. For each, run `_DOI_RE.findall(paper.full_text)` → normalise → deduplicate.
3. Skip DOIs already present in state (check `paper.doi` of existing records).
4. Cap candidates at `max_per_paper` per source paper (take first N unique DOIs).
5. For each candidate DOI: `client.lookup_doi(doi)` → `_s2_paper_to_record` →
   dedup → `state.add_paper`. Sleep 1s between requests (rate limit).
6. Print: `f"Citation chaining: {len(papers_with_text)} papers scanned, {added} new papers added."`

### Why no LLM
DOI regex on structured reference sections has near-100% recall.  Using an LLM
for citation extraction would add latency and API cost with no accuracy benefit.

### When to call
```python
# After get_full_text_article has been called for initial papers
for pmid in all_pmids:
    get_full_text_article(pmid, state)

new_pmids = expand_via_citations(state, max_per_paper=5)
```

---

## Example Recommended Call Order in the Verification Loop (the agent should decide the actual order and calling)

```
ITERATION 0  (initial search)
─────────────────────────────────────────────────────
search_pubmed_llm(claim, state, llm, max_results=10)
search_semantic_scholar(query, state, max_results=10)   ← NEW: parallel source

for pmid in all_new_pmids:
    get_full_text_article(pmid, state)
    extract_and_add_facts(llm, pmid, state)

populate_paper_features(state)
result = check_sufficiency(state, llm)

ITERATION 1+  (gap filling)
─────────────────────────────────────────────────────
for gap in result.gaps:
    search_for_gap(gap.description, state)              ← existing

search_semantic_scholar_recommendations(state)          ← NEW: graph expand
expand_via_citations(state, max_per_paper=5)            ← NEW: citation mine

for pmid in newly_added_pmids:
    get_full_text_article(pmid, state)
    extract_and_add_facts(llm, pmid, state)

populate_paper_features(state)
result = check_sufficiency(state, llm)
```

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `S2_API_KEY` | Optional | Semantic Scholar API key (removes rate limit cap) |

Without `S2_API_KEY`, `S2Client` enforces 1 req/s with `time.sleep(1)`. The
unauthenticated tier allows ~5,000 reqs/day which is sufficient for a
verification run of ≤8 iterations.

---

## Files Changed / Created

| File | Change |
|------|--------|
| `src/pkevolve/search/semantic_scholar.py` | **New** — `S2Client` with `search`, `recommendations`, `lookup_doi` |
| `src/pkevolve/verification/evidence_api.py` | **Add** `search_semantic_scholar`, `search_semantic_scholar_recommendations`, `expand_via_citations`, `_s2_paper_to_record` |
| `src/pkevolve/verification/README.md` | **Update** search method table with 3 new functions |
