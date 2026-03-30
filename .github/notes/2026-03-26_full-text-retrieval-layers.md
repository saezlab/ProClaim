# Full-Text Retrieval: Layer 1.5 + Layer 4 — 2026-03-26

**Branch:** `fix/full_text`

## Summary

Added two new retrieval layers to `full_text.py` to reduce false negatives caused by papers that silently yield no text. **Layer 1.5** queries the Semantic Scholar Graph API for OA PDF links and extracts body text via pymupdf, covering papers on Frontiers, PLoS, and similar publishers not indexed under NCBI PMC. **Layer 4** fetches the PubMed structured abstract (with IMRAD section labels) as a guaranteed last-resort fallback, ensuring `fetch_full_text` almost never returns `None`. Previously, all full-text failures resulted in `None` which caused `extract_and_add_facts` to silently skip the paper; Layer 4 eliminates this silent discard path for the common case of paywalled 1990s–2000s SIGNOR-era papers.

## Modified Files

| File | Changes |
|------|---------|
| `src/pkevolve/verification/full_text.py` | Added `_fetch_semantic_scholar()` (Layer 1.5) and `_fetch_pubmed_structured_abstract()` (Layer 4); added `_S2_GRAPH_BASE` constant; wired both into `fetch_full_text()`; updated module docstring |
| `src/pkevolve/verification/evidence_api.py` | Updated `get_full_text_article()` docstring to reflect the new 6-layer chain; clarified that the final `paper.abstract` fallback is now rarely reached |

## Architecture

```
fetch_full_text(pmid, doi, title)
│
├── Layer 1   PMC Open Access XML       (NCBI efetch → PMC XML)
├── Layer 1b  Europe PMC REST API       (broader OA, author manuscripts)
├── Layer 1.5 Semantic Scholar OA PDF   ← NEW
│             • GET /graph/v1/paper/PMID:{pmid}?fields=openAccessPdf
│             • downloads PDF → pymupdf text extraction
│             • silently skips on 429 (rate limit)
├── Layer 2   INDRA literature          (PMC + Elsevier via indra)
├── Layer 3   Unpaywall + PDF           (OA PDF via Unpaywall API)
└── Layer 4   PubMed structured abstract ← NEW
              • GET efetch?db=pubmed&id={pmid}&retmode=xml
              • parses <AbstractText Label="..."> with section labels
              • prefixes result with "[Abstract only — full text unavailable]"
              • ensures non-None return for downstream fact extraction
```

## Key Design Decisions

- **Layer 1.5 position (after Europe PMC, before INDRA):** Semantic Scholar PDF download is fast when the URL resolves but can hit 403s from blocking publishers (MDPI, Elsevier direct links). Placing it before the heavier INDRA layer keeps it cheap-first while leaving INDRA as the publisher-authenticated fallback.

- **Silent 429 handling in Layer 1.5:** The public S2 API allows ~100 req/5 min without a key. Rather than raising or sleeping, a 429 response logs at DEBUG and falls through to the next layer. This avoids blocking the pipeline when rate limits are hit during batch evaluation runs.

- **Abstract prefix `[Abstract only — full text unavailable]`:** Signals to the LLM subagent that the text is abstract-level, allowing calibrated confidence in extracted facts. Also allows filtering or weighting by the feature extractor in future iterations.

- **Layer 4 does not replace `paper.abstract`:** `get_full_text_article()` still falls back to `paper.abstract` (already fetched at search time) if Layer 4 also fails (e.g. network error). This makes Layer 4 an additional retrieval attempt, not a dependency injection.

- **Title cross-validation applied in Layer 1.5:** Consistent with all other layers — if the PDF content doesn't contain enough title words, it is discarded to prevent serving a wrong paper's text.
