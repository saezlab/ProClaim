# Fix Verification Endpoint & Full-Text Retrieval — 2025-02-25

**Branch:** `RLM`

## Summary

Fixed two critical bugs in the evidence verification subsystem that caused the notebook demo (`evidence_report.ipynb`) to fail silently: (1) the Z.AI OpenAI-compatible endpoint URL was wrong (`/api/openai` → `/api/paas/v4/`), causing all LLM subagent calls to fail, and (2) full-text paper retrieval was limited to PMC Open Access XML only, missing the majority of papers. Added a new layered full-text retrieval module and updated endpoint configuration across the codebase.

## New Files

| File | Purpose |
|------|---------|
| `src/proclaim/verification/full_text.py` | Layered full-text retrieval: PMC E-utilities → INDRA `get_full_text()` → Unpaywall+PDF (`pymupdf`). Each layer falls back to the next. INDRA is import-time optional. |

## Modified Files

| File | Changes |
|------|---------|
| `src/proclaim/verification/evidence_api.py` | Replaced inline PMC-only `get_full_text_article()` (~95 lines) with a thin wrapper delegating to `full_text.fetch_full_text()`. Wired DOI into `_search_and_add()` and `find_related_articles()` PaperRecord creation. |
| `src/proclaim/verification/data_models.py` | Added `doi: Optional[str] = None` field to `PaperRecord`. |
| `src/proclaim/verification/subagents.py` | Increased `extract_facts()` paper text truncation from 6,000 → 16,000 chars (GLM-4.6 supports 200K context). |
| `src/proclaim/verification/repl_orchestrator.py` | Changed `DEFAULT_BASE_URL` to `https://api.z.ai/api/paas/v4/`. Added `ZAI_API_KEY` to env-var resolution order. |
| `scripts/verification/demo_evidence_programming.py` | Changed `GLM_OPENAI_BASE` to `/api/paas/v4/`. Added `ZAI_API_KEY` to env-var resolution. |
| `pyproject.toml` | Added `[project.optional-dependencies] fulltext = ["pymupdf>=1.24", "requests"]`. INDRA noted as manual-install-only (pysb build fails on Python 3.12). |
| `doc/ClaudeAgentSDK/CLAUDE.md` | Updated model names table with correct Anthropic + OpenAI-compatible endpoint columns. Added warning against `/api/openai`. |
| `.github/copilot-instructions.md` | Updated LLM client pattern with correct Z.AI endpoint. Added `ZAI_API_KEY` to env vars. Added verification subsystem module listing. |
| `doc/implementation_plan.md` | Fixed example endpoint URL. |

## Architecture

```
evidence_api.get_full_text_article(paper, state)
        │
        ▼
full_text.fetch_full_text(pmid, doi, title, max_chars)
        │
        ├─ Layer 1: _fetch_pmc(pmid)
        │    PMC E-utilities efetch XML → extract <body> text
        │    Cross-validates title via _title_overlap()
        │
        ├─ Layer 2: _fetch_indra(pmid)        [optional]
        │    indra.literature.get_full_text(pmid, 'pmid')
        │    Skipped if indra not installed
        │
        └─ Layer 3: _fetch_unpaywall_pdf(doi)  [optional]
             Unpaywall API → PDF URL → pymupdf text extraction
             Requires UNPAYWALL_EMAIL env var
```

## Key Design Decisions

- **Layered fallback instead of single source:** PMC OA only covers ~30% of biomedical papers. Adding INDRA (Elsevier, PMC full) and Unpaywall broadens coverage without breaking existing functionality.
- **INDRA as import-time optional:** INDRA's dependency chain pulls in `pysb>=1.3.0` which fails to build on Python 3.12 (broken `ez_setup.py`). Making it optional means the module works without INDRA installed.
- **DOI on PaperRecord:** Needed for Unpaywall lookups (Layer 3). PubMed E-utilities return DOIs in article metadata, now captured during search.
- **Title cross-validation for PMC:** PMC efetch sometimes returns the wrong article for a given PMID-to-PMCID mapping. `_title_overlap()` compares word sets to discard mismatches.
- **16K char truncation for extract_facts:** GLM-4.6 supports 200K context. The previous 6K limit threw away most full-text content, defeating the purpose of retrieval.

## Bug Fixes

### Fixed: Wrong Z.AI OpenAI-compatible endpoint
- **Symptom:** All `llm()` subagent calls returned empty strings → 0 facts extracted → manual fact entry required
- **Root cause:** Endpoint was `https://api.z.ai/api/openai` — this path does not exist
- **Fix:** Changed to `https://api.z.ai/api/paas/v4/` in 4 files
- **Verification:** Endpoint now returns proper API responses (confirmed with test call; got 429 "insufficient balance" rather than connection error)

### Fixed: Full-text retrieval limited to PMC OA
- **Symptom:** Only 5 of 52 papers returned body text; rest fell back to abstracts (800-2000 chars)
- **Root cause:** Single-source PMC XML fetch with no fallback
- **Fix:** New `full_text.py` with 3-layer fallback chain
- **Verification:** `_fetch_pmc('35562995')` now returns 17,172 chars of body text for the SRC/CTTN paper

---

## Remaining Issues (from notebook analysis)

### 1. Z.AI account balance depleted
The entire first round of fact extraction (12 papers) failed with `429 Insufficient balance`. This is an account/billing issue, not a code bug — the endpoint URL fix is correct. Account needs recharging.

### 2. Sufficiency checker permanently stuck on INSUFFICIENT
Every sufficiency check (iterations 2–6) returned the same single gap: `contradictory_evidence: Conflicting evidence: no clear majority stance`. Confidence stayed at ~0.47–0.52 despite growing from 71 to 173 facts. The heuristic counts irrelevant REFUTE/NEUTRAL facts (e.g., "paper does not mention cortactin") equally with genuine refutations. Needs: filter or reclassify off-topic stances before computing the ratio, or switch to an LLM-based sufficiency judgment.

### 3. `EvidenceState.load()` crashes with directory path
Passing a workspace directory to `EvidenceState.load()` raises `IsADirectoryError`. The method expects `evidence_state.json` but should auto-append the filename when given a directory.

### 4. PMC title cross-validation false positives
`_title_overlap()` rejected valid PMC articles for PMIDs 35060920 and 34692908, forcing fallback to abstracts. The word-overlap threshold or matching logic may be too strict.

### 5. Misleading "full text" logging
Several papers log `Full text retrieved for PMID ...: <2000 chars` then use the abstract path. The full_text module returns short PMC content that `evidence_api` then treats as abstract-length. Either the logging or the min-length threshold should be adjusted.

### 6. ImportError in notebook cell
Cell 32 imports `add_facts_from_dicts` from `subagents` — it actually lives in `evidence_api`. This is a notebook-specific typo, not a library bug.

### 7. Zero facts from some full-text papers
PMID 22811067 (16K chars full text) and PMID 11165943 (abstract) yielded 0 facts. The extraction prompt may need better handling of papers tangential to the claim.
