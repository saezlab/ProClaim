# Changes — 2026-03-30

**Branch:** `fix/pid_search`

---

## 1. Auto-timestamped Output Directory

### Summary

Removed the hardcoded `notebook_demo` default output directory from `VerificationSettings`. When `output_dir` is omitted from the YAML config, the system now generates a unique, human-sortable directory name per run (`results/verification/YYYYMMDD_HHMMSS_xxxxxxx/`). The timestamp is computed once at instance-creation time so that all path properties (`resolved_workspace`, `resolved_notebook_path`, etc.) resolve consistently within a single run.

### Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/config.py` | Replaced hardcoded `"notebook_demo"` fallback in `resolved_output_dir` with a `PrivateAttr`-cached timestamp; added `uuid4` suffix for parallel-process safety |
| `experiments/test_config.yaml` | Removed `output_dir` field — no longer required |

### Key Design Decisions

- **`PrivateAttr` for caching** — `resolved_output_dir` is a `@property`, so a plain `datetime.now()` call would produce a different value on every access, creating multiple directories per run. Storing the timestamp as a `PrivateAttr` with `default_factory` pins it to the moment the `VerificationSettings` object is constructed.
- **Seconds + 6-hex UUID suffix** — seconds-precision alone allows collisions when subprocesses start within the same second. Appending `uuid4().hex[:6]` (1-in-16M collision probability) makes parallel batch runs safe without sacrificing human readability.
- **Format: `YYYYMMDD_HHMMSS_xxxxxxx`** — lexicographic sort equals chronological sort, making `ls` and log inspection straightforward.

---

## 2. Semantic Scholar Search in Agent Prompt

### Summary

The verification agent's system prompt and API function list did not expose the three Semantic Scholar functions (`search_semantic_scholar`, `search_semantic_scholar_recommendations`, `expand_via_citations`). The agent could only use PubMed, missing keyword search, recommendation-based discovery, and citation chaining via S2.

### Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/evidence_api.py` | Added `search_semantic_scholar`, `search_semantic_scholar_recommendations`, `expand_via_citations` to the `_api_funcs` list so their signatures are injected into the agent's system prompt |
| `src/pkevolve/verification/evidence_programming.py` | Updated `SYSTEM_PROMPT` workflow step 3 (initial search) to include S2 keyword search at iteration 0; updated step 12 (gap-filling) to list S2 recommendations and citation chaining as strategies |

---

## 3. Citation Chaining: Store Reference DOIs on PaperRecord

### Summary

Reference DOIs from JATS XML `<back><ref-list>` were previously appended as a text block to `paper.full_text`, polluting the body text sent to fact-extraction subagents. Refactored to store DOIs as a structured `list[str]` on `PaperRecord.reference_dois`, persisted in `evidence_state.json` via Pydantic serialization. `expand_via_citations` now reads DOIs directly from this field instead of regex-scanning full text.

### Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/data_models.py` | Added `reference_dois: list[str] = Field(default_factory=list)` to `PaperRecord` |
| `src/pkevolve/verification/full_text.py` | `_extract_reference_dois()` returns `list[str]` instead of text block; `_fetch_pmc()` and `_fetch_europepmc()` return `tuple[Optional[str], list[str]]`; `fetch_full_text()` returns `tuple[Optional[str], list[str]]` — body text and DOIs separately |
| `src/pkevolve/verification/evidence_api.py` | `get_full_text_article()` unpacks `(text, ref_dois)` and stores `paper.reference_dois`; `expand_via_citations()` reads from `paper.reference_dois` instead of regex-scanning `full_text` |

### Key Design Decisions

- **Structured field over text appending** — DOI strings carry no scientific content; appending them to `full_text` wastes ~8% of context for fact extraction. A `list[str]` field is cleaner, serializes to JSON automatically, and decouples citation chaining from text processing.
- **Tuple return from `fetch_full_text`** — Layers 1 and 1b (JATS XML sources) provide reference DOIs; other layers (S2 PDF, INDRA, Unpaywall, PubMed abstract) return an empty list. The tuple `(text, reference_dois)` keeps the API uniform across all layers.
- **No regex scanning needed** — `expand_via_citations` previously used `_DOI_RE.findall(paper.full_text)` which was fragile (dependent on DOIs appearing in body text). Structured DOIs from `<pub-id pub-id-type="doi">` elements are authoritative.

### Verification

End-to-end test on PMID 24365180: `get_full_text_article` stores 47 DOIs on `paper.reference_dois`, `full_text` is clean (no `References (DOIs):` block), and `expand_via_citations(max_per_paper=5)` adds 5 new papers from S2 lookups.

---

## 4. `fetch_full_text`: Return Untruncated Text

### Summary

`fetch_full_text` was silently slicing every returned text to `max_chars` (default 50 000). This discarded potentially evidence-bearing content without any notification, making it impossible to know whether fact extraction was working on a truncated corpus. The function now returns the full retrieved text regardless of length, and logs a `WARNING` when the length exceeds `max_chars` so operators can tune the limit if needed.

### Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/full_text.py` | Replaced six `return text[:max_chars], ...` call-sites with a shared `_maybe_warn_and_return` inner helper; updated `max_chars` docstring to reflect new semantics (warn-only, no truncation) |

### Key Design Decisions

- **Inner helper `_maybe_warn_and_return`** — avoids repeating the length check at every early-exit point while keeping the logic local to the function (not worth a module-level helper for a one-off pattern).
- **`max_chars` parameter kept** — removing it would be a breaking API change. It now serves as a "soft limit" threshold for the warning, which lets callers still configure at what size they want to be alerted.
- **`WARNING` level** — chosen over `INFO` because exceeding the previous hard limit is actionable (the caller may need to increase context-window budget or chunk the text).

---

## 5. Disable `expand_via_citations` in Agent Prompt

### Summary

Removed `expand_via_citations` from the agent's visible API surface. The function itself is retained in `evidence_api.py` for future use, but the agent can no longer discover or call it. Citation chaining needs further reliability improvements before being re-enabled.

### Modified Files

| File | Change |
|------|--------|
| `src/pkevolve/verification/evidence_api.py` | Removed `expand_via_citations` from the `_api_funcs` list — signature is no longer injected into the system prompt |
| `src/pkevolve/verification/evidence_programming.py` | Removed step 12c (`expand_via_citations`) from the gap-filling workflow; former step 12d (formulate_gap_queries) renumbered to 12c |

### Key Design Decisions

- **Function kept, exposure removed** — the implementation is preserved so the feature can be re-enabled with a one-line change once the underlying reliability issues are resolved.
- **No API breakage** — callers that import `expand_via_citations` directly from `evidence_api` are unaffected.
