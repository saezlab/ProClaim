# RLM REPL Post-Integration Hardening — 2026-02-24

**Branch:** `RLM`

## Summary

Five rounds of diagnose → fix → re-run on the evidence programming REPL
system resolved 18 runtime issues discovered during end-to-end demo runs
(`demo_evidence_programming.py --claim "SRC directly down-regulates CTTN"`).
Fixes span structural wiring, fact extraction quality, PMC full-text
retrieval correctness, LLM callable robustness, and schema visibility for
the orchestrator LLM. All changes are backward-compatible; no API
signatures were altered.

## Modified Files

| File | Changes |
|------|---------|
| `src/proclaim/verification/evidence_state.py` | Added `checkpoint_save()` auto-persistence; added `extracted_pmids: list[str]` field |
| `src/proclaim/verification/evidence_api.py` | Added `schema_docs()` with auto-generated Pydantic field listings including `PaperRecord`; field alias normalization (`statement`→`text`); fact deduplication; `add_facts_from_dicts()` rejects unknown PMIDs; `extract_and_add_facts()` tries full text before abstract and tracks `extracted_pmids`; `get_full_text_article()` filters elink by `pubmed_pmc` LinkName and cross-validates PMC text by title word overlap (≥25%) |
| `src/proclaim/verification/kernel_runner.py` | `inject_prelude()` now injects a robust `llm()` callable with 3× retry, exponential backoff, and `None`/empty `resp.choices` guard |
| `src/proclaim/verification/subagents.py` | `extract_facts()` returns `[]` if `llm()` returns empty string instead of passing `None` to parser |
| `src/proclaim/verification/data_models.py` | `PaperRecord.abstract` default changed from required to `""` (allows synthetic/summary records) |
| `src/proclaim/verification/repl_orchestrator.py` | System prompt updated with "Grounded Evidence Only" anti-fabrication rules and verdict quality gate |
| `scripts/verification/demo_evidence_programming.py` | SYSTEM_PROMPT `llm()` template updated with retry/backoff pattern matching kernel prelude; anti-fabrication rules added |
| `doc/implementation_plan.md` | §2.1 schema updated (`abstract` default); §4.2 `llm()` example replaced with robust pattern; new §8 Stage 3.5 documents all 18 fixes |

## Bug Fixes

### Round 1 — Structural Wiring (6 fixes)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Empty `evidence_state.json` on disk | State never persisted from kernel memory | Added `checkpoint_save()` to `evidence_state.py`, called from prelude |
| `KeyError: 'text'` in `add_facts_from_dicts` | LLM produced `statement` key instead of `text` | Added field alias normalization (`statement`, `fact_text`, `evidence`, `description` → `text`) |
| Coverage always 0 | No subclaim decomposition; `coverage` dict stayed empty | Documented in schema docs; LLM now sets subclaims properly |
| Duplicate facts accumulated | No dedup check when adding facts | `add_facts_from_dicts()` skips facts with identical `(text, source_pmid)` |
| Stale disk renderers | Renderers read from disk but state was only in memory | Renderers now work from `evidence_state.json` written by `checkpoint_save()` |
| No `llm()` in kernel | Subagent functions couldn't call LLM | `inject_prelude()` wires `llm()` callable from `llm_base_url`/`llm_api_key`/`llm_model` params |

### Round 2 — Extraction Quality (4 fixes)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Facts from abstracts only | `extract_and_add_facts` never tried full text | Now calls `get_full_text_article()` first, falls back to `get_paper_text()` |
| Facts cite unknown PMIDs | LLM fabricated PMIDs not in `state.papers` | `add_facts_from_dicts()` validates `source_pmid ∈ state.papers` (warn) |
| LLM uses wrong attribute patterns | No schema docs in prompt | `schema_docs()` auto-generates field listings from Pydantic models |
| LLM fabricates facts | No anti-fabrication instruction | "CRITICAL: Grounded Evidence Only" section added to both system prompts |

### Round 3 — PMC Full-Text Retrieval (3 fixes)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Wrong PMC article retrieved | elink returns `pubmed_pmc_refs` (papers that *cite* the target) | Filter elink results to `LinkName == "pubmed_pmc"` only |
| Mismatched full text accepted | No validation that PMC content matches the requested paper | Title cross-validation: discard if < 25% word overlap between PubMed title and PMC text title |
| Facts with bad PMIDs still added | Round 2 only warned, didn't reject | `add_facts_from_dicts()` now skips (not just warns) facts with unknown PMIDs |

### Round 4 — LLM Callable Robustness (3 fixes)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `TypeError: 'NoneType' object is not subscriptable` | `resp.choices` is `None`/empty; `resp.choices[0]` crashes | `llm()` guards against `None`/empty choices, retries 3× with exponential backoff |
| All 12 papers → 0 facts extracted | Single crash in `llm()` kills entire extraction loop | `llm()` returns `""` on exhaustion (graceful degradation) |
| `extract_facts` crashes on `None` | `_parse_facts_response` receives `None` when `llm()` fails | `extract_facts()` returns `[]` if response is empty |

### Round 5 — Schema Visibility (2 fixes)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `ValidationError: abstract Field required` | `PaperRecord.abstract` had no default; LLM omitted it for synthetic records | Default changed to `""` |
| `ValidationError: authors Input should be a valid list` | LLM passed `"Author Name"` instead of `["Author Name"]` | `schema_docs()` now includes `PaperRecord` fields with types and defaults; pitfall hint added |

## Key Design Decisions

- **Retry with backoff over fail-fast**: The `llm()` callable retries 3× with `2^attempt` second delays. An empty-string return is preferred over raising — it lets `extract_facts` return `[]` gracefully and the loop continues to the next paper.
- **Reject over warn for bad PMIDs**: Round 2 only warned when `source_pmid ∉ state.papers`. Round 3 upgraded this to rejection (skip the fact entirely) because warnings were ignored by the LLM and polluted the fact store.
- **Title cross-validation for PMC text**: A lightweight check (≥ 25% word overlap between PubMed title and PMC first-line title) catches the elink `pubmed_pmc_refs` misrouting without requiring a separate API call.
- **Schema docs auto-generated from Pydantic**: `schema_docs()` uses `model_fields` introspection so the prompt stays in sync with the code. Adding a field to `PaperRecord` or `Fact` automatically updates the LLM's system prompt.
- **`abstract` defaults to `""`**: This is a deliberate relaxation. Real PubMed records always have abstracts (populated by the search functions), but synthetic summary records created by the LLM (e.g., `SEARCH_SUMMARY`) don't need one.
