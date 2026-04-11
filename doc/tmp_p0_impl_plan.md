---
status: tmp — delete after P0 lands and testing passes
scope: P0 only from doc/accuracy_note.md (§ "P0 — block experiments until done")
audience: AI coding agent
---

# P0 implementation plan (tmp)

This plan implements **only** the P0 items from
[doc/accuracy_note.md](accuracy_note.md). P1 / P2 are explicitly out of scope
— after P0 lands, the user will run a test, review the result, and then
ask for the next plan.

The accuracy note and cost report were written in a different test branch.
Code on this branch (`refactor/cost-accuracy`) may differ slightly from the
notes — the *core problems* still exist, but file line numbers in the
notes are not authoritative. Verify the current state of each file before
editing.

---

## Scope

Two tasks from accuracy_note.md §P0:

1. **Model-id fix** — config uses `./qwen3.5-9b`; make it agree with what
   vLLM serves so the fact extractor never loops on 404s.
2. **Verbosity reduction** — stop spamming `print()` with per-item progress,
   HTTP traces, retry messages, and spaCy warnings. Keep only a single
   one-line result summary per top-level call in `print()`. Everything
   else must go to `logger` (captured in `run.log`, not in `nb_execute`
   tool-result text).

**Do not touch** anything in the P1 / P2 / "out of scope" sections of
accuracy_note.md. In particular: don't modify `identify_gaps`, don't
remove tools from `function_docs`, don't change the system prompt, don't
add an early-exit on sufficiency plateau, don't change `max_iterations`,
don't merge notebook cells.

---

## Task 1 — model-id fix

### Current state (verified)

- [experiments/configs/signor_eval_config.yaml:32](experiments/configs/signor_eval_config.yaml#L32) sets `subagent_model: ./qwen3.5-9b`
- [experiments/configs/test_config.yaml:37](experiments/configs/test_config.yaml#L37) already sets `subagent_model: qwen3.5-9b` (no prefix — good)
- Both launchers already register three aliases via `--served-model-name`:
  - [scripts/vllm_node_setup.sh:147](scripts/vllm_node_setup.sh#L147)
  - [scripts/start_vllm_workstation.sh:229](scripts/start_vllm_workstation.sh#L229)
  registers `"${MODEL_NAME}"`, `"./${MODEL_NAME}"`, `"${MODEL}"` (full path)
- [src/pkevolve/verification/llm_factory.py:100](src/pkevolve/verification/llm_factory.py#L100) passes `model` verbatim to the OpenAI client

So the launchers should already tolerate `./qwen3.5-9b`. The note still
flags it as a bug because the cost-debug run hit the 404. Root cause in
this branch is most likely the **config**, not the launcher. Fix the
config, and add a defensive assertion.

### Changes

1. **Drop the `./` prefix in the config** so it matches the plain form
   that works even if a launcher path skips the alias registration:

   - [experiments/configs/signor_eval_config.yaml:32](experiments/configs/signor_eval_config.yaml#L32):
     `subagent_model: ./qwen3.5-9b` → `subagent_model: qwen3.5-9b`
   - Update the example comments on lines 28–31 of the same file to show
     the non-prefixed form as the recommended canonical style.

2. **Audit every other `*_config.yaml`** in [experiments/configs/](experiments/configs/)
   for `subagent_model:` entries with a `./` prefix. There is currently
   one other baseline config family — normalise any that still use the
   prefixed form. Do NOT touch configs that point at cloud APIs
   (claude-*, glm-*, etc.) — only the local-vLLM qwen entries.

3. **Add a one-line preflight check** in
   [src/pkevolve/verification/llm_factory.py](src/pkevolve/verification/llm_factory.py)
   so the next time a mismatch happens it fails loudly on the first call
   instead of silently burning a 3-retry loop per paper:

   - In `make_llm`, right before returning, issue a single lightweight
     `client.models.list()` call wrapped in `try/except`. If the call
     succeeds and `model` is not in the returned id set, raise
     `ValueError(f"make_llm: model {model!r} not served by {base_url}. "
     f"Available: {sorted(available)}")`.
   - If the `models.list()` call itself raises (endpoint unreachable,
     auth error), swallow it silently — we don't want to block startup
     on endpoints that don't implement `/v1/models`.
   - Skip the preflight entirely when `base_url` points at
     `api.anthropic.com` (Anthropic doesn't expose a models list
     compatible with OpenAI's shape). Check by substring match.

### Acceptance

- A fresh run of any config in `experiments/configs/` with a local vLLM
  qwen backend does **not** produce any `Error code: 404 - {'error':
  {'message': 'The model ... does not exist'` in `run.log` or in
  `nb_execute` tool results.
- Running `make_llm(base_url="http://localhost:8000/v1/", api_key="EMPTY",
  model="nonexistent-model")` raises `ValueError` immediately (instead of
  returning a callable that fails 3× per call).

---

## Task 2 — verbosity reduction

### Background (non-obvious part)

The stated files to sweep in accuracy_note.md are:

- `evidence_api.py`
- `subagents.py`
- `feature_tools.py`
- `full_text.py`

Current state on this branch:

| file | `print(` count | logger already imported? |
|---|---:|:---:|
| `evidence_api.py` | 82 | yes (line 33) |
| `subagents.py` | 2 | yes |
| `feature_tools.py` | 0 | yes — **no work needed** |
| `full_text.py` | 0 | yes — **no work needed** |

So the real work is in [evidence_api.py](src/pkevolve/verification/evidence_api.py)
and [subagents.py](src/pkevolve/verification/subagents.py). The note's
list is still correct as a sweep target — `feature_tools.py` and
`full_text.py` just happen to already be clean on this branch.

### Rule for keeping vs. moving `print()`

For each `print(` call, decide:

- **Keep as `print()`** if and only if it is a **single, one-line
  terminal summary of a top-level user-callable function**, produced
  exactly once per invocation. Examples of things to keep:
  - `search_pubmed`: one-liner `"PubMed: found N, added M new. PMIDs: ..."`
    — keep (shortened, see below).
  - `extract_and_add_facts`: one-liner `"extract_and_add_facts: done.
    facts=N papers=M"` — keep (replacing the multi-line "processing…
    / ✓ / ✗ / completed" block).
  - `check_sufficiency` (any backend): one-liner
    `"check_sufficiency: label=<sufficient|insufficient>
    prob=<x.xxx> papers=<N>"` — keep.
  - `filter_papers_by_stance`: one-liner `"filter_papers_by_stance:
    kept N, removed M"` — keep.
  - `compress_evidence`: one-liner
    `"compress_evidence: <before> → <after> tokens"` — keep.
  - `emit_verdict`: the existing `"Verdict emitted: ..."` line — keep.
  - `setup_kernel`: the existing `"Kernel ready. ..."` line — keep.

- **Move to `logger.info`** if it is per-item progress, per-call banner
  headers, or multi-line dumps that the agent does not need to see
  turn-to-turn:
  - Per-PMID lines (`"  ✓ PMID ...: extracted N facts"`,
    `"_extract_and_add_facts_single: using abstract..."`,
    `"Paper {pmid} not found..."`, `"Full text retrieved for PMID..."`)
  - Per-iteration banner blocks (`"=== SUFFICIENCY CHECK ==="`, MLP
    feature dumps, gap lists) inside `_check_sufficiency_mlp`,
    `_check_sufficiency_llm`, `_check_sufficiency_haiku` — the agent
    gets the same information back in structured form via the
    `SufficiencyResult` object it receives.
  - `"=== SUFFICIENCY HISTORY ==="` block in `get_sufficiency_history`
    — replace with a single-line summary, push the table into
    `logger.debug`.
  - `"=== PAPER FILTERING ==="` block in `filter_papers_by_stance`
    — same treatment.

- **Move to `logger.warning`** if it carries a `⚠️` / `⚠` / `Error`
  prefix. Drop the emoji; use the warning level to convey severity.
  - `"⚠️ No papers found in initial search..."`
  - `"⚠ extract_and_add_facts: error processing PMID ..."`
  - `"Error querying elink ..."`, `"Error fetching related article ..."`

- **Move to `logger.debug`** if it exists purely for post-hoc debugging
  (the `"[LLM Query] ..."` prefix in `search_pubmed_llm`, the `"Refined
  queries: ..."` line, the `print(summary)` in `get_evidence_summary`,
  the kernel-internal `_extract_and_add_facts_single` traces).

### Concrete hot-spots to rewrite

Use these as landmarks; do **not** limit changes to only these. After
editing, grep for any remaining `^\s*print\(` in `evidence_api.py` and
`subagents.py` and triage each per the rule above.

1. **`extract_and_add_facts`** ([src/pkevolve/verification/evidence_api.py:1188](src/pkevolve/verification/evidence_api.py#L1188))
   — this is the single worst offender (cost report §4: three 12K-char
   tool results came from this function's `print()` stream). Replace
   the seven `print()` calls (lines ~1234, 1244, 1247, 1260, 1262,
   1270) with:
   - `logger.info` for "processing N papers with K workers"
   - `logger.debug` for the per-PMID ✓/✗ lines
   - `logger.warning` for the "error processing PMID" line
   - Keep exactly one `print()` at the end:
     `print(f"extract_and_add_facts: done. facts={total_facts} papers={len(pmids_to_process)}")`

2. **`_check_sufficiency_mlp` / `_llm` / `_haiku`** ([src/pkevolve/verification/evidence_api.py:1820](src/pkevolve/verification/evidence_api.py#L1820)
   onwards, ~60 lines of prints across three variants) — replace each
   multi-line banner block with:
   - `logger.info` for the full block
   - Keep exactly one `print()` at the end of each variant:
     `print(f"check_sufficiency[{backend}]: label={label} prob={prob:.3f} "
           f"papers={current_paper_count} gaps={len(gaps)}")`

3. **`search_pubmed` / `search_pubmed_llm` / `search_for_gap`**
   (around [src/pkevolve/verification/evidence_api.py:367](src/pkevolve/verification/evidence_api.py#L367)
   and :379) — the multi-line `print(...)` on line 372 and the three
   `print` calls on lines 404, 408, 411 collapse to one line each.

4. **`refine_search_for_failed_papers`** (around line 484) — three
   prints, all per-call. Collapse to one `"refined_search: queries=N
   new_pmids=M"` summary.

5. **`_extract_and_add_facts_single`** (around [src/pkevolve/verification/evidence_api.py:1150](src/pkevolve/verification/evidence_api.py#L1150))
   — all four prints are per-PMID debug output. Move **all** to
   `logger.debug`. This function is internal; the wrapper (task 1
   item above) owns the user-visible summary.

6. **`get_sufficiency_history`** (around line 2060) — replace the
   seven prints building an ASCII table with `logger.debug` for the
   full table and one `print()` line giving the trend summary.

7. **`filter_papers_by_stance`** (around line 2175) — four prints.
   Collapse to one.

8. **`update_synthesis` / `add_conflict`** (around lines 1089 / 1108)
   — one `print()` each, per-call. Replace with
   `logger.info` only; don't print anything (these are the "Category A
   dead code" tools from the accuracy note, but since P1 is out of
   scope we just quiet them down; do NOT remove the functions).

9. **`subagents.py` lines 172 and 415** — both are debugging prints
   that duplicate a `logger.warning` or are pure debug output. Move
   to `logger.debug` (line 172) and delete the redundant `print()`
   on line 415 — the `logger.warning(...)` two lines above already
   covers it.

### One surgical addition: silence third-party noise at kernel start

The cost-debug run captured `HTTP Request: POST ...` (from `httpx`'s
INFO logger) and spaCy's `W095` `UserWarning` inside `nb_execute`
results. These come from third-party libraries, not our code — moving
our own prints won't silence them. The surgical fix is to install a
few suppressions **once**, inside
[`setup_kernel`](src/pkevolve/verification/evidence_api.py#L2280),
so every kernel launched by `notebook_mcp` inherits them.

Add at the very top of `setup_kernel`, before any other imports or
state construction:

```python
import logging as _logging
import warnings as _warnings
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("httpcore").setLevel(_logging.WARNING)
_logging.getLogger("urllib3").setLevel(_logging.WARNING)
_warnings.filterwarnings(
    "ignore",
    message=r".*\[W095\].*",
    category=UserWarning,
)
```

Do **not** touch `logging.basicConfig` in
[evidence_programming.py:329](src/pkevolve/verification/evidence_programming.py#L329)
— that is the CLI entry point and already wires a FileHandler to
`run.log`; leaving it as-is preserves the human-readable log.

### Acceptance

Run these checks after the edits:

1. `grep -cE '^\s*print\(' src/pkevolve/verification/evidence_api.py`
   should drop from 82 to no more than **~12** — one per top-level
   user-callable function's terminal summary, plus `setup_kernel`'s
   `"Kernel ready."` line and the two verdict lines in `emit_verdict`.
   If it's higher, something was missed; if it's much lower, a
   legitimate summary line got deleted.

2. `grep -cE '^\s*print\(' src/pkevolve/verification/subagents.py`
   should be **0**.

3. Re-run a single claim from the test config (`test_config.yaml` is
   already set up; user will drive this). In the resulting
   `evidence_report.ipynb`, the biggest `nb_execute` tool result by
   char count should be **under 3,000 chars** (was 12,337 before).
   `extract_and_add_facts`'s cell output should be a single line
   like `extract_and_add_facts: done. facts=12 papers=15`.

4. `run.log` should still contain all the per-PMID lines, HTTP retry
   lines (if any), and classifier feature dumps — the information is
   preserved, just not replayed through the orchestrator's context
   window.

---

## Out of scope (reminder)

Do **not** do any of the following in this branch — they are P1/P2 and
will be scoped separately:

- Upgrade `identify_gaps` / extend its signature / enrich its prompt
- Encode the upstream-fix principle in the system prompt
- Plateau-based early exit on sufficiency
- Remove any tool from `function_docs` (no `update_synthesis`,
  `add_conflict`, `compress_evidence`, `get_paper_text`, etc. removals
  — just quiet their prints where applicable)
- Trim verdict definitions or workflow rules in the system prompt
- Collapse `nb_render_*` HTML tables
- Any per-iteration `query()` refresh / notebook serialisation change
- Any `max_iterations` change

---

## Handoff

When done, report:
- `git diff --stat` summary
- `grep -cE '^\s*print\(' src/pkevolve/verification/{evidence_api,subagents}.py`
- Confirmation that the preflight in `make_llm` triggers on a bad model id
- Any file where the sweep turned up a `print()` that didn't fit the
  rule cleanly (so the user can decide case-by-case)

Then stop. User will run a test round before the next plan is written.
