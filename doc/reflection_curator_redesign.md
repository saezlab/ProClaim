# Reflection & Curator Redesign

## Background and Motivation

ProClaim's workbook-based verification loop (introduced in the refactor/workbook branch)
regressed accuracy from 0.75 to 0.65 on the SIGNOR benchmark compared to the prior
no-workbook version.

Root-cause analysis identified three connected problems:

1. **Reflection uses hardcoded enum classifications** (`retrieval`, `extraction`,
   `framing`, `budget`, `other`). The reflection LLM systematically misuses the
   `budget` class to mean "I believe further search is futile" rather than its
   intended meaning of "the workbook shows `forced-verdict-imminent`". This causes
   premature `emit_verdict` with zero extracted facts, which maps to REFUTE under
   the current verdict label definition.

2. **No curator agent exists.** The reflection → curation link is advisory only:
   reflection writes a 4-field record to `reflection.json`, the workbook renders it
   as Section 9 text, and the planner optionally reads it. There is no agent whose
   job is to synthesise the reflection and add actionable guidance to the workbook.
   This is the core misapplication of the ACE architecture (Generator → Reflector →
   Curator) that ProClaim was inspired by.

3. **The workbook accumulates no claim-specific wisdom.** Every turn the workbook is
   rebuilt from `evidence_state.json` alone. There is nowhere for cross-turn
   observations (e.g., "queries using MAPK9 return irrelevant kinase papers; try
   JNK2/IB2 aliases") to live. The planner must rediscover the same dead ends
   repeatedly.

This document specifies the changes needed to fix all three problems.

---

## Design Overview

The revised loop per turn:

```
1. Execute planner action → observe → append to action_ledger
2. Auto-curation from EvidenceState (deterministic, unchanged)
3. Orchestrator: detect stall signals (unchanged)
4. Reflect  — reads volatile workbook + stall signals + recent paper abstracts
            — writes richer free-form record (no enum fields)
5. Curate   — reads workbook + reflection record + recent paper abstracts
            — appends zero or more notes to workspace/curator_notes.jsonl
            — does NOT modify EvidenceState
6. Rebuild workbook (now includes Section 2b from curator_notes.jsonl)
7. Planner reads full workbook → one action
```

The three agents map to ACE roles:
- Planner ≈ Generator (executes actions)
- Reflect LLM ≈ Reflector (diagnoses what happened and why)
- Curator LLM ≈ Curator (synthesises diagnosis into workbook guidance)

---

## Scope: What Changes and What Does Not

**Changes:**
- `src/proclaim/verification/reflection.py` — schema + prompt
- `src/proclaim/verification/prompts.py` — reflection prompt overhaul + new curator prompts
- `src/proclaim/verification/workbook.py` — new Section 2b renderer + updated `build_workbook_parts` signature
- `src/proclaim/verification/evidence_programming_direct.py` — add curator call, update ledger logging
- **New file** `src/proclaim/verification/curator.py`

**Unchanged:**
- `evidence_api.py`, `evidence_state.py`, `data_models.py`
- `curation.py` (auto-curation from EvidenceState continues unchanged)
- `config.py`, `action_ledger.py`, `verdict_packet.py`, `compressor.py`
- `workbook.py` sections 1–9 content (only Section 2b is added)

---

## 1. `reflection.py` — Schema Changes

### Remove

- `class ReflectionClassification(str, Enum)` — delete entirely
- `class ReflectionNextFamily(str, Enum)` — delete entirely
- Fields from `ReflectionRecord`: `classification`, `proposed_next_family`

### Updated `ReflectionRecord`

```python
class ReflectionRecord(BaseModel):
    """Richer free-form reflection schema.

    Fields:
      diagnosis       — observable state summary: what we see right now.
      root_cause      — mechanistic explanation: WHY progress stalled or what
                        drove the current state (not just restating the state).
      key_insight     — a transferable observation the planner or curator can act
                        on this turn and future turns (e.g., alias gap, query
                        framing issue, evidence type mismatch).
      next_suggestion — concrete, specific recommendation for the planner's next
                        action.  Free-form sentence, not an enum family name.
      override_invoked — optional guardrail ID (e.g. "GR1") whose stated override
                         condition this reflection claims to satisfy. Keep as-is.
      is_stall        — True when stall signals were passed and this reflection is
                        a full diagnostic.  False on routine (light) turns.
    """
    turn: int
    timestamp: str
    diagnosis: str
    root_cause: str
    key_insight: str
    next_suggestion: str
    override_invoked: Optional[str] = None
    is_stall: bool = False

    @field_validator("diagnosis", "root_cause", "key_insight", "next_suggestion")
    @classmethod
    def _clip(cls, v: str) -> str:
        v = (v or "").strip()
        return v[:399] + "…" if len(v) > 400 else v
```

### Updated `write_reflection` signature

```python
def write_reflection(
    workspace: Path,
    *,
    turn: int,
    diagnosis: str,
    root_cause: str,
    key_insight: str,
    next_suggestion: str,
    override_invoked: Optional[str] = None,
    is_stall: bool = False,
) -> ReflectionRecord:
```

### Updated `run_reflection` tool schema

Replace the `submit_reflection` tool with this schema:

```python
{
    "type": "function",
    "function": {
        "name": "submit_reflection",
        "description": (
            "Submit a structured reflection for the current verification turn. "
            "On routine (non-stall) turns, diagnosis and next_suggestion are "
            "sufficient; root_cause and key_insight may be brief. "
            "On stall turns, provide full analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "What we observe right now (state summary, one sentence)."
                },
                "root_cause": {
                    "type": "string",
                    "description": (
                        "WHY this state occurred. On routine turns: brief (e.g., "
                        "'First turn, no evidence yet'). On stall turns: mechanistic "
                        "explanation of the failure."
                    )
                },
                "key_insight": {
                    "type": "string",
                    "description": (
                        "A transferable observation the planner or curator should act "
                        "on. On routine turns: empty string. On stall turns: "
                        "e.g., 'Papers use JNK2/IB2 but queries use MAPK9'."
                    )
                },
                "next_suggestion": {
                    "type": "string",
                    "description": (
                        "Specific, actionable recommendation for the planner's next "
                        "action. Not a category label — a concrete instruction, "
                        "e.g., 'Run extract_and_add_facts on the 3 new papers from "
                        "the S2 search' or 'Call emit_verdict — Section 7 shows "
                        "forced-verdict-imminent and 8 SUPPORT facts support the claim'."
                    )
                },
                "override_invoked": {
                    "type": ["string", "null"],
                    "description": (
                        "Guardrail ID (e.g. 'GR1') whose override condition this "
                        "reflection satisfies, or null."
                    )
                }
            },
            "required": [
                "diagnosis", "root_cause", "key_insight",
                "next_suggestion", "override_invoked"
            ]
        }
    }
}
```

### Updated `run_reflection` return

Parse the tool call and call `write_reflection` with all fields. Set `is_stall=stall_signals.any_fired`.

---

## 2. `prompts.py` — Reflection Prompt Overhaul

### Replace `REFLECTION_SYSTEM_PROMPT`

```python
REFLECTION_SYSTEM_PROMPT = """\
You are a reflection module for a scientific claim verification loop.

The loop has a planner that executes one evidence-gathering action per turn
(search, extract, populate features, check sufficiency, emit verdict, etc.).
You run every turn. Your job is to read the current workbook state and produce
a structured diagnosis that helps the planner and curator on the next turn.

## Turn types

**Routine turn** (stall signals = none):
- diagnosis: one sentence describing where we are.
- root_cause: brief (e.g. "First turn, no evidence gathered yet").
- key_insight: empty string — nothing unusual to surface.
- next_suggestion: the natural next step given the workflow state.

**Stall turn** (one or more stall signals passed):
- diagnosis: what we observe (e.g., "18 papers extracted, 0 facts found").
- root_cause: WHY this happened — do not just restate the observation.
  Good: "Queries use gene symbol MAPK8IP2 but retrieved papers describe the
  protein under scaffold aliases IB2 / JIP-2; extraction context lacks these."
  Bad: "Extraction yielded zero facts."
- key_insight: a transferable observation the planner or curator can act on
  (e.g., "Inhibition in this system may be mediated by ubiquitin-dependent
  degradation rather than direct kinase activity; look for E3-ligase papers").
- next_suggestion: concrete instruction for the planner's next action.

## Critical constraint on emit_verdict

NEVER suggest `emit_verdict` (in next_suggestion) unless Section 7 of the
workbook explicitly reads "forced-verdict-imminent" or "check-sufficiency-imminent".

If Section 7 shows "Forced-verdict risk: low" or "not yet checked", do NOT
suggest emitting a verdict even if multiple extraction rounds returned 0 facts.
In that case root_cause should explain the extraction failure and next_suggestion
should propose a different retrieval or extraction angle.

"We tried twice and found nothing" is NOT a budget justification — it is an
extraction or retrieval problem that needs a different strategy.

## Guardrail overrides

Set override_invoked to the relevant GR ID only when the reflection's root_cause
and key_insight justify the specific override condition stated in Section 2 of
the workbook. Pass null otherwise."""
```

### Replace `REFLECTION_USER_PROMPT`

```python
REFLECTION_USER_PROMPT = """\
Stall signals: {stall_signals}
(If "(none)", this is a routine turn.)

Recent paper titles and abstracts (up to {n_papers} papers currently in state):
{paper_summaries}

Current workbook state (volatile sections):
{workbook_volatile}

Submit the structured reflection."""
```

Placeholder notes:
- `stall_signals` — "; "-joined list of reason strings, or "(none)"
- `n_papers` — count of papers in state passed to this call
- `paper_summaries` — see Section 4 below for format
- `workbook_volatile` — volatile tail of workbook (unchanged from current)

---

## 3. New `curator.py`

Create `src/proclaim/verification/curator.py`.

### Data model

```python
CURATOR_NOTES_FILENAME = "curator_notes.jsonl"
_NOTE_MAX_CHARS = 1200
_MAX_NOTES_IN_WORKBOOK = 8  # show only the most recent N notes in the workbook

class CuratorNote(BaseModel):
    turn: int
    timestamp: str
    content: str           # markdown prose the planner will read
    rationale: str         # why this note is useful (not shown in workbook)
```

### Persistence

```python
def append_curator_note(workspace: Path, note: CuratorNote) -> None:
    """Append one curator note to workspace/curator_notes.jsonl."""

def load_curator_notes(workspace: Path, *, n: int = _MAX_NOTES_IN_WORKBOOK) -> list[CuratorNote]:
    """Return the last n notes, or [] if file does not exist."""
```

### `run_curator` function

```python
def run_curator(
    *,
    workspace: Path,
    workbook_volatile: str,
    reflection: ReflectionRecord,
    paper_summaries: str,
    turn: int,
    model: str,
) -> tuple[Optional[CuratorNote], Optional[dict]]:
    """Run the curator LLM and optionally append a note to curator_notes.jsonl.

    Returns (note, usage) where note is None when curator decided no update
    is needed (no_update=True), and usage is the token-usage dict for billing.
    """
```

The curator uses a forced tool call (`submit_curator_note`):

```python
_CURATOR_TOOL_NAME = "submit_curator_note"

def _build_curator_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": _CURATOR_TOOL_NAME,
                "description": (
                    "Submit a curator note to append to the workbook's Strategy Notes "
                    "section. Call with no_update=true when there is nothing useful to add."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": (
                                "Markdown text (≤1200 chars) to append to Section 2b "
                                "of the workbook. Should be directly actionable for "
                                "the planner: alias suggestions, search angle changes, "
                                "extraction framing notes, conflict resolution hints, "
                                "or gap characterisation. Omit if no_update=true."
                            )
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this note will help (internal, not shown to planner)."
                        },
                        "no_update": {
                            "type": "boolean",
                            "description": (
                                "Set true when the reflection and current state give "
                                "you nothing useful to add. The workbook will not change."
                            )
                        }
                    },
                    "required": ["no_update"]
                }
            }
        }
    ]
```

### Curator prompts (add to `prompts.py`)

```python
CURATOR_SYSTEM_PROMPT = """\
You are a curator for a scientific claim verification loop.

After each turn, a reflection module diagnoses what happened and why.
Your job is to read the reflection and the current workbook, then decide
whether to add a note to the workbook's Strategy Notes section (Section 2b)
that will help the planner on future turns.

## What to add

Add a note when the reflection's key_insight or root_cause contains information
that the planner should keep in mind across turns. Good notes:

- Alias / terminology mappings:
  "Extraction context update: papers in this area use JNK2 / IB2 / JIP-2 /
  JNK-interacting protein 2 interchangeably with MAPK9 / MAPK8IP2. Add these
  to the extraction context before the next extraction round."

- Search angle suggestions:
  "Query reframe: direct inhibition via kinase activity not found; try the
  angle of ubiquitin-mediated degradation. Suggested query terms: 'E3 ligase
  RING domain degradation [target protein]'."

- Evidence type clarifications:
  "Claim framing note: 'directly activates' in this kinase context includes
  scaffold-mediated co-localisation that increases phosphorylation. Extraction
  should accept co-localisation + activation evidence as SUPPORT."

- Gap characterisation:
  "Gap identified: no papers address the direct biochemical mechanism.
  The retrieved corpus shows indirect pathway evidence only. Consider
  searching for in vitro kinase assay or pull-down studies specifically."

- Conflict resolution guidance (when contradictory facts exist):
  "Contradiction note: PMID X reports inhibition in neuronal cells;
  PMID Y reports activation in HEK293 cells. Cell-type specificity may
  explain the conflict. Weight context-matched studies more heavily."

## What NOT to add

- Do not repeat information already visible in the workbook.
- Do not add a note if the reflection was a routine (non-stall) turn with
  no key insight (key_insight is empty or trivial).
- Do not prescribe which specific evidence API function to call — that is
  the planner's job.
- Do not add more than one note per turn (one focused note beats several vague ones).

## Paper context

You are given titles and abstracts of papers currently in the evidence state.
Use these to identify terminology gaps, alias patterns, or relevant mechanisms
that the extraction context may be missing. The reflection may have already
flagged a terminology gap — your note should operationalise it."""


CURATOR_USER_PROMPT = """\
Reflection from this turn:
- Diagnosis: {diagnosis}
- Root cause: {root_cause}
- Key insight: {key_insight}
- Next suggestion: {next_suggestion}
- Is stall turn: {is_stall}

Recent paper titles and abstracts:
{paper_summaries}

Current workbook (volatile sections, so you know what is already visible):
{workbook_volatile}

Existing curator notes already in Section 2b (do not repeat these):
{existing_notes}

Submit your curator note, or set no_update=true if nothing useful to add."""
```

---

## 4. Paper Summaries Format

Both the reflection and curator calls receive a `paper_summaries` string.

Build this in the orchestrator before calling `run_reflection` and `run_curator`:

```python
def _build_paper_summaries(state: EvidenceState, *, max_papers: int = 20) -> str:
    """Return a compact title+abstract block for up to max_papers papers in state."""
    if not state.papers:
        return "(no papers retrieved yet)"
    lines = []
    for pmid, paper in list(state.papers.items())[:max_papers]:
        title = (paper.title or "").strip() or "(no title)"
        abstract = (paper.abstract or "").strip()
        if abstract:
            abstract = abstract[:400] + ("…" if len(abstract) > 400 else "")
        else:
            abstract = "(no abstract)"
        lines.append(f"PMID {pmid}: {title}\n  {abstract}")
    return "\n\n".join(lines)
```

Place this helper either in `evidence_programming_direct.py` (local) or in `curator.py`
and import it into the orchestrator.

---

## 5. `workbook.py` — Section 2b

### New renderer

```python
CURATOR_NOTES_MAX_DISPLAY = 8  # import from curator.py or define locally

def _section_curator_notes(notes: list["CuratorNote"]) -> str:
    """Render Section 2b — accumulated claim-specific strategy notes from the curator."""
    header = "## 2b. Strategy Notes (curator-accumulated)"
    if not notes:
        return f"{header}\n- _(no curator notes yet)_"
    lines = [header]
    for note in notes:
        lines.append(f"\n[Turn {note.turn}]")
        lines.append(note.content.strip())
    return "\n".join(lines)
```

### Updated `build_workbook_parts`

Add `curator_notes: Optional[list[CuratorNote]] = None` parameter.

In the body, after `if curation_queue is None: ...`, add:
```python
if curator_notes is None:
    from proclaim.verification.curator import load_curator_notes
    curator_notes = load_curator_notes(workspace)
```

Insert `_section_curator_notes(curator_notes)` into `volatile_sections` as the **first
entry**, before `_section_header(...)`. This positions it prominently so the planner
reads it near the top of the volatile context.

```python
volatile_sections = [
    _section_curator_notes(curator_notes),   # NEW — first
    _section_header(...),
    _section_state_snapshot(state),
    ...
]
```

### Updated `_section_next_turn_guidance`

Adapt to use the new `ReflectionRecord` fields (no `classification`, no `proposed_next_family`):

```python
def _section_next_turn_guidance(reflection: Optional[ReflectionRecord]) -> str:
    header = "## 9. Next-Turn Guidance"
    if reflection is None:
        return f"{header}\n- _(no reflection recorded yet)_"

    stall_marker = " [STALL TURN]" if reflection.is_stall else ""
    lines = [
        header,
        f"- Reflection (turn {reflection.turn}{stall_marker}):",
        f"    - Diagnosis: {reflection.diagnosis}",
        f"    - Root cause: {reflection.root_cause}",
    ]
    if reflection.key_insight:
        lines.append(f"    - Key insight: {reflection.key_insight}")
    lines.append(f"    - Suggested next action: {reflection.next_suggestion}")
    if reflection.override_invoked:
        lines.append(f"    - Guardrail override invoked: {reflection.override_invoked}")
    if reflection.next_suggestion.lower().startswith("emit") or "emit_verdict" in reflection.next_suggestion:
        lines.append(
            "    - When emitting the verdict, base reasoning on Section 4a "
            "(Known Evidence Digest). Do not derive the verdict from action "
            "history, reflection diagnosis, or guardrails."
        )
    return "\n".join(lines)
```

---

## 6. `evidence_programming_direct.py` — Orchestrator Integration

### Import changes

Add to local imports inside the function (or at module level):
```python
from proclaim.verification.curator import run_curator, load_curator_notes
```

### After the reflection block (after `if reflect_record is not None: ...`)

Insert the curator phase. The curator runs unconditionally when a reflection record
exists and `evidence_state.json` is present (same gate as reflection):

```python
        # Phase 4b: curator.  Reads workbook + reflection + paper abstracts,
        # optionally appends a note to curator_notes.jsonl.
        # Failures are non-fatal; the workbook renders without curator notes.
        curator_note, curator_usage = None, None
        if reflect_record is not None:
            _paper_summaries = _build_paper_summaries(current_state)
            _curator_t0 = time.monotonic()
            try:
                curator_note, curator_usage = run_curator(
                    workspace=workspace,
                    workbook_volatile=volatile_tail,
                    reflection=reflect_record,
                    paper_summaries=_paper_summaries,
                    turn=call_count,
                    model=agent_model,
                )
            except Exception as exc:
                logger.warning("Curator LLM call failed: %s", exc)
            _curator_latency = time.monotonic() - _curator_t0

            if curator_usage:
                tracker.record(
                    "curator_llm",
                    input_tokens=curator_usage.get("prompt_tokens", 0) or 0,
                    output_tokens=curator_usage.get("completion_tokens", 0) or 0,
                    latency=_curator_latency,
                    call_number=call_count,
                    cache_read_tokens=curator_usage.get("cache_read_input_tokens", 0) or 0,
                    cache_write_tokens=curator_usage.get("cache_creation_input_tokens", 0) or 0,
                )
            if curator_note is not None:
                logger.info(
                    "Curator note appended (turn %d): %.80s",
                    call_count,
                    curator_note.content,
                )
```

### Ledger logging update

In the `_record_action` call for the `reflect` action, update `arguments` to use
the new field names:

```python
_record_action(
    workspace=workspace,
    turn=call_count,
    tool_name="reflect",
    arguments={
        "is_stall": reflect_record.is_stall,
        "override_invoked": reflect_record.override_invoked or None,
    },
    raw_output=reflect_record.diagnosis,
    before=_reflect_snap,
    after=_reflect_snap,
)
```

### Pass `paper_summaries` to `run_reflection`

Update the `run_reflection` call to include `paper_summaries`:

```python
_paper_summaries = _build_paper_summaries(current_state)
reflect_record, reflect_usage = run_reflection(
    workspace=workspace,
    workbook_volatile=volatile_tail,
    stall_signals=signals,
    paper_summaries=_paper_summaries,
    turn=call_count,
    model=agent_model,
)
```

And update `run_reflection` signature accordingly:
```python
def run_reflection(
    *,
    workspace: Path,
    workbook_volatile: str,
    stall_signals: StallSignals,
    paper_summaries: str,
    turn: int,
    model: str,
) -> tuple[Optional[ReflectionRecord], Optional[dict]]:
```

---

## 7. Guardrails in Section 2 — Minimal Update

The static guardrails (GR1–GR6) remain in the stable prefix unchanged in structure,
but the `budget` override description in the prompt-facing guardrail text should be
clarified. In `DEFAULT_GUARDRAILS` in `workbook.py`, update GR5:

```python
{
    "id": "GR5",
    "rule": (
        "Do not consume the final turns on retrieval if verdict readiness is "
        "already high enough for check_sufficiency plus emit_verdict."
    ),
    "source": "migrated from URGENT-turns prompt branch",
    "override": (
        "turns remaining > 2 and sufficiency still insufficient, AND "
        "Section 7 does not yet show forced-verdict-imminent"
    ),
},
```

No other guardrail changes are needed. The curator's Section 2b notes serve as the
dynamic, claim-specific constraint layer.

---

## 8. `__all__` and Module Exports

`curator.py` should export:
```python
__all__ = [
    "CURATOR_NOTES_FILENAME",
    "CuratorNote",
    "append_curator_note",
    "load_curator_notes",
    "run_curator",
]
```

`reflection.py` should remove from `__all__`:
- `ReflectionClassification`
- `ReflectionNextFamily`

And keep/add:
- `ReflectionRecord`, `StallSignals`, `write_reflection`, `load_reflection`,
  `clear_reflection`, `detect_stall_signals`, `run_reflection`, `REFLECTION_FILENAME`

---

## 9. Error Handling and Non-Fatal Guarantees

Both `run_reflection` and `run_curator` must never crash the outer orchestrator loop.
Pattern to follow (already used for reflection — replicate for curator):

```python
try:
    curator_note, curator_usage = run_curator(...)
except Exception as exc:
    logger.warning("Curator LLM call failed: %s", exc)
    curator_note, curator_usage = None, None
```

If curator notes cannot be loaded for the workbook render, render with an empty list:
```python
try:
    curator_notes = load_curator_notes(workspace)
except Exception:
    curator_notes = []
```

---

## 10. Testing Checklist

After implementation, verify with a smoke run:

```bash
uv run python experiments/run_signor_eval.py \
    --input-csv datasets/signor.csv \
    --config experiments/configs/signor_smoke_config.yaml \
    --limit 3 --reps 1 --run-tag curator-smoke
```

Check in the smoke run's workspace:
- `curator_notes.jsonl` exists and contains valid JSON lines
- `workbook.md` contains Section 2b with at least one note after an extraction stall
- `reflection.json` has keys `root_cause`, `key_insight`, `next_suggestion` (no `classification`)
- `action_ledger.jsonl` reflect entries have `is_stall` key instead of `classification`
- No `KeyError` or `AttributeError` in `run.log` related to `classification` or `proposed_next_family`

For a regression check against the baseline:
```bash
uv run python scripts/analysis/workbook_metrics.py \
    --baseline-dir results/signor_direct_eval_20260427_221617 \
    --workbook-dir results/<new_run_timestamp>
```

Target: accuracy ≥ 0.70 (recovering at least half the regression from 0.75 → 0.65).

---

## Appendix: File Change Summary

| File | Change type | Key changes |
|------|-------------|-------------|
| `reflection.py` | Modify | Remove enums; add `root_cause`, `key_insight`, `next_suggestion`, `is_stall` to `ReflectionRecord`; update `run_reflection` signature |
| `prompts.py` | Modify | Replace `REFLECTION_SYSTEM_PROMPT`, `REFLECTION_USER_PROMPT`; add `CURATOR_SYSTEM_PROMPT`, `CURATOR_USER_PROMPT` |
| `workbook.py` | Modify | Add `_section_curator_notes`; update `build_workbook_parts` to load/render Section 2b; update `_section_next_turn_guidance` for new fields |
| `evidence_programming_direct.py` | Modify | Add curator phase after reflection; update ledger arguments; add `_build_paper_summaries` helper |
| `curator.py` | **New** | `CuratorNote` model; `append_curator_note`; `load_curator_notes`; `run_curator` with forced tool call |
