"""
Curator — synthesises the per-turn reflection into claim-specific guidance.

This is the third ACE-style role in the verification loop:

    Planner  ≈ Generator  (executes one evidence action per turn)
    Reflect  ≈ Reflector  (diagnoses what happened and why)
    Curator  ≈ Curator    (synthesises the diagnosis into workbook guidance)

After each turn's reflection, the curator LLM reads the reflection record,
the volatile workbook, and the abstracts of papers currently in state, then
optionally appends ONE note to ``workspace/curator_notes.jsonl``.  Those notes
are rendered as workbook Section 2b ("Strategy Notes"), giving the planner a
place where cross-turn observations (alias gaps, query-framing changes,
evidence-type clarifications) accumulate instead of being rediscovered every
turn.

The curator never mutates ``EvidenceState`` — its only side effect is the
append-only notes file.  Failures are non-fatal: the workbook renders without
curator notes when the file is missing or the LLM call fails.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator

from proclaim.verification.reflection import ReflectionRecord


CURATOR_NOTES_FILENAME = "curator_notes.jsonl"
_NOTE_MAX_CHARS = 1200
_MAX_NOTES_IN_WORKBOOK = 8  # show only the most recent N notes in the workbook


class CuratorNote(BaseModel):
    """One claim-specific strategy note appended by the curator.

    ``content`` is the markdown prose the planner reads in Section 2b.
    ``rationale`` records why the note is useful and is kept out of the
    workbook (audit only).
    """

    turn: int
    timestamp: str
    content: str
    rationale: str

    @field_validator("content")
    @classmethod
    def _clip_content(cls, v: str) -> str:
        v = (v or "").strip()
        return v[: _NOTE_MAX_CHARS - 1] + "…" if len(v) > _NOTE_MAX_CHARS else v


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def append_curator_note(workspace: Path, note: CuratorNote) -> None:
    """Append one curator note to ``workspace/curator_notes.jsonl``."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / CURATOR_NOTES_FILENAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write(note.model_dump_json() + "\n")


def load_curator_notes(
    workspace: Path, *, n: int = _MAX_NOTES_IN_WORKBOOK
) -> list[CuratorNote]:
    """Return the last ``n`` notes, or ``[]`` if the file does not exist.

    Malformed lines are skipped so a single bad row never blocks the workbook.
    """
    path = Path(workspace) / CURATOR_NOTES_FILENAME
    if not path.exists():
        return []
    notes: list[CuratorNote] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                notes.append(CuratorNote.model_validate_json(line))
            except Exception:
                continue
    except Exception:
        return []
    return notes[-n:] if n and n > 0 else notes


# ---------------------------------------------------------------------------
# Curator LLM call
# ---------------------------------------------------------------------------
#
# The curator uses the same forced-tool-call pattern as reflection: a single
# ``submit_curator_note`` tool with ``tool_choice`` set so the model must call
# it.  ``no_update=true`` lets the model decline cleanly when the reflection
# gives it nothing useful to add.

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
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this note will help (internal, not shown to planner).",
                        },
                        "no_update": {
                            "type": "boolean",
                            "description": (
                                "Set true when the reflection and current state give "
                                "you nothing useful to add. The workbook will not change."
                            ),
                        },
                    },
                    "required": ["no_update"],
                },
            },
        }
    ]


def run_curator(
    *,
    workspace: Path,
    workbook_volatile: str,
    reflection: ReflectionRecord,
    paper_summaries: str,
    turn: int,
    model: str,
) -> tuple[Optional[CuratorNote], Optional[dict]]:
    """Run the curator LLM and optionally append a note to ``curator_notes.jsonl``.

    Returns ``(note, usage)`` where ``note`` is ``None`` when the curator
    decided no update is needed (``no_update=true``) or the call produced no
    usable note, and ``usage`` is the litellm token-usage dict for billing
    (``None`` if the API call failed before producing a response).
    """
    import litellm
    from proclaim.verification.prompts import (
        CURATOR_SYSTEM_PROMPT,
        CURATOR_USER_PROMPT,
    )

    existing = load_curator_notes(workspace)
    if existing:
        existing_notes = "\n".join(
            f"[Turn {n.turn}] {n.content}" for n in existing
        )
    else:
        existing_notes = "(none yet)"

    user_text = CURATOR_USER_PROMPT.format(
        diagnosis=reflection.diagnosis or "(none)",
        root_cause=reflection.root_cause or "(none)",
        key_insight=reflection.key_insight or "(none)",
        next_suggestion=reflection.next_suggestion or "(none)",
        is_stall=reflection.is_stall,
        paper_summaries=paper_summaries or "(no papers retrieved yet)",
        workbook_volatile=workbook_volatile,
        existing_notes=existing_notes,
    )
    messages = [
        {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    tools = _build_curator_tools()

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=1024,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": _CURATOR_TOOL_NAME},
            },
        )
    except Exception:
        # API call failed before producing a response — nothing to bill.
        return None, None

    usage: Optional[dict] = None
    u = getattr(response, "usage", None)
    if u:
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }

    # Tokens are already spent; parse/persist failures still return ``usage``.
    try:
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return None, usage
        parsed = json.loads(tool_calls[0].function.arguments)
    except Exception:
        return None, usage

    if parsed.get("no_update"):
        return None, usage

    content = str(parsed.get("content", "")).strip()
    if not content:
        # no_update was false but no content given — treat as no-op.
        return None, usage

    note = CuratorNote(
        turn=turn,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        content=content,
        rationale=str(parsed.get("rationale", "")).strip(),
    )
    try:
        append_curator_note(workspace, note)
    except Exception:
        return None, usage

    return note, usage


__all__ = [
    "CURATOR_NOTES_FILENAME",
    "CuratorNote",
    "append_curator_note",
    "load_curator_notes",
    "run_curator",
]
