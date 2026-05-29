"""
Reflection — structured meta-judgment for the bounded ReAct loop.

Phase 3 scope (see doc/IMPROVEMENT_PLAN.md):

* Persist the latest planner-produced reflection at
  ``workspace/reflection.json`` (single object, overwritten).
* Detect "stall signals" from ``action_ledger.jsonl`` so the orchestrator
  can prepend a ``<reflection-required>`` notice to the next planner turn.
* Render the latest reflection as a lead-in line in workbook Section 5.

A reflection is produced by the outer planner via the
``record_reflection`` evidence-API tool.  A reflection turn does not
search or extract — it only records the structured diagnosis.  This
keeps reflection symmetric with other actions in the ledger while
giving downstream guardrails (Phase 4) a first-class artifact to
inspect for "override condition met" semantics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from proclaim.verification.action_ledger import ActionLedger


REFLECTION_FILENAME = "reflection.json"

# Diagnosis-length cap so the workbook lead-in stays a single short line.
_DIAGNOSIS_MAX_CHARS = 300


class ReflectionClassification(str, Enum):
    RETRIEVAL = "retrieval"
    EXTRACTION = "extraction"
    FRAMING = "framing"
    BUDGET = "budget"
    OTHER = "other"


class ReflectionNextFamily(str, Enum):
    SEARCH = "search"
    EXTRACT = "extract"
    RE_EXTRACT = "re_extract"
    FEATURE_POPULATE = "feature_populate"
    FILTER = "filter"
    CHECK_SUFFICIENCY = "check_sufficiency"
    EMIT_VERDICT = "emit_verdict"
    CURATE = "curate"
    NONE = "none"


class ReflectionRecord(BaseModel):
    """Minimal 4-field reflection schema (plus turn / timestamp bookkeeping).

    Field intent:

    * ``diagnosis``            — short free-text *why* progress stalled.
    * ``classification``       — coarse cause category.
    * ``proposed_next_family`` — what action family should run next.
    * ``override_invoked``     — guardrail id (e.g. ``"GR1"``) whose
                                 override condition this reflection
                                 claims to satisfy, or ``None``.
    """

    turn: int
    timestamp: str
    diagnosis: str
    classification: ReflectionClassification
    proposed_next_family: ReflectionNextFamily
    override_invoked: Optional[str] = None

    @field_validator("diagnosis")
    @classmethod
    def _clip_diagnosis(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) > _DIAGNOSIS_MAX_CHARS:
            v = v[: _DIAGNOSIS_MAX_CHARS - 1] + "…"
        return v


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_reflection(
    workspace: Path,
    *,
    turn: int,
    diagnosis: str,
    classification: str,
    proposed_next_family: str,
    override_invoked: Optional[str] = None,
) -> ReflectionRecord:
    """Persist a reflection record (overwrites any previous file).

    Raises ``ValueError`` for unknown enum values so a malformed planner
    call surfaces clearly rather than silently storing junk.
    """
    record = ReflectionRecord(
        turn=turn,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        diagnosis=diagnosis,
        classification=ReflectionClassification(classification),
        proposed_next_family=ReflectionNextFamily(proposed_next_family),
        override_invoked=override_invoked or None,
    )
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / REFLECTION_FILENAME).write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )
    return record


def load_reflection(workspace: Path) -> Optional[ReflectionRecord]:
    path = Path(workspace) / REFLECTION_FILENAME
    if not path.exists():
        return None
    try:
        return ReflectionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_reflection(workspace: Path) -> None:
    """Delete ``reflection.json`` if it exists (used after the next planner action
    consumes the override, so a fresh stall must trigger a fresh reflection)."""
    path = Path(workspace) / REFLECTION_FILENAME
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Stall-signal detection
# ---------------------------------------------------------------------------


class StallSignals(BaseModel):
    """Bundle of stall-trigger flags derived from the action ledger and
    sufficiency history.  ``any_fired`` is True iff at least one signal
    fired; ``reasons`` is a short list of human-readable strings used as
    the ``<reflection-required signals="...">`` attribute.
    """

    search_drought: bool = False
    extraction_drought: bool = False
    sufficiency_stall: bool = False
    action_repetition: bool = False
    reasons: list[str] = Field(default_factory=list)

    @property
    def any_fired(self) -> bool:
        return (
            self.search_drought
            or self.extraction_drought
            or self.sufficiency_stall
            or self.action_repetition
        )


# Thresholds — kept module-level so tests and future Phase 4 tuning can
# import and override them cleanly.
SEARCH_DROUGHT_WINDOW = 2          # last N search rows
EXTRACT_DROUGHT_WINDOW = 2         # last M extract rows
ACTION_REPETITION_K = 3            # K consecutive same-family rows


def _target_head(target: str) -> str:
    """Return the function-name head (before any "(") for grouping."""
    if not target:
        return ""
    return target.split("(", 1)[0].strip()


def _is_search_target(target: str) -> bool:
    head = _target_head(target).lower()
    return head.startswith("search_") or head == "find_related_articles"


def _is_extract_target(target: str) -> bool:
    head = _target_head(target).lower()
    return head.startswith("extract_") or head == "add_facts_from_dicts"


def detect_stall_signals(workspace: Path) -> StallSignals:
    """Inspect ``action_ledger.jsonl`` + ``evidence_state.json`` for the
    four stall conditions defined in the Phase 3 plan.

    Reads from disk so callers don't need to thread state through.
    Returns an empty ``StallSignals`` when there is no ledger yet.
    """
    workspace = Path(workspace)
    ledger = ActionLedger(workspace)
    records = ledger.read_all()
    signals = StallSignals()

    if not records:
        return signals

    # --- Search drought -------------------------------------------------
    search_rows = [r for r in records if _is_search_target(r.target)]
    if len(search_rows) >= SEARCH_DROUGHT_WINDOW:
        tail = search_rows[-SEARCH_DROUGHT_WINDOW:]
        if all(r.delta.get("papers", 0) <= 0 for r in tail):
            signals.search_drought = True
            signals.reasons.append(
                f"search drought: last {SEARCH_DROUGHT_WINDOW} searches added 0 papers"
            )

    # --- Extraction drought --------------------------------------------
    extract_rows = [r for r in records if _is_extract_target(r.target)]
    if len(extract_rows) >= EXTRACT_DROUGHT_WINDOW:
        tail = extract_rows[-EXTRACT_DROUGHT_WINDOW:]
        if all(r.delta.get("facts", 0) <= 0 for r in tail):
            signals.extraction_drought = True
            signals.reasons.append(
                f"extraction drought: last {EXTRACT_DROUGHT_WINDOW} extractions added 0 facts"
            )

    # --- Sufficiency stall (read from evidence_state.json) -------------
    state_path = workspace / "evidence_state.json"
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            hist = data.get("sufficiency_history") or []
        except Exception:
            hist = []
        if len(hist) >= 2:
            prev_conf = float(hist[-2].get("confidence", 0.0))
            last_conf = float(hist[-1].get("confidence", 0.0))
            if last_conf <= prev_conf:
                signals.sufficiency_stall = True
                signals.reasons.append(
                    f"sufficiency stall: confidence {prev_conf:.2f} → {last_conf:.2f}"
                )

    # --- Action-family repetition --------------------------------------
    heads = [_target_head(r.target).lower() for r in records]
    if len(heads) >= ACTION_REPETITION_K:
        tail_heads = heads[-ACTION_REPETITION_K:]
        if tail_heads[0] and all(h == tail_heads[0] for h in tail_heads):
            signals.action_repetition = True
            signals.reasons.append(
                f"action repetition: '{tail_heads[0]}' called {ACTION_REPETITION_K} times in a row"
            )

    return signals


# ---------------------------------------------------------------------------
# Reflect LLM call (Phase 3)
# ---------------------------------------------------------------------------
#
# Reflection is produced by a dedicated LLM call (not by the outer planner,
# and not by the qwen subagent).  It runs only when the orchestrator's
# stall-signal detector fires, so cost is bounded.  The reflect LLM uses
# the *same* model as the planner — high-stakes diagnosis deserves the
# same reasoning capacity as the action loop.
#
# Inputs: only the workbook's volatile tail (Header, State Snapshot,
# Action Loop Record, Recuration Queue, Verdict Readiness, Raw Artifact
# Index) plus the stall signals.  The stable prefix (Claim Frame +
# Guardrails) lives in the reflect system prompt so the same boilerplate
# is shared across calls.
#
# Output is schema-bound via forced tool use: we expose a single
# ``submit_reflection`` tool and set ``tool_choice`` so the model has to
# call it.  The tool arguments come back as a JSON string under
# ``response.choices[0].message.tool_calls[0].function.arguments``.
# Anthropic + OpenAI both honour this, and the JSON-schema ``enum``
# constraints inside the tool definition give us schema validation at the
# model level — no markdown-fence munging needed.


_REFLECTION_TOOL_NAME = "submit_reflection"


def _build_reflection_tools() -> list[dict]:
    """JSON-schema tool spec used to force a structured reflection output.

    Enum values mirror ``ReflectionClassification`` and
    ``ReflectionNextFamily`` so the model can't return an unknown label.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": _REFLECTION_TOOL_NAME,
                "description": (
                    "Submit a structured reflection diagnosis for the verification "
                    "loop's current stall.  All four fields are required; pass "
                    "override_invoked=null when no guardrail override applies."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "diagnosis": {
                            "type": "string",
                            "description": (
                                "One short sentence on why progress stalled. "
                                "Focus on cause, not state."
                            ),
                        },
                        "classification": {
                            "type": "string",
                            "enum": [c.value for c in ReflectionClassification],
                        },
                        "proposed_next_family": {
                            "type": "string",
                            "enum": [f.value for f in ReflectionNextFamily],
                        },
                        "override_invoked": {
                            "type": ["string", "null"],
                            "description": (
                                "Guardrail id (e.g. 'GR1') whose override "
                                "condition this reflection authorises, or null."
                            ),
                        },
                    },
                    "required": [
                        "diagnosis",
                        "classification",
                        "proposed_next_family",
                        "override_invoked",
                    ],
                },
            },
        }
    ]


def run_reflection(
    *,
    workspace: Path,
    workbook_volatile: str,
    stall_signals: StallSignals,
    turn: int,
    model: str,
) -> tuple[Optional[ReflectionRecord], Optional[dict]]:
    """Run a single reflect LLM call and persist the result.

    Returns ``(record, usage)`` where ``record`` is the persisted
    :class:`ReflectionRecord` (or ``None`` if the call produced no usable
    reflection) and ``usage`` is a plain dict of litellm token-usage fields
    for the orchestrator's ``CostTracker`` (or ``None`` if the API call
    failed before producing a response).

    The orchestrator continues without a reflection lead-in when ``record``
    is ``None``; the stall still surfaces through the workbook's Action Loop
    Record.  ``usage`` is reported separately so reflect tokens are billed
    even when the tool-call payload is missing or malformed.
    """
    import litellm
    from proclaim.verification.prompts import (
        REFLECTION_SYSTEM_PROMPT,
        REFLECTION_USER_PROMPT,
    )

    signals_str = "; ".join(stall_signals.reasons) if stall_signals.reasons else "(none)"
    user_text = REFLECTION_USER_PROMPT.format(
        stall_signals=signals_str,
        workbook_volatile=workbook_volatile,
    )
    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    tools = _build_reflection_tools()

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=1024,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": _REFLECTION_TOOL_NAME},
            },
        )
    except Exception:
        # API call failed before producing a response — nothing to bill.
        return None, None

    # Capture token usage from the completed call, mirroring the planner's
    # field reads (see evidence_programming_direct.py) so the keys line up
    # with CostTracker.record(...).
    usage: Optional[dict] = None
    u = getattr(response, "usage", None)
    if u:
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }

    # Tokens are already spent at this point; parse/persist failures still
    # return ``usage`` so the orchestrator can bill the call.
    try:
        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return None, usage
        parsed = json.loads(tool_calls[0].function.arguments)
        record = write_reflection(
            workspace,
            turn=turn,
            diagnosis=str(parsed.get("diagnosis", "")).strip()
            or "(reflect LLM returned empty diagnosis)",
            classification=str(parsed.get("classification", "other")).lower(),
            proposed_next_family=str(parsed.get("proposed_next_family", "none")).lower(),
            override_invoked=parsed.get("override_invoked") or None,
        )
    except Exception:
        return None, usage

    return record, usage


__all__ = [
    "REFLECTION_FILENAME",
    "ReflectionClassification",
    "ReflectionNextFamily",
    "ReflectionRecord",
    "StallSignals",
    "SEARCH_DROUGHT_WINDOW",
    "EXTRACT_DROUGHT_WINDOW",
    "ACTION_REPETITION_K",
    "write_reflection",
    "load_reflection",
    "clear_reflection",
    "detect_stall_signals",
    "run_reflection",
]
