"""
Action ledger — append-only record of tool calls and their effects.

Phase 2 scope (see doc/IMPROVEMENT_PLAN.md):

* Persist a concise ``ActionRecord`` per tool call in
  ``workspace/action_ledger.jsonl``.
* Compute pre/post state deltas from ``EvidenceState`` so the workbook's
  Section 4 (Action Loop Record) can show what actually changed.
* Spill large raw outputs to ``workspace/artifacts/`` and reference them
  via short handles instead of inlining them into the workbook.

The ledger is the orchestrator's anti-repetition memory: each turn the
planner sees the last few records (action type, target, observation,
delta, diagnosis, next step, artifact handle) rendered in Section 4 of
``workbook.md``.

Design notes
------------

* Append-only.  Records are never edited; later corrections are added as
  new records with a diagnosis explaining the change.
* Snapshots are derived purely from ``EvidenceState`` — no extra storage.
* Action classification uses simple substring heuristics on the executed
  command/args.  When the heuristic cannot identify a known evidence-API
  call the record falls back to the bare tool name (``bash``,
  ``read_file``, ``web_search``).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from proclaim.verification.evidence_state import EvidenceState


LEDGER_FILENAME = "action_ledger.jsonl"
ARTIFACTS_DIRNAME = "artifacts"

# Spill threshold for raw outputs.  Anything longer than this (chars) is
# written to ``workspace/artifacts/`` and referenced by handle.
DEFAULT_SPILL_THRESHOLD = 4000

# Keep the action loop record bounded — the workbook only shows the most
# recent ``DEFAULT_WORKBOOK_WINDOW`` entries (per the improvement plan
# "last 3 to 5 meaningful action records").
DEFAULT_WORKBOOK_WINDOW = 5


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ActionRecord(BaseModel):
    """One entry in the action ledger.

    Field intent:

    * ``action_id``        – monotonically increasing index inside the run.
    * ``turn``             – outer LLM call number this action belongs to.
    * ``timestamp``        – UTC ISO timestamp.
    * ``action_type``      – classified family (search, extract, …).
    * ``target``           – query string, PMID list, function name, …
    * ``observation``      – one-line high-signal summary of what changed.
    * ``delta``            – non-zero state deltas as a small dict.
    * ``diagnosis``        – why progress did or did not happen (optional).
    * ``next_step``        – planner-facing hint about the next action.
    * ``artifact_handle``  – ``artifact://...`` pointer or None.
    * ``raw_size_chars``   – char length of the raw tool output.
    """

    action_id: int
    turn: int
    timestamp: str
    action_type: str
    target: str
    observation: str
    delta: dict[str, Any] = Field(default_factory=dict)
    diagnosis: Optional[str] = None
    next_step: Optional[str] = None
    artifact_handle: Optional[str] = None
    raw_size_chars: int = 0


class StateSnapshot(BaseModel):
    """Compact snapshot of mutable ``EvidenceState`` fields, used for diffs."""

    papers: int = 0
    papers_with_summary: int = 0
    papers_with_full_text: int = 0
    facts: int = 0
    extracted_pmids: int = 0
    extraction_context: int = 0
    conflicts: int = 0
    unique_source_pmids: int = 0
    fact_stance_counts: dict[str, int] = Field(default_factory=dict)
    sufficiency_checks: int = 0
    sufficiency_label: Optional[str] = None
    sufficiency_confidence: float = 0.0
    open_gaps: int = 0
    iteration: int = 0


# ---------------------------------------------------------------------------
# Snapshots & deltas
# ---------------------------------------------------------------------------


def snapshot_state(state: EvidenceState) -> StateSnapshot:
    """Materialize a ``StateSnapshot`` from an ``EvidenceState`` instance."""
    papers = state.papers
    facts = state.facts

    stance_counts: Counter = Counter(str(f.stance) for f in facts)

    last_suf = state.sufficiency_history[-1] if state.sufficiency_history else None

    return StateSnapshot(
        papers=len(papers),
        papers_with_summary=sum(1 for p in papers.values() if p.summary),
        papers_with_full_text=sum(1 for p in papers.values() if p.full_text),
        facts=len(facts),
        extracted_pmids=len(state.extracted_pmids or []),
        extraction_context=len(state.extraction_context or []),
        conflicts=len(state.conflicts or []),
        unique_source_pmids=len({f.source_pmid for f in facts}),
        fact_stance_counts=dict(stance_counts),
        sufficiency_checks=len(state.sufficiency_history or []),
        sufficiency_label=str(last_suf.label) if last_suf else None,
        sufficiency_confidence=float(last_suf.confidence) if last_suf else 0.0,
        open_gaps=len(last_suf.gaps) if last_suf else 0,
        iteration=state.iteration,
    )


def _snapshot_from_disk(workspace: Path) -> StateSnapshot:
    """Read evidence_state.json and produce a snapshot, with a safe default."""
    state_path = workspace / "evidence_state.json"
    if not state_path.exists():
        return StateSnapshot()
    try:
        state = EvidenceState.load(state_path)
    except Exception:
        return StateSnapshot()
    return snapshot_state(state)


def compute_delta(before: StateSnapshot, after: StateSnapshot) -> dict[str, Any]:
    """Return only the fields that changed, as a compact dict.

    Scalar fields are emitted as ``+N`` / ``-N`` integers (or floats for
    confidence).  Stance count diffs are nested under ``"facts_by_stance"``.
    Sufficiency label transitions are emitted as ``"sufficiency"`` with the
    new label + confidence.  Empty dicts are filtered out so the workbook
    never shows ``"delta: {}"`` cruft for actions that didn't move state.
    """
    delta: dict[str, Any] = {}

    scalar_fields = (
        "papers",
        "papers_with_summary",
        "papers_with_full_text",
        "facts",
        "extracted_pmids",
        "extraction_context",
        "conflicts",
        "unique_source_pmids",
        "sufficiency_checks",
        "open_gaps",
        "iteration",
    )
    for name in scalar_fields:
        b = getattr(before, name)
        a = getattr(after, name)
        if a != b:
            delta[name] = a - b

    stance_delta: dict[str, int] = {}
    keys = set(before.fact_stance_counts) | set(after.fact_stance_counts)
    for k in keys:
        diff = after.fact_stance_counts.get(k, 0) - before.fact_stance_counts.get(k, 0)
        if diff != 0:
            stance_delta[k] = diff
    if stance_delta:
        delta["facts_by_stance"] = stance_delta

    if (
        before.sufficiency_label != after.sufficiency_label
        or abs(before.sufficiency_confidence - after.sufficiency_confidence) > 1e-6
    ):
        if after.sufficiency_label is not None:
            delta["sufficiency"] = {
                "label": after.sufficiency_label,
                "confidence": round(after.sufficiency_confidence, 3),
            }

    return delta


# ---------------------------------------------------------------------------
# Action classification (heuristic)
# ---------------------------------------------------------------------------

# (function name → action_type) — recognised evidence-API and infra calls.
_ACTION_FAMILIES: tuple[tuple[str, str], ...] = (
    ("setup_workspace", "setup"),
    ("search_pubmed_llm", "search"),
    ("search_pubmed", "search"),
    ("search_semantic_scholar_dual", "search"),
    ("search_semantic_scholar_recommendations", "search"),
    ("search_semantic_scholar", "search"),
    ("search_for_gap", "search"),
    ("formulate_gap_queries", "search"),
    ("refine_search_for_failed_papers", "search-refine"),
    ("extract_and_add_facts", "extract"),
    ("add_facts_from_dicts", "extract"),
    ("populate_paper_features", "feature-populate"),
    ("filter_papers_by_stance", "curate"),
    ("check_sufficiency", "sufficiency-check"),
    ("get_sufficiency_history", "sufficiency-check"),
    ("add_extraction_context_note", "curate"),
    ("emit_verdict", "verdict"),
)


_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def classify_action(tool_name: str, arguments: dict) -> tuple[str, str]:
    """Return ``(action_type, target)`` for a tool invocation.

    ``tool_name`` is the OpenAI function-calling name (``bash``, ``read_file``,
    ``web_search``).  For ``bash`` we scan the command for known evidence-API
    function names and use the first match.
    """
    if tool_name == "web_search":
        q = str(arguments.get("query", ""))[:120]
        return "web-search", q
    if tool_name == "read_file":
        return "read-file", str(arguments.get("path", ""))[:120]
    if tool_name != "bash":
        return tool_name, ""

    command = str(arguments.get("command", ""))

    # First evidence-API call mentioned in the command wins.  Iterate the
    # registry in declaration order so more specific names (e.g.
    # ``refine_search_for_failed_papers``) match before generic ``search_…``
    # if both appear.
    for fn_name, family in _ACTION_FAMILIES:
        if re.search(rf"\b{re.escape(fn_name)}\s*\(", command):
            return family, _extract_call_target(command, fn_name)

    # Fall back to the first identifier-shaped call we can spot.
    match = _CALL_RE.search(command)
    if match:
        return "bash", match.group(1)
    return "bash", command[:60].replace("\n", " ").strip()


def _extract_call_target(command: str, fn_name: str) -> str:
    """Best-effort extraction of the first useful argument to a call."""
    pattern = re.compile(rf"\b{re.escape(fn_name)}\s*\((.*?)\)", re.DOTALL)
    m = pattern.search(command)
    if not m:
        return fn_name
    arg_blob = m.group(1).strip()
    arg_blob = arg_blob.replace("\n", " ")
    if len(arg_blob) > 120:
        arg_blob = arg_blob[:117] + "..."
    return f"{fn_name}({arg_blob})" if arg_blob else fn_name


# ---------------------------------------------------------------------------
# Observation summary & diagnosis
# ---------------------------------------------------------------------------


def summarize_observation(
    action_type: str,
    raw_output: str,
    delta: dict[str, Any],
) -> str:
    """Build a one-line, planner-friendly observation string.

    Preference order:

    1. Salient delta facts (new papers, new facts, sufficiency).
    2. First non-empty line of raw output (clipped).
    3. ``"(no output)"`` fallback.
    """
    parts: list[str] = []

    if "papers" in delta:
        parts.append(f"papers {_signed(delta['papers'])}")
    if "facts" in delta:
        parts.append(f"facts {_signed(delta['facts'])}")
    if "facts_by_stance" in delta:
        by_stance = ", ".join(
            f"{k} {_signed(v)}" for k, v in sorted(delta["facts_by_stance"].items())
        )
        parts.append(f"stance({by_stance})")
    if "conflicts" in delta:
        parts.append(f"conflicts {_signed(delta['conflicts'])}")
    if "extracted_pmids" in delta:
        parts.append(f"extracted-cache {_signed(delta['extracted_pmids'])}")
    if "sufficiency" in delta:
        suf = delta["sufficiency"]
        parts.append(f"sufficiency→{suf['label']}@{suf['confidence']:.2f}")
    if "iteration" in delta:
        parts.append(f"iter {_signed(delta['iteration'])}")

    if parts:
        return "; ".join(parts)

    if not raw_output:
        return "(no output)"
    head = raw_output.strip().splitlines()[0] if raw_output.strip() else ""
    head = head.strip()
    if len(head) > 160:
        head = head[:157] + "..."
    return head or "(no output)"


def diagnose(
    action_type: str,
    delta: dict[str, Any],
    raw_output: str,
) -> Optional[str]:
    """Classify a failure mode when an action produced no useful progress.

    Returns ``None`` when the action plausibly made progress.  The intent is
    to flag the obvious dead-ends (zero new papers, zero new facts, declining
    sufficiency, errors) so the planner can pivot instead of repeating.
    """
    out_lower = raw_output.lower() if raw_output else ""
    if "[error" in out_lower or "traceback" in out_lower:
        # Capture the first line that looks like an error to aid debugging
        for line in raw_output.splitlines():
            stripped = line.strip()
            if stripped.startswith("[ERROR") or "Error" in stripped:
                return f"error: {stripped[:160]}"
        return "error: tool returned an error"

    if action_type == "search" and not delta.get("papers"):
        return "no new papers retained"
    if action_type == "extract" and not delta.get("facts"):
        if delta.get("extracted_pmids", 0) > 0:
            return "papers extracted but zero new facts"
        return "no facts added (extraction not triggered or all already cached)"
    if action_type == "sufficiency-check":
        suf = delta.get("sufficiency")
        if isinstance(suf, dict) and suf.get("label", "").lower() != "sufficient":
            return f"still {suf.get('label')} @ {suf.get('confidence', 0):.2f}"
    if action_type == "feature-populate" and not raw_output.strip():
        return "no features populated (may already be cached)"

    return None


def suggest_next_step(
    action_type: str,
    delta: dict[str, Any],
    diagnosis: Optional[str],
) -> Optional[str]:
    """A short, mechanical hint for what the planner might consider next.

    These are heuristic nudges — the planner remains free to choose any
    action consistent with the active guardrails.  Returning ``None``
    keeps the workbook compact when no useful hint is obvious.
    """
    if diagnosis and diagnosis.startswith("error"):
        return "inspect artifact and adjust call signature"
    if action_type == "search" and not delta.get("papers"):
        return "try refine_search_for_failed_papers or reformulate the query"
    if action_type == "extract" and not delta.get("facts"):
        return "add_extraction_context_note for missing aliases, then retry"
    if action_type == "feature-populate":
        return "filter_papers_by_stance then check_sufficiency"
    if action_type == "curate":
        return "check_sufficiency"
    if action_type == "sufficiency-check":
        suf = delta.get("sufficiency")
        if isinstance(suf, dict) and suf.get("label", "").lower() == "sufficient":
            return "emit_verdict"
        return "address top gap via gap-targeted search"
    return None


def _signed(n: int) -> str:
    return f"+{n}" if n >= 0 else str(n)


# ---------------------------------------------------------------------------
# Artifact spill
# ---------------------------------------------------------------------------


def spill_to_artifact(
    workspace: Path,
    raw_output: str,
    *,
    action_id: int,
    action_type: str,
    threshold: int = DEFAULT_SPILL_THRESHOLD,
) -> Optional[str]:
    """If ``raw_output`` exceeds ``threshold`` chars, write it to disk and
    return an ``artifact://...`` handle.  Otherwise return ``None``.

    Files live under ``workspace/artifacts/`` with deterministic names so
    re-running the same workspace doesn't shuffle them.
    """
    if not raw_output or len(raw_output) <= threshold:
        return None

    artifacts_dir = workspace / ARTIFACTS_DIRNAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(action_type)
    filename = f"action_{action_id:04d}_{slug}.txt"
    (artifacts_dir / filename).write_text(raw_output, encoding="utf-8")
    return f"artifact://{ARTIFACTS_DIRNAME}/{filename}"


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "action"


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------


class ActionLedger:
    """Append-only JSONL writer/reader for ``ActionRecord``s.

    The on-disk file lives at ``workspace/action_ledger.jsonl``.  We avoid
    rewriting the entire file on append; reads tolerate occasional partial
    lines (truncated by an interrupted run) by skipping them.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.path = self.workspace / LEDGER_FILENAME

    # -- write side --------------------------------------------------------

    def next_action_id(self) -> int:
        """Return the next action id (1-based, monotonically increasing)."""
        if not self.path.exists():
            return 1
        # Cheap line-count — files stay small (one row per tool call).
        with self.path.open("rb") as f:
            return sum(1 for _ in f) + 1

    def append(self, record: ActionRecord) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -- read side ---------------------------------------------------------

    def read_all(self) -> list[ActionRecord]:
        if not self.path.exists():
            return []
        out: list[ActionRecord] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(ActionRecord.model_validate_json(raw))
            except Exception:
                # Partial/corrupted line — skip rather than fail the run.
                continue
        return out

    def recent(self, n: int = DEFAULT_WORKBOOK_WINDOW) -> list[ActionRecord]:
        records = self.read_all()
        return records[-n:] if n > 0 else records

    def artifact_handles(self) -> list[str]:
        return [r.artifact_handle for r in self.read_all() if r.artifact_handle]


# ---------------------------------------------------------------------------
# Top-level helper for the orchestrator
# ---------------------------------------------------------------------------


def record_action(
    *,
    workspace: Path,
    turn: int,
    tool_name: str,
    arguments: dict,
    raw_output: str,
    before: StateSnapshot,
    after: StateSnapshot,
    spill_threshold: int = DEFAULT_SPILL_THRESHOLD,
) -> ActionRecord:
    """Convenience entry point used by the orchestrator after each tool call.

    Builds the delta, classifies the action, summarises observation +
    diagnosis, optionally spills the raw output, persists the record, and
    returns it for any callers that want to log it inline.
    """
    ledger = ActionLedger(workspace)
    action_id = ledger.next_action_id()
    action_type, target = classify_action(tool_name, arguments)
    delta = compute_delta(before, after)
    observation = summarize_observation(action_type, raw_output, delta)
    diag = diagnose(action_type, delta, raw_output)
    next_step = suggest_next_step(action_type, delta, diag)
    handle = spill_to_artifact(
        workspace,
        raw_output,
        action_id=action_id,
        action_type=action_type,
        threshold=spill_threshold,
    )

    record = ActionRecord(
        action_id=action_id,
        turn=turn,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        action_type=action_type,
        target=target,
        observation=observation,
        delta=delta,
        diagnosis=diag,
        next_step=next_step,
        artifact_handle=handle,
        raw_size_chars=len(raw_output or ""),
    )
    ledger.append(record)
    return record


def load_recent_records(
    workspace: Path,
    n: int = DEFAULT_WORKBOOK_WINDOW,
) -> list[ActionRecord]:
    """Convenience helper for the workbook renderer."""
    return ActionLedger(workspace).recent(n)


def load_artifact_handles(workspace: Path) -> list[str]:
    return ActionLedger(workspace).artifact_handles()


__all__ = [
    "ActionRecord",
    "ActionLedger",
    "StateSnapshot",
    "LEDGER_FILENAME",
    "ARTIFACTS_DIRNAME",
    "DEFAULT_SPILL_THRESHOLD",
    "DEFAULT_WORKBOOK_WINDOW",
    "snapshot_state",
    "compute_delta",
    "classify_action",
    "summarize_observation",
    "diagnose",
    "suggest_next_step",
    "spill_to_artifact",
    "record_action",
    "load_recent_records",
    "load_artifact_handles",
]
