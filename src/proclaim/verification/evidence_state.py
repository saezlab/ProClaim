"""
EvidenceState — central data structure for the verification loop.

Pydantic-based with JSON persistence and audit logging.
In REPL mode, the state lives as a Python variable in the kernel.
In MCP mode, it is loaded/saved on each tool call.
Extension points:
  - Thread safety (add RLock on mutations)
  - Budget-aware get_context(budget_tokens)
  - tiktoken-based token_count()
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from proclaim.verification.data_models import (
    Conflict,
    Fact,
    PaperRecord,
    SufficiencyResult,
)


class EvidenceState(BaseModel):
    """
    Container for all evidence gathered during a verification loop.

    Pydantic v2 model with JSON persistence. Supports mutation via
    add_paper/add_fact methods and serialization via save/load.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Class-level guard: maximum sufficiency checks allowed
    MAX_ITERATIONS: int = 8

    claim: str
    subclaims: list[str] = Field(default_factory=list)
    papers: dict[str, PaperRecord] = Field(default_factory=dict)
    facts: list[Fact] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    coverage: dict[str, float] = Field(default_factory=dict)
    synthesis: dict[str, str] = Field(default_factory=dict)
    extracted_pmids: list[str] = Field(default_factory=list)
    extraction_context: list[str] = Field(default_factory=list)
    sufficiency_history: list[SufficiencyResult] = Field(default_factory=list)
    iteration: int = 0
    token_estimate: int = 0
    # Track paper count per iteration for minimum paper requirements
    papers_per_iteration: list[int] = Field(default_factory=list)
    # In-memory trace log — accumulated during REPL sessions,
    # written to disk only by checkpoint_save() (excluded from evidence_state.json).
    trace: list[dict] = Field(default_factory=list, exclude=True)

    # Private: workspace path for auto-persistence.  Set by init_new()
    # and load().  When set, every mutation auto-saves to disk so
    # renderers and external tools always see current state.
    _workspace: Optional[Path] = None

    # -- Auto-persistence --------------------------------------------------

    def _auto_save(self) -> None:
        """Persist state to workspace if one is configured.

        Writes evidence_state.json on every mutation, and also flushes
        trace.json whenever the in-memory trace is non-empty.  This ensures
        the trace survives even if emit_verdict() is never called.
        """
        if self._workspace is not None:
            self.save(self._workspace / "evidence_state.json")
            if self.trace:
                trace_path = self._workspace / "trace.json"
                trace_path.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")

    # -- Mutation methods --------------------------------------------------

    def add_paper(self, paper: PaperRecord) -> None:
        """Add a paper keyed by PMID. Overwrites if PMID already present."""
        self.papers[paper.pmid] = paper
        self._auto_save()

    def add_fact(self, fact: Fact) -> None:
        """Append a fact to the evidence."""
        self.facts.append(fact)
        self._auto_save()

    def add_conflict(self, conflict: Conflict) -> None:
        """Record a conflict between two facts."""
        self.conflicts.append(conflict)
        self._auto_save()

    def add_extraction_context(self, note: str) -> None:
        """Append a context note and clear the extraction cache for full re-extraction."""
        self.extraction_context.append(note)
        self.extracted_pmids.clear()
        self._auto_save()

    # -- Query methods -----------------------------------------------------

    def clone(self) -> "EvidenceState":
        """Deep copy — produces an independent copy of the entire state."""
        return self.model_copy(deep=True)

    def token_count(self) -> int:
        """
        Approximate token count (char-based: ~4 chars per token).

        Backbone uses character length / 4 as a rough proxy.
        Extension: replace with tiktoken encoding.
        """
        total_chars = 0
        for paper in self.papers.values():
            if paper.summary:
                total_chars += len(paper.summary)
            elif paper.abstract:
                total_chars += len(paper.abstract)
        for fact in self.facts:
            total_chars += len(fact.text)
        return total_chars // 4

    def get_context(self) -> str:
        """
        Concatenate all summaries and facts into a prompt-ready string.

        Backbone: returns everything — no budget filtering.
        Extension: add budget_tokens parameter, prioritize L2 > L1 > L0.
        """
        sections: list[str] = []

        # Paper summaries (L1) or abstracts (L0)
        if self.papers:
            sections.append("=== Evidence from Papers ===")
            for pmid, paper in self.papers.items():
                text = paper.summary if paper.summary else paper.abstract
                sections.append(f"[paper ID:{pmid}] {paper.title}\n{text}")

        # Extracted facts
        if self.facts:
            sections.append("\n=== Extracted Facts ===")
            for fact in self.facts:
                sections.append(
                    f"[{fact.stance}] {fact.text} (paper ID:{fact.source_pmid})"
                )

        return "\n\n".join(sections)

    # -- Persistence -------------------------------------------------------

    def append_trace(self, operation: str, details: dict) -> None:
        """Append an entry to the in-memory trace log."""
        self.trace.append({
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        })

    def save(self, path: Optional[Path] = None) -> None:
        """Write state as JSON to disk.

        If *path* is omitted, falls back to ``<workspace>/evidence_state.json``
        (requires that ``_workspace`` was set via ``init_new`` or ``load``).
        Raises ``ValueError`` when no path can be determined.

        If *path* points to a directory, ``evidence_state.json`` is appended
        automatically so callers don't need to remember the filename.
        """
        if path is None:
            if self._workspace is None:
                raise ValueError(
                    "save() called without a path and no workspace is set. "
                    "Pass a file path or initialize with init_new(workspace=...)."
                )
            path = self._workspace / "evidence_state.json"
        else:
            path = Path(path)
            if path.is_dir():
                path = path / "evidence_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def checkpoint_save(self, workspace) -> None:
        """Save both evidence_state.json and trace.json to workspace.

        Convenience method for REPL sessions where you want to persist
        both state and the accumulated trace log at a checkpoint.
        Accepts either a ``Path`` or ``str``.
        """
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        self.save(workspace / "evidence_state.json")
        if self.trace:
            trace_path = workspace / "trace.json"
            trace_path.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvidenceState":
        """Load state from a JSON file on disk."""
        path = Path(path)
        state = cls.model_validate_json(path.read_text(encoding="utf-8"))
        # Infer workspace from path (expects workspace/evidence_state.json)
        if path.name == "evidence_state.json":
            state._workspace = path.parent
        return state

    @classmethod
    def init_new(
        cls,
        claim: str,
        subclaims: Optional[list[str]] = None,
        workspace: Optional[Path] = None,
    ) -> "EvidenceState":
        """
        Create a new evidence state and optionally persist to workspace.

        Args:
            claim: The claim to be verified.
            subclaims: Sub-claims to verify individually.
                       Defaults to [claim] (no decomposition).
            workspace: If provided, saves state to workspace/evidence_state.json.
        """
        subs = subclaims if subclaims is not None else [claim]
        state = cls(claim=claim, subclaims=subs)
        if workspace is not None:
            workspace = Path(workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            state._workspace = workspace
            state.save(workspace / "evidence_state.json")
        return state

    def __repr__(self) -> str:
        return (
            f"EvidenceState(claim={self.claim!r}, "
            f"papers={len(self.papers)}, facts={len(self.facts)})"
        )


class TraceLog:
    """Append-only audit log for tool invocations."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, operation: str, details: dict) -> None:
        """Append a timestamped entry to the trace log."""
        entries = self.read()
        entries.append({
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def read(self) -> list[dict]:
        """Read all trace entries."""
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return []
