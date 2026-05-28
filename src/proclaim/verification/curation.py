"""
Recuration queue — explicit to-do surface for evidence re-curation.

Phase 3 scope (see doc/IMPROVEMENT_PLAN.md):

* Persist a structured queue at ``workspace/curation_queue.json`` with the
  six buckets that map 1:1 to workbook Section 6.
* Provide auto-enqueue helpers that the orchestrator and evidence_api
  call when state changes (sufficiency gaps, conflicts, extraction
  context updates, zero-fact papers).
* Provide an explicit ``enqueue_curation(...)`` API the planner can call
  from bash when reflection decides additional curation is needed.

Each entry carries a ``reason`` (why it was queued) and a ``source``
("auto" | "reflection" | "planner") so the planner can distinguish
machine-derived suggestions from its own intent.

Entries are *not* removed automatically by this module — Phase 3 leaves
de-queueing implicit (the planner can call ``enqueue_curation`` with the
same bucket / id to overwrite, or call ``dequeue_curation``).  This keeps
the queue's behaviour predictable until Phase 4 wires enforcement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field

from proclaim.verification.evidence_state import EvidenceState


CURATION_QUEUE_FILENAME = "curation_queue.json"


# ---------------------------------------------------------------------------
# Bucket entry schemas
# ---------------------------------------------------------------------------


Source = Literal["auto", "reflection", "planner"]


class PaperBucketEntry(BaseModel):
    """Entry for the four paper-keyed buckets (re_rank, re_extract,
    filtered_to_revisit, zero_fact)."""

    pmid: str
    reason: str
    source: Source = "auto"


class GapBucketEntry(BaseModel):
    description: str
    subclaim: str = ""
    priority: str = "medium"  # "high" | "medium" | "low"
    gap_type: str = ""
    source: Source = "auto"


class ContradictionBucketEntry(BaseModel):
    conflict_id: str
    summary: str
    source: Source = "auto"


BUCKETS_PAPER: tuple[str, ...] = (
    "re_rank",
    "re_extract",
    "filtered_to_revisit",
    "zero_fact",
)
BUCKETS_OTHER: tuple[str, ...] = ("gaps", "contradictions")
ALL_BUCKETS: tuple[str, ...] = BUCKETS_PAPER + BUCKETS_OTHER


class CurationQueue(BaseModel):
    """Six-bucket queue — single object on disk, regenerated each turn."""

    re_rank: list[PaperBucketEntry] = Field(default_factory=list)
    re_extract: list[PaperBucketEntry] = Field(default_factory=list)
    filtered_to_revisit: list[PaperBucketEntry] = Field(default_factory=list)
    zero_fact: list[PaperBucketEntry] = Field(default_factory=list)
    gaps: list[GapBucketEntry] = Field(default_factory=list)
    contradictions: list[ContradictionBucketEntry] = Field(default_factory=list)

    # -- IO ----------------------------------------------------------------

    @classmethod
    def load(cls, workspace: Path) -> "CurationQueue":
        path = Path(workspace) / CURATION_QUEUE_FILENAME
        if not path.exists():
            return cls()
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupted file — start fresh rather than crash the run.
            return cls()

    def save(self, workspace: Path) -> Path:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / CURATION_QUEUE_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    # -- Mutation ----------------------------------------------------------

    def _bucket_paper(self, name: str) -> list[PaperBucketEntry]:
        return getattr(self, name)

    def add_paper(
        self,
        bucket: str,
        pmid: str,
        reason: str,
        source: Source = "auto",
    ) -> bool:
        """Add ``pmid`` to a paper bucket if not already present.  Returns
        True when a new entry was created."""
        if bucket not in BUCKETS_PAPER:
            raise ValueError(f"Unknown paper bucket: {bucket!r}")
        bucket_list = self._bucket_paper(bucket)
        if any(e.pmid == pmid for e in bucket_list):
            return False
        bucket_list.append(PaperBucketEntry(pmid=pmid, reason=reason, source=source))
        return True

    def add_gap(
        self,
        description: str,
        *,
        subclaim: str = "",
        priority: str = "medium",
        gap_type: str = "",
        source: Source = "auto",
    ) -> bool:
        if any(e.description == description for e in self.gaps):
            return False
        self.gaps.append(
            GapBucketEntry(
                description=description,
                subclaim=subclaim,
                priority=priority,
                gap_type=gap_type,
                source=source,
            )
        )
        return True

    def add_contradiction(
        self,
        conflict_id: str,
        summary: str,
        source: Source = "auto",
    ) -> bool:
        if any(e.conflict_id == conflict_id for e in self.contradictions):
            return False
        self.contradictions.append(
            ContradictionBucketEntry(
                conflict_id=conflict_id, summary=summary, source=source
            )
        )
        return True

    def remove(self, bucket: str, key: str) -> bool:
        """Remove an entry by its primary key.  Returns True iff removed.

        ``key`` is interpreted as ``pmid`` for paper buckets, ``description``
        for gaps, and ``conflict_id`` for contradictions.
        """
        if bucket in BUCKETS_PAPER:
            bucket_list = self._bucket_paper(bucket)
            before = len(bucket_list)
            setattr(
                self,
                bucket,
                [e for e in bucket_list if e.pmid != key],
            )
            return len(getattr(self, bucket)) != before
        if bucket == "gaps":
            before = len(self.gaps)
            self.gaps = [e for e in self.gaps if e.description != key]
            return len(self.gaps) != before
        if bucket == "contradictions":
            before = len(self.contradictions)
            self.contradictions = [
                e for e in self.contradictions if e.conflict_id != key
            ]
            return len(self.contradictions) != before
        raise ValueError(f"Unknown bucket: {bucket!r}")


# ---------------------------------------------------------------------------
# Auto-enqueue helpers
# ---------------------------------------------------------------------------


def auto_enqueue_from_sufficiency(
    workspace: Path, state: EvidenceState
) -> CurationQueue:
    """Mirror the latest sufficiency gaps into the ``gaps`` bucket."""
    queue = CurationQueue.load(workspace)
    if state.sufficiency_history:
        last = state.sufficiency_history[-1]
        for gap in last.gaps:
            gt = getattr(gap.gap_type, "value", str(gap.gap_type))
            pri = getattr(gap.priority, "value", str(gap.priority))
            queue.add_gap(
                description=gap.description,
                subclaim=gap.subclaim,
                priority=pri,
                gap_type=gt,
                source="auto",
            )
    queue.save(workspace)
    return queue


def auto_enqueue_from_conflicts(
    workspace: Path, state: EvidenceState
) -> CurationQueue:
    """Mirror unresolved conflicts into the ``contradictions`` bucket."""
    queue = CurationQueue.load(workspace)
    for conflict in state.conflicts:
        summary = (conflict.description or "")[:160]
        queue.add_contradiction(conflict.id, summary, source="auto")
    queue.save(workspace)
    return queue


def auto_enqueue_zero_fact(
    workspace: Path, state: EvidenceState
) -> CurationQueue:
    """Mark papers that were extracted but produced zero facts."""
    queue = CurationQueue.load(workspace)
    extracted = set(state.extracted_pmids or [])
    if not extracted:
        queue.save(workspace)
        return queue

    pmids_with_facts = {f.source_pmid for f in state.facts}
    for pmid in extracted:
        if pmid in pmids_with_facts:
            continue
        if pmid not in state.papers:
            continue  # may have been filtered out
        queue.add_paper(
            "zero_fact",
            pmid,
            reason="extracted but produced zero facts — extraction context may need adjustment",
            source="auto",
        )
    queue.save(workspace)
    return queue


def auto_enqueue_re_extract(
    workspace: Path,
    pmids: Iterable[str],
    *,
    reason: str,
    source: Source = "auto",
) -> CurationQueue:
    """Add the given PMIDs to the ``re_extract`` bucket.

    Used by ``add_extraction_context_note`` to mark papers whose previous
    extraction was invalidated by a new context note.
    """
    queue = CurationQueue.load(workspace)
    for pmid in pmids:
        queue.add_paper("re_extract", pmid, reason=reason, source=source)
    queue.save(workspace)
    return queue


def auto_enqueue_filtered(
    workspace: Path,
    pmids: Iterable[str],
    *,
    reason: str,
    source: Source = "auto",
) -> CurationQueue:
    """Add the given PMIDs to the ``filtered_to_revisit`` bucket.

    Used by ``filter_papers_by_stance`` to record papers that were removed
    from ``state.papers`` because they had only default-stance facts.  The
    planner can later revisit them if the claim frame changes.
    """
    queue = CurationQueue.load(workspace)
    for pmid in pmids:
        queue.add_paper("filtered_to_revisit", pmid, reason=reason, source=source)
    queue.save(workspace)
    return queue


def run_auto_curation(workspace: Path) -> Optional[CurationQueue]:
    """Run all auto-enqueue passes from disk state.

    Convenience wrapper for the orchestrator: loads ``evidence_state.json``
    once, then refreshes the gaps / contradictions / zero-fact buckets.
    Returns the updated queue, or ``None`` when no state file exists.
    """
    state_path = Path(workspace) / "evidence_state.json"
    if not state_path.exists():
        return None
    try:
        state = EvidenceState.load(state_path)
    except Exception:
        return None
    auto_enqueue_from_sufficiency(workspace, state)
    auto_enqueue_from_conflicts(workspace, state)
    auto_enqueue_zero_fact(workspace, state)
    return CurationQueue.load(workspace)


# ---------------------------------------------------------------------------
# Workbook rendering helper
# ---------------------------------------------------------------------------


_BUCKET_LABELS: dict[str, str] = {
    "re_rank": "Re-rank",
    "re_extract": "Re-extract",
    "filtered_to_revisit": "Filtered to revisit",
    "zero_fact": "Zero-fact papers needing claim-frame adjustment",
    "gaps": "Gaps / new subclaims needing targeted search",
    "contradictions": "Contradictions needing resolution",
}


def _clip(text: str, limit: int = 140) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) > limit:
        s = s[: max(limit - 1, 1)] + "…"
    return s


def render_curation_section(queue: Optional[CurationQueue]) -> str:
    """Build the markdown for workbook Section 6 from a CurationQueue."""
    lines = ["## 6. Recuration Queue"]
    if queue is None:
        queue = CurationQueue()

    for bucket in ALL_BUCKETS:
        label = _BUCKET_LABELS[bucket]
        entries = getattr(queue, bucket)
        if not entries:
            lines.append(f"- {label}: (none)")
            continue
        lines.append(f"- {label}:")
        for e in entries[:6]:  # cap each bucket for workbook bound
            if bucket in BUCKETS_PAPER:
                lines.append(
                    f"    - PMID {e.pmid} — {_clip(e.reason)} [source: {e.source}]"
                )
            elif bucket == "gaps":
                pri = f"[{e.priority}] " if e.priority else ""
                gt = f"{e.gap_type}: " if e.gap_type else ""
                lines.append(
                    f"    - {pri}{gt}{_clip(e.description)} [source: {e.source}]"
                )
            elif bucket == "contradictions":
                lines.append(
                    f"    - {e.conflict_id} — {_clip(e.summary)} [source: {e.source}]"
                )
        if len(entries) > 6:
            lines.append(f"    - _(+{len(entries) - 6} more)_")

    return "\n".join(lines)


__all__ = [
    "CURATION_QUEUE_FILENAME",
    "ALL_BUCKETS",
    "BUCKETS_PAPER",
    "BUCKETS_OTHER",
    "PaperBucketEntry",
    "GapBucketEntry",
    "ContradictionBucketEntry",
    "CurationQueue",
    "auto_enqueue_from_sufficiency",
    "auto_enqueue_from_conflicts",
    "auto_enqueue_zero_fact",
    "auto_enqueue_re_extract",
    "auto_enqueue_filtered",
    "run_auto_curation",
    "render_curation_section",
]
