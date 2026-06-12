"""
Markdown workbook — compact planning surface for the verification agent.

Phase 1 scope (see doc/IMPROVEMENT_PLAN.md):
  - Serialize a bounded markdown workbook from EvidenceState.
  - Provide stable section writers so the schema is uniform across turns.
  - Replace the full execution log as the planner's primary context.

Phase 2 scope (this revision):
  - Render Section 5 (Action Loop Record) from action_ledger.jsonl entries
    as a compact bullet list (one bullet per tool call).
  - Render Section 8 (Raw Artifact Index) from artifact handles persisted
    by the orchestrator alongside the ledger.

The workbook is regenerated each turn at ``workspace/workbook.md``.
The execution log (``workspace/execution_log.py``) remains untouched and
continues to be produced for audit / notebook rendering.

Sections covered (per the improvement plan):
  1. Claim Frame           — claim, subclaims, consensus and stopping rule
  2. Guardrails            — read-only operational policy
  3. Header                — run identity, iteration, turn budget, status
  4. State Snapshot        — paper / fact / sufficiency aggregates
  4a. Known Evidence Digest— Phase 4: facts by stance/PMID + verdict packet path
  5. Action Loop Record    — last N entries from the action ledger
  6. Recuration Queue      — Phase 3 curation buckets
  7. Verdict Readiness     — derived from sufficiency_history + turn budget
  8. Raw Artifact Index    — handles for spilled raw outputs
  9. Next-Turn Guidance    — per-turn reflection: diagnosis + next family
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from proclaim.verification.action_ledger import (
    ARTIFACTS_DIRNAME,
    DEFAULT_WORKBOOK_WINDOW,
    ActionRecord,
    load_artifact_handles,
    load_recent_records,
)
from proclaim.verification.curation import CurationQueue, render_curation_section
from proclaim.verification.evidence_state import EvidenceState
from proclaim.verification.reflection import (
    ReflectionRecord,
    load_reflection,
)

if TYPE_CHECKING:
    from proclaim.verification.curator import CuratorNote


WORKBOOK_SCHEMA_VERSION = "workbook.v1"
WORKBOOK_FILENAME = "workbook.md"


# ---------------------------------------------------------------------------
# Default guardrails — migrated from the existing direct-mode prompt.
# Each guardrail keeps the rule, why it exists, and the explicit override
# condition that reflection may invoke.  Section 5 of the workbook is
# read-only to the planner.
# ---------------------------------------------------------------------------

DEFAULT_GUARDRAILS: list[dict[str, str]] = [
    {
        "id": "GR1",
        "rule": "Do not rerun the same search query unless reflection records a changed rationale.",
        "source": "migrated from direct-mode prompt",
        "override": "reflection records a changed rationale",
    },
    {
        "id": "GR2",
        "rule": "Do not re-extract the same paper PMID under the same claim frame and extraction-context version.",
        "source": "migrated from extract_and_add_facts cache semantics",
        "override": "extraction_context note added or claim frame changed",
    },
    {
        "id": "GR3",
        "rule": "Do not perform broad search immediately after an extraction failure unless reflection classifies the problem as retrieval.",
        "source": "migrated from refine_search_for_failed_papers workflow",
        "override": "reflection classifies the failure as a retrieval problem",
    },
    {
        "id": "GR4",
        "rule": "Do not repeat a full status narration when no material state change occurred.",
        "source": "new — context-budget guardrail",
        "override": "state delta is non-empty",
    },
    {
        "id": "GR5",
        "rule": "Do not consume the final turns on retrieval if verdict readiness is already high enough for check_sufficiency plus emit_verdict.",
        "source": "migrated from URGENT-turns prompt branch in evidence_programming_direct.py",
        "override": (
            "turns remaining > 2 and sufficiency still insufficient, AND "
            "Section 7 does not yet show forced-verdict-imminent"
        ),
    },
    {
        "id": "GR6",
        "rule": "Propose one next action only.",
        "source": "new — bounded ReAct loop",
        "override": "(never; this is a hard contract)",
    },
    {
        "id": "GR7",
        "rule": "Always call populate_paper_features(state) before check_sufficiency(state, llm).",
        "source": "migrated from Feature Computation for MLP Classifier section of system prompt",
        "override": "(never; MLP classifier requires populated features)",
    },
    {
        "id": "GR8",
        "rule": "Always call filter_papers_by_stance(state) before check_sufficiency to remove papers with only default-stance facts.",
        "source": "migrated from Feature Computation for MLP Classifier section of system prompt",
        "override": "(never; required for stable sufficiency signal)",
    },
    {
        "id": "GR9",
        "rule": "Never fabricate facts. Every fact must come from a paper already in state.papers.",
        "source": "migrated from CRITICAL: Grounded Evidence Only section of system prompt",
        "override": "(never; hard correctness invariant)",
    },
]


# ---------------------------------------------------------------------------
# Section writers — each returns a markdown string for one workbook section.
# Writers are pure: they read EvidenceState + a small context dict and never
# mutate either.
# ---------------------------------------------------------------------------


def _section_header(
    state: EvidenceState,
    *,
    workspace: Path,
    max_turns: int,
    turns_used: int,
    sufficiency_threshold: float,
    max_iterations: int,
) -> str:
    iteration = state.iteration
    # ``turns_used`` is the per-iteration turn count (reset after each
    # check_sufficiency), so ``turns_remaining`` is the budget left in THIS
    # iteration — not the whole run.
    turns_remaining = max(max_turns - turns_used, 0)
    is_last_iteration = iteration >= max_iterations - 1
    status = _derive_status(
        state, turns_remaining, sufficiency_threshold, workspace,
        is_last_iteration=is_last_iteration,
    )

    lines = [
        "## 3. Header",
        f"- Claim: {state.claim}",
        f"- Workspace: {workspace}",
        f"- Iteration: {iteration} / {max_iterations}",
        f"- Turns remaining this iteration: {turns_remaining} / {max_turns}",
        f"- Status: {status}",
        f"- Schema version: {WORKBOOK_SCHEMA_VERSION}",
    ]
    return "\n".join(lines)


def _section_claim_frame(
    state: EvidenceState,
    *,
    label_cfg,
    sufficiency_threshold: float,
    max_iterations: int,
) -> str:
    verdict_names = " | ".join(label_cfg.verdict_names())
    subclaim_lines = (
        "\n".join(f"  - {sc}" for sc in state.subclaims)
        if state.subclaims
        else "  - (none — claim not yet decomposed)"
    )

    stopping_rule = (
        f"Stop when sufficiency confidence >= {sufficiency_threshold:.2f} "
        f"or after {max_iterations} iterations; "
        f"force a verdict when the turn budget is nearly exhausted."
    )

    lines = [
        "## 1. Claim Frame",
        f"- Input claim: {state.claim}",
        f"- Verdict labels: {verdict_names}",
        "- Target question: whether the literature overall supports, refutes, or leaves the claim uncertain",
        "- Verdict label definitions:",
    ]
    for line in label_cfg.verdict_prompt_block().splitlines():
        lines.append(f"    {line}")
    lines.append("- Subclaims:")
    lines.append(subclaim_lines)
    lines.append("- Consensus notes:")
    lines.append("  - Prefer multiple independent sources over a single decisive paper.")
    lines.append("  - Resolve contradictions explicitly before emitting a verdict.")
    lines.append("- Interpretation notes:")
    lines.append("  - Treat synonyms / aliases captured in extraction context as on-claim.")
    lines.append("  - Scope, mechanism class, and compartmental constraints can flip relevance — keep them explicit.")
    lines.append(f"- Stopping rule: {stopping_rule}")
    return "\n".join(lines)


def _section_state_snapshot(state: EvidenceState) -> str:
    papers = state.papers
    facts = state.facts

    total_papers = len(papers)
    papers_with_full_text = sum(1 for p in papers.values() if p.full_text)
    extracted_pmids = set(state.extracted_pmids or [])

    # Per-paper fact counts (used both for zero-fact bucket and aggregates).
    facts_by_pmid: Counter = Counter()
    for f in facts:
        facts_by_pmid[f.source_pmid] += 1
    zero_fact_papers = [
        pmid for pmid in papers if pmid in extracted_pmids and facts_by_pmid.get(pmid, 0) == 0
    ]

    stance_counts: Counter = Counter(str(f.stance) for f in facts)
    unique_sources = len({f.source_pmid for f in facts})

    rows = [
        ("Papers retrieved", total_papers),
        ("Papers with full text", papers_with_full_text),
        ("Papers extracted (cache)", len(extracted_pmids)),
        ("Zero-fact papers (extracted, no facts)", len(zero_fact_papers)),
        ("Total facts", len(facts)),
        ("Unique source PMIDs (facts)", unique_sources),
        ("Conflicts", len(state.conflicts)),
    ]
    # Add a row per configured stance so any rebuilt enum still renders.
    for stance, count in sorted(stance_counts.items()):
        rows.append((f"Facts: {stance}", count))

    table = ["| Metric | Value |", "| --- | ---: |"]
    for label, value in rows:
        table.append(f"| {label} | {value} |")

    suff_trend = _format_sufficiency_trend(state)
    open_gaps = _format_open_gaps(state)

    lines = ["## 4. State Snapshot", *table, "", "- Sufficiency trend:"]
    lines.extend(suff_trend)
    lines.append("- Open gaps:")
    lines.extend(open_gaps)

    # Show the active (latest) extraction context note so the planner knows
    # what aliases are in scope without scanning historical notes.  Older notes
    # are superseded and omitted; the full history is in the trace log.
    if state.extraction_context:
        lines.append("- Extraction context in force (latest):")
        lines.append(f"  - {state.extraction_context[-1]}")
    else:
        lines.append("- Extraction context in force: (none)")

    return "\n".join(lines)


def _section_known_evidence_digest(
    state: EvidenceState,
    *,
    verdict_packet_path: Optional[Path] = None,
    max_facts_per_stance: int = 8,
) -> str:
    """Render Section 4a — a compact, fact-centered digest of the full corpus.

    Phase 4 (recovery plan): the planner previously saw counts and gaps but not
    a statement of *what is known*, which anchored verdicts on "missing
    evidence" rather than the extracted facts. This volatile section surfaces
    facts grouped by stance/PMID (capped tighter than the full verdict packet so
    the workbook stays bounded) and points at the written verdict packet for the
    complete view.

    Numbered ``4a`` deliberately, so the existing Section IDs (5–9) downstream
    code and the prompt reference do not have to renumber.
    """
    from proclaim.verification.verdict_packet import build_known_facts_digest

    lines = ["## 4a. Known Evidence Digest"]
    digest = build_known_facts_digest(state, max_facts_per_stance=max_facts_per_stance)
    lines.append(digest)
    if verdict_packet_path is not None:
        lines.append("")
        lines.append(
            f"- Full verdict packet (all facts, no caps): `{verdict_packet_path.name}`"
        )
    return "\n".join(lines)


def _section_action_loop_record(records: Sequence[ActionRecord]) -> str:
    """Render the last few action ledger entries as a compact bullet list.

    Each bullet is a single line of the form::

        - #<id> <target> — <observation>[; <diagnosis>][ <artifact_handle>]

    ``observation`` already carries the delta-derived summary, so we don't
    render the raw ``delta`` dict here.  We deliberately do *not* render
    ``action_type`` (the old heuristic was unreliable for mixed bash cells)
    or a planner-facing next-step nudge — those belong in Section 9
    (Next-Turn Guidance).
    """
    header = "## 5. Action Loop Record"

    if not records:
        return f"{header}\n- _(empty — no tool calls recorded yet for this run)_"

    lines = [header]
    for rec in records:
        target = _clip(rec.target, 80) or "(no target)"
        summary = _clip(rec.observation, 140) or "(no output)"
        bullet = f"- #{rec.action_id} {target} — {summary}"
        if rec.diagnosis:
            bullet += f"; {_clip(rec.diagnosis, 140)}"
        if rec.artifact_handle:
            bullet += f" [{rec.artifact_handle}]"
        lines.append(bullet)
    return "\n".join(lines)


def _clip(value: Optional[str], limit: int) -> str:
    """Collapse newlines and clip to ``limit`` chars for a single-line bullet."""
    if not value:
        return ""
    s = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(s) > limit:
        s = s[: max(limit - 1, 1)] + "…"
    return s


def _section_guardrails(guardrails: Iterable[dict[str, str]]) -> str:
    blocks = ["## 2. Guardrails"]
    for gr in guardrails:
        blocks.append(f"- {gr['id']}:")
        blocks.append(f"    - source: {gr['source']}")
        blocks.append(f"    - rule: {gr['rule']}")
        blocks.append(f"    - override: {gr['override']}")
        blocks.append("    - status: active")
    blocks.append(
        "\nThese guardrails are read-only to the agent. Reflection may "
        "note when an override condition has been met; it must not edit or "
        "rewrite the rules themselves."
    )
    return "\n".join(blocks)


def _section_curator_notes(notes: list["CuratorNote"]) -> str:
    """Render Section 2b — accumulated claim-specific strategy notes from the curator.

    These notes are the dynamic, claim-specific constraint layer: the curator
    appends cross-turn observations (alias gaps, query reframes, evidence-type
    clarifications) that the planner would otherwise rediscover every turn.
    Positioned near the top of the volatile context so the planner reads them
    before deciding its action.
    """
    header = "## 2b. Strategy Notes (curator-accumulated)"
    if not notes:
        return f"{header}\n- _(no curator notes yet)_"
    lines = [header]
    for note in notes:
        lines.append(f"\n[Turn {note.turn}]")
        lines.append(note.content.strip())
    return "\n".join(lines)


def _section_recuration_queue(queue: Optional[CurationQueue]) -> str:
    """Render Section 6 from a CurationQueue (or an empty default)."""
    return render_curation_section(queue)


def _section_verdict_readiness(
    state: EvidenceState,
    *,
    turns_remaining: int,
    sufficiency_threshold: float,
    is_last_iteration: bool = False,
) -> str:
    if state.sufficiency_history:
        last = state.sufficiency_history[-1]
        sufficiency_label = last.label
        confidence = last.confidence
        gap_summaries = [g.description for g in last.gaps]
    else:
        last = None
        sufficiency_label = "(not yet checked)"
        confidence = 0.0
        gap_summaries = []

    has_non_default_facts = _has_non_default_facts(state)
    stagnated = _sufficiency_stagnated(state)

    # Verdict readiness is true when any of the Phase 3 escapes hold. Each is a
    # deterministic signal that further retrieval is unlikely to change the
    # outcome, so the loop should pivot to emitting a verdict.
    ready = False
    ready_reason: Optional[str] = None
    if sufficiency_label.lower() == "sufficient" and confidence >= sufficiency_threshold:
        ready = True
        ready_reason = "sufficiency met threshold"
    elif last is not None and _only_low_diversity_caveat(last):
        ready = True
        ready_reason = (
            "sufficient; only blocked by a soft low-source-diversity caveat"
        )
    elif has_non_default_facts and stagnated:
        ready = True
        ready_reason = (
            "sufficiency confidence flat/declining over recent checks while "
            "directional facts already exist — stagnation escape"
        )
    elif has_non_default_facts and (is_last_iteration or turns_remaining <= 2):
        ready = True
        ready_reason = (
            "final iteration / turn budget low and directional facts exist — "
            "emit a verdict on current evidence"
        )

    missing_lines: list[str] = []
    if not ready and gap_summaries:
        for g in gap_summaries[:5]:
            missing_lines.append(f"    - {g}")
    if not missing_lines:
        missing_lines.append("    - (none recorded)")

    blockers = []
    if ready:
        blockers.append(f"(none — verdict can be emitted: {ready_reason})")
    else:
        if sufficiency_label.lower() != "sufficient":
            blockers.append(f"sufficiency label is {sufficiency_label!r}")
        if confidence < sufficiency_threshold:
            blockers.append(
                f"confidence {confidence:.2f} below threshold {sufficiency_threshold:.2f}"
            )
        if not has_non_default_facts:
            blockers.append("no directional (non-default-stance) facts extracted yet")
        if not blockers:
            blockers.append("(none — verdict can be emitted)")

    if turns_remaining <= 2 and not ready:
        if is_last_iteration:
            forced_risk = (
                "HIGH — final iteration, turn budget low; expect forced verdict "
                "if no resolution this turn."
            )
        else:
            forced_risk = (
                "MEDIUM — iteration turn budget low; expect a forced "
                "check_sufficiency to close out this iteration."
            )
    else:
        forced_risk = "low"

    lines = [
        "## 7. Verdict Readiness",
        f"- Sufficiency: {sufficiency_label} (confidence {confidence:.2f})",
        f"- Ready for verdict: {'yes' if ready else 'no'}",
        "- Blockers:",
        *(f"    - {b}" for b in blockers),
        "- Minimum additional evidence needed:",
        *missing_lines,
        f"- Forced-verdict risk: {forced_risk}",
    ]
    return "\n".join(lines)


def _section_raw_artifact_index(handles: Sequence[str], *, limit: int = 20) -> str:
    """Render artifact handles persisted in the action ledger.

    The list is capped at ``limit`` to keep the workbook bounded; older
    handles remain in ``action_ledger.jsonl`` for full audit.  The planner
    can request a specific artifact via the ``read_file`` tool by following
    the workspace-relative path embedded in each handle.
    """
    if not handles:
        return (
            "## 8. Raw Artifact Index\n"
            f"- (none — no raw outputs have been spilled to `{ARTIFACTS_DIRNAME}/` yet)"
        )

    # Most-recent first for planner relevance.
    recent = list(reversed(handles))
    if len(recent) > limit:
        recent = recent[:limit]
        suffix = (
            f"- _(showing latest {limit}; older handles remain in "
            "`action_ledger.jsonl`)_"
        )
    else:
        suffix = ""

    lines = ["## 8. Raw Artifact Index"]
    lines.extend(f"- {h}" for h in recent)
    if suffix:
        lines.append(suffix)
    return "\n".join(lines)


def _section_next_turn_guidance(reflection: Optional[ReflectionRecord]) -> str:
    """Render Section 9 — the reflection module's recommendation for this turn.

    Placed at the very end of the volatile workbook so it sits next to the
    orchestrator's "make ONE tool call" instruction in the user message —
    the salient position for an action decision.

    The reflection is regenerated every turn from the latest workbook state
    (per-turn reflect), so this section always reflects current strategic
    context, not a stale stall diagnosis.
    """
    header = "## 9. Next-Turn Guidance"
    if reflection is None:
        return (
            f"{header}\n"
            "- _(no reflection recorded yet — first turn, or reflect call failed)_"
        )

    stall_marker = " [STALL TURN]" if reflection.is_stall else ""
    lines = [
        header,
        f"- Reflection (turn {reflection.turn}{stall_marker}):",
        f"    - Diagnosis: {reflection.diagnosis.strip() or '(empty)'}",
        f"    - Root cause: {reflection.root_cause.strip() or '(empty)'}",
    ]
    if reflection.key_insight.strip():
        lines.append(f"    - Key insight: {reflection.key_insight.strip()}")
    lines.append(f"    - Suggested next action: {reflection.next_suggestion.strip() or '(empty)'}")
    if reflection.override_invoked:
        lines.append(f"    - Guardrail override invoked: {reflection.override_invoked}")
    # Phase 4: when the recommendation is to emit a verdict, anchor the planner
    # on the fact-centered Known Evidence Digest / verdict packet rather than on
    # this action/reflection history — the latter biases the verdict toward
    # "missing evidence" instead of what the facts actually say.
    _ns = reflection.next_suggestion.lower()
    if _ns.startswith("emit") or "emit_verdict" in _ns:
        lines.append(
            "    - When emitting the verdict, base `reasoning`, `key_evidence`, and "
            "`gaps_remaining` on Section 4a (Known Evidence Digest) / the verdict "
            "packet — check each fact's direction against the claim. Do not derive "
            "the verdict from action history, reflection diagnosis, or guardrails."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_status(
    state: EvidenceState,
    turns_remaining: int,
    sufficiency_threshold: float,
    workspace: Path,
    *,
    is_last_iteration: bool = False,
) -> str:
    # When this iteration's turn budget is nearly spent, the urgency depends on
    # whether it is the FINAL iteration.  Final iteration → wrap up the run with
    # a verdict.  Earlier iteration → just close out the iteration with a
    # sufficiency check (further iterations remain).
    if (workspace / "verdict.json").exists():
        return "complete (verdict emitted)"
    if state.sufficiency_history:
        last = state.sufficiency_history[-1]
        sufficient = (
            last.label.lower() == "sufficient"
            and last.confidence >= sufficiency_threshold
        )
        if sufficient:
            return f"ready-for-verdict (sufficiency=sufficient, confidence={last.confidence:.2f})"
        # Phase 3 escapes: a soft low-diversity caveat, or stagnant confidence
        # with directional facts already on hand, both mean further retrieval is
        # unlikely to help — flag verdict readiness rather than "collecting".
        if _only_low_diversity_caveat(last):
            return "ready-for-verdict (sufficient; soft low-diversity caveat only)"
        if _has_non_default_facts(state) and _sufficiency_stagnated(state):
            return (
                f"ready-for-verdict (stagnation escape: confidence flat/declining "
                f"at {last.confidence:.2f} with directional facts)"
            )
        if turns_remaining <= 2:
            kind = "forced-verdict-imminent" if is_last_iteration else "check-sufficiency-imminent"
            return (
                f"{kind} (sufficiency={last.label}, "
                f"confidence={last.confidence:.2f}, turns_remaining={turns_remaining})"
            )
        return f"collecting-evidence (sufficiency={last.label}, confidence={last.confidence:.2f})"
    if turns_remaining <= 2:
        kind = "forced-verdict-imminent" if is_last_iteration else "check-sufficiency-imminent"
        return f"{kind} (no sufficiency check yet, turns_remaining={turns_remaining})"
    return "collecting-evidence (no sufficiency check yet)"


def _sufficiency_stagnated(
    state: EvidenceState, window: int = 3, min_delta: float = 0.02
) -> bool:
    """Return True when sufficiency confidence is flat or declining.

    Deterministic stagnation signal for the verdict-readiness escape: once the
    classifier has produced at least ``window`` checks and the confidence over
    that window neither rises by ``min_delta`` nor recovers from a decline,
    further retrieval is unlikely to help and the loop should pivot to a
    verdict (provided non-default facts already exist — that gate lives in
    Section 7, not here).

    Mirrors the trend logic in ``evidence_api.get_sufficiency_history`` but
    collapses "flat" and "declining" into a single boolean.
    """
    hist = state.sufficiency_history
    if len(hist) < window:
        return False
    recent = [r.confidence for r in hist[-window:]]
    score_range = max(recent) - min(recent)
    declining = recent[-1] < recent[0]
    return score_range < min_delta or declining


def _has_non_default_facts(state: EvidenceState) -> bool:
    """Return True when at least one fact carries a non-default stance.

    Default-stance (NEUTRAL) facts are relevant context but cannot, on their
    own, anchor a SUPPORT/REFUTE verdict; the stagnation escape requires real
    directional evidence before recommending a verdict.
    """
    from proclaim.verification.config import get_label_config

    default_stance = get_label_config().default_stance
    return any(str(f.stance) != default_stance for f in state.facts)


def _only_low_diversity_caveat(result) -> bool:
    """Return True when a SUFFICIENT result's only open gap is a soft caveat.

    A SUFFICIENT check whose remaining gaps are all LOW_DIVERSITY is blocked
    only by the soft source-diversity warning (Phase 3), not by a real
    evidence gap — so it should still count as verdict-ready.
    """
    from proclaim.verification.data_models import GapType

    if result.label.lower() != "sufficient" or not result.gaps:
        return False
    return all(g.gap_type == GapType.LOW_DIVERSITY for g in result.gaps)


def _format_sufficiency_trend(state: EvidenceState) -> list[str]:
    hist = state.sufficiency_history
    if not hist:
        return ["    - (no sufficiency checks recorded)"]
    lines = []
    for i, res in enumerate(hist):
        lines.append(
            f"    - check {i + 1}: {res.label} (confidence {res.confidence:.2f}, "
            f"{len(res.gaps)} gaps)"
        )
    return lines


def _format_open_gaps(state: EvidenceState) -> list[str]:
    if not state.sufficiency_history:
        return ["    - (no sufficiency checks recorded)"]
    gaps = state.sufficiency_history[-1].gaps
    if not gaps:
        return ["    - (none — last sufficiency check reported no gaps)"]
    lines = []
    for g in gaps:
        gt = getattr(g.gap_type, "value", str(g.gap_type))
        pri = getattr(g.priority, "value", str(g.priority))
        lines.append(f"    - [{pri}] {gt}: {g.description}")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Render order: STABLE sections first (Claim Frame + Guardrails, both invariant
# within a single run) followed by VOLATILE sections (everything that changes
# turn-over-turn).  The two groups are separated by a sentinel marker so the
# orchestrator can drop a prompt-cache breakpoint exactly at the boundary.
# Section IDs (1–8) follow the schema in doc/IMPROVEMENT_PLAN.md and are
# preserved in headings, only the *render* order is rearranged.
STABLE_VOLATILE_MARKER = "<!-- ---- workbook: stable | volatile ---- -->"


def build_workbook_parts(
    state: EvidenceState,
    *,
    workspace: Path,
    max_turns: int,
    turns_used: int,
    sufficiency_threshold: float,
    max_iterations: int,
    label_cfg,
    guardrails: Optional[Iterable[dict[str, str]]] = None,
    action_records: Optional[Sequence[ActionRecord]] = None,
    artifact_handles: Optional[Sequence[str]] = None,
    action_window: int = DEFAULT_WORKBOOK_WINDOW,
    reflection: Optional[ReflectionRecord] = None,
    curation_queue: Optional[CurationQueue] = None,
    curator_notes: Optional[list["CuratorNote"]] = None,
    verdict_packet_path: Optional[Path] = None,
) -> tuple[str, str]:
    """Build the workbook as a (stable_prefix, volatile_tail) pair.

    The stable prefix is invariant for the duration of a run and is the
    cache-friendly portion; the volatile tail is regenerated each turn.
    Both strings already include trailing newlines.

    ``action_records``, ``artifact_handles``, ``reflection``,
    ``curation_queue`` and ``curator_notes`` default to reading from disk
    under ``workspace``.  Pass explicit values to short-circuit disk I/O in
    tests.
    """
    # ``turns_used`` / ``turns_remaining`` are per-iteration (reset after each
    # check_sufficiency).  ``is_last_iteration`` decides whether a nearly-spent
    # turn budget means "force a verdict" (last iteration) or merely "force a
    # sufficiency check to close out this iteration" (earlier iterations).
    turns_remaining = max(max_turns - turns_used, 0)
    is_last_iteration = state.iteration >= max_iterations - 1
    grs = list(guardrails if guardrails is not None else DEFAULT_GUARDRAILS)

    if action_records is None:
        action_records = load_recent_records(workspace, n=action_window)
    if artifact_handles is None:
        artifact_handles = load_artifact_handles(workspace)
    if reflection is None:
        reflection = load_reflection(workspace)
    if curation_queue is None:
        curation_queue = CurationQueue.load(workspace)
    if curator_notes is None:
        from proclaim.verification.curator import load_curator_notes

        try:
            curator_notes = load_curator_notes(workspace)
        except Exception:
            curator_notes = []

    stable_sections = [
        "# ProClaim Workbook",
        _section_claim_frame(
            state,
            label_cfg=label_cfg,
            sufficiency_threshold=sufficiency_threshold,
            max_iterations=max_iterations,
        ),
        _section_guardrails(grs),
    ]

    volatile_sections = [
        _section_curator_notes(curator_notes),
        _section_header(
            state,
            workspace=workspace,
            max_turns=max_turns,
            turns_used=turns_used,
            sufficiency_threshold=sufficiency_threshold,
            max_iterations=max_iterations,
        ),
        _section_state_snapshot(state),
        _section_known_evidence_digest(
            state, verdict_packet_path=verdict_packet_path
        ),
        _section_action_loop_record(action_records),
        _section_recuration_queue(curation_queue),
        _section_verdict_readiness(
            state,
            turns_remaining=turns_remaining,
            sufficiency_threshold=sufficiency_threshold,
            is_last_iteration=is_last_iteration,
        ),
        _section_raw_artifact_index(artifact_handles),
        _section_next_turn_guidance(reflection),
    ]

    stable_prefix = "\n\n".join(stable_sections) + "\n\n"
    volatile_tail = "\n\n".join(volatile_sections) + "\n"
    return stable_prefix, volatile_tail


def build_workbook(
    state: EvidenceState,
    *,
    workspace: Path,
    max_turns: int,
    turns_used: int,
    sufficiency_threshold: float,
    max_iterations: int,
    label_cfg,
    guardrails: Optional[Iterable[dict[str, str]]] = None,
    action_records: Optional[Sequence[ActionRecord]] = None,
    artifact_handles: Optional[Sequence[str]] = None,
    action_window: int = DEFAULT_WORKBOOK_WINDOW,
    reflection: Optional[ReflectionRecord] = None,
    curation_queue: Optional[CurationQueue] = None,
    curator_notes: Optional[list["CuratorNote"]] = None,
    verdict_packet_path: Optional[Path] = None,
) -> str:
    """Render the full markdown workbook (stable prefix + marker + volatile tail).

    Section IDs (1–8) follow the schema in ``doc/IMPROVEMENT_PLAN.md``.  The
    *render* order is stable-first so the orchestrator can place a prompt-cache
    breakpoint at the boundary marker.
    """
    stable, volatile = build_workbook_parts(
        state,
        workspace=workspace,
        max_turns=max_turns,
        turns_used=turns_used,
        sufficiency_threshold=sufficiency_threshold,
        max_iterations=max_iterations,
        label_cfg=label_cfg,
        guardrails=guardrails,
        action_records=action_records,
        artifact_handles=artifact_handles,
        action_window=action_window,
        reflection=reflection,
        curation_queue=curation_queue,
        curator_notes=curator_notes,
        verdict_packet_path=verdict_packet_path,
    )
    return f"{stable}{STABLE_VOLATILE_MARKER}\n\n{volatile}"


def write_workbook(
    state: EvidenceState,
    *,
    workspace: Path,
    max_turns: int,
    turns_used: int,
    sufficiency_threshold: float,
    max_iterations: int,
    label_cfg,
    guardrails: Optional[Iterable[dict[str, str]]] = None,
    action_records: Optional[Sequence[ActionRecord]] = None,
    artifact_handles: Optional[Sequence[str]] = None,
    action_window: int = DEFAULT_WORKBOOK_WINDOW,
    reflection: Optional[ReflectionRecord] = None,
    curation_queue: Optional[CurationQueue] = None,
    curator_notes: Optional[list["CuratorNote"]] = None,
) -> Path:
    """Serialize the workbook to ``workspace/workbook.md`` and return the path.

    Also writes the clean verdict packet (``verdict_packet.md``) alongside the
    workbook so the forced-verdict path and the planner share one fact-centered
    view. Packet failures are non-fatal — workbook rendering must not depend on
    it.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    verdict_packet_path: Optional[Path] = None
    try:
        from proclaim.verification.verdict_packet import write_verdict_packet

        verdict_packet_path = write_verdict_packet(
            state, workspace=workspace, label_cfg=label_cfg
        )
    except Exception:  # pragma: no cover - packet is an aid, not a dependency
        verdict_packet_path = None

    text = build_workbook(
        state,
        workspace=workspace,
        max_turns=max_turns,
        turns_used=turns_used,
        sufficiency_threshold=sufficiency_threshold,
        max_iterations=max_iterations,
        label_cfg=label_cfg,
        guardrails=guardrails,
        action_records=action_records,
        artifact_handles=artifact_handles,
        action_window=action_window,
        reflection=reflection,
        curation_queue=curation_queue,
        curator_notes=curator_notes,
        verdict_packet_path=verdict_packet_path,
    )
    path = workspace / WORKBOOK_FILENAME
    path.write_text(text)
    return path
