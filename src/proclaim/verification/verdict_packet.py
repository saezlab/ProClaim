"""
Verdict packet — a clean, fact-centered evidence view for verdict emission.

Recovery plan Phase 4 (see doc/workbook_reflection_recovery_implementation_plan.md):
the planner and the forced-verdict path previously saw counts, gaps, action-loop
history, and reflection guidance, but not a compact statement of what is actually
*known*. That anchored the final LLM on "missing evidence" instead of the
extracted facts, and the old direction-correction behavior was lost.

This module renders, deterministically and with **no extra LLM call**, two
artifacts derived purely from ``EvidenceState``:

  * ``build_known_facts_digest`` — facts grouped by stance then PMID, with paper
    titles and short fact snippets, plus a capped list of retrieved papers that
    yielded no directional facts.
  * ``build_verdict_packet`` — the digest wrapped with the claim, verdict label
    definitions, full-corpus counts, and the sufficiency result presented as
    *classifier metadata* rather than as evidence.

Both operate on the **full corpus** (``state.papers`` / ``state.facts``), not the
sufficiency candidate subset, so a post-filter state still surfaces every
retained paper and fact. By construction the packet excludes workbook control
text — Action Loop Record, reflection diagnosis, guardrails, recommended-next-
action, and re-extract queues all live outside ``EvidenceState`` and are never
read here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from proclaim.verification.evidence_state import EvidenceState


VERDICT_PACKET_FILENAME = "verdict_packet.md"

# Keep individual fact snippets bounded so the packet stays compact even when a
# single paper yields a long extracted statement.
_FACT_SNIPPET_CHARS = 240
_TITLE_CHARS = 160
# Cap the "no directional facts" list — these papers are context, not evidence.
_MAX_ZERO_FACT_PAPERS = 10


def _clip(value: Optional[str], limit: int) -> str:
    """Collapse whitespace and clip ``value`` to ``limit`` characters."""
    if not value:
        return ""
    s = " ".join(str(value).split())
    if len(s) > limit:
        s = s[: max(limit - 1, 1)] + "…"
    return s


def _default_stance() -> str:
    from proclaim.verification.config import get_label_config

    return get_label_config().default_stance


def _stance_order(state: EvidenceState) -> list[str]:
    """Stance label names in configured order, with any unexpected stances
    appended so no fact is silently dropped from the digest."""
    from proclaim.verification.config import get_label_config

    names = list(get_label_config().stance_names())
    seen = set(names)
    for f in state.facts:
        s = str(f.stance)
        if s not in seen:
            names.append(s)
            seen.add(s)
    return names


def build_known_facts_digest(
    state: EvidenceState, *, max_facts_per_stance: int = 12
) -> str:
    """Render a compact, fact-centered digest of the full evidence corpus.

    Includes total paper/fact counts, stance counts, facts grouped by stance
    then PMID (with paper titles and short snippets), and a capped list of
    retrieved papers that produced no directional (non-default-stance) facts.

    ``max_facts_per_stance`` bounds the number of facts shown per stance group;
    any overflow is summarized as a count so the digest stays bounded without
    hiding that more evidence exists.
    """
    default_stance = _default_stance()
    facts = state.facts
    papers = state.papers

    stance_counts: Counter = Counter(str(f.stance) for f in facts)

    lines: list[str] = []
    lines.append(
        f"Corpus: {len(papers)} retrieved paper(s), {len(facts)} extracted fact(s)."
    )
    if stance_counts:
        counts_str = ", ".join(
            f"{name}: {stance_counts.get(name, 0)}" for name in _stance_order(state)
        )
        lines.append(f"Fact stance counts: {counts_str}.")
    if state.conflicts:
        lines.append(f"Recorded conflicts between facts: {len(state.conflicts)}.")

    # Facts grouped by stance, then by source PMID.
    facts_by_stance: dict[str, list] = defaultdict(list)
    for f in facts:
        facts_by_stance[str(f.stance)].append(f)

    pmids_with_facts: set[str] = {f.source_pmid for f in facts}

    for stance in _stance_order(state):
        group = facts_by_stance.get(stance, [])
        if not group:
            continue
        lines.append("")
        lines.append(f"### {stance} facts ({len(group)})")

        by_pmid: dict[str, list] = defaultdict(list)
        for f in group:
            by_pmid[f.source_pmid].append(f)

        shown = 0
        truncated = False
        for pmid, pmid_facts in by_pmid.items():
            if shown >= max_facts_per_stance:
                truncated = True
                break
            title = _clip(getattr(papers.get(pmid), "title", ""), _TITLE_CHARS) or "(title unavailable)"
            lines.append(f"- PMID {pmid} — {title}")
            for f in pmid_facts:
                if shown >= max_facts_per_stance:
                    truncated = True
                    break
                snippet = _clip(f.text, _FACT_SNIPPET_CHARS) or "(empty fact text)"
                lines.append(f"    - {snippet}")
                shown += 1
        if truncated:
            remaining = len(group) - shown
            if remaining > 0:
                lines.append(
                    f"    - … and {remaining} more {stance} fact(s) not shown."
                )

    # Retrieved papers that yielded no directional facts — context, not evidence.
    directional_pmids = {
        f.source_pmid for f in facts if str(f.stance) != default_stance
    }
    no_directional = [pmid for pmid in papers if pmid not in directional_pmids]
    if no_directional:
        lines.append("")
        lines.append(
            f"### Retrieved papers with no directional ({default_stance}-excluded) "
            f"facts ({len(no_directional)})"
        )
        for pmid in no_directional[:_MAX_ZERO_FACT_PAPERS]:
            title = _clip(getattr(papers.get(pmid), "title", ""), _TITLE_CHARS) or "(title unavailable)"
            note = "" if pmid in pmids_with_facts else " — no facts extracted"
            lines.append(f"- PMID {pmid} — {title}{note}")
        overflow = len(no_directional) - _MAX_ZERO_FACT_PAPERS
        if overflow > 0:
            lines.append(f"- … and {overflow} more retained but non-directional paper(s).")

    return "\n".join(lines)


def build_verdict_packet(
    state: EvidenceState, *, label_cfg, max_facts_per_stance: int = 20
) -> str:
    """Render the clean verdict packet for the final verdict LLM.

    Contains the claim, verdict label definitions, full-corpus counts, the
    sufficiency result as *classifier metadata*, the known-facts digest, and any
    remaining caveats — clearly separated from the evidence.

    It deliberately excludes all workbook control text (action loop, reflection
    diagnosis, guardrails, recommended-next-action, re-extract queues): those are
    not part of ``EvidenceState`` and must not anchor the verdict.
    """
    lines: list[str] = ["# Verdict Packet", ""]
    lines.append(f"## Claim\n{state.claim}")

    lines.append("")
    lines.append("## Verdict label definitions")
    lines.append(label_cfg.verdict_prompt_block())

    if state.subclaims:
        lines.append("")
        lines.append("## Subclaims")
        for sc in state.subclaims:
            lines.append(f"- {sc}")

    # Interpretation notes — kept generic and task-agnostic. The point is to
    # steer the verdict toward consensus-level reasoning and direction
    # correction, not single-paper support.
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append(
        "- Judge the claim against the literature consensus, not a single decisive paper."
    )
    lines.append(
        "- Direction is decisive: if the corpus attests the relation in the OPPOSITE "
        "direction to the claim, return REFUTE (not UNCERTAIN); do not chain a caveat "
        "into SUPPORT."
    )
    lines.append(
        "- \"Directly\" is judged between the two named entities; a relation that runs "
        "through a distinct intermediate entity does not substantiate a direct claim — "
        "treat mediated-only evidence as UNCERTAIN."
    )
    lines.append(
        "- A direct interaction in the claimed direction is sufficient for SUPPORT even "
        "if the functional magnitude is unquantified, absent contradicting evidence."
    )
    lines.append(
        "- Synonyms / aliases noted during extraction count as on-claim; scope, mechanism "
        "class, and compartment can flip relevance."
    )

    # Sufficiency result is classifier metadata, NOT evidence.
    lines.append("")
    lines.append("## Sufficiency classifier (metadata, not evidence)")
    if state.sufficiency_history:
        last = state.sufficiency_history[-1]
        lines.append(
            f"- Latest result: {last.label} (confidence {last.confidence:.2f}); "
            f"{len(state.sufficiency_history)} check(s) run."
        )
    else:
        lines.append("- No sufficiency check has been run.")

    lines.append("")
    lines.append("## Known evidence digest")
    lines.append(
        build_known_facts_digest(state, max_facts_per_stance=max_facts_per_stance)
    )

    # Caveats: the latest sufficiency gaps, surfaced as caveats and explicitly
    # separated from the evidence above.
    caveats: list[str] = []
    if state.sufficiency_history:
        for g in state.sufficiency_history[-1].gaps:
            gt = getattr(g.gap_type, "value", str(g.gap_type))
            caveats.append(f"[{gt}] {g.description}")
    lines.append("")
    lines.append("## Remaining caveats (do not treat as evidence)")
    if caveats:
        for c in caveats[:8]:
            lines.append(f"- {c}")
    else:
        lines.append("- (none recorded)")

    return "\n".join(lines) + "\n"


def write_verdict_packet(
    state: EvidenceState, *, workspace: Path, label_cfg
) -> Path:
    """Serialize the verdict packet to ``workspace/verdict_packet.md``.

    Returns the written path. Callers that render the workbook should treat a
    failure here as non-fatal (the packet is an aid, not a hard dependency).
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / VERDICT_PACKET_FILENAME
    path.write_text(build_verdict_packet(state, label_cfg=label_cfg), encoding="utf-8")
    return path
