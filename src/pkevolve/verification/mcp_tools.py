"""
MCP Tool Server for Evidence Programming.

Thin MCP wrappers over the pure-Python evidence_api module.
Each tool loads state from disk, delegates to evidence_api, saves state back,
and appends to the trace log. In REPL mode the evidence_api functions are
called directly without these wrappers.

Provides 13 tools for the orchestrator agent to manipulate evidence state.
Uses the FastMCP server from the mcp Python SDK.

Launch:  python -m pkevolve.verification.mcp_tools
Connect: Claude Agent SDK connects via stdio transport.
"""

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pkevolve.verification.data_models import (
    Stance,
    VerificationVerdict,
)
from pkevolve.verification.evidence_state import EvidenceState, TraceLog
import pkevolve.verification.evidence_api as api

logger = logging.getLogger(__name__)

mcp = FastMCP("evidence-tools")


# ---------------------------------------------------------------------------
# Disk I/O helpers (MCP-only — not used in REPL mode)
# ---------------------------------------------------------------------------


def _load_state(workspace: str) -> EvidenceState:
    """Load evidence state from workspace directory."""
    state_path = Path(workspace) / "evidence_state.json"
    return EvidenceState.load(state_path)


def _save_state(workspace: str, state: EvidenceState) -> None:
    """Save evidence state to workspace directory."""
    state_path = Path(workspace) / "evidence_state.json"
    state.save(state_path)


def _trace(workspace: str, operation: str, details: dict) -> None:
    """Append to trace log."""
    trace_path = Path(workspace) / "trace.json"
    TraceLog(trace_path).append(operation, details)


# ---------------------------------------------------------------------------
# Retrieval Tools (5)
# ---------------------------------------------------------------------------


@mcp.tool()
def formulate_pubmed_query(claim: str, workspace: str) -> str:
    """Convert a natural-language scientific claim into a structured PubMed
    search query. Extracts gene/protein symbols and maps biological action
    verbs to noun forms. Call this before search_pubmed when you have a
    natural-language claim rather than a pre-formulated query."""
    query = api.formulate_pubmed_query(claim)
    _trace(workspace, "formulate_pubmed_query", {"claim": claim, "query": query})
    return query


@mcp.tool()
def search_pubmed(query: str, workspace: str, max_results: int = 5) -> str:
    """Search PubMed for papers relevant to a query and add them to the
    evidence state. Returns list of PMIDs added (skips duplicates)."""
    state = _load_state(workspace)
    added_pmids = api.search_pubmed(query, state, max_results)
    _save_state(workspace, state)
    _trace(workspace, "search_pubmed", {
        "query": query, "added": len(added_pmids),
    })
    return (
        f"Added {len(added_pmids)} new papers. "
        f"PMIDs: {', '.join(added_pmids) if added_pmids else 'none'}"
    )


@mcp.tool()
def search_for_gap(
    gap_description: str, gap_type: str, workspace: str, max_results: int = 3,
) -> str:
    """Retrieve papers targeting a specific evidence gap identified by the
    classifier. Uses gap_type context to guide the search."""
    state = _load_state(workspace)
    added_pmids = api.search_for_gap(gap_description, state, max_results)
    _save_state(workspace, state)
    _trace(workspace, "search_for_gap", {
        "gap_description": gap_description, "gap_type": gap_type,
        "added": len(added_pmids),
    })
    return (
        f"Added {len(added_pmids)} new papers for gap. "
        f"PMIDs: {', '.join(added_pmids) if added_pmids else 'none'}"
    )


@mcp.tool()
def search_pubmed_progressive(
    claim: str, workspace: str, max_results_per_tier: int = 5,
) -> str:
    """Progressive PubMed search with automatic query broadening.

    Tries queries from most specific (exact gene pair + mechanism) to broader
    (bridge queries connecting each gene to shared biology). Stops escalating
    once enough papers are found.

    Use this instead of formulate_pubmed_query + search_pubmed for initial
    evidence retrieval, especially when the gene pair may not co-occur in
    any single paper.
    """
    state = _load_state(workspace)
    added_pmids = api.search_pubmed_progressive(
        claim, state, max_results_per_tier,
    )
    _save_state(workspace, state)
    _trace(workspace, "search_pubmed_progressive", {
        "claim": claim, "total_added": len(added_pmids),
    })
    return f"Progressive search complete. Added {len(added_pmids)} papers."


@mcp.tool()
def find_related_articles(
    pmid: str, workspace: str, max_results: int = 5,
) -> str:
    """Find articles related to a given PMID using PubMed's citation
    co-occurrence algorithm (\"Similar Articles\"). Useful when keyword
    searches return no results -- this discovers papers through citation
    links rather than text matching. Adds results to the evidence state."""
    state = _load_state(workspace)
    added_pmids = api.find_related_articles(pmid, state, max_results)
    _save_state(workspace, state)
    _trace(workspace, "find_related_articles", {
        "seed_pmid": pmid, "added": len(added_pmids),
    })
    return (
        f"Found related articles for PMID {pmid}, "
        f"added {len(added_pmids)} new. "
        f"PMIDs: {', '.join(added_pmids) if added_pmids else 'none'}"
    )


@mcp.tool()
def get_full_text_article(pmid: str, workspace: str) -> str:
    """Attempt to retrieve full text for a paper via PubMed Central Open
    Access. Updates the paper's full_text field in the evidence state.
    Falls back to abstract if full text is not available in PMC OA."""
    state = _load_state(workspace)
    text = api.get_full_text_article(pmid, state)
    _save_state(workspace, state)
    _trace(workspace, "get_full_text_article", {"pmid": pmid, "chars": len(text)})
    paper = state.papers.get(pmid)
    title = paper.title if paper else ""
    return f"PMID: {pmid}\nTitle: {title}\n\n{text[:8000]}"


# ---------------------------------------------------------------------------
# Evidence State Tools (5)
# ---------------------------------------------------------------------------


@mcp.tool()
def get_evidence_summary(workspace: str) -> str:
    """Get a summary of the current evidence state: paper count, fact count,
    coverage per subclaim, conflicts, and latest sufficiency result."""
    state = _load_state(workspace)
    summary = api.get_evidence_summary(state)
    _trace(workspace, "get_evidence_summary", {})
    return summary


@mcp.tool()
def add_facts(facts_json: str, workspace: str) -> str:
    """Add extracted facts to the evidence state. Input: JSON array of fact
    objects with keys: text, stance (SUPPORT/REFUTE/NEUTRAL), source_pmid,
    relevant_subclaims (optional list), confidence (optional float)."""
    state = _load_state(workspace)
    try:
        facts_data = json.loads(facts_json)
    except json.JSONDecodeError as e:
        return f"Error parsing facts JSON: {e}"

    added = api.add_facts_from_dicts(facts_data, state)
    _save_state(workspace, state)
    _trace(workspace, "add_facts", {"count": added})
    return f"Added {added} facts. Coverage updated."


@mcp.tool()
def get_paper_text(pmid: str, workspace: str) -> str:
    """Retrieve the full abstract/text for a specific paper by PMID."""
    state = _load_state(workspace)
    text = api.get_paper_text(pmid, state)
    _trace(workspace, "get_paper_text", {"pmid": pmid})
    return text


@mcp.tool()
def update_synthesis(subclaim: str, synthesis_text: str, workspace: str) -> str:
    """Update the evidence synthesis for a specific subclaim."""
    state = _load_state(workspace)
    api.update_synthesis(subclaim, synthesis_text, state)
    _save_state(workspace, state)
    _trace(workspace, "update_synthesis", {"subclaim": subclaim})
    return f"Synthesis updated for: {subclaim}"


@mcp.tool()
def add_conflict(
    fact_a_id: str,
    fact_b_id: str,
    description: str,
    severity: float,
    workspace: str,
) -> str:
    """Record a detected conflict between two facts."""
    state = _load_state(workspace)
    conflict_id = api.add_conflict(fact_a_id, fact_b_id, description, severity, state)
    _save_state(workspace, state)
    _trace(workspace, "add_conflict", {"conflict_id": conflict_id})
    return f"Conflict recorded: {conflict_id}"


# ---------------------------------------------------------------------------
# Sufficiency Tool (1)
# ---------------------------------------------------------------------------


@mcp.tool()
def check_sufficiency(workspace: str) -> str:
    """Run the sufficiency classifier on the current evidence state. Returns
    label, confidence, and identified gaps. This is cheap (no LLM call) --
    call it frequently to guide your next action. Think of this as running
    your test suite."""
    state = _load_state(workspace)

    # Enforce iteration limit
    if state.iteration >= EvidenceState.MAX_ITERATIONS:
        return (
            f"ITERATION LIMIT REACHED ({EvidenceState.MAX_ITERATIONS}/"
            f"{EvidenceState.MAX_ITERATIONS}). "
            "You must now call emit_verdict to produce your final verdict. "
            "No further evidence gathering is permitted."
        )

    result = api.check_sufficiency(state)
    _save_state(workspace, state)
    _trace(workspace, "check_sufficiency", {
        "label": result.label,
        "confidence": result.confidence,
        "gaps": len(result.gaps),
    })

    # Format feedback for MCP response
    lines = [
        f"=== SUFFICIENCY CHECK (iteration {state.iteration}/{EvidenceState.MAX_ITERATIONS}) ===",
        f"Label: {result.label}",
        f"Confidence: {result.confidence:.3f}",
        f"Threshold: 0.80",
        f"Status: {'SUFFICIENT' if result.confidence >= 0.80 and result.label != 'INSUFFICIENT' else 'INSUFFICIENT'}",
    ]
    if result.gaps:
        lines.append(f"\nIdentified gaps ({len(result.gaps)}):")
        for gap in result.gaps:
            lines.append(f"  [{gap.priority:.1f}] {gap.gap_type.value}: {gap.description}")
            lines.append(f"         Subclaim: {gap.subclaim}")
    else:
        lines.append("\nNo specific gaps identified.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compression Tool (1)
# ---------------------------------------------------------------------------


@mcp.tool()
def compress_evidence(workspace: str, target_tokens: int = 40000) -> str:
    """Compress the evidence state to fit within token budget while preserving
    sufficiency. Uses L1 deduplication. Reports tokens before and after."""
    state = _load_state(workspace)
    before = state.token_count()

    compressed = api.compress_evidence(state, target_tokens)
    after = compressed.token_count()

    _save_state(workspace, compressed)
    _trace(workspace, "compress_evidence", {
        "before": before, "after": after, "target": target_tokens,
    })

    return (
        f"Compression complete. Level: L1 (deduplication). "
        f"Tokens: {before} -> {after}. "
        f"{'Within budget.' if after <= target_tokens else 'Still over budget.'}"
    )


# ---------------------------------------------------------------------------
# Verdict Tool (1)
# ---------------------------------------------------------------------------


@mcp.tool()
def emit_verdict(
    verdict: str,
    confidence: float,
    reasoning: str,
    key_evidence_json: str,
    gaps_remaining_json: str,
    workspace: str,
) -> str:
    """Emit the final verification verdict. Call this when sufficiency is
    reached or max iterations exceeded. Writes verdict to workspace/verdict.json."""
    try:
        key_evidence = json.loads(key_evidence_json)
    except json.JSONDecodeError:
        key_evidence = [key_evidence_json]

    try:
        gaps_remaining = json.loads(gaps_remaining_json)
    except json.JSONDecodeError:
        gaps_remaining = [gaps_remaining_json]

    v = api.emit_verdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        key_evidence=key_evidence,
        gaps_remaining=gaps_remaining,
        state=_load_state(workspace),
        workspace=Path(workspace),
    )
    _trace(workspace, "emit_verdict", {
        "verdict": verdict, "confidence": confidence,
    })
    return f"Verdict emitted: {verdict} (confidence: {confidence:.2f}). Saved to {Path(workspace) / 'verdict.json'}."


if __name__ == "__main__":
    mcp.run()
