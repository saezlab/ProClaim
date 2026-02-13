"""
MCP Tool Server for Evidence Programming.

Provides 13 tools for the orchestrator agent to manipulate evidence state.
Uses the FastMCP server from the mcp Python SDK.

Launch:  python -m pkevolve.verification.mcp_tools
Connect: Claude Agent SDK connects via stdio transport.
"""

import json
import logging
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pkevolve.verification.classifier import SufficiencyClassifier
from pkevolve.verification.compressor import SufficiencyPreservingCompressor
from pkevolve.verification.data_models import (
    Conflict,
    Fact,
    PaperRecord,
    Stance,
    VerificationVerdict,
)
from pkevolve.verification.evidence_state import EvidenceState, TraceLog

logger = logging.getLogger(__name__)

mcp = FastMCP("evidence-tools")

# Shared classifier and compressor instances
_classifier = SufficiencyClassifier()
_compressor = SufficiencyPreservingCompressor()


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

# Biological action verbs → noun forms for PubMed queries
_BIO_VERB_TO_NOUN: dict[str, str] = {
    "activates": "activation",
    "inhibits": "inhibition",
    "phosphorylates": "phosphorylation",
    "binds": "binding",
    "regulates": "regulation",
    "suppresses": "suppression",
    "promotes": "promotion",
    "blocks": "blocking",
    "induces": "induction",
    "represses": "repression",
    "ubiquitinates": "ubiquitination",
    "methylates": "methylation",
    "acetylates": "acetylation",
    "deactivates": "deactivation",
    "up-regulates": "up-regulation",
    "down-regulates": "down-regulation",
    "upregulates": "up-regulation",
    "downregulates": "down-regulation",
}

_STOP_WORDS: set[str] = {
    "does", "is", "are", "the", "a", "an", "of", "in", "to", "and",
    "or", "that", "this", "it", "by", "with", "from", "for", "on",
    "at", "be", "was", "were", "been", "being", "have", "has", "had",
    "do", "did", "will", "would", "could", "should", "may", "might",
    "can", "shall", "not", "no", "its", "their", "our", "your",
    "directly", "indirectly", "via", "through",
}


@mcp.tool()
def formulate_pubmed_query(claim: str, workspace: str) -> str:
    """Convert a natural-language scientific claim into a structured PubMed
    search query. Extracts gene/protein symbols and maps biological action
    verbs to noun forms. Call this before search_pubmed when you have a
    natural-language claim rather than a pre-formulated query."""
    # Extract likely gene/protein symbols: uppercase, 2-10 chars, may
    # contain digits and hyphens (e.g. MAPK1, H3-3A, TP53, BCL2L1)
    symbols = re.findall(r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b", claim)

    # Map biological verbs to their noun forms
    claim_lower = claim.lower().rstrip(".")
    bio_terms: list[str] = []
    for verb, noun in _BIO_VERB_TO_NOUN.items():
        if verb in claim_lower:
            bio_terms.append(noun)

    # Fallback: use non-stop words if no symbols were extracted
    if not symbols:
        words = re.findall(r"\b\w+\b", claim)
        symbols = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]

    # Build query: symbols joined with AND, bio terms as context
    parts: list[str] = []
    if symbols:
        parts.append(" AND ".join(symbols))
    if bio_terms:
        parts.append("(" + " OR ".join(bio_terms) + ")")

    query = " AND ".join(parts) if parts else claim
    _trace(workspace, "formulate_pubmed_query", {"claim": claim, "query": query})
    return query


def _search_and_add(
    query: str, workspace: str, max_results: int = 5,
    _retries: int = 2, _delay: float = 1.0,
) -> tuple[int, int, list[str]]:
    """Internal: search PubMed and add papers to state.

    Returns (found_count, added_count, added_pmids).
    Retries on transient XML parse errors (e.g. NCBI rate limits).
    """
    import time
    from pkevolve.search.custom_pubmed import RelevancePubMedSearcher

    state = _load_state(workspace)
    searcher = RelevancePubMedSearcher()

    papers = []
    for attempt in range(_retries + 1):
        try:
            papers = searcher.search(query, max_results=max_results)
            break
        except Exception:
            if attempt < _retries:
                time.sleep(_delay * (attempt + 1))
            else:
                # All retries exhausted — return 0 results rather than crash
                return 0, 0, []

    added_pmids: list[str] = []
    for paper in papers:
        pmid = paper.paper_id
        if pmid in state.papers:
            continue
        record = PaperRecord(
            pmid=pmid,
            title=paper.title or "",
            abstract=paper.abstract or "",
            authors=[a for a in getattr(paper, "authors", [])],
            source="pubmed",
        )
        state.add_paper(record)
        added_pmids.append(pmid)

    state.token_estimate = state.token_count()
    _save_state(workspace, state)
    return len(papers), len(added_pmids), added_pmids


def _extract_symbol_subtokens(symbol: str) -> list[str]:
    """Extract family-level sub-tokens from a gene symbol.

    Examples: H3-3A → ['H3'], MAPK1 → ['MAPK'], BCL2L1 → ['BCL2']
    """
    subtokens: list[str] = []
    # Split on hyphens: H3-3A → H3
    parts = symbol.split("-")
    if len(parts) > 1 and len(parts[0]) >= 2:
        subtokens.append(parts[0])
    # Strip trailing digits: MAPK1 → MAPK
    prefix = re.sub(r"\d+$", "", symbol)
    if prefix and prefix != symbol and len(prefix) >= 2:
        subtokens.append(prefix)
    return list(set(subtokens))


def _generate_tiered_queries(
    symbols: list[str], bio_terms: list[str],
) -> list[tuple[str, str]]:
    """Generate queries from most specific to broadest.

    Returns list of (tier_name, query_string) tuples.
    Tier order:
      1. strict_pair_mechanism — all symbols AND bio terms
      2. strict_pair           — all symbols, no bio terms
      3. bridge_mechanism      — each symbol + bio terms
      4. bridge_cross          — symbol + sub-token of other symbol + bio term
    """
    queries: list[tuple[str, str]] = []

    # Tier 1: All symbols AND bio terms
    if symbols and bio_terms:
        q = " AND ".join(symbols) + " AND (" + " OR ".join(bio_terms) + ")"
        queries.append(("strict_pair_mechanism", q))

    # Tier 2: All symbols only
    if len(symbols) >= 2:
        q = " AND ".join(symbols)
        queries.append(("strict_pair", q))

    # Tier 3: Each symbol + bio terms (bridge through mechanism)
    if bio_terms:
        for sym in symbols:
            q = f"{sym} AND ({' OR '.join(bio_terms)})"
            queries.append(("bridge_mechanism", q))

    # Tier 4: Cross-symbol bridge queries
    if len(symbols) >= 2:
        for i, sym in enumerate(symbols):
            other_syms = [s for j, s in enumerate(symbols) if j != i]
            for other in other_syms:
                for subtoken in _extract_symbol_subtokens(other):
                    if subtoken != sym:
                        bridge_q = f"{sym} {subtoken}"
                        if bio_terms:
                            bridge_q += f" {bio_terms[0]}"
                        queries.append(("bridge_cross", bridge_q))

    return queries


@mcp.tool()
def search_pubmed(query: str, workspace: str, max_results: int = 5) -> str:
    """Search PubMed for papers relevant to a query and add them to the
    evidence state. Returns list of PMIDs added (skips duplicates)."""
    found, added, added_pmids = _search_and_add(query, workspace, max_results)
    _trace(workspace, "search_pubmed", {
        "query": query, "found": found, "added": added,
    })
    return (
        f"Found {found} papers, added {added} new. "
        f"PMIDs added: {', '.join(added_pmids) if added_pmids else 'none'}"
    )


@mcp.tool()
def search_for_gap(
    gap_description: str, gap_type: str, workspace: str, max_results: int = 3
) -> str:
    """Retrieve papers targeting a specific evidence gap identified by the
    classifier. Uses gap_type context to guide the search."""
    return search_pubmed(
        query=gap_description, workspace=workspace, max_results=max_results
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
    # Parse claim components (same logic as formulate_pubmed_query)
    symbols = re.findall(r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b", claim)

    claim_lower = claim.lower().rstrip(".")
    bio_terms: list[str] = []
    for verb, noun in _BIO_VERB_TO_NOUN.items():
        if verb in claim_lower:
            bio_terms.append(noun)

    if not symbols:
        words = re.findall(r"\b\w+\b", claim)
        symbols = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]

    tiers = _generate_tiered_queries(symbols, bio_terms)

    import time

    total_added = 0
    tier_results: list[str] = []

    for idx, (tier_name, query) in enumerate(tiers):
        # Rate-limit: NCBI allows 3 req/s without API key
        if idx > 0:
            time.sleep(0.4)

        found, added, added_pmids = _search_and_add(
            query, workspace, max_results=max_results_per_tier,
        )
        total_added += added
        tier_results.append(
            f"  [{tier_name}] {query} → found {found}, added {added}"
        )
        _trace(workspace, "search_pubmed_progressive_tier", {
            "tier": tier_name, "query": query,
            "found": found, "added": added,
        })

        # Stop escalating once we have papers from bridge queries or beyond
        if total_added >= max_results_per_tier and tier_name not in (
            "strict_pair_mechanism", "strict_pair",
        ):
            break

    lines = [f"Progressive search for: {claim}"]
    lines.extend(tier_results)
    lines.append(f"\nTotal papers added: {total_added}")

    _trace(workspace, "search_pubmed_progressive", {
        "claim": claim, "total_added": total_added,
        "tiers_tried": len(tier_results),
    })

    return "\n".join(lines)


# NCBI E-utilities base URLs (shared with RelevancePubMedSearcher)
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ELINK_URL = f"{_EUTILS_BASE}/elink.fcgi"
_EFETCH_URL = f"{_EUTILS_BASE}/efetch.fcgi"
_PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


@mcp.tool()
def find_related_articles(
    pmid: str, workspace: str, max_results: int = 5
) -> str:
    """Find articles related to a given PMID using PubMed's citation
    co-occurrence algorithm (\"Similar Articles\"). Useful when keyword
    searches return no results -- this discovers papers through citation
    links rather than text matching. Adds results to the evidence state."""
    import requests
    from xml.etree import ElementTree as ET

    # Step 1: Use elink to find related PMIDs
    link_params = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "id": pmid,
        "cmd": "neighbor_score",
        "retmode": "xml",
    }
    try:
        link_resp = requests.get(_ELINK_URL, params=link_params, timeout=30)
        link_resp.raise_for_status()
    except requests.RequestException as e:
        _trace(workspace, "find_related_articles", {"pmid": pmid, "error": str(e)})
        return f"Error querying PubMed elink for PMID {pmid}: {e}"

    link_root = ET.fromstring(link_resp.content)

    # Extract related PMIDs (skip the query PMID itself)
    related_pmids: list[str] = []
    for link in link_root.findall(".//LinkSetDb/Link"):
        rid = link.findtext("Id", "")
        if rid and rid != pmid:
            related_pmids.append(rid)
        if len(related_pmids) >= max_results:
            break

    if not related_pmids:
        _trace(workspace, "find_related_articles", {"pmid": pmid, "found": 0})
        return f"No related articles found for PMID {pmid}."

    # Step 2: Fetch metadata for related PMIDs
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(related_pmids),
        "retmode": "xml",
    }
    try:
        fetch_resp = requests.get(_EFETCH_URL, params=fetch_params, timeout=30)
        fetch_resp.raise_for_status()
    except requests.RequestException as e:
        _trace(workspace, "find_related_articles", {"pmid": pmid, "error": str(e)})
        return f"Error fetching related article metadata: {e}"

    fetch_root = ET.fromstring(fetch_resp.content)

    # Step 3: Parse and add to evidence state
    state = _load_state(workspace)
    added_pmids: list[str] = []

    for article in fetch_root.findall(".//PubmedArticle"):
        try:
            art_pmid = article.findtext(".//PMID", "")
            if not art_pmid or art_pmid in state.papers:
                continue

            title = article.findtext(".//ArticleTitle", "") or ""
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""

            authors: list[str] = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())

            record = PaperRecord(
                pmid=art_pmid,
                title=title,
                abstract=abstract or "",
                authors=authors,
                source="pubmed",
            )
            state.add_paper(record)
            added_pmids.append(art_pmid)
        except Exception as e:
            logger.warning("Error parsing related article: %s", e)

    state.token_estimate = state.token_count()
    _save_state(workspace, state)
    _trace(workspace, "find_related_articles", {
        "seed_pmid": pmid, "found": len(related_pmids), "added": len(added_pmids),
    })

    return (
        f"Found {len(related_pmids)} related articles for PMID {pmid}, "
        f"added {len(added_pmids)} new. "
        f"PMIDs added: {', '.join(added_pmids) if added_pmids else 'none'}"
    )


@mcp.tool()
def get_full_text_article(pmid: str, workspace: str) -> str:
    """Attempt to retrieve full text for a paper via PubMed Central Open
    Access. Updates the paper's full_text field in the evidence state.
    Falls back to abstract if full text is not available in PMC OA."""
    import requests
    from xml.etree import ElementTree as ET

    state = _load_state(workspace)
    paper = state.papers.get(pmid)
    if not paper:
        return f"Paper {pmid} not found in evidence state."

    # Return cached full text if already fetched
    if paper.full_text:
        _trace(workspace, "get_full_text_article", {"pmid": pmid, "cached": True})
        return (
            f"PMID: {paper.pmid}\n"
            f"Title: {paper.title}\n\n"
            f"{paper.full_text[:8000]}"
        )

    # Step 1: Convert PMID to PMCID via elink
    link_params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "xml",
    }
    pmcid = None
    try:
        link_resp = requests.get(_ELINK_URL, params=link_params, timeout=30)
        link_resp.raise_for_status()
        link_root = ET.fromstring(link_resp.content)
        pmc_link = link_root.find(".//LinkSetDb/Link/Id")
        if pmc_link is not None and pmc_link.text:
            pmcid = pmc_link.text
    except requests.RequestException as e:
        logger.warning("elink PMID->PMC failed for %s: %s", pmid, e)

    if not pmcid:
        _trace(workspace, "get_full_text_article", {
            "pmid": pmid, "pmcid": None, "status": "no_pmc_record",
        })
        return (
            f"No PMC full text available for PMID {pmid}. "
            f"Falling back to abstract.\n\n"
            f"Title: {paper.title}\n\n"
            f"{paper.abstract}"
        )

    # Step 2: Fetch full text XML from PMC
    fetch_params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "xml",
        "retmode": "xml",
    }
    try:
        fetch_resp = requests.get(_EFETCH_URL, params=fetch_params, timeout=60)
        fetch_resp.raise_for_status()
        fetch_root = ET.fromstring(fetch_resp.content)
    except requests.RequestException as e:
        _trace(workspace, "get_full_text_article", {
            "pmid": pmid, "pmcid": pmcid, "error": str(e),
        })
        return (
            f"Error fetching PMC full text for PMID {pmid} (PMC{pmcid}): {e}. "
            f"Falling back to abstract.\n\n{paper.abstract}"
        )

    # Step 3: Extract body text from the JATS XML
    body_parts: list[str] = []
    for elem in fetch_root.iter():
        if elem.tag in ("p", "title") and elem.text:
            body_parts.append(elem.text.strip())

    full_text = "\n\n".join(body_parts) if body_parts else ""

    if not full_text:
        _trace(workspace, "get_full_text_article", {
            "pmid": pmid, "pmcid": pmcid, "status": "empty_body",
        })
        return (
            f"PMC record found (PMC{pmcid}) but body text could not be "
            f"extracted. Falling back to abstract.\n\n{paper.abstract}"
        )

    # Step 4: Store in evidence state
    paper.full_text = full_text
    state.token_estimate = state.token_count()
    _save_state(workspace, state)

    _trace(workspace, "get_full_text_article", {
        "pmid": pmid, "pmcid": pmcid, "chars": len(full_text),
    })

    return (
        f"PMID: {paper.pmid} (PMC{pmcid})\n"
        f"Title: {paper.title}\n\n"
        f"{full_text[:8000]}"
    )


# ---------------------------------------------------------------------------
# Evidence State Tools (5)
# ---------------------------------------------------------------------------


@mcp.tool()
def get_evidence_summary(workspace: str) -> str:
    """Get a summary of the current evidence state: paper count, fact count,
    coverage per subclaim, conflicts, and latest sufficiency result."""
    state = _load_state(workspace)

    lines = [
        f"Claim: {state.claim}",
        f"Iteration: {state.iteration}",
        f"Papers: {len(state.papers)}",
        f"Facts: {len(state.facts)} ("
        f"support: {sum(1 for f in state.facts if f.stance == Stance.SUPPORT)}, "
        f"refute: {sum(1 for f in state.facts if f.stance == Stance.REFUTE)})",
        f"Conflicts: {len(state.conflicts)}",
        f"Token estimate: ~{state.token_estimate}",
        "",
        "Coverage per subclaim:",
    ]
    for sc in state.subclaims:
        cov = state.coverage.get(sc, 0.0)
        synth = "yes" if sc in state.synthesis else "no"
        lines.append(f"  [{cov:.1%}] [synth:{synth}] {sc}")

    if state.sufficiency_history:
        latest = state.sufficiency_history[-1]
        lines.extend([
            "",
            f"Latest sufficiency: {latest.label} (confidence: {latest.confidence:.2f})",
            f"Gaps: {len(latest.gaps)}",
        ])
        for gap in latest.gaps[:5]:
            lines.append(f"  - [{gap.gap_type.value}] {gap.description}")

    _trace(workspace, "get_evidence_summary", {})
    return "\n".join(lines)


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

    added = 0
    for item in facts_data:
        stance_str = item.get("stance", "NEUTRAL").upper()
        if stance_str not in ("SUPPORT", "REFUTE", "NEUTRAL"):
            stance_str = "NEUTRAL"

        fact = Fact(
            id=f"fact_{len(state.facts) + added}",
            text=item.get("text", ""),
            stance=Stance(stance_str),
            source_pmid=item.get("source_pmid", ""),
            relevant_subclaims=item.get("relevant_subclaims", []),
            confidence=item.get("confidence", 0.5),
        )
        state.add_fact(fact)
        added += 1

    # Recompute coverage
    for sc in state.subclaims:
        relevant = [
            f for f in state.facts
            if sc in f.relevant_subclaims and f.stance != Stance.NEUTRAL
        ]
        state.coverage[sc] = min(1.0, len(relevant) / 3.0)

    state.token_estimate = state.token_count()
    _save_state(workspace, state)
    _trace(workspace, "add_facts", {"count": added})
    return f"Added {added} facts. Coverage updated."


@mcp.tool()
def get_paper_text(pmid: str, workspace: str) -> str:
    """Retrieve the full abstract/text for a specific paper by PMID."""
    state = _load_state(workspace)
    paper = state.papers.get(pmid)
    if not paper:
        return f"Paper {pmid} not found in evidence state."

    _trace(workspace, "get_paper_text", {"pmid": pmid})
    return (
        f"PMID: {paper.pmid}\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors[:5])}\n\n"
        f"{paper.abstract}"
    )


@mcp.tool()
def update_synthesis(subclaim: str, synthesis_text: str, workspace: str) -> str:
    """Update the evidence synthesis for a specific subclaim."""
    state = _load_state(workspace)
    state.synthesis[subclaim] = synthesis_text
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
    conflict = Conflict(
        id=f"conflict_{len(state.conflicts)}",
        fact_a_id=fact_a_id,
        fact_b_id=fact_b_id,
        description=description,
        severity=severity,
    )
    state.add_conflict(conflict)
    _save_state(workspace, state)
    _trace(workspace, "add_conflict", {"conflict_id": conflict.id})
    return f"Conflict recorded: {conflict.id}"


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
    if state.iteration >= 8:
        return (
            "ITERATION LIMIT REACHED (8/8). "
            "You must now call emit_verdict to produce your final verdict. "
            "No further evidence gathering is permitted."
        )

    result = _classifier(state)

    # Persist to state history
    state.sufficiency_history.append(result)
    state.iteration += 1
    _save_state(workspace, state)
    _trace(workspace, "check_sufficiency", {
        "label": result.label,
        "confidence": result.confidence,
        "gaps": len(result.gaps),
    })

    # Format feedback
    lines = [
        f"=== SUFFICIENCY CHECK (iteration {state.iteration}/8) ===",
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

    compressed = _compressor.compress(state, state.claim)

    after = compressed.token_count()
    compressed.token_estimate = after
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

    v = VerificationVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        key_evidence=key_evidence,
        gaps_remaining=gaps_remaining,
    )

    verdict_path = Path(workspace) / "verdict.json"
    verdict_path.write_text(v.model_dump_json(indent=2))

    _trace(workspace, "emit_verdict", {
        "verdict": verdict, "confidence": confidence,
    })

    return f"Verdict emitted: {verdict} (confidence: {confidence:.2f}). Saved to {verdict_path}."


if __name__ == "__main__":
    mcp.run()
