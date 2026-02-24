"""
Evidence API — pure Python library for evidence manipulation.

All functions operate on EvidenceState objects in-memory. No disk I/O,
no MCP dependency. This is the importable API surface that both the REPL
kernel and the MCP wrappers call.

The MCP tools in mcp_tools.py become thin wrappers around these functions,
adding only load/save/trace bookends.

In REPL mode, the LLM calls these functions directly::

    papers = search_pubmed("MAPK1 activation", state)
    result = check_sufficiency(state)
    state = compress_evidence(state, state.claim)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from pkevolve.verification.classifier import SufficiencyClassifier
from pkevolve.verification.compressor import SufficiencyPreservingCompressor
from pkevolve.verification.data_models import (
    Conflict,
    Fact,
    PaperRecord,
    Stance,
    SufficiencyResult,
    VerificationVerdict,
)
from pkevolve.verification.evidence_state import EvidenceState

logger = logging.getLogger(__name__)

# Shared singleton instances
_classifier = SufficiencyClassifier()
_compressor = SufficiencyPreservingCompressor()

# Maximum sufficiency checks before forced verdict
MAX_ITERATIONS = 8

# ---------------------------------------------------------------------------
# Biological verb → noun mapping (for PubMed query formulation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MaxIterationsExceeded(Exception):
    """Raised when the iteration limit is reached."""

    def __init__(self, *args, state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state


# ---------------------------------------------------------------------------
# Query Formulation
# ---------------------------------------------------------------------------


def formulate_pubmed_query(claim: str) -> str:
    """Convert a natural-language claim into a structured PubMed query.

    Extracts gene/protein symbols and maps biological action verbs to
    noun forms. Returns the query string.
    """
    symbols = re.findall(r"\b[A-Z][A-Z0-9](?:[A-Z0-9\-]{0,8})\b", claim)

    claim_lower = claim.lower().rstrip(".")
    bio_terms: list[str] = []
    for verb, noun in _BIO_VERB_TO_NOUN.items():
        if verb in claim_lower:
            bio_terms.append(noun)

    if not symbols:
        words = re.findall(r"\b\w+\b", claim)
        symbols = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]

    parts: list[str] = []
    if symbols:
        parts.append(" AND ".join(symbols))
    if bio_terms:
        parts.append("(" + " OR ".join(bio_terms) + ")")

    return " AND ".join(parts) if parts else claim


def _extract_symbol_subtokens(symbol: str) -> list[str]:
    """Extract family-level sub-tokens from a gene symbol."""
    subtokens: list[str] = []
    parts = symbol.split("-")
    if len(parts) > 1 and len(parts[0]) >= 2:
        subtokens.append(parts[0])
    prefix = re.sub(r"\d+$", "", symbol)
    if prefix and prefix != symbol and len(prefix) >= 2:
        subtokens.append(prefix)
    return list(set(subtokens))


def _generate_tiered_queries(
    symbols: list[str], bio_terms: list[str],
) -> list[tuple[str, str]]:
    """Generate queries from most specific to broadest.

    Returns list of (tier_name, query_string) tuples.
    """
    queries: list[tuple[str, str]] = []

    if symbols and bio_terms:
        q = " AND ".join(symbols) + " AND (" + " OR ".join(bio_terms) + ")"
        queries.append(("strict_pair_mechanism", q))

    if len(symbols) >= 2:
        q = " AND ".join(symbols)
        queries.append(("strict_pair", q))

    if bio_terms:
        for sym in symbols:
            q = f"{sym} AND ({' OR '.join(bio_terms)})"
            queries.append(("bridge_mechanism", q))

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


# ---------------------------------------------------------------------------
# Retrieval Functions
# ---------------------------------------------------------------------------


def _search_and_add(
    query: str,
    state: EvidenceState,
    max_results: int = 5,
    _retries: int = 2,
    _delay: float = 1.0,
) -> tuple[int, int, list[str]]:
    """Search PubMed and add papers to state in-memory.

    Returns (found_count, added_count, added_pmids).
    """
    from pkevolve.search.custom_pubmed import RelevancePubMedSearcher

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
    return len(papers), len(added_pmids), added_pmids


def search_pubmed(
    query: str, state: EvidenceState, max_results: int = 5,
) -> list[str]:
    """Search PubMed and add papers to state. Returns list of added PMIDs."""
    found, added, added_pmids = _search_and_add(query, state, max_results)
    print(
        f"PubMed: found {found}, added {added} new. "
        f"PMIDs: {', '.join(added_pmids) if added_pmids else 'none'}"
    )
    return added_pmids


def search_pubmed_progressive(
    claim: str,
    state: EvidenceState,
    max_results_per_tier: int = 5,
) -> list[str]:
    """Progressive PubMed search with automatic query broadening.

    Tries queries from most specific to broadest, stopping once enough
    papers are found. Returns list of all added PMIDs.
    """
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

    total_added = 0
    all_added_pmids: list[str] = []
    tier_results: list[str] = []

    for idx, (tier_name, query) in enumerate(tiers):
        if idx > 0:
            time.sleep(0.4)

        found, added, added_pmids = _search_and_add(
            query, state, max_results=max_results_per_tier,
        )
        total_added += added
        all_added_pmids.extend(added_pmids)
        tier_results.append(
            f"  [{tier_name}] {query} → found {found}, added {added}"
        )

        if total_added >= max_results_per_tier and tier_name not in (
            "strict_pair_mechanism", "strict_pair",
        ):
            break

    print(f"Progressive search for: {claim}")
    for line in tier_results:
        print(line)
    print(f"Total papers added: {total_added}")

    return all_added_pmids


def search_for_gap(
    gap_description: str,
    state: EvidenceState,
    max_results: int = 3,
) -> list[str]:
    """Search for papers targeting a specific evidence gap."""
    return search_pubmed(gap_description, state, max_results)


def find_related_articles(
    pmid: str,
    state: EvidenceState,
    max_results: int = 5,
) -> list[str]:
    """Find related PubMed articles via citation co-occurrence.

    Returns list of added PMIDs.
    """
    import requests
    from xml.etree import ElementTree as ET

    _EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    _ELINK_URL = f"{_EUTILS_BASE}/elink.fcgi"
    _EFETCH_URL = f"{_EUTILS_BASE}/efetch.fcgi"

    # Step 1: elink to find related PMIDs
    link_params = {
        "dbfrom": "pubmed", "db": "pubmed", "id": pmid,
        "cmd": "neighbor_score", "retmode": "xml",
    }
    try:
        link_resp = requests.get(_ELINK_URL, params=link_params, timeout=30)
        link_resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error querying elink for PMID {pmid}: {e}")
        return []

    link_root = ET.fromstring(link_resp.content)
    related_pmids: list[str] = []
    for link in link_root.findall(".//LinkSetDb/Link"):
        rid = link.findtext("Id", "")
        if rid and rid != pmid:
            related_pmids.append(rid)
        if len(related_pmids) >= max_results:
            break

    if not related_pmids:
        print(f"No related articles found for PMID {pmid}.")
        return []

    # Step 2: Fetch metadata
    fetch_params = {
        "db": "pubmed", "id": ",".join(related_pmids), "retmode": "xml",
    }
    try:
        fetch_resp = requests.get(_EFETCH_URL, params=fetch_params, timeout=30)
        fetch_resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching related article metadata: {e}")
        return []

    fetch_root = ET.fromstring(fetch_resp.content)
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
                pmid=art_pmid, title=title, abstract=abstract or "",
                authors=authors, source="pubmed",
            )
            state.add_paper(record)
            added_pmids.append(art_pmid)
        except Exception as e:
            logger.warning("Error parsing related article: %s", e)

    state.token_estimate = state.token_count()
    print(
        f"Related articles for PMID {pmid}: found {len(related_pmids)}, "
        f"added {len(added_pmids)} new."
    )
    return added_pmids


def get_full_text_article(pmid: str, state: EvidenceState) -> str:
    """Attempt to retrieve full text via PubMed Central Open Access.

    Updates the paper's full_text field in the state. Returns the text
    (full text or abstract fallback).
    """
    import requests
    from xml.etree import ElementTree as ET

    _EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    _ELINK_URL = f"{_EUTILS_BASE}/elink.fcgi"
    _EFETCH_URL = f"{_EUTILS_BASE}/efetch.fcgi"

    paper = state.papers.get(pmid)
    if not paper:
        print(f"Paper {pmid} not found in evidence state.")
        return ""

    if paper.full_text:
        return paper.full_text[:8000]

    # Convert PMID to PMCID
    link_params = {
        "dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "xml",
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
        print(f"No PMC full text for PMID {pmid}. Using abstract.")
        return paper.abstract

    # Fetch full text XML from PMC
    fetch_params = {
        "db": "pmc", "id": pmcid, "rettype": "xml", "retmode": "xml",
    }
    try:
        fetch_resp = requests.get(_EFETCH_URL, params=fetch_params, timeout=60)
        fetch_resp.raise_for_status()
        fetch_root = ET.fromstring(fetch_resp.content)
    except requests.RequestException as e:
        print(f"Error fetching PMC full text: {e}. Using abstract.")
        return paper.abstract

    body_parts: list[str] = []
    for elem in fetch_root.iter():
        if elem.tag in ("p", "title") and elem.text:
            body_parts.append(elem.text.strip())

    full_text = "\n\n".join(body_parts) if body_parts else ""

    if not full_text:
        print(f"PMC record found but body empty. Using abstract.")
        return paper.abstract

    paper.full_text = full_text
    state.token_estimate = state.token_count()
    print(f"Full text retrieved for PMID {pmid} (PMC{pmcid}): {len(full_text)} chars")
    return full_text[:8000]


def get_paper_text(pmid: str, state: EvidenceState) -> str:
    """Get the abstract/text for a paper by PMID."""
    paper = state.papers.get(pmid)
    if not paper:
        print(f"Paper {pmid} not found in evidence state.")
        return ""
    return (
        f"PMID: {paper.pmid}\nTitle: {paper.title}\n"
        f"Authors: {', '.join(paper.authors[:5])}\n\n{paper.abstract}"
    )


# ---------------------------------------------------------------------------
# Evidence Mutation Functions
# ---------------------------------------------------------------------------


def add_facts_from_dicts(
    facts_data: list[dict], state: EvidenceState,
) -> int:
    """Add facts to state from a list of dicts.

    Each dict should have: text, stance, source_pmid.
    Optional: relevant_subclaims, confidence.

    Returns number of facts added.
    """
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
    _recompute_coverage(state)
    state.token_estimate = state.token_count()
    print(f"Added {added} facts. Coverage updated.")
    return added


def _recompute_coverage(state: EvidenceState) -> None:
    """Recompute per-subclaim coverage scores."""
    for sc in state.subclaims:
        relevant = [
            f for f in state.facts
            if sc in f.relevant_subclaims and f.stance != Stance.NEUTRAL
        ]
        state.coverage[sc] = min(1.0, len(relevant) / 3.0)


def update_synthesis(
    subclaim: str, synthesis_text: str, state: EvidenceState,
) -> None:
    """Update the evidence synthesis for a subclaim."""
    state.synthesis[subclaim] = synthesis_text
    print(f"Synthesis updated for: {subclaim}")


def add_conflict(
    fact_a_id: str,
    fact_b_id: str,
    description: str,
    severity: float,
    state: EvidenceState,
) -> str:
    """Record a conflict between two facts. Returns conflict id."""
    conflict = Conflict(
        id=f"conflict_{len(state.conflicts)}",
        fact_a_id=fact_a_id,
        fact_b_id=fact_b_id,
        description=description,
        severity=severity,
    )
    state.add_conflict(conflict)
    print(f"Conflict recorded: {conflict.id}")
    return conflict.id


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def get_evidence_summary(state: EvidenceState) -> str:
    """Get a human-readable summary of the current evidence state."""
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

    summary = "\n".join(lines)
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Sufficiency
# ---------------------------------------------------------------------------


def check_sufficiency(state: EvidenceState) -> SufficiencyResult:
    """Run the sufficiency classifier on the current evidence state.

    Appends result to state.sufficiency_history and increments iteration.
    Raises MaxIterationsExceeded if the iteration limit is reached.
    """
    if state.iteration >= MAX_ITERATIONS:
        raise MaxIterationsExceeded(
            f"Iteration limit ({MAX_ITERATIONS}) reached. "
            "Call emit_verdict() to produce your final verdict.",
            state=state,
        )

    result = _classifier(state)

    state.sufficiency_history.append(result)
    state.iteration += 1

    # Print feedback
    status = (
        "SUFFICIENT" if result.confidence >= 0.80 and result.label != "INSUFFICIENT"
        else "INSUFFICIENT"
    )
    print(f"=== SUFFICIENCY CHECK (iteration {state.iteration}/{MAX_ITERATIONS}) ===")
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Status: {status}")
    if result.gaps:
        print(f"\nIdentified gaps ({len(result.gaps)}):")
        for gap in result.gaps:
            print(f"  [{gap.priority:.1f}] {gap.gap_type.value}: {gap.description}")
    else:
        print("\nNo specific gaps identified.")

    return result


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def compress_evidence(
    state: EvidenceState, target_tokens: int = 40000,
) -> EvidenceState:
    """Compress evidence state while preserving sufficiency.

    Returns a new (compressed) EvidenceState. The caller should reassign::

        state = compress_evidence(state)
    """
    before = state.token_count()
    compressed = _compressor.compress(state, state.claim)
    after = compressed.token_count()
    compressed.token_estimate = after

    within = "Within budget." if after <= target_tokens else "Still over budget."
    print(f"Compression: {before} → {after} tokens. {within}")
    return compressed


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def emit_verdict(
    verdict: str,
    confidence: float,
    reasoning: str,
    key_evidence: list[str],
    gaps_remaining: list[str],
    state: EvidenceState,
    workspace: Optional[Path] = None,
) -> VerificationVerdict:
    """Emit the final verification verdict.

    Optionally writes verdict.json to workspace.
    Returns the VerificationVerdict object.
    """
    v = VerificationVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        key_evidence=key_evidence,
        gaps_remaining=gaps_remaining,
    )

    if workspace is not None:
        verdict_path = Path(workspace) / "verdict.json"
        verdict_path.write_text(v.model_dump_json(indent=2))
        print(f"Verdict emitted: {verdict} (confidence: {confidence:.2f}). Saved to {verdict_path}.")
    else:
        print(f"Verdict emitted: {verdict} (confidence: {confidence:.2f}).")

    return v
