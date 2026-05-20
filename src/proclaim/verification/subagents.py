"""
Subagent functions for evidence programming.

Converts the Claude Agent SDK subagent definitions (.claude/agents/*.md) into
callable Python functions. Each function takes an ``llm`` callable and data,
constructs a prompt, calls ``llm(prompt)``, and parses the structured response.

The ``llm`` callable is injected by the orchestrator:
  - In REPL mode: a closure over openai.OpenAI.chat.completions.create
  - In SDK mode: the recursive llm() primitive

Usage::

    from proclaim.verification.subagents import extract_facts

    # Define your LLM callable
    def llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="glm-4.6",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    facts = extract_facts(llm, paper_text, claim, subclaims)
"""

import json
import logging
import os
import re
from typing import Callable, Optional

from proclaim.verification.data_models import (
    Conflict,
    Fact,
    Gap,
    Stance,
)
from proclaim.verification.prompts import (
    EXTRACT_FACTS,
    SYNTHESIZE_SUBCLAIM,
    DETECT_CONFLICTS,
    IDENTIFY_GAPS,
    FORMULATE_GAP_QUERIES,
    REFINE_SEARCH_QUERY,
)

logger = logging.getLogger(__name__)

_EVIDENCE_DEBUG = os.environ.get("EVIDENCE_DEBUG", "0") == "1"

# Type alias for the LLM callable
LLMCallable = Callable[[str], str]

# Regex to strip <think> reasoning tags from LLM responses
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _clean_llm_json(text: str) -> str:
    """Strip thinking tags and markdown code fences from an LLM response."""
    text = _THINK_RE.sub("", text).strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return text


def _extract_json_array(text: str) -> list | None:
    """Try to extract a JSON array from *text*.

    Applies ``_clean_llm_json`` first, then attempts direct parse and
    bracket-based extraction as a fallback. The fallback searches from the
    first ``[`` to the last ``]``.
    
    Returns the parsed list or ``None`` if no valid JSON array could be found.
    """
    text = _clean_llm_json(text)
    
    # 1. Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
        
    # 2. Heuristic extraction: find first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    
    if start != -1 and end != -1 and start < end:
        array_text = text[start:end+1]
        try:
            data = json.loads(array_text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
            
    # 3. Fallback to bracket pair matching across the string
    # Try multiple sub-strings if there are multiple arrays (rare, but possible)
    current_start = text.find("[")
    while current_start != -1:
        # Find closing bracket from end to avoid nested brackets issues
        current_end = text.rfind("]")
        while current_end > current_start:
            try:
                candidate = text[current_start:current_end+1]
                data = json.loads(candidate)
                if isinstance(data, list):
                    return data
            except Exception:
                current_end = text.rfind("]", current_start, current_end)
        current_start = text.find("[", current_start + 1)
        
    return None


# ---------------------------------------------------------------------------
# Fact Extraction
# ---------------------------------------------------------------------------


def extract_facts(
    llm: LLMCallable,
    paper_text: str,
    claim: str,
    subclaims: list[str],
    source_pmid: str,
    extraction_context: list[str] | None = None,
) -> list[Fact]:
    """Extract stance-labeled facts from a paper.

    Args:
        llm: Callable that takes a prompt string and returns a response string.
        paper_text: Full text or abstract of the paper.
        claim: The claim being verified.
        subclaims: List of subclaims the facts should be mapped to.
        source_pmid: Paper ID of the source paper.
        extraction_context: Optional list of supplementary notes (synonym mappings,
            disambiguation, scope clarifications) injected between subclaims and paper.

    Returns:
        List of Fact objects with stance labels and subclaim mappings.
    """
    from proclaim.verification.config import get_label_config
    label_cfg = get_label_config()

    subclaims_str = "\n".join(f"  - {sc}" for sc in subclaims)
    stance_block = label_cfg.stance_prompt_block()
    stance_options = label_cfg.stance_options_str()
    if extraction_context:
        notes = "\n".join(f"- {note}" for note in extraction_context)
        context_block = f"Supplementary extraction context (use to interpret the paper correctly):\n{notes}\n"
    else:
        context_block = ""
    prompt = EXTRACT_FACTS.format(
        stance_options=stance_options,
        stance_block=stance_block,
        claim=claim,
        subclaims_str=subclaims_str,
        context_block=context_block,
        source_pmid=source_pmid,
        paper_text=paper_text[:50000],
    )

    response = llm(prompt)
    if not response or response.strip() == "[]":
        if _EVIDENCE_DEBUG:
            print(f"extract_facts: No facts found for paper ID {source_pmid}")
        return []
    return _parse_facts_response(response, source_pmid)


def _parse_facts_response(response: str, source_pmid: str) -> list[Fact]:
    """Parse LLM response into Fact objects."""
    data = _extract_json_array(response)
    if data is None:
        logger.warning(
            "Could not parse facts JSON from LLM response (first 500 chars): %s",
            response[:500],
        )
        return []

    from proclaim.verification.config import get_label_config
    label_cfg = get_label_config()

    facts: list[Fact] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        raw_stance = item.get("stance", label_cfg.default_stance)
        stance_str = label_cfg.validate_stance(raw_stance)
        facts.append(Fact(
            id=f"fact_auto_{i}",
            text=item.get("text", ""),
            stance=stance_str,
            source_pmid=item.get("source_pmid", source_pmid),
            relevant_subclaims=item.get("relevant_subclaims", []),
            confidence=item.get("confidence", 0.5),
        ))
    return facts


# ---------------------------------------------------------------------------
# Evidence Synthesis
# ---------------------------------------------------------------------------


def synthesize_subclaim(
    llm: LLMCallable,
    facts: list[Fact],
    subclaim: str,
) -> str:
    """Synthesize evidence for a specific subclaim.

    Args:
        llm: LLM callable.
        facts: Facts relevant to this subclaim.
        subclaim: The subclaim to synthesize evidence for.

    Returns:
        Synthesis text (under 200 words).
    """
    if not facts:
        return f"No facts found for subclaim: {subclaim}"

    facts_str = "\n".join(
        f"  [{f.stance}] {f.text} (paper ID:{f.source_pmid}, conf:{f.confidence:.2f})"
        for f in facts
    )
    prompt = SYNTHESIZE_SUBCLAIM.format(
        subclaim=subclaim,
        facts_str=facts_str,
    )

    return llm(prompt).strip()


# ---------------------------------------------------------------------------
# Conflict Detection
# ---------------------------------------------------------------------------


def detect_conflicts(
    llm: LLMCallable,
    facts: list[Fact],
) -> list[dict]:
    """Detect contradictions among extracted facts.

    Args:
        llm: LLM callable.
        facts: All extracted facts.

    Returns:
        List of conflict dicts with keys: fact_a_id, fact_b_id,
        description, severity.
    """
    if len(facts) < 2:
        return []

    facts_str = "\n".join(
        f"  [{f.id}] [{f.stance}] {f.text} (paper ID:{f.source_pmid})"
        for f in facts
    )
    prompt = DETECT_CONFLICTS.format(
        facts_str=facts_str,
    )

    response = llm(prompt).strip()

    data = _extract_json_array(response)
    if data is None:
        logger.warning(
            "Could not parse conflicts JSON from LLM response (first 500 chars): %s",
            response[:500],
        )
        return []

    conflicts: list[dict] = []
    for item in data:
        if isinstance(item, dict) and "fact_a_id" in item:
            conflicts.append({
                "fact_a_id": item["fact_a_id"],
                "fact_b_id": item["fact_b_id"],
                "description": item.get("description", ""),
                "severity": float(item.get("severity", 0.5)),
            })
    return conflicts


# ---------------------------------------------------------------------------
# Gap Identification
# ---------------------------------------------------------------------------


def identify_gaps(
    llm: LLMCallable,
    claim: str,
    subclaims: list[str],
    facts: list[Fact],
) -> list[Gap]:
    """Identify evidence gaps using LLM reasoning over extracted facts.

    The LLM inspects the claim, subclaims, and stance-labeled facts to
    determine what evidence is missing, conflicting, or weak.

    Args:
        llm: LLM callable.
        claim: The claim being verified.
        subclaims: List of subclaims.
        facts: All extracted facts so far.

    Returns:
        List of Gap objects with gap_type, description, and priority.
    """
    from proclaim.verification.config import get_label_config
    label_cfg = get_label_config()

    facts_str = "\n".join(
        f"  [{f.stance}] {f.text} (paper ID:{f.source_pmid})"
        for f in facts
    ) or "  (no facts extracted yet)"

    subclaims_str = "\n".join(f"  - {sc}" for sc in subclaims)

    # Count evidence stats per configured stance label
    stance_counts = {}
    for name in label_cfg.stance_names():
        stance_counts[name] = sum(1 for f in facts if f.stance == name)
    unique_sources = len(set(f.source_pmid for f in facts))
    counts_str = ", ".join(f"{name.lower()}: {cnt}" for name, cnt in stance_counts.items())

    prompt = IDENTIFY_GAPS.format(
        claim=claim,
        subclaims_str=subclaims_str,
        num_facts=len(facts),
        counts_str=counts_str,
        unique_sources=unique_sources,
        facts_str=facts_str,
    )

    response = llm(prompt)
    if not response:
        return _fallback_gaps(claim, subclaims, facts)
    return _parse_gaps_response(response, claim, subclaims, facts)


def _parse_gaps_response(
    response: str,
    claim: str,
    subclaims: list[str],
    facts: list[Fact],
) -> list[Gap]:
    """Parse LLM response into Gap objects with fallback."""
    data = _extract_json_array(response)
    if data is None:
        logger.warning(
            "Could not parse gaps JSON from LLM response (first 500 chars): %s",
            response[:500],
        )
        if _EVIDENCE_DEBUG:
            print(
                f"[identify_gaps] JSON parse failed. Raw response (first 300 chars):\n"
                f"{response[:300]}"
            )
        return _fallback_gaps(claim, subclaims, facts)

    from proclaim.verification.data_models import GapType, GapPriority
    valid_types = {gt.value for gt in GapType}
    valid_priorities = {gp.value for gp in GapPriority}

    gaps: list[Gap] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        gap_type_str = item.get("gap_type", "")
        if gap_type_str not in valid_types:
            gap_type_str = "missing_subclaim_evidence"
        priority_str = str(item.get("priority", "medium")).lower()
        if priority_str not in valid_priorities:
            priority_str = "medium"
        gaps.append(Gap(
            subclaim=item.get("subclaim", claim),
            gap_type=GapType(gap_type_str),
            description=item.get("description", "Evidence gap identified by LLM"),
            priority=GapPriority(priority_str),
        ))

    if not gaps:
        return _fallback_gaps(claim, subclaims, facts)

    priority_order = {GapPriority.HIGH: 0, GapPriority.MEDIUM: 1, GapPriority.LOW: 2}
    return sorted(gaps, key=lambda g: priority_order[g.priority])


def _fallback_gaps(
    claim: str,
    subclaims: list[str],
    facts: list[Fact],
) -> list[Gap]:
    """Minimal fallback when LLM gap identification fails."""
    from proclaim.verification.data_models import GapType, GapPriority
    return [Gap(
        subclaim=claim,
        gap_type=GapType.MISSING_SUBCLAIM,
        description="Need more evidence to reach sufficiency threshold",
        priority=GapPriority.MEDIUM,
    )]


# ---------------------------------------------------------------------------
# Gap Query Formulation
# ---------------------------------------------------------------------------


def formulate_gap_queries(
    llm: LLMCallable,
    gaps: list[Gap],
) -> list[str]:
    """Translate gap predictions into targeted PubMed search queries.

    Args:
        llm: LLM callable.
        gaps: Gap objects from the sufficiency classifier.

    Returns:
        List of PubMed query strings (3-8 words each).
    """
    if not gaps:
        return []

    gaps_str = "\n".join(
        f"  [{g.gap_type.value}] Subclaim: {g.subclaim} | Priority: {g.priority.value} | {g.description}"
        for g in gaps
    )
    prompt = FORMULATE_GAP_QUERIES.format(
        gaps_str=gaps_str,
    )

    response = llm(prompt).strip()

    text = response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [str(q) for q in queries if isinstance(q, str)]
    except json.JSONDecodeError:
        pass

    # Fallback: try line-by-line
    return [line.strip().strip('"').strip("'")
            for line in response.split("\n")
            if line.strip() and not line.strip().startswith(("#", "-", "["))]


def refine_search_query(
    llm: LLMCallable,
    claim: str,
    subclaims: list[str],
    failed_papers: list[dict],
) -> list[str]:
    """Analyze why certain papers yielded no facts and generate refined search queries.

    This function is used when extract_and_add_facts returns 0 facts for some papers.
    Instead of retrying the same papers, we analyze why they were irrelevant and
    generate more precise PubMed queries to find better papers.

    Args:
        llm: LLM callable.
        claim: The claim being verified.
        subclaims: List of subclaims.
        failed_papers: List of dicts with keys: pmid, title, abstract.

    Returns:
        List of refined PubMed query strings (3-8 words each).
    """
    if not failed_papers:
        return []

    subclaims_str = "\n".join(f"  - {sc}" for sc in subclaims)

    papers_str = "\n".join(
        f"  paper ID:{p.get('pmid', 'unknown')}\n"
        f"  Title: {p.get('title', 'N/A')}\n"
        f"  Abstract: {p.get('abstract', 'N/A')[:200]}...\n"
        for p in failed_papers[:5]  # Limit to first 5 to avoid token overflow
    )

    prompt = REFINE_SEARCH_QUERY.format(
        claim=claim,
        subclaims_str=subclaims_str,
        papers_str=papers_str,
    )

    response = llm(prompt).strip()

    # Parse JSON response (same logic as formulate_gap_queries)
    text = response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [str(q) for q in queries if isinstance(q, str)]
    except json.JSONDecodeError:
        pass

    # Fallback: try line-by-line
    return [line.strip().strip('"').strip("'")
            for line in response.split("\n")
            if line.strip() and not line.strip().startswith(("#", "-", "["))]
