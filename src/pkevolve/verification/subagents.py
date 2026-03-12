"""
Subagent functions for evidence programming.

Converts the Claude Agent SDK subagent definitions (.claude/agents/*.md) into
callable Python functions. Each function takes an ``llm`` callable and data,
constructs a prompt, calls ``llm(prompt)``, and parses the structured response.

The ``llm`` callable is injected by the orchestrator:
  - In REPL mode: a closure over openai.OpenAI.chat.completions.create
  - In SDK mode: the recursive llm() primitive

Usage::

    from pkevolve.verification.subagents import extract_facts

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
import re
from typing import Callable, Optional

from pkevolve.verification.data_models import (
    Conflict,
    Fact,
    Gap,
    Stance,
)

logger = logging.getLogger(__name__)

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
) -> list[Fact]:
    """Extract stance-labeled facts from a paper.

    Args:
        llm: Callable that takes a prompt string and returns a response string.
        paper_text: Full text or abstract of the paper.
        claim: The claim being verified.
        subclaims: List of subclaims the facts should be mapped to.
        source_pmid: PMID of the source paper.

    Returns:
        List of Fact objects with stance labels and subclaim mappings.
    """
    subclaims_str = "\n".join(f"  - {sc}" for sc in subclaims)
    prompt = f"""\
You are a scientific fact extraction specialist.

Given a paper and a claim with subclaims, extract every atomic fact relevant to the claim.

For each fact provide a JSON object with:
- "text": factual statement (one sentence, self-contained)
- "stance": "SUPPORT" if it supports the claim, "REFUTE" if it contradicts, "NEUTRAL" if relevant but neither
- "source_pmid": "{source_pmid}"
- "relevant_subclaims": list of subclaim strings this fact addresses
- "confidence": 0.0-1.0, how clearly the paper states this

Rules:
- Be precise. Do not infer beyond what the paper states.
- If a paper does not address a subclaim, do not manufacture facts.
- Each fact must be independently verifiable from the source paper.

Claim: {claim}

Subclaims:
{subclaims_str}

Paper (PMID: {source_pmid}):
{paper_text[:16000]}

Output ONLY a JSON array of fact objects. No other text."""

    response = llm(prompt)
    if not response:
        print(f"extract_facts: LLM returned empty response for PMID {source_pmid}")
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

    facts: list[Fact] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        stance_str = item.get("stance", "NEUTRAL").upper()
        if stance_str not in ("SUPPORT", "REFUTE", "NEUTRAL"):
            stance_str = "NEUTRAL"
        facts.append(Fact(
            id=f"fact_auto_{i}",
            text=item.get("text", ""),
            stance=Stance(stance_str),
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
        f"  [{f.stance.value}] {f.text} (PMID:{f.source_pmid}, conf:{f.confidence:.2f})"
        for f in facts
    )
    prompt = f"""\
You are an evidence synthesis specialist.

Given a subclaim and relevant facts, produce a synthesis that:
1. States weight of evidence (mostly supporting, mostly refuting, mixed, insufficient)
2. Summarizes key supporting facts with PMID citations
3. Summarizes contradicting facts with PMID citations
4. Notes quality and diversity of sources
5. Identifies what additional evidence would strengthen the assessment

Keep synthesis under 200 words. Be precise about what evidence does and does not show.

Subclaim: {subclaim}

Facts:
{facts_str}

Output the synthesis as plain text."""

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
        f"  [{f.id}] [{f.stance.value}] {f.text} (PMID:{f.source_pmid})"
        for f in facts
    )
    prompt = f"""\
You are a scientific conflict detection specialist.

Given a list of extracted facts, identify pairs that contradict each other.
For each conflict, provide a JSON object with:
- "fact_a_id": ID of first fact
- "fact_b_id": ID of second fact
- "description": nature of the contradiction
- "severity": 0.0-1.0 (0 = minor methodological difference, 1 = direct contradiction)

Facts:
{facts_str}

Output ONLY a JSON array of conflict objects. If no conflicts, output []."""

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
    facts_str = "\n".join(
        f"  [{f.stance.value}] {f.text} (PMID:{f.source_pmid})"
        for f in facts
    ) or "  (no facts extracted yet)"

    subclaims_str = "\n".join(f"  - {sc}" for sc in subclaims)

    # Count basic evidence stats for context
    n_support = sum(1 for f in facts if f.stance == Stance.SUPPORT)
    n_refute = sum(1 for f in facts if f.stance == Stance.REFUTE)
    n_neutral = sum(1 for f in facts if f.stance == Stance.NEUTRAL)
    unique_sources = len(set(f.source_pmid for f in facts))

    prompt = f"""\
You are an evidence gap analyst for scientific claim verification.

The current evidence has been judged INSUFFICIENT. Your task: inspect the
extracted facts and identify specific gaps — what is missing, conflicting,
or weak — so the system can search for additional evidence.

Claim: {claim}

Subclaims:
{subclaims_str}

Extracted facts ({len(facts)} total — {n_support} support, {n_refute} refute, {n_neutral} neutral, from {unique_sources} unique sources):
{facts_str}

Gap types to choose from:
- missing_subclaim_evidence: a subclaim lacks supporting facts
- contradictory_evidence: conflicting evidence needs resolution
- low_source_diversity: too few independent sources
- weak_stance_evidence: evidence exists but is weak/indirect
- missing_mechanism: mechanistic explanation is missing
- missing_quantitative: quantitative data (dose-response, effect sizes) is missing
- missing_temporal: temporal/longitudinal data is missing
- missing_population: population-specific evidence is missing

Output ONLY a JSON array of gap objects. Each gap:
{{
  "subclaim": "the relevant subclaim text",
  "gap_type": "one of the gap types above",
  "description": "specific description of what is missing",
  "priority": "high" | "medium" | "low"
}}

Identify 1-4 gaps, ordered from highest to lowest priority. Output [] if no specific gaps."""

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
        print(
            f"[identify_gaps] JSON parse failed. Raw response (first 300 chars):\n"
            f"{response[:300]}"
        )
        return _fallback_gaps(claim, subclaims, facts)

    from pkevolve.verification.data_models import GapType, GapPriority
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
    from pkevolve.verification.data_models import GapType, GapPriority
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
    prompt = f"""\
You are an evidence retrieval query specialist.

Given gap predictions from the sufficiency classifier, formulate targeted PubMed
search queries (3-8 words each) to close each gap. Do NOT re-analyze the evidence.

Gap types and query strategies:
- missing_subclaim_evidence: search directly for the subclaim topic
- contradictory_evidence: search for meta-analyses or reviews
- low_source_diversity: use different terminology or adjacent fields
- weak_stance_evidence: search for RCTs, large cohorts
- missing_mechanism: search for mechanistic or pathway studies
- missing_quantitative: search for dose-response, effect size studies
- missing_temporal: search for longitudinal or time-course studies
- missing_population: search for studies in the specific population

Gaps:
{gaps_str}

Output ONLY a JSON array of query strings. Example: ["MAPK1 phosphorylation mechanism", "H3 histone modification review"]"""

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
