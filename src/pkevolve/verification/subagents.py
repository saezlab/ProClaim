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
    from pkevolve.verification.config import get_label_config
    label_cfg = get_label_config()

    subclaims_str = "\n".join(f"  - {sc}" for sc in subclaims)
    stance_block = label_cfg.stance_prompt_block()
    stance_options = label_cfg.stance_options_str()
    prompt = f"""\
You are a scientific fact extraction specialist.

Given a paper and a claim with subclaims, extract every atomic fact that addresses the claim, accounting for varying terminology, synonyms, or aliases used in the text.

For each fact provide a JSON object with:
- "text": factual statement (one sentence, self-contained)
- "stance": one of {stance_options}
- "source_pmid": "{source_pmid}"
- "relevant_subclaims": list of subclaim strings this fact addresses
- "confidence": 0.0-1.0, how clearly the paper states this

Stance definitions:
{stance_block}

Rules:
- Base facts strictly on the provided text. Recognize equivalent terms, but do not hallucinate logical leaps not present in the paper.
- If a paper does not address a subclaim, do not manufacture facts.
- Each fact must be independently verifiable from the source paper.
Now extract facts for the following:

Claim: {claim}

Subclaims:
{subclaims_str}

Paper (PMID: {source_pmid}):
{paper_text[:50000]}

Output ONLY a JSON array of fact objects. No other text."""

    response = llm(prompt)
    if not response or response.strip() == "[]":
        logger.debug("extract_facts: No facts found for PMID %s", source_pmid)
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

    from pkevolve.verification.config import get_label_config
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
        f"  [{f.stance}] {f.text} (PMID:{f.source_pmid}, conf:{f.confidence:.2f})"
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
        f"  [{f.id}] [{f.stance}] {f.text} (PMID:{f.source_pmid})"
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
    from pkevolve.verification.config import get_label_config
    label_cfg = get_label_config()

    facts_str = "\n".join(
        f"  [{f.stance}] {f.text} (PMID:{f.source_pmid})"
        for f in facts
    ) or "  (no facts extracted yet)"

    subclaims_str = "\n".join(f"  - {sc}" for sc in subclaims)

    # Count evidence stats per configured stance label
    stance_counts = {}
    for name in label_cfg.stance_names():
        stance_counts[name] = sum(1 for f in facts if f.stance == name)
    unique_sources = len(set(f.source_pmid for f in facts))
    counts_str = ", ".join(f"{name.lower()}: {cnt}" for name, cnt in stance_counts.items())

    prompt = f"""\
You are an evidence gap analyst for scientific claim verification.

The current evidence has been judged INSUFFICIENT. Your task: inspect the
extracted facts and identify specific gaps — what is missing, conflicting,
or weak — so the system can search for additional evidence.

Claim: {claim}

Subclaims:
{subclaims_str}

Extracted facts ({len(facts)} total — {counts_str}, from {unique_sources} unique sources):
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
        f"  PMID:{p.get('pmid', 'unknown')}\n"
        f"  Title: {p.get('title', 'N/A')}\n"
        f"  Abstract: {p.get('abstract', 'N/A')[:200]}...\n"
        for p in failed_papers[:5]  # Limit to first 5 to avoid token overflow
    )

    prompt = f"""\
You are an evidence retrieval specialist for scientific claim verification.

The system attempted to extract facts from the following papers but found NO relevant evidence.
Your task: analyze why these papers are irrelevant to the claim, then generate MORE PRECISE
PubMed search queries to find papers that actually contain relevant evidence.

Claim: {claim}

Subclaims:
{subclaims_str}

Papers that yielded 0 facts:
{papers_str}

Common reasons for irrelevance:
- Papers discuss similar topics but in different contexts or domains
- Papers mention the subject matter only indirectly or as background
- Papers study related but distinct phenomena or mechanisms
- Initial query was too broad and retrieved tangentially related papers
- Papers lack the specific type of evidence needed for verification

Your task:
1. Analyze why the above papers were irrelevant to the claim
2. Generate 5-10 MORE SPECIFIC PubMed queries that:
   - Keep queries SHORT (2-4 key terms max) to maximize recall
   - Add ONE contextual constraint that distinguishes relevant papers
   - Use OR operators for synonyms rather than long AND chains
   - Avoid overly specific methodology terms (e.g., "co-immunoprecipitation")
   - Use proper PubMed Boolean operators sparingly

CRITICAL: PubMed interprets space-separated terms as AND. Keep queries SHORT.
Too specific = 0 results. Balance precision with recall.

Output ONLY a JSON array of query strings (2-4 words each).
Example for a PPI claim: ["MAPK1 H3F3A phosphorylation", "ERK2 histone H3.3"]
Example for a clinical claim: ["JAK2 V617F lymphoid leukemia", "JAK2 mutation B-ALL"]"""

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
