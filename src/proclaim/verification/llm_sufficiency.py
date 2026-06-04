"""LLM-based sufficiency classifier for ProClaim.

Provides a drop-in alternative to the MLP classifier that uses a Qwen subagent
(via the existing ``llm`` callable) to assess whether retrieved papers provide
sufficient evidence to support or refute a claim.

Public API
----------
check_sufficiency_llm(state, llm, threshold, fallback_score, max_parse_retries)
    → (label, score, raw_response)
"""

from __future__ import annotations

import json
import logging
import re

from proclaim.verification.evidence_state import EvidenceState
from proclaim.verification.prompts import LLM_SUFFICIENCY_PROMPT as PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference builder
# ---------------------------------------------------------------------------


def build_reference_text(state: EvidenceState) -> str:
    """Build the REFERENCE block from EvidenceState.papers."""
    if not state.papers:
        return "No papers were found."

    parts: list[str] = []
    for pmid, paper in state.papers.items():
        meta: dict = paper.metadata.model_dump() if paper.metadata else {}
        nlp_raw: dict = paper.nlp.model_dump() if paper.nlp else {}
        nlp_filtered = {
            k: v for k, v in nlp_raw.items()
            if k not in ("claim_entities", "evidence_entities")
        }
        block = (
            f"Paper {pmid}:\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.abstract}\n"
            f"Metadata: {json.dumps(meta)}\n"
            f"NLP: {json.dumps(nlp_filtered)}"
        )
        parts.append(block)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Score parser
# ---------------------------------------------------------------------------


def parse_score(response_text: str) -> float | None:
    """Extract sufficiency_score from the LLM's JSON EVALUATION block."""
    obj_start = response_text.find("{")
    obj_end = response_text.rfind("}")
    if obj_start != -1 and obj_end != -1:
        try:
            obj = json.loads(response_text[obj_start:obj_end + 1])
            if "sufficiency_score" in obj:
                val = float(obj["sufficiency_score"])
                return max(0.0, min(1.0, val))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    for pattern in (
        r'"sufficiency_score"\s*:\s*([0-9]*\.?[0-9]+)',
        r'"Sufficient Support Score"\s*:\s*([0-9]*\.?[0-9]+)',
        r'"score"\s*:\s*([0-9]*\.?[0-9]+)',
    ):
        matches = re.findall(pattern, response_text)
        if matches:
            val = float(matches[-1])
            return max(0.0, min(1.0, val))

    return None


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def check_sufficiency_llm(
    state: EvidenceState,
    llm,
    threshold: float = 0.5,
    fallback_score: float = 0.0,
    max_parse_retries: int = 2,
) -> tuple[str, float, str]:
    """Run LLM-based sufficiency check.

    Returns:
        (label, score, raw_response)
        where label is 'sufficient' | 'insufficient'
    """
    reference_text = build_reference_text(state)
    prompt = (
        PROMPT_TEMPLATE
        .replace("{claim}", state.claim)
        .replace("{reference}", reference_text)
    )

    raw_response = llm(prompt)
    score = parse_score(raw_response)

    for attempt in range(max_parse_retries):
        if score is not None:
            break
        logger.warning(
            "LLM sufficiency: parse failed (attempt %d/%d), retrying",
            attempt + 1, max_parse_retries,
        )
        raw_response = llm(prompt)
        score = parse_score(raw_response)

    if score is None:
        logger.error(
            "LLM sufficiency: all retries exhausted. Using fallback score=%.2f. "
            "Raw response (first 300 chars): %s",
            fallback_score, raw_response[:300],
        )
        score = fallback_score

    label = "sufficient" if score >= threshold else "insufficient"
    return label, score, raw_response
