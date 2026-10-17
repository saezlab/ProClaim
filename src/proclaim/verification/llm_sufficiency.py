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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a scientific evaluator. Given a CLAIM and a REFERENCE (paper titles, \
abstracts, and NLP features), assess whether the REFERENCE contains sufficient \
evidence to support or refute the CLAIM.

Assign a sufficiency score between 0.0 and 1.0:
- 1.0: The REFERENCE contains decisive evidence — either clearly supporting OR \
clearly refuting the CLAIM. Both directions are equally "sufficient". Papers that \
unanimously refute the claim deserve the same high score as papers that unanimously \
support it.
- 0.0: The REFERENCE is entirely irrelevant (entities, topic, and semantics are \
unrelated to the claim), or no papers were found.
- Values in between reflect partial relevance, inconclusive evidence, or genuinely \
conflicting evidence. Conflicting evidence means the papers are relevant and credible, \
but split in direction — after weighing their content, methodology, and credibility, \
no clear verdict can be reached. Do NOT judge conflict by the ratio of papers alone; \
a single high-quality paper can outweigh several weaker ones.

### FEATURE DEFINITIONS
Each paper in the REFERENCE includes pre-computed Metadata and NLP fields:

Metadata:
- publication_year: Year the paper was published.
- log_impact_factor: log(1 + journal impact factor). Higher = more prestigious venue.
- normalized_citation_count: Citations / paper age. Measures community attention.
- author_h_index_max: Highest H-index among all authors. Proxy for author credibility.

NLP (computed relative to the CLAIM):
- claim_entity_coverage: Fraction of claim biomedical entities also in the paper (0–1).
- semantic_similarity: SBERT cosine similarity between claim and abstract (0–1).
- nli_entailment: DeBERTa NLI probability that the nli_best_chunk_text ENTAILS the claim (0–1).
- nli_contradiction: Probability the nli_best_chunk_text CONTRADICTS (refutes) the claim (0–1).
- nli_neutral: Probability the nli_best_chunk_text is NEUTRAL (0–1). Sum ≈ 1.
- nli_best_chunk_text: The most relevant text passage used to compute NLI scores.

### CLAIM
{claim}

### REFERENCE
{reference}

First, output a brief ### EXPLANATION (plain text reasoning).
Then output ### EVALUATION as a JSON object with exactly these keys:

```json
{
  "sufficiency_score": <float 0.0–1.0>,
  "reasoning": "<one-sentence summary>"
}
```
Output ONLY the EXPLANATION block followed by the EVALUATION JSON block. No other text.\
"""

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
