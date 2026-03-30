"""
Shared prompt templates used by all baselines that make a final verdict call.

Using the same prompt across baselines is critical for fair comparison — the
only variable should be what evidence is provided, not how the LLM is asked
to render a verdict.
"""

VERIFICATION_SYSTEM_PROMPT = """You are a scientific claim verification expert.

Given a claim and retrieved evidence, determine whether the claim is:
- SUPPORT: The evidence supports the claim
- REFUTE: The evidence contradicts the claim
- NEI: There is not enough information to determine

Respond in JSON format:
{
    "label": "SUPPORT" | "REFUTE" | "NEI",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "evidence": ["PMID1", "PMID2", ...]
}"""

VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL = """You are a scientific claim verification expert.

Given a claim, determine whether it is:
- SUPPORT: Your knowledge supports the claim
- REFUTE: Your knowledge contradicts the claim
- NEI: You do not have enough information to determine

Respond in JSON format:
{
    "label": "SUPPORT" | "REFUTE" | "NEI",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}"""

VERIFICATION_USER_TEMPLATE = """Claim: {claim}

Retrieved Evidence:
{evidence}

Based on the above evidence, classify the claim."""

LLM_ONLY_USER_TEMPLATE = """Claim: {claim}

Based on your scientific knowledge, classify the claim."""

DECOMPOSITION_PROMPT = """Decompose the following scientific claim into
independently verifiable atomic facts. Each fact should be a single
statement that can be checked against scientific literature.

Output as a JSON list of strings.

Claim: {claim}"""

QUERY_GENERATION_PROMPT = """Generate a PubMed search query to find
evidence about the following. Output only the query string.

Topic: {topic}"""
