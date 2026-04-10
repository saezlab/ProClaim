"""
Shared prompt templates used by all baselines that make a final verdict call.

Using the same prompt across baselines is critical for fair comparison — the
only variable should be what evidence is provided, not how the LLM is asked
to render a verdict.

Label definitions are read from ``LabelConfig`` so that changing the verdict
taxonomy (names, descriptions, number of labels) in one place propagates to
every baseline prompt automatically.
"""

from __future__ import annotations

from baselines.shared.label_utils import get_label_config

from pkevolve.verification.config import LabelConfig

# ── Default label config (re-use shared singleton) ────────────────────

_DEFAULT_LABELS = get_label_config()


# ── Builder functions (use these when you need a custom LabelConfig) ──


def build_verification_system_prompt(labels: LabelConfig | None = None) -> str:
    """System prompt for retrieval-augmented claim verification."""
    lc = labels or _DEFAULT_LABELS
    verdict_block = "\n".join(
        f"- {name} — {desc}" for name, desc in lc.verdict_labels.items()
    )
    label_options = " | ".join(f'"{n}"' for n in lc.verdict_names())
    return f"""You are a scientific claim verification expert.

Given a claim and retrieved evidence passages, assign one of the following verdicts:

{verdict_block}

Respond with valid JSON only — no markdown fences, no extra keys:
{{
    "label": {label_options},
    "reasoning": "One or two sentences citing specific evidence",
    "evidence": ["PMID1", "PMID2"]
}}"""


def build_verification_system_prompt_no_retrieval(labels: LabelConfig | None = None) -> str:
    """System prompt for parametric-knowledge-only claim verification."""
    lc = labels or _DEFAULT_LABELS
    # Build verdict definitions reframed for no-retrieval context
    no_retrieval_defs = {
        name: _no_retrieval_description(name, desc)
        for name, desc in lc.verdict_labels.items()
    }
    verdict_block = "\n".join(
        f"- {name} — {desc}" for name, desc in no_retrieval_defs.items()
    )
    label_options = " | ".join(f'"{n}"' for n in lc.verdict_names())
    return f"""You are a scientific claim verification expert.

Given a claim, assess whether it is supported by established scientific knowledge:

{verdict_block}

Respond with valid JSON only — no markdown fences, no extra keys:
{{
    "label": {label_options},
    "reasoning": "One or two sentences explaining your verdict"
}}"""


def _no_retrieval_description(name: str, retrieval_desc: str) -> str:
    """Reframe a retrieval-oriented verdict description for the no-retrieval setting."""
    upper = name.upper()
    if upper == "SUPPORT":
        return (
            "Your knowledge of the scientific literature supports the claim "
            "as true or highly likely true based on established experimental evidence."
        )
    elif upper == "REFUTE":
        return (
            "Your knowledge of the scientific literature contradicts the claim, "
            "or the claim is not substantiated by any known experimental evidence."
        )
    elif upper == "UNCERTAIN":
        return (
            "The scientific literature is ambiguous, incomplete, or conflicting "
            "on this claim, or your knowledge is insufficient to make a reliable determination."
        )
    # For any custom label, fall back to the retrieval description as-is
    return retrieval_desc


# ── Module-level constants (backward compatible) ──────────────────────

VERIFICATION_SYSTEM_PROMPT = build_verification_system_prompt()
VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL = build_verification_system_prompt_no_retrieval()

VERIFICATION_USER_TEMPLATE = """Claim: {claim}

Retrieved Evidence:
{evidence}

Classify the claim based solely on the evidence passages above. Output JSON."""

LLM_ONLY_USER_TEMPLATE = """Claim: {claim}

Classify this claim based on your scientific knowledge. Output JSON."""
