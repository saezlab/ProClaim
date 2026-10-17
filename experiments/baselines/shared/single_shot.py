"""Shared helpers for single-shot baseline verdict calls."""

from __future__ import annotations

from dataclasses import dataclass

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import normalize_label
from baselines.shared.llm import LLMBackend


@dataclass
class SingleShotVerdict:
    text: str
    predicted_label: str
    confidence: float
    reasoning: str
    evidence: list[str]
    summary: dict


def run_single_shot_verdict(
    *,
    llm: LLMBackend,
    tracker: CostTracker,
    system_prompt: str,
    user_prompt: str,
    response_format: str = "json_object",
) -> SingleShotVerdict:
    """Execute a single LLM verdict call, track usage, and parse the result."""
    if response_format == "text":
        text, input_tokens, output_tokens = llm.complete_text(system=system_prompt, user=user_prompt)
    else:
        text, input_tokens, output_tokens = llm.complete(
            system=system_prompt,
            user=user_prompt,
            response_format=response_format,
        )
    tracker.record("llm_call", input_tokens, output_tokens)

    parsed = llm.parse_json(text)
    raw_label = parsed.get("label", "UNCERTAIN")
    predicted = normalize_label(raw_label)
    confidence = float(parsed.get("confidence", 0.0))
    reasoning = parsed.get("reasoning", text[:500] if text else "")
    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]

    return SingleShotVerdict(
        text=text,
        predicted_label=predicted,
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence,
        summary=tracker.summary(),
    )