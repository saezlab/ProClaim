"""
LLM-only baseline — parametric knowledge ceiling, no retrieval.

Sends only the claim to the LLM and asks it to classify from memory.
This measures how much of the answer is already encoded in pretraining,
which sets the upper bound for what retrieval can improve upon.

Cost: 1 LLM call per claim (~500–1000 tokens at temperature=0).
"""

from __future__ import annotations

import logging
import time

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import normalize_label
from baselines.shared.llm import LLMBackend
from baselines.shared.prompts import (
    LLM_ONLY_USER_TEMPLATE,
    VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL,
)
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)


class LLMOnly:
    """Classify claims using only the LLM's parametric (pretraining) knowledge."""

    name = "llm_only"

    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm
        self.tracker = CostTracker(model=llm.model)

    def verify(self, claim_id: str, claim: str, gold_label: str) -> BaselineResult:
        self.tracker.reset()
        t0 = time.monotonic()

        user_msg = LLM_ONLY_USER_TEMPLATE.format(claim=claim)

        text, in_tok, out_tok = self.llm.complete(
            system=VERIFICATION_SYSTEM_PROMPT_NO_RETRIEVAL,
            user=user_msg,
        )
        self.tracker.record("llm_call", in_tok, out_tok, time.monotonic() - t0)

        parsed = self.llm.parse_json(text)
        raw_label = parsed.get("label", "NEI")
        predicted = normalize_label(raw_label)
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning", text[:500] if text else "")

        summary = self.tracker.summary()

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=confidence,
            reasoning=reasoning,
            input_tokens=summary["input_tokens"],
            output_tokens=summary["output_tokens"],
            cost_usd=summary["cost_usd"],
            latency_seconds=summary["latency_seconds"],
            baseline_name=self.name,
            model=self.llm.model,
        )
