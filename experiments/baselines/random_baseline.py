"""
Random baseline — establishes the chance-level performance floor.

Draws uniformly at random from {SUPPORT, REFUTE, UNCERTAIN} for every claim.
Cost: 0 tokens, 0 LLM calls.
"""

from __future__ import annotations

import random

from baselines.shared.label_utils import normalize_label
from baselines.shared.verdict import BaselineResult


class RandomBaseline:
    """Uniform random label assignment.  No retrieval, no LLM calls."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def verify(self, claim_id: str, claim: str, gold_label: str) -> BaselineResult:
        label = self._rng.choice(["SUPPORT", "REFUTE", "UNCERTAIN"])
        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=label,
            confidence=1.0 / 3.0,
            reasoning="Randomly sampled from {SUPPORT, REFUTE, UNCERTAIN}.",
            baseline_name=self.name,
        )
