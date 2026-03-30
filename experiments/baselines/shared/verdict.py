"""
Shared result schemas for evidence programming baselines.

Every baseline returns a ``BaselineResult`` per claim.  The evaluation
harness aggregates ``BaselineResult`` objects into metrics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """Minimal structured verdict emitted by any baseline."""

    label: str  # SUPPORT | REFUTE | NEI (canonical)
    confidence: float = 0.0
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)  # PMIDs or snippets


class BaselineResult(BaseModel):
    """One claim evaluated by one baseline for use in the evaluation harness."""

    claim_id: str
    claim: str
    gold_label: str  # canonical (SUPPORT | REFUTE | NEI)
    predicted_label: str  # canonical
    confidence: float = 0.0
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)

    # Cost / usage
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0

    # Metadata
    baseline_name: str = ""
    model: str = ""
    dataset: str = ""

    # Whether the prediction matched the gold label after normalization
    @property
    def correct(self) -> bool:
        return self.predicted_label == self.gold_label
