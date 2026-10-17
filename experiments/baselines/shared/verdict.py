"""
Shared result schemas for evidence programming baselines.

Every baseline returns a ``BaselineResult`` per claim.  The evaluation
harness aggregates ``BaselineResult`` objects into metrics.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


def extract_json(text: str) -> dict:
    """Pull the first ``{...}`` object out of free-form agent text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"label": None, "reasoning": ""}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"label": None, "reasoning": text[:500]}


class Verdict(BaseModel):
    """Minimal structured verdict emitted by any baseline."""

    label: str  # SUPPORT | REFUTE | UNCERTAIN (canonical)
    confidence: float = 0.0
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)  # PMIDs or snippets


class BaselineResult(BaseModel):
    """One claim evaluated by one baseline for use in the evaluation harness."""

    claim_id: str
    claim: str
    gold_label: str  # canonical (SUPPORT | REFUTE | UNCERTAIN)
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

    # Non-empty => this claim produced no verdict (resource cap hit or crash).
    # metrics() excludes these rather than scoring them as UNCERTAIN.
    error: str = ""

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.gold_label
