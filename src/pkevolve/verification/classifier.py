"""
Sufficiency Classifier φ — learned stopping criterion.

Backbone: heuristic rule-based classifier.
Same interface as the future trained MLP — the control loop calls
  classifier(state) -> dict
regardless of whether it's a heuristic or a neural network.

Extension points:
  - SufficiencyClassifier(nn.Module) with 16-feature input, 3 output heads
  - Signal extraction (verbalized confidence, self-consistency, entropy)
  - Training data generation (k ∈ {1,3,5,10,15,20})
  - Calibration analysis (ECE)
  - Gap type prediction (8-class multi-label)
"""

import logging
from typing import Dict, List, Any

from pkevolve.verification.evidence_state import EvidenceState

logger = logging.getLogger(__name__)


class SufficiencyClassifier:
    """
    Heuristic classifier (backbone).

    Drop-in replacement for the trained MLP — same interface.
    Returns dict with keys: label, confidence, gaps.
    """

    def __init__(self, min_facts: int = 3, majority_threshold: float = 0.7):
        """
        Args:
            min_facts: Minimum number of facts required for sufficiency.
            majority_threshold: Fraction of facts in one stance needed
                                for a SUFFICIENT verdict.
        """
        self.min_facts = min_facts
        self.majority_threshold = majority_threshold

    def __call__(self, state: EvidenceState) -> Dict[str, Any]:
        """
        Classify whether the evidence state is sufficient.

        Returns:
            dict with keys:
              - label: SUFFICIENT_SUPPORT | SUFFICIENT_REFUTE | INSUFFICIENT
              - confidence: float in [0, 1]
              - gaps: list of gap descriptions (empty if sufficient)
        """
        n_facts = len(state.facts)
        support = sum(1 for f in state.facts if f.stance == "SUPPORT")
        refute = sum(1 for f in state.facts if f.stance == "REFUTE")
        total = max(n_facts, 1)

        support_frac = support / total
        refute_frac = refute / total

        if n_facts >= self.min_facts and support_frac >= self.majority_threshold:
            result = {
                "label": "SUFFICIENT_SUPPORT",
                "confidence": support_frac,
                "gaps": [],
            }
        elif n_facts >= self.min_facts and refute_frac >= self.majority_threshold:
            result = {
                "label": "SUFFICIENT_REFUTE",
                "confidence": refute_frac,
                "gaps": [],
            }
        else:
            gaps: List[str] = []
            if n_facts < self.min_facts:
                gaps.append("need_more_evidence")
            if n_facts >= self.min_facts and max(support_frac, refute_frac) < self.majority_threshold:
                gaps.append("conflicting_evidence")
            result = {
                "label": "INSUFFICIENT",
                "confidence": max(support_frac, refute_frac),
                "gaps": gaps if gaps else ["need_more_evidence"],
            }

        logger.info(
            "classifier: facts=%d, support=%d, refute=%d → %s (conf=%.2f)",
            n_facts, support, refute, result["label"], result["confidence"],
        )
        return result
