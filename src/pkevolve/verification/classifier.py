"""
Sufficiency Classifier -- heuristic implementation.

Returns SufficiencyResult (Pydantic model) for interface compatibility
with the future trained MLP classifier.

Backbone: heuristic rule-based classifier.
Same interface as the future trained MLP — the control loop calls
  classifier(state) -> SufficiencyResult
regardless of whether it's a heuristic or a neural network.

Extension points:
  - SufficiencyClassifier(nn.Module) with 16-feature input, 3 output heads
  - Signal extraction (verbalized confidence, self-consistency, entropy)
  - Training data generation (k in {1,3,5,10,15,20})
  - Calibration analysis (ECE)
  - Gap type prediction (8-class multi-label)
"""

import logging

from pkevolve.verification.data_models import (
    Gap,
    GapType,
    Stance,
    SufficiencyResult,
)
from pkevolve.verification.evidence_state import EvidenceState

logger = logging.getLogger(__name__)


class SufficiencyClassifier:
    """
    Heuristic classifier returning Pydantic SufficiencyResult.

    Drop-in replacement interface:
        classifier(state: EvidenceState) -> SufficiencyResult

    The future MLP classifier will implement the same __call__ signature.
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

    def __call__(self, state: EvidenceState) -> SufficiencyResult:
        """
        Classify whether the evidence state is sufficient.

        Returns:
            SufficiencyResult with label, confidence, and gaps.
        """
        n_facts = len(state.facts)
        support = sum(1 for f in state.facts if f.stance == Stance.SUPPORT)
        refute = sum(1 for f in state.facts if f.stance == Stance.REFUTE)
        total = max(n_facts, 1)

        support_frac = support / total
        refute_frac = refute / total

        if n_facts >= self.min_facts and support_frac >= self.majority_threshold:
            result = SufficiencyResult(
                label="SUFFICIENT_SUPPORT",
                confidence=support_frac,
                gaps=[],
            )
        elif n_facts >= self.min_facts and refute_frac >= self.majority_threshold:
            result = SufficiencyResult(
                label="SUFFICIENT_REFUTE",
                confidence=refute_frac,
                gaps=[],
            )
        else:
            gaps = self._identify_gaps(state, n_facts, support_frac, refute_frac)
            result = SufficiencyResult(
                label="INSUFFICIENT",
                confidence=max(support_frac, refute_frac),
                gaps=gaps,
            )

        logger.info(
            "classifier: facts=%d, support=%d, refute=%d -> %s (conf=%.2f)",
            n_facts, support, refute, result.label, result.confidence,
        )
        return result

    def _identify_gaps(
        self,
        state: EvidenceState,
        n_facts: int,
        support_frac: float,
        refute_frac: float,
    ) -> list[Gap]:
        """Heuristic gap identification using GapType enum."""
        gaps: list[Gap] = []

        if n_facts < self.min_facts:
            for sc in state.subclaims:
                relevant = [
                    f for f in state.facts if sc in f.relevant_subclaims
                ]
                if len(relevant) < 2:
                    gaps.append(Gap(
                        subclaim=sc,
                        gap_type=GapType.MISSING_SUBCLAIM,
                        description=f"Only {len(relevant)} facts found for: {sc}",
                        priority=0.9,
                    ))

        if n_facts >= self.min_facts and max(support_frac, refute_frac) < self.majority_threshold:
            gaps.append(Gap(
                subclaim=state.claim,
                gap_type=GapType.CONTRADICTORY,
                description="Conflicting evidence: no clear majority stance",
                priority=0.8,
            ))

        # Check source diversity
        unique_sources = len(set(f.source_pmid for f in state.facts)) if state.facts else 0
        if unique_sources < max(len(state.subclaims), 2):
            gaps.append(Gap(
                subclaim=state.claim,
                gap_type=GapType.LOW_DIVERSITY,
                description=f"Only {unique_sources} unique sources",
                priority=0.5,
            ))

        # Fallback
        if not gaps:
            gaps.append(Gap(
                subclaim=state.claim,
                gap_type=GapType.MISSING_SUBCLAIM,
                description="Need more evidence to reach sufficiency threshold",
                priority=0.7,
            ))

        return sorted(gaps, key=lambda g: g.priority, reverse=True)
