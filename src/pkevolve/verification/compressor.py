"""
Sufficiency-Preserving Compression — primary technical novelty.

Backbone: L1 deduplication only (lossless).
Same interface as the full compressor — controller doesn't know the difference.

Extension points:
  - L2: Synthesis refresh (LLM-based, lossy, coverage-gated)
  - L3: Aggressive synthesis (LLM-based, lossy, guarded)
  - _check_invariant(): classifier comparison |φ(S') - φ(S)| ≤ ε
  - async compress() for L2/L3 LLM calls
  - Invariant violation rate tracking
  - ε sensitivity sweep
"""

import logging
from typing import Optional

from pkevolve.verification.evidence_state import EvidenceState

logger = logging.getLogger(__name__)


class SufficiencyPreservingCompressor:
    """
    Backbone: L1 deduplication only.

    Same interface as the full compressor — controller doesn't know the
    difference. The constructor accepts llm_client and classifier for
    interface compatibility with the full version— they are unused in
    the backbone.
    """

    def __init__(
        self,
        llm_client=None,
        classifier=None,
        epsilon: float = 0.05,
    ):
        """
        Args:
            llm_client: Unused in backbone. For L2/L3 LLM calls in extension.
            classifier: Unused in backbone. For invariant checking in extension.
            epsilon: Unused in backbone. Tolerance for sufficiency invariant.
        """
        self.llm_client = llm_client
        self.classifier = classifier
        self.epsilon = epsilon

    def compress(self, state: EvidenceState, claim: str) -> EvidenceState:
        """
        L1: Remove duplicate facts (exact text + stance match).

        Args:
            state: Current evidence state.
            claim: The claim being verified (unused in backbone;
                   needed by L2/L3 for synthesis prompts).

        Returns:
            A new EvidenceState with duplicates removed.
            Original state is NOT mutated.
        """
        compressed = state.clone()

        seen: set = set()
        unique_facts = []
        for fact in compressed.facts:
            key = (fact.text.strip().lower(), fact.stance)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)

        removed = len(compressed.facts) - len(unique_facts)
        compressed.facts = unique_facts

        logger.info(
            "compress L1: %d facts → %d facts (%d duplicates removed)",
            len(state.facts), len(unique_facts), removed,
        )
        return compressed
