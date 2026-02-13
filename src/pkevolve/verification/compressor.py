"""
Sufficiency-Preserving Compression.

L1 deduplication only (lossless). Removes duplicate facts with identical
(text, stance) pairs. Returns a new EvidenceState — original is not mutated.
"""

import logging

from pkevolve.verification.evidence_state import EvidenceState

logger = logging.getLogger(__name__)


class SufficiencyPreservingCompressor:
    """L1 deduplication compressor."""

    def compress(self, state: EvidenceState, claim: str) -> EvidenceState:
        """
        Remove duplicate facts (exact text + stance match).

        Args:
            state: Current evidence state.
            claim: The claim being verified.

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
