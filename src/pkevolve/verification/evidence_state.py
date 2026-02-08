"""
EvidenceState — central data structure for the verification loop.

Backbone: plain container, no locking, char-based token approximation.
Extension points:
  - Thread safety (add RLock on mutations)
  - Conflict tracking (add_conflict, Conflict dataclass)
  - Coverage/synthesis tracking
  - Budget-aware get_context(budget_tokens)
  - tiktoken-based token_count()
  - Audit log
"""

import copy
from typing import Dict, List, Optional

from pkevolve.verification.data_models import Fact, PaperRecord


class EvidenceState:
    """
    Container for all evidence gathered during a verification loop.

    Backbone: sequential access only — no locking.
    The interface is designed so locking can be added later without
    changing callers.
    """

    def __init__(self, claim: str, subclaims: Optional[List[str]] = None):
        """
        Initialize with a claim and optional subclaims.

        Args:
            claim: The claim to be verified.
            subclaims: Sub-claims to verify individually.
                       Defaults to [claim] (no decomposition).
        """
        self.claim: str = claim
        self.subclaims: List[str] = subclaims if subclaims is not None else [claim]
        self.papers: Dict[str, PaperRecord] = {}
        self.facts: List[Fact] = []

    # -- Mutation methods --------------------------------------------------

    def add_paper(self, paper: PaperRecord) -> None:
        """Add a paper keyed by PMID. Overwrites if PMID already present."""
        self.papers[paper.pmid] = paper

    def add_fact(self, fact: Fact) -> None:
        """Append a fact to the evidence."""
        self.facts.append(fact)

    # -- Query methods -----------------------------------------------------

    def clone(self) -> "EvidenceState":
        """Deep copy — produces an independent copy of the entire state."""
        return copy.deepcopy(self)

    def token_count(self) -> int:
        """
        Approximate token count (char-based: ~4 chars per token).

        Backbone uses character length / 4 as a rough proxy.
        Extension: replace with tiktoken encoding.
        """
        total_chars = 0
        for paper in self.papers.values():
            if paper.summary:
                total_chars += len(paper.summary)
            elif paper.abstract:
                total_chars += len(paper.abstract)
        for fact in self.facts:
            total_chars += len(fact.text)
        # Rough approximation: 1 token ≈ 4 characters
        return total_chars // 4

    def get_context(self) -> str:
        """
        Concatenate all summaries and facts into a prompt-ready string.

        Backbone: returns everything — no budget filtering.
        Extension: add budget_tokens parameter, prioritize L2 > L1 > L0.
        """
        sections: List[str] = []

        # Paper summaries (L1) or abstracts (L0)
        if self.papers:
            sections.append("=== Evidence from Papers ===")
            for pmid, paper in self.papers.items():
                text = paper.summary if paper.summary else paper.abstract
                sections.append(f"[PMID:{pmid}] {paper.title}\n{text}")

        # Extracted facts
        if self.facts:
            sections.append("\n=== Extracted Facts ===")
            for fact in self.facts:
                sections.append(
                    f"[{fact.stance}] {fact.text} (PMID:{fact.source_pmid})"
                )

        return "\n\n".join(sections)

    def __repr__(self) -> str:
        return (
            f"EvidenceState(claim={self.claim!r}, "
            f"papers={len(self.papers)}, facts={len(self.facts)})"
        )
