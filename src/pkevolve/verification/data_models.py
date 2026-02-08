"""
Core data models for the Evidence Verification system.

PaperRecord and Fact are the atomic data units stored in EvidenceState.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PaperRecord:
    """A single paper retrieved from PubMed or other sources."""

    pmid: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    summary: Optional[str] = None  # L1 — set by summarize_paper tool

    def text_for_summarization(self) -> str:
        """Return the best available text for LLM summarization."""
        if self.full_text:
            return self.full_text
        return self.abstract


@dataclass
class Fact:
    """A stance-labeled fact extracted from a paper."""

    text: str
    stance: str  # SUPPORT | REFUTE | NEUTRAL
    source_pmid: str
    relevant_subclaims: List[str] = field(default_factory=list)
