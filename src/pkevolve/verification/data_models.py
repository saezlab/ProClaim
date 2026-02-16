"""
Core data models for the Evidence Verification system.

Pydantic v2 models with backward-compatible field names.
PaperRecord and Fact are the atomic data units stored in EvidenceState.
Additional types: Stance, GapType, Conflict, Gap, SufficiencyResult, VerificationVerdict.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Stance(str, Enum):
    """Stance of a fact relative to the claim."""
    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    NEUTRAL = "NEUTRAL"


class GapType(str, Enum):
    """Types of evidence gaps the classifier can identify."""
    MISSING_SUBCLAIM = "missing_subclaim_evidence"
    CONTRADICTORY = "contradictory_evidence"
    LOW_DIVERSITY = "low_source_diversity"
    WEAK_STANCE = "weak_stance_evidence"
    MISSING_MECHANISM = "missing_mechanism"
    MISSING_QUANTITATIVE = "missing_quantitative"
    MISSING_TEMPORAL = "missing_temporal"
    MISSING_POPULATION = "missing_population"


class PaperRecord(BaseModel):
    """A single paper retrieved from PubMed or other sources."""

    pmid: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    summary: Optional[str] = None  # L1 — set by summarize_paper tool
    authors: list[str] = Field(default_factory=list)
    source: str = "pubmed"  # pubmed | semantic_scholar

    def text_for_summarization(self) -> str:
        """Return the best available text for LLM summarization."""
        if self.full_text:
            return self.full_text
        return self.abstract


class Fact(BaseModel):
    """A stance-labeled fact extracted from a paper."""

    id: str = ""
    text: str
    stance: Stance
    source_pmid: str
    relevant_subclaims: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class Conflict(BaseModel):
    """A detected conflict between two facts."""

    id: str
    fact_a_id: str
    fact_b_id: str
    description: str
    severity: float  # 0-1


class Gap(BaseModel):
    """An identified evidence gap for a subclaim."""

    subclaim: str
    gap_type: GapType
    description: str
    priority: float  # 0-1


class SufficiencyResult(BaseModel):
    """Result from the sufficiency classifier."""

    label: str  # SUFFICIENT_SUPPORT | SUFFICIENT_REFUTE | INSUFFICIENT
    confidence: float  # 0-1, calibrated
    gaps: list[Gap]


class VerificationVerdict(BaseModel):
    """Structured output schema for the orchestrator's final answer."""

    verdict: str  # SUPPORT | REFUTE | INSUFFICIENT
    confidence: float
    reasoning: str
    key_evidence: list[str]
    gaps_remaining: list[str]


class PaperFeatureVector(BaseModel):
    """Per-paper metadata feature vector for the sufficiency classifier.

    All fields are Optional because API calls may fail.
    Derived features (log_impact_factor, normalized_citation_count) are
    computed from raw values during extraction.
    """

    pmid: str

    # --- Metadata features ---
    publication_year: Optional[int] = None
    log_impact_factor: Optional[float] = None          # log(1 + IF)
    normalized_citation_count: Optional[float] = None  # citations / age
    author_h_index_max: Optional[int] = None           # max h-index among authors
