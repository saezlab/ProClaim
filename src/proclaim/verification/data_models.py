"""
Core data models for the Evidence Verification system.

Pydantic v2 models with backward-compatible field names.
PaperRecord and Fact are the atomic data units stored in EvidenceState.
Additional types: Stance, GapType, Conflict, Gap, SufficiencyResult, VerificationVerdict.
"""

import enum as _enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, Field


# ---------------------------------------------------------------------------
# Dynamic Stance enum
# ---------------------------------------------------------------------------


def _make_stance_enum(labels: dict[str, str]) -> type:
    """Create a dynamic ``Stance`` str-Enum from label definitions.

    Each key in *labels* becomes both the member name and its value.
    The resulting enum inherits from ``str``, so members compare equal to
    their string values: ``Stance.SUPPORT == "SUPPORT"`` → ``True``.

    ``__str__`` and ``__format__`` are overridden so that f-strings and
    ``print()`` render the **value** (e.g. ``"SUPPORT"``) instead of the
    default ``"Stance.SUPPORT"`` representation.
    """
    members = {name: name for name in labels}
    StanceEnum = _enum.Enum("Stance", members, type=str)

    # Override so f"{stance}" and str(stance) return the value, not "Stance.NAME"
    def _str(self):
        return self.value

    def _format(self, spec):
        return format(self.value, spec)

    StanceEnum.__str__ = _str
    StanceEnum.__format__ = _format
    # Preserve descriptions as a class attribute for prompt generation
    StanceEnum._descriptions = labels
    return StanceEnum


# Default Stance enum — redefined at runtime by ``rebuild_stance_enum()``
Stance = _make_stance_enum({
    "SUPPORT": "The fact directly supports or corroborates the claim.",
    "REFUTE": "The fact directly contradicts or refutes the claim.",
    "NEUTRAL": "The fact is relevant to the claim but neither clearly supports nor refutes it.",
})


def _validate_stance_field(v: Any) -> Any:
    """Pydantic BeforeValidator for the ``Fact.stance`` field.

    Reads the *current* module-level ``Stance`` enum (which may have been
    rebuilt with custom labels) and coerces the input value to a member.
    Falls back to the configured default stance for unrecognised values.
    """
    import proclaim.verification.data_models as _dm
    CurrentStance = _dm.Stance
    if isinstance(v, CurrentStance):
        return v
    s = str(v).upper().strip()
    try:
        return CurrentStance(s)
    except (ValueError, KeyError):
        from proclaim.verification.config import get_label_config
        default = get_label_config().default_stance
        return CurrentStance(default)


# Annotated type used by Fact.stance — dynamically resolves to the current Stance enum
DynamicStance = Annotated[Any, BeforeValidator(_validate_stance_field)]


def rebuild_stance_enum(labels: dict[str, str]) -> type:
    """Rebuild the module-level ``Stance`` enum from new label definitions.

    After this call, any new ``Fact`` objects will validate ``stance``
    against the **new** enum members.  No ``Fact.model_rebuild()`` is
    needed because the ``DynamicStance`` field uses a ``BeforeValidator``
    that reads the current ``Stance`` reference at validation time.

    Called automatically by ``config.set_label_config()`` when the user
    provides custom stance labels.

    Args:
        labels: Mapping from label name → description.  Keys become enum
            member names and values.

    Returns:
        The newly created Stance enum class.
    """
    global Stance
    Stance = _make_stance_enum(labels)
    return Stance


class GapType(str, _enum.Enum):
    """Types of evidence gaps the classifier can identify."""
    MISSING_SUBCLAIM = "missing_subclaim_evidence"
    CONTRADICTORY = "contradictory_evidence"
    LOW_DIVERSITY = "low_source_diversity"
    WEAK_STANCE = "weak_stance_evidence"
    MISSING_MECHANISM = "missing_mechanism"
    MISSING_QUANTITATIVE = "missing_quantitative"
    MISSING_TEMPORAL = "missing_temporal"
    MISSING_POPULATION = "missing_population"


class GapPriority(str, _enum.Enum):
    """Discrete priority levels for evidence gaps."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PaperRecord(BaseModel):
    """A single paper retrieved from PubMed or other sources."""

    pmid: str
    title: str
    abstract: str = ""
    full_text: Optional[str] = None
    summary: Optional[str] = None  # L1 — set by summarize_paper tool
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    source: str = "pubmed"  # pubmed | semantic_scholar
    reference_dois: list[str] = Field(default_factory=list)
    metadata: Optional["PaperFeatureVector"] = None
    nlp: Optional["NLPFeatureVector"] = None

    def text_for_summarization(self) -> str:
        """Return the best available text for LLM summarization."""
        if self.full_text:
            return self.full_text
        return self.abstract


class Fact(BaseModel):
    """A stance-labeled fact extracted from a paper.

    The ``stance`` field uses a dynamic ``BeforeValidator`` that always
    resolves against the *current* module-level ``Stance`` enum.  This
    means the enum can be rebuilt at runtime (via ``rebuild_stance_enum()``)
    without needing to call ``Fact.model_rebuild()``.
    """

    id: str = ""
    text: str
    stance: DynamicStance
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
    priority: GapPriority


class SufficiencyResult(BaseModel):
    """Result from the sufficiency classifier."""

    label: str  # "sufficient" | "insufficient"
    confidence: float  # 0-1, calibrated
    gaps: list[Gap]


class VerificationVerdict(BaseModel):
    """Structured output schema for the orchestrator's final answer."""

    verdict: str  # SUPPORT | REFUTE | UNCERTAIN
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


class NLPFeatureVector(BaseModel):
    """NLP features for a single claim–evidence pair.

    Entity Overlap Ratio: Jaccard similarity of named entities
    detected in the claim and evidence texts.
    """
    claim_entity_coverage: Optional[float] = None  # Recall: |Claim & Evidence| / |Claim|
    semantic_similarity: Optional[float] = None    # SBERT cosine similarity (0.0 - 1.0)
    claim_entities: list[str] = Field(default_factory=list)
    evidence_entities: list[str] = Field(default_factory=list)
    
    nli_entailment: Optional[float] = None
    nli_contradiction: Optional[float] = None
    nli_neutral: Optional[float] = None
    nli_best_chunk_text: Optional[str] = None
