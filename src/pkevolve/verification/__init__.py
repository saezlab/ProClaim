"""
Metacognitive Evidence Verification system.

Backbone implementation — minimal viable forms of each component
with clean extension points for later refinement.

Imports are lazy to avoid pulling in heavy dependencies (mcp, pydantic)
when only core data structures are needed.
"""


def __getattr__(name: str):
    """Lazy imports — only load a module when its symbol is accessed."""
    # Data models
    if name in ("PaperRecord", "Fact", "Stance", "GapType", "Conflict",
                "Gap", "SufficiencyResult", "VerificationVerdict"):
        import pkevolve.verification.data_models as dm
        return getattr(dm, name)
    # Evidence state
    if name == "EvidenceState":
        from pkevolve.verification.evidence_state import EvidenceState
        return EvidenceState
    if name == "TraceLog":
        from pkevolve.verification.evidence_state import TraceLog
        return TraceLog
    # Classifier
    if name == "SufficiencyClassifier":
        from pkevolve.verification.classifier import SufficiencyClassifier
        return SufficiencyClassifier
    # Compressor
    if name == "SufficiencyPreservingCompressor":
        from pkevolve.verification.compressor import SufficiencyPreservingCompressor
        return SufficiencyPreservingCompressor
    raise AttributeError(f"module 'pkevolve.verification' has no attribute {name!r}")


__all__ = [
    # Data models
    "PaperRecord",
    "Fact",
    "Stance",
    "GapType",
    "Conflict",
    "Gap",
    "SufficiencyResult",
    "VerificationVerdict",
    # State
    "EvidenceState",
    "TraceLog",
    # Classifier
    "SufficiencyClassifier",
    # Compressor
    "SufficiencyPreservingCompressor",
]
