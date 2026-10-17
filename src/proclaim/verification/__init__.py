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
    if name in ("PaperRecord", "Fact", "Stance", "GapType", "GapPriority", "Conflict",
                "Gap", "SufficiencyResult", "VerificationVerdict"):
        import proclaim.verification.data_models as dm
        return getattr(dm, name)
    # Configuration
    if name in ("VerificationSettings", "APISettings", "LLMSettings",
                "get_settings"):
        import proclaim.verification.config as cfg_mod
        return getattr(cfg_mod, name)
    # Evidence state
    if name == "EvidenceState":
        from proclaim.verification.evidence_state import EvidenceState
        return EvidenceState
    if name == "TraceLog":
        from proclaim.verification.evidence_state import TraceLog
        return TraceLog
    # Compressor
    if name == "SufficiencyPreservingCompressor":
        from proclaim.verification.compressor import SufficiencyPreservingCompressor
        return SufficiencyPreservingCompressor
    raise AttributeError(f"module 'proclaim.verification' has no attribute {name!r}")


__all__ = [
    # Data models
    "PaperRecord",
    "Fact",
    "Stance",
    "GapType",
    "GapPriority",
    "Conflict",
    "Gap",
    "SufficiencyResult",
    "VerificationVerdict",
    # State
    "EvidenceState",
    "TraceLog",
    # Configuration
    "VerificationSettings",
    "APISettings",
    "LLMSettings",
    "get_settings",
    # Compressor
    "SufficiencyPreservingCompressor",
]
