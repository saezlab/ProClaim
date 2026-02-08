"""
Metacognitive Evidence Verification system.

Backbone implementation — minimal viable forms of each component
with clean extension points for later refinement.

Imports are lazy to avoid pulling in heavy dependencies (openai, fitz)
when only core data structures are needed.
"""


def __getattr__(name: str):
    """Lazy imports — only load a module when its symbol is accessed."""
    if name in ("PaperRecord", "Fact"):
        from pkevolve.verification.data_models import PaperRecord, Fact
        return PaperRecord if name == "PaperRecord" else Fact
    if name == "EvidenceState":
        from pkevolve.verification.evidence_state import EvidenceState
        return EvidenceState
    if name == "VerificationTools":
        from pkevolve.verification.tools import VerificationTools
        return VerificationTools
    if name == "SufficiencyClassifier":
        from pkevolve.verification.classifier import SufficiencyClassifier
        return SufficiencyClassifier
    if name == "SufficiencyPreservingCompressor":
        from pkevolve.verification.compressor import SufficiencyPreservingCompressor
        return SufficiencyPreservingCompressor
    if name == "MetacognitiveController":
        from pkevolve.verification.controller import MetacognitiveController
        return MetacognitiveController
    raise AttributeError(f"module 'pkevolve.verification' has no attribute {name!r}")


__all__ = [
    "PaperRecord",
    "Fact",
    "EvidenceState",
    "VerificationTools",
    "SufficiencyClassifier",
    "SufficiencyPreservingCompressor",
    "MetacognitiveController",
]
