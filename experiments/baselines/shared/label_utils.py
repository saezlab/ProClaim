"""
Label normalization utilities for evidence programming baselines.

All datasets use different label vocabularies. This module maps every
raw label to the canonical {SUPPORT, REFUTE, UNCERTAIN} taxonomy so that
metrics can be compared across datasets and baselines.
"""

from __future__ import annotations

# Mapping from raw dataset labels → canonical labels
LABEL_MAP: dict[str, str] = {
    # Canonical (pass-through)
    "SUPPORT": "SUPPORT",
    "REFUTE": "REFUTE",
    "UNCERTAIN": "UNCERTAIN",
    # SciFact-Open
    "CONTRADICT": "REFUTE",
    # SIGNOR
    "SUPPORTED": "SUPPORT",
    "WRONG": "REFUTE",
    # CIViC-Fact
    "SUPPORTS": "SUPPORT",
    "REFUTES": "REFUTE",
    # ConnectomeDB
    "REFUTED": "REFUTE",
    # Common variants
    "NEI": "UNCERTAIN",
    "NOT ENOUGH INFORMATION": "UNCERTAIN",
    "NOT_ENOUGH_INFORMATION": "UNCERTAIN",
}

CANONICAL_LABELS = frozenset({"SUPPORT", "REFUTE", "UNCERTAIN"})


def normalize_label(label: str) -> str:
    """Map a raw dataset label to the canonical {SUPPORT, REFUTE, UNCERTAIN} taxonomy.

    Unknown labels default to UNCERTAIN.
    """
    return LABEL_MAP.get(label.upper().strip().replace("-", "_"), "UNCERTAIN")
