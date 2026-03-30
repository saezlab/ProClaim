"""
Label normalization utilities for evidence programming baselines.

All datasets use different label vocabularies. This module maps every
raw label to the canonical {SUPPORT, REFUTE, NEI} taxonomy so that
metrics can be compared across datasets and baselines.
"""

from __future__ import annotations

# Mapping from raw dataset labels → canonical labels
LABEL_MAP: dict[str, str] = {
    # Canonical (pass-through)
    "SUPPORT": "SUPPORT",
    "REFUTE": "REFUTE",
    "NEI": "NEI",
    # SciFact-Open
    "CONTRADICT": "REFUTE",
    # SIGNOR*
    "SUPPORTED": "SUPPORT",
    "WRONG": "REFUTE",
    "UNCERTAIN": "NEI",
    # CIViC-Fact
    "SUPPORTS": "SUPPORT",
    "REFUTES": "REFUTE",
    # ConnectomeDB
    "REFUTED": "REFUTE",
    # Evidence Programming system
    "UNCERTAIN": "NEI",
    # Common variants
    "NOT ENOUGH INFORMATION": "NEI",
    "NOT_ENOUGH_INFORMATION": "NEI",
}

CANONICAL_LABELS = frozenset({"SUPPORT", "REFUTE", "NEI"})


def normalize_label(label: str) -> str:
    """Map a raw dataset label to the canonical {SUPPORT, REFUTE, NEI} taxonomy.

    Unknown labels default to NEI.
    """
    return LABEL_MAP.get(label.upper().strip().replace("-", "_"), "NEI")
