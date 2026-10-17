"""Label normalization and verdict-label helpers.

Maps each dataset's raw labels to the canonical {SUPPORT, REFUTE, UNCERTAIN}
taxonomy, and exposes a shared LabelConfig plus prompt-fragment helpers.
"""

from __future__ import annotations

from proclaim.verification.config import LabelConfig

# ── Shared LabelConfig singleton ──────────────────────────────────────

_LABEL_CONFIG = LabelConfig()


def get_label_config() -> LabelConfig:
    """Return the shared default :class:`LabelConfig` instance."""
    return _LABEL_CONFIG


# ── Verdict-label helpers (delegate to active LabelConfig) ───────────


def verdict_names() -> list[str]:
    """Canonical verdict label names, e.g. ``["SUPPORT", "REFUTE", "UNCERTAIN"]``."""
    return _LABEL_CONFIG.verdict_names()


def verdict_options_str() -> str:
    """Pipe-separated quoted options, e.g. ``'"SUPPORT" | "REFUTE" | "UNCERTAIN"'``."""
    return " | ".join(f'"{n}"' for n in _LABEL_CONFIG.verdict_names())


def verdict_or_str() -> str:
    """Or-separated options for prose, e.g. ``'"SUPPORT" or "REFUTE" or "UNCERTAIN"'``."""
    return " or ".join(f'"{n}"' for n in _LABEL_CONFIG.verdict_names())


def verdict_defs_block(indent: str = "   ") -> str:
    """Multi-line verdict definitions block for LLM prompts.

    Each line is formatted as ``{indent}- "{NAME}" — {description}``.
    """
    lines = []
    for name, desc in _LABEL_CONFIG.verdict_labels.items():
        lines.append(f'{indent}- "{name}" — {desc}')
    return "\n".join(lines)


def validate_verdict(value: str) -> str:
    """Validate a verdict string against configured labels (case-insensitive).

    Returns the canonical label name if recognised, otherwise passes through.
    """
    return _LABEL_CONFIG.validate_verdict(value)


# ── Raw-label → canonical mapping ────────────────────────────────────

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
    # ConnectomeDB
    "REFUTED": "REFUTE",
    # # CIViC-Fact
    # "SUPPORTS": "SUPPORT",
    # "REFUTES": "REFUTE",
    # # Common variants
    # "NEI": "UNCERTAIN",
    # "NOT ENOUGH INFORMATION": "UNCERTAIN",
    # "NOT_ENOUGH_INFORMATION": "UNCERTAIN",
}

CANONICAL_LABELS = frozenset({"SUPPORT", "REFUTE", "UNCERTAIN"})


def normalize_label(label: str) -> str:
    """Map a raw dataset label to the canonical {SUPPORT, REFUTE, UNCERTAIN} taxonomy.

    Unknown labels default to UNCERTAIN.
    """
    return LABEL_MAP.get(label.upper().strip().replace("-", "_"), "UNCERTAIN")
