"""
Dataset adapters for Evidence Programming.

Convert SIGNOR edges and SciFact claims to a common Claim format
so the orchestrator is dataset-agnostic.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """Common claim format for all datasets."""

    id: str
    text: str
    gold_label: Optional[str] = None  # SUPPORT | REFUTE | None
    metadata: dict = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Abstract base for dataset adapters."""

    @abstractmethod
    def load(self) -> list[Claim]:
        """Load all claims from the dataset."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def iter_claims(self, max_claims: int = 0) -> Iterator[Claim]:
        """Iterate over claims with optional limit."""
        claims = self.load()
        if max_claims > 0:
            claims = claims[:max_claims]
        yield from claims


class SignorAdapter(DatasetAdapter):
    """
    Adapter for SIGNOR protein-protein interaction edges.

    Reads true_positive_edges.csv and true_negative_edges.csv from data/signor/.
    Uses construct_signor_question() to generate natural language claims.

    Gold labels:
      - true_positive -> SUPPORT (the edge is real)
      - true_negative -> REFUTE (the edge is an artifact)
    """

    def __init__(self, data_dir: Path, label: str = "both"):
        """
        Args:
            data_dir: Path to data/signor/ directory.
            label: "true_positive", "true_negative", or "both".
        """
        self.data_dir = data_dir
        self.label = label
        self._claims: Optional[list[Claim]] = None

    def load(self) -> list[Claim]:
        if self._claims is not None:
            return self._claims

        from proclaim.utils.signor_utils import (
            construct_signor_question,
            load_signor_data,
        )

        claims: list[Claim] = []
        labels_to_load = (
            ["true_positive", "true_negative"]
            if self.label == "both"
            else [self.label]
        )

        for lbl in labels_to_load:
            csv_path = self.data_dir / f"{lbl}_edges.csv"
            if not csv_path.exists():
                logger.warning("SIGNOR CSV not found: %s", csv_path)
                continue

            df = load_signor_data(str(csv_path))
            if df is None or df.empty:
                logger.warning("No data loaded from %s", csv_path)
                continue

            gold = "SUPPORT" if lbl == "true_positive" else "REFUTE"

            for idx, row in df.iterrows():
                source = str(row["ENTITYA"])
                target = str(row["ENTITYB"])
                effect = str(row["EFFECT"])

                claim_text = construct_signor_question(source, target, effect)
                claim_id = f"signor_{lbl}_{idx}_{source}_{target}"

                claims.append(Claim(
                    id=claim_id,
                    text=claim_text,
                    gold_label=gold,
                    metadata={
                        "source": source,
                        "target": target,
                        "effect": effect,
                        "label": lbl,
                        "pmid": str(row.get("PMID", "")),
                        "mechanism": str(row.get("MECHANISM", "")),
                        "dataset": "signor",
                    },
                ))

        self._claims = claims
        return claims

    def __len__(self) -> int:
        return len(self.load())


class SciFactAdapter(DatasetAdapter):
    """
    Adapter for SciFact claim verification dataset.

    Reads SciFact JSON format (list of objects with id, claim, label fields).
    Normalizes SUPPORTS -> SUPPORT, REFUTES -> REFUTE.
    """

    def __init__(self, claims_path: Path):
        self.claims_path = claims_path
        self._claims: Optional[list[Claim]] = None

    def load(self) -> list[Claim]:
        if self._claims is not None:
            return self._claims

        data = json.loads(self.claims_path.read_text())
        claims: list[Claim] = []

        for item in data:
            gold = None
            original_label = item.get("label")
            if original_label == "SUPPORTS":
                gold = "SUPPORT"
            elif original_label == "REFUTES":
                gold = "REFUTE"

            claims.append(Claim(
                id=f"scifact_{item['id']}",
                text=item["claim"],
                gold_label=gold,
                metadata={
                    "dataset": "scifact",
                    "original_label": original_label,
                    "evidence_doc_ids": list(item.get("evidence", {}).keys()),
                },
            ))

        self._claims = claims
        return claims

    def __len__(self) -> int:
        return len(self.load())
