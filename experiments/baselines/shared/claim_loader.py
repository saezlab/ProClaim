"""Shared dataset-CSV loader for baseline drivers."""

from __future__ import annotations

import csv
from pathlib import Path


def load_claim_csv(
    dataset: str,
    datasets_dir: Path | str,
    limit: int | None = None,
) -> list[dict]:
    """Load ``{dataset}.csv`` into claim dicts for :class:`EvaluationHarness`.

    SIGNOR carries a ``flip`` column (the original and flipped variants of a
    claim share one ``id``), so flipped rows take a ``_flip`` suffix to keep
    ``claim_id`` unique. ``dir_id`` is the id in cached-run directory form
    (``:`` and spaces replaced by ``_``); it is identical to ``base_id`` for
    SIGNOR and differs only for ConnectomeDB.
    """
    csv_path = Path(datasets_dir) / f"{dataset}.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    claims = []
    for r in rows:
        flip = str(r.get("flip", "")).strip().lower() == "true"
        base_id = r["id"]
        claims.append({
            "claim_id": base_id + ("_flip" if flip else ""),
            "base_id": base_id,
            "dir_id": base_id.replace(":", "_").replace(" ", "_"),
            "flip": flip,
            "claim": r["claim"],
            "gold_label": r["label"],
            "dataset": dataset,
        })
    return claims[:limit] if limit is not None else claims
