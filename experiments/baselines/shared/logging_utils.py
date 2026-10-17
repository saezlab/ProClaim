"""Shared helpers for writing per-claim baseline logs."""

from __future__ import annotations

from pathlib import Path


def write_claim_log(
    log_dir: Path | None,
    claim_id: str,
    sections: list[tuple[str, str]],
) -> None:
    """Write a per-claim log with named sections if logging is enabled."""
    if log_dir is None:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{claim_id}.log"
    with open(log_path, "w") as handle:
        for title, body in sections:
            handle.write(f"=== {title} ===\n")
            if body:
                handle.write(body)
            handle.write("\n\n")