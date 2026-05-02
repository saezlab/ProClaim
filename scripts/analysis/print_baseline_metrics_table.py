#!/usr/bin/env python3
"""Render baseline metrics JSON files as ASCII tables and save a log copy.

Usage:
  uv run python scripts/analysis/print_baseline_metrics_table.py
  uv run python scripts/analysis/print_baseline_metrics_table.py --dataset signor
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "metrics"

BASELINE_ORDER = {
    "llm_only": 0,
    "retrieval": 1,
    "react": 2,
    "ace": 3,
    "fire": 4,
    "safe": 5,
    "open_scholar": 6,
}

VARIANT_ORDER = {
    "-": 0,
    "web": 1,
    "s2": 2,
    "s2/top5": 3,
}

MODEL_ORDER = {
    "anthropic--claude-sonnet-4-6": 0,
    "vertex_ai--gemini-2.5-flash": 1,
    "-": 99,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print baseline metrics as ASCII tables and save the report as a log file."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing baseline metrics JSON files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where the ASCII table log will be written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Restrict output to one or more datasets. Repeat the flag to include multiple datasets.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["baseline_order", "macro_f1", "accuracy", "weighted_fpr", "weighted_fnr", "total_cost_usd"],
        default="baseline_order",
        help="Sort rows within each dataset table. Default keeps a fixed baseline order across datasets.",
    )
    parser.add_argument(
        "--no-std",
        action="store_false",
        dest="use_std",
        help="Disable showing standard deviation in metric values (default: show std if available).",
    )
    return parser.parse_args()


def metric_value(metrics: dict, key: str, field: str = "mean", default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, dict):
        return float(value.get(field, default))
    return float(value)


def metric_with_std(metrics: dict, key: str, use_std: bool, decimals: int = 2) -> str:
    mean = metric_value(metrics, key, "mean")
    if use_std:
        std = metric_value(metrics, key, "std")
        return f"{mean:.{decimals}f}±{std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def parse_metric_path(metrics_path: Path, results_dir: Path) -> tuple[str, str, str]:
    relative_parts = metrics_path.relative_to(results_dir).parts
    directory_parts = list(relative_parts[:-1])
    baseline = directory_parts[0] if directory_parts else "unknown"
    model = "-"
    variant_parts: list[str] = []

    if baseline == "retrieval":
        if len(directory_parts) >= 2:
            variant_parts.append(directory_parts[1])
        if len(directory_parts) >= 3:
            model = directory_parts[2]
        if len(directory_parts) > 3:
            variant_parts.extend(directory_parts[3:])
    elif baseline in {"react", "safe"}:
        if len(directory_parts) >= 2:
            variant_parts.append(directory_parts[1])
        if len(directory_parts) >= 3:
            model = directory_parts[2]
        if len(directory_parts) > 3:
            variant_parts.extend(directory_parts[3:])
    elif len(directory_parts) >= 2:
        model = directory_parts[1]
        if len(directory_parts) > 2:
            variant_parts.extend(directory_parts[2:])

    variant = "/".join(variant_parts) if variant_parts else "-"
    return baseline, variant, model


def collect_rows(
    results_dir: Path,
    dataset_filters: set[str],
    sort_by: str,
    use_std: bool = True,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for metrics_path in sorted(results_dir.rglob("*_metrics.json")):
        dataset = metrics_path.stem.removesuffix("_metrics")
        if dataset_filters and dataset not in dataset_filters:
            continue

        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)

        baseline, variant, model = parse_metric_path(metrics_path, results_dir)
        rows.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "variant": variant,
                "model": model,
                "repeats": str(int(metrics.get("n_repeats", 0))),
                "n": str(int(metrics.get("n", 0))),
                "accuracy": metric_with_std(metrics, "accuracy", use_std),
                "macro_f1": '--',
                "macro_fpr": metric_with_std(metrics, "macro_fpr", use_std),
                "macro_fnr": metric_with_std(metrics, "macro_fnr", use_std),
                "total_cost_usd": metric_with_std(metrics, "total_cost_usd", use_std, decimals=3),
                "sort_value": metric_value(metrics, sort_by),
            }
        )
    return rows


def row_sort_key(row: dict[str, str | float], sort_by: str) -> tuple[object, ...]:
    baseline = str(row["baseline"])
    variant = str(row["variant"])
    model = str(row["model"])

    baseline_rank = BASELINE_ORDER.get(baseline, 999)
    variant_rank = VARIANT_ORDER.get(variant, 999)
    model_rank = MODEL_ORDER.get(model, 999)

    if sort_by == "baseline_order":
        return (baseline_rank, variant_rank, model_rank, baseline, variant, model)

    metric_direction = -1.0
    if sort_by in {"weighted_fpr", "weighted_fnr", "total_cost_usd"}:
        metric_direction = 1.0

    return (
        metric_direction * float(row["sort_value"]),
        baseline_rank,
        variant_rank,
        model_rank,
        baseline,
        variant,
        model,
    )


def build_ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    body_lines = [
        "& " + " & ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " &"
        for row in rows
    ]
    return "\n".join([separator, header_line, separator, *body_lines, separator])


def render_report(rows: list[dict[str, str | float]], results_dir: Path, sort_by: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    grouped_rows: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["dataset"])].append(row)

    lines = [
        "Baseline Metrics Summary",
        f"Generated: {timestamp}",
        f"Results dir: {results_dir}",
        f"Sort key: {sort_by}",
        "",
    ]

    if not grouped_rows:
        lines.append("No metrics files found.")
        return "\n".join(lines)

    headers = ["baseline", "variant", "model", "repeats", "n", "accuracy", "macro_f1", "macro_fpr", "macro_fnr", "total_cost_usd"]
    for dataset in sorted(grouped_rows):
        dataset_rows = sorted(grouped_rows[dataset], key=lambda row: row_sort_key(row, sort_by))
        table_rows = [[str(row[column]) for column in headers] for row in dataset_rows]
        lines.append(f"Dataset: {dataset}")
        lines.append(build_ascii_table(headers, table_rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    use_std = args.use_std
    dataset_filters = set(args.dataset)

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    rows = collect_rows(results_dir, dataset_filters, args.sort_by, use_std=use_std)
    report = render_report(rows, results_dir, args.sort_by)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"baseline_metrics_{timestamp}.log"
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(report, end="")
    print(f"Saved report to {log_path}")


if __name__ == "__main__":
    main()