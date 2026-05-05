#!/usr/bin/env python3
"""Render baseline metrics as a paper-ready LaTeX table.

Usage:
  uv run python scripts/analysis/print_baseline_metrics_table.py
  uv run python scripts/analysis/print_baseline_metrics_table.py --dataset signor
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.shared.evaluate import EvaluationHarness

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "metrics"

DATASET_ORDER = ["signor", "connectomedb"]
DATASET_TITLES = {
    "signor": "SIGNOR-Fact",
    "connectomedb": "ConnectomeDB-Fact",
}
DATASET_HEADER_COLORS = {
    "signor": "headerteal",
    "connectomedb": "headerlavender",
}
OURS_JSONL = {
    "signor": PROJECT_ROOT / "results/baselines/signor_direct_eval_20260427_221617/signor_seed100.jsonl",
    "connectomedb": PROJECT_ROOT / "results/baselines/connectomedb_eval_20260424_171117/connectomedb_seed100.jsonl",
}


@dataclass(frozen=True)
class RowSpec:
    label: str
    baseline: str | None = None
    variant: str = "-"
    model: str = "-"
    section: str | None = None
    is_rule: bool = False
    is_ours: bool = False
    dataset_scope: frozenset[str] | None = None


ROW_SPECS = [
    RowSpec(label="Random", baseline="random"),
    RowSpec(label="", is_rule=True),
    RowSpec(label="", section="LLM-only"),
    RowSpec(label="Claude-Sonnet-4.6", baseline="llm_only", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="Gemini-2.5-Flash", baseline="llm_only", model="vertex_ai--gemini-2.5-flash"),
    RowSpec(label="", is_rule=True),
    RowSpec(label="", section="Static retrieval S2 top 5"),
    RowSpec(label="Claude-Sonnet-4.6", baseline="retrieval", variant="s2/top5", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="Gemini-2.5-Flash", baseline="retrieval", variant="s2/top5", model="vertex_ai--gemini-2.5-flash"),
    RowSpec(label="", is_rule=True),
    RowSpec(label="", section="Adaptive retrieval"),
    RowSpec(label="ReAct + S2", baseline="react", variant="s2", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="FIRE", baseline="fire", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="SAFE", baseline="safe", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="OS-Sonnet-4.6", baseline="open_scholar", model="anthropic--claude-sonnet-4-6"),
    RowSpec(
        label="OS-Sonnet-4.6 + oracle",
        baseline="open_scholar",
        model="oracle",
        dataset_scope=frozenset({"signor"}),
    ),
    RowSpec(label="", is_rule=True),
    RowSpec(label=r"\textbf{ProClaim} (ours)", is_ours=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print baseline metrics as a paste-ready LaTeX table."
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
        help=f"Directory where the LaTeX table will be written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_ORDER,
        default=[],
        help="Restrict output to one or more datasets. Repeat the flag to include multiple datasets.",
    )
    parser.add_argument(
        "--label",
        default="tab:main_results",
        help="LaTeX label written after the tabular environment.",
    )
    return parser.parse_args()


def metric_value(metrics: dict[str, object], key: str, field: str = "mean") -> float:
    value = metrics[key]
    if isinstance(value, dict):
        return float(value[field])
    return float(value)


def parse_metric_path(metrics_path: Path, results_dir: Path) -> tuple[str, str, str, str]:
    relative_parts = metrics_path.relative_to(results_dir).parts
    directory_parts = list(relative_parts[:-1])
    baseline = directory_parts[0] if directory_parts else "unknown"
    model = "-"
    variant_parts: list[str] = []

    if baseline in {"retrieval", "react"}:
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

    dataset = metrics_path.stem.removesuffix("_metrics")
    variant = "/".join(variant_parts) if variant_parts else "-"
    return dataset, baseline, variant, model


def collect_metrics(results_dir: Path) -> dict[tuple[str, str, str, str], dict[str, object]]:
    metrics_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for metrics_path in sorted(results_dir.rglob("*_metrics.json")):
        dataset, baseline, variant, model = parse_metric_path(metrics_path, results_dir)
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)

        if "total_input_tokens" not in metrics or "total_output_tokens" not in metrics:
            jsonl_path = metrics_path.with_name(f"{dataset}_seed100.jsonl")
            if jsonl_path.exists():
                jsonl_metrics = EvaluationHarness.metrics(EvaluationHarness.load(jsonl_path))
                metrics.setdefault("total_input_tokens", jsonl_metrics["total_input_tokens"])
                metrics.setdefault("total_output_tokens", jsonl_metrics["total_output_tokens"])

        metrics_by_key[(dataset, baseline, variant, model)] = metrics
    return metrics_by_key


def load_ours_metrics() -> dict[str, dict[str, object]]:
    ours: dict[str, dict[str, object]] = {}
    for dataset, jsonl_path in OURS_JSONL.items():
        if not jsonl_path.exists():
            continue
        ours[dataset] = EvaluationHarness.metrics(EvaluationHarness.load(jsonl_path))
    return ours


def format_number(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.2f}"
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def average_total_tokens(metrics: dict[str, object]) -> float:
    total_examples = metric_value(metrics, "n")
    if total_examples <= 0:
        return 0.0
    total_input_tokens = metric_value(metrics, "total_input_tokens")
    total_output_tokens = metric_value(metrics, "total_output_tokens")
    return (total_input_tokens + total_output_tokens) / total_examples


def format_token_millions(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value / 1_000_000:.3f}"
    if bold:
        return rf"\textbf{{{text}}}"
    return text


def delta_color(delta_value: float) -> str:
    magnitude = abs(delta_value)
    if magnitude <= 0.10:
        return "deltas"
    if magnitude <= 0.20:
        return "deltam"
    if magnitude <= 0.30:
        return "deltal"
    return "deltaxl"


def format_delta(delta_value: float | None) -> str:
    if delta_value is None:
        return "--"
    sign = "+" if delta_value > 0 else ""
    return rf"\cellcolor{{{delta_color(delta_value)}}}{sign}{delta_value:.2f}"


def is_best(value: float | None, best_value: float | None) -> bool:
    return value is not None and best_value is not None and abs(value - best_value) < 1e-9


def build_row_metrics(
    datasets: list[str],
    metrics_by_key: dict[tuple[str, str, str, str], dict[str, object]],
    ours_metrics: dict[str, dict[str, object]],
) -> dict[int, dict[str, dict[str, float] | None]]:
    row_metrics: dict[int, dict[str, dict[str, float] | None]] = {}
    for index, spec in enumerate(ROW_SPECS):
        dataset_map: dict[str, dict[str, float] | None] = {}
        for dataset in datasets:
            if spec.section or spec.is_rule:
                dataset_map[dataset] = None
                continue
            if spec.dataset_scope is not None and dataset not in spec.dataset_scope:
                dataset_map[dataset] = None
                continue
            if spec.is_ours:
                metrics = ours_metrics.get(dataset)
            else:
                metrics = metrics_by_key.get((dataset, spec.baseline or "", spec.variant, spec.model))
            if metrics is None:
                dataset_map[dataset] = None
                continue
            accuracy = metric_value(metrics, "accuracy")
            ours_accuracy = metric_value(ours_metrics[dataset], "accuracy") if dataset in ours_metrics else None
            dataset_map[dataset] = {
                "accuracy": accuracy,
                "delta": None if spec.is_ours or ours_accuracy is None else ours_accuracy - accuracy,
                "macro_fpr": metric_value(metrics, "macro_fpr"),
                "macro_fnr": metric_value(metrics, "macro_fnr"),
                "avg_total_tokens": average_total_tokens(metrics),
            }
        row_metrics[index] = dataset_map
    return row_metrics


def compute_bests(
    datasets: list[str],
    row_metrics: dict[int, dict[str, dict[str, float] | None]],
) -> dict[str, dict[str, float | None]]:
    bests: dict[str, dict[str, float | None]] = {}
    for dataset in datasets:
        accuracy_values: list[float] = []
        delta_values: list[float] = []
        fpr_values: list[float] = []
        fnr_values: list[float] = []
        token_values: list[float] = []
        for per_dataset in row_metrics.values():
            metrics = per_dataset.get(dataset)
            if metrics is None:
                continue
            accuracy_values.append(metrics["accuracy"])
            fpr_values.append(metrics["macro_fpr"])
            fnr_values.append(metrics["macro_fnr"])
            token_values.append(metrics["avg_total_tokens"])
            if metrics["delta"] is not None:
                delta_values.append(metrics["delta"])
        bests[dataset] = {
            "accuracy": max(accuracy_values) if accuracy_values else None,
            "delta": min(delta_values) if delta_values else None,
            "macro_fpr": min(fpr_values) if fpr_values else None,
            "macro_fnr": min(fnr_values) if fnr_values else None,
            "avg_total_tokens": min(token_values) if token_values else None,
        }
    return bests


def tabular_spec(datasets: list[str]) -> str:
    dataset_blocks = ["c c *{3}{c}" for _ in datasets]
    return "@{} l | " + " | ".join(dataset_blocks) + " @{}"


def dataset_header_line(datasets: list[str]) -> str:
    parts = [" "]
    for index, dataset in enumerate(datasets):
        suffix = "|" if index < len(datasets) - 1 else ""
        parts.append(
            rf"\multicolumn{{5}}{{c{suffix}}}{{\cellcolor{{{DATASET_HEADER_COLORS[dataset]}}}\textbf{{{DATASET_TITLES[dataset]}}}}}"
        )
    return " & ".join(parts) + r" \\"


def column_header_line(datasets: list[str]) -> str:
    headers = [r"\textbf{Method}"]
    for _dataset in datasets:
        headers.extend([
            r"$\textbf{AGR}\uparrow$",
            r"$\boldsymbol{\Delta}\uparrow$",
            r"$\textbf{FPR}\downarrow$",
            r"$\textbf{FNR}\downarrow$",
            r"$\textbf{Tokens (M)}\downarrow$",
        ])
    return " & ".join(headers) + r" \\"


def render_table(
    datasets: list[str],
    row_metrics: dict[int, dict[str, dict[str, float] | None]],
    bests: dict[str, dict[str, float | None]],
    label: str,
) -> str:
    column_count = 1 + 5 * len(datasets)
    lines = [
        rf"\begin{{tabular}}{{{tabular_spec(datasets)}}}",
        r"\toprule",
        dataset_header_line(datasets),
        column_header_line(datasets),
        r"\midrule",
    ]

    for index, spec in enumerate(ROW_SPECS):
        if spec.is_rule:
            lines.append(r"\midrule")
            continue
        if spec.section:
            lines.append(rf"\multicolumn{{{column_count}}}{{@{{}}l}}{{\textit{{{spec.section}}}}} \\[2pt]")
            continue

        cells = [spec.label]
        for dataset in datasets:
            metrics = row_metrics[index][dataset]
            if metrics is None:
                cells.extend(["--", "--", "--", "--", "--"])
                continue
            cells.extend([
                format_number(
                    metrics["accuracy"],
                    bold=is_best(metrics["accuracy"], bests[dataset]["accuracy"]),
                ),
                format_delta(metrics["delta"]),
                format_number(
                    metrics["macro_fpr"],
                    bold=is_best(metrics["macro_fpr"], bests[dataset]["macro_fpr"]),
                ),
                format_number(
                    metrics["macro_fnr"],
                    bold=is_best(metrics["macro_fnr"], bests[dataset]["macro_fnr"]),
                ),
                format_token_millions(
                    metrics["avg_total_tokens"],
                    bold=is_best(metrics["avg_total_tokens"], bests[dataset]["avg_total_tokens"]),
                ),
            ])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])
    return "\n".join(lines) + "\n"


def output_filename(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return f"baseline_metrics_{datasets[0]}.tex"
    return "baseline_metrics_main.tex"


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    datasets = [dataset for dataset in DATASET_ORDER if not args.dataset or dataset in set(args.dataset)]

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    if not datasets:
        raise SystemExit("No datasets selected.")

    metrics_by_key = collect_metrics(results_dir)
    ours_metrics = load_ours_metrics()
    row_metrics = build_row_metrics(datasets, metrics_by_key, ours_metrics)
    bests = compute_bests(datasets, row_metrics)
    report = render_table(datasets, row_metrics, bests, args.label)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(datasets)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(report, end="")


if __name__ == "__main__":
    main()