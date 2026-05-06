#!/usr/bin/env python3
"""Render a pooled SIGNOR + ConnectomeDB baseline table as LaTeX.

Usage:
  uv run python scripts/analysis/print_aggregated_baseline_metrics_table.py
  uv run python scripts/analysis/print_aggregated_baseline_metrics_table.py --dataset signor --dataset connectomedb
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.shared.evaluate import EvaluationHarness
from experiments.baselines.shared.verdict import BaselineResult

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "metrics"

DATASET_ORDER = ["signor", "connectomedb"]
DATASET_TITLES = {
    "signor": "SIGNOR",
    "connectomedb": "ConnectomeDB",
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
    RowSpec(label="OpenScholar", baseline="open_scholar", model="anthropic--claude-sonnet-4-6"),
    # RowSpec(
    #     label="OS-Sonnet-4.6 + oracle",
    #     baseline="open_scholar",
    #     model="oracle",
    #     dataset_scope=frozenset({"signor"}),
    # ),
    RowSpec(label="", is_rule=True),
    RowSpec(label=r"\textbf{ProClaim} (ours)", is_ours=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print pooled baseline metrics as a paste-ready LaTeX table."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing baseline result JSONL files (default: {DEFAULT_RESULTS_DIR}).",
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
        help="Datasets to pool. Repeat the flag to include multiple datasets. Default pools SIGNOR and ConnectomeDB.",
    )
    parser.add_argument(
        "--label",
        default="tab:aggregate_results",
        help="LaTeX label written after the tabular environment.",
    )
    return parser.parse_args()


def result_subdir(spec: RowSpec) -> Path | None:
    if spec.baseline is None:
        return None
    path = Path(spec.baseline)
    if spec.baseline == "retrieval":
        variant_parts = [] if spec.variant == "-" else spec.variant.split("/")
        if variant_parts:
            path /= variant_parts[0]
        if spec.model != "-":
            path /= Path(spec.model)
        if len(variant_parts) > 1:
            path /= Path(*variant_parts[1:])
        return path
    if spec.variant != "-":
        path /= Path(spec.variant)
    if spec.model != "-":
        path /= Path(spec.model)
    return path


def dataset_result_path(results_dir: Path, spec: RowSpec, dataset: str) -> Path | None:
    if spec.is_ours:
        path = OURS_JSONL.get(dataset)
        if path is None:
            return None
        return path.resolve()

    subdir = result_subdir(spec)
    if subdir is None:
        return None
    return (results_dir / subdir / f"{dataset}_seed100.jsonl").resolve()


def load_pooled_results(paths: list[Path]) -> list[BaselineResult]:
    pooled: list[BaselineResult] = []
    for path in paths:
        pooled.extend(EvaluationHarness.load(path))
    return pooled


def metric_value(metrics: dict[str, object], key: str) -> float:
    value = metrics[key]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def build_row_metrics(
    datasets: list[str],
    results_dir: Path,
) -> dict[int, dict[str, float] | None]:
    row_metrics: dict[int, dict[str, float] | None] = {}

    ours_paths = [
        dataset_result_path(results_dir, spec=RowSpec(label="", is_ours=True), dataset=dataset)
        for dataset in datasets
    ]
    ours_available = all(path is not None and path.exists() for path in ours_paths)
    ours_metrics = None
    if ours_available:
        ours_metrics = EvaluationHarness.metrics(load_pooled_results([path for path in ours_paths if path is not None]))

    for index, spec in enumerate(ROW_SPECS):
        if spec.section or spec.is_rule:
            row_metrics[index] = None
            continue
        if spec.dataset_scope is not None and not set(datasets).issubset(spec.dataset_scope):
            row_metrics[index] = None
            continue

        result_paths = [
            dataset_result_path(results_dir, spec, dataset)
            for dataset in datasets
        ]
        if not all(path is not None and path.exists() for path in result_paths):
            row_metrics[index] = None
            continue

        metrics = EvaluationHarness.metrics(load_pooled_results([path for path in result_paths if path is not None]))
        accuracy = metric_value(metrics, "accuracy")
        row_metrics[index] = {
            "accuracy": accuracy,
            "delta": None if spec.is_ours or ours_metrics is None else metric_value(ours_metrics, "accuracy") - accuracy,
            "macro_fpr": metric_value(metrics, "macro_fpr"),
            "macro_fnr": metric_value(metrics, "macro_fnr"),
        }
    return row_metrics


def compute_bests(row_metrics: dict[int, dict[str, float] | None]) -> dict[str, float | None]:
    accuracy_values: list[float] = []
    delta_values: list[float] = []
    fpr_values: list[float] = []
    fnr_values: list[float] = []

    for metrics in row_metrics.values():
        if metrics is None:
            continue
        accuracy_values.append(metrics["accuracy"])
        fpr_values.append(metrics["macro_fpr"])
        fnr_values.append(metrics["macro_fnr"])
        if metrics["delta"] is not None:
            delta_values.append(metrics["delta"])

    return {
        "accuracy": max(accuracy_values) if accuracy_values else None,
        "delta": min(delta_values) if delta_values else None,
        "macro_fpr": min(fpr_values) if fpr_values else None,
        "macro_fnr": min(fnr_values) if fnr_values else None,
    }


def format_number(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.2f}"
    if bold:
        return rf"\underline{{{text}}}"
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


def tabular_spec() -> str:
    return "@{} l | c c c c @{}"


def table_title(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return DATASET_TITLES[datasets[0]]
    return "ProClaim-eval"


def table_header_color(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return DATASET_HEADER_COLORS[datasets[0]]
    return DATASET_HEADER_COLORS["signor"]


def dataset_header_line(datasets: list[str]) -> str:
    return (
        " "
        + " & "
        + rf"\multicolumn{{4}}{{c}}{{\cellcolor{{{table_header_color(datasets)}}}\textbf{{{table_title(datasets)}}}}}"
        + r" \\"
    )


def column_header_line() -> str:
    headers = [
        r"\textbf{Method}",
        r"Agreement$\uparrow$",
        r"$\boldsymbol{\Delta}\uparrow$",
        r"FPR$\downarrow$",
        r"FNR$\downarrow$",
    ]
    return " & ".join(headers) + r" \\"


def render_table(
    datasets: list[str],
    row_metrics: dict[int, dict[str, float] | None],
    bests: dict[str, float | None],
    label: str,
) -> str:
    column_count = 5
    lines = [
        rf"\begin{{tabular}}{{{tabular_spec()}}}",
        r"\toprule",
        dataset_header_line(datasets),
        column_header_line(),
        r"\midrule",
    ]

    for index, spec in enumerate(ROW_SPECS):
        if spec.is_rule:
            lines.append(r"\midrule")
            continue
        if spec.section:
            lines.append(rf"\multicolumn{{{column_count}}}{{@{{}}l}}{{\textit{{{spec.section}}}}} \\[2pt]")
            continue
        metrics = row_metrics[index]
        if metrics is None:
            lines.append(" & ".join([spec.label, "--", "--", "--", "--"]) + " \\\\")
            continue

        lines.append(
            " & ".join(
                [
                    spec.label,
                    format_number(metrics["accuracy"], bold=is_best(metrics["accuracy"], bests["accuracy"])),
                    format_delta(metrics["delta"]),
                    format_number(metrics["macro_fpr"], bold=is_best(metrics["macro_fpr"], bests["macro_fpr"])),
                    format_number(metrics["macro_fnr"], bold=is_best(metrics["macro_fnr"], bests["macro_fnr"])),
                ]
            )
            + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])
    return "\n".join(lines) + "\n"


def output_filename(datasets: list[str]) -> str:
    dataset_part = "+".join(datasets)
    return f"baseline_metrics_aggregated_{dataset_part}.tex"


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    datasets = [dataset for dataset in DATASET_ORDER if not args.dataset or dataset in set(args.dataset)]

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    if not datasets:
        raise SystemExit("No datasets selected.")

    row_metrics = build_row_metrics(datasets, results_dir)
    bests = compute_bests(row_metrics)
    report = render_table(datasets, row_metrics, bests, args.label)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(datasets)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(report, end="")


if __name__ == "__main__":
    main()