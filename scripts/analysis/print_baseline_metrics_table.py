#!/usr/bin/env python3
"""Render baseline metrics as a paper-ready LaTeX table.

Usage:
  uv run python scripts/analysis/print_baseline_metrics_table.py
  uv run python scripts/analysis/print_baseline_metrics_table.py --dataset signor
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.shared.evaluate import EvaluationHarness
from experiments.baselines.shared.label_utils import normalize_label

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
OURS_DIR = PROJECT_ROOT / "results/baselines/proclaim/latest"

AGGREGATED_KEYS = (
    "accuracy",
    "macro_fpr",
    "macro_fnr",
    "avg_cost_usd",
    "total_input_tokens",
    "total_output_tokens",
)


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
    RowSpec(label="", section="LLM Only"),
    RowSpec(label="Claude Sonnet 4.6", baseline="llm_only", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="Gemini 2.5 Flash", baseline="llm_only", model="vertex_ai--gemini-2.5-flash"),
    RowSpec(label="", is_rule=True),
    RowSpec(label="", section="Static Retrieval S2 Top 5"),
    RowSpec(label="Claude Sonnet 4.6", baseline="retrieval", variant="s2/top5", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="Gemini 2.5 Flash", baseline="retrieval", variant="s2/top5", model="vertex_ai--gemini-2.5-flash"),
    RowSpec(label="", is_rule=True),
    RowSpec(label="", section="Static Retrieval Web Top 5"),
    RowSpec(label="Claude Sonnet 4.6", baseline="retrieval", variant="web/top5", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="Gemini 2.5 Flash", baseline="retrieval", variant="web/top5", model="vertex_ai--gemini-2.5-flash"),
    RowSpec(label="", is_rule=True),
    RowSpec(label="", section="Adaptive Retrieval"),
    RowSpec(label="ReAct + S2", baseline="react", variant="s2", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="ReAct + Web", baseline="react", variant="web", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="FIRE", baseline="fire", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="SAFE", baseline="safe", model="anthropic--claude-sonnet-4-6"),
    RowSpec(label="OpenScholar + S2", baseline="open_scholar", model="anthropic--claude-sonnet-4-6"),
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
    parser.add_argument(
        "--cost-table",
        action="store_true",
        help="Print a separate LaTeX table with average USD cost instead of the main metrics table.",
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


def aggregate_scalar(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": float(mean), "std": float(std)}


def compute_unrounded_metrics(results: list[object]) -> dict[str, float]:
    labels = ["SUPPORT", "REFUTE", "UNCERTAIN"]

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    tn: dict[str, int] = defaultdict(int)

    correct = 0
    total = len(results)

    for result in results:
        pred = normalize_label(result.predicted_label)
        gold = normalize_label(result.gold_label)
        if pred == gold:
            correct += 1
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    for label in labels:
        tn[label] = total - tp[label] - fp[label] - fn[label]

    per_class_fpr: dict[str, float] = {}
    per_class_fnr: dict[str, float] = {}
    for label in labels:
        fp_denom = fp[label] + tn[label]
        fn_denom = fn[label] + tp[label]
        per_class_fpr[label] = fp[label] / fp_denom if fp_denom > 0 else 0.0
        per_class_fnr[label] = fn[label] / fn_denom if fn_denom > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.0
    macro_fpr = sum(per_class_fpr.values()) / len(labels) if labels else 0.0
    macro_fnr = sum(per_class_fnr.values()) / len(labels) if labels else 0.0
    total_cost_usd = sum(result.cost_usd for result in results)
    avg_cost_usd = total_cost_usd / total if total > 0 else 0.0
    total_input_tokens = sum(result.input_tokens for result in results)
    total_output_tokens = sum(result.output_tokens for result in results)

    return {
        "n": total,
        "accuracy": accuracy,
        "macro_fpr": macro_fpr,
        "macro_fnr": macro_fnr,
        "avg_cost_usd": avg_cost_usd,
        "total_input_tokens": float(total_input_tokens),
        "total_output_tokens": float(total_output_tokens),
    }


def aggregate_seed_metrics(metrics_path: Path) -> dict[str, object] | None:
    dataset = metrics_path.stem.removesuffix("_metrics")
    run_metrics: list[dict[str, object]] = []

    for jsonl_path in sorted(metrics_path.parent.glob(f"{dataset}_seed*.jsonl")):
        results = EvaluationHarness.load(jsonl_path)
        if not results:
            continue
        run_metrics.append(compute_unrounded_metrics(results))

    if len(run_metrics) < 2:
        return None

    max_examples = max(int(metric_value(metrics, "n")) for metrics in run_metrics)
    complete_runs = [metrics for metrics in run_metrics if int(metric_value(metrics, "n")) == max_examples]
    if len(complete_runs) < 2:
        return None

    aggregated: dict[str, object] = {
        "n": max_examples,
        "n_repeats": len(complete_runs),
    }
    for key in AGGREGATED_KEYS:
        aggregated[key] = aggregate_scalar([metric_value(metrics, key) for metrics in complete_runs])
    return aggregated


def collect_metrics(results_dir: Path) -> dict[tuple[str, str, str, str], dict[str, object]]:
    metrics_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for metrics_path in sorted(results_dir.rglob("*_metrics.json")):
        dataset, baseline, variant, model = parse_metric_path(metrics_path, results_dir)
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)

        aggregated_metrics = aggregate_seed_metrics(metrics_path)
        if aggregated_metrics is not None:
            metrics.update(aggregated_metrics)

        if "total_input_tokens" not in metrics or "total_output_tokens" not in metrics:
            jsonl_path = metrics_path.with_name(f"{dataset}_seed100.jsonl")
            if jsonl_path.exists():
                jsonl_metrics = compute_unrounded_metrics(EvaluationHarness.load(jsonl_path))
                metrics.setdefault("total_input_tokens", jsonl_metrics["total_input_tokens"])
                metrics.setdefault("total_output_tokens", jsonl_metrics["total_output_tokens"])

        metrics_by_key[(dataset, baseline, variant, model)] = metrics
    return metrics_by_key


def load_ours_metrics() -> dict[str, dict[str, object]]:
    ours: dict[str, dict[str, object]] = {}
    for dataset in DATASET_ORDER:
        seed_paths = sorted(OURS_DIR.glob(f"{dataset}_seed*.jsonl"))
        if not seed_paths:
            continue
        run_metrics = [compute_unrounded_metrics(EvaluationHarness.load(p)) for p in seed_paths]
        if len(run_metrics) == 1:
            ours[dataset] = run_metrics[0]
        else:
            max_n = max(int(m["n"]) for m in run_metrics)
            complete = [m for m in run_metrics if int(m["n"]) == max_n]
            aggregated: dict[str, object] = {"n": max_n, "n_repeats": len(complete)}
            for key in AGGREGATED_KEYS:
                aggregated[key] = aggregate_scalar([metric_value(m, key) for m in complete])
            ours[dataset] = aggregated
    return ours


def format_number(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.2f}"
    if bold:
        return rf"\underline{{{text}}}"
    return text


def metric_standard_error(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    n_repeats = metrics.get("n_repeats")
    if not isinstance(value, dict) or "std" not in value or n_repeats is None:
        return None
    repeats = float(n_repeats)
    if repeats <= 1:
        return None
    return float(value["std"]) / math.sqrt(repeats)


def format_number_with_se(value: float | None, se: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    value_text = format_number(value, bold=bold)
    if se is None:
        return value_text
    return rf"{value_text} {{\scriptsize$\pm$ {se:.2f}}}"


def average_total_tokens(metrics: dict[str, object]) -> float:
    total_examples = metric_value(metrics, "n")
    if total_examples <= 0:
        return 0.0
    total_input_tokens = metric_value(metrics, "total_input_tokens")
    total_output_tokens = metric_value(metrics, "total_output_tokens")
    return (total_input_tokens + total_output_tokens) / total_examples


def average_total_tokens_standard_error(metrics: dict[str, object]) -> float | None:
    total_examples = metric_value(metrics, "n")
    n_repeats = metrics.get("n_repeats")
    total_input_tokens = metrics.get("total_input_tokens")
    total_output_tokens = metrics.get("total_output_tokens")
    if (
        total_examples <= 0
        or n_repeats is None
        or not isinstance(total_input_tokens, dict)
        or not isinstance(total_output_tokens, dict)
        or "std" not in total_input_tokens
        or "std" not in total_output_tokens
    ):
        return None
    repeats = float(n_repeats)
    if repeats <= 1:
        return None
    combined_std = math.sqrt(float(total_input_tokens["std"]) ** 2 + float(total_output_tokens["std"]) ** 2)
    return combined_std / math.sqrt(repeats) / total_examples


def format_token_millions(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value / 1_000_000:.3f}"
    if bold:
        return rf"\underline{{{text}}}"
    return text


def format_token_millions_with_se(value: float | None, se: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    value_text = format_token_millions(value, bold=bold)
    if se is None:
        return value_text
    return rf"{value_text} {{\scriptsize$\pm$ {se / 1_000_000:.3f}}}"


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
            dataset_map[dataset] = {
                "accuracy": metric_value(metrics, "accuracy"),
                "accuracy_se": metric_standard_error(metrics, "accuracy"),
                "macro_fpr": metric_value(metrics, "macro_fpr"),
                "macro_fpr_se": metric_standard_error(metrics, "macro_fpr"),
                "macro_fnr": metric_value(metrics, "macro_fnr"),
                "macro_fnr_se": metric_standard_error(metrics, "macro_fnr"),
                "avg_cost_usd": metric_value(metrics, "avg_cost_usd"),
                "avg_cost_usd_se": metric_standard_error(metrics, "avg_cost_usd"),
                "avg_total_tokens": average_total_tokens(metrics),
                "avg_total_tokens_se": average_total_tokens_standard_error(metrics),
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
        fpr_values: list[float] = []
        fnr_values: list[float] = []
        cost_values: list[float] = []
        token_values: list[float] = []
        for per_dataset in row_metrics.values():
            metrics = per_dataset.get(dataset)
            if metrics is None:
                continue
            accuracy_values.append(metrics["accuracy"])
            fpr_values.append(metrics["macro_fpr"])
            fnr_values.append(metrics["macro_fnr"])
            cost_values.append(metrics["avg_cost_usd"])
            token_values.append(metrics["avg_total_tokens"])
        bests[dataset] = {
            "accuracy": max(accuracy_values) if accuracy_values else None,
            "macro_fpr": min(fpr_values) if fpr_values else None,
            "macro_fnr": min(fnr_values) if fnr_values else None,
            "avg_cost_usd": min(cost_values) if cost_values else None,
            "avg_total_tokens": min(token_values) if token_values else None,
        }
    return bests


def tabular_spec(datasets: list[str]) -> str:
    dataset_blocks = ["*{3}{c}" for _ in datasets]
    return "@{} l | " + " | ".join(dataset_blocks) + " @{}"


def cost_tabular_spec(datasets: list[str]) -> str:
    dataset_blocks = ["c" for _ in datasets]
    return "@{} l | " + " | ".join(dataset_blocks) + " @{}"


def dataset_header_line(datasets: list[str]) -> str:
    parts = [" "]
    for index, dataset in enumerate(datasets):
        suffix = "|" if index < len(datasets) - 1 else ""
        parts.append(
            rf"\multicolumn{{3}}{{c{suffix}}}{{\cellcolor{{{DATASET_HEADER_COLORS[dataset]}}}\textbf{{{DATASET_TITLES[dataset]}}}}}"
        )
    return " & ".join(parts) + r" \\"


def cost_dataset_header_line(datasets: list[str]) -> str:
    parts = [" "]
    for index, dataset in enumerate(datasets):
        suffix = "|" if index < len(datasets) - 1 else ""
        parts.append(
            rf"\multicolumn{{1}}{{c{suffix}}}{{\cellcolor{{{DATASET_HEADER_COLORS[dataset]}}}\textbf{{{DATASET_TITLES[dataset]}}}}}"
        )
    return " & ".join(parts) + r" \\" 


def column_header_line(datasets: list[str]) -> str:
    headers = [r"\textbf{Method}"]
    for _dataset in datasets:
        headers.extend([
            r"Agreement$\uparrow$",
            r"FPR$\downarrow$",
            r"FNR$\downarrow$",
        ])
    return " & ".join(headers) + r" \\"


def cost_column_header_line(datasets: list[str]) -> str:
    headers = [r"\textbf{Method}"]
    for _dataset in datasets:
        headers.append(r"Cost (USD)$\downarrow$")
    return " & ".join(headers) + r" \\" 


def format_usd(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.3f}"
    if bold:
        return rf"\underline{{{text}}}"
    return text


def format_usd_with_se(value: float | None, se: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    value_text = format_usd(value, bold=bold)
    if se is None:
        return value_text
    return rf"{value_text} {{\scriptsize$\pm$ {se:.3f}}}"


def render_table(
    datasets: list[str],
    row_metrics: dict[int, dict[str, dict[str, float] | None]],
    bests: dict[str, dict[str, float | None]],
    label: str,
) -> str:
    column_count = 1 + 3 * len(datasets)
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
                cells.extend(["--", "--", "--"])
                continue
            cells.extend([
                format_number_with_se(
                    metrics["accuracy"],
                    metrics["accuracy_se"],
                    bold=is_best(metrics["accuracy"], bests[dataset]["accuracy"]),
                ),
                format_number_with_se(
                    metrics["macro_fpr"],
                    metrics["macro_fpr_se"],
                    bold=is_best(metrics["macro_fpr"], bests[dataset]["macro_fpr"]),
                ),
                format_number_with_se(
                    metrics["macro_fnr"],
                    metrics["macro_fnr_se"],
                    bold=is_best(metrics["macro_fnr"], bests[dataset]["macro_fnr"]),
                ),
            ])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])
    return "\n".join(lines) + "\n"


def render_cost_table(
    datasets: list[str],
    row_metrics: dict[int, dict[str, dict[str, float] | None]],
    bests: dict[str, dict[str, float | None]],
) -> str:
    column_count = 1 + len(datasets)
    lines = [
        rf"\begin{{tabular}}{{{cost_tabular_spec(datasets)}}}",
        r"\toprule",
        cost_dataset_header_line(datasets),
        cost_column_header_line(datasets),
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
                cells.append("--")
                continue
            cells.append(
                format_usd_with_se(
                    metrics["avg_cost_usd"],
                    metrics["avg_cost_usd_se"],
                    bold=is_best(metrics["avg_cost_usd"], bests[dataset]["avg_cost_usd"]),
                )
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return "\n".join(lines) + "\n"


def output_filename(datasets: list[str], cost_table: bool) -> str:
    prefix = "baseline_cost" if cost_table else "baseline_metrics"
    if len(datasets) == 1:
        return f"{prefix}_{datasets[0]}.tex"
    return f"{prefix}_main.tex"


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
    report = render_cost_table(datasets, row_metrics, bests) if args.cost_table else render_table(datasets, row_metrics, bests, args.label)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(datasets, args.cost_table)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(report, end="")


if __name__ == "__main__":
    main()