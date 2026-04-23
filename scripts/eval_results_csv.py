#!/usr/bin/env python3
"""Evaluate a connectomedb results.csv using EvaluationHarness.metrics().

Usage
-----
uv run python scripts/eval_results_csv.py results/connectomedb_eval_20260420_012842/results.csv
uv run python scripts/eval_results_csv.py results/connectomedb_eval_20260420_012842/results.csv --save
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.baselines.shared.evaluate import EvaluationHarness
from experiments.baselines.shared.label_utils import normalize_label
from experiments.baselines.shared.verdict import BaselineResult


def load_results(csv_path: Path) -> list[BaselineResult]:
    results = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            results.append(
                BaselineResult(
                    claim_id=row["CDB_ID"],
                    claim=row["Claim_String"],
                    gold_label=normalize_label(row["Label"]),
                    predicted_label=normalize_label(row["Agent_Verdict"]),
                    confidence=float(row["Agent_Confidence"] or 0),
                    reasoning=row["Reasoning_Snippet"],
                    input_tokens=int(row["Input_Tokens"] or 0),
                    output_tokens=int(row["Output_Tokens"] or 0),
                    cost_usd=float(row["Cost_Estimate"] or 0),
                    baseline_name="connectomedb_eval",
                    dataset="connectomedb",
                )
            )
    return results


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*52}")
    print(f"  ConnectomeDB Evaluation  (n={metrics['n']})")
    print(f"{'='*52}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"  Macro FPR  : {metrics['macro_fpr']:.4f}")
    print(f"  Macro FNR  : {metrics['macro_fnr']:.4f}")
    print(f"  Wtd FPR    : {metrics['weighted_fpr']:.4f}")
    print(f"  Wtd FNR    : {metrics['weighted_fnr']:.4f}")
    print(f"  Total cost : ${metrics['total_cost_usd']:.4f}")
    print(f"  Avg cost   : ${metrics['avg_cost_usd']:.6f}")
    print(f"  Input tok  : {metrics['total_input_tokens']:,}")
    print(f"  Output tok : {metrics['total_output_tokens']:,}")
    print()
    print(f"  {'Class':<12} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'FNR':>6}  TP  FP  FN  TN")
    print(f"  {'-'*70}")
    for label, pc in metrics["per_class"].items():
        print(
            f"  {label:<12} {pc['precision']:>6.4f} {pc['recall']:>6.4f} {pc['f1']:>6.4f} "
            f"{pc['fpr']:>6.4f} {pc['fnr']:>6.4f}  {pc['tp']:>3} {pc['fp']:>3} {pc['fn']:>3} {pc['tn']:>3}"
        )
    print(f"{'='*52}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a connectomedb results.csv.")
    parser.add_argument("csv_path", help="Path to results.csv")
    parser.add_argument(
        "--save", action="store_true", help="Save metrics.json alongside the CSV"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    results = load_results(csv_path)
    metrics = EvaluationHarness.metrics(results)
    print_metrics(metrics)

    if args.save:
        out_path = csv_path.parent / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
