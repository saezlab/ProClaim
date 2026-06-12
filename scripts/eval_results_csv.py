#!/usr/bin/env python3
"""Evaluate a ProClaim results.csv using EvaluationHarness.metrics().

Usage
-----
uv run python scripts/eval_results_csv.py results/connectomedb_eval_20260424_171117/results.csv
uv run python scripts/eval_results_csv.py results/signor_direct_eval_20260605_163241/results.csv
uv run python scripts/eval_results_csv.py results/.../results.csv --save
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


SCHEMAS = {
    "connectomedb": {
        "id_column": "CDB_ID",
        "gold_column": "Label",
        "input_tokens_column": "Input_Tokens",
        "dataset": "connectomedb",
        "baseline_name": "connectomedb_eval",
        "title": "ConnectomeDB",
        "flipped_column": None,
    },
    "signor_direct": {
        "id_column": "SIGNOR_ID",
        "gold_column": "Flipped_Label",
        "input_tokens_column": "Total_Input_Tokens",
        "dataset": "signor_direct",
        "baseline_name": "signor_direct_eval",
        "title": "SignorDirect",
        "flipped_column": "Is_Flipped",
    },
}


def detect_schema(fieldnames: list[str] | None) -> dict:
    if not fieldnames:
        raise ValueError("CSV is missing a header row.")
    field_set = set(fieldnames)
    if "CDB_ID" in field_set and "Label" in field_set:
        return SCHEMAS["connectomedb"]
    if "SIGNOR_ID" in field_set and "Flipped_Label" in field_set:
        return SCHEMAS["signor_direct"]
    raise ValueError(
        "Unsupported results CSV schema. Expected ConnectomeDB or signor_direct columns."
    )


def load_results(csv_path: Path, schema: dict) -> list[BaselineResult]:
    results = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            results.append(
                BaselineResult(
                    claim_id=row[schema["id_column"]],
                    claim=row["Claim_String"],
                    gold_label=normalize_label(row[schema["gold_column"]]),
                    predicted_label=normalize_label(row["Agent_Verdict"]),
                    confidence=float(row["Agent_Confidence"] or 0),
                    reasoning=row["Reasoning_Snippet"],
                    input_tokens=int(row[schema["input_tokens_column"]] or 0),
                    output_tokens=int(row["Output_Tokens"] or 0),
                    cost_usd=float(row["Cost_Estimate"] or 0),
                    baseline_name=schema["baseline_name"],
                    dataset=schema["dataset"],
                )
            )
    return results


def load_token_breakdown(csv_path: Path) -> dict:
    totals = {
        "input_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "total_input_tokens": 0,
        "output_tokens": 0,
    }
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            totals["input_tokens"] += int(row.get("Input_Tokens") or 0)
            totals["cache_creation_tokens"] += int(row.get("Cache_Creation_Tokens") or 0)
            totals["cache_read_tokens"] += int(row.get("Cache_Read_Tokens") or 0)
            totals["total_input_tokens"] += int(row.get("Total_Input_Tokens") or 0)
            totals["output_tokens"] += int(row.get("Output_Tokens") or 0)
    return totals


def count_flipped(csv_path: Path, flipped_column: str | None) -> int | None:
    if not flipped_column:
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return sum(1 for row in rows if row.get(flipped_column) == "True")


def print_metrics(metrics: dict, schema: dict, tokens: dict, n_flipped: int | None = None) -> None:
    title = schema["title"]
    header = f"  {title} Evaluation  (n={metrics['n']})"
    if n_flipped is not None:
        header = f"  {title} Evaluation  (n={metrics['n']}, flipped={n_flipped}/{metrics['n']})"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"  Macro FPR  : {metrics['macro_fpr']:.4f}")
    print(f"  Macro FNR  : {metrics['macro_fnr']:.4f}")
    print(f"  Wtd FPR    : {metrics['weighted_fpr']:.4f}")
    print(f"  Wtd FNR    : {metrics['weighted_fnr']:.4f}")
    print()
    print(f"  {'─'*40}")
    print(f"  Tokens")
    print(f"  {'─'*40}")
    print(f"  Input (non-cache)  : {tokens['input_tokens']:>12,}")
    print(f"  Cache write        : {tokens['cache_creation_tokens']:>12,}")
    print(f"  Cache read         : {tokens['cache_read_tokens']:>12,}")
    print(f"  Total input        : {tokens['total_input_tokens']:>12,}")
    print(f"  Output             : {tokens['output_tokens']:>12,}")
    print()
    print(f"  {'─'*40}")
    print(f"  Cost")
    print(f"  {'─'*40}")
    print(f"  From Cost_Estimate : ${metrics['total_cost_usd']:>10.4f}   (avg ${metrics['avg_cost_usd']:.6f})")
    print()
    print(f"  {'Class':<12} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'FNR':>6}  TP  FP  FN  TN")
    print(f"  {'-'*70}")
    for label, pc in metrics["per_class"].items():
        print(
            f"  {label:<12} {pc['precision']:>6.4f} {pc['recall']:>6.4f} {pc['f1']:>6.4f} "
            f"{pc['fpr']:>6.4f} {pc['fnr']:>6.4f}  {pc['tp']:>3} {pc['fp']:>3} {pc['fn']:>3} {pc['tn']:>3}"
        )
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ProClaim results.csv.")
    parser.add_argument("csv_path", help="Path to results.csv")
    parser.add_argument(
        "--save", action="store_true", help="Save metrics.json alongside the CSV"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        schema = detect_schema(reader.fieldnames)

    results = load_results(csv_path, schema)
    metrics = EvaluationHarness.metrics(results)
    tokens = load_token_breakdown(csv_path)
    n_flipped = count_flipped(csv_path, schema["flipped_column"])
    print_metrics(metrics, schema=schema, tokens=tokens, n_flipped=n_flipped)

    if args.save:
        out = {**metrics, "token_breakdown": tokens}
        out_path = csv_path.parent / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
