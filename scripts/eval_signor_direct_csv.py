#!/usr/bin/env python3
"""Evaluate a signor_direct results.csv using EvaluationHarness.metrics().

Usage
-----
uv run python scripts/eval_signor_direct_csv.py results/signor_direct_eval_20260503_151018/results.csv
uv run python scripts/eval_signor_direct_csv.py results/signor_direct_eval_20260427_221617/results_baseline_cost.csv --save
uv run python scripts/eval_signor_direct_csv.py results/... --model claude-haiku-4-5
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.baselines.shared.cost_tracker import CostTracker
from experiments.baselines.shared.evaluate import EvaluationHarness
from experiments.baselines.shared.label_utils import normalize_label
from experiments.baselines.shared.verdict import BaselineResult


def load_results(csv_path: Path) -> list[BaselineResult]:
    results = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            results.append(
                BaselineResult(
                    claim_id=row["SIGNOR_ID"],
                    claim=row["Claim_String"],
                    gold_label=normalize_label(row["Flipped_Label"]),
                    predicted_label=normalize_label(row["Agent_Verdict"]),
                    confidence=float(row["Agent_Confidence"] or 0),
                    reasoning=row["Reasoning_Snippet"],
                    input_tokens=int(row["Total_Input_Tokens"] or 0),
                    output_tokens=int(row["Output_Tokens"] or 0),
                    cost_usd=float(row["Cost_Estimate"] or 0),
                    baseline_name="signor_direct_eval",
                    dataset="signor_direct",
                )
            )
    return results


def load_token_breakdown(csv_path: Path) -> dict:
    """Aggregate token counts with cache breakdown from CSV columns."""
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


def compute_costs(tokens: dict, in_price: float, out_price: float) -> tuple[float, float]:
    """Return (cost_with_cache, cost_no_cache) using the same pricing for fair comparison."""
    inp = tokens["input_tokens"]
    write = tokens["cache_creation_tokens"]
    read = tokens["cache_read_tokens"]
    out = tokens["output_tokens"]
    cost_with = (
        (inp / 1_000_000) * in_price
        + (write / 1_000_000) * in_price * 1.25
        + (read / 1_000_000) * in_price * 0.1
        + (out / 1_000_000) * out_price
    )
    cost_no = (
        ((inp + write + read) / 1_000_000) * in_price
        + (out / 1_000_000) * out_price
    )
    return round(cost_with, 4), round(cost_no, 4)


def print_metrics(
    metrics: dict,
    n_flipped: int,
    n_total: int,
    tokens: dict,
    cost_with_cache: float,
    cost_no_cache: float,
    in_price: float = 3.0,
    out_price: float = 15.0,
    price_label: str = "",
) -> None:
    n = metrics["n"]
    print(f"\n{'='*60}")
    print(f"  SignorDirect Evaluation  (n={n}, flipped={n_flipped}/{n_total})")
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
    print(f"  Cache write        : {tokens['cache_creation_tokens']:>12,}  (\u00d71.25 price)")
    print(f"  Cache read         : {tokens['cache_read_tokens']:>12,}  (\u00d70.10 price)")
    print(f"  Output             : {tokens['output_tokens']:>12,}")
    print()
    print(f"  {'─'*40}")
    label_str = f"  [{price_label}  in=${in_price}/M  out=${out_price}/M]" if price_label else ""
    print(f"  Cost{label_str}")
    print(f"  {'─'*40}")
    print(f"  With cache pricing : ${cost_with_cache:>10.4f}   (avg ${cost_with_cache/n:.6f})")
    print(f"  No cache (full $/M): ${cost_no_cache:>10.4f}   (avg ${cost_no_cache/n:.6f})")
    print()
    print(f"  {'Class':<12} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'FNR':>6}  TP  FP  FN  TN")
    print(f"  {'-'*70}")
    for label, pc in metrics["per_class"].items():
        print(
            f"  {label:<12} {pc['precision']:>6.4f} {pc['recall']:>6.4f} {pc['f1']:>6.4f} "
            f"{pc['fpr']:>6.4f} {pc['fnr']:>6.4f}  {pc['tp']:>3} {pc['fp']:>3} {pc['fn']:>3} {pc['tn']:>3}"
        )
    print(f"{'='*60}\n")


def resolve_pricing(model: str | None, in_price: float | None, out_price: float | None) -> tuple[float, float, str]:
    """Return (in_price, out_price, label) from model name or explicit prices."""
    if in_price is not None and out_price is not None:
        return in_price, out_price, "custom"
    if model:
        model_key = model.split("/", 1)[-1] if "/" in model else model
        pricing = CostTracker.DEFAULT_PRICING.get(model_key, CostTracker.FALLBACK_PRICING)
        ip, op = pricing
        return (in_price or ip), (out_price or op), model_key
    # default: sonnet-4-6
    ip, op = CostTracker.DEFAULT_PRICING["claude-sonnet-4-6"]
    return ip, op, "claude-sonnet-4-6 (default)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a signor_direct results.csv.")
    parser.add_argument("csv_path", help="Path to results.csv")
    parser.add_argument("--save", action="store_true", help="Save metrics.json alongside the CSV")
    parser.add_argument("--model", default=None,
                        help="Model name for pricing lookup (e.g. claude-haiku-4-5, claude-sonnet-4-6)")
    parser.add_argument("--in-price", type=float, default=None,
                        help="Override input token price USD/1M")
    parser.add_argument("--out-price", type=float, default=None,
                        help="Override output token price USD/1M")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path) as f:
        all_rows = list(csv.DictReader(f))
    n_total = len(all_rows)
    n_flipped = sum(1 for r in all_rows if r.get("Is_Flipped") == "True")

    in_price, out_price, price_label = resolve_pricing(args.model, args.in_price, args.out_price)

    results = load_results(csv_path)
    metrics = EvaluationHarness.metrics(results)
    tokens = load_token_breakdown(csv_path)
    cost_with_cache, cost_no_cache = compute_costs(tokens, in_price, out_price)

    print_metrics(metrics, n_flipped=n_flipped, n_total=n_total,
                  tokens=tokens, cost_with_cache=cost_with_cache, cost_no_cache=cost_no_cache,
                  in_price=in_price, out_price=out_price, price_label=price_label)

    if args.save:
        out = {**metrics, "token_breakdown": tokens,
               "cost_with_cache": cost_with_cache, "cost_no_cache": cost_no_cache,
               "pricing": {"model": price_label, "in_price": in_price, "out_price": out_price}}
        out_path = csv_path.parent / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
