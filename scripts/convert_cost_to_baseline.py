#!/usr/bin/env python3
"""
Convert a results.csv to use baseline-style cost calculation.

The original Cost_Estimate uses cache-aware pricing:
    cost = input*in_p + cache_creation*in_p*1.25 + cache_read*in_p*0.1 + output*out_p

The baseline CostTracker uses only input+output tokens (no cache):
    cost = input*in_p + output*out_p

The token values themselves are unchanged — Input_Tokens in the CSV already
equals LiteLLM's prompt_tokens (non-cache input only), which is exactly what
the new CostTracker.record() would receive.

Output: <input_dir>/results_baseline_cost.csv  (original file untouched)
  - Cost_Estimate     overwritten with baseline formula
  - Cost_Estimate_Old added (original cache-aware value)

Usage:
    python scripts/convert_cost_to_baseline.py <results.csv> [--model MODEL]
    python scripts/convert_cost_to_baseline.py results/signor_direct_eval_20260427_221617/results.csv
    python scripts/convert_cost_to_baseline.py results/connectomedb_eval_20260424_171117/results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6":           (5.00, 25.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-haiku-4-5":          (1.00,  5.00),
    "claude-haiku-4-5-20251001": (1.00,  5.00),
    "gemini-3.1-pro-preview":    (2.00, 12.00),
    "gemini-3-pro-preview":      (2.00, 12.00),
    "gemini-2.5-pro":            (1.25, 10.00),
    "gemini-2.5-flash":          (0.30,  2.50),
}
FALLBACK_PRICING = (3.00, 15.00)


def convert(csv_path: Path, model: str, out_path: Path) -> None:
    in_price, out_price = PRICING.get(model, FALLBACK_PRICING)
    print(f"Model: {model}  |  pricing: in=${in_price}/out=${out_price} per 1M tokens")

    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    if not rows:
        print("No rows found.")
        return

    missing = {"Input_Tokens", "Output_Tokens", "Cost_Estimate"} - set(rows[0].keys())
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    total_old = 0.0
    total_new = 0.0

    for row in rows:
        in_tok  = int(row.get("Input_Tokens")  or 0)
        out_tok = int(row.get("Output_Tokens") or 0)
        old_cost = float(row.get("Cost_Estimate") or 0)

        new_cost = round(
            (in_tok  / 1_000_000) * in_price
            + (out_tok / 1_000_000) * out_price,
            6,
        )
        row["Cost_Estimate_Old"] = f"{old_cost:.6f}"
        row["Cost_Estimate"] = f"{new_cost:.6f}"
        total_old += old_cost
        total_new += new_cost

    # Insert Cost_Estimate_Old right after Cost_Estimate
    fieldnames = list(rows[0].keys())
    fieldnames.remove("Cost_Estimate_Old")
    fieldnames.insert(fieldnames.index("Cost_Estimate") + 1, "Cost_Estimate_Old")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"Rows: {n}")
    print(f"Total cost (old cache-aware): ${total_old:.4f}")
    print(f"Total cost (new baseline):    ${total_new:.4f}")
    print(f"Difference:                   ${total_old - total_new:+.4f}  ({(total_old/total_new - 1)*100:+.1f}%)")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_path", help="Path to input results.csv")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model key for pricing lookup (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <input_dir>/results_baseline_cost.csv)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else csv_path.parent / "results_baseline_cost.csv"
    convert(csv_path, args.model, out_path)


if __name__ == "__main__":
    main()
