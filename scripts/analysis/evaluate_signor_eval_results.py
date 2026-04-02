"""
Convert signor_eval CSV results to BaselineResult objects and run the
evaluation pipeline.

CSV schema (signor_eval_results_*.csv):
    SIGNOR_ID, ENTITYA, ENTITYB, Original_Label, Flipped_Label,
    Claim_String, Is_Flipped, Repetition, Agent_Verdict, Agent_Confidence,
    Reasoning_Snippet, Output_Directory, Input_Tokens, Output_Tokens,
    Cache_Creation_Tokens, Cache_Read_Tokens, Total_Input_Tokens, Cost_Estimate

Gold label mapping:
    Flipped_Label column is always the ground truth for the claim that was
    actually presented (SUPPORTED/WRONG/UNCERTAIN → SUPPORT/REFUTE/NEI).

Usage::

    uv run python scripts/analysis/evaluate_signor_eval_results.py \
        results/signor_eval_results_march_30.csv

    # Save metrics JSON
    uv run python scripts/analysis/evaluate_signor_eval_results.py \
        results/signor_eval_results_march_30.csv \
        --output results/signor_eval_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from experiments.baselines.shared.evaluate import EvaluationHarness
from experiments.baselines.shared.label_utils import normalize_label
from experiments.baselines.shared.verdict import BaselineResult


def csv_to_baseline_results(
    csv_path: Path,
    baseline_name: str = "evidence_programming",
    dataset: str = "SIGNOR",
) -> list[BaselineResult]:
    """Convert a signor_eval CSV file into a list of BaselineResult objects.

    Args:
        csv_path: Path to the CSV file produced by run_signor_eval.py.
        baseline_name: Value to set in the ``baseline_name`` field.
        dataset: Value to set in the ``dataset`` field.

    Returns:
        List of BaselineResult objects ready for EvaluationHarness.metrics().
    """
    results: list[BaselineResult] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signor_id = row["SIGNOR_ID"]
            is_flipped = row["Is_Flipped"].strip().lower() in ("true", "1", "yes")
            repetition = row.get("Repetition", "1").strip()

            # Unique claim identifier that distinguishes flipped vs. original
            claim_id = f"{signor_id}_flip_{is_flipped}_rep_{repetition}"

            # Flipped_Label is the ground truth for the claim as presented
            gold_label = normalize_label(row["Flipped_Label"])
            predicted_label = normalize_label(row["Agent_Verdict"])

            results.append(
                BaselineResult(
                    claim_id=claim_id,
                    claim=row["Claim_String"],
                    gold_label=gold_label,
                    predicted_label=predicted_label,
                    confidence=float(row.get("Agent_Confidence") or 0.0),
                    reasoning=row.get("Reasoning_Snippet", ""),
                    input_tokens=int(row.get("Total_Input_Tokens") or 0),
                    output_tokens=int(row.get("Output_Tokens") or 0),
                    cost_usd=float(row.get("Cost_Estimate") or 0.0),
                    baseline_name=baseline_name,
                    dataset=dataset,
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate signor_eval CSV results using the shared evaluation pipeline."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the signor_eval results CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--baseline-name",
        default="evidence_programming",
        help="Baseline name tag written into each result (default: evidence_programming).",
    )
    args = parser.parse_args()

    results = csv_to_baseline_results(args.csv_path, baseline_name=args.baseline_name)
    metrics = EvaluationHarness.metrics(results)

    print(json.dumps(metrics, indent=2))

    if args.output:
        EvaluationHarness.save_metrics(metrics, args.output)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
