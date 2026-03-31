#!/usr/bin/env python3
"""
Run baselines on pre-processed dataset CSVs from the datasets directory.

Expects CSVs with at minimum: id, claim, label columns.

For stochastic baselines (random), multiple repeats are run and metrics are
aggregated as mean ± std across repeats. Each repeat uses seed + repeat_index
so results are reproducible.

Usage
-----
# Random baseline, 10 repeats, seed=100 (defaults):
uv run python experiments/run_baselines_datasets.py \\
    --datasets-dir /home/ail/workspace/connectomeDB_data/datasets \\
    --datasets signor connectomedb \\
    --baseline random

# Custom seed / repeats:
uv run python experiments/run_baselines_datasets.py \\
    --datasets-dir /home/ail/workspace/connectomeDB_data/datasets \\
    --datasets signor connectomedb \\
    --baseline random --seed 42 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))  # for experiments/baselines/

from baselines.shared.evaluate import EvaluationHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baselines_datasets")


def load_claims(csv_path: Path) -> list[dict]:
    """Load a pre-processed dataset CSV into a list of claim dicts."""
    df = pd.read_csv(csv_path)
    claims = []
    for _, row in df.iterrows():
        claims.append(
            {
                "claim_id": str(row["id"]),
                "claim": str(row["claim"]),
                "gold_label": str(row["label"]),
                "dataset": str(row.get("dataset", csv_path.stem)),
            }
        )
    return claims


def build_baseline(name: str, seed: int):
    if name == "random":
        from baselines.random_baseline import RandomBaseline
        return RandomBaseline(seed=seed)
    else:
        raise ValueError(f"Unknown baseline: {name!r}")


def _mean_std(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {"mean": round(mean, 4), "std": round(math.sqrt(variance), 4)}


def aggregate_metrics(all_runs: list[dict]) -> dict:
    """Aggregate a list of per-repeat metric dicts into mean ± std."""
    scalar_keys = ["accuracy", "macro_f1", "binary_f1", "binary_precision", "binary_recall"]
    per_class_labels = ["SUPPORT", "REFUTE", "NEI"]
    per_class_subkeys = ["precision", "recall", "f1"]

    agg: dict = {
        "n": all_runs[0]["n"],
        "n_repeats": len(all_runs),
    }

    for key in scalar_keys:
        agg[key] = _mean_std([r[key] for r in all_runs])

    agg["per_class"] = {}
    for label in per_class_labels:
        agg["per_class"][label] = {}
        for sub in per_class_subkeys:
            agg["per_class"][label][sub] = _mean_std(
                [r["per_class"][label][sub] for r in all_runs]
            )

    agg["total_cost_usd"] = {"mean": 0.0, "std": 0.0}
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--datasets-dir",
        default="/home/ail/workspace/connectomeDB_data/datasets",
        help="Directory containing dataset CSV files.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["signor", "connectomedb"],
        help="Dataset names (without .csv extension).",
    )
    p.add_argument("--baseline", default="random", choices=["random"], help="Baseline to run.")
    p.add_argument("--seed", type=int, default=100, help="Base random seed. Each repeat i uses seed+i.")
    p.add_argument("--repeats", type=int, default=10, help="Number of independent repeats (for stochastic baselines).")
    p.add_argument(
        "--output-dir",
        default="results/baselines",
        help="Directory to save results.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets_dir = Path(args.datasets_dir)
    output_dir = PROJECT_ROOT / args.output_dir

    logger.info("Baseline: %s  repeats=%d  base_seed=%d", args.baseline, args.repeats, args.seed)

    all_metrics: dict[str, dict] = {}

    for dataset_name in args.datasets:
        csv_path = datasets_dir / f"{dataset_name}.csv"
        if not csv_path.exists():
            logger.warning("CSV not found, skipping: %s", csv_path)
            continue

        logger.info("=== Dataset: %s ===", dataset_name)
        claims = load_claims(csv_path)
        logger.info("  Loaded %d claims", len(claims))

        repeat_metrics: list[dict] = []

        for rep in range(args.repeats):
            seed = args.seed + rep
            baseline = build_baseline(args.baseline, seed)
            harness = EvaluationHarness(baseline, dataset_name=dataset_name)
            results = harness.run(claims)
            m = EvaluationHarness.metrics(results)
            repeat_metrics.append(m)
            logger.info(
                "  [repeat %d/%d seed=%d]  accuracy=%.4f  macro_f1=%.4f  binary_f1=%.4f",
                rep + 1, args.repeats, seed,
                m["accuracy"], m["macro_f1"], m["binary_f1"],
            )

            # Save per-claim results for this repeat
            out_path = output_dir / args.baseline / f"{dataset_name}_seed{seed}.jsonl"
            EvaluationHarness.save(results, out_path)

        agg = aggregate_metrics(repeat_metrics)

        # Save aggregated metrics
        metrics_path = output_dir / args.baseline / f"{dataset_name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(agg, f, indent=2)

        logger.info(
            "  AGGREGATED (%d repeats): accuracy=%.4f±%.4f  macro_f1=%.4f±%.4f  binary_f1=%.4f±%.4f",
            args.repeats,
            agg["accuracy"]["mean"], agg["accuracy"]["std"],
            agg["macro_f1"]["mean"], agg["macro_f1"]["std"],
            agg["binary_f1"]["mean"], agg["binary_f1"]["std"],
        )
        logger.info("  Metrics saved to %s", metrics_path)

        all_metrics[dataset_name] = agg

    # Print summary
    print("\n=== SUMMARY ===")
    for dataset_name, agg in all_metrics.items():
        print(f"\n[{dataset_name.upper()}]  n={agg['n']}  repeats={agg['n_repeats']}")
        for key in ["accuracy", "macro_f1", "binary_f1"]:
            m = agg[key]
            print(f"  {key:20s}: {m['mean']:.4f} ± {m['std']:.4f}")
        print("  Per-class (mean ± std):")
        for label, prf in agg["per_class"].items():
            p, r, f1 = prf["precision"], prf["recall"], prf["f1"]
            print(f"    {label:8s}  P={p['mean']:.3f}±{p['std']:.3f}  R={r['mean']:.3f}±{r['std']:.3f}  F1={f1['mean']:.3f}±{f1['std']:.3f}")


if __name__ == "__main__":
    main()
