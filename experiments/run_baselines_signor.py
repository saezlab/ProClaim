#!/usr/bin/env python3
"""
Baseline evaluation on the SIGNOR ground-truth dataset.

Supports running the ``random`` and ``llm_only`` baselines (and more as they
are implemented), with optional flip-variant generation following the logic in
``run_signor_eval.py``.

Usage examples
--------------
# Random baseline, all 66 edges, no flips:
uv run python experiments/run_baselines_signor.py --baseline random

# LLM-only baseline, both forward and flipped variants:
uv run python experiments/run_baselines_signor.py --baseline llm_only --flip

# Limit rows for a quick smoke test:
uv run python experiments/run_baselines_signor.py --baseline llm_only --limit 5

# Custom model and output dir:
uv run python experiments/run_baselines_signor.py \\
    --baseline llm_only \\
    --model glm-4-plus \\
    --base-url https://api.z.ai/api/paas/v4/ \\
    --output-dir results/baselines/llm_only_glm4
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from the project root via `uv run python`
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))  # for experiments/baselines/

from baselines.shared.evaluate import EvaluationHarness
from baselines.shared.label_utils import normalize_label

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baselines_signor")

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------

DEFAULT_INPUT_GLOB = "/hps/nobackup/saezrodriguez/shared_datasets/signor*/ground_truth.csv"

# ---------------------------------------------------------------------------
# Claim construction — mirrors run_signor_eval.py exactly
# ---------------------------------------------------------------------------

_POSITIVE_EFFECTS = {
    "up-regulates",
    "up-regulates activity",
    "up-regulates quantity",
    "up-regulates quantity by expression",
}
_NEGATIVE_EFFECTS = {
    "down-regulates",
    "down-regulates activity",
    "down-regulates quantity by destabilization",
}


def construct_signor_claim(source: str, target: str, interaction: str, flip: bool = False) -> str:
    """Build the natural-language claim string for a SIGNOR edge.

    Flip logic (same as run_signor_eval.py):
      - Only positive (activation) edges are flipped to inhibition.
      - Down-regulating and non-directional edges are never flipped.
    """
    is_positive = interaction in _POSITIVE_EFFECTS
    is_negative = interaction in _NEGATIVE_EFFECTS

    if flip and is_positive:
        is_positive = False
        is_negative = True

    if is_positive:
        return (
            f"{source} directly activates {target} "
            f"(either through post-translational modification, complex formation, "
            f"stabilization, or regulation of expression)."
        )
    elif is_negative:
        return (
            f"{source} directly inhibits {target} "
            f"(either through post-translational modification, complex formation, "
            f"destabilization, or regulation of expression)."
        )
    else:
        return f"{source} directly interacts with {target} (e.g., physical binding)."


def get_flipped_label(original_label: str, flip: bool) -> str:
    if not flip:
        return original_label
    mapped = original_label.upper()
    if mapped == "SUPPORTED":
        return "WRONG"
    elif mapped == "WRONG":
        return "SUPPORTED"
    return original_label  # UNCERTAIN stays UNCERTAIN


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_signor_claims(
    csv_path: str,
    include_flipped: bool = False,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Load SIGNOR ground truth CSV and build a list of claim dicts.

    Each dict has: claim_id, claim, gold_label, metadata (entitya, entityb,
    effect, is_flipped).
    """
    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.head(limit)

    claims: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sid = str(row.get("SIGNOR_ID", f"ROW_{_}"))
        entity_a = str(row.get("ENTITYA", "UnknownA"))
        entity_b = str(row.get("ENTITYB", "UnknownB"))
        effect = str(row.get("EFFECT", "unknown"))
        orig_label = str(row.get("Label", "UNCERTAIN"))

        flip_values = [False, True] if include_flipped else [False]
        for flip in flip_values:
            # Only generate a flipped variant for positive edges
            if flip and effect not in _POSITIVE_EFFECTS:
                continue
            claim_str = construct_signor_claim(entity_a, entity_b, effect, flip=flip)
            label = get_flipped_label(orig_label, flip=flip)
            claim_id = f"{sid}_flip={flip}"
            claims.append(
                {
                    "claim_id": claim_id,
                    "claim": claim_str,
                    "gold_label": label,
                    "signor_id": sid,
                    "entitya": entity_a,
                    "entityb": entity_b,
                    "effect": effect,
                    "is_flipped": flip,
                }
            )
    return claims


# ---------------------------------------------------------------------------
# Baseline factory
# ---------------------------------------------------------------------------

def build_baseline(name: str, args: argparse.Namespace):
    """Instantiate the requested baseline."""
    if name == "random":
        from baselines.random_baseline import RandomBaseline
        return RandomBaseline(seed=args.seed)

    elif name == "llm_only":
        from baselines.llm_only import LLMOnly
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=args.model,
            base_url=args.base_url,
            temperature=0.0,  # deterministic
        )
        return LLMOnly(llm=llm)

    else:
        raise ValueError(f"Unknown baseline: {name!r}. Supported: random, llm_only")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baselines on the SIGNOR ground-truth dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["random", "llm_only"],
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--input-csv",
        default="",
        help="Path to SIGNOR ground_truth.csv. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Include flipped variants for positive edges (doubles SIGNOR to ~110 claims).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of CSV rows to process (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used by RandomBaseline).",
    )
    parser.add_argument(
        "--model",
        default="glm-4-plus",
        help="LLM model identifier (used by llm_only baseline).",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.z.ai/api/paas/v4/",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for results (defaults to results/baselines/<baseline>_signor/).",
    )
    args = parser.parse_args()

    # ---- Locate dataset ----
    csv_path = args.input_csv
    if not csv_path:
        matches = glob.glob(DEFAULT_INPUT_GLOB)
        if not matches:
            raise FileNotFoundError(
                f"Could not find SIGNOR ground truth at: {DEFAULT_INPUT_GLOB}"
            )
        csv_path = matches[0]
    logger.info("Loading SIGNOR claims from %s", csv_path)

    claims = load_signor_claims(csv_path, include_flipped=args.flip, limit=args.limit)
    logger.info("Loaded %d claim variants (flip=%s)", len(claims), args.flip)

    # ---- Output directory ----
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        tag = args.model.replace("/", "_") if args.baseline == "llm_only" else ""
        folder = f"{args.baseline}_signor" + (f"_{tag}" if tag else "")
        out_dir = PROJECT_ROOT / "results" / "baselines" / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build and run baseline ----
    baseline = build_baseline(args.baseline, args)
    harness = EvaluationHarness(baseline, dataset_name="SIGNOR")

    logger.info("Running baseline: %s", args.baseline)
    results = harness.run(claims)

    # ---- Save results ----
    results_path = out_dir / "results.jsonl"
    harness.save(results, results_path)

    metrics = harness.metrics(results)
    metrics_path = out_dir / "metrics.json"
    harness.save_metrics(metrics, metrics_path)

    # ---- Pretty-print summary ----
    logger.info("=" * 60)
    logger.info("Baseline: %s | Dataset: SIGNOR | N=%d", args.baseline, metrics["n"])
    logger.info("Accuracy:     %.4f", metrics["accuracy"])
    logger.info("Macro-F1:     %.4f", metrics["macro_f1"])
    logger.info("Binary-F1:    %.4f  (SUPPORT vs REFUTE, excl NEI)", metrics["binary_f1"])
    logger.info("Per-class F1: SUPPORT=%.4f  REFUTE=%.4f  NEI=%.4f",
                metrics["per_class"]["SUPPORT"]["f1"],
                metrics["per_class"]["REFUTE"]["f1"],
                metrics["per_class"]["NEI"]["f1"])
    logger.info("Total cost:   $%.6f (%d input / %d output tokens)",
                metrics["total_cost_usd"],
                metrics["total_input_tokens"],
                metrics["total_output_tokens"])
    logger.info("Results:      %s", results_path)
    logger.info("Metrics:      %s", metrics_path)

    # Also save a human-readable summary next to the results
    summary = {
        "baseline": args.baseline,
        "model": getattr(args, "model", ""),
        "dataset": "SIGNOR",
        "n_claims": len(claims),
        "include_flipped": args.flip,
        "seed": args.seed,
        **metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
