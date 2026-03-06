"""
Relabel negative_conflict samples based on the conflict ratio threshold τ.

For each negative_conflict sample:
- Compute r_minority = min(N_sup, N_con) / (N_sup + N_con)
- If r_minority >= τ → keep as negative_conflict (label 0)
- If r_minority < τ → relabel to positive_support or positive_contradict (label 1)
  based on majority stance

Uses ground truth SciFact labels (not NLI-predicted ratios).

Usage:
    python scripts/sufficiency_classifier/relabel_by_threshold.py \
        --input data/classifier_train_data_with_labels.json \
        --output data/ablation/tau_0.20.json \
        --threshold 0.20 \
        --verbose
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_conflict_ratio(sample: Dict[str, Any]) -> Tuple[float, int, int]:
    """
    Compute the minority conflict ratio from ground truth SciFact labels.

    Returns:
        r_minority: min(N_sup, N_con) / (N_sup + N_con)
        N_sup: number of supporting papers (ground truth)
        N_con: number of contradicting papers (ground truth)
    """
    # Use ground truth counts from SciFact labels
    N_sup = sample.get("n_support", 0)
    N_con = sample.get("n_contradict", 0)

    # Handle edge case
    if N_sup + N_con == 0:
        return 0.0, N_sup, N_con

    r_minority = min(N_sup, N_con) / (N_sup + N_con)

    return r_minority, N_sup, N_con


def relabel_sample(sample: Dict[str, Any], threshold: float, verbose: bool = False) -> Dict[str, Any]:
    """
    Apply threshold-based relabeling to a single sample.

    Args:
        sample: Original sample dictionary
        threshold: τ threshold value
        verbose: Whether to log relabeling decisions

    Returns:
        Modified sample with potentially updated pool_type and target_y
    """
    # Create a copy to avoid modifying original
    new_sample = sample.copy()

    pool_type = sample.get("pool_type", "")

    # Only process negative_conflict samples
    if pool_type != "negative_conflict":
        return new_sample

    # Skip samples without SciFact labels
    if "n_support" not in sample or "n_contradict" not in sample:
        if verbose:
            logging.warning(
                f"SKIP: claim_id={sample.get('claim_id')}, "
                f"num_papers={sample.get('num_papers')} - No SciFact labels available"
            )
        return new_sample

    # Compute conflict ratio
    r_minority, N_sup, N_con = compute_conflict_ratio(sample)

    # Apply threshold logic
    if r_minority >= threshold:
        # Keep as negative_conflict
        new_sample["pool_type"] = "negative_conflict"
        new_sample["target_y"] = 0

        if verbose:
            logging.info(
                f"KEEP: claim_id={sample.get('claim_id')}, "
                f"num_papers={sample.get('num_papers')}, "
                f"r_minority={r_minority:.3f} >= {threshold:.3f}, "
                f"N_sup={N_sup}, N_con={N_con} → negative_conflict (label 0)"
            )
    else:
        # Relabel based on majority stance
        if N_sup > N_con:
            new_pool_type = "positive_support"
        elif N_con > N_sup:
            new_pool_type = "positive_contradict"
        else:
            # Edge case: N_sup == N_con, but r_minority < threshold
            # This shouldn't happen if threshold < 0.5
            new_pool_type = "positive_support"  # Default to support
            if verbose:
                logging.warning(
                    f"EDGE CASE: claim_id={sample.get('claim_id')}, "
                    f"N_sup={N_sup} == N_con={N_con}, defaulting to positive_support"
                )

        new_sample["pool_type"] = new_pool_type
        new_sample["target_y"] = 1

        if verbose:
            logging.info(
                f"RELABEL: claim_id={sample.get('claim_id')}, "
                f"num_papers={sample.get('num_papers')}, "
                f"r_minority={r_minority:.3f} < {threshold:.3f}, "
                f"N_sup={N_sup}, N_con={N_con} → {new_pool_type} (label 1)"
            )

    return new_sample


def relabel_dataset(
    input_path: str,
    output_path: str,
    threshold: float,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Relabel an entire dataset based on threshold.

    Returns:
        Statistics dictionary with counts
    """
    logging.info(f"Loading dataset from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    logging.info(f"Processing {len(data)} samples with threshold τ={threshold}")

    # Track statistics
    stats = {
        "total": len(data),
        "negative_conflict_original": 0,
        "negative_conflict_kept": 0,
        "negative_conflict_relabeled": 0,
        "negative_conflict_skipped_no_labels": 0,
        "other_pool_types": 0,
        "final_positive": 0,
        "final_negative": 0
    }

    # Process each sample
    relabeled_data = []
    for sample in data:
        original_pool_type = sample.get("pool_type", "")

        if original_pool_type == "negative_conflict":
            stats["negative_conflict_original"] += 1

        new_sample = relabel_sample(sample, threshold, verbose)
        relabeled_data.append(new_sample)

        # Update statistics
        if original_pool_type == "negative_conflict":
            if new_sample["pool_type"] == "negative_conflict":
                stats["negative_conflict_kept"] += 1
            elif "n_support" not in sample or "n_contradict" not in sample:
                stats["negative_conflict_skipped_no_labels"] += 1
            else:
                stats["negative_conflict_relabeled"] += 1
        else:
            stats["other_pool_types"] += 1

        if new_sample["target_y"] == 1:
            stats["final_positive"] += 1
        else:
            stats["final_negative"] += 1

    # Save relabeled dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving relabeled dataset to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(relabeled_data, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Relabel negative_conflict samples by threshold")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--threshold", type=float, required=True, help="τ threshold value")
    parser.add_argument("--verbose", action="store_true", help="Log per-sample decisions")

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        logging.error(f"Threshold must be between 0.0 and 1.0, got {args.threshold}")
        return

    # Run relabeling
    stats = relabel_dataset(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        verbose=args.verbose
    )

    # Print summary
    logging.info("\n" + "="*60)
    logging.info(f"RELABELING SUMMARY (τ={args.threshold})")
    logging.info("="*60)
    logging.info(f"Total samples:                         {stats['total']}")
    logging.info(f"Original negative_conflict samples:    {stats['negative_conflict_original']}")
    logging.info(f"  - Kept as negative_conflict:         {stats['negative_conflict_kept']}")
    logging.info(f"  - Relabeled to positive:             {stats['negative_conflict_relabeled']}")
    logging.info(f"  - Skipped (no labels):               {stats['negative_conflict_skipped_no_labels']}")
    logging.info(f"Other pool types (unchanged):          {stats['other_pool_types']}")
    logging.info(f"Final positive (label 1):              {stats['final_positive']}")
    logging.info(f"Final negative (label 0):              {stats['final_negative']}")

    if stats['final_negative'] > 0:
        logging.info(f"Final class ratio (pos/neg):           {stats['final_positive'] / stats['final_negative']:.3f}")

    logging.info("="*60)


if __name__ == "__main__":
    main()
