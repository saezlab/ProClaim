"""
Generate all dataset variants for τ-threshold ablation study.

This script generates 13 dataset variants with different τ thresholds:
- τ = 0.00 (baseline - all conflicts are kept as negative_conflict regardless of imbalance)
- τ = 0.11 (relabel if minority < 11.1%, i.e. 8S:1C)
- τ = 0.13 (relabel if minority < 12.5%, i.e. 7S:1C)
- τ = 0.14 (relabel if minority < 14.3%, i.e. 6S:1C)
- τ = 0.17 (relabel if minority < 16.7%, i.e. 5S:1C)
- τ = 0.20 (relabel if minority < 20.0%, i.e. 4S:1C, 1S:4C)
- τ = 0.25 (relabel if minority < 25.0%, i.e. 3S:1C, 1S:3C)
- τ = 0.29 (relabel if minority < 28.6%, i.e. 5S:2C)
- τ = 0.33 (relabel if minority < 33.3%, i.e. 2S:1C, 1S:2C)
- τ = 0.38 (relabel if minority < 37.5%, i.e. 5S:3C)
- τ = 0.40 (relabel if minority < 40.0%, i.e. 3S:2C, 2S:3C)
- τ = 0.43 (relabel if minority < 42.9%, i.e. 4S:3C)
- τ = 0.50 (most aggressive relabeling - only perfectly balanced 50/50 are kept as conflict)

Usage:
    python scripts/sufficiency_classifier/generate_ablation_datasets.py \
        --input data/classifier_train_data_with_labels.json \
        --output_dir data/ablation

Output files:
    data/ablation/tau_0.00.json
    data/ablation/tau_0.11.json
    ...
    data/ablation/tau_0.50.json
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict

try:
    from tau_config import TAU_THRESHOLDS
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from tau_config import TAU_THRESHOLDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_datasets(
    input_path: str,
    output_dir: str,
    thresholds: List[float] = None
) -> Dict[float, Dict[str, int]]:
    """
    Generate all dataset variants for different τ thresholds.

    Args:
        input_path: Path to input dataset with SciFact labels
        output_dir: Output directory for generated datasets
        thresholds: List of τ values to use (defaults to TAU_THRESHOLDS)
        
    Returns:
        Dictionary mapping τ → statistics
    """
    if thresholds is None:
        thresholds = TAU_THRESHOLDS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Generating {len(thresholds)} dataset variants")
    logging.info(f"Input: {input_path}")
    logging.info(f"Output directory: {output_dir}")

    all_stats = {}

    for tau in thresholds:
        logging.info(f"\n{'='*60}")
        logging.info(f"Generating dataset for τ={tau:.2f}")
        logging.info(f"{'='*60}")

        output_path = output_dir / f"tau_{tau:.2f}.json"

        # Call relabel_by_threshold.py
        cmd = [
            "python",
            "scripts/sufficiency_classifier/tau_ablation/relabel_by_threshold.py",
            "--input", input_path,
            "--output", str(output_path),
            "--threshold", str(tau)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Failed to generate dataset for τ={tau}")
            logging.error(result.stderr)
            continue

        # Parse statistics from stderr (logging output goes to stderr)
        stats = parse_stats_from_output(result.stderr)
        all_stats[tau] = stats

        logging.info(f"✓ Generated {output_path}")
        logging.info(f"  Final positive: {stats.get('final_positive', '?')}")
        logging.info(f"  Final negative: {stats.get('final_negative', '?')}")

    return all_stats


def parse_stats_from_output(output: str) -> Dict[str, int]:
    """Parse statistics from relabel_by_threshold.py output."""
    stats = {}

    for line in output.split('\n'):
        # Strip ANSI color codes and extra whitespace
        line = line.strip()

        if "Total samples:" in line:
            stats['total'] = int(line.split()[-1])
        elif "Original negative_conflict samples:" in line:
            stats['negative_conflict_original'] = int(line.split()[-1])
        elif "- Kept as negative_conflict:" in line:
            stats['negative_conflict_kept'] = int(line.split()[-1])
        elif "- Relabeled to positive:" in line:
            stats['negative_conflict_relabeled'] = int(line.split()[-1])
        elif "Final positive (label 1):" in line:
            stats['final_positive'] = int(line.split()[-1])
        elif "Final negative (label 0):" in line:
            stats['final_negative'] = int(line.split()[-1])

    return stats


def generate_summary_report(all_stats: Dict[float, Dict[str, int]], output_dir: Path):
    """Generate a summary CSV and JSON report of all datasets."""

    summary_data = []
    for tau in sorted(all_stats.keys()):
        stats = all_stats[tau]
        final_pos = stats.get('final_positive', 0)
        final_neg = stats.get('final_negative', 1)  # Avoid division by zero
        summary_data.append({
            "tau": tau,
            "total": stats.get('total', 0),
            "positive": final_pos,
            "negative": final_neg,
            "pos_neg_ratio": final_pos / max(final_neg, 1),
            "neg_conflict_kept": stats.get('negative_conflict_kept', 0),
            "neg_conflict_relabeled": stats.get('negative_conflict_relabeled', 0)
        })

    # Save JSON summary
    summary_json = output_dir / "dataset_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logging.info(f"\n✓ Saved summary to {summary_json}")

    # Save CSV summary
    summary_csv = output_dir / "dataset_summary.csv"
    with open(summary_csv, 'w') as f:
        # Header
        f.write("tau,total,positive,negative,pos_neg_ratio,neg_conflict_kept,neg_conflict_relabeled\n")
        # Data rows
        for row in summary_data:
            f.write(f"{row['tau']:.2f},{row['total']},{row['positive']},{row['negative']},"
                   f"{row['pos_neg_ratio']:.3f},{row['neg_conflict_kept']},{row['neg_conflict_relabeled']}\n")
    logging.info(f"✓ Saved summary to {summary_csv}")

    # Print summary table
    logging.info("\n" + "="*80)
    logging.info("DATASET GENERATION SUMMARY")
    logging.info("="*80)
    logging.info(f"{'τ':>6} | {'Total':>6} | {'Pos':>6} | {'Neg':>6} | {'Ratio':>6} | {'Kept':>6} | {'Relabeled':>10}")
    logging.info("-"*80)
    for row in summary_data:
        logging.info(
            f"{row['tau']:>6.2f} | {row['total']:>6} | {row['positive']:>6} | {row['negative']:>6} | "
            f"{row['pos_neg_ratio']:>6.3f} | {row['neg_conflict_kept']:>6} | {row['neg_conflict_relabeled']:>10}"
        )
    logging.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate ablation datasets for all τ thresholds")
    parser.add_argument("--input", type=str,
                       default="data/classifier_train_data_with_labels.json",
                       help="Input dataset with SciFact labels")
    parser.add_argument("--output_dir", type=str,
                       default="data/ablation",
                       help="Output directory for generated datasets")
    parser.add_argument("--thresholds", type=float, nargs="+",
                       default=TAU_THRESHOLDS,
                       help="List of τ thresholds to generate")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        logging.info("Please ensure the dataset has been augmented with SciFact labels.")
        logging.info("Run: python scripts/sufficiency_classifier/augment_existing_data.py")
        return

    # Generate all datasets
    all_stats = generate_datasets(
        input_path=str(input_path),
        output_dir=args.output_dir,
        thresholds=args.thresholds
    )

    # Generate summary report
    generate_summary_report(all_stats, Path(args.output_dir))

    logging.info("\n✓ All datasets generated successfully!")
    logging.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
