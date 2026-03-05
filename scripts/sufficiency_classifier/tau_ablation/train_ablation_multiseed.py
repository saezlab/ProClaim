"""
Train MLP classifiers for τ-threshold ablation with multiple random seeds.

This script trains models for all combinations of:
- 6 τ thresholds: {0.0, 0.16, 0.2, 0.25, 0.33, 0.5}
- 3 random seeds: {42, 123, 456}
Total: 18 models

This allows us to distinguish signal (consistent across seeds) from noise (varies across seeds).

Usage:
    python scripts/sufficiency_classifier/train_ablation_multiseed.py \
        --splits_dir data/ablation/splits \
        --output_dir results/ablation/models_multiseed \
        --seeds 42 123 456 \
        --features important

    # Or run in background
    nohup python scripts/sufficiency_classifier/train_ablation_multiseed.py \
        > results/ablation/training_multiseed.log 2>&1 &
"""

import argparse
import json
import logging
import subprocess
import time
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from tau_config import TAU_THRESHOLDS
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from tau_config import TAU_THRESHOLDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(
    train_data_path: str,
    output_dir: str,
    tau: float,
    seed: int,
    features: str = "important",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64
) -> Dict[str, Any]:
    """
    Train a single model for a given (τ, seed) combination.

    Returns:
        Training metrics and metadata
    """
    tau_str = f"{tau:.2f}"
    model_dir = Path(output_dir) / f"tau_{tau_str}_seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*80}")
    logging.info(f"Training: τ={tau_str}, seed={seed}")
    logging.info(f"{'='*80}")

    # Build command - use PYTHONHASHSEED for full reproducibility
    cmd = [
        "env", f"PYTHONHASHSEED={seed}",
        "uv", "run", "python",
        "scripts/sufficiency_classifier/train_mlp_classifier.py",
        "--data_path", train_data_path,
        "--output_dir", str(model_dir),
        "--seed", str(seed),
        "--features", features,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--hidden_dim", str(hidden_dim)
    ]

    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        logging.error(f"Training failed for τ={tau_str}, seed={seed}")
        logging.error(result.stderr[-500:])  # Last 500 chars
        return {
            "tau": tau,
            "seed": seed,
            "status": "failed",
            "error": result.stderr[-200:],
            "elapsed_time": elapsed_time
        }

    # Parse metrics from training output
    metrics = parse_training_metrics(result.stderr)
    metrics["tau"] = tau
    metrics["seed"] = seed
    metrics["status"] = "success"
    metrics["elapsed_time"] = elapsed_time
    metrics["model_dir"] = str(model_dir)

    logging.info(f"✓ Completed in {elapsed_time:.1f}s")
    logging.info(f"  Accuracy: {metrics.get('test_accuracy', 0):.4f}")
    logging.info(f"  F1:       {metrics.get('test_f1', 0):.4f}")

    return metrics


def parse_training_metrics(output: str) -> Dict[str, float]:
    """Parse test metrics from training script output."""
    metrics = {}

    for line in output.split('\n'):
        line = line.strip()

        if "Accuracy:" in line:
            try:
                metrics['test_accuracy'] = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
        elif "Precision:" in line:
            try:
                metrics['test_precision'] = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
        elif "Recall:" in line:
            try:
                metrics['test_recall'] = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
        elif "F1 Score:" in line:
            try:
                metrics['test_f1'] = float(line.split()[-1])
            except (ValueError, IndexError):
                pass

    return metrics


def compute_seed_statistics(all_metrics: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    """
    Compute mean and std across seeds for each τ.

    Returns:
        Dict mapping τ → {mean_acc, std_acc, mean_f1, std_f1, ...}
    """
    from collections import defaultdict

    # Group by tau
    by_tau = defaultdict(list)
    for m in all_metrics:
        if m.get('status') == 'success':
            by_tau[m['tau']].append(m)

    # Compute statistics
    stats = {}
    for tau, metrics_list in sorted(by_tau.items()):
        accs = [m['test_accuracy'] for m in metrics_list]
        f1s = [m['test_f1'] for m in metrics_list]
        precs = [m['test_precision'] for m in metrics_list]
        recs = [m['test_recall'] for m in metrics_list]

        stats[tau] = {
            'n_seeds': len(metrics_list),
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'mean_precision': np.mean(precs),
            'std_precision': np.std(precs),
            'mean_recall': np.mean(recs),
            'std_recall': np.std(recs),
            'individual_seeds': metrics_list
        }

    return stats


def generate_summary_report(all_metrics: List[Dict[str, Any]], stats: Dict[float, Dict[str, Any]], output_dir: Path):
    """Generate summary report with cross-seed analysis."""

    # Save raw metrics
    metrics_json = output_dir / "training_metrics_all_seeds.json"
    with open(metrics_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logging.info(f"\n✓ Saved raw metrics to {metrics_json}")

    # Save statistics
    stats_json = output_dir / "training_statistics.json"
    stats_serializable = {}
    for tau, stat in stats.items():
        stats_serializable[f"{tau:.2f}"] = {
            k: v for k, v in stat.items() if k != 'individual_seeds'
        }
    with open(stats_json, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    logging.info(f"✓ Saved statistics to {stats_json}")

    # Save CSV with mean ± std
    report_csv = output_dir / "training_summary_multiseed.csv"
    with open(report_csv, 'w') as f:
        f.write("tau,n_seeds,mean_acc,std_acc,mean_f1,std_f1,mean_prec,std_prec,mean_rec,std_rec\n")
        for tau in sorted(stats.keys()):
            s = stats[tau]
            f.write(
                f"{tau:.2f},{s['n_seeds']},"
                f"{s['mean_accuracy']:.4f},{s['std_accuracy']:.4f},"
                f"{s['mean_f1']:.4f},{s['std_f1']:.4f},"
                f"{s['mean_precision']:.4f},{s['std_precision']:.4f},"
                f"{s['mean_recall']:.4f},{s['std_recall']:.4f}\n"
            )
    logging.info(f"✓ Saved summary CSV to {report_csv}")

    # Print summary table
    logging.info("\n" + "="*100)
    logging.info("MULTI-SEED ABLATION SUMMARY (Mean ± Std across seeds)")
    logging.info("="*100)
    logging.info(f"{'τ':>6} | {'Seeds':>5} | {'Accuracy':>18} | {'F1 Score':>18} | {'Precision':>18} | {'Recall':>18}")
    logging.info("-"*100)
    for tau in sorted(stats.keys()):
        s = stats[tau]
        logging.info(
            f"{tau:>6.2f} | {s['n_seeds']:>5} | "
            f"{s['mean_accuracy']:>6.4f} ± {s['std_accuracy']:>6.4f} | "
            f"{s['mean_f1']:>6.4f} ± {s['std_f1']:>6.4f} | "
            f"{s['mean_precision']:>6.4f} ± {s['std_precision']:>6.4f} | "
            f"{s['mean_recall']:>6.4f} ± {s['std_recall']:>6.4f}"
        )
    logging.info("="*100)

    # Identify best τ by mean F1
    best_tau = max(stats.items(), key=lambda x: x[1]['mean_f1'])
    logging.info(f"\n✓ Best τ by mean F1: {best_tau[0]:.2f} (F1 = {best_tau[1]['mean_f1']:.4f} ± {best_tau[1]['std_f1']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train ablation models with multiple random seeds")
    parser.add_argument("--splits_dir", type=str,
                       default="data/ablation/splits",
                       help="Directory containing train/test splits")
    parser.add_argument("--output_dir", type=str,
                       default="results/ablation/models_multiseed",
                       help="Output directory for trained models")
    parser.add_argument("--thresholds", type=float, nargs="+",
                       default=None,
                       help="List of τ thresholds to train (defaults to tau_config.TAU_THRESHOLDS)")
    parser.add_argument("--seeds", type=int, nargs="+",
                       default=[42, 123, 456, 789, 1011],
                       help="List of random seeds to use")
    parser.add_argument("--features", type=str, default="important",
                       help="Feature preset to use")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64,
                       help="Hidden layer dimension")

    args = parser.parse_args()

    if args.thresholds is None:
        args.thresholds = TAU_THRESHOLDS

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_combinations = len(args.thresholds) * len(args.seeds)
    logging.info(f"{'='*80}")
    logging.info(f"MULTI-SEED τ-THRESHOLD ABLATION STUDY")
    logging.info(f"{'='*80}")
    logging.info(f"Training {n_combinations} models:")
    logging.info(f"  τ values: {args.thresholds}")
    logging.info(f"  Seeds: {args.seeds}")
    logging.info(f"  Combinations: {len(args.thresholds)} × {len(args.seeds)} = {n_combinations}")
    logging.info(f"  Output: {output_dir}")

    # Train all combinations
    all_metrics = []
    completed = 0
    start_time = time.time()

    for tau in args.thresholds:
        tau_str = f"{tau:.2f}"
        train_data_path = splits_dir / f"tau_{tau_str}_train.json"

        if not train_data_path.exists():
            logging.error(f"Training data not found: {train_data_path}")
            continue

        for seed in args.seeds:
            logging.info(f"\n[{completed+1}/{n_combinations}] τ={tau_str}, seed={seed}")

            metrics = train_model(
                train_data_path=str(train_data_path),
                output_dir=str(output_dir),
                tau=tau,
                seed=seed,
                features=args.features,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim
            )
            all_metrics.append(metrics)
            completed += 1

            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            eta = avg_time * (n_combinations - completed)
            logging.info(f"Progress: {completed}/{n_combinations} ({completed/n_combinations*100:.1f}%) | "
                        f"ETA: {eta/60:.1f} min")

    # Compute statistics across seeds
    stats = compute_seed_statistics(all_metrics)

    # Generate summary report
    generate_summary_report(all_metrics, stats, output_dir)

    total_time = time.time() - start_time
    logging.info(f"\n✓ All training completed in {total_time/60:.1f} minutes!")
    logging.info(f"Models saved to: {output_dir}")

    # Check for failures
    failed = [m for m in all_metrics if m.get('status') != 'success']
    if failed:
        logging.warning(f"\n⚠ {len(failed)} training job(s) failed:")
        for m in failed:
            logging.warning(f"  τ={m.get('tau')}, seed={m.get('seed')}")


if __name__ == "__main__":
    main()
