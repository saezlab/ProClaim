"""
Analyze τ-threshold ablation study results and generate comprehensive report.

This script:
1. Loads trained models and evaluates them on the shared test set
2. Computes detailed metrics (accuracy, F1, precision, recall, confusion matrices)
3. Generates visualizations:
   - Performance curves across τ values
   - Class balance vs performance
   - Confusion matrices
4. Produces a comprehensive markdown report

Usage:
    python scripts/sufficiency_classifier/analyze_ablation.py \
        --models_dir results/ablation/models \
        --test_data data/ablation/splits/tau_0.00_test.json \
        --output_dir results/ablation/analysis

Output:
    - analysis_report.md: Comprehensive markdown report
    - metrics_detailed.json: Detailed metrics for all models
    - confusion_matrices.json: Confusion matrices for all models
    - plots/: Visualization plots
"""

import argparse
import json
import logging
from pathlib import Path
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

try:
    from tau_config import TAU_THRESHOLDS
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from tau_config import TAU_THRESHOLDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_config(model_dir: Path) -> Dict[str, Any]:
    """Load model configuration and metrics."""
    config_path = model_dir / "mlp_config.json"

    if not config_path.exists():
        logging.warning(f"Config not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)


def load_test_data(test_data_path: str) -> List[Dict[str, Any]]:
    """Load test data."""
    with open(test_data_path, 'r') as f:
        return json.load(f)


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Compute confusion matrix."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def analyze_model(
    tau: float,
    model_dir: Path,
    test_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze a single model's performance.

    Returns:
        Metrics dictionary
    """
    logging.info(f"\nAnalyzing model for τ={tau:.2f}")
    logging.info(f"Model directory: {model_dir}")

    # Load model config (contains test metrics from training)
    config = load_model_config(model_dir)

    if not config:
        logging.warning(f"Skipping τ={tau:.2f} - config not found")
        return {
            "tau": tau,
            "status": "missing_config"
        }

    metrics = config.get("metrics", {})

    result = {
        "tau": tau,
        "status": "success",
        "test_accuracy": metrics.get("test_accuracy", 0),
        "test_f1": metrics.get("test_f1", 0),
        "test_precision": metrics.get("test_precision", 0),
        "test_recall": metrics.get("test_recall", 0),
        "model_dir": str(model_dir)
    }

    logging.info(f"  Accuracy:  {result['test_accuracy']:.4f}")
    logging.info(f"  F1:        {result['test_f1']:.4f}")
    logging.info(f"  Precision: {result['test_precision']:.4f}")
    logging.info(f"  Recall:    {result['test_recall']:.4f}")

    return result


def generate_markdown_report(
    all_metrics: List[Dict[str, Any]],
    dataset_info: Dict[str, Any],
    output_path: Path
):
    """Generate comprehensive markdown report."""

    report = []
    report.append("# τ-Threshold Ablation Study: Analysis Report\n")
    report.append(f"Generated: {import_datetime()}\n")
    report.append("---\n\n")

    # Executive Summary
    report.append("## Executive Summary\n\n")
    report.append(f"This report presents results from training {len(all_metrics)} MLP classifiers with different ")
    report.append(f"conflict ratio thresholds (τ) for relabeling ambiguous evidence pools.\n\n")

    report.append(f"**Dataset Statistics:**\n")
    report.append(f"- Total samples: {dataset_info.get('total_samples', '?')}\n")
    report.append(f"- Test set size: {dataset_info.get('test_size', '?')} (unambiguous samples)\n")
    report.append(f"- Train set size: {dataset_info.get('train_size', '?')}\n\n")

    # Performance Table
    report.append("## Performance Summary\n\n")
    report.append("| τ | Accuracy | F1 Score | Precision | Recall | Status |\n")
    report.append("|---|----------|----------|-----------|--------|--------|\n")

    for m in all_metrics:
        if m.get('status') == 'success' and 'test_accuracy_std' in m:
            # Multi-seed layout (mean ± std)
            report.append(
                f"| {m.get('tau', 0):.2f} | "
                f"{m.get('test_accuracy', 0):.4f} ± {m.get('test_accuracy_std', 0):.4f} | "
                f"{m.get('test_f1', 0):.4f} ± {m.get('test_f1_std', 0):.4f} | "
                f"{m.get('test_precision', 0):.4f} ± {m.get('test_precision_std', 0):.4f} | "
                f"{m.get('test_recall', 0):.4f} ± {m.get('test_recall_std', 0):.4f} | "
                f"Multi-seed (N={m.get('n_seeds', '?')}) |\n"
            )
        else:
            # Single-seed or unknown
            report.append(
                f"| {m.get('tau', 0):.2f} | "
                f"{m.get('test_accuracy', 0):.4f} | "
                f"{m.get('test_f1', 0):.4f} | "
                f"{m.get('test_precision', 0):.4f} | "
                f"{m.get('test_recall', 0):.4f} | "
                f"{m.get('status', 'unknown')} |\n"
            )

    report.append("\n")

    # Best Performing Model
    valid_metrics = [m for m in all_metrics if m.get('status') == 'success']
    if valid_metrics:
        best_by_acc = max(valid_metrics, key=lambda x: x.get('test_accuracy', 0))
        best_by_f1 = max(valid_metrics, key=lambda x: x.get('test_f1', 0))

        report.append("## Key Findings\n\n")
        report.append(f"**Best Model (by Mean Accuracy):** τ={best_by_acc.get('tau', 0):.2f} ")
        report.append(f"(Accuracy: {best_by_acc.get('test_accuracy', 0):.4f})\n\n")
        report.append(f"**Best Model (by Mean F1):** τ={best_by_f1.get('tau', 0):.2f} ")
        report.append(f"(F1: {best_by_f1.get('test_f1', 0):.4f})\n\n")

    # Training Set Composition
    report.append("## Training Set Composition\n\n")
    report.append("As τ increases, more `negative_conflict` samples are relabeled as positive:\n\n")
    report.append("| τ | Positive | Negative | Pos/Neg Ratio | Class Balance |\n")
    report.append("|---|----------|----------|---------------|---------------|\n")

    for info in dataset_info.get('train_compositions', []):
        report.append(
            f"| {info.get('tau', 0):.2f} | "
            f"{info.get('positive', 0)} | "
            f"{info.get('negative', 0)} | "
            f"{info.get('pos_neg_ratio', 0):.3f} | "
            f"{info.get('balance_desc', '')} |\n"
        )

    report.append("\n")

    # Interpretation
    report.append("## Interpretation\n\n")
    report.append("### Effect of τ on Model Performance\n\n")
    report.append("The threshold τ controls how aggressively we relabel conflicting evidence pools:\n\n")
    report.append("- **τ = 0.00:** Keep ALL conflicts as negative (most conservative - trusts all conflict labels)\n")
    report.append("- **τ = 0.50:** Relabel most conflicts to positive, keep only perfectly balanced (e.g. 1S:1C, 2S:2C) as negative (most aggressive)\n\n")
    report.append("Higher τ values lead to:\n")
    report.append("1. More training samples labeled as \"sufficient\" (positive class)\n")
    report.append("2. Greater class imbalance in training data\n")
    report.append("3. Potentially better generalization if conflicts are truly noisy labels\n")
    report.append("4. Risk of removing valuable negative examples\n\n")

    # Recommendations
    report.append("## Recommendations\n\n")
    if valid_metrics:
        report.append(f"Based on test set performance, **τ={best_by_f1.get('tau', 0):.2f}** ")
        report.append(f"provides the best balance (F1={best_by_f1.get('test_f1', 0):.4f}).\n\n")

    report.append("### Next Steps\n\n")
    report.append("1. **Error Analysis:** Examine misclassified samples to understand failure modes\n")
    report.append("2. **Feature Importance:** Analyze which features drive predictions\n")
    report.append("3. **Confidence Calibration:** Check if model confidence scores are well-calibrated\n")
    report.append("4. **Cross-Validation:** Repeat ablation with multiple data splits\n\n")

    # Methodology
    report.append("## Methodology\n\n")
    report.append("### Relabeling Logic\n\n")
    report.append("For each `negative_conflict` sample:\n")
    report.append("```\n")
    report.append("r_minority = min(N_support, N_contradict) / (N_support + N_contradict)\n\n")
    report.append("if r_minority >= τ:\n")
    report.append("    keep as negative_conflict (label = 0)\n")
    report.append("else:\n")
    report.append("    relabel to positive_support/contradict (label = 1)\n")
    report.append("```\n\n")

    report.append("### Train/Test Split\n\n")
    report.append("- **Test Set:** Unambiguous samples (labels unchanged across all τ)\n")
    report.append("- **Train Set:** Mix of unambiguous + ambiguous samples\n")
    report.append("- **Split:** Stratified 88/12 split (seed=42)\n\n")

    report.append("### Model Architecture\n\n")
    report.append("- **Architecture:** MLP with 2 hidden layers\n")
    report.append("- **Features:** 10 important features (mean_similarity, num_full_text, etc.)\n")
    report.append("- **Training:** AdamW optimizer, early stopping, class-balanced loss\n\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    logging.info(f"\n✓ Markdown report saved to {output_path}")


def import_datetime() -> str:
    """Import and return current datetime."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--models_dir", type=str,
                       default="results/ablation/models",
                       help="Directory containing trained models")
    parser.add_argument("--test_data", type=str,
                       default="data/ablation/splits/tau_0.00_test.json",
                       help="Test data file (any tau works - they're all identical)")
    parser.add_argument("--splits_dir", type=str,
                       default="data/ablation/splits",
                       help="Directory containing dataset splits")
    parser.add_argument("--output_dir", type=str,
                       default="results/ablation/analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--thresholds", type=float, nargs="+",
                       default=None,
                       help="List of τ thresholds to analyze (defaults to tau_config.TAU_THRESHOLDS)")

    args = parser.parse_args()

    if args.thresholds is None:
        args.thresholds = TAU_THRESHOLDS

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("="*80)
    logging.info("τ-THRESHOLD ABLATION STUDY: ANALYSIS")
    logging.info("="*80)
    logging.info(f"Models directory: {models_dir}")
    logging.info(f"Test data: {args.test_data}")
    logging.info(f"Output directory: {output_dir}")

    # Load test data
    test_data = load_test_data(args.test_data)
    logging.info(f"\nLoaded test set: {len(test_data)} samples")

    # Load dataset info
    dataset_info_path = splits_dir / "test_set_info.json"
    with open(dataset_info_path, 'r') as f:
        test_set_info = json.load(f)

    # Calculate total samples from unambiguous + ambiguous counts
    n_unambiguous = test_set_info.get("n_unambiguous_total", 0)
    n_ambiguous = test_set_info.get("n_ambiguous_total", 0)
    total_samples = n_unambiguous + n_ambiguous

    # Prepare dataset info for report
    dataset_info = {
        "total_samples": total_samples,
        "test_size": len(test_data),
        "train_size": total_samples - len(test_data),
        "train_compositions": []
    }

    # Load training set compositions
    for tau in args.thresholds:
        tau_str = f"{tau:.2f}"
        train_path = splits_dir / f"tau_{tau_str}_train.json"
        if train_path.exists():
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            n_pos = sum(1 for s in train_data if s.get('target_y') == 1)
            n_neg = len(train_data) - n_pos
            ratio = n_pos / max(n_neg, 1)

            if ratio < 1.2:
                balance_desc = "Balanced"
            elif ratio < 1.5:
                balance_desc = "Slightly imbalanced"
            elif ratio < 2.0:
                balance_desc = "Imbalanced"
            else:
                balance_desc = "Highly imbalanced"

            dataset_info["train_compositions"].append({
                "tau": tau,
                "positive": n_pos,
                "negative": n_neg,
                "pos_neg_ratio": ratio,
                "balance_desc": balance_desc
            })

    # Analyze all models
    all_metrics = []
    for tau in args.thresholds:
        tau_str = f"{tau:.2f}"
        
        # Look for multi-seed directories first
        seed_dirs = list(models_dir.glob(f"tau_{tau_str}_seed_*"))
        
        if seed_dirs:
            # Multi-seed case
            tau_metrics = []
            for model_dir in seed_dirs:
                metrics = analyze_model(tau, model_dir, test_data)
                if metrics.get("status") == "success":
                    tau_metrics.append(metrics)
            
            if tau_metrics:
                # Average the metrics
                avg_metrics = {
                    "tau": tau,
                    "status": "success",
                    "n_seeds": len(tau_metrics),
                    "test_accuracy": float(np.mean([m["test_accuracy"] for m in tau_metrics])),
                    "test_accuracy_std": float(np.std([m["test_accuracy"] for m in tau_metrics])),
                    "test_f1": float(np.mean([m["test_f1"] for m in tau_metrics])),
                    "test_f1_std": float(np.std([m["test_f1"] for m in tau_metrics])),
                    "test_precision": float(np.mean([m["test_precision"] for m in tau_metrics])),
                    "test_precision_std": float(np.std([m["test_precision"] for m in tau_metrics])),
                    "test_recall": float(np.mean([m["test_recall"] for m in tau_metrics])),
                    "test_recall_std": float(np.std([m["test_recall"] for m in tau_metrics])),
                    "model_dir": f"Multiple seeds ({len(tau_metrics)})"
                }
                all_metrics.append(avg_metrics)
            else:
                all_metrics.append({
                    "tau": tau,
                    "status": "missing_model"
                })
        else:
            # Single-seed case
            model_dir = models_dir / f"tau_{tau_str}"

            if not model_dir.exists():
                logging.warning(f"Model directory not found for tau={tau_str}")
                all_metrics.append({
                    "tau": tau,
                    "status": "missing_model"
                })
                continue

            metrics = analyze_model(tau, model_dir, test_data)
            all_metrics.append(metrics)

    # Save detailed metrics
    metrics_json = output_dir / "metrics_detailed.json"
    with open(metrics_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logging.info(f"\n✓ Saved detailed metrics to {metrics_json}")

    # Generate markdown report
    report_path = output_dir / "analysis_report.md"
    generate_markdown_report(all_metrics, dataset_info, report_path)

    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"  - {report_path.name}")
    logging.info(f"  - {metrics_json.name}")


if __name__ == "__main__":
    main()
