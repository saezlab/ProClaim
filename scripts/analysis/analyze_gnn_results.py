#!/usr/bin/env python3
"""
Analyze GNN results.

1. Loads true_positive_edges.csv and true_negative_edges.csv to establish ground truth.
2. Loads gnn_test_predictions.csv for predictions.
3. Calculates performance metrics (Accuracy, Precision, Recall, F1) for:
   - All edges
   - True Positive set
   - True Negative set
4. Determines the optimal threshold for the GNN predictions.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import metrics
from typing import Dict, Tuple, Set

import matplotlib.pyplot as plt

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "signor"
RESULTS_DIR = PROJECT_ROOT / "results/gnn_analysis"
GNN_PRED_FILE = PROJECT_ROOT / "src/pkevolve/gnn/gnn_test_predictions.csv"
TP_EDGES_FILE = DATA_DIR / "true_positive_edges.csv"
TN_EDGES_FILE = DATA_DIR / "true_negative_edges.csv"

def load_edges(file_path: Path) -> Set[Tuple[str, str]]:
    """Load edges from a CSV file and return a set of (source, target) tuples."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    df = pd.read_csv(file_path)
    # Assuming columns 'ENTITYA' and 'ENTITYB' exist based on previous inspection
    if 'ENTITYA' not in df.columns or 'ENTITYB' not in df.columns:
        raise ValueError(f"File {file_path} missing required columns 'ENTITYA', 'ENTITYB'")
        
    edges = set()
    for _, row in df.iterrows():
        s = str(row['ENTITYA']).strip()
        t = str(row['ENTITYB']).strip()
        edges.add((s, t))
    return edges

def calculate_set_metrics(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate metrics for a dataframe given a threshold."""
    if len(df) == 0:
        return {}

    y_true = df['label_updated'].values
    y_prob = df['prediction'].values
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion Matrix
    # Labels must be in [0, 1] for confusion_matrix
    # But if df only has class 1 (only TP in set), confusion_matrix shape varies
    # So we force labels=[0, 1]
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Metrics
    # Handle single class cases for sklearn metrics gracefully if needed, 
    # but zero_division=0 usually handles precision/recall/f1.
    # Accuracy works fine.
    
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        # ROC AUC needs both classes to be present in y_true usually
        if len(np.unique(y_true)) > 1:
            roc_auc = metrics.roc_auc_score(y_true, y_prob)
        else:
            roc_auc = 0.0 
    except ValueError:
        roc_auc = 0.0

    return {
        'total': len(df),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def print_metrics(metrics_dict: Dict[str, float], title: str):
    """Pretty print metrics."""
    print(f"\n{title}")
    print("-" * 40)
    if not metrics_dict:
        print("No samples.")
        return
    
    print(f"  Total Samples: {metrics_dict['total']}")
    print(f"  Accuracy:      {metrics_dict['accuracy']:.4f}")
    print(f"  Precision:     {metrics_dict['precision']:.4f}")
    print(f"  Recall:        {metrics_dict['recall']:.4f}")
    print(f"  F1 Score:      {metrics_dict['f1']:.4f}")
    print(f"  Macro F1:      {metrics_dict['f1_macro']:.4f}")
    print(f"  ROC AUC:       {metrics_dict['roc_auc']:.4f}")
    print(f"  Confusion Matrix: TP={metrics_dict['tp']}, TN={metrics_dict['tn']}, FP={metrics_dict['fp']}, FN={metrics_dict['fn']}")

def main():
    print("=" * 80)
    print("GNN Performance Analysis")
    print("=" * 80)

    # 1. Load Ground Truth Sets
    print(f"Loading True Positive edges from {TP_EDGES_FILE}...")
    tp_edges = load_edges(TP_EDGES_FILE)
    print(f"  Found {len(tp_edges)} TP edges.")
    
    print(f"Loading True Negative edges from {TN_EDGES_FILE}...")
    tn_edges = load_edges(TN_EDGES_FILE)
    print(f"  Found {len(tn_edges)} TN edges.")

    # 2. Load Predictions
    print(f"Loading Predictions from {GNN_PRED_FILE}...")
    if not GNN_PRED_FILE.exists():
        print(f"Error: Prediction file not found: {GNN_PRED_FILE}")
        sys.exit(1)
        
    pred_df = pd.read_csv(GNN_PRED_FILE)
    print(f"  Loaded {len(pred_df)} predictions.")

    # 3. Assign Updated Labels
    # We ignore the 'label' column in the CSV and use TP/TN sets
    labels = []
    types = []
    
    valid_mask = []

    # Using sets for fast lookup
    # Need to handle potential whitespace trimming
    for _, row in pred_df.iterrows():
        s = str(row['source']).strip()
        t = str(row['target']).strip()
        
        is_tp = (s, t) in tp_edges
        is_tn = (s, t) in tn_edges
        
        if is_tp and is_tn:
            # Ambiguous - should not happen if sets are disjoint
            print(f"Warning: Edge ({s}, {t}) found in BOTH TP and TN sets. Treating as TP.")
            labels.append(1)
            types.append('tp_set')
            valid_mask.append(True)
        elif is_tp:
            labels.append(1)
            types.append('tp_set')
            valid_mask.append(True)
        elif is_tn:
            labels.append(0)
            types.append('tn_set')
            valid_mask.append(True)
        else:
            # Edge in prediction but not in current modified sets
            labels.append(-1)
            types.append('unknown')
            valid_mask.append(False)

    pred_df['label_updated'] = labels
    pred_df['set_type'] = types
    
    # Filter only valid edges
    original_count = len(pred_df)
    valid_df = pred_df[valid_mask].copy()
    dropped_count = original_count - len(valid_df)
    
    print(f"\nAligned predictions with modified sets.")
    print(f"  Matched: {len(valid_df)}")
    print(f"  Dropped (not in new sets): {dropped_count}")
    
    if len(valid_df) == 0:
        print("Error: No overlapping edges found between predictions and new sets.")
        sys.exit(1) # Exit cleanly but with error code

    # 4. Collect Plot Data and Calculate Metrics at 0.5
    print("\nCalculating metrics and generating plots...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    thresholds = np.arange(0.00, 1.01, 0.01)
    
    # We will simply report for 0.5 as requested
    report_thresh = 0.5
    
    # Data for plotting
    plot_data = {
        'thresholds': [],
        'acc_all': [],
        'acc_pos': [],
        'acc_neg': []
    }
    
    y_true = valid_df['label_updated'].values
    y_prob = valid_df['prediction'].values
    
    # Indices for subsets
    pos_mask = (valid_df['set_type'] == 'tp_set').values
    neg_mask = (valid_df['set_type'] == 'tn_set').values
    
    y_true_pos = y_true[pos_mask]
    y_prob_pos = y_prob[pos_mask]
    
    y_true_neg = y_true[neg_mask]
    y_prob_neg = y_prob[neg_mask]

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
            
        # Calculate Accuracies for plotting
        acc_all = metrics.accuracy_score(y_true, y_pred)
        
        # Pos set accuracy (Sensitivity/Recall)
        if len(y_true_pos) > 0:
            y_pred_pos = (y_prob_pos >= t).astype(int)
            acc_pos = metrics.accuracy_score(y_true_pos, y_pred_pos)
        else:
            acc_pos = 0.0
            
        # Neg set accuracy (Specificity)
        if len(y_true_neg) > 0:
            y_pred_neg = (y_prob_neg >= t).astype(int)
            acc_neg = metrics.accuracy_score(y_true_neg, y_pred_neg)
        else:
            acc_neg = 0.0
            
        plot_data['thresholds'].append(t)
        plot_data['acc_all'].append(acc_all)
        plot_data['acc_pos'].append(acc_pos)
        plot_data['acc_neg'].append(acc_neg)
    
    # --- Plotting --- (Combined)
    print("Generating combined plot...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['thresholds'], plot_data['acc_all'], label='All Edges', color='blue', linewidth=2)
    plt.plot(plot_data['thresholds'], plot_data['acc_pos'], label='True Positive Set', color='green', linewidth=2)
    plt.plot(plot_data['thresholds'], plot_data['acc_neg'], label='True Negative Set', color='orange', linewidth=2)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Remove title as requested
    # plt.title('Accuracy vs Threshold') 
    
    combined_plot_path = RESULTS_DIR / 'gnn_accuracy_vs_threshold.pdf'
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    
    print(f"Combined plot saved to {combined_plot_path}")

    # 5. Report Performance at Threshold 0.5
    print(f"\n" + "="*40)
    print(f"Performance Analysis at Threshold {report_thresh:.2f}")
    print("="*40)
    
    # All Edges
    m_all = calculate_set_metrics(valid_df, threshold=report_thresh)
    print_metrics(m_all, f"All Edges")
    
    # TP Set Performance
    tp_subset = valid_df[valid_df['set_type'] == 'tp_set']
    m_tp = calculate_set_metrics(tp_subset, threshold=report_thresh)
    print_metrics(m_tp, f"True Positive Set")
    
    # TN Set Performance
    tn_subset = valid_df[valid_df['set_type'] == 'tn_set']
    m_tn = calculate_set_metrics(tn_subset, threshold=report_thresh)
    print_metrics(m_tn, f"True Negative Set")

    # Save metrics to JSON
    json_results = {
        "threshold": report_thresh,
        "all_edges": m_all,
        "true_positive_set": m_tp,
        "true_negative_set": m_tn
    }
    
    json_path = RESULTS_DIR / 'gnn_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nMetrics saved to {json_path}")

if __name__ == "__main__":
    main()
