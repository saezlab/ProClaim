#!/usr/bin/env python3
"""
Analyze QA results for GLM-4.6.

This script reads QA evaluation results from langgraph/results/reasoning/glm-4.6/no_search,
then performs analysis.
It filters for repeat0 only.
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
QA_RESULTS_DIR = PROJECT_ROOT / "langgraph/results/reasoning/glm-4.6/no_search"

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse interaction information from filename.
    
    Expected format: SOURCE_TARGET_repeatX.json
    Example: AKT1_PPP1CA_repeat0.json
    
    Returns:
        Dictionary with 'source', 'target', 'interaction' keys
    """
    stem = filename.replace('.json', '')
    parts = stem.split('_')
    
    # We expect at least Source, Target, repeatX
    if len(parts) < 3:
         return {'source': None, 'target': None, 'interaction': None}

    # Assuming format Source_Target_repeatX
    # If the source or target has underscores, this might be ambiguous, 
    # but based on seen examples (AKT1_PPP1CA), it seems standard.
    # The last part is repeatX.
    
    return {
        'source': parts[0],
        'target': parts[1],
        'interaction': 'Unknown' # Interaction type is not in the filename for these results
    }

def load_qa_results() -> Dict[str, List[Dict]]:
    """
    Load QA results from the specific GLM result directories.
    
    Returns:
        Dictionary with keys 'true_positive_edges' and 'true_negative_edges',
        each containing a list of result dictionaries.
    """
    results = {
        'true_positive_edges': [],
        'true_negative_edges': []
    }
    
    # Map directory names to edge types
    # true_edges -> true_positive_edges
    # false_edges -> true_negative_edges
    dir_mapping = {
        'true_edges': 'true_positive_edges',
        'false_edges': 'true_negative_edges'
    }

    for dir_name, edge_type in dir_mapping.items():
        edge_dir = QA_RESULTS_DIR / dir_name
        
        if not edge_dir.exists():
            print(f"Warning: {edge_dir} does not exist")
            continue
            
        # Filter for repeat0 only and exclude raw_ files
        json_files = list(edge_dir.glob("*_repeat0.json"))
        # Filter out raw_ files
        json_files = [f for f in json_files if not f.name.startswith("raw_")]
        
        print(f"Loading {len(json_files)} QA result files from {dir_name} (repeat0 only)...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['edge_type'] = edge_type
                    data['filename'] = json_file.name
                    results[edge_type].append(data)
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
                
    return results

def convert_to_dataframe(qa_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Convert QA results to a pandas DataFrame for analysis.
    
    Args:
        qa_results: Dictionary from load_qa_results()
        
    Returns:
        DataFrame with columns for analysis
    """
    rows = []

    for edge_type, results_list in qa_results.items():
        for result in results_list:
            # Parse filename to get source, target
            file_info = parse_filename(result['filename'])
            
            # Extract data from flat JSON structure
            # Structure: {"reasoning": "...", "answer_text": "...", "answer": bool, "reasoning_content": "..."}
            
            row = {
                'filename': result['filename'],
                'edge_type': edge_type, # 'true_positive_edges' or 'true_negative_edges'
                'ground_truth': edge_type == 'true_positive_edges',
                'source': file_info['source'],
                'target': file_info['target'],
                'interaction': file_info['interaction'],
                'question': "N/A", # Not present in these files
                
                'prediction': result.get('answer'), # Boolean
                'reasoning': result.get('reasoning'),
                'reasoning_content': result.get('reasoning_content', ''),
                # context_length not available
            }
            rows.append(row)

    return pd.DataFrame(rows)

def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, F1 (Binary & Macro).
    
    IMPORTANT: NaN/None predictions are treated as a third category (Unknown).
    This means they are NOT counted as TP, TN, FP, or FN.
    Accuracy is calculated as (TP + TN) / Total Samples.
    
    Args:
        df: DataFrame from convert_to_dataframe()
        
    Returns:
        Dictionary with metrics
    """
    if len(df) == 0:
        return {}

    metrics = {}
    
    # Do NOT fill NaN (treat "None" as third category)
    predictions = df['prediction']
    
    # Calculate confusion matrix components
    # Note: comparison with NaN returns False, so NaN is excluded from all these
    tp = ((df['ground_truth'] == True) & (predictions == True)).sum()
    tn = ((df['ground_truth'] == False) & (predictions == False)).sum()
    fp = ((df['ground_truth'] == False) & (predictions == True)).sum()
    fn = ((df['ground_truth'] == True) & (predictions == False)).sum()

    # Calculate counts for valid answers
    total_pos = (df['ground_truth'] == True).sum()
    total_neg = (df['ground_truth'] == False).sum()
    nan_count = predictions.isna().sum()

    # Calculate Positive Class Metrics
    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / total_pos if total_pos > 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    # Calculate Negative Class Metrics (for Macro F1)
    # Precision Negative = TN / (TN + FN) (Predicted False)
    pred_neg = tn + fn
    precision_neg = tn / pred_neg if pred_neg > 0 else 0
    # Recall Negative = TN / Total Negatives
    recall_neg = tn / total_neg if total_neg > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    # Macro F1
    macro_f1 = (f1_pos + f1_neg) / 2

    return {
        'accuracy': accuracy,
        'precision': precision_pos,
        'recall': recall_pos,
        'f1': f1_pos,
        'macro_f1': macro_f1,
        'f1_neg': f1_neg,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_samples': int(total),
        'nan_count': int(nan_count)
    }

def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print(f"QA RESULTS ANALYSIS (GLM-4.6 NoSearch)")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = PROJECT_ROOT / "results" / "qa_analysis_nosearch" / "glm-4.6"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load data
    print(f"Loading QA results...")
    qa_results = load_qa_results()

    # Convert to DataFrame
    print("\nConverting to DataFrame...")
    df = convert_to_dataframe(qa_results)
    
    if len(df) == 0:
        print("No results found. Exiting.")
        return

    print(f"Total samples: {len(df)}\n")

    # Save DataFrame to CSV
    csv_path = output_dir / 'qa_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved DataFrame to {csv_path}\n")

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(df)

    # Print metrics
    print("\nPerformance Metrics (All Samples):")
    print("-" * 80)
    print(f"  Total Samples: {metrics['total_samples']}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score (Binary):  {metrics['f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print(f"  NaN/Unknown: {metrics['nan_count']}")
    
    # Calculate metrics by edge type
    edge_metrics = {}
    for edge_type in ['true_positive_edges', 'true_negative_edges']:
        print(f"\n{edge_type.replace('_', ' ').title()}:")
        subset_df = df[df['edge_type'] == edge_type]
        m = calculate_metrics(subset_df)
        edge_metrics[edge_type] = m
        
        if m:
            print(f"  Total Samples: {m['total_samples']}")
            print(f"  Accuracy:  {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1 Score (Binary):  {m['f1']:.4f}")
            print(f"  Macro F1: {m['macro_f1']:.4f}")
            print(f"  TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")
            print(f"  NaN/Unknown: {m['nan_count']}")
        else:
            print("  No samples found.")

    # Save metrics to JSON
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'edge_type_metrics': edge_metrics,
            'total_samples': len(df)
        }, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All results saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
