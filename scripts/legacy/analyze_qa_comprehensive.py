#!/usr/bin/env python3
"""
Analyze QA and sufficient context results.

This script reads QA evaluation results from results/qa_comprehensive and sufficient context
results from data/papers, then performs analysis.

Usage:
    uv run python scripts/analyze_qa_comprehensive.py --model gpt-oss-120b --scenario full_text --ordering rater
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_QA_BASE = PROJECT_ROOT / "results/qa_comprehensive"
PAPERS_DATA_DIR = PROJECT_ROOT / "data/papers/qa_search"
RESULTS_ANALYSIS_BASE = PROJECT_ROOT / "results/qa_comprehensive_analysis"

# Load matplotlib style
STYLE_FILE = PROJECT_ROOT / "src/pkevolve/utils/rw_visualization.mplstyle"
if STYLE_FILE.exists():
    plt.style.use(str(STYLE_FILE))


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse interaction information from filename.
    
    Filename format: SOURCE_TARGET_INTERACTION.json
    Example: AKT1_GSK3A_down-regulates.json
    """
    stem = filename.replace('.json', '')
    parts = stem.split('_')
    
    if len(parts) < 3:
        return {'source': None, 'target': None, 'interaction': None}
        
    return {
        'source': parts[0],
        'target': parts[1],
        'interaction': '_'.join(parts[2:])
    }


def load_sufficient_support_data() -> Dict[str, Dict]:
    """
    Load sufficient support data from data/papers/qa_search directory.
    Returns a dictionary mapping filename -> support data.
    """
    paper_lookup = {}
    
    # We look in both true_positive and true_negative folders in qa_search
    # The actual structure is data/papers/qa_search/{true_positive|true_negative}/*.json
    # We'll just glob recursively or check specific folders.
    
    search_dirs = [
        PAPERS_DATA_DIR / "true_positive",
        PAPERS_DATA_DIR / "true_negative"
    ]
    
    total_files = 0
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        json_files = list(search_dir.glob("*.json"))
        # print(f"Loading {len(json_files)} papers from {search_dir.name}...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Key by filename
                    paper_lookup[json_file.name] = {
                        'title_abstract_sufficient_support': data.get('title_abstract_sufficient_support'),
                        'full_text_sufficient_support': data.get('full_text_sufficient_support')
                    }
                    total_files += 1
            except Exception as e:
                # print(f"Error loading {json_file.name}: {e}")
                pass
                
    print(f"Loaded sufficient support data for {total_files} papers.")
    return paper_lookup


def load_qa_results(model: str, scenario: str, ordering: str) -> Dict[str, List[Dict]]:
    """
    Load QA results from results/qa_comprehensive/{model}/{scenario}/{ordering}
    """
    results = {
        'true_positive_edges': [],
        'true_negative_edges': []
    }
    
    # Construct base path
    base_path = RESULTS_QA_BASE / model / scenario / ordering
    
    # Check if this specific ordering folder exists (e.g. 'shuffled' or 'rater')
    if not base_path.exists():
        # Fallback for folder structure variation check? No, enforce specific structure for now.
        print(f"Error: Results directory does not exist: {base_path}")
        return results

    print(f"Loading results from: {base_path}")

    for edge_type in ['true_positive_edges', 'true_negative_edges']:
        # Map folder names - they map to the subfolders inside ordering
        # qa_comprehensive.py saves them into 'true_positive' and 'true_negative' subfolders
        # Note: The subfolder name is 'true_positive' not 'true_positive_edges' in the results dir
        folder_name = "true_positive" if "positive" in edge_type else "true_negative"
        edge_dir = base_path / folder_name
        
        if not edge_dir.exists():
            # print(f"Warning: {edge_dir} does not exist")
            continue
            
        json_files = list(edge_dir.glob("*.json"))
        print(f"Loading {len(json_files)} QA result files from {edge_type} ({folder_name})...")
        
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


def convert_to_dataframe(qa_results: Dict[str, List[Dict]], 
                         paper_lookup: Dict[str, Dict],
                         scenario_arg: str) -> pd.DataFrame:
    """
    Convert QA results to DataFrame.
    """
    rows = []
    
    for edge_type, results_list in qa_results.items():
        for result in results_list:
            file_info = parse_filename(result['filename'])
            
            # --- AGGREGATE SUFFICIENT SUPPORT ---
            # result['paper_order'] contains the list of filenames used
            paper_order = result.get('paper_order', [])
            
            # If paper_order is empty (e.g. error case or old format), we have no support info
            # We track counts and binary status (if ANY paper is sufficient)
            
            abstract_suff_count = 0
            fulltext_suff_count = 0
            papers_found = 0
            missing_scores_count = 0
            
            for paper_filename in paper_order:
                if paper_filename in paper_lookup:
                    p_data = paper_lookup[paper_filename]
                    
                    # Handle possible None values if key exists but value is null, or if we defaulted to None
                    abs_val = p_data.get('title_abstract_sufficient_support')
                    ft_val = p_data.get('full_text_sufficient_support')
                    
                    if abs_val is not None:
                        abstract_suff_count += int(abs_val)
                    else:
                        missing_scores_count += 1
                        
                    if ft_val is not None:
                        fulltext_suff_count += int(ft_val)
                    
                    papers_found += 1
                else:
                    # potentially warn if paper not found?
                    pass
            
            # Binary flag: Is there at least one sufficient paper?
            is_abstract_sufficient = 1 if abstract_suff_count > 0 else 0
            is_fulltext_sufficient = 1 if fulltext_suff_count > 0 else 0
            
            # --- EXTRACT PREDICTION ---
            current_scenario = result.get('scenario', scenario_arg)
            res_data = result.get('result', {})
            prediction = res_data.get('prediction')
            reasoning = res_data.get('reasoning')
            
            # Prediction is kept as raw boolean (or None) to match analyze_qa_results.py logic
            
            row = {
                'filename': result['filename'],
                'edge_type': edge_type,
                'ground_truth': edge_type == 'true_positive_edges',
                'source': file_info['source'],
                'target': file_info['target'],
                'interaction': file_info['interaction'],
                'question': result.get('question'),
                'scenario': current_scenario,
                'ordering': result.get('ordering', 'unknown'),
                
                # Sufficient support (Aggregated)
                'title_abstract_sufficient_support': is_abstract_sufficient,
                'full_text_sufficient_support': is_fulltext_sufficient,
                'abstract_sufficient_count': abstract_suff_count,
                'fulltext_sufficient_count': fulltext_suff_count,
                'papers_lookup_found': papers_found,
                'missing_scores_count': missing_scores_count,
                
                # Prediction Data
                'prediction': prediction,
                'reasoning': reasoning,
                
                # Metadata
                'supporting_papers_count': result.get('supporting_papers_count'),
                'paper_limit_reached': res_data.get('context_limit_exceeded', False)
            }
            rows.append(row)
            
    return pd.DataFrame(rows)


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics for the current dataframe.
    """
    if df.empty:
        return {}
        
def calculate_metrics_by_condition(df_subset: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    """
    Helper function to calculate metrics for a given subset and prediction column.
    Matches logic from analyze_qa_results.py
    """
    if len(df_subset) == 0:
        return None

    # Do not fill NaN
    predictions = df_subset[pred_col]

    tp = ((df_subset['ground_truth'] == True) & (predictions == True)).sum()
    tn = ((df_subset['ground_truth'] == False) & (predictions == False)).sum()
    fp = ((df_subset['ground_truth'] == False) & (predictions == True)).sum()
    fn = ((df_subset['ground_truth'] == True) & (predictions == False)).sum()

    total_pos = (df_subset['ground_truth'] == True).sum()
    total_neg = (df_subset['ground_truth'] == False).sum()

    # Positive Class Metrics
    accuracy = (tp + tn) / len(df_subset)
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / total_pos if total_pos > 0 else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    # Negative Class Metrics (for Macro F1)
    # Precision Negative = TN / (TN + FN)
    pred_neg = tn + fn
    precision_neg = tn / pred_neg if pred_neg > 0 else 0
    # Recall Negative = TN / Total Negatives
    recall_neg = tn / total_neg if total_neg > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    macro_f1 = (f1_pos + f1_neg) / 2

    # Count NaN values for tracking
    nan_count = df_subset[pred_col].isna().sum()

    return {
        'total_samples': len(df_subset),
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
        'nan_count': int(nan_count)
    }

def calculate_comprehensive_metrics(df: pd.DataFrame, scenario_arg: str, sufficient_support_col: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for conditions similarly to analyze_qa_results.py
    for the current single scenario.
    
    Breakdown by sufficient support count (0 to 5).
    """
    metrics = {}
    pred_col = 'prediction'

    # Clean legacy call: scenarios list is just the one we are running
    # We want: 
    #   - All Edges
    #   - True Positive Edges
    #   - True Negative Edges
    # multiplied by Sufficient Support Count (0-5)
    
    edge_filters = [
        ('all', 'All Edges'),
        ('true_positive_edges', 'True Positive'),
        ('true_negative_edges', 'True Negative')
    ]
    
    for edge_filter, edge_label in edge_filters:
        # Filter by edge type
        if edge_filter == 'all':
            df_edge = df
        else:
            df_edge = df[df['edge_type'] == edge_filter]

        if df_edge.empty:
            continue

        # 1. All samples
        key_all = f"{scenario_arg}_{edge_filter}_all"
        result = calculate_metrics_by_condition(df_edge, pred_col)
        if result:
            metrics[key_all] = result

        # 2. Breakdown by Sufficient Support Count (0 to 5)
        # We assume max 5 papers, but we can iterate through unique values found + safeguard 0-5
        
        for count in range(6):
            df_suff = df_edge[df_edge[sufficient_support_col] == count]
            key_suff = f"{scenario_arg}_{edge_filter}_sufficient_{count}"
            result = calculate_metrics_by_condition(df_suff, pred_col)
            if result:
                metrics[key_suff] = result
            
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze QA results.")
    parser.add_argument('--model', type=str, default='gpt-oss-120b', help="Model name.")
    parser.add_argument('--scenario', type=str, required=True, choices=['title_abstract', 'full_text'],
                        help="Scenario to analyze.")
    parser.add_argument('--ordering', type=str, default='shuffled', choices=['shuffled', 'rater'],
                        help="ordering used (shuffled/rater).")
                        
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"ANALYZING: {args.model} | {args.scenario} | {args.ordering}")
    print("=" * 80)
    
    # 1. Load Data
    qa_results = load_qa_results(args.model, args.scenario, args.ordering)
    suff_data = load_sufficient_support_data()
    
    # 2. Convert to DataFrame
    df = convert_to_dataframe(qa_results, suff_data, args.scenario)
    
    if df.empty:
        print("No data found to analyze.")
        return

    # 3. Output Directory
    output_dir = RESULTS_ANALYSIS_BASE / args.model / args.scenario / args.ordering
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Calculate Comprehensive Metrics
    # Determine the sufficient support column to use for breakdown
    # WE USE THE COUNT COLUMN NOW
    if args.scenario == 'title_abstract':
        suff_col = 'abstract_sufficient_count'
    else:
        suff_col = 'fulltext_sufficient_count'
        
    print("Calculating comprehensive metrics (buckets 0-5)...")
    metrics = calculate_comprehensive_metrics(df, args.scenario, suff_col)
    
    # Print metrics
    print("\nPerformance Metrics:")
    print("-" * 80)
    
    # Printing helper
    def print_metric_block(title, m):
        if not m:
            return
        print(f"\n{title}:")
        print(f"  Total Samples: {m['total_samples']}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score (Binary):  {m['f1']:.4f}")
        print(f"  Macro F1: {m['macro_f1']:.4f}")
        print(f"  TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")
        print(f"  Invalid/Unknown: {m['nan_count']}")

    # Order of printing: All, then break down by True Positive Edges, True Negative Edges
    # And for each of those, break down by Sufficient Support Count
    
    edge_types = [
         ('all', 'All Edges'),
         ('true_positive_edges', 'True Positive Edges'),
         ('true_negative_edges', 'True Negative Edges')
    ]
    
    support_types = [('all', 'All Samples')] + \
                    [(f'sufficient_{i}', f'Sufficient Support Count = {i}') for i in range(6)]
    
    for edge_filter, edge_title in edge_types:
        print(f"\n{'='*40}")
        print(f"{edge_title}")
        print(f"{'='*40}")
        
        for supp_suffix, supp_title in support_types:
            key = f"{args.scenario}_{edge_filter}_{supp_suffix}"
            if key in metrics:
                print_metric_block(supp_title, metrics[key])
            else:
                # print(f"\n{supp_title}: No data")
                pass

    # 5. Save Results
    csv_path = output_dir / "analysis_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to: {csv_path}")
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")
    
    # Option: Plotting could be added here similar to before if needed
    # (Leaving out complex plotting for now to focus on data flow first)

if __name__ == "__main__":
    main()
