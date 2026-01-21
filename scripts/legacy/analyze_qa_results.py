#!/usr/bin/env python3
"""
Analyze QA and sufficient context results.

This script reads QA evaluation results from results/qa and sufficient context
results from data/papers, then performs analysis.
uv run python analyze_qa_results.py gpt-oss-120b
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
QA_RESULTS_DIR = PROJECT_ROOT / "results/qa/signor_ref"
PAPERS_DATA_DIR = PROJECT_ROOT / "data/papers/signor"

# Load matplotlib style
STYLE_FILE = PROJECT_ROOT / "src/pkevolve/utils/rw_visualization.mplstyle"
if STYLE_FILE.exists():
    plt.style.use(str(STYLE_FILE))


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse interaction information from filename.

    Filename format: SOURCE_TARGET_INTERACTION.json
    Example: AKT1_GSK3A_down-regulates.json

    Returns:
        Dictionary with 'source', 'target', 'interaction' keys
    """
    stem = filename.replace('.json', '')
    parts = stem.split('_')

    if len(parts) < 3:
        return {'source': None, 'target': None, 'interaction': None}

    return {
        'source': parts[0],
        'target': parts[1],
        'interaction': '_'.join(parts[2:])  # Handle interactions with underscores
    }


def load_sufficient_support_data() -> Dict[str, List[Dict]]:
    """
    Load sufficient support data from data/papers/signor directory.

    Returns:
        Dictionary with keys 'true_positive_edges' and 'true_negative_edges',
        each containing a list of paper data dictionaries with sufficient support info.
    """
    results = {
        'true_positive_edges': [],
        'true_negative_edges': []
    }

    for edge_type in ['true_positive_edges', 'true_negative_edges']:
        edge_dir = PAPERS_DATA_DIR / edge_type

        if not edge_dir.exists():
            print(f"Warning: {edge_dir} does not exist")
            continue

        json_files = list(edge_dir.glob("*.json"))
        print(f"Loading {len(json_files)} sufficient support files from {edge_type}...")

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


def load_qa_results(model_name: str = "gpt-oss-120b") -> Dict[str, List[Dict]]:
    """
    Load QA results from results/qa directory.

    Args:
        model_name: Model directory name (default: gpt-oss-120b)

    Returns:
        Dictionary with keys 'true_positive_edges' and 'true_negative_edges',
        each containing a list of result dictionaries.
    """
    results = {
        'true_positive_edges': [],
        'true_negative_edges': []
    }

    model_dir = QA_RESULTS_DIR / model_name

    for edge_type in ['true_positive_edges', 'true_negative_edges']:
        edge_dir = model_dir / edge_type

        if not edge_dir.exists():
            print(f"Warning: {edge_dir} does not exist")
            continue

        json_files = list(edge_dir.glob("*.json"))
        print(f"Loading {len(json_files)} QA result files from {edge_type}...")

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
                         sufficient_support_data: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Convert QA results and sufficient support data to a pandas DataFrame for analysis.

    Args:
        qa_results: Dictionary from load_qa_results()
        sufficient_support_data: Dictionary from load_sufficient_support_data()

    Returns:
        DataFrame with columns for each scenario and metadata
    """
    # Create a lookup dictionary for sufficient support data by filename
    support_lookup = {}
    for edge_type, data_list in sufficient_support_data.items():
        for data in data_list:
            support_lookup[data['filename']] = {
                'title_abstract_sufficient_support': data.get('title_abstract_sufficient_support'),
                'full_text_sufficient_support': data.get('full_text_sufficient_support')
            }

    rows = []

    for edge_type, results_list in qa_results.items():
        for result in results_list:
            # Parse filename to get source, target, interaction
            file_info = parse_filename(result['filename'])

            # Get sufficient support data
            support_data = support_lookup.get(result['filename'], {
                'title_abstract_sufficient_support': None,
                'full_text_sufficient_support': None
            })

            row = {
                'filename': result['filename'],
                'edge_type': edge_type,
                'ground_truth': edge_type == 'true_positive_edges',
                'source': file_info['source'],
                'target': file_info['target'],
                'interaction': file_info['interaction'],
                'question': result['question'],

                # Sufficient support
                'title_abstract_sufficient_support': support_data['title_abstract_sufficient_support'],
                'full_text_sufficient_support': support_data['full_text_sufficient_support'],

                # Title + Abstract scenario
                'abstract_prediction': result['scenario_title_abstract']['prediction'],
                'abstract_context_length': result['scenario_title_abstract']['search_context_length'],
                'abstract_reasoning_length': result['scenario_title_abstract']['reasoning_length'],

                # Title + Abstract + Full text scenario
                'fulltext_prediction': result['scenario_title_abstract_fulltext']['prediction'],
                'fulltext_context_length': result['scenario_title_abstract_fulltext']['search_context_length'],
                'fulltext_reasoning_length': result['scenario_title_abstract_fulltext']['reasoning_length'],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy, precision, recall, F1 (Binary & Macro) for both scenarios.

    IMPORTANT: NaN/None predictions are treated as a third category (Unknown).
    This means they are NOT counted as TP, TN, FP, or FN.
    Accuracy is calculated as (TP + TN) / Total Samples.
    Recall is calculated as TP / Total Positives (so Unknowns are penalized).

    Args:
        df: DataFrame from convert_to_dataframe()

    Returns:
        Dictionary with metrics for 'abstract' and 'fulltext' scenarios
    """
    metrics = {}

    for scenario in ['abstract', 'fulltext']:
        pred_col = f'{scenario}_prediction'

        # Do NOT fill NaN (treat "None" as third category)
        predictions = df[pred_col]

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

        metrics[scenario] = {
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

    return metrics


def calculate_metrics_by_condition(df_subset: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    """
    Helper function to calculate metrics for a given subset and prediction column.

    IMPORTANT: NaN/None predictions are treated as a third category (Unknown).
    Accuracy = (TP+TN)/Total. Recall = TP/Total_Pos.

    Args:
        df_subset: Subset of DataFrame to calculate metrics on
        pred_col: Column name for predictions

    Returns:
        Dictionary with metrics
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
        'total_samples': len(df_subset),
        'nan_count': int(nan_count)
    }


def calculate_comprehensive_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for all 18 conditions:
    - 6 pie chart categories (title_abstract × 3 edge types, full_text × 3 edge types)
    - 3 support conditions each (all, sufficient=1, sufficient=0)

    Args:
        df: DataFrame from convert_to_dataframe()

    Returns:
        Dictionary with metrics for all conditions
    """
    metrics = {}

    # Define scenarios matching the 6 pie charts
    scenarios = [
        ('title_abstract', 'abstract_prediction', 'title_abstract_sufficient_support',
         [('true_negative_edges', 'True Negative'), ('true_positive_edges', 'True Positive'), ('all', 'All Edges')]),
        ('full_text', 'fulltext_prediction', 'full_text_sufficient_support',
         [('true_negative_edges', 'True Negative'), ('true_positive_edges', 'True Positive'), ('all', 'All Edges')])
    ]

    for context_type, pred_col, support_col, edge_types in scenarios:
        for edge_filter, edge_label in edge_types:
            # Filter by edge type
            if edge_filter == 'all':
                df_edge = df
            else:
                df_edge = df[df['edge_type'] == edge_filter]

            # Calculate metrics for 3 support conditions
            # 1. All samples
            key_all = f"{context_type}_{edge_filter}_all"
            result = calculate_metrics_by_condition(df_edge, pred_col)
            if result:
                metrics[key_all] = result

            # 2. Sufficient support = 1
            df_sufficient_1 = df_edge[df_edge[support_col] == 1]
            key_suff_1 = f"{context_type}_{edge_filter}_sufficient_1"
            result = calculate_metrics_by_condition(df_sufficient_1, pred_col)
            if result:
                metrics[key_suff_1] = result

            # 3. Sufficient support = 0
            df_sufficient_0 = df_edge[df_edge[support_col] == 0]
            key_suff_0 = f"{context_type}_{edge_filter}_sufficient_0"
            result = calculate_metrics_by_condition(df_sufficient_0, pred_col)
            if result:
                metrics[key_suff_0] = result

    return metrics


def analyze_prediction_agreement(df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze agreement between abstract and fulltext predictions.

    IMPORTANT: NaN/None predictions are treated as a third category (Unknown).
    They are considered 'Incorrect' when compared to Ground Truth.

    Returns:
        Dictionary with counts for each agreement category
    """
    # Do not fill NaN
    abstract_pred = df['abstract_prediction']
    fulltext_pred = df['fulltext_prediction']

    # Calculate agreement (not strictly used in return dict but good to have)
    # Note: simple equality returns False for NaN, so NaNs technically 'disagree' with everything unless we handle it
    df['predictions_agree'] = (abstract_pred == fulltext_pred) | (abstract_pred.isna() & fulltext_pred.isna())

    df['both_correct'] = (abstract_pred == df['ground_truth']) & \
                         (fulltext_pred == df['ground_truth'])
    df['abstract_only_correct'] = (abstract_pred == df['ground_truth']) & \
                                   (fulltext_pred != df['ground_truth'])
    df['fulltext_only_correct'] = (abstract_pred != df['ground_truth']) & \
                                    (fulltext_pred == df['ground_truth'])
    df['both_wrong'] = (abstract_pred != df['ground_truth']) & \
                       (fulltext_pred != df['ground_truth'])

    agreement_counts = {
        'Both Correct': int(df['both_correct'].sum()),
        'Abstract Only': int(df['abstract_only_correct'].sum()),
        'Fulltext Only': int(df['fulltext_only_correct'].sum()),
        'Both Wrong': int(df['both_wrong'].sum())
    }

    return agreement_counts


def plot_sufficient_support_pie_charts(df: pd.DataFrame, output_dir: Path):
    """
    Plot 6 pie charts showing sufficient support distribution:
    - Row 1: Title+Abstract support for True Negative, True Positive, and All edges
    - Row 2: Full Text support for True Negative, True Positive, and All edges

    Args:
        df: DataFrame with sufficient support columns
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # fig.suptitle('Sufficient Support Distribution', fontsize=20, y=0.995)

    # Define colors for support levels
    colors = ['#e41a1c', '#4daf4a']  # Red for 0, Green for 1
    labels = ['Not Sufficient', 'Sufficient']

    # Row 1: Title + Abstract sufficient support
    scenarios = [
        ('true_negative_edges', 'True Negative Edges (0)'),
        ('true_positive_edges', 'True Positive Edges (1)'),
        ('all', 'All Edges')
    ]

    for idx, (edge_filter, title) in enumerate(scenarios):
        ax = axes[0, idx]

        # Filter data
        if edge_filter == 'all':
            data_subset = df
        else:
            data_subset = df[df['edge_type'] == edge_filter]

        # Count support levels
        support_counts = data_subset['title_abstract_sufficient_support'].value_counts().sort_index()

        # Ensure both 0 and 1 are present
        counts = [support_counts.get(0, 0), support_counts.get(1, 0)]
        total = sum(counts)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )

        ax.set_title(f'Title + Abstract\n{title}', fontsize=14, pad=10)

        # Make percentage text bold and add absolute count
        for autotext, count in zip(autotexts, counts):
            autotext.set_text(f"{autotext.get_text()}\n({count})")
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    # Row 2: Full Text sufficient support
    for idx, (edge_filter, title) in enumerate(scenarios):
        ax = axes[1, idx]

        # Filter data
        if edge_filter == 'all':
            data_subset = df
        else:
            data_subset = df[df['edge_type'] == edge_filter]

        # Count support levels
        support_counts = data_subset['full_text_sufficient_support'].value_counts().sort_index()

        # Ensure both 0 and 1 are present
        counts = [support_counts.get(0, 0), support_counts.get(1, 0)]
        total = sum(counts)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )

        ax.set_title(f'Full Text\n{title}', fontsize=14, pad=10)

        # Make percentage text bold and add absolute count
        for autotext, count in zip(autotexts, counts):
            autotext.set_text(f"{autotext.get_text()}\n({count})")
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    plt.tight_layout()

    # Save as PDF
    output_path_pdf = output_dir / 'sufficient_support_pie_charts.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved sufficient support pie charts to {output_path_pdf}")

    # Save as PNG
    output_path_png = output_dir / 'sufficient_support_pie_charts.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved sufficient support pie charts to {output_path_png}")

    plt.close()


def main(model_name: str = "gpt-oss-120b"):
    """Main analysis pipeline.

    Args:
        model_name: Model name to analyze (default: gpt-oss-120b)
    """
    print("=" * 80)
    print(f"QA RESULTS ANALYSIS - Model: {model_name}")
    print("=" * 80)
    print()

    # Create output directory for this specific model
    output_dir = PROJECT_ROOT / "results" / "qa_analysis" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load data
    print(f"Loading QA results for model: {model_name}...")
    qa_results = load_qa_results(model_name=model_name)

    print("\nLoading sufficient support data...")
    sufficient_support_data = load_sufficient_support_data()

    # Convert to DataFrame
    print("\nConverting to DataFrame...")
    df = convert_to_dataframe(qa_results, sufficient_support_data)
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
    for scenario in ['abstract', 'fulltext']:
        title = 'Title + Abstract' if scenario == 'abstract' else 'Title + Abstract + Fulltext'
        print(f"\n{title}:")
        m = metrics[scenario]
        print(f"  Total Samples: {m['total_samples']}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score (Binary):  {m['f1']:.4f}")
        print(f"  Macro F1: {m['macro_f1']:.4f}")
        print(f"  TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")

    # Calculate comprehensive metrics (18 conditions)
    print("\n" + "=" * 80)
    print("Comprehensive Metrics (18 Conditions):")
    print("=" * 80)
    comprehensive_metrics = calculate_comprehensive_metrics(df)

    # Organize and print by context type and edge type
    context_labels = {
        'title_abstract': 'Title + Abstract',
        'full_text': 'Full Text'
    }
    edge_labels = {
        'true_negative_edges': 'True Negative Edges',
        'true_positive_edges': 'True Positive Edges',
        'all': 'All Edges'
    }
    support_labels = {
        'all': 'All Samples',
        'sufficient_1': 'Sufficient Support = 1',
        'sufficient_0': 'Sufficient Support = 0'
    }

    for context in ['title_abstract', 'full_text']:
        print(f"\n{'='*80}")
        print(f"{context_labels[context]}")
        print(f"{'='*80}")

        for edge_type in ['true_negative_edges', 'true_positive_edges', 'all']:
            print(f"\n{'-'*80}")
            print(f"{edge_labels[edge_type]}")
            print(f"{'-'*80}")

            for support_type in ['all', 'sufficient_1', 'sufficient_0']:
                key = f"{context}_{edge_type}_{support_type}"
                if key in comprehensive_metrics:
                    m = comprehensive_metrics[key]
                    print(f"\n  {support_labels[support_type]}:")
                    print(f"    Total Samples: {m['total_samples']}")
                    print(f"    Accuracy:  {m['accuracy']:.4f}")
                    print(f"    Precision: {m['precision']:.4f}")
                    print(f"    Recall:    {m['recall']:.4f}")
                    print(f"    F1 Score (Binary):  {m['f1']:.4f}")
                    print(f"    Macro F1: {m['macro_f1']:.4f}")
                    print(f"    TP: {m['tp']}, TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}")

    # Analyze prediction agreement
    print("\n" + "-" * 80)
    print("Prediction Agreement Analysis:")
    print("-" * 80)
    agreement_counts = analyze_prediction_agreement(df)
    for category, count in agreement_counts.items():
        print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")

    # Sufficient support statistics
    print("\n" + "-" * 80)
    print("Sufficient Support Statistics:")
    print("-" * 80)
    print("\nTitle + Abstract Sufficient Support:")
    for edge_type in ['true_negative_edges', 'true_positive_edges']:
        subset = df[df['edge_type'] == edge_type]
        support_counts = subset['title_abstract_sufficient_support'].value_counts().sort_index()
        total = len(subset)
        print(f"  {edge_type}:")
        print(f"    Not Sufficient (0): {support_counts.get(0, 0)} ({support_counts.get(0, 0)/total*100:.1f}%)")
        print(f"    Sufficient (1): {support_counts.get(1, 0)} ({support_counts.get(1, 0)/total*100:.1f}%)")

    print("\nFull Text Sufficient Support:")
    for edge_type in ['true_negative_edges', 'true_positive_edges']:
        subset = df[df['edge_type'] == edge_type]
        support_counts = subset['full_text_sufficient_support'].value_counts().sort_index()
        total = len(subset)
        print(f"  {edge_type}:")
        print(f"    Not Sufficient (0): {support_counts.get(0, 0)} ({support_counts.get(0, 0)/total*100:.1f}%)")
        print(f"    Sufficient (1): {support_counts.get(1, 0)} ({support_counts.get(1, 0)/total*100:.1f}%)")

    # Basic statistics
    print("\n" + "-" * 80)
    print("Context Length Statistics:")
    print("-" * 80)
    print(f"\nAbstract context length:")
    print(f"  Mean: {df['abstract_context_length'].mean():.0f}")
    print(f"  Median: {df['abstract_context_length'].median():.0f}")
    print(f"  Min: {df['abstract_context_length'].min():.0f}")
    print(f"  Max: {df['abstract_context_length'].max():.0f}")

    print(f"\nFulltext context length:")
    print(f"  Mean: {df['fulltext_context_length'].mean():.0f}")
    print(f"  Median: {df['fulltext_context_length'].median():.0f}")
    print(f"  Min: {df['fulltext_context_length'].min():.0f}")
    print(f"  Max: {df['fulltext_context_length'].max():.0f}")

    print("\n" + "-" * 80)
    print("Reasoning Length Statistics:")
    print("-" * 80)
    print(f"\nAbstract reasoning length:")
    print(f"  Mean: {df['abstract_reasoning_length'].mean():.0f}")
    print(f"  Median: {df['abstract_reasoning_length'].median():.0f}")

    print(f"\nFulltext reasoning length:")
    print(f"  Mean: {df['fulltext_reasoning_length'].mean():.0f}")
    print(f"  Median: {df['fulltext_reasoning_length'].median():.0f}")

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    plot_sufficient_support_pie_charts(df, output_dir)

    # Save metrics to JSON
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'comprehensive_metrics': comprehensive_metrics,
            'agreement': agreement_counts,
            'total_samples': len(df)
        }, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")

    # Save comprehensive metrics to CSV
    save_comprehensive_metrics_to_csv(comprehensive_metrics, output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


def save_comprehensive_metrics_to_csv(metrics_dict: Dict[str, Dict[str, float]], output_dir: Path):
    """
    Save comprehensive metrics to a CSV file.

    Args:
        metrics_dict: Dictionary with metrics for all conditions
        output_dir: Directory to save the CSV
    """
    rows = []

    # Define order of iteration to reconstruct metadata from keys
    scenarios = ['title_abstract', 'full_text']
    edge_types = ['true_negative_edges', 'true_positive_edges', 'all']
    support_conditions = ['all', 'sufficient_1', 'sufficient_0']

    for context in scenarios:
        for edge_type in edge_types:
            for support in support_conditions:
                key = f"{context}_{edge_type}_{support}"

                if key in metrics_dict:
                    data = metrics_dict[key].copy()

                    # Add metadata
                    data['context_type'] = context
                    data['edge_type'] = edge_type
                    data['support_condition'] = support

                    rows.append(data)

    if not rows:
        print("No metrics found to save to CSV.")
        return

    df_metrics = pd.DataFrame(rows)

    # Reorder columns
    first_cols = ['context_type', 'edge_type', 'support_condition',
                  'total_samples', 'accuracy', 'precision', 'recall', 'f1', 'macro_f1']
    remaining_cols = [c for c in df_metrics.columns if c not in first_cols]

    df_metrics = df_metrics[first_cols + remaining_cols]

    output_path = output_dir / 'comprehensive_metrics_summary.csv'
    df_metrics.to_csv(output_path, index=False)
    print(f"Saved comprehensive metrics summary to {output_path}")


if __name__ == "__main__":
    import sys

    # Check if model name is provided as command line argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "gpt-oss-120b"  # Default model

    main(model_name=model_name)
