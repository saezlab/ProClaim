#!/usr/bin/env python3
"""
Create unambiguous test set for τ-threshold ablation study.

This script:
1. Loads all 6 dataset variants (tau_0.00, 0.16, 0.20, 0.25, 0.33, 0.50)
2. Identifies samples that have the same label across ALL τ values (unambiguous)
3. Creates stratified train/test splits with unambiguous samples in test set
4. Saves splits for each τ value with identical test sets
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    from tau_config import TAU_THRESHOLDS
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from tau_config import TAU_THRESHOLDS


def stratified_train_test_split(
    indices: List[int],
    labels: List[int],
    test_size: float,
    random_seed: int
) -> Tuple[List[int], List[int]]:
    """
    Perform stratified train/test split manually.

    Args:
        indices: List of sample indices
        labels: List of labels (0 or 1) corresponding to indices
        test_size: Fraction of data to use for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, test_indices)
    """
    random.seed(random_seed)

    # Group indices by label
    positive_indices = [idx for idx, label in zip(indices, labels) if label == 1]
    negative_indices = [idx for idx, label in zip(indices, labels) if label == 0]

    # Shuffle both groups
    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    # Split each group
    n_positive_test = int(len(positive_indices) * test_size)
    n_negative_test = int(len(negative_indices) * test_size)

    positive_test = positive_indices[:n_positive_test]
    positive_train = positive_indices[n_positive_test:]

    negative_test = negative_indices[:n_negative_test]
    negative_train = negative_indices[n_negative_test:]

    # Combine
    test_indices = positive_test + negative_test
    train_indices = positive_train + negative_train

    # Shuffle combined sets
    random.shuffle(test_indices)
    random.shuffle(train_indices)

    return train_indices, test_indices


def load_datasets(data_dir: Path) -> Dict[float, List[dict]]:
    """Load all tau dataset variants."""
    tau_values = TAU_THRESHOLDS
    datasets = {}

    print("Loading datasets...")
    for tau in tau_values:
        file_path = data_dir / f"tau_{tau:.2f}.json"
        with open(file_path, 'r') as f:
            datasets[tau] = json.load(f)
        print(f"  tau_{tau:.2f}: {len(datasets[tau])} samples")

    return datasets


def identify_unambiguous_samples(datasets: Dict[float, List[dict]]) -> Tuple[List[int], List[int]]:
    """
    Identify samples that have the same label across all τ values.

    Returns:
        Tuple of (unambiguous_indices, ambiguous_indices)
    """
    tau_values = sorted(datasets.keys())
    n_samples = len(datasets[tau_values[0]])

    unambiguous_indices = []
    ambiguous_indices = []

    print("\nAnalyzing label consistency across τ values...")

    # Track statistics
    label_change_patterns = defaultdict(int)
    pool_type_changes = defaultdict(int)

    for i in range(n_samples):
        # Get labels and pool types for this sample across all tau values
        labels = [datasets[tau][i]['target_y'] for tau in tau_values]
        pool_types = [datasets[tau][i]['pool_type'] for tau in tau_values]
        claim_id = datasets[tau_values[0]][i]['claim_id']

        # Check if all labels are the same
        if len(set(labels)) == 1:
            unambiguous_indices.append(i)
        else:
            ambiguous_indices.append(i)

            # Track the pattern of label changes
            label_pattern = tuple(labels)
            label_change_patterns[label_pattern] += 1

            # Track pool type transitions
            pool_type_set = set(pool_types)
            if len(pool_type_set) > 1:
                pool_type_changes[f"{pool_types[0]} -> {pool_types[-1]}"] += 1

    print(f"\nLabel consistency analysis:")
    print(f"  Unambiguous samples: {len(unambiguous_indices)} ({len(unambiguous_indices)/n_samples*100:.1f}%)")
    print(f"  Ambiguous samples: {len(ambiguous_indices)} ({len(ambiguous_indices)/n_samples*100:.1f}%)")

    print(f"\nTop 5 label change patterns:")
    for pattern, count in sorted(label_change_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pattern}: {count} samples")

    print(f"\nPool type transitions in ambiguous samples:")
    for transition, count in sorted(pool_type_changes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {transition}: {count} samples")

    return unambiguous_indices, ambiguous_indices


def analyze_unambiguous_distribution(dataset: List[dict], unambiguous_indices: List[int]) -> None:
    """Analyze the distribution of labels and pool types in unambiguous samples."""
    print("\nUnambiguous sample distribution:")

    # Count labels
    labels = [dataset[i]['target_y'] for i in unambiguous_indices]
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    print(f"  Positive: {n_positive} ({n_positive/len(labels)*100:.1f}%)")
    print(f"  Negative: {n_negative} ({n_negative/len(labels)*100:.1f}%)")

    # Count pool types
    pool_types = [dataset[i]['pool_type'] for i in unambiguous_indices]
    pool_type_counts = defaultdict(int)
    for pt in pool_types:
        pool_type_counts[pt] += 1

    print(f"\n  Pool type distribution:")
    for pt, count in sorted(pool_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pt}: {count} ({count/len(pool_types)*100:.1f}%)")


def create_splits(
    datasets: Dict[float, List[dict]],
    unambiguous_indices: List[int],
    ambiguous_indices: List[int],
    test_size: float = 0.15,
    random_seed: int = 42
) -> Dict[float, Dict[str, List[dict]]]:
    """
    Create train/test splits for each τ value.

    Test set contains ALL unambiguous samples.
    Train set contains remaining ambiguous samples + some unambiguous for stratification.

    Actually, let's make it simpler:
    - Test set: stratified sample from unambiguous samples (15%)
    - Train set: remaining unambiguous + all ambiguous samples

    Returns:
        Dict mapping tau -> {'train': [...], 'test': [...]}
    """
    tau_values = sorted(datasets.keys())
    base_tau = tau_values[0]
    base_dataset = datasets[base_tau]

    # Get labels for unambiguous samples from base dataset
    unambiguous_labels = [base_dataset[i]['target_y'] for i in unambiguous_indices]

    print(f"\nCreating stratified train/test split...")
    print(f"  Test size: {test_size*100:.0f}%")
    print(f"  Random seed: {random_seed}")

    # Split unambiguous samples into train and test
    unambiguous_train_idx, unambiguous_test_idx = stratified_train_test_split(
        indices=unambiguous_indices,
        labels=unambiguous_labels,
        test_size=test_size,
        random_seed=random_seed
    )

    # Test set consists only of unambiguous samples
    test_indices = set(unambiguous_test_idx)
    # Train set consists of unambiguous_train + all ambiguous samples
    train_indices = set(unambiguous_train_idx + ambiguous_indices)

    print(f"\nSplit statistics:")
    print(f"  Test set: {len(test_indices)} samples (all unambiguous)")
    print(f"  Train set: {len(train_indices)} samples ({len(unambiguous_train_idx)} unambiguous + {len(ambiguous_indices)} ambiguous)")

    # Create splits for each tau value
    splits = {}
    for tau in tau_values:
        dataset = datasets[tau]

        train_data = [dataset[i] for i in sorted(train_indices)]
        test_data = [dataset[i] for i in sorted(test_indices)]

        splits[tau] = {
            'train': train_data,
            'test': test_data
        }

        # Print statistics for this tau value
        train_labels = [d['target_y'] for d in train_data]
        test_labels = [d['target_y'] for d in test_data]

        print(f"\n  tau_{tau:.2f}:")
        print(f"    Train: {len(train_data)} samples, {sum(train_labels)} positive ({sum(train_labels)/len(train_labels)*100:.1f}%)")
        print(f"    Test: {len(test_data)} samples, {sum(test_labels)} positive ({sum(test_labels)/len(test_labels)*100:.1f}%)")

    # Verify test sets are identical across all tau values
    print("\nVerifying test set consistency...")
    base_test = splits[tau_values[0]]['test']
    base_test_claim_ids = [d['claim_id'] for d in base_test]
    base_test_labels = [d['target_y'] for d in base_test]

    all_consistent = True
    for tau in tau_values[1:]:
        test_data = splits[tau]['test']
        test_claim_ids = [d['claim_id'] for d in test_data]
        test_labels = [d['target_y'] for d in test_data]

        if test_claim_ids != base_test_claim_ids:
            print(f"  ERROR: Test claim_ids differ for tau={tau}")
            all_consistent = False
        elif test_labels != base_test_labels:
            print(f"  ERROR: Test labels differ for tau={tau}")
            all_consistent = False

    if all_consistent:
        print("  ✓ All test sets are identical (same samples, same labels)")

    return splits


def save_splits(
    splits: Dict[float, Dict[str, List[dict]]],
    unambiguous_indices: List[int],
    ambiguous_indices: List[int],
    output_dir: Path
) -> None:
    """Save train/test splits and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving splits to {output_dir}...")

    # Save train/test splits for each tau value
    for tau, split_data in splits.items():
        train_file = output_dir / f"tau_{tau:.2f}_train.json"
        test_file = output_dir / f"tau_{tau:.2f}_test.json"

        with open(train_file, 'w') as f:
            json.dump(split_data['train'], f, indent=2)

        with open(test_file, 'w') as f:
            json.dump(split_data['test'], f, indent=2)

        print(f"  Saved tau_{tau:.2f}: train={len(split_data['train'])}, test={len(split_data['test'])}")

    # Save metadata about the test set
    base_tau = sorted(splits.keys())[0]
    test_data = splits[base_tau]['test']

    metadata = {
        'description': 'Unambiguous test set for τ-threshold ablation study',
        'n_test_samples': len(test_data),
        'n_unambiguous_total': len(unambiguous_indices),
        'n_ambiguous_total': len(ambiguous_indices),
        'test_claim_ids': [d['claim_id'] for d in test_data],
        'test_labels': [d['target_y'] for d in test_data],
        'test_pool_types': [d['pool_type'] for d in test_data],
        'label_distribution': {
            'positive': sum([d['target_y'] for d in test_data]),
            'negative': len(test_data) - sum([d['target_y'] for d in test_data])
        },
        'random_seed': 42,
        'test_size': 0.15
    }

    metadata_file = output_dir / 'test_set_info.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata to {metadata_file}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Create unambiguous test set for ablation study")
    parser.add_argument("--ablation_dir", type=str, default=None,
                       help="Directory containing ablation datasets")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for splits")
    parser.add_argument("--test_size", type=float, default=0.1808,
                       help="Fraction of unambiguous samples for test set (default: 0.1808 for ~109 samples)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = Path(args.ablation_dir) if args.ablation_dir else project_root / 'data' / 'ablation'
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'splits'

    print("="*80)
    print("Creating Unambiguous Test Set for τ-Threshold Ablation Study")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test size: {args.test_size:.4f}")
    print(f"Random seed: {args.seed}")

    # Load all datasets
    datasets = load_datasets(data_dir)

    # Identify unambiguous samples
    unambiguous_indices, ambiguous_indices = identify_unambiguous_samples(datasets)

    # Analyze unambiguous distribution
    analyze_unambiguous_distribution(datasets[0.00], unambiguous_indices)

    # Create train/test splits
    splits = create_splits(
        datasets=datasets,
        unambiguous_indices=unambiguous_indices,
        ambiguous_indices=ambiguous_indices,
        test_size=args.test_size,
        random_seed=args.seed
    )

    # Save splits
    save_splits(splits, unambiguous_indices, ambiguous_indices, output_dir)

    print("\n" + "="*80)
    print("Successfully created unambiguous test set!")
    print("="*80)


if __name__ == "__main__":
    main()
