"""
Analyze SUPPORT/CONTRADICT ratio distribution in negative_conflict samples.

This script provides two analysis views:

1. Generated Data Analysis (Actual Training Data):
   Analyzes the final combinations produced by generate_classifier_data.py.
   (e.g., the actual proportion of kS:kC seen by the MLP during training)
   Run: uv run scripts/sufficiency_classifier/analyze_conflict_ratios.py --generated data/classifier_train_data.json

2. Raw Claim-Level Analysis (Source Data):
   Analyzes the original evidenced SciFact claims before combination expansion.
   (e.g., how many papers have "kS vs kC" total evidence before mixing)
   Run: uv run scripts/sufficiency_classifier/analyze_conflict_ratios.py --claims ... --corpus ...

uv run scripts/sufficiency_classifier/analyze_conflict_ratios.py --generated data/classifier_train_data.json --save results/ratio_analysis.json

"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict

# Paths
DEFAULT_CLAIMS = "/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/claims.jsonl"
DEFAULT_CORPUS = "/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl"


def load_jsonl(filepath: Path) -> List[Dict]:
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data


# ---------------------------------------------------------------------------
# Generated Data Analysis (Reads outputs of generate_classifier_data.py)
# ---------------------------------------------------------------------------

def analyze_generated_ratios(data_path: Path) -> Dict:
    with open(data_path, "r") as f:
        data = json.load(f)

    stats = {
        "total_samples": len(data),
        "pool_counts": Counter(d.get("pool_type", "unknown") for d in data),
        "total_conflict": 0,
        "combo_sizes": Counter(),
        "support_counts": Counter(),
        "contradict_counts": Counter(),
        "ratio_distribution": Counter(),
        "total_support_papers": 0,
        "total_contradict_papers": 0,
    }

    conflict_data = [d for d in data if d.get("pool_type") == "negative_conflict"]
    stats["total_conflict"] = len(conflict_data)

    if not conflict_data:
        return stats

    for d in conflict_data:
        ns = d.get("n_support", 0)
        nc = d.get("n_contradict", 0)
        size = ns + nc
        ratio_str = f"{ns}S:{nc}C"

        stats["combo_sizes"][size] += 1
        stats["support_counts"][ns] += 1
        stats["contradict_counts"][nc] += 1
        stats["ratio_distribution"][ratio_str] += 1
        stats["total_support_papers"] += ns
        stats["total_contradict_papers"] += nc

    return stats


def print_generated_analysis(stats: Dict, data_path: Path):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"GENERATED DATA RATIO ANALYSIS")
    print(f"Source: {data_path}")
    print(sep)

    print(f"\n📊 Overall Training Dataset Distribution:")
    print(f"  Total samples: {stats['total_samples']}")
    for pool, count in sorted(stats["pool_counts"].items()):
        print(f"  {pool}: {count} ({count/stats['total_samples']*100:.1f}%)")

    n_conflict = stats["total_conflict"]
    if n_conflict == 0:
        print("\n❌ No negative_conflict samples found in this dataset.")
        return

    print(f"\n🔬 Zooming into {n_conflict} 'negative_conflict' samples:")

    print(f"\n� Combination size distribution (Total papers per sample):")
    for size, c in sorted(stats["combo_sizes"].items()):
        print(f"  Size {size:2d}: {c:4d} samples ({c/n_conflict*100:5.1f}%)")

    print(f"\n⚖️  SUPPORT:CONTRADICT Ratio Distribution:")
    top_ratios = sorted(stats["ratio_distribution"].items(), key=lambda x: -x[1])
    for ratio_str, c in top_ratios:
        print(f"  {ratio_str:8s}: {c:4d} samples ({c/n_conflict*100:5.1f}%)")

    total_papers = stats["total_support_papers"] + stats["total_contradict_papers"]
    pct_sup = stats["total_support_papers"] / total_papers * 100
    pct_con = stats["total_contradict_papers"] / total_papers * 100

    print(f"\n📊 Aggregate Paper Statistics across all conflict samples:")
    print(f"  Total SUPPORT papers:    {stats['total_support_papers']} ({pct_sup:.1f}%)")
    print(f"  Total CONTRADICT papers: {stats['total_contradict_papers']} ({pct_con:.1f}%)")
    print(f"  Total papers:            {total_papers}")
    print(sep)


# ---------------------------------------------------------------------------
# Raw Claim-Level Analysis
# ---------------------------------------------------------------------------

def analyze_claim_ratios(claims: List[Dict], corpus: Dict[str, Dict]) -> Dict:
    stats: Dict = {
        "total_claims": len(claims),
        "claims_with_evidence": 0,
        "claims_with_conflict": 0,
        "support_counts": Counter(),
        "contradict_counts": Counter(),
        "ratio_distribution": Counter(),
    }

    for claim in claims:
        evidence = claim.get("evidence", {})
        if not evidence:
            continue
        stats["claims_with_evidence"] += 1

        labels = [info.get("label", "") for doc_id, info in evidence.items() if doc_id in corpus]
        if not labels:
            continue

        unique = set(labels)
        valid = {"SUPPORT", "CONTRADICT"}
        if len(unique & valid) < 2:
            continue

        n_sup = labels.count("SUPPORT")
        n_con = labels.count("CONTRADICT")
        ratio_str = f"{n_sup}S:{n_con}C"

        stats["claims_with_conflict"] += 1
        stats["support_counts"][n_sup] += 1
        stats["contradict_counts"][n_con] += 1
        stats["ratio_distribution"][ratio_str] += 1

    return stats


def print_claim_analysis(stats: Dict):
    sep = "=" * 80
    print(f"\n{sep}")
    print("RAW CLAIM-LEVEL RATIO ANALYSIS (Before combination mixing)")
    print(sep)

    n = stats["claims_with_conflict"]
    print(f"\n📊 Overall:")
    print(f"  Total claims:         {stats['total_claims']}")
    print(f"  Claims with evidence: {stats['claims_with_evidence']}")
    print(f"  Claims with conflict: {n}")

    print(f"\n⚖️  SUPPORT:CONTRADICT ratio per claim (Original SciFact labels):")
    for ratio, count in sorted(stats["ratio_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {ratio:8s}: {count:3d} claims ({count/n*100:5.1f}%)")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze SUPPORT/CONTRADICT ratios.")

    # Mutually exclusive groups: must provide --generated OR --claims/--corpus
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generated", type=str, metavar="PATH",
                       help="Path to generated training JSON (e.g., data/classifier_train_data.json)")
    group.add_argument("--claims", type=str, metavar="PATH",
                       help="Path to raw claims.jsonl (requires --corpus)")

    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS,
                        help="Path to corpus.jsonl (used with --claims)")
    parser.add_argument("--save", type=str, metavar="PATH",
                        help="Path to save the analysis report as JSON (e.g., results/ratio_analysis.json)")

    args = parser.parse_args()

    if args.generated:
        # Generated Data Analysis
        data_path = Path(args.generated)
        if not data_path.exists():
            print(f"File not found: {data_path}")
            return
        print(f"Analyzing generated dataset: {data_path}")
        stats = analyze_generated_ratios(data_path)
        print_generated_analysis(stats, data_path)
        
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n✅ Generated analysis saved to: {save_path}")

    elif args.claims:
        # Raw Claim-Level Analysis
        print(f"Loading claims from: {args.claims}")
        claims = load_jsonl(Path(args.claims))
        print(f"Loading corpus from: {args.corpus}")
        corpus = {str(d["doc_id"]): d for d in load_jsonl(Path(args.corpus))}

        stats = analyze_claim_ratios(claims, corpus)
        print_claim_analysis(stats)
        
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n✅ Claim-level analysis saved to: {save_path}")


if __name__ == "__main__":
    main()
