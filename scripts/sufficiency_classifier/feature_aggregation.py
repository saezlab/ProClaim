"""
Test script for feature aggregation across multiple evidence papers.

This script loads pre-extracted features from extract_features_scifact.py
and tests the aggregation functions defined in Section 2 of classifier_plan.md.

For a single claim with multiple evidence papers (N >= 2), it:
1. Loads local feature vectors {v1, v2, ..., vN}
2. Applies aggregation functions (Metadata, NLP, Cross-Features)
3. Produces a global vector V_global
4. Saves the aggregated features for classifier training

Usage:
    # First, check available feature files
    ls -lt results/sufficiency_classifier/scifact_feature_extraction/features_*.json

    # Then run aggregation with actual filename
    uv run scripts/sufficiency_classifier/feature_aggregation.py \\
        --features-file results/sufficiency_classifier/scifact_feature_extraction/features_YYYYMMDD_HHMMSS.json \\
        --min-papers 2

    # For demo/testing with provided sample data
    uv run scripts/sufficiency_classifier/feature_aggregation.py \
        --features-file results/sufficiency_classifier/scifact_feature_extraction/test_features_demo.json

    uv run scripts/sufficiency_classifier/feature_aggregation.py \
        --features-file results/sufficiency_classifier/scifact_feature_extraction/test_features_demo.json \
        --min-papers 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))


class FeatureAggregator:
    """
    Aggregates local feature vectors from multiple papers into a global vector.

    Based on Section 2 of classifier_plan.md:
    - Metadata Aggregation (Prior Credibility)
    - NLP Feature Aggregation (Semantic Signal & Consensus)
    - Cross-Features (Signal × Quality Interaction)
    """

    def __init__(self, current_year: Optional[int] = None):
        """
        Initialize the aggregator.

        Args:
            current_year: The current year for temporal calculations.
                         If None, uses datetime.now().year
        """
        self.current_year = current_year if current_year is not None else datetime.now().year
        self.lambda_decay = 0.1  # Temporal decay parameter

    def aggregate_metadata(self, papers: list[dict]) -> dict[str, float]:
        """
        Aggregate metadata features from multiple papers.

        Args:
            papers: List of paper feature records with metadata_features

        Returns:
            Dictionary of aggregated metadata features
        """
        if not papers:
            return {}

        # Extract valid metadata
        valid_papers = [p for p in papers if p.get("metadata_features")]
        if not valid_papers:
            return {"num_papers": len(papers)}

        # Extract fields
        years = [m["publication_year"] for m in
                [p["metadata_features"] for p in valid_papers]
                if m.get("publication_year")]

        log_ifs = [m["log_impact_factor"] for m in
                  [p["metadata_features"] for p in valid_papers]
                  if m.get("log_impact_factor")]

        norm_citations = [m["normalized_citation_count"] for m in
                         [p["metadata_features"] for p in valid_papers]
                         if m.get("normalized_citation_count")]

        h_indices = [m["author_h_index_max"] for m in
                    [p["metadata_features"] for p in valid_papers]
                    if m.get("author_h_index_max")]

        agg = {
            "num_papers": len(papers),
            "num_papers_with_metadata": len(valid_papers)
        }

        # Authority & Quality
        if log_ifs:
            agg["max_log_IF"] = float(np.max(log_ifs))
            agg["mean_log_IF"] = float(np.mean(log_ifs))

        if h_indices:
            agg["max_h_index"] = float(np.max(h_indices))
            agg["avg_max_h_index"] = float(np.mean(h_indices))

        # Community Attention
        if norm_citations:
            agg["max_norm_citation"] = float(np.max(norm_citations))
            agg["mean_norm_citation"] = float(np.mean(norm_citations))

        # Temporal Footprint
        if years:
            agg["latest_year_age"] = self.current_year - max(years)
            agg["year_span"] = max(years) - min(years)
            agg["oldest_year"] = int(min(years))
            agg["newest_year"] = int(max(years))

        return agg

    def aggregate_nlp(self, papers: list[dict]) -> dict[str, float]:
        """
        Aggregate NLP features from multiple papers.

        Based on Section 2 of classifier_plan.md, computes:
        - Voting & Consensus:
            - entailment_ratio: Proportion where argmax(NLI) == Entailment
            - contradiction_ratio: Proportion where argmax(NLI) == Contradiction
            - controversy_index: Shannon entropy of stance distribution
        - Statistical Pooling:
            - max_entity_coverage, mean_entity_coverage
            - max_similarity, mean_similarity

        Args:
            papers: List of paper feature records with nlp_features

        Returns:
            Dictionary of aggregated NLP features (only those specified in plan)
        """
        if not papers:
            return {}

        # Extract valid NLP features
        valid_papers = [p for p in papers if p.get("nlp_features")]
        if not valid_papers:
            return {}

        nlp_features = [p["nlp_features"] for p in valid_papers]

        # Extract NLI scores
        entailments = [f["nli_entailment"] for f in nlp_features
                      if f.get("nli_entailment") is not None]
        contradictions = [f["nli_contradiction"] for f in nlp_features
                         if f.get("nli_contradiction") is not None]
        neutrals = [f["nli_neutral"] for f in nlp_features
                   if f.get("nli_neutral") is not None]

        # Entity coverage
        coverages = [f["claim_entity_coverage"] for f in nlp_features
                    if f.get("claim_entity_coverage") is not None]

        # Semantic similarity
        similarities = [f["semantic_similarity"] for f in nlp_features
                       if f.get("semantic_similarity") is not None]

        agg = {}

        # Voting & Consensus (argmax-based stance)
        if entailments and contradictions and neutrals:
            n_samples = len(entailments)

            # Count stance votes (argmax)
            stance_votes = {"entailment": 0, "contradiction": 0, "neutral": 0}
            for e, c, n in zip(entailments, contradictions, neutrals):
                probs = [e, c, n]
                max_idx = np.argmax(probs)
                if max_idx == 0:
                    stance_votes["entailment"] += 1
                elif max_idx == 1:
                    stance_votes["contradiction"] += 1
                else:
                    stance_votes["neutral"] += 1

            # Ratios (only entailment and contradiction as per plan)
            agg["entailment_ratio"] = stance_votes["entailment"] / n_samples
            agg["contradiction_ratio"] = stance_votes["contradiction"] / n_samples

            # Controversy Index (Entropy)
            # Note: p_neu is computed internally for entropy but not returned as a feature
            p_ent = agg["entailment_ratio"]
            p_con = agg["contradiction_ratio"]
            p_neu = stance_votes["neutral"] / n_samples

            # Shannon entropy H = -Σ p_i log2(p_i)
            entropy = 0.0
            for p in [p_ent, p_con, p_neu]:
                if p > 0:
                    entropy -= p * np.log2(p)
            agg["controversy_index"] = float(entropy)

        # Statistical Pooling
        if coverages:
            agg["max_entity_coverage"] = float(np.max(coverages))
            agg["mean_entity_coverage"] = float(np.mean(coverages))

        if similarities:
            agg["max_similarity"] = float(np.max(similarities))
            agg["mean_similarity"] = float(np.mean(similarities))

        return agg

    def compute_cross_features(self, papers: list[dict]) -> dict[str, float]:
        """
        Compute cross-features: Signal × Quality interaction.

        Three weighting schemes:
        1. Quality-Weighted Stance (by log_impact_factor)
        2. Attention-Weighted Stance (by normalized_citation_count)
        3. Temporal-Decay Stance (by recency)

        Each scheme produces:
        - Weighted_Entailment
        - Weighted_Contradiction

        Args:
            papers: List of paper feature records

        Returns:
            Dictionary of cross-features
        """
        if not papers:
            return {}

        # Filter papers with both metadata and NLP features
        valid_papers = [
            p for p in papers
            if p.get("metadata_features") and p.get("nlp_features")
        ]

        if not valid_papers:
            return {}

        cross = {}

        # 1. Quality-Weighted Stance (by log_IF)
        weighted_ent_if = 0.0
        weighted_con_if = 0.0
        for p in valid_papers:
            meta = p["metadata_features"]
            nlp = p["nlp_features"]

            log_if = meta.get("log_impact_factor")
            p_ent = nlp.get("nli_entailment")
            p_con = nlp.get("nli_contradiction")

            if log_if is not None and p_ent is not None and p_con is not None:
                weighted_ent_if += log_if * p_ent
                weighted_con_if += log_if * p_con

        cross["weighted_entailment_IF"] = float(weighted_ent_if)
        cross["weighted_contradiction_IF"] = float(weighted_con_if)

        # 2. Attention-Weighted Stance (by normalized_citation_count)
        weighted_ent_cit = 0.0
        weighted_con_cit = 0.0
        for p in valid_papers:
            meta = p["metadata_features"]
            nlp = p["nlp_features"]

            norm_cit = meta.get("normalized_citation_count")
            p_ent = nlp.get("nli_entailment")
            p_con = nlp.get("nli_contradiction")

            if norm_cit is not None and p_ent is not None and p_con is not None:
                weighted_ent_cit += norm_cit * p_ent
                weighted_con_cit += norm_cit * p_con

        cross["weighted_entailment_citation"] = float(weighted_ent_cit)
        cross["weighted_contradiction_citation"] = float(weighted_con_cit)

        # 3. Temporal-Decay Stance (by recency)
        weighted_ent_time = 0.0
        weighted_con_time = 0.0
        for p in valid_papers:
            meta = p["metadata_features"]
            nlp = p["nlp_features"]

            year = meta.get("publication_year")
            p_ent = nlp.get("nli_entailment")
            p_con = nlp.get("nli_contradiction")

            if year is not None and p_ent is not None and p_con is not None:
                # Temporal decay weight: e^(-λ * age)
                age = self.current_year - year
                weight = np.exp(-self.lambda_decay * age)
                weighted_ent_time += weight * p_ent
                weighted_con_time += weight * p_con

        cross["weighted_entailment_temporal"] = float(weighted_ent_time)
        cross["weighted_contradiction_temporal"] = float(weighted_con_time)

        return cross

    def aggregate_all(self, papers: list[dict]) -> dict:
        """
        Aggregate all features (metadata, NLP, cross-features) into a global vector.

        Args:
            papers: List of paper feature records

        Returns:
            Dictionary containing all aggregated features
        """
        metadata_agg = self.aggregate_metadata(papers)
        nlp_agg = self.aggregate_nlp(papers)
        cross_features = self.compute_cross_features(papers)

        return {
            "metadata_aggregation": metadata_agg,
            "nlp_aggregation": nlp_agg,
            "cross_features": cross_features
        }


def load_extracted_features(filepath: Path) -> list[dict]:
    """Load extracted features from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def group_by_claim(features: list[dict]) -> dict[int, list[dict]]:
    """
    Group feature records by claim_id.

    Returns:
        Dictionary mapping claim_id -> list of paper feature records
    """
    grouped = {}
    for record in features:
        claim_id = record["claim_id"]
        if claim_id not in grouped:
            grouped[claim_id] = []
        grouped[claim_id].append(record)
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Test Feature Aggregation for SciFact")
    parser.add_argument(
        "--features-file",
        type=str,
        required=True,
        help="Path to extracted features JSON file (from extract_features_scifact.py)"
    )
    parser.add_argument(
        "--min-papers",
        type=int,
        default=2,
        help="Minimum number of papers required per claim for aggregation test (default: 2)"
    )
    args = parser.parse_args()

    features_path = Path(args.features_file)
    if not features_path.exists():
        print(f"Error: Features file not found: {features_path}")
        print("\nAvailable feature files:")
        result_dir = PROJECT_ROOT / "results" / "claude_sdk" / "scifact_feature_extraction"
        if result_dir.exists():
            feature_files = sorted(result_dir.glob("features_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if feature_files:
                for i, f in enumerate(feature_files[:5], 1):
                    print(f"  {i}. {f.relative_to(PROJECT_ROOT)}")
                if len(feature_files) > 5:
                    print(f"  ... and {len(feature_files) - 5} more")
            else:
                print("  (No feature files found)")
                print("\nRun extract_features_scifact.py first to generate features:")
                print("  uv run scripts/sufficiency_classifier/extract_features_scifact.py --limit 5")
        return

    print(f"Loading extracted features from: {features_path}")
    features = load_extracted_features(features_path)
    print(f"Loaded {len(features)} feature records.")

    # Group by claim
    grouped = group_by_claim(features)
    print(f"\nFound {len(grouped)} unique claims.")

    # Filter claims with multiple papers
    multi_paper_claims = {
        cid: papers for cid, papers in grouped.items()
        if len(papers) >= args.min_papers
    }

    print(f"Claims with >= {args.min_papers} papers: {len(multi_paper_claims)}")

    if not multi_paper_claims:
        print("No claims with multiple papers found. Cannot test aggregation.")
        print("Run extract_features_scifact.py with --limit or without --doc-id to extract multiple papers per claim.")
        return

    # Initialize aggregator (uses current year by default)
    aggregator = FeatureAggregator()

    # Test aggregation on each claim
    aggregated_results = []

    for claim_id, papers in multi_paper_claims.items():
        print(f"\n{'='*80}")
        print(f"Claim ID: {claim_id}")
        print(f"Number of Papers: {len(papers)}")
        print(f"Claim Text: {papers[0]['claim_text'][:100]}...")

        # Perform aggregation
        agg_features = aggregator.aggregate_all(papers)

        # Display results
        print("\n--- Metadata Aggregation ---")
        for k, v in agg_features["metadata_aggregation"].items():
            print(f"  {k}: {v}")

        print("\n--- NLP Aggregation ---")
        for k, v in agg_features["nlp_aggregation"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\n--- Cross-Features ---")
        for k, v in agg_features["cross_features"].items():
            print(f"  {k}: {v:.4f}")

        # Collect for output
        aggregated_results.append({
            "claim_id": claim_id,
            "claim_text": papers[0]["claim_text"],
            "evidence_papers": [
                {
                    "doc_id": p["doc_id"],
                    "pmid": p.get("found_pmid"),
                    "evidence_label": p.get("evidence_label")
                }
                for p in papers
            ],
            "aggregated_features": agg_features
        })

    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = PROJECT_ROOT / "results" / "claude_sdk" / "scifact_feature_extraction" / f"aggregated_{timestamp}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved {len(aggregated_results)} aggregated feature vectors to:")
    print(f"  {out_file}")


if __name__ == "__main__":
    main()
