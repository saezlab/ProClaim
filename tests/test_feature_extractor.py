#!/usr/bin/env python3
"""
Test script for PaperFeatureExtractor.

Extracts metadata features for a few known PMIDs and prints the results.
"""

import json
import sys
from pathlib import Path

# Path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pkevolve.verification.feature_extractor import PaperFeatureExtractor


# Well-known PMIDs for testing
TEST_PMIDS = [
    "36194155",  # Nature 2022 — should have high IF
    "38954691",  # Recent paper — should have citations
    "16476930",  # Older paper (2006) — tests age normalization
]


def main():
    extractor = PaperFeatureExtractor()

    print("=" * 60)
    print("PaperFeatureExtractor — Metadata Test")
    print("=" * 60)

    for pmid in TEST_PMIDS:
        print(f"\n--- PMID: {pmid} ---")
        features = extractor.extract_metadata(pmid)
        print(json.dumps(features.model_dump(), indent=2))

        # Quick validation
        populated = sum(
            1 for v in features.model_dump().values()
            if v is not None
        )
        total = len(features.model_dump())
        print(f"  Fields populated: {populated}/{total}")

    # Batch test
    print(f"\n{'=' * 60}")
    print("Batch extraction test")
    print("=" * 60)
    batch = extractor.extract_batch(TEST_PMIDS[:2])
    print(f"Batch returned {len(batch)} feature vectors")
    for fv in batch:
        print(f"  PMID {fv.pmid}: year={fv.publication_year}, "
              f"log_IF={fv.log_impact_factor:.2f}, "
              f"norm_cit={fv.normalized_citation_count:.2f}, "
              f"author_h={fv.author_h_index_max}")


if __name__ == "__main__":
    main()
