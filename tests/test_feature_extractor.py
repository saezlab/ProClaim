#!/usr/bin/env python3
"""
Test script for PaperFeatureExtractor and NLP features.

Extracts metadata features for a few known PMIDs and prints the results.
And tests the new NLP features (Entity Overlap Ratio).
"""

import json
import sys
from pathlib import Path

# Path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pkevolve.verification.feature_extractor import PaperFeatureExtractor, compute_entity_overlap

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
        if features:
            print(json.dumps(features.model_dump(), indent=2))
            
            # Quick validation
            populated = sum(
                1 for v in features.model_dump().values()
                if v is not None
            )
            total = len(features.model_dump())
            print(f"  Fields populated: {populated}/{total}")
        else:
            print("  Failed to extract metadata.")

    # Batch test
    print(f"\n{'=' * 60}")
    print("Batch extraction test")
    print("=" * 60)
    batch = extractor.extract_batch(TEST_PMIDS[:2])
    print(f"Batch returned {len(batch)} feature vectors")
    for fv in batch:
        if fv:
             print(f"  PMID {fv.pmid}: year={fv.publication_year}, "
                  f"log_IF={fv.log_impact_factor:.2f} if fv.log_impact_factor else None, "
                  f"norm_cit={fv.normalized_citation_count:.2f} if fv.normalized_citation_count else None, "
                  f"author_h={fv.author_h_index_max}")

    # NLP Feature Test
    print(f"\n{'=' * 60}")
    print("NLP Feature Test (General NER)")
    print("=" * 60)

    # Case 1: Overlap
    # Entities: Barack Obama (PERSON), Paris (GPE)
    claim = "Barack Obama visited Paris."
    # Entities: Barack Obama (PERSON), France (GPE)
    evidence = "Barack Obama is the former president of the USA and visited France."
    
    # Expected overlap: Barack Obama (normalized: barack obama)
    print(f"\nClaim: {claim}")
    print(f"Evidence: {evidence}")
    nlp_vec = compute_entity_overlap(claim, evidence)
    print("Result:", json.dumps(nlp_vec.model_dump(), indent=2))

    if nlp_vec.entity_overlap_ratio is not None and nlp_vec.entity_overlap_ratio > 0:
        print("  [PASS] Overlap detected.")
    else:
        print("  [FAIL] No overlap detected.")

    # Case 2: No Overlap
    claim2 = "Apple released a new iPhone."
    evidence2 = "Microsoft updated Windows."
    print(f"\nClaim: {claim2}")
    print(f"Evidence: {evidence2}")
    nlp_vec2 = compute_entity_overlap(claim2, evidence2)
    print("Result:", json.dumps(nlp_vec2.model_dump(), indent=2))
    
    if nlp_vec2.entity_overlap_ratio == 0.0:
        print("  [PASS] Correctly detected zero overlap.")
    else:
        print(f"  [FAIL] Expected 0.0, got {nlp_vec2.entity_overlap_ratio}")

    # Case 3: Programmatic SciFact-Open Test
    print(f"\n{'=' * 60}")
    print("SciFact-Open Dataset Test (Programmatic)")
    print("=" * 60)

    claims_path = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/claims.jsonl")
    corpus_path = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl")

    if not claims_path.exists() or not corpus_path.exists():
        print("Skipping SciFact-Open test: Dataset files not found at expected path.")
        return

    # 1. First, scan corpus to get a set of available doc_ids (limit to first 10000 to be fast)
    # and cache their abstracts.
    corpus_map = {}
    print("  Scanning corpus for available documents...")
    try:
        with open(corpus_path) as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                try:
                    doc = json.loads(line)
                    corpus_map[int(doc["doc_id"])] = doc
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception as e:
        print(f"Error reading corpus file: {e}")
        return

    print(f"  Cached {len(corpus_map)} documents.")

    # 2. Find a claim that references one of these docs
    target_claim = None
    target_doc_id = None
    target_sentences = []
    target_label = None
    
    try:
        with open(claims_path) as f:
            for line in f:
                claim_data = json.loads(line)
                if claim_data.get("evidence"):
                    # Check if any evidence doc is in our corpus map
                    for doc_id_str, info in claim_data["evidence"].items():
                        did = int(doc_id_str)
                        if did in corpus_map:
                            target_doc_id = did
                            target_sentences = info.get("sentences", [])
                            target_label = info.get("label")
                            target_claim = claim_data
                            break
                if target_claim:
                    break
    except Exception as e:
        print(f"Error reading claims file: {e}")
        return

    if not target_claim:
        print("No claim found referencing the cached corpus documents (try increasing limit).")
        return

    # 3. Get evidence text
    doc = corpus_map[target_doc_id]
    abstract = doc.get("abstract", [])
    
    if target_sentences and all(idx < len(abstract) for idx in target_sentences):
        # Use rationale sentences if valid
        evidence_parts = [abstract[idx] for idx in target_sentences]
        evidence_text = " ".join(evidence_parts)
    else:
        # Fallback to full abstract
        evidence_text = " ".join(abstract)

    print(f"Claim ID: {target_claim['id']}")
    print(f"Claim Text: {target_claim['claim']}")
    print(f"Evidence Doc ID: {target_doc_id}")
    print(f"Label: {target_label}")
    print(f"Evidence Text Snippet: {evidence_text[:200]}...")
    
    # Compute Overlap
    try:
        nlp_vec_sci = compute_entity_overlap(target_claim["claim"], evidence_text)
        print("Result:", json.dumps(nlp_vec_sci.model_dump(), indent=2))

        if nlp_vec_sci.entity_overlap_ratio is not None and nlp_vec_sci.entity_overlap_ratio > 0:
            print("  [PASS] Real-world overlap detected.")
        else:
             print("  [WARNING] No overlap detected (this depends on the specific example).")
    except Exception as e:
        print(f"Error computing overlap: {e}")


if __name__ == "__main__":
    main()
