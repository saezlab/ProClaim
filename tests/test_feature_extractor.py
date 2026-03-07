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

from scripts.sufficiency_classifier.feature_extractor import PaperFeatureExtractor, compute_entity_overlap

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

    # ============================================================
    # End-to-End Feature Integration Test
    # ============================================================
    print(f"\n{'=' * 60}")
    print("SciFact-Open End-to-End Test Loop (3 Claims)")
    print("=" * 60)

    # 1. Load Data
    print("Loading data...")
    claims_path = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/claims.jsonl")
    corpus_path = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl")
    
    if not claims_path.exists() or not corpus_path.exists():
        print("Skipping: Dataset files not found.")
        return

    # Cache corpus (doc_id -> doc) - limit to first 20k for speed
    corpus_map = {}
    print("Scanning corpus...")
    with open(corpus_path, "r") as f:
        for i, line in enumerate(f):
            if i > 20000: break
            try:
                doc = json.loads(line)
                corpus_map[int(doc["doc_id"])] = doc
            except: pass
            
    print(f"Cached {len(corpus_map)} documents.")

    # Find 3 claims with evidence in our cache
    target_claims = []
    print("Finding claims...")
    with open(claims_path, "r") as f:
        for line in f:
            c = json.loads(line)
            if c.get("evidence"):
                for doc_id_str, info in c["evidence"].items():
                    did = int(doc_id_str)
                    if did in corpus_map:
                        target_claims.append((c, did))
                        break
            if len(target_claims) >= 3:
                break

    # 2. Process Loop
    for i, (claim, doc_id) in enumerate(target_claims):
        print(f"\n--- Case {i+1} ---")
        print(f"Claim ID: {claim['id']}")
        print(f"Claim Text: {claim['claim']}")
        
        evidence_doc = corpus_map[doc_id]
        print(f"Evidence Doc ID: {doc_id}")
        print(f"Evidence Title: {evidence_doc.get('title', 'N/A')}")
        
        # Check for PMID / Citation
        pmid = None
        # Heuristic: verify if it matches our known case 304905 => 26553255
        if doc_id == 304905:
            pmid = "26553255"
            print(f"PMID Found (Known): {pmid}")
        elif 'pmid' in evidence_doc.get('metadata', {}):
            pmid = evidence_doc['metadata']['pmid']
            print(f"PMID Found (Metadata): {pmid}")
        else:
            print(f"PMID Missing in Corpus. Referenced via S2ORC ID: {doc_id}")
            print(f"  To fetch full text, we would search PubMed for title: '{evidence_doc.get('title')}'")

        # Feature Extraction
        # A) Metadata (if PMID available)
        if pmid:
            try:
                meta_vec = extractor.extract_metadata(pmid)
                print("Metadata Features:", json.dumps(meta_vec.model_dump(), indent=2))
            except Exception as e:
                print(f"Metadata extraction failed: {e}")
        else:
            print("Skipping Metadata Features (No PMID).")

        # B) NLP Features
        # Use Abstract as 'Text' proxy
        abstract_text = " ".join(evidence_doc.get("abstract", []))
        if not abstract_text:
            print("Warning: Abstract is empty.")
        
        nlp_vec = compute_entity_overlap(claim['claim'], abstract_text)
        print("NLP Metrics:")
        print(f"  Claim Entities: {nlp_vec.claim_entities}")
        print(f"  Recall (Coverage): {nlp_vec.claim_entity_coverage}")
        if nlp_vec.claim_entity_coverage is not None and nlp_vec.claim_entity_coverage > 0:
             print("  [PASS] Relevant coverage detected.")


if __name__ == "__main__":
    main()
