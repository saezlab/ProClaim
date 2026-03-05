"""
Generate classifier training data (aggregated feature vectors) for the
Sufficiency MLP classifier.

Pipeline:
1. Load scifact-open claims + original SciFact corpus (evidence)
2. Build evidence pools (positive / negative-conflict / negative-noise)
3. Per-paper feature extraction:
   - NLP: semantic similarity (SBERT), NLI best-chunk (DeBERTa),
          entity coverage (scispacy MCP)
   - Metadata: via PaperFeatureExtractor (using cached PMIDs)
4. Aggregate features across papers via FeatureAggregator
5. Output JSON compatible with train_mlp_classifier.py

Usage:
    # First, resolve PMIDs (one-time)
    uv run scripts/sufficiency_classifier/resolve_pmids.py

    # Then generate training data
    uv run scripts/sufficiency_classifier/generate_classifier_data.py \
      --limit_claims 3 \
      --output data/classifier_train_data.json

    uv run scripts/sufficiency_classifier/generate_classifier_data.py --output data/classifier_train_data.json

"""

import asyncio
import argparse
import json
import random
import logging
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from pkevolve.verification.feature_extractor import PaperFeatureExtractor
from scripts.sufficiency_classifier.extract_features_scifact import (
    SemanticSimilarityComputer,
    NLIEentailmentComputer,
    BiomedicalEntityExtractor,
    compute_recall_from_entities,
    clean_text,
)
from scripts.sufficiency_classifier.feature_aggregation import FeatureAggregator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed feature schema — must match train_mlp_classifier.py EXPECTED_FEATURES
EXPECTED_FEATURES = [
    # Metadata
    "num_papers", "num_papers_with_metadata", "num_full_text",
    "max_log_IF", "mean_log_IF",
    "max_h_index", "avg_max_h_index", "max_norm_citation", "mean_norm_citation",
    "latest_year_age", "year_span",
    # NLP
    "entailment_ratio", "contradiction_ratio", "controversy_index",
    "max_entity_coverage", "mean_entity_coverage", "max_similarity", "mean_similarity",
    # Cross-Features
    "weighted_entailment_IF", "weighted_contradiction_IF",
    "weighted_entailment_citation", "weighted_contradiction_citation",
    "weighted_entailment_temporal", "weighted_contradiction_temporal",
]


def flatten_features(agg: Dict) -> Dict[str, float]:
    """Flatten aggregated features into a fixed-size dict with all EXPECTED_FEATURES.

    Missing features are filled with 0.0 so every sample has the same keys.
    """
    flat: Dict[str, float] = {}
    for section in ("metadata_aggregation", "nlp_aggregation", "cross_features"):
        flat.update(agg.get(section, {}))

    return {k: float(flat.get(k, 0.0)) if isinstance(flat.get(k), (int, float)) else 0.0
            for k in EXPECTED_FEATURES}


# Paths
CLAIMS_PATH = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/claims.jsonl")
CORPUS_PATH = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl")
PMID_CACHE_FILE = PROJECT_ROOT / "data" / "doc_id_to_pmid_cache.json"
FULL_TEXT_DIR = PROJECT_ROOT / "results" / "claude_sdk" / "scifact_feature_extraction"


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file, skipping malformed lines."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data


def load_pmid_cache() -> Dict[str, str]:
    """Load cached doc_id -> PMID mapping."""
    if PMID_CACHE_FILE.exists():
        with open(PMID_CACHE_FILE) as f:
            cache = json.load(f)
        logging.info(f"Loaded PMID cache: {sum(1 for v in cache.values() if v)}/{len(cache)} resolved")
        return cache
    logging.warning("No PMID cache found. Run resolve_pmids.py first for metadata features.")
    return {}


# ---------------------------------------------------------------------------
# Feature Extraction (Per-Paper)
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts per-paper features using the same logic as extract_features_scifact.py.

    For each (claim, evidence_doc) pair, produces:
    {
        "doc_id": ...,
        "metadata_features": { ... } | None,
        "nlp_features": {
            "semantic_similarity": float,
            "claim_entity_coverage": float,
            "nli_entailment": float,
            "nli_contradiction": float,
            "nli_neutral": float,
        } | None,
    }
    """

    def __init__(self, pmid_cache: Dict[str, str], entity_extractor: BiomedicalEntityExtractor):
        self.meta_extractor = PaperFeatureExtractor()
        self.sim_computer = SemanticSimilarityComputer()
        self.nli_computer = NLIEentailmentComputer()
        self.entity_extractor = entity_extractor
        self.pmid_cache = pmid_cache
        # Cache claim entities to avoid redundant MCP calls
        self._claim_entity_cache: Dict[str, list] = {}

    async def _extract_entities_with_retry(self, text: str, label: str = "", max_retries: int = 3) -> list:
        """Extract entities via scispacy MCP with retry logic."""
        for attempt in range(1, max_retries + 1):
            entities = await self.entity_extractor.extract(text)
            if entities:
                logging.debug(f"Entity extraction ({label}): {len(entities)} entities found")
                return entities
            if attempt < max_retries:
                logging.warning(
                    f"Entity extraction ({label}) returned empty (attempt {attempt}/{max_retries}), retrying..."
                )
                await asyncio.sleep(0.5)  # brief pause before retry
        logging.warning(f"Entity extraction ({label}) failed after {max_retries} attempts")
        return []

    async def _get_claim_entities(self, claim_text: str) -> list:
        """Get entities for a claim (cached)."""
        if claim_text not in self._claim_entity_cache:
            entities = await self._extract_entities_with_retry(
                claim_text, label=f"claim: '{claim_text[:50]}...'"
            )
            self._claim_entity_cache[claim_text] = entities
        return self._claim_entity_cache[claim_text]

    def _extract_metadata(self, doc_id: str) -> Dict | None:
        """Extract metadata features via PaperFeatureExtractor using cached PMID."""
        pmid = self.pmid_cache.get(doc_id, "")
        if not pmid or not pmid.isdigit():
            return None
        try:
            meta = self.meta_extractor.extract_metadata(pmid)
            if meta:
                return meta.model_dump()
        except Exception as e:
            logging.warning(f"Metadata extraction failed for PMID {pmid}: {e}")
        return None

    async def _extract_nlp(self, claim_text: str, evidence_text: str) -> Dict | None:
        """
        Extract NLP features using best-chunk strategy from extract_features_scifact.py.

        - Semantic similarity: max-pool over chunks
        - NLI: find the most opinionated chunk (similarity-weighted), return unweighted probs
        - Entity coverage: via scispacy MCP
        """
        if not evidence_text or len(evidence_text.strip()) < 20:
            return None

        try:
            import torch

            # 1. Semantic Similarity
            chunk_similarities = self.sim_computer.compute(claim_text, evidence_text)
            similarity = float(chunk_similarities.max().item())

            # 2. NLI (best-chunk strategy)
            nli_scores = self.nli_computer.compute(claim_text, evidence_text)
            raw_probs = nli_scores["raw_probs"]  # (num_chunks, 3)
            ent_idx = nli_scores["ent_idx"]
            con_idx = nli_scores["con_idx"]
            neu_idx = nli_scores["neu_idx"]

            # Align lengths (chunking logic should match, but be safe)
            min_len = min(len(chunk_similarities), len(raw_probs))
            sims = chunk_similarities[:min_len].unsqueeze(1).cpu()

            if not isinstance(raw_probs, torch.Tensor):
                raw_probs = torch.tensor(raw_probs, device='cpu')
            probs = raw_probs[:min_len]

            # Weighted probs: similarity * NLI prob
            weighted_probs = probs * sims

            # Find chunk with highest opinionated weighted score
            opinion_weighted = torch.max(weighted_probs[:, ent_idx], weighted_probs[:, con_idx])
            best_idx = torch.argmax(opinion_weighted).item()

            # Return UNWEIGHTED probs of the best chunk
            best_probs = probs[best_idx].tolist()

            # 3. Entity Coverage (scispacy MCP with retry)
            claim_entities = await self._get_claim_entities(claim_text)
            evidence_entities = await self._extract_entities_with_retry(
                evidence_text, label=f"evidence ({len(evidence_text)} chars)"
            )
            coverage = compute_recall_from_entities(claim_entities, evidence_entities)

            if not claim_entities:
                logging.warning(f"No claim entities extracted - coverage will be 0")
            elif not evidence_entities:
                logging.warning(f"No evidence entities extracted - coverage will be 0")

            return {
                "semantic_similarity": similarity,
                "claim_entity_coverage": coverage,
                "nli_entailment": float(best_probs[ent_idx]),
                "nli_contradiction": float(best_probs[con_idx]),
                "nli_neutral": float(best_probs[neu_idx]),
            }

        except Exception as e:
            logging.warning(f"NLP extraction failed: {e}")
            return None

    async def extract(self, claim_text: str, doc: Dict) -> Dict:
        """Extract all features for a single (claim, document) pair."""
        doc_id = str(doc["doc_id"])

        # Strategy: Use full_text if available, else corpus abstract
        has_full_text = False
        full_text_path = FULL_TEXT_DIR / f"doc_{doc_id}" / "full_text.txt"
        if full_text_path.exists():
            with open(full_text_path, "r") as f:
                full_text_content = f.read()
            if full_text_content and len(full_text_content) > 1000:
                evidence_text = clean_text(full_text_content)
                has_full_text = True
                logging.debug(f"Doc {doc_id}: using full text ({len(evidence_text)} chars)")
            else:
                evidence_text = self._get_abstract_text(doc)
        else:
            evidence_text = self._get_abstract_text(doc)

        metadata = self._extract_metadata(doc_id)
        nlp = await self._extract_nlp(claim_text, evidence_text)

        return {
            "doc_id": doc_id,
            "has_full_text": has_full_text,
            "metadata_features": metadata,
            "nlp_features": nlp,
        }

    @staticmethod
    def _get_abstract_text(doc: Dict) -> str:
        """Get evidence text from corpus abstract."""
        abstract = doc.get("abstract", [])
        text = " ".join(abstract) if abstract else ""
        return clean_text(text)


# ---------------------------------------------------------------------------
# Evidence Pool Construction (Section 4 of classifier_plan.md)
# ---------------------------------------------------------------------------

def build_evidence_pools(
    claims: List[Dict],
    corpus: Dict[str, Dict],
    full_text_dir: Path,
    num_noise: int = 2,
    seed: int = 42,
    max_combos_per_claim: int = 41,
    noise_paper_counts: List[int] | None = None,
    target_neg_conflict: int | None = None,
    target_neg_noise: int | None = None,
) -> List[Tuple[Dict, List[Dict], int, str, List[str], List[str]]]:
    """
    Build (claim, papers, target_y, pool_type, doc_ids, labels) tuples.

    Positive (y=1): Unanimous SUPPORT or CONTRADICT
    Negative conflict (y=0): Mixed SUPPORT + CONTRADICT (expanded via combinations)
    Negative noise (y=0): NEI claims paired with random irrelevant papers (multiple variants)

    Args:
        max_combos_per_claim: Cap on number of combination-pools per negative_conflict claim.
        noise_paper_counts: List of paper counts for noise variants, e.g. [1, 2, 3].
                           If None, defaults to [num_noise].
        target_neg_conflict: If set, subsample negative_conflict pools to this target.
        target_neg_noise: If set, subsample negative_noise pools to this target.

    Returns:
        List of tuples: (claim, papers, target_y, pool_type, doc_ids, labels)
        where doc_ids is a list of document IDs and labels is a list of ground truth labels.
    """
    rng = random.Random(seed)
    pools = []
    neg_conflict_pools = []
    neg_noise_pools = []
    all_doc_ids = list(corpus.keys())

    if noise_paper_counts is None:
        noise_paper_counts = [num_noise]

    # Pre-compute documents with full text paths for noise sampling
    docs_with_full_text = []
    logging.info("Starting docs_with_full_text pre-computation...")
    if full_text_dir.exists():
        for doc_dir in full_text_dir.iterdir():
            if doc_dir.is_dir() and doc_dir.name.startswith("doc_"):
                doc_id_str = doc_dir.name[4:]
                if doc_id_str in corpus and (doc_dir / "full_text.txt").exists():
                    docs_with_full_text.append(doc_id_str)
            
    if not docs_with_full_text:
        logging.warning("No documents with full text found for noise sampling. Falling back to all corpus docs.")
        docs_with_full_text = all_doc_ids

    logging.info(f"Finished docs_with_full_text pre-computation. Found {len(docs_with_full_text)} docs. Starting claim loop...")
    
    for i, claim in enumerate(claims):
        if i % 10 == 0:
            logging.info(f"Processing claim {i+1}/{len(claims)}...")
        claim_text = claim["claim"]
        evidence = claim.get("evidence", {})

        # --- NEI Claims (no evidence): pair with random noise papers ---
        if not evidence:
            candidates = list(docs_with_full_text)
            for n_papers in noise_paper_counts:
                if len(candidates) >= n_papers:
                    sampled = rng.sample(candidates, n_papers)
                    noise_docs = [corpus[did] for did in sampled]
                    # NEI papers have no ground truth labels
                    noise_labels = ["NEI"] * len(sampled)
                    neg_noise_pools.append((claim, noise_docs, 0, "negative_noise", sampled, noise_labels))
            continue

        rel_docs = []
        labels = []
        doc_ids = []
        for doc_id_str, info in evidence.items():
            if doc_id_str in corpus:
                rel_docs.append(corpus[doc_id_str])
                labels.append(info.get("label", ""))
                doc_ids.append(doc_id_str)

        if not rel_docs:
            continue

        # --- Positive Samples (y=1): Unanimous consensus ---
        unique_labels = list(set(labels))
        valid_labels = ["SUPPORT", "CONTRADICT"]

        if len(unique_labels) == 1 and unique_labels[0] in valid_labels:
            # Generate sub-pools of size 1..N
            for n in range(1, len(rel_docs) + 1):
                sub_pool = rel_docs[:n]
                sub_doc_ids = doc_ids[:n]
                sub_labels = labels[:n]
                pools.append((claim, sub_pool, 1, f"positive_{unique_labels[0].lower()}", sub_doc_ids, sub_labels))

        # --- Negative Conflict (y=0): Conflicting evidence (expanded via combinations) ---
        elif any(l in valid_labels for l in unique_labels) and len([l for l in unique_labels if l in valid_labels]) > 1:
            # Generate valid combinations of size 2..N that retain both SUPPORT and CONTRADICT.
            # Use stratified sampling by (n_support, n_contradict) ratio bucket so that the
            # final sample is balanced across ratios (e.g. 1S:1C, 2S:1C, 2S:2C, 3S:2C ...)
            # rather than dominated by kS:1C cases.
            import math
            n = len(rel_docs)

            # --- Helper: build a combo entry ---
            def _make_combo(combo_indices):
                combo_docs = [rel_docs[idx] for idx in combo_indices]
                combo_doc_ids = [doc_ids[idx] for idx in combo_indices]
                combo_labels_list = [labels[idx] for idx in combo_indices]
                ns = combo_labels_list.count("SUPPORT")
                nc = combo_labels_list.count("CONTRADICT")
                return (claim, combo_docs, 0, "negative_conflict", combo_doc_ids, combo_labels_list), (ns, nc)

            # --- Collect combos bucketed by (n_support, n_contradict) ---
            # For very large search spaces, cap each bucket independently via reservoir sampling
            # to avoid memory explosion before we even reach the stratified-sampling step.
            total_combos_estimate = 2 ** n
            per_bucket_cap = max(1, max_combos_per_claim)  # generous per-bucket cap

            buckets: dict = {}  # (ns, nc) -> list of combos
            bucket_seen: dict = {}  # (ns, nc) -> count seen (for reservoir)

            use_reservoir = total_combos_estimate > max_combos_per_claim * 100

            if use_reservoir:
                logging.info(f"Claim {claim['id']}: Using stratified reservoir sampling for {n} docs")
                early_stop = False
                total_seen = 0
                for r in range(2, n + 1):
                    if early_stop:
                        break
                    for combo_indices in combinations(range(n), r):
                        combo_label_set = set(labels[idx] for idx in combo_indices)
                        if "SUPPORT" not in combo_label_set or "CONTRADICT" not in combo_label_set:
                            continue
                        entry, key = _make_combo(combo_indices)
                        total_seen += 1
                        bucket_seen[key] = bucket_seen.get(key, 0) + 1
                        bucket = buckets.setdefault(key, [])
                        if len(bucket) < per_bucket_cap:
                            bucket.append(entry)
                        else:
                            # Reservoir replace
                            j = rng.randint(0, bucket_seen[key] - 1)
                            if j < per_bucket_cap:
                                bucket[j] = entry
                        if total_seen > max_combos_per_claim * 1000:
                            logging.info(f"Claim {claim['id']}: Early stopping after {total_seen} combinations")
                            early_stop = True
                            break
            else:
                # Enumerate all valid combos, group by ratio bucket
                for r in range(2, n + 1):
                    for combo_indices in combinations(range(n), r):
                        combo_label_set = set(labels[idx] for idx in combo_indices)
                        if "SUPPORT" not in combo_label_set or "CONTRADICT" not in combo_label_set:
                            continue
                        entry, key = _make_combo(combo_indices)
                        buckets.setdefault(key, []).append(entry)

            # --- Stratified sampling: allocate equal slots across ratio buckets ---
            # Each bucket gets at most ceil(max_combos / n_buckets) samples, then
            # if total still exceeds the cap we do a final trim.
            ratio_keys = sorted(buckets.keys())
            n_buckets = len(ratio_keys)
            if n_buckets == 0:
                claim_combos = []
            else:
                per_bucket_quota = max(1, math.ceil(max_combos_per_claim / n_buckets))
                claim_combos = []
                for key in ratio_keys:
                    bucket = buckets[key]
                    if len(bucket) > per_bucket_quota:
                        bucket = rng.sample(bucket, per_bucket_quota)
                    claim_combos.extend(bucket)
                # Final trim if total overshoots (can happen due to ceiling)
                if len(claim_combos) > max_combos_per_claim:
                    claim_combos = rng.sample(claim_combos, max_combos_per_claim)

            neg_conflict_pools.extend(claim_combos)

    # --- Subsample to targets if specified ---
    if target_neg_conflict is not None and len(neg_conflict_pools) > target_neg_conflict:
        neg_conflict_pools = rng.sample(neg_conflict_pools, target_neg_conflict)
    if target_neg_noise is not None and len(neg_noise_pools) > target_neg_noise:
        neg_noise_pools = rng.sample(neg_noise_pools, target_neg_noise)

    logging.info(
        f"Pool counts before balancing: positive={len(pools)}, "
        f"neg_conflict={len(neg_conflict_pools)}, neg_noise={len(neg_noise_pools)}"
    )

    pools.extend(neg_conflict_pools)
    pools.extend(neg_noise_pools)

    return pools


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Generate classifier training data")
    parser.add_argument("--claims", default=str(CLAIMS_PATH))
    parser.add_argument("--corpus", default=str(CORPUS_PATH),
                        help="Original SciFact corpus (evidence docs)")
    parser.add_argument("--limit_claims", type=int, default=None,
                        help="Limit number of claims to process (for testing)")
    parser.add_argument("--num_noise", type=int, default=2,
                        help="Number of noise papers per claim for negative samples")
    parser.add_argument("--max_combos_per_claim", type=int, default=30,
                        help="Max combinations per negative_conflict claim")
    parser.add_argument("--noise_paper_counts", type=int, nargs="+", default=[1, 2, 3],
                        help="List of paper counts for noise variants (default: 1 2 3)")
    parser.add_argument("--target_neg_conflict", type=int, default=None,
                        help="Target number of negative_conflict pools (subsample if exceeded)")
    parser.add_argument("--target_neg_noise", type=int, default=191,
                        help="Target number of negative_noise pools (subsample if exceeded)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/classifier_train_data.json")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. Load Data
    logging.info("Loading SciFact data...")
    claims = load_jsonl(Path(args.claims))
    claims_with_evidence = [c for c in claims if c.get("evidence")]
    claims_nei = [c for c in claims if not c.get("evidence")]
    logging.info(f"Claims with evidence: {len(claims_with_evidence)}, NEI claims: {len(claims_nei)}")

    corpus_list = load_jsonl(Path(args.corpus))
    corpus = {str(d["doc_id"]): d for d in corpus_list}
    logging.info(f"Corpus: {len(corpus)} documents")

    pmid_cache = load_pmid_cache()

    # 2. Build Pools (all claims including NEI)
    all_claims = claims_with_evidence + claims_nei
    if args.limit_claims:
        active_claims = random.sample(
            all_claims,
            min(args.limit_claims, len(all_claims))
        )
    else:
        active_claims = all_claims

    pools = build_evidence_pools(
        active_claims, corpus, FULL_TEXT_DIR,
        num_noise=args.num_noise,
        seed=args.seed,
        max_combos_per_claim=args.max_combos_per_claim,
        noise_paper_counts=args.noise_paper_counts,
        target_neg_conflict=args.target_neg_conflict,
        target_neg_noise=args.target_neg_noise,
    )
    logging.info(f"Generated {len(pools)} total pools.")

    # Distribution
    dist = {}
    for p in pools:
        dist[p[3]] = dist.get(p[3], 0) + 1
    logging.info(f"Pool Distribution: {dist}")

    # 3. Setup Feature Extraction (async for scispacy MCP)
    logging.info("Initializing models (SBERT, NLI, scispacy MCP)...")
    entity_extractor = BiomedicalEntityExtractor()
    await entity_extractor.__aenter__()

    try:
        extractor = FeatureExtractor(pmid_cache, entity_extractor)
        aggregator = FeatureAggregator()

        # 4. Extract & Aggregate
        final_dataset = []

        for i, (claim, docs, y, p_type, doc_ids, doc_labels) in enumerate(pools):
            logging.info(
                f"Pool {i+1}/{len(pools)} | Type: {p_type} | y={y} | "
                f"Papers: {len(docs)}"
            )

            extracted_papers = []
            for d in docs:
                feat = await extractor.extract(claim["claim"], d)
                extracted_papers.append(feat)

            agg = aggregator.aggregate_all(extracted_papers)
            flat = flatten_features(agg)

            # Add num_full_text (not from aggregator, computed directly)
            flat["num_full_text"] = float(sum(
                1 for p in extracted_papers if p.get("has_full_text", False)
            ))

            # Count ground truth labels
            n_support = doc_labels.count("SUPPORT")
            n_contradict = doc_labels.count("CONTRADICT")
            n_nei = doc_labels.count("NEI")

            # Keep sample if it has any meaningful features
            if any(v != 0.0 for v in flat.values()):
                final_dataset.append({
                    "claim_id": claim["id"],
                    "claim_text": claim["claim"],
                    "target_y": y,
                    "pool_type": p_type,
                    "num_papers": len(docs),
                    "doc_ids": doc_ids,
                    "scifact_labels": doc_labels,
                    "n_support": n_support,
                    "n_contradict": n_contradict,
                    "n_nei": n_nei,
                    "features": flat,
                })

        # 5. Save
        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(final_dataset, f, indent=2)

        logging.info(f"Saved {len(final_dataset)} training samples to {out_path}")

        # Summary
        pos = sum(1 for d in final_dataset if d["target_y"] == 1)
        neg = sum(1 for d in final_dataset if d["target_y"] == 0)
        logging.info(f"Positive (y=1): {pos}, Negative (y=0): {neg}")

    finally:
        await entity_extractor.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
