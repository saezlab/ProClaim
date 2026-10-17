"""
Paper Feature Extractor — metadata feature extraction for the classifier.

Extracts per-paper metadata features (publication year, impact factor,
citation count, author h-index) by reusing search/paper_utils.py functions
and querying OpenAlex for author-level metrics.

Usage:
    from scripts.sufficiency_classifier.feature_extractor import PaperFeatureExtractor

    extractor = PaperFeatureExtractor()
    features = extractor.extract_metadata("36194155")
    print(features.model_dump_json(indent=2))
"""

import logging
import math
import time
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from proclaim.search.paper_utils import (
    get_journal_info_by_pmid,
    get_openalex_citation_count,
    get_pubmed_metadata,
    get_session,
)
from proclaim.verification.data_models import NLPFeatureVector, PaperFeatureVector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# General NER — Entity Overlap Ratio
# ---------------------------------------------------------------------------

_spacy_nlp = None


def _get_spacy_nlp():
    """Load spaCy NLP model (singleton, loaded on first call)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            # Try loading scispaCy model
            _spacy_nlp = spacy.load("en_core_sci_sm")
        except OSError:
            # Fallback or try to import it directly if not linked
            try:
                import en_core_sci_sm
                _spacy_nlp = en_core_sci_sm.load()
            except ImportError:
                print("Warning: en_core_sci_sm not found. Falling back to en_core_web_sm.")
                _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def extract_entities(text: str) -> set[str]:
    """Extract named entities from text using scispaCy (or fallback).

    Uses a biomedical NER model (en_core_sci_sm) to extract entities like
    genes, diseases, chemicals, etc.

    Returns a set of lowercased entity surface forms.
    Empty string returns an empty set.
    """
    if not text or not text.strip():
        return set()
    nlp = _get_spacy_nlp()
    doc = nlp(text)
    
    # scispaCy is trained to detect biomedical entities directly in doc.ents
    return {ent.text.lower() for ent in doc.ents}


def compute_entity_coverage(claim: str, evidence_text: str) -> NLPFeatureVector:
    """Compute Claim Entity Coverage (Recall) against an evidence text.

    Uses Recall: |Claim ∩ Evidence| / |Claim| where Claim and Evidence are
    the named entity sets extracted by scispaCy.

    Args:
        claim: The scientific claim text.
        evidence_text: The evidence text (abstract, sentence, etc.).

    Returns:
        NLPFeatureVector with extracted entities and coverage ratio.
        Ratio is None if the claim has no entities.
    """
    claim_ents = extract_entities(claim)
    evidence_ents = extract_entities(evidence_text)

    # Claim Entity Coverage (Recall)
    coverage = None
    if claim_ents:
        coverage = len(claim_ents & evidence_ents) / len(claim_ents)
    elif not claim_ents: 
        # If claim has no entities, coverage is arguably 1.0 (trivial) or None.
        # Let's say None or 0.0? Usually None if not applicable.
        pass

    return NLPFeatureVector(
        claim_entity_coverage=coverage,
        claim_entities=sorted(claim_ents),
        evidence_entities=sorted(evidence_ents),
    )


# ---------------------------------------------------------------------------
# OpenAlex author h-index helpers
# ---------------------------------------------------------------------------

def _get_author_ids_from_openalex(pmid: str) -> list[str]:
    """Return OpenAlex author IDs for a given PMID.

    Queries the OpenAlex works endpoint to get authorships, then extracts
    each author's OpenAlex ID (e.g. 'https://openalex.org/A5048677171').
    """
    url = f"https://api.openalex.org/works/pmid:{pmid}"
    try:
        resp = get_session().get(url)
        if resp.status_code == 200:
            data = resp.json()
            ids = []
            for authorship in data.get("authorships", []):
                author_obj = authorship.get("author", {})
                aid = author_obj.get("id")
                if aid:
                    ids.append(aid)
            return ids
    except Exception as exc:
        logger.warning("Failed to get author IDs for PMID %s: %s", pmid, exc)
    return []


def _get_author_h_index(author_id: str) -> int | None:
    """Fetch h-index for a single OpenAlex author ID.

    Args:
        author_id: Full OpenAlex URL, e.g. 'https://openalex.org/A5048677171'.
    """
    # Convert URL to API endpoint
    api_url = author_id.replace("https://openalex.org/", "https://api.openalex.org/authors/")
    try:
        resp = get_session().get(api_url)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("summary_stats", {}).get("h_index")
    except Exception as exc:
        logger.warning("Failed to get h-index for %s: %s", author_id, exc)
    return None


def get_max_author_h_index(pmid: str, delay: float = 0.15) -> int | None:
    """Get the maximum h-index among all authors for a paper.

    Args:
        pmid: PubMed ID.
        delay: Seconds to wait between OpenAlex author API calls.

    Returns:
        Maximum h-index across all authors, or None if unavailable.
    """
    author_ids = _get_author_ids_from_openalex(pmid)
    if not author_ids:
        return None

    h_indices: list[int] = []
    for aid in author_ids:
        h = _get_author_h_index(aid)
        if h is not None:
            h_indices.append(h)
        time.sleep(delay)

    return max(h_indices) if h_indices else None


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class PaperFeatureExtractor:
    """Extract metadata features from a paper given its PMID.

    Reuses existing API functions from search/paper_utils.py:
      - get_pubmed_metadata()        → publication year
      - get_journal_info_by_pmid()   → impact factor (OpenAlex 2yr mean citedness)
      - get_openalex_citation_count() → citation count

    Additionally queries OpenAlex for author-level h-index.

    Computes derived features:
      - log_impact_factor:           log(1 + IF)
      - normalized_citation_count:   citations / (current_year - pub_year + 1)
    """

    def __init__(self, rate_limit_delay: float = 0.35, cache_path: str = "data/metadata_cache.json"):
        """
        Args:
            rate_limit_delay: Seconds to wait between major API call groups
                to avoid NCBI/OpenAlex rate limits (default: 0.35s).
            cache_path: Path to the JSON cache file to store extracted metadata.
        """
        self._delay = rate_limit_delay
        
        # Setup Cache
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self._cache_file = project_root / cache_path
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache from {self._cache_file}: {e}")
        return {}

    def _save_cache(self):
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Write safely using a temporary file to avoid corruption on restart
            temp_file = self._cache_file.with_suffix('.tmp')
            with open(temp_file, "w") as f:
                json.dump(self._cache, f, indent=2)
            os.replace(temp_file, self._cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache to {self._cache_file}: {e}")

    def extract_metadata(self, pmid: str) -> PaperFeatureVector:
        """Extract metadata features for a single paper.

        Args:
            pmid: PubMed ID.

        Returns:
            PaperFeatureVector with populated metadata fields.
            Fields may be None if the corresponding API call failed.
        """
        # --- 0. Check Cache ---
        if pmid in self._cache:
            cached_data = self._cache[pmid]
            logger.debug("Loaded metadata for PMID %s from cache", pmid)
            return PaperFeatureVector(**cached_data)

        logger.info("Extracting metadata features for PMID %s", pmid)

        # --- 1. Publication year from PubMed ---
        pub_year = None
        try:
            meta = get_pubmed_metadata(pmid)
            date_str = meta.get("published_date")
            if date_str:
                year_str = date_str.split("-")[0]
                pub_year = int(year_str)
        except Exception as exc:
            logger.warning("Failed to get pub date for %s: %s", pmid, exc)

        time.sleep(self._delay)

        # --- 2. Impact factor from PubMed + OpenAlex ---
        raw_if = None
        try:
            journal = get_journal_info_by_pmid(pmid)
            raw_if = journal.get("impact_factor")
        except Exception as exc:
            logger.warning("Failed to get journal info for %s: %s", pmid, exc)

        time.sleep(self._delay)

        # --- 3. Citation count from OpenAlex ---
        raw_citations = None
        try:
            raw_citations = get_openalex_citation_count(pmid)
        except Exception as exc:
            logger.warning("Failed to get citations for %s: %s", pmid, exc)

        time.sleep(self._delay)

        # --- 4. Max author h-index from OpenAlex ---
        author_h_max = get_max_author_h_index(pmid, delay=0.15)

        # --- 5. Compute derived features ---
        log_if = None
        if raw_if is not None:
            log_if = math.log(1.0 + raw_if)

        norm_citations = None
        current_year = datetime.now(timezone.utc).year
        if raw_citations is not None and pub_year is not None:
            age = max(current_year - pub_year + 1, 1)
            norm_citations = raw_citations / age

        result_vector = PaperFeatureVector(
            pmid=pmid,
            publication_year=pub_year,
            log_impact_factor=log_if,
            normalized_citation_count=norm_citations,
            author_h_index_max=author_h_max,
        )

        # Save to cache
        self._cache[pmid] = result_vector.model_dump()
        self._save_cache()

        return result_vector

    def extract_batch(
        self, pmids: list[str]
    ) -> list[PaperFeatureVector]:
        """Extract metadata features for multiple papers.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of PaperFeatureVector, one per PMID.
        """
        results = []
        for i, pmid in enumerate(pmids):
            logger.info("Batch progress: %d/%d", i + 1, len(pmids))
            features = self.extract_metadata(pmid)
            results.append(features)
        return results
