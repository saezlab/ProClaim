"""
Feature extraction tools for sufficiency classifier.

Extracts NLP and metadata features for papers to support the MLP sufficiency classifier.
Integrates entity coverage, semantic similarity, NLI entailment, and metadata extraction.

Usage:
    from pkevolve.verification.feature_tools import (
        compute_entity_coverage,
        SemanticSimilarityComputer,
        NLIEntailmentComputer,
        PaperFeatureExtractor,
    )

    # Entity coverage
    nlp_features = compute_entity_coverage("MAPK1 activates H3-3A", evidence_text)

    # Semantic similarity
    sim_computer = SemanticSimilarityComputer()
    similarity = sim_computer.compute(claim, evidence_text)

    # NLI entailment
    nli_computer = NLIEntailmentComputer()
    nli_scores = nli_computer.compute(claim, evidence_text)

    # Metadata features
    meta_extractor = PaperFeatureExtractor()
    metadata = meta_extractor.extract_metadata("36194155")
"""

import atexit
import logging
import math
import subprocess
import threading
import time
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from pkevolve.search.paper_utils import (
    get_journal_info_by_pmid,
    get_openalex_citation_count,
    get_pubmed_metadata,
    get_session,
)
from pkevolve.verification.data_models import NLPFeatureVector, PaperFeatureVector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity Coverage — scispaCy NER via .venv310 subprocess
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VENV310_PYTHON = _PROJECT_ROOT / ".venv310" / "bin" / "python"
_NER_WORKER = _PROJECT_ROOT / "scripts" / "ner_worker.py"

_ner_proc: subprocess.Popen | None = None
_ner_lock = threading.Lock()


def _start_ner_process() -> "subprocess.Popen | None":
    if not _VENV310_PYTHON.exists() or not _NER_WORKER.exists():
        logger.warning(
            "NER worker unavailable (missing %s). Entity coverage will be empty.",
            _VENV310_PYTHON,
        )
        return None

    logger.info("Starting NER worker subprocess (%s)...", _VENV310_PYTHON)
    proc = subprocess.Popen(
        [str(_VENV310_PYTHON), str(_NER_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        ready = json.loads(proc.stdout.readline())
        if ready.get("ready"):
            logger.info("NER worker ready (model: %s)", ready.get("model"))
            atexit.register(proc.terminate)
            return proc
        logger.warning("NER worker startup failed: %s", ready.get("error"))
    except Exception as exc:
        logger.warning("NER worker bad startup message: %s", exc)
    proc.terminate()
    return None


def _get_ner_process() -> "subprocess.Popen | None":
    global _ner_proc
    with _ner_lock:
        if _ner_proc is None or _ner_proc.poll() is not None:
            _ner_proc = _start_ner_process()
        return _ner_proc


def extract_entities(text: str) -> set[str]:
    """Extract named entities via the .venv310 scispaCy subprocess.

    Uses en_core_sci_sm (biomedical NER). Returns lowercased entity strings.
    Returns empty set if the subprocess is unavailable or the text is empty.
    """
    if not text or not text.strip():
        return set()

    proc = _get_ner_process()
    if proc is None:
        return set()

    with _ner_lock:
        try:
            proc.stdin.write(json.dumps({"text": text}) + "\n")
            proc.stdin.flush()
            result = json.loads(proc.stdout.readline())
            return set(result.get("entities", []))
        except Exception as exc:
            logger.warning("NER subprocess error: %s", exc)
            return set()


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
# Semantic Similarity — SBERT
# ---------------------------------------------------------------------------

class SemanticSimilarityComputer:
    """Compute SBERT cosine similarity between claim and evidence.

    For long evidence texts that exceed the model's token limit (~512 tokens),
    the evidence is split into overlapping chunks. Each chunk is compared
    against the claim and the maximum similarity is returned (max-pooling).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 256, chunk_overlap: int = 64):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SBERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def compute(self, claim: str, evidence: str) -> float:
        """Compute cosine similarity. Returns max similarity over evidence chunks."""
        from sentence_transformers.util import cos_sim

        if not claim or not evidence:
            return 0.0

        chunks = self._chunk_text(evidence)
        claim_emb = self.model.encode(claim, convert_to_tensor=True, show_progress_bar=False)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True, batch_size=32, show_progress_bar=False)

        # cos_sim returns a (1, N) tensor
        similarities = cos_sim(claim_emb, chunk_embs)
        return float(similarities.max().item())


# ---------------------------------------------------------------------------
# NLI Entailment — CrossEncoder
# ---------------------------------------------------------------------------

class NLIEntailmentComputer:
    """Compute NLI probabilities using a cross-encoder model.

    For long evidence texts that exceed the model's token limit,
    the evidence is split into overlapping chunks and the chunk with
    the highest opinionated score (entailment or contradiction) is selected.
    Returns the NLI probabilities for that best chunk.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", chunk_size: int = 256, chunk_overlap: int = 64):
        from sentence_transformers import CrossEncoder
        import torch
        logger.info(f"Loading NLI model: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def compute(self, claim: str, evidence: str) -> dict[str, float | str | None]:
        """Compute NLI scores. Returns probabilities for the best chunk.

        Returns:
            Dictionary with keys:
            - nli_entailment: float
            - nli_contradiction: float
            - nli_neutral: float
            - nli_best_chunk_text: str (the chunk with highest opinionated score)
        """
        import torch
        import numpy as np

        if not claim or not evidence:
            return {
                "nli_entailment": 0.0,
                "nli_contradiction": 0.0,
                "nli_neutral": 1.0,
                "nli_best_chunk_text": None,
            }

        chunks = self._chunk_text(evidence)
        pairs = [[claim, chunk] for chunk in chunks]

        logits = self.model.predict(pairs, show_progress_bar=False)

        if isinstance(logits, list):
            logits = np.array(logits)

        scores_tensor = torch.tensor(logits)
        if len(scores_tensor.shape) == 1:
            scores_tensor = scores_tensor.unsqueeze(0)

        probs = torch.nn.functional.softmax(scores_tensor, dim=-1)  # shape (num_chunks, 3)

        # Map label indices
        id2label = getattr(self.model.config, 'id2label', {})

        # Default mapping for cross-encoder/nli-deberta-v3-*
        ent_idx, con_idx, neu_idx = 1, 0, 2
        for idx, label in id2label.items():
            if not isinstance(label, str):
                continue
            label = label.lower()
            if "entail" in label:
                ent_idx = int(idx)
            elif "contradict" in label:
                con_idx = int(idx)
            elif "neutral" in label:
                neu_idx = int(idx)

        # Find best chunk: highest opinionated score (entailment or contradiction)
        opinion_scores = torch.max(probs[:, ent_idx], probs[:, con_idx])
        best_chunk_idx = torch.argmax(opinion_scores).item()

        # Extract probabilities for the best chunk
        best_probs = probs[best_chunk_idx].tolist()
        best_chunk_text = chunks[best_chunk_idx] if best_chunk_idx < len(chunks) else None

        return {
            "nli_entailment": float(best_probs[ent_idx]),
            "nli_contradiction": float(best_probs[con_idx]),
            "nli_neutral": float(best_probs[neu_idx]),
            "nli_best_chunk_text": best_chunk_text,
        }


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
# Metadata Feature Extractor
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
        project_root = Path(__file__).resolve().parent.parent.parent
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
            features = self.extract_metadata(pmid)
            results.append(features)
        return results
