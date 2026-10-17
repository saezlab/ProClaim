"""NLP helpers for generate_classifier_data.py.

SemanticSimilarityComputer and NLIEentailmentComputer return per-chunk scores, which
generate_classifier_data.py pools itself. Entities come from the scispaCy NER worker
(scripts/ner_worker.py in the .venv310 environment) via proclaim.verification.feature_tools.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from proclaim.verification.feature_tools import extract_entities  # noqa: E402


class BiomedicalEntityExtractor:
    """Async wrapper around the scispaCy NER worker."""

    async def __aenter__(self) -> "BiomedicalEntityExtractor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def extract(self, text: str) -> list[str]:
        if not text:
            return []
        return sorted(await asyncio.to_thread(extract_entities, text))


def compute_recall_from_entities(claim_ents: list[str], text_ents: list[str]) -> float:
    """Fraction of claim entities that also occur in the text (case-insensitive)."""
    if not claim_ents:
        return 0.0
    claim_set = {e.lower() for e in claim_ents}
    text_set = {e.lower() for e in text_ents}
    return len(claim_set & text_set) / len(claim_set)


def clean_text(text: str) -> str:
    """Remove XML/HTML tags and normalise whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SemanticSimilarityComputer:
    """Compute SBERT cosine similarity between claim and evidence.

    For long evidence texts that exceed the model's token limit (~512 tokens),
    the evidence is split into overlapping chunks. Each chunk is compared
    against the claim and the maximum similarity is returned (max-pooling).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 256, chunk_overlap: int = 64):
        from sentence_transformers import SentenceTransformer
        print(f"  Loading SBERT model: {model_name}...")
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

        chunks = self._chunk_text(evidence)
        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True, batch_size=32)

        # cos_sim returns a (1, N) tensor
        similarities = cos_sim(claim_emb, chunk_embs).squeeze(0) # shape: (num_chunks,)
        return similarities


class NLIEentailmentComputer:
    """Compute NLI probabilities using a cross-encoder model.

    For long evidence texts that exceed the model's token limit,
    the evidence is split into overlapping chunks and max probabilities
    are returned for entailment, contradiction, and neutral.
    Also returns similarities-weighted NLI scores.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", chunk_size: int = 256, chunk_overlap: int = 64):
        from sentence_transformers import CrossEncoder
        import torch
        print(f"  Loading NLI model: {model_name}...")
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

    def compute(self, claim: str, evidence: str) -> dict[str, float]:
        """Compute NLI scores. Returns max probabilities across chunks."""
        import torch

        chunks = self._chunk_text(evidence)
        pairs = [[claim, chunk] for chunk in chunks]
        
        logits = self.model.predict(pairs)
        
        import numpy as np
        if isinstance(logits, list):
            logits = np.array(logits)
        
        scores_tensor = torch.tensor(logits)
        if len(scores_tensor.shape) == 1:
            scores_tensor = scores_tensor.unsqueeze(0)
            
        probs = torch.nn.functional.softmax(scores_tensor, dim=-1) # shape (num_chunks, 3)

        id2label = getattr(self.model.config, 'id2label', {})
        
        # Default mapping for cross-encoder/nli-deberta-v3-*
        ent_idx, con_idx, neu_idx = 1, 0, 2
        for idx, label in id2label.items():
            if not isinstance(label, str): continue
            label = label.lower()
            if "entail" in label:
                ent_idx = int(idx)
            elif "contradict" in label:
                con_idx = int(idx)
            elif "neutral" in label:
                neu_idx = int(idx)
            
        # Original max probs across all chunks
        max_probs = probs.max(dim=0).values.tolist()

        return {
            "nli_contradiction": float(max_probs[con_idx]),
            "nli_entailment": float(max_probs[ent_idx]),
            "nli_neutral": float(max_probs[neu_idx]),
            "raw_probs": probs,
            "ent_idx": ent_idx,
            "con_idx": con_idx,
            "neu_idx": neu_idx,
            "chunks": chunks
        }
