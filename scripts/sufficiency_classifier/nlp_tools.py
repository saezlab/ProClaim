"""
NLP Tools for Verification Agent.

Contains classes for:
1. Biomedical Entity Extraction (via MCP connection to scispacy server)
2. Semantic Similarity Computation (via sentence-transformers)
"""

import json
import logging
import os
import sys
from pathlib import Path
from contextlib import AsyncExitStack

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Locate project root to find server script
# This file is in scripts/sufficiency_classifier/nlp_tools.py
# Root is ../../
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


class BiomedicalEntityExtractor:
    """
    MCP Client for the BioNER Scispacy Server.
    Manages the connection to the local MCP server running in a separate environment.
    """
    def __init__(self):
        self.server_script = str(PROJECT_ROOT / "src/servers/scispacy_server.py")
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        env = os.environ.copy()
        python_exe = str(PROJECT_ROOT / ".venv310/bin/python")
        
        if os.path.exists(python_exe):
            params = StdioServerParameters(command=python_exe, args=[self.server_script], env=env)
            logger.info(f"Starting BiomedicalEntityExtractor with command: {python_exe} {self.server_script}")
        else:
            logger.warning(f"Warning: {python_exe} not found. Fallback to 'uv run'.")
            params = StdioServerParameters(command="uv", args=["run", self.server_script], env=env)
            logger.info(f"Starting BiomedicalEntityExtractor with command: uv run {self.server_script}")

        self.read, self.write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def extract(self, text: str) -> list[str]:
        """Call the MCP server to extract entities."""
        if not text:
            return []
        try:
            result = await self.session.call_tool("extract_entities", arguments={"text": text})
            # FastMCP returns result as JSON string in text content within the tool result
            if result.content and hasattr(result.content[0], "text"):
                content = result.content[0].text
                return json.loads(content)
            return []
        except Exception as e:
            logger.error(f"MCP Extraction error: {e}")
            return []


def compute_recall_from_entities(claim_ents: list[str], text_ents: list[str]) -> float:
    """Compute entity recall (coverage) based on extracted entity lists."""
    if not claim_ents:
        return 0.0
    
    # Normalize
    claim_set = set(e.lower() for e in claim_ents)
    text_set = set(e.lower() for e in text_ents)
    
    # intersection
    overlap = claim_set.intersection(text_set)
    return len(overlap) / len(claim_set)


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
        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True, batch_size=32)

        # cos_sim returns a (1, N) tensor
        similarities = cos_sim(claim_emb, chunk_embs)
        return float(similarities.max().item())
