

"""
Script to extract features for SciFact-Open claims/evidence by combining:
1. PubMed Search & Retrieval (via Claude SDK + mcp-simple-pubmed)
2. Metadata Extraction (via PaperFeatureExtractor)
3. NLP Feature Extraction (Recall/Overlap via scispacy)

Usage:
    uv run scripts/claude_sdk/extract_features_scifact.py --limit 3
"""

import asyncio
import re
import json
import argparse
import random
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from claude_agent_sdk import ClaudeAgentOptions, query, AssistantMessage, UserMessage, ToolUseBlock, ToolResultBlock

# Local imports
from pkevolve.verification.feature_extractor import PaperFeatureExtractor
from pkevolve.verification.data_models import PaperFeatureVector, NLPFeatureVector

class BiomedicalEntityExtractor:
    """
    MCP Client for the BioNER Scispacy Server.
    Manages the connection to the local MCP server running in Python 3.10.
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
        else:
            print(f"Warning: {python_exe} not found. Fallback to 'uv run'.")
            params = StdioServerParameters(command="uv", args=["run", self.server_script], env=env)

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
            # FastMCP returns result as JSON string in text content
            if result.content and hasattr(result.content[0], "text"):
                content = result.content[0].text
                print(content)
                return json.loads(content)
            return []
        except Exception as e:
            print(f"    ⚠️ MCP Extraction error: {e}")
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



# -----------------------------------------------------------------------------
# 1. PubMed Search & Retrieval (Async)
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean text by removing XML/HTML tags and normalizing whitespace."""
    if not text:
        return ""
    # Remove XML/HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def run_search_and_retrieval(options, doc_id: int, title: str, known_pmid: str, run_id: str) -> Dict[str, Any]:
    """
    Ask Claude to find a paper on PubMed and retrieve its full text.
    Saves results to per-doc output directory.
    Returns the path to the output directory and success status.
    """
    output_dir = PROJECT_ROOT / "results" / "claude_sdk" / "scifact_feature_extraction" / f"doc_{doc_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have results (skip if fully processed)
    if (output_dir / "result.json").exists():
        print(f"  [Skipping] Results already exist for Doc {doc_id}")
        return {"output_dir": output_dir, "success": True, "cached": True}

    if known_pmid and known_pmid != "None" and known_pmid.isdigit():
        prompt = (
            f"I have a scientific paper with PMID: {known_pmid}. "
            f"Title: '{title}'. "
            f"Please retrieve its full text using get_paper_fulltext with this PMID."
        )
        print(f"  Retrieving: '{title}' (PMID: {known_pmid})...")
    else:
        prompt = (
            f"I have a scientific paper title: '{title}'. "
            f"Please do the following:\n"
            f"1. Call search_pubmed to find this paper by title\n"
            f"2. Call get_paper_fulltext with the PMID from search results to get the full text\n"
        )
        print(f"  Searching: '{title}'...")

    responses = []
    tool_calls = []
    tool_results = []
    
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, 'text'):
                        responses.append(block.text)
                    
                    if isinstance(block, ToolUseBlock):
                        tool_calls.append({
                            "name": block.name,
                            "id": block.id,
                            "input": block.input
                        })
                        print(f"    -> Tool: {block.name}")
                        
            elif isinstance(message, UserMessage):
                if hasattr(message, 'content') and message.content:
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            content_str = str(block.content) if block.content else ""
                            
                            # Check for truncation error and load from file
                            # "Error: result (...) exceeds maximum allowed tokens. Output has been saved to /path/to/file.txt."
                            match = re.search(r"Output has been saved to\s+(.*?\.txt)", content_str)
                            if match:
                                saved_path = Path(match.group(1))
                                if saved_path.exists():
                                    print(f"    ⚠️ Output truncated. Reading from: {saved_path}")
                                    try:
                                        with open(saved_path, "r") as f:
                                            # Read content (could be huge, be careful?)
                                            # For now read all
                                            content_str = f.read()
                                    except Exception as e:
                                        print(f"    ❌ Failed to read saved output: {e}")

                            tool_results.append({
                                "tool_use_id": block.tool_use_id,
                                "content": content_str,
                                "is_error": getattr(block, 'is_error', False)
                            })

        full_response = "\n".join(responses)
        
        doc_result = {
            "doc_id": doc_id,
            "search_title": title,
            "known_pmid": known_pmid,
            "response": full_response,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "success": True
        }

        # Save complete result JSON
        with open(output_dir / "result.json", "w") as f:
            json.dump(doc_result, f, indent=2, ensure_ascii=False)
            
        # Extract and save tool products
        tool_id_to_name = {tc["id"]: tc["name"] for tc in tool_calls}
        
        for tr in tool_results:
            tool_name = tool_id_to_name.get(tr["tool_use_id"], "")
            
            if "get_paper_fulltext" in tool_name and not tr.get("is_error"):
                with open(output_dir / "full_text.txt", "w") as f:
                    f.write(tr["content"])
                    
            elif "search_pubmed" in tool_name and not tr.get("is_error"):
                with open(output_dir / "search_results.txt", "w") as f:
                    f.write(tr["content"])

        return {"output_dir": output_dir, "success": True, "cached": False}

    except Exception as e:
        print(f"  ❌ Error in search: {e}")
        return {"output_dir": output_dir, "success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# 2. Main Processing Loop
# -----------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Integrated Feature Extraction for SciFact-Open")
    parser.add_argument("--limit", type=int, default=3, help="Number of random claims to process")
    parser.add_argument("--claim-id", type=int, help="Process specific claim ID")
    parser.add_argument("--doc-id", type=int, help="Process specific evidence Doc ID (requires valid claim)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5-20250929")
    args = parser.parse_args()

    # --- Setup Configuration & Env ---
    env_vars = {}
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    env_vars[k] = v.strip('"\'')
    
    if "CLAUDE_API_KEY" not in env_vars:
        print("Error: CLAUDE_API_KEY not found in .env")
        return

    if not shutil.which("uvx"):
        print("Error: 'uvx' not found. Install uv first.")
        return

    full_env = os.environ.copy()
    full_env.update(env_vars)
    full_env["ANTHROPIC_API_KEY"] = env_vars.get("CLAUDE_API_KEY", full_env.get("CLAUDE_API_KEY"))

    # Credentials for PubMed (Env > Args)
    pubmed_email = full_env.get("PUBMED_EMAIL")
    pubmed_api_key = full_env.get("PUBMED_API_KEY")
    
    if not pubmed_email:
        print("Error: PUBMED_EMAIL not found in .env")
        return

    options = ClaudeAgentOptions(
        model=args.model,
        cwd=str(PROJECT_ROOT),
        env=full_env,
        allowed_tools=[
            "mcp__pubmed__search_pubmed",
            "mcp__pubmed__get_paper_fulltext",
        ],
        disallowed_tools=["Write", "WebSearch", "WebFetch", "Bash", "Task"],
        mcp_servers={
            "pubmed": {
                "command": "uvx",
                "args": ["--python", "3.12", "mcp-simple-pubmed"],
                "env": {
                    "PUBMED_EMAIL": pubmed_email,
                    **({"PUBMED_API_KEY": pubmed_api_key} if pubmed_api_key else {}),
                },
            }
        },
    )

    # --- Load Data ---
    print("Loading SciFact Data...")
    claims_path = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/claims.jsonl")
    corpus_path = Path("/hps/nobackup/saezrodriguez/shared_datasets/scifact-open/data/corpus.jsonl")
    
    # Load corpus (doc_id -> doc)
    corpus = {}
    if corpus_path.exists():
        with open(corpus_path, "r") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    corpus[int(doc["doc_id"])] = doc
                except: pass
    print(f"Loaded {len(corpus)} documents.")

    # Load claims
    target_claims = []
    with open(claims_path, "r") as f:
        all_claims = [json.loads(line) for line in f]
    
    # Filter for claims that have evidence in our corpus
    viable_claims = []
    for c in all_claims:
        if not c.get("evidence"): continue
        # Check if at least one evidence doc is in our corpus
        has_corpus_doc = any(int(did) in corpus for did in c["evidence"].keys())
        if has_corpus_doc:
            viable_claims.append(c)
            
    if args.claim_id:
        target_claims = [c for c in viable_claims if c["id"] == args.claim_id]
    else:
        target_claims = random.sample(viable_claims, min(args.limit, len(viable_claims)))
    
    print(f"Selected {len(target_claims)} claims for processing.")
    
    # --- Processing Loop ---
    feature_extractor = PaperFeatureExtractor()
    extracted_features_list = []

    # Initialize NLP components
    sim_computer = SemanticSimilarityComputer()
    nli_computer = NLIEentailmentComputer()
    extractor = BiomedicalEntityExtractor()
    await extractor.__aenter__()

    for i, claim in enumerate(target_claims):
        print(f"\n[{i+1}/{len(target_claims)}] Processing Claim ID: {claim['id']}")
        
        for doc_id_str, evidence_info in claim["evidence"].items():
            doc_id = int(doc_id_str)
            if doc_id not in corpus:
                continue
                
            doc = corpus[doc_id]
            title = doc.get("title", "")
            known_pmid = str(doc.get("pmid")) if doc.get("pmid") else None
            
            # 1. Search & Retrieve (Claude)
            search_res = await run_search_and_retrieval(options, doc_id, title, known_pmid, f"run_{i}")
            
            output_dir = search_res.get("output_dir")
            found_pmid = None
            full_text_content = ""
            
            # Analyze Search Results
            if output_dir and (output_dir / "result.json").exists():
                with open(output_dir / "result.json", "r") as f:
                    run_data = json.load(f)
                    
                # Try to extract PMID from successful tool calls
                for tc in run_data.get("tool_calls", []):
                    if "get_paper_fulltext" in tc["name"]:
                        # Input might be {"pmid": "12345"}
                        if isinstance(tc.get("input"), dict):
                            found_pmid = tc["input"].get("pmid")
                
                # If tool call didn't have it, maybe it was passed as known_pmid
                if not found_pmid and known_pmid:
                    found_pmid = known_pmid
                    
                # Load full text if available
                if (output_dir / "full_text.txt").exists():
                    with open(output_dir / "full_text.txt", "r") as f:
                        full_text_content = f.read()

            # 2. Metadata Features
            meta_features = None
            if found_pmid:
                print(f"    Extracting Metadata for PMID: {found_pmid}...")
                try:
                    # Note: We trust extract_metadata to handle errors gracefully (returns None fields)
                    meta_vec = feature_extractor.extract_metadata(found_pmid)
                    if meta_vec:
                        meta_features = meta_vec.model_dump()
                except Exception as e:
                    print(f"    ⚠️ Metadata extraction error: {e}")

            # 3. NLP Features (Recall)
            # Strategy: Use full_text if available, else corpus abstract
            text_source = "none"
            text_for_nlp = ""
            



            if full_text_content and len(full_text_content) > 100:
                text_for_nlp = clean_text(full_text_content)
                text_source = "full_text_pubmed"
            elif doc.get("abstract"):
                 text_for_nlp = " ".join(doc["abstract"])
                 text_source = "abstract_corpus"

            nlp_vec = None
            if text_for_nlp:
                print(f"    Computing NLP Features (Source: {text_source})...")
                try:
                    # Extract entities via MCP
                    claim_entities = await extractor.extract(claim["claim"])
                    evidence_entities = await extractor.extract(text_for_nlp)
                    
                    # Compute Coverage
                    coverage = compute_recall_from_entities(claim_entities, evidence_entities)

                    # Compute Semantic Similarity (SBERT) for each chunk
                    chunk_similarities = sim_computer.compute(claim["claim"], text_for_nlp)
                    similarity = float(chunk_similarities.max().item())
                    print(f"    Semantic Similarity (max): {similarity:.4f}")
                    
                    # Compute NLI Entailment
                    nli_scores = nli_computer.compute(claim["claim"], text_for_nlp)
                    print(f"    NLI Max Scores: Entail={nli_scores['nli_entailment']:.4f}, Contradict={nli_scores['nli_contradiction']:.4f}, Neutral={nli_scores['nli_neutral']:.4f}")
                    
                    # Calculate weighted maximums
                    raw_probs = nli_scores["raw_probs"] # shape: (num_chunks, 3)
                    
                    # Optional: Handle length mismatch conceptually (though chunking logic is identical)
                    min_len = min(len(chunk_similarities), len(raw_probs))
                    chunk_similarities = chunk_similarities[:min_len].unsqueeze(1).cpu() # Move to CPU
                    raw_probs = raw_probs[:min_len] 
                    
                    # Ensure raw_probs is a tensor
                    import torch
                    if not isinstance(raw_probs, torch.Tensor):
                        raw_probs = torch.tensor(raw_probs, device='cpu')
                    
                    weighted_probs = raw_probs * chunk_similarities
                    
                    ent_idx, con_idx, neu_idx = nli_scores["ent_idx"], nli_scores["con_idx"], nli_scores["neu_idx"]
                    
                    # Find the chunk with the highest *opinionated* weighted score (Entailment or Contradiction)
                    # We ignore highest weighted *neutral* score because we want the chunk that makes the strongest claim
                    opinion_weighted_scores = torch.max(weighted_probs[:, ent_idx], weighted_probs[:, con_idx])
                    best_chunk_idx = torch.argmax(opinion_weighted_scores).item()
                    
                    # Extract the *unweighted* probabilities and text of this single best chunk
                    best_probs = raw_probs[best_chunk_idx].tolist()
                    best_entailment = float(best_probs[ent_idx])
                    best_contradiction = float(best_probs[con_idx])
                    best_neutral = float(best_probs[neu_idx])
                    
                    chunks = nli_scores.get("chunks", [])
                    best_chunk_text = chunks[best_chunk_idx] if best_chunk_idx < len(chunks) else None
                    
                    print(f"    NLI Best Chunk Scores: Entail={best_entailment:.4f}, Contradict={best_contradiction:.4f}, Neutral={best_neutral:.4f}")

                    nlp_vec = {
                        "claim_entity_coverage": coverage,
                        "semantic_similarity": similarity,
                        "claim_entities": claim_entities,
                        "evidence_entities": evidence_entities,
                        "nli_entailment": best_entailment,
                        "nli_contradiction": best_contradiction,
                        "nli_neutral": best_neutral,
                        "nli_best_chunk_text": best_chunk_text
                    }
                except Exception as e:
                    print(f"    ⚠️ NLP Extraction Failed: {e}")
            
            # 4. Combine Results
            feature_record = {
                "claim_id": claim["id"],
                "doc_id": doc_id,
                "claim_text": claim["claim"],
                "evidence_label": evidence_info.get("label"),
                "found_pmid": found_pmid,
                "final_text_source": text_source,
                "metadata_features": meta_features,
                "nlp_features": nlp_vec
            }
            
            extracted_features_list.append(feature_record)
            
            # Rate limit/pause
            await asyncio.sleep(1)

    # Close MCP Client
    await extractor.__aexit__(None, None, None)

    # Save Final Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = PROJECT_ROOT / "results" / "claude_sdk" / "scifact_feature_extraction" / f"features_{timestamp}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w") as f:
        json.dump(extracted_features_list, f, indent=2)
        
    print(f"\nSaved {len(extracted_features_list)} feature records to: {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
