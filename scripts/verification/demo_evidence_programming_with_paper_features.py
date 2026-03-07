#!/usr/bin/env python3
"""
Demo: Verification Agent with Feature Extraction.

Integrates the Evidence Programming loop with:
1.  Async MCP Client for `mcp-simple-pubmed` (Search & Retrieval).
2.  Metadata Feature Extraction (FeatureExtractor).
3.  NLP Feature Extraction (BiomedicalEntityExtractor, SemanticSimilarity).

Usage:
  uv run scripts/verification/demo_evidence_programming_with_paper_features.py \
      --preset local --claim "Does p53 activate BAX?"

  # Anthropic (OpenAI-compatible)
  uv run scripts/verification/demo_evidence_programming_with_paper_features.py \
      --preset anthropic --claim "Does GNAS directly activate ADCY1?"

  # GLM
  uv run scripts/verification/demo_evidence_programming_with_paper_features.py \
      --preset glm --claim "Does p53 activate BAX?"

  # Custom endpoint
  uv run python scripts/verification/demo_evidence_programming_with_paper_features.py \
      --base-url http://my-server:8000/v1 --api-key my-key \
      --model my-model --claim "Does p53 activate BAX?"
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import shutil
import sys
import tempfile
import asyncio
import argparse
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from pkevolve.verification.evidence_state import EvidenceState  # noqa: E402
from pkevolve.verification.data_models import (  # noqa: E402
    PaperFeatureVector, NLPFeatureVector
)
from pkevolve.verification.mcp_tools import (  # noqa: E402
    add_facts,
    check_sufficiency,
    compress_evidence,
    emit_verdict,
    find_related_articles,
    get_evidence_summary,
    search_for_gap,
    # We replace the search/get_text tools with our own MCP client calls
    # but still use some logic from there if needed, or reimplement.
)
from scripts.sufficiency_classifier.feature_extractor import PaperFeatureExtractor  # noqa: E402
from scripts.sufficiency_classifier.nlp_tools import (  # noqa: E402
    BiomedicalEntityExtractor,
    SemanticSimilarityComputer,
    compute_recall_from_entities,
)
# Re-import LLMClient from original script (or copy it here for simplicity)
# Since we copied the file, we can keep the class definition here.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint Presets & LLM Client
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "local_oss_120b": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "model": "gpt-oss-120b",
    },
    "local_glm_5": {
        "base_url": "http://codon-gpu-001.ebi.ac.uk:8000/v1/",
        "api_key": "EMPTY",
        "model": "glm-5-fp8",
    },
    "glm": {
        "base_url": "https://api.z.ai/api/anthropic",
        "api_key_env": "GLM_API_KEY",
        "model": "glm-5",
        "backend": "anthropic",
    },
}

class LLMClient:
    """Thin wrapper supporting OpenAI-compatible and Anthropic endpoints."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 backend: str = "openai"):
        self.model = model
        self.backend = backend

        if backend == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key, base_url=base_url)
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        """Single-turn chat completion. Returns the assistant message text."""
        if self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content


# ---------------------------------------------------------------------------
# PubMed MCP Client
# ---------------------------------------------------------------------------

class PubMedClient:
    """
    Client for the mcp-simple-pubmed server.
    """
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        if not shutil.which("uvx"):
            raise RuntimeError("'uvx' not found. Install uv first.")

        # Credentials
        env = os.environ.copy()
        pubmed_email = env.get("PUBMED_EMAIL")
        pubmed_api_key = env.get("PUBMED_API_KEY")

        if not pubmed_email:
            logger.warning("PUBMED_EMAIL not set in environment. Queries may be rate limited.")
        
        server_env = env.copy()
        if pubmed_email:
            server_env["PUBMED_EMAIL"] = pubmed_email
        if pubmed_api_key:
            server_env["PUBMED_API_KEY"] = pubmed_api_key

        # Command: uvx --python 3.12 mcp-simple-pubmed
        command = "uvx"
        args = ["--python", "3.12", "mcp-simple-pubmed"]
        
        logger.info(f"Starting PubMed MCP server: {command} {' '.join(args)}")
        
        params = StdioServerParameters(command=command, args=args, env=server_env)
        self.read, self.write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def search(self, query: str) -> list[str]:
        """Call search_pubmed tool. Returns list of PMIDs (strings)."""
        logger.info(f"Searching PubMed for: {query}")
        try:
            result = await self.session.call_tool("search_pubmed", arguments={"query": query})
            # expects a formatted string or json? mcp-simple-pubmed returns a text summary usually.
            # Let's inspect the output.
            # If it returns a list of PMIDs, we need to parse it. 
            # Actually mcp-simple-pubmed 'search_pubmed' usually returns a formatted string list of results.
            # But wait, looking at extract_features_scifact.py, it uses search_pubmed then get_paper_fulltext.
            # Let's assume it returns a string we can parse or containing PMIDs?
            # Actually, standard mcp-simple-pubmed returns a string description.
            # We might need to ask the user to refine what tool to use or how to parse.
            # Let's assume for now we parse the text output for PMIDs if it's not JSON.
            # Typically mcp-simple-pubmed returns: "1. Title (PMID: 12345)\n..."
            
            content = result.content[0].text if result.content else ""
            if not content:
                return []
            
            # fastmcp mcp-simple-pubmed returns JSON lists of papers.
            import json
            import re
            try:
                data = json.loads(content)
                pmids = [str(item["pmid"]) for item in data if "pmid" in item]
                if pmids:
                    return list(set(pmids))
            except json.JSONDecodeError:
                pass
            
            # Fallback regex to find PMIDs (case insensitive just in case)
            pmids = re.findall(r"(?:PMID|pmid)[\s:]*(\d+)", content)
            return list(set(pmids))
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    async def get_paper_fulltext(self, pmid: str) -> str:
        """Call get_paper_fulltext tool."""
        try:
            result = await self.session.call_tool("get_paper_fulltext", arguments={"pmid": pmid})
            if result.content and hasattr(result.content[0], "text"):
                return result.content[0].text
            return ""
        except Exception as e:
            logger.error(f"get_paper_fulltext error for {pmid}: {e}")
            return ""


# ---------------------------------------------------------------------------
# Helpers (LLM)
# ---------------------------------------------------------------------------
# Wrappers to make sync LLM calls 'async-compatible' (blocking is fine for now in this loop)

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[1:end])
    return text.strip()


def llm_extract_facts(
    client: LLMClient, claim: str, paper_text: str,
    pmid: str, subclaims: list[str],
) -> str:
    """Ask LLM to extract stance-labeled facts. Returns JSON string."""
    system = (
        "You are a fact extractor for scientific claim verification.\n"
        "Extract atomic facts from the paper text relevant to the claim.\n\n"
        "STANCE GUIDELINES:\n"
        "- SUPPORT: The fact provides direct or mechanistic evidence FOR the claim.\n"
        "  Example: If the claim is 'A activates B', a fact showing A phosphorylates B "
        "is SUPPORT.\n"
        "- REFUTE: The fact provides evidence AGAINST the claim. This includes:\n"
        "  * Evidence that a DIFFERENT protein (not the one in the claim) performs the "
        "action on the target.\n"
        "  * Evidence that the claimed interaction does not occur.\n"
        "  * Evidence that the mechanism described in the claim is wrong.\n"
        "- NEUTRAL: The fact is tangentially related but does not directly address "
        "the claim.\n\n"
        "IMPORTANT: When evidence describes the target protein's function/regulation "
        "by a DIFFERENT kinase/activator than what the claim states, classify as REFUTE "
        "with moderate confidence, since it suggests the claimed interaction may not be "
        "the primary mechanism.\n\n"
        "Return a JSON array of objects with keys:\n"
        '  text (string), stance (SUPPORT|REFUTE|NEUTRAL), source_pmid (string),\n'
        '  relevant_subclaims (list of strings), confidence (float 0-1).\n'
        "Return ONLY the JSON array."
    )
    user = (
        f"Claim: {claim}\n"
        f"Subclaims: {json.dumps(subclaims)}\n"
        f"PMID: {pmid}\n\n"
        f"Paper text:\n{paper_text[:4000]}"
    )
    return _strip_code_fences(client.chat(system, user))


def llm_formulate_gap_query(
    client: LLMClient, gap_description: str, gap_type: str, claim: str,
    failed_queries: list[str] | None = None,
) -> str:
    """Ask LLM to turn a gap into a PubMed search query string."""
    system = (
        "You are a PubMed query formulator. Given an evidence gap, "
        "formulate a targeted PubMed search query to fill that gap.\n"
        "Use gene/protein symbols with OR for common synonyms/aliases.\n"
        "Do NOT use field tags like [Title/Abstract].\n"
        "Keep the query broad enough to return results -- avoid ANDing "
        "more than 2-3 concepts.\n"
        "IMPORTANT: If both genes together return 0 results, try querying "
        "ONE gene at a time combined with a functional term (e.g., pathway, "
        "signaling, phosphorylation) rather than ANDing both gene symbols.\n"
        "Return ONLY the query string, no explanation."
    )
    user_parts = [
        f"Claim: {claim}",
        f"Gap type: {gap_type}",
        f"Gap: {gap_description}",
    ]
    if failed_queries:
        user_parts.append(
            f"\nPrevious queries that returned ZERO results "
            f"(do NOT repeat these):\n"
            + "\n".join(f"  - {q}" for q in failed_queries)
        )
    user_parts.append(
        "\nFormulate a PubMed query to find papers addressing this gap."
    )
    return client.chat(system, "\n".join(user_parts)).strip().strip('"').strip("'")


def llm_verdict(client: LLMClient, claim: str, summary: str) -> dict:
    """Ask LLM for final verdict. Returns dict with verdict fields."""
    system = (
        "You are a scientific claim verifier. Based on the evidence summary,\n"
        "provide a final verdict.\n\n"
        "Return a JSON object with keys:\n"
        "  verdict (SUPPORT|REFUTE|INSUFFICIENT), confidence (float 0-1),\n"
        "  reasoning (string), key_evidence (list of strings),\n"
        "  gaps_remaining (list of strings).\n"
        "Return ONLY the JSON object."
    )
    user = f"Claim: {claim}\n\nEvidence summary:\n{summary}"
    raw = _strip_code_fences(client.chat(system, user, temperature=0.1))
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main evidence programming loop
# ---------------------------------------------------------------------------


async def run_demo(
    claim: str,
    workspace: Path,
    client: LLMClient,
    max_iterations: int = 5,
    threshold: float = 0.80,
) -> None:
    """Execute the evidence programming loop."""
    ws = str(workspace)

    # --- Init ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"CLAIM: {claim}")
    print(f"{'=' * 60}")

    EvidenceState.init_new(claim, subclaims=[claim], workspace=workspace)
    print(f"[INIT] Workspace: {workspace}\n")

    # --- Initialize Tools ---
    print("[INIT] initializing tools...")
    
    # 1. Metadata Feature Extractor
    feature_extractor = PaperFeatureExtractor()
    
    # 2. Semantic Similarity Computer
    sim_computer = SemanticSimilarityComputer()
    
    # 3. Entity Extractor & PubMed Client (Async Contexts)
    entity_extractor = BiomedicalEntityExtractor()
    pubmed_client = PubMedClient()

    async with entity_extractor, pubmed_client:
        
        # --- Initial Search ---
        print("[SEARCH] Initial PubMed search (via mcp-simple-pubmed) ...")
        # We don't have search_pubmed_progressive here easily unless we reimplement the tier logic.
        # For this demo, let's start with a simple query derived from the claim.
        # Ideally, we call LLM to formulate query if simple claim fails, or just use claim.
        # Let's try direct claim search first.
        initial_query = claim
        
        pmids = await pubmed_client.search(initial_query)
        print(f"  Found {len(pmids)} PMIDs for query: '{initial_query}'")
        
        if not pmids:
             # Fallback to LLM query formulation
            print("\n[SEARCH] 0 papers, using LLM to formulate query …")
            llm_query = llm_formulate_gap_query(client, f"No papers found for: {claim}", "missing_evidence", claim)
            print(f"  LLM query: {llm_query}")
            pmids = await pubmed_client.search(llm_query)
            print(f"  Found {len(pmids)} PMIDs")

        
        processed_pmids: set[str] = set()
        failed_queries: list[str] = []
        used_seed_pmids: set[str] = set()
        
        # Add found PMIDs to state (as placeholders, we'll fetch details next)
        from pkevolve.verification.data_models import PaperRecord
        state = EvidenceState.load(workspace / "evidence_state.json")
        for pmid in pmids:
            if pmid not in state.papers:
                # We need title/abstract. mcp-simple-pubmed search output might have had it but we parsed PMIDs.
                # We will fetch text mostly in the extract phase.
                # Create a placeholder record
                state.add_paper(PaperRecord(pmid=pmid, title="Fetching...", abstract="", source="pubmed"))
        state.save(workspace / "evidence_state.json")


        for iteration in range(1, max_iterations + 1):
            print(f"\n{'─' * 60}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'─' * 60}")

        # -- Extract facts from new papers -----------------------------------
            state = EvidenceState.load(workspace / "evidence_state.json")
            new_pmids = [p for p in state.papers if p not in processed_pmids]

            if new_pmids:
                print(f"\n[EXTRACT] {len(new_pmids)} new paper(s) to process")
                
                # --- Feature Extraction Batch ---
                # Pre-calculate claim entities once
                claim_entities = await entity_extractor.extract(claim)
                print(f"  [NLP] Claim Entities: {claim_entities}")

                for pmid in new_pmids:
                    paper = state.papers[pmid]
                    print(f"  PMID {pmid}: Fetching text & Extracting features...")
                    
                    # 1. Get Text (Full Text via MCP)
                    text_content = await pubmed_client.get_paper_fulltext(pmid)
                    text_source = "abstract"
                    if text_content and "Abstract" not in text_content and len(text_content) > 1000:
                         # Heuristic: if it's long, it's likely full text
                         paper.full_text = text_content
                         text_source = "full_text"
                    elif text_content:
                         paper.abstract = text_content # Fallback if only abstract returned or found
                    
                    # Ensure we have some text
                    text_to_analyze = paper.full_text or paper.abstract or ""
                    if not text_to_analyze:
                        print(f"    Skipping {pmid} (no text found)")
                        processed_pmids.add(pmid)
                        continue

                    # 2. Metadata Features
                    try:
                        meta_vec = feature_extractor.extract_metadata(pmid)
                        paper.metadata = meta_vec
                        print(f"    [META] Year: {meta_vec.publication_year}, IF log: {meta_vec.log_impact_factor:.2f} (raw IF likely ~{2.71**meta_vec.log_impact_factor - 1:.1f})")
                    except Exception as e:
                        print(f"    [META] Error: {e}")

                    # 3. NLP Features
                    try:
                        ev_entities = await entity_extractor.extract(text_to_analyze[:10000]) # limit length for NER
                        overlap = compute_recall_from_entities(claim_entities, ev_entities)
                        similarity = sim_computer.compute(claim, text_to_analyze)
                        
                        paper.nlp = NLPFeatureVector(
                            entity_overlap_ratio=None, # Not calculating Jaccard for now
                            claim_entity_coverage=overlap,
                            semantic_similarity=similarity,
                            claim_entities=claim_entities,
                            evidence_entities=ev_entities
                        )
                        print(f"    [NLP] Coverage: {overlap:.2f}, Similarity: {similarity:.2f}")
                    except Exception as e:
                        print(f"    [NLP] Error: {e}")

                    # Save updated paper record
                    state.papers[pmid] = paper
                    state.save(workspace / "evidence_state.json")

                    # 4. LLM Fact Extraction
                    try:
                        print(f"    [Reasoning] Extracting facts...")
                        facts_json = llm_extract_facts(client, claim, text_to_analyze, pmid, state.subclaims)
                        msg = add_facts(facts_json, ws)
                        print(f"      {msg}")
                    except Exception as exc:
                        print(f"      Error extracting facts: {exc}")
                    
                    processed_pmids.add(pmid)

            else:
                print("\n[EXTRACT] No new papers.")

            # -- Sufficiency check -----------------------------------------------
            print("\n[SUFFICIENCY] Running classifier (Heuristic) ...")
            suff_text = check_sufficiency(ws)
            for line in suff_text.splitlines():
                print(f"  {line}")

            # Check logic
            iteration_limit = "ITERATION LIMIT" in suff_text
            if iteration_limit:
                print("\n  → Iteration limit reached.")
                break

            state = EvidenceState.load(workspace / "evidence_state.json")
            latest = state.sufficiency_history[-1] if state.sufficiency_history else None
            
            if latest and latest.label != "INSUFFICIENT" and latest.confidence >= threshold:
                print(f"\n  → Evidence SUFFICIENT ({latest.label}, conf={latest.confidence:.2f})")
                break

            # -- Gap-targeted retrieval ------------------------------------------
            if latest and latest.gaps:
                for gap in latest.gaps[:2]:
                    print(f"\n[GAP] {gap.gap_type.value}: {gap.description}")
                    try:
                        query = llm_formulate_gap_query(
                            client, gap.description, gap.gap_type.value, claim,
                            failed_queries=failed_queries,
                        )
                        print(f"  Query: {query}")
                        
                        # Use MCP Search
                        new_gap_pmids = await pubmed_client.search(query)
                        print(f"  Found {len(new_gap_pmids)} papers.")
                        
                        added_count = 0
                        for gp in new_gap_pmids:
                            if gp not in state.papers:
                                state.add_paper(PaperRecord(pmid=gp, title="Fetching...", abstract="", source="pubmed"))
                                added_count += 1
                        state.save(workspace / "evidence_state.json")
                        
                        if added_count == 0:
                            failed_queries.append(query)
                            
                    except Exception as exc:
                        print(f"  Error: {exc}")

            # -- Compress if needed -- 
            state = EvidenceState.load(workspace / "evidence_state.json")
            if state.token_estimate > 40_000:
                print(f"\n[COMPRESS] {state.token_estimate} tokens > 40 000 budget")
                print(f"  {compress_evidence(ws)}")


    # --- Final verdict ------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("FINAL VERDICT")
    print(f"{'─' * 60}")

    summary = get_evidence_summary(ws)
    print(f"\n[SUMMARY]\n{summary}\n")

    print("[VERDICT] Generating verdict via LLM …")
    try:
        vd = llm_verdict(client, claim, summary)
        emit_verdict(
            verdict=vd["verdict"],
            confidence=vd["confidence"],
            reasoning=vd["reasoning"],
            key_evidence_json=json.dumps(vd.get("key_evidence", [])),
            gaps_remaining_json=json.dumps(vd.get("gaps_remaining", [])),
            workspace=ws,
        )
        print(f"\n{'=' * 60}")
        print(f"  VERDICT:    {vd['verdict']}")
        print(f"  CONFIDENCE: {vd['confidence']:.2f}")
        print(f"  REASONING:  {vd['reasoning']}")
        print(f"{'=' * 60}")
    except Exception as exc:
        print(f"  Error generating verdict: {exc}")

    print(f"\nWorkspace: {workspace}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verification Agent with Feature Extraction.",
    )
    parser.add_argument(
        "--claim", required=True, help="Scientific claim to verify.",
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS), default=None,
        help="Endpoint preset: local | glm | anthropic.",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", help="API key (overrides preset/env).")
    parser.add_argument("--model", help="Model identifier (overrides preset).")
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Max evidence iterations (default: 5).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Sufficiency confidence threshold (default: 0.80).",
    )
    parser.add_argument("--output-dir", help="Output workspace directory.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Resolve endpoint configuration --------------------------------------
    if args.preset:
        cfg = PRESETS[args.preset].copy()
        env_var = cfg.pop("api_key_env", None)
        if "api_key" not in cfg and env_var:
            cfg["api_key"] = os.environ.get(env_var, "")
            if not cfg["api_key"]:
                print(f"ERROR: Set {env_var} for preset '{args.preset}'")
                sys.exit(1)
        if args.base_url:
            cfg["base_url"] = args.base_url
        if args.model:
            cfg["model"] = args.model
        if args.api_key:
            cfg["api_key"] = args.api_key
    elif args.base_url:
        cfg = {
            "base_url": args.base_url,
            "api_key": args.api_key or "EMPTY",
            "model": args.model or "default",
        }
    else:
        print("ERROR: Provide --preset or --base-url.")
        sys.exit(1)

    client = LLMClient(
        base_url=cfg["base_url"], api_key=cfg["api_key"], model=cfg["model"],
        backend=cfg.get("backend", "openai"),
    )
    print(f"Endpoint: {cfg['base_url']}  Model: {cfg['model']}  Backend: {cfg.get('backend', 'openai')}")

    # -- Workspace -----------------------------------------------------------
    workspace = Path(args.output_dir) if args.output_dir else Path(
        tempfile.mkdtemp(prefix="demo_ep_")
    )
    
    asyncio.run(run_demo(
        claim=args.claim,
        workspace=workspace,
        client=client,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
    ))


if __name__ == "__main__":
    main()
