
"""
Test script to run the NLP feature extraction pipeline on a 'Mismatch' scenario using REAL SciFact data and FULL TEXT retrieval.

Purpose:
1. Select a random SciFact claim.
2. Select a random evidence document from the corpus that is NOT associated with this claim.
3. Perform full text retrieval for this unrelated document (simulating a "retrieved but irrelevant" result).
4. Compute attributes (Entity Coverage, Semantic Similarity) and expect low scores.
"""

import asyncio
import sys
import json
import random
import os
import shutil
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from claude_agent_sdk import ClaudeAgentOptions
from scripts.claude_sdk.extract_features_scifact import (
    BiomedicalEntityExtractor, 
    SemanticSimilarityComputer,
    compute_recall_from_entities,
    run_search_and_retrieval,
    clean_text,
    PROJECT_ROOT as SCRIPT_PROJECT_ROOT
)

async def main():
    # --- Setup Logic from extract_features_scifact.py ---
    print("Setting up environment and agents...")
    env_vars = {}
    env_path = SCRIPT_PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    env_vars[k] = v.strip('"\'')
    
    if "CLAUDE_API_KEY" not in env_vars:
        print("Error: CLAUDE_API_KEY not found in .env")
        return

    full_env = os.environ.copy()
    full_env.update(env_vars)
    full_env["ANTHROPIC_API_KEY"] = env_vars.get("CLAUDE_API_KEY", full_env.get("CLAUDE_API_KEY"))
    
    pubmed_email = full_env.get("PUBMED_EMAIL")
    pubmed_api_key = full_env.get("PUBMED_API_KEY")

    if not pubmed_email:
        print("Error: PUBMED_EMAIL not found in .env")
        return

    options = ClaudeAgentOptions(
        model="claude-3-5-sonnet-20241022", # Use a capable model
        cwd=str(SCRIPT_PROJECT_ROOT),
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
    
    # Load corpus (limit to first 5000 for speed if needed, but we need random access)
    corpus = {}
    corpus_keys = []
    if corpus_path.exists():
        with open(corpus_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    doc = json.loads(line)
                    did = int(doc["doc_id"])
                    corpus[did] = doc
                    corpus_keys.append(did)
                    if i > 50000: break # Load substantial amount
                except: pass
    print(f"Loaded {len(corpus)} documents.")

    # Load claims
    with open(claims_path, "r") as f:
        all_claims = [json.loads(line) for line in f]
    
    # Filter for claims that have evidence in our corpus
    viable_claims = []
    for c in all_claims:
        if not c.get("evidence"): continue
        has_corpus_doc = any(int(did) in corpus for did in c["evidence"].keys())
        if has_corpus_doc:
            viable_claims.append(c)

    # --- Selection Logic ---
    # Pick 1 random viable claim
    target_claim = random.choice(viable_claims)
    claim_id = target_claim["id"]
    claim_text = target_claim["claim"]
    
    # Get associated evidence IDs
    evidence_ids = {int(did) for did in target_claim["evidence"].keys()}
    
    # Pick a random doc that is NOT in evidence_ids
    while True:
        random_doc_id = random.choice(corpus_keys)
        if random_doc_id not in evidence_ids:
            break
            
    mismatch_doc = corpus[random_doc_id]
    mismatch_title = mismatch_doc.get("title", "")
    mismatch_pmid = str(mismatch_doc.get("pmid")) if mismatch_doc.get("pmid") else None

    print(f"\n--- Test Case: Mismatch (Real Data) ---")
    print(f"Claim (ID {claim_id}): {claim_text}")
    print(f"Unrelated Doc (ID {random_doc_id}): {mismatch_title}")
    if mismatch_pmid:
        print(f"Unrelated PMID: {mismatch_pmid}")
    print("-" * 60)

    # --- Execution ---
    
    # 1. Search & Retrieve (Claude)
    # We pretend this random doc is the "retrieved" evidence
    print(f"Retrieving full text for Doc {random_doc_id}...")
    search_res = await run_search_and_retrieval(
        options, 
        random_doc_id, 
        mismatch_title, 
        mismatch_pmid, 
        f"test_mismatch_{claim_id}"
    )
    
    output_dir = search_res.get("output_dir")
    full_text_content = ""
    
    if output_dir:
        # Check full text
        ft_path = output_dir / "full_text.txt"
        if ft_path.exists():
            with open(ft_path, "r") as f:
                full_text_content = f.read()
            print(f"  [SUCCESS] Retrieved full text ({len(full_text_content)} chars)")
        else:
            print("  [WARNING] Full text not found. Using abstract as fallback.")
            if mismatch_doc.get("abstract"):
                full_text_content = " ".join(mismatch_doc["abstract"])

    text_for_nlp = clean_text(full_text_content)
    
    if not text_for_nlp:
        print("  [ERROR] No text available for NLP features. Aborting.")
        return

    # 2. Compute NLP Features
    print("Initializing NLP components...")
    sim_computer = SemanticSimilarityComputer()
    extractor = BiomedicalEntityExtractor()
    await extractor.__aenter__()

    try:
        print("Computing features...")
        
        # A) Entity Extraction
        claim_ents = await extractor.extract(claim_text)
        evidence_ents = await extractor.extract(text_for_nlp) # This effectively truncates if too long? No, extract handles it via chunking usually or model limit
        # The default ExtractEntities tool might have a limit. For now assume it works or returns partial.
        
        # B) Coverage
        coverage = compute_recall_from_entities(claim_ents, evidence_ents)
        print(f"  Entity Coverage:   {coverage:.4f} (Expected: Low/0.0)")

        # C) Semantic Similarity
        similarity = sim_computer.compute(claim_text, text_for_nlp)
        print(f"  SBERT Similarity:  {similarity:.4f} (Expected: In-domain mismatch typically 0.3-0.5)")

        # Validation
        passed = True
        if coverage > 0.3:
            print(f"❌ Coverage unreasonably high for mismatch ({coverage:.4f})")
            passed = False
        else:
             print(f"✅ Coverage within expected low range")
             
        # In-domain biomedical text (e.g. Cancer A vs Cancer B) often has similarity 0.3-0.5 due to shared vocabulary.
        # We only flag if it's deceptively high (>0.6).
        if similarity > 0.6: 
            print(f"❌ Similarity unreasonably high for mismatch ({similarity:.4f})")
            passed = False
        else:
             print(f"✅ Similarity within expected in-domain noise range (< 0.6)")
             
    finally:
        await extractor.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
