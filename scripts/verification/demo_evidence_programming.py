#!/usr/bin/env python3
"""
Demo: Portable Evidence Programming Agent.

Shows the evidence programming loop end-to-end using any OpenAI-compatible
LLM endpoint. Calls MCP tool functions directly as Python — no MCP server
protocol or Claude Agent SDK needed.

The LLM is only used for 3 steps:
  1. Fact extraction     — read paper, produce structured facts
  2. Gap query formulation — turn classifier feedback into PubMed queries
  3. Verdict reasoning   — synthesize evidence into a final judgment

Everything else (search, sufficiency check, compression) is deterministic.

Usage:
  # Local vLLM endpoint
  python scripts/verification/demo_evidence_programming.py \
      --preset local --claim "Does p53 activate BAX?"

  # Anthropic (OpenAI-compatible)
  python scripts/verification/demo_evidence_programming.py \
      --preset anthropic --claim "Does GNAS directly activate ADCY1?"

  # GLM
  python scripts/verification/demo_evidence_programming.py \
      --preset glm --claim "Does p53 activate BAX?"

  # Custom endpoint
  python scripts/verification/demo_evidence_programming.py \
      --base-url http://my-server:8000/v1 --api-key my-key \
      --model my-model --claim "Does p53 activate BAX?"
"""

import json
import logging
import os
import sys
import argparse
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from pkevolve.verification.evidence_state import EvidenceState  # noqa: E402
from pkevolve.verification.mcp_tools import (  # noqa: E402
    add_facts,
    check_sufficiency,
    compress_evidence,
    emit_verdict,
    find_related_articles,
    get_evidence_summary,
    get_full_text_article,
    get_paper_text,
    search_for_gap,
    search_pubmed,
    search_pubmed_progressive,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint Presets
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
    # "glm": {
    #     "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    #     "api_key_env": "GLM_API_KEY",
    #     "model": "glm-4.6",
    # },
    # "anthropic": {
    #     "base_url": "https://api.anthropic.com/v1/",
    #     "api_key_env": "ANTHROPIC_API_KEY",
    #     "model": "claude-sonnet-4-5-20250929",
    # },
}


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around openai.OpenAI for any compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        """Single-turn chat completion. Returns the assistant message text."""
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
# LLM-powered steps (3 total)
# ---------------------------------------------------------------------------


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


def run_demo(
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

    # --- Initial retrieval (progressive) ------------------------------------
    print("[SEARCH] Progressive search …")
    result = search_pubmed_progressive(claim, ws)
    for line in result.splitlines():
        print(f"  {line}")

    # If progressive search still found nothing, ask LLM for a query
    state = EvidenceState.load(workspace / "evidence_state.json")
    if not state.papers:
        print("\n[SEARCH] Progressive search returned 0 papers, "
              "using LLM to reformulate …")
        try:
            llm_query = llm_formulate_gap_query(
                client, f"No papers found for: {claim}",
                "missing_subclaim_evidence", claim,
            )
            print(f"  LLM query: {llm_query}")
            result = search_pubmed(llm_query, ws, max_results=10)
            print(f"  {result}")
        except Exception as exc:
            print(f"  LLM query formulation error: {exc}")

    processed_pmids: set[str] = set()
    failed_queries: list[str] = []  # Track queries that returned 0
    used_seed_pmids: set[str] = set()  # Track PMIDs already used for related articles

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'─' * 60}")

        # -- Extract facts from new papers -----------------------------------
        state = EvidenceState.load(workspace / "evidence_state.json")
        new_pmids = [p for p in state.papers if p not in processed_pmids]

        if new_pmids:
            print(f"\n[EXTRACT] {len(new_pmids)} new paper(s) to process")
            for pmid in new_pmids:
                # Try full text first, fall back to abstract
                text = get_full_text_article(pmid, ws)
                if "Falling back to abstract" in text or "not found" in text:
                    text = get_paper_text(pmid, ws)
                    print(f"  PMID {pmid}: extracting facts (abstract) …")
                else:
                    print(f"  PMID {pmid}: extracting facts (full text) …")
                try:
                    facts_json = llm_extract_facts(
                        client, claim, text, pmid, state.subclaims,
                    )
                    msg = add_facts(facts_json, ws)
                    print(f"    {msg}")
                except Exception as exc:
                    print(f"    Error: {exc}")
                processed_pmids.add(pmid)
        else:
            print("\n[EXTRACT] No new papers.")

        # -- Sufficiency check -----------------------------------------------
        print("\n[SUFFICIENCY] Running classifier …")
        suff_text = check_sufficiency(ws)
        for line in suff_text.splitlines():
            print(f"  {line}")

        # Determine outcome from structured state
        iteration_limit = "ITERATION LIMIT" in suff_text
        if iteration_limit:
            print("\n  → Iteration limit reached. Proceeding to verdict.")
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
                    result = search_for_gap(query, gap.gap_type.value, ws)
                    print(f"  {result}")
                    # Track queries that found nothing
                    if "added 0 new" in result and "Found 0" in result:
                        failed_queries.append(query)
                except Exception as exc:
                    print(f"  Error: {exc}")

        # -- Related articles fallback ---------------------------------------
        state = EvidenceState.load(workspace / "evidence_state.json")
        unprocessed = [p for p in state.papers if p not in processed_pmids]
        if not unprocessed and state.papers:
            # No new papers from keyword search; try citation graph
            # Rotate through seed PMIDs to discover new neighborhoods
            candidate_seeds = [
                p for p in state.papers if p not in used_seed_pmids
            ]
            if candidate_seeds:
                seed_pmid = candidate_seeds[0]
                used_seed_pmids.add(seed_pmid)
                print(f"\n[RELATED] Keyword search found no new papers. "
                      f"Trying citation graph from PMID {seed_pmid} …")
                result = find_related_articles(seed_pmid, ws)
                print(f"  {result}")
            else:
                print("\n[RELATED] All seed PMIDs exhausted for citation graph.")

        # -- Compress if needed ----------------------------------------------
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
        emit_verdict(
            verdict="INSUFFICIENT", confidence=0.0,
            reasoning=f"Verdict generation failed: {exc}",
            key_evidence_json="[]",
            gaps_remaining_json='["Verdict generation failed"]',
            workspace=ws,
        )

    print(f"\nWorkspace: {workspace}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evidence Programming demo — any OpenAI-compatible endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
    )
    print(f"Endpoint: {cfg['base_url']}  Model: {cfg['model']}")

    # -- Workspace -----------------------------------------------------------
    workspace = Path(args.output_dir) if args.output_dir else Path(
        tempfile.mkdtemp(prefix="demo_ep_")
    )

    run_demo(
        claim=args.claim,
        workspace=workspace,
        client=client,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
