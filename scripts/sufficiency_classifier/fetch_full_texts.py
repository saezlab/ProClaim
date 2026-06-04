"""
Fetch full text from PubMed for SciFact evidence documents using
mcp-simple-pubmed MCP server directly (no Claude SDK needed).

Reads doc_id -> PMID mapping from cache, calls get_paper_fulltext
via MCP client, and saves results to the expected directory structure.

Usage:
    uv run scripts/sufficiency_classifier/fetch_full_texts.py
    uv run scripts/sufficiency_classifier/fetch_full_texts.py --limit 10
    uv run scripts/sufficiency_classifier/fetch_full_texts.py --skip-existing
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from contextlib import AsyncExitStack

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PMID_CACHE_FILE = PROJECT_ROOT / "data" / "doc_id_to_pmid_cache.json"
FULL_TEXT_DIR = PROJECT_ROOT / "results" / "claude_sdk" / "scifact_feature_extraction"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_env() -> dict:
    """Load .env file for PubMed credentials."""
    env_path = PROJECT_ROOT / ".env"
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip("\"'")
    return env_vars


class PubMedMCPClient:
    """Direct MCP client for mcp-simple-pubmed server."""

    def __init__(self, pubmed_email: str, pubmed_api_key: str = None):
        self.pubmed_email = pubmed_email
        self.pubmed_api_key = pubmed_api_key
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        env = {
            "PUBMED_EMAIL": self.pubmed_email,
        }
        if self.pubmed_api_key:
            env["PUBMED_API_KEY"] = self.pubmed_api_key

        # Inherit PATH so uvx can be found
        full_env = os.environ.copy()
        full_env.update(env)

        params = StdioServerParameters(
            command="uvx",
            args=["--python", "3.12", "mcp-simple-pubmed"],
            env=full_env,
        )

        self.read, self.write = await self.exit_stack.enter_async_context(
            stdio_client(params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )
        await self.session.initialize()
        logging.info("MCP PubMed server connected.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def get_full_text(self, pmid: str) -> str | None:
        """Call get_paper_fulltext MCP tool and return the text content."""
        try:
            result = await self.session.call_tool(
                "get_paper_fulltext",
                arguments={"pmid": pmid},
            )
            if result.content and hasattr(result.content[0], "text"):
                text = result.content[0].text
                # Check if it's an error message
                if text and not text.startswith("Error"):
                    return text
                else:
                    logging.warning(f"PMID {pmid}: server returned error: {text[:100]}")
                    return None
            return None
        except Exception as e:
            logging.warning(f"PMID {pmid}: MCP call failed: {e}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Fetch full texts from PubMed via MCP")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of docs to fetch (for testing)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip docs that already have full_text.txt")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests (seconds)")
    args = parser.parse_args()

    # Load credentials
    env_vars = load_env()
    email = env_vars.get("PUBMED_EMAIL")
    api_key = env_vars.get("PUBMED_API_KEY")

    if not email:
        logging.error("PUBMED_EMAIL not found in .env")
        sys.exit(1)

    # Load PMID cache
    if not PMID_CACHE_FILE.exists():
        logging.error(f"PMID cache not found: {PMID_CACHE_FILE}")
        logging.error("Run resolve_pmids.py first.")
        sys.exit(1)

    with open(PMID_CACHE_FILE) as f:
        pmid_cache = json.load(f)

    # Filter to docs with valid PMIDs
    docs_to_fetch = []
    for doc_id, pmid in pmid_cache.items():
        if not pmid or not pmid.strip().isdigit():
            continue

        output_dir = FULL_TEXT_DIR / f"doc_{doc_id}"
        full_text_path = output_dir / "full_text.txt"

        if args.skip_existing and full_text_path.exists():
            continue

        docs_to_fetch.append((doc_id, pmid.strip()))

    logging.info(f"PMID cache: {len(pmid_cache)} entries, "
                 f"{sum(1 for v in pmid_cache.values() if v)} with PMIDs")
    logging.info(f"Docs to fetch: {len(docs_to_fetch)}")

    if args.limit:
        docs_to_fetch = docs_to_fetch[:args.limit]
        logging.info(f"Limited to {len(docs_to_fetch)} docs")

    if not docs_to_fetch:
        logging.info("Nothing to fetch. Done.")
        return

    # Connect to MCP server and fetch
    async with PubMedMCPClient(email, api_key) as client:
        success = 0
        failed = 0
        skipped_short = 0

        for i, (doc_id, pmid) in enumerate(docs_to_fetch):
            logging.info(f"[{i+1}/{len(docs_to_fetch)}] Doc {doc_id} (PMID {pmid})...")

            text = await client.get_full_text(pmid)

            if text and len(text) > 1000:
                output_dir = FULL_TEXT_DIR / f"doc_{doc_id}"
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / "full_text.txt", "w") as f:
                    f.write(text)
                success += 1
                logging.info(f"  ✓ Saved ({len(text)} chars)")
            elif text:
                skipped_short += 1
                logging.warning(f"  ⚠ Text too short ({len(text)} chars), skipped")
            else:
                failed += 1
                logging.warning(f"  ✗ No full text available")

            # Rate limit
            await asyncio.sleep(args.delay)

    logging.info(f"\nDone! Success: {success}, Failed: {failed}, Too short: {skipped_short}")
    logging.info(f"Full texts saved to: {FULL_TEXT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
