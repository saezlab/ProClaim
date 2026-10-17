"""MCP stdio server exposing ProClaim's PubMed + Semantic Scholar search.

Thin wrappers around ``react_baseline``'s ``_pubmed_search``/``_s2_search``, so
the evidence-source ablation reuses the same retrieval path as ProClaim itself.
Wired in via ``ClaudeAgentOptions.mcp_servers`` in claude_code_baseline.py.

NOTE: the tool docstrings below are sent to the model as tool descriptions —
they are prompt text, not documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The SDK's MCP client ignores a stdio server's "cwd", so `-m` can't resolve
# the package; insert the path here instead.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from mcp.server.fastmcp import FastMCP

from baselines.react_baseline import _pubmed_search, _s2_search

mcp = FastMCP("pubmed-s2")


@mcp.tool()
def search_pubmed(query: str) -> str:
    """Search PubMed for academic papers relevant to verifying a claim.

    This is the primary biomedical literature source — use it first. Returns
    paper titles, authors, abstracts, and metadata. Use specific, targeted
    queries mentioning key entities and relationships from the claim.
    """
    return _pubmed_search(query, k=5)


@mcp.tool()
def search_semantic_scholar(query: str) -> str:
    """Search Semantic Scholar for academic papers relevant to verifying a
    claim.

    Complements PubMed — it additionally covers preprints (e.g. bioRxiv) and
    non-MEDLINE sources PubMed does not index. Returns paper titles, authors,
    abstracts, and metadata. Use specific, targeted queries mentioning key
    entities and relationships from the claim.
    """
    return _s2_search(query, k=5)


if __name__ == "__main__":
    mcp.run()
