"""
LLM + static retrieval baseline — fixed-k RAG with configurable search backend.

Uses the claim text to search for evidence, retrieves the top-k results,
concatenates them as context, and asks the LLM for a verdict in a single call.
No iteration, no feedback loop.

This is the "paste search results into ChatGPT" baseline for scientific
claim verification.

Search backends:
  ``search_backend="s2"`` (default):
    Semantic Scholar relevance search (``/paper/search``).
  ``search_backend="web"``:
    Web search via DuckDuckGo (``ddgs`` package, free, no API key).

Cost: 1 search API call + 1 LLM call per claim.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import normalize_label
from baselines.shared.llm import LLMBackend
from baselines.shared.prompts import (
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_TEMPLATE,
)
from baselines.shared.verdict import BaselineResult
from pkevolve.search.semantic_scholar import S2Client, S2RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query processors — transform claim into a focused search query
# ---------------------------------------------------------------------------

def query_process_signor(claim: str) -> str:
    """Strip parenthetical boilerplate from SIGNOR claims.

    SIGNOR claims look like:
        "GNAS directly activates ADCY1 (either through post-translational
         modification, complex formation, or direct regulation of expression)."
    The parenthetical drowns out the entity names in keyword search.
    """
    return re.sub(r"\s*\([^)]*\)\s*", " ", claim).strip()


def query_process_connectomedb(claim: str) -> str:
    """Extract entity names from ConnectomeDB claims.

    ConnectomeDB claims look like:
        "In the context of protein-protein interactions, A2M as ligand
         directly interacts with HSPA5 as receptor."
    Even after stripping the prefix, the remaining boilerplate ("as ligand
    directly interacts with ... as receptor") drowns out entity names in
    keyword search.  Extract just the two protein names.
    """
    m = re.search(
        r"(\S+)\s+as\s+ligand\s+directly\s+interacts\s+with\s+(\S+)\s+as\s+receptor",
        claim,
    )
    if m:
        return f"{m.group(1)} {m.group(2)} protein interaction"
    # Last resort: strip the prefix before the comma
    _, _, rest = claim.partition(",")
    return rest.strip() if rest else claim


# Registry for convenient lookup by dataset name
QUERY_PROCESSORS: dict[str, Callable[[str, dict | None], str]] = {
    "signor": query_process_signor,
    "connectomedb": query_process_connectomedb,
}


# ---------------------------------------------------------------------------
# Evidence formatting helpers
# ---------------------------------------------------------------------------

def _format_s2_passages(papers: list[dict]) -> str:
    """Format a list of S2 paper dicts into numbered evidence passages."""
    parts: list[str] = []
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            abstract = "(no abstract available)"

        # Extract PMID if present
        ext_ids = paper.get("externalIds") or {}
        pmid = ext_ids.get("PubMed", "")
        pmid_str = f"  PMID: {pmid}" if pmid else ""

        parts.append(
            f"[{i}] {title} ({year}){pmid_str}\n{abstract}"
        )
    return "\n\n".join(parts)


def _format_web_passages(snippets: list[dict]) -> str:
    """Format a list of web search result dicts into numbered evidence passages."""
    parts: list[str] = []
    for i, item in enumerate(snippets, 1):
        title = item.get("title", "Untitled")
        url = item.get("link", "")
        snippet = item.get("snippet", item.get("body", ""))
        url_str = f"  {url}" if url else ""
        parts.append(f"[{i}] {title}{url_str}\n{snippet}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Web search backend (DuckDuckGo)
# ---------------------------------------------------------------------------

_DDG_LOCK = threading.Lock()


def _ddg_search(query: str, k: int = 5) -> list[dict]:
    """Query DuckDuckGo via the ddgs package (free, no API key)."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("ddgs not installed. Run: pip install ddgs")

    results: list[dict] = []
    try:
        with _DDG_LOCK:
            ddgs = DDGS()
            for result in ddgs.text(query, max_results=k):
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                    "snippet": result.get("body", ""),
                })
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)

    return results


def _web_search(query: str, k: int = 5, _max_retries: int = 3) -> list[dict]:
    """Execute a web search with retries on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_max_retries):
        try:
            return _ddg_search(query, k=k)
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt * 2
            logger.warning(
                "Web search error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, _max_retries, wait, exc,
            )
            time.sleep(wait)
    logger.error("Web search failed after %d retries: %s", _max_retries, last_exc)
    return []


# ---------------------------------------------------------------------------
# Retrieval class
# ---------------------------------------------------------------------------

class RetrievalBaseline:
    """Retrieve top-k results from S2 or the web, then classify with one LLM call.

    Parameters
    ----------
    llm:
        ``LLMBackend`` instance for the classification call.
    top_k:
        Number of search results to retrieve.
    search_backend:
        ``"s2"`` (default) for Semantic Scholar, ``"web"`` for web search.
    query_process:
        Optional function to transform claim into a focused search query.
    """

    name = "retrieval"

    def __init__(
        self,
        llm: LLMBackend,
        top_k: int = 5,
        search_backend: str = "s2",
        query_process: Callable[[str, dict | None], str] | None = None,
    ) -> None:
        if search_backend not in ("web", "s2"):
            raise ValueError(f"search_backend must be 'web' or 's2', got {search_backend!r}")
        self.llm = llm
        self.top_k = top_k
        self.search_backend = search_backend
        self.query_process = query_process
        self.tracker = CostTracker(model=llm.model)
        if search_backend == "s2":
            self._s2 = S2Client()
        self.log_dir: Path | None = None

    # ------------------------------------------------------------------
    # Search dispatch
    # ------------------------------------------------------------------

    def _search_s2(self, query: str) -> tuple[str, list[dict]]:
        """Search Semantic Scholar and return (formatted_evidence, raw_papers)."""
        papers = self._s2.search(query, limit=self.top_k)
        evidence_text = _format_s2_passages(papers) if papers else ""
        return evidence_text, papers

    def _search_web(self, query: str) -> tuple[str, list[dict]]:
        """Search the web and return (formatted_evidence, raw_results)."""
        results = _web_search(query, k=self.top_k)
        evidence_text = _format_web_passages(results) if results else ""
        return evidence_text, results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        *,
        context: dict | None = None,
    ) -> BaselineResult:
        self.tracker.reset()
        t0 = time.monotonic()

        # 1. Build search query
        if self.query_process is not None:
            query = self.query_process(claim)
        else:
            query = claim

        # 2. Retrieve evidence
        try:
            if self.search_backend == "s2":
                evidence_text, raw_results = self._search_s2(query)
            else:
                evidence_text, raw_results = self._search_web(query)
        except S2RateLimitError as exc:
            logger.error("S2 rate-limit exhausted for claim %s: %s", claim_id, exc)
            return BaselineResult(
                claim_id=claim_id,
                claim=claim,
                gold_label=normalize_label(gold_label),
                predicted_label="UNCERTAIN",
                reasoning=f"S2 rate-limit exhausted: {exc}",
                baseline_name=self.name,
                model=self.llm.model,
                latency_seconds=round(time.monotonic() - t0, 2),
            )

        self.tracker.record(
            f"{self.search_backend}_search", latency=time.monotonic() - t0, query=query
        )

        if not raw_results:
            logger.warning(
                "No %s results for claim %s — defaulting to UNCERTAIN",
                self.search_backend, claim_id,
            )
            return BaselineResult(
                claim_id=claim_id,
                claim=claim,
                gold_label=normalize_label(gold_label),
                predicted_label="UNCERTAIN",
                reasoning=f"No results found via {self.search_backend} search.",
                baseline_name=self.name,
                model=self.llm.model,
                latency_seconds=round(time.monotonic() - t0, 2),
            )

        # 3. Single LLM call with retrieval prompt
        user_msg = VERIFICATION_USER_TEMPLATE.format(
            claim=claim, evidence=evidence_text
        )
        t_llm = time.monotonic()
        text, in_tok, out_tok = self.llm.complete(
            system=VERIFICATION_SYSTEM_PROMPT,
            user=user_msg,
        )
        self.tracker.record("llm_call", in_tok, out_tok, time.monotonic() - t_llm)

        # 4. Parse response
        parsed = self.llm.parse_json(text)
        raw_label = parsed.get("label", "UNCERTAIN")
        predicted = normalize_label(raw_label)
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning", text[:500] if text else "")
        cited = parsed.get("evidence", [])
        if isinstance(cited, str):
            cited = [cited]

        summary = self.tracker.summary()

        # Write per-claim log
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{claim_id}.log"
            with open(log_path, "w") as lf:
                lf.write(f"=== claim_id: {claim_id} ===\n")
                lf.write(f"=== claim ===\n{claim}\n\n")
                lf.write(f"=== {self.search_backend} results ({len(raw_results)}) ===\n")
                for r in raw_results:
                    if self.search_backend == "s2":
                        title = r.get('title', 'Untitled')
                        pmid = (r.get('externalIds') or {}).get('PubMed', '')
                        lf.write(f"  [{pmid or 'no-pmid'}] {title}\n")
                    else:
                        title = r.get('title', 'Untitled')
                        link = r.get('link', '')
                        lf.write(f"  {title}  {link}\n")
                lf.write(f"\n=== predicted: {predicted} ===\n")
                lf.write(f"=== reasoning ===\n{reasoning}\n")
                lf.write(f"=== raw LLM response ===\n{text}\n")

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=confidence,
            reasoning=reasoning,
            evidence=cited,
            input_tokens=summary["input_tokens"],
            output_tokens=summary["output_tokens"],
            cost_usd=summary["cost_usd"],
            latency_seconds=summary["latency_seconds"],
            baseline_name=self.name,
            model=self.llm.model,
        )
