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
import time
from collections.abc import Callable
from pathlib import Path

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import normalize_label
from baselines.shared.llm import LLMBackend
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.prompts import (
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_TEMPLATE,
)
from baselines.shared.search_utils import (
    ddg_text_search_with_retry,
    format_ddg_passages,
    format_s2_passages,
)
from baselines.shared.single_shot import run_single_shot_verdict
from baselines.shared.verdict import BaselineResult
from proclaim.search.semantic_scholar import S2Client, S2RateLimitError

logger = logging.getLogger(__name__)

_SIGNOR_CLAIM_RE = re.compile(r"^(?P<base>.*?)\s*\([^)]*\)\s*\.?$")
_CONNECTOMEDB_CLAIM_RE = re.compile(
    r"^(?P<ligand>\S+)\s+as\s+ligand\s+directly\s+interacts(?:\s+extracellularly)?\s+with\s+(?P<receptor>\S+)\s+as\s+receptor\.?$"
)

# ---------------------------------------------------------------------------
# Query processors — transform claim into a focused search query
# ---------------------------------------------------------------------------

def query_process_signor(claim: str) -> str:
    """Strip parenthetical boilerplate from SIGNOR claims.

    SIGNOR claims look like:
        "GNAS directly activates ADCY1."
    """
    match = _SIGNOR_CLAIM_RE.match(claim.strip())
    # if not match:
    #     logger.warning("SIGNOR claim did not match expected format: %r", claim)
    #     return claim.strip()
    return match.group("base").strip()


def query_process_connectomedb(claim: str) -> str:
    """Extract entity names from ConnectomeDB claims.

    ConnectomeDB claims look like:
        "A2M as ligand directly interacts extracellularly with HSPA5
         as receptor."
    The boilerplate ("as ligand directly interacts extracellularly with ...
    as receptor") drowns out entity names in keyword search. Extract just the
    two protein names.
    """
    m = _CONNECTOMEDB_CLAIM_RE.match(claim.strip())
    if m:
        return f"{m.group('ligand')} {m.group('receptor')} protein interaction"

    logger.warning("ConnectomeDB claim did not match expected format: %r", claim)
    return claim.strip()


# Registry for convenient lookup by dataset name
QUERY_PROCESSORS: dict[str, Callable[[str, dict | None], str]] = {
    "signor": query_process_signor,
    "connectomedb": query_process_connectomedb,
}


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
        evidence_text = format_s2_passages(papers) if papers else ""
        return evidence_text, papers

    def _search_web(self, query: str) -> tuple[str, list[dict]]:
        """Search the web and return (formatted_evidence, raw_results)."""
        results = ddg_text_search_with_retry(query, k=self.top_k)
        evidence_text = format_ddg_passages(results, empty_message="") if results else ""
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
        verdict = run_single_shot_verdict(
            llm=self.llm,
            tracker=self.tracker,
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_msg,
        )

        # Write per-claim log
        result_lines = []
        for result in raw_results:
            if self.search_backend == "s2":
                title = result.get("title", "Untitled")
                pmid = (result.get("externalIds") or {}).get("PubMed", "")
                result_lines.append(f"  [{pmid or 'no-pmid'}] {title}")
            else:
                title = result.get("title", "Untitled")
                link = result.get("link", "")
                result_lines.append(f"  {title}  {link}")
        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}", ""),
                ("claim", claim),
                ("system prompt", VERIFICATION_SYSTEM_PROMPT),
                ("user prompt", user_msg),
                (f"{self.search_backend} results ({len(raw_results)})", "\n".join(result_lines)),
                (f"predicted: {verdict.predicted_label}", ""),
                ("reasoning", verdict.reasoning),
                ("raw LLM response", verdict.text),
            ],
        )

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=verdict.predicted_label,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            evidence=verdict.evidence,
            input_tokens=verdict.summary["input_tokens"],
            output_tokens=verdict.summary["output_tokens"],
            cost_usd=verdict.summary["cost_usd"],
            latency_seconds=verdict.summary["latency_seconds"],
            baseline_name=self.name,
            model=self.llm.model,
        )
