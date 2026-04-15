"""
LLM + static S2 retrieval baseline — fixed-k Semantic Scholar RAG.

Uses the claim text to search Semantic Scholar, retrieves the top-k
abstracts, concatenates them as context, and asks the LLM for a verdict
in a single call.  No iteration, no feedback loop.

This is the "paste search results into ChatGPT" baseline for scientific
claim verification, using Semantic Scholar as the retrieval backend.

Search backends (checked in order):
  1. S2 relevance search (``/paper/search``) with the raw claim as query.
  2. Falls back to UNCERTAIN if the API returns no results.

Cost: 1 S2 API call + 1 LLM call per claim.
"""

from __future__ import annotations

import logging
import re
import time
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


def _format_passages(papers: list[dict]) -> str:
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


class S2Retrieval:
    """Retrieve top-k abstracts from Semantic Scholar, then classify with one LLM call."""

    name = "s2_retrieval"

    def __init__(
        self,
        llm: LLMBackend,
        top_k: int = 5,
        strip_parens: bool = True,
    ) -> None:
        self.llm = llm
        self.top_k = top_k
        self.strip_parens = strip_parens
        self.tracker = CostTracker(model=llm.model)
        self._s2 = S2Client()
        self.log_dir: Path | None = None

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

        # 1. Retrieve from Semantic Scholar
        # Optionally strip parenthetical boilerplate so the query focuses on
        # key entities.  Long parenthetical clauses (e.g. SIGNOR mechanism
        # descriptions) dominate S2's relevance ranking and drown out the
        # entity names that actually distinguish one claim from another.
        if self.strip_parens:
            query = re.sub(r"\s*\([^)]*\)\s*", " ", claim).strip()
        else:
            query = claim
        try:
            papers = self._s2.search(query, limit=self.top_k)
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
            "s2_search", latency=time.monotonic() - t0, query=query
        )

        if not papers:
            logger.warning("No S2 results for claim %s — defaulting to UNCERTAIN", claim_id)
            return BaselineResult(
                claim_id=claim_id,
                claim=claim,
                gold_label=normalize_label(gold_label),
                predicted_label="UNCERTAIN",
                reasoning="No papers found via Semantic Scholar search.",
                baseline_name=self.name,
                model=self.llm.model,
                latency_seconds=round(time.monotonic() - t0, 2),
            )

        # 2. Format passages as evidence
        evidence_text = _format_passages(papers)

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
                lf.write(f"=== S2 papers ({len(papers)}) ===\n")
                for p in papers:
                    title = p.get('title', 'Untitled')
                    pmid = (p.get('externalIds') or {}).get('PubMed', '')
                    lf.write(f"  [{pmid or 'no-pmid'}] {title}\n")
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
