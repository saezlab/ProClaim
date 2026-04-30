"""
SAFE baseline — search-augmented factuality evaluation adapted to claim verification.

This baseline implements the claim-rating portion of SAFE (Wei et al., NeurIPS
2024): the model iteratively proposes search queries, accumulates retrieved
evidence, and then renders a final verdict from that evidence. The original SAFE
pipeline decomposes long-form generations into atomic facts first; here the input
datasets already provide atomic claims, so this wrapper applies the rating loop
directly to each claim.

Differences vs the original SAFE paper:
  - Input is a single atomic claim, so no fact decomposition step is needed.
  - Final prediction uses the shared claim-verification taxonomy
    {SUPPORT, REFUTE, UNCERTAIN} instead of SAFE's binary
    {Supported, Not Supported} labels.
  - Search can target either the open web (default, closest to SAFE) or
    Semantic Scholar for literature-only ablations.

Cost: 1–(max_steps + 1) LLM calls per claim + searches.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import (
    normalize_label,
    validate_verdict,
    verdict_defs_block,
    verdict_names,
)
from baselines.shared.llm import LLMBackend
from baselines.shared.verdict import BaselineResult
from pkevolve.search.semantic_scholar import S2Client, S2RateLimitError

logger = logging.getLogger(__name__)

_VERDICT_NAMES = verdict_names()
_VERDICT_DEFS = verdict_defs_block()

_NEXT_SEARCH_PROMPT = """\
Instructions:
1. You have been given a scientific CLAIM and some accumulated KNOWLEDGE.
2. Your goal is to find one additional search query that would provide new,
   useful evidence for verifying the CLAIM.
3. The query should target evidence not already covered by the KNOWLEDGE.
4. Prefer concise, entity-focused scientific queries.
5. Return only the query inside a markdown code block.

KNOWLEDGE:
{knowledge}

CLAIM:
{claim}
"""

_FINAL_ANSWER_PROMPT = """\
Instructions:
1. You have been given a scientific CLAIM and some KNOWLEDGE.
2. Determine whether the CLAIM should be labeled as one of the following:
{verdict_defs}
3. Use SUPPORT when the knowledge directly supports or strongly implies the claim.
4. Use REFUTE when the knowledge directly contradicts the claim.
5. Use UNCERTAIN when the knowledge is mixed, indirect, or insufficient.
6. Think step-by-step and summarize the strongest evidence before the answer.
7. End with a final answer in square brackets, using exactly one of:
   [{verdict_names}]

KNOWLEDGE:
{knowledge}

CLAIM:
{claim}
"""

_DDG_LOCK = threading.Lock()


@dataclass
class _SearchResult:
    query: str
    result: str


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    if not match:
        return None
    query = match.group(1).strip()
    return query or None


def _extract_square_bracket_answer(text: str) -> str | None:
    matches = re.findall(r"\[([^\]]+)\]", text)
    if not matches:
        return None
    return matches[-1].strip()


def _token_overlap(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _count_similar(target: str, history: list[str], threshold: float = 0.9) -> int:
    return sum(1 for item in history if _token_overlap(target, item) >= threshold)


def _format_s2_results(papers: list[dict]) -> str:
    parts: list[str] = []
    for index, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = (paper.get("abstract") or "").strip() or "(no abstract available)"
        ext_ids = paper.get("externalIds") or {}
        pmid = ext_ids.get("PubMed", "")
        pmid_text = f" PMID:{pmid}" if pmid else ""
        parts.append(f"[{index}] {title} ({year}){pmid_text}\n{abstract}")
    return "\n\n".join(parts) if parts else "No relevant papers found."


def _ddg_search(query: str, k: int) -> str:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("ddgs not installed. Run: pip install ddgs") from exc

    snippets: list[str] = []
    try:
        with _DDG_LOCK:
            ddgs = DDGS()
            for result in ddgs.text(query, max_results=k):
                title = result.get("title", "")
                body = result.get("body", "")
                href = result.get("href", "")
                parts = [part for part in (title, body, href) if part]
                if parts:
                    snippets.append(" | ".join(parts))
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)

    return "\n".join(snippets) if snippets else "No relevant web results found."


class SAFEBaseline:
    """SAFE claim-rating baseline using iterative search and a final evidence judgment."""

    name = "safe"

    def __init__(
        self,
        llm: LLMBackend,
        *,
        max_steps: int = 5,
        max_retries: int = 10,
        max_tolerance: int = 2,
        num_search_results: int = 3,
        search_backend: str = "web",
    ) -> None:
        if search_backend not in ("web", "s2"):
            raise ValueError(f"search_backend must be 'web' or 's2', got {search_backend!r}")

        self._llm = llm
        self.model = llm.model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.max_tolerance = max_tolerance
        self.num_search_results = num_search_results
        self.search_backend = search_backend
        self.log_dir: Path | None = None

        self._s2 = S2Client() if search_backend == "s2" else None

        model_key = self.model.split("/", 1)[-1] if "/" in self.model else self.model
        in_price, out_price = CostTracker.DEFAULT_PRICING.get(
            model_key, CostTracker.FALLBACK_PRICING,
        )
        self._in_price = in_price
        self._out_price = out_price

    def verify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        *,
        context: dict | None = None,
    ) -> BaselineResult:
        del context
        t0 = time.monotonic()
        usage = {"input_tokens": 0, "output_tokens": 0}
        searches: list[_SearchResult] = []

        try:
            answer, reasoning = self._safe_loop(claim, searches, usage)
        except Exception as exc:
            logger.error("SAFE error for %s: %s", claim_id, exc)
            answer, reasoning = None, f"ERROR: {exc}"

        latency = time.monotonic() - t0
        predicted = validate_verdict(answer) if answer else "UNCERTAIN"

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{claim_id}.log"
            with open(log_path, "w") as handle:
                handle.write(f"=== claim_id: {claim_id} ===\n")
                handle.write(f"=== claim ===\n{claim}\n\n")
                handle.write(f"=== answer: {answer} ===\n")
                handle.write(f"=== search_backend: {self.search_backend} ===\n")
                handle.write(f"=== searches ({len(searches)}) ===\n")
                for search in searches:
                    handle.write(f"Q: {search.query}\nR: {search.result[:1200]}\n\n")
                handle.write(f"=== reasoning ===\n{reasoning}\n")

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=0.0,
            reasoning=reasoning[:1000] if reasoning else "",
            evidence=[search.query for search in searches],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=(
                usage["input_tokens"] / 1_000_000 * self._in_price
                + usage["output_tokens"] / 1_000_000 * self._out_price
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )

    def _safe_loop(
        self,
        claim: str,
        searches: list[_SearchResult],
        usage: dict[str, int],
    ) -> tuple[str | None, str]:
        for _ in range(self.max_steps):
            query = self._next_query(claim, searches, usage)
            if not query:
                break

            if _count_similar(query, [search.query for search in searches]) >= self.max_tolerance:
                logger.info("SAFE early stop: repeated query for claim %r: %s", claim, query)
                break

            result = self._search(query)
            if _count_similar(result, [search.result for search in searches]) >= self.max_tolerance:
                logger.info("SAFE early stop: repeated evidence for claim %r", claim)
                break

            searches.append(_SearchResult(query=query, result=result))

        return self._final_answer(claim, searches, usage)

    def _next_query(
        self,
        claim: str,
        searches: list[_SearchResult],
        usage: dict[str, int],
    ) -> str | None:
        knowledge = self._knowledge_block(searches)
        prompt = _NEXT_SEARCH_PROMPT.format(knowledge=knowledge, claim=claim)

        for _ in range(self.max_retries):
            response, in_tok, out_tok = self._llm.complete_text(system="", user=prompt)
            usage["input_tokens"] += in_tok
            usage["output_tokens"] += out_tok
            query = _extract_code_block(response)
            if query:
                return query

        logger.warning("SAFE could not parse next query for claim %r", claim)
        return None

    def _final_answer(
        self,
        claim: str,
        searches: list[_SearchResult],
        usage: dict[str, int],
    ) -> tuple[str | None, str]:
        knowledge = self._knowledge_block(searches)
        prompt = _FINAL_ANSWER_PROMPT.format(
            knowledge=knowledge,
            claim=claim,
            verdict_defs=_VERDICT_DEFS,
            verdict_names=" | ".join(_VERDICT_NAMES),
        )

        last_response = ""
        for _ in range(self.max_retries):
            response, in_tok, out_tok = self._llm.complete_text(system="", user=prompt)
            usage["input_tokens"] += in_tok
            usage["output_tokens"] += out_tok
            last_response = response
            answer = _extract_square_bracket_answer(response)
            if answer and validate_verdict(answer) in _VERDICT_NAMES:
                return validate_verdict(answer), response

        logger.warning("SAFE could not parse final answer for claim %r", claim)
        return None, last_response

    def _knowledge_block(self, searches: list[_SearchResult]) -> str:
        if not searches:
            return "N/A"
        return "\n\n".join(
            f"Search {index}: {search.query}\n{search.result}"
            for index, search in enumerate(searches, 1)
        )

    def _search(self, query: str) -> str:
        if self.search_backend == "s2":
            try:
                papers = self._s2.search(query, limit=self.num_search_results)
            except S2RateLimitError as exc:
                logger.error("S2 rate-limit exhausted during SAFE search: %s", exc)
                return "Semantic Scholar rate limit exhausted."
            return _format_s2_results(papers)

        return _ddg_search(query, k=self.num_search_results)