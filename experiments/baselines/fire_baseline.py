"""
FIRE baseline — iterative retrieval-augmented fact-checking.

Reimplements the core FIRE (Xie et al., NAACL 2025) loop using the shared
LLMBackend (litellm) for LLM calls and Google search for web retrieval.  Runs
in-process — no subprocess or separate venv needed.

FIRE iteratively decides whether to (a) issue a web search query or (b) render
a final verdict, based on accumulated search results.

FIRE outputs use the same canonical taxonomy:
  - Support   → SUPPORT
  - Refute    → REFUTE
  - Uncertain → UNCERTAIN

Search backends:
    ``search_backend="web"`` (default):
        DuckDuckGo search via ``ddgs`` (free, no API key).
    ``search_backend="s2"``:
        Semantic Scholar relevance search (``S2_API_KEY`` optional but recommended).

Prerequisites:
  - An LLM API key recognised by litellm (e.g. ``ANTHROPIC_API_KEY``).
  - ``pip install ddgs`` (already in project deps).

Cost: 1–N LLM calls per claim (N ≤ max_steps × max_retries) + searches.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.llm import LLMBackend
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.label_utils import (
    normalize_label,
    validate_verdict,
    verdict_defs_block,
    verdict_names,
    verdict_or_str,
)
from baselines.shared.search_utils import (
    ddg_text_search_with_retry,
    format_ddg_body_results,
    format_s2_basic_results,
)
from baselines.shared.verdict import BaselineResult
from pkevolve.search.semantic_scholar import S2Client, S2RateLimitError

logger = logging.getLogger(__name__)

# ── Verdict labels (loaded from shared label_utils) ──────────────────

_VERDICT_OPTIONS = verdict_or_str()
_VERDICT_DEFS = verdict_defs_block()

# ── FIRE prompts (from Xie et al.) ──────────────────────────────────

_SYS_PROMPT = "You are a fact-checking agent responsible for verifying the accuracy of claims."

_FINAL_ANSWER_OR_NEXT_SEARCH = f"""\
Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Before presenting your conclusion, think through the process step-by-step. \
Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
4. If the KNOWLEDGE allows you to confidently make a decision, output the final \
answer as a JSON object in the following format:
   {{{{
     "final_answer": {_VERDICT_OPTIONS}
   }}}}
{_VERDICT_DEFS}
5. If the KNOWLEDGE is insufficient to make a judgment, issue ONE Google Search \
query that could provide additional evidence. Output the search query in JSON \
format, as follows:
   {{{{
     "search_query": "Your Google search query here"
   }}}}
6. The query should aim to obtain new information not already present in the \
KNOWLEDGE, specifically helpful for verifying the STATEMENT's accuracy.

KNOWLEDGE:
{{knowledge}}

STATEMENT:
{{statement}}"""

_MUST_HAVE_FINAL_ANSWER = f"""\
Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Before presenting your final answer, think step-by-step and show your reasoning. \
Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
4. Your final answer should be {_VERDICT_OPTIONS}.
{_VERDICT_DEFS}
5. Format your final answer as a JSON object in the following structure:
   {{{{
     "final_answer": {_VERDICT_OPTIONS}
   }}}}

KNOWLEDGE:
{{knowledge}}

STATEMENT:
{{statement}}"""

# ── Helpers ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from model output."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _token_overlap(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _count_similar(target: str, history: list[str], threshold: float = 0.9) -> int:
    """Count how many strings in *history* are similar to *target*."""
    return sum(1 for h in history if _token_overlap(target, h) >= threshold)


@dataclass
class _SearchResult:
    query: str
    result: str


# ── Baseline class ───────────────────────────────────────────────────


class FIREBaseline:
    """Verify claims via the FIRE iterative retrieval loop (in-process, LLMBackend).

    Parameters
    ----------
    llm:
        Shared ``LLMBackend`` instance.
    max_steps:
        Maximum number of iterative search steps.
    max_retries:
        Maximum retries per step when the LLM returns unparseable output.
    max_tolerance:
        Maximum number of similar queries/results before early stopping
        (FIRE default is 2).
    num_search_results:
        Number of search results per query.
    """

    name = "fire"

    def __init__(
        self,
        llm: LLMBackend,
        *,
        max_steps: int = 5,
        max_retries: int = 10,
        max_tolerance: int = 2,
        num_search_results: int = 5,
        search_backend: str = "web",
    ) -> None:
        if search_backend not in ("web", "s2"):
            raise ValueError(f"search_backend must be 'web' or 's2', got {search_backend!r}")

        self._llm = llm
        self.model = self._llm.model
        self.temperature = self._llm.temperature
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.max_tolerance = max_tolerance
        self.num_search_results = num_search_results
        self.search_backend = search_backend
        self.log_dir: Path | None = None

        self._search_backend = "s2" if search_backend == "s2" else "ddg"
        self._s2 = S2Client() if search_backend == "s2" else None
        logger.info("FIRE search backend: %s", self._search_backend)

        # Cost estimation — reuse CostTracker pricing table
        in_price, out_price = CostTracker.pricing_for(self.model)
        self._in_price = in_price    # USD per 1M input tokens
        self._out_price = out_price  # USD per 1M output tokens

    # ── Public interface ─────────────────────────────────────────────

    def verify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        context: dict | None = None,
    ) -> BaselineResult:
        t0 = time.monotonic()
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        searches: list[_SearchResult] = []
        prompts: list[str] = []

        try:
            answer, reasoning = self._fire_loop(claim, searches, total_usage, prompts)
        except Exception as exc:
            logger.error("FIRE error for %s: %s", claim_id, exc)
            answer, reasoning = None, f"ERROR: {exc}"

        latency = time.monotonic() - t0

        # Write per-claim log
        prompt_block = "\n\n".join(
            f"--- prompt {index} ---\n{prompt}"
            for index, prompt in enumerate(prompts, 1)
        )
        search_block = "\n\n".join(
            f"  Q: {search.query}\n  R: {search.result[:200]}"
            for search in searches
        )
        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}", ""),
                ("claim", claim),
                ("system prompt", _SYS_PROMPT),
                (f"prompts ({len(prompts)})", prompt_block),
                (f"answer: {answer}", ""),
                (f"search_backend: {self.search_backend}", ""),
                (f"searches ({len(searches)})", search_block),
                ("reasoning", reasoning),
            ],
        )

        # Map FIRE's answer → canonical labels
        if answer is not None:
            predicted = validate_verdict(answer)
        else:
            predicted = "UNCERTAIN"

        evidence = [s.query for s in searches]

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=0.0,
            reasoning=reasoning[:1000] if reasoning else "",
            evidence=evidence,
            input_tokens=total_usage["input_tokens"],
            output_tokens=total_usage["output_tokens"],
            cost_usd=(
                total_usage["input_tokens"] / 1_000_000 * self._in_price
                + total_usage["output_tokens"] / 1_000_000 * self._out_price
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )

    # ── Core FIRE loop ───────────────────────────────────────────────

    def _fire_loop(
        self,
        claim: str,
        searches: list[_SearchResult],
        usage: dict[str, int],
        prompts: list[str],
    ) -> tuple[str | None, str]:
        """Run the iterative search-or-answer loop.  Returns (answer, reasoning)."""
        for _ in range(self.max_steps):
            result = self._step(claim, searches, usage, prompts)
            if result is None:
                break
            if isinstance(result, str):
                # "_Early_Stop" — force final answer
                break
            if isinstance(result, tuple):
                return result
            # _SearchResult — already appended inside _step

        return self._force_final(claim, searches, usage, prompts)

    def _step(
        self,
        claim: str,
        searches: list[_SearchResult],
        usage: dict[str, int],
        prompts: list[str],
    ) -> tuple[str, str] | _SearchResult | str | None:
        """One FIRE iteration: ask the LLM to decide or search."""
        knowledge = "\n".join(s.result for s in searches) or "N/A"
        prompt = _FINAL_ANSWER_OR_NEXT_SEARCH.format(
            knowledge=knowledge, statement=claim,
        ).strip()
        prompts.append(prompt)

        for _ in range(self.max_retries):
            text, in_tok, out_tok = self._llm_call(prompt)
            usage["input_tokens"] += in_tok
            usage["output_tokens"] += out_tok

            parsed = _extract_json(text)
            if parsed is None:
                continue

            if "final_answer" in parsed:
                return (parsed["final_answer"], text)

            if "search_query" in parsed:
                q = parsed["search_query"]

                # Tolerance check: early-stop if queries or results are repetitive
                query_history = [s.query for s in searches]
                result_history = [s.result for s in searches]
                tol = self.max_tolerance

                if (len(query_history) >= tol - 1
                        and _count_similar(q, query_history[-(tol - 1):]) >= tol - 1):
                    logger.info("FIRE early stop: repetitive queries")
                    return "_Early_Stop"

                if (len(result_history) >= tol
                        and _count_similar(result_history[-1],
                                           result_history[-tol:-1]) >= tol - 1):
                    logger.info("FIRE early stop: repetitive search results")
                    return "_Early_Stop"

                snippet = self._search(q)
                sr = _SearchResult(query=q, result=snippet)
                searches.append(sr)
                return sr

        return None

    def _force_final(
        self,
        claim: str,
        searches: list[_SearchResult],
        usage: dict[str, int],
        prompts: list[str],
    ) -> tuple[str | None, str]:
        """Force the LLM to produce a final verdict."""
        knowledge = "\n".join(s.result for s in searches) or "N/A"
        prompt = _MUST_HAVE_FINAL_ANSWER.format(
            knowledge=knowledge, statement=claim,
        ).strip()
        prompts.append(prompt)

        for _ in range(self.max_retries):
            text, in_tok, out_tok = self._llm_call(prompt)
            usage["input_tokens"] += in_tok
            usage["output_tokens"] += out_tok
            parsed = _extract_json(text)
            if parsed and "final_answer" in parsed:
                fa = parsed["final_answer"]
                if validate_verdict(fa) in verdict_names():
                    return (fa, text)
        return (None, "Failed to extract final answer from FIRE loop")

    # ── LLM call via shared LLMBackend ─────────────────────────────

    def _llm_call(self, user_prompt: str) -> tuple[str, int, int]:
        """Single LLM completion.  Returns (text, input_tokens, output_tokens)."""
        return self._llm.complete_text(system=_SYS_PROMPT, user=user_prompt)

    def _search(self, query: str) -> str:
        """Dispatch FIRE retrieval to the configured search backend."""
        if self.search_backend == "s2":
            try:
                papers = self._s2.search(query, limit=self.num_search_results)
            except S2RateLimitError as exc:
                logger.error("S2 rate-limit exhausted during FIRE search: %s", exc)
                return "Semantic Scholar rate limit exhausted."
            return format_s2_basic_results(
                papers,
                empty_message="No relevant papers found on Semantic Scholar",
            )

        return format_ddg_body_results(
            ddg_text_search_with_retry(query, k=self.num_search_results),
            empty_message="No good Google Search result was found",
        )
