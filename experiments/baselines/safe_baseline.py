"""SAFE baseline via the upstream SAFE repository implementation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import (
    normalize_label,
    verdict_defs_block,
)
from baselines.shared.llm import LLMBackend
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.upstream_adapters import (
    SAFE_REPO_ROOT,
    SafeModelAdapter,
    do_search,
    import_upstream_module,
)
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)

class SAFEBaseline:
    """Verify claims via the original SAFE atomic fact rating loop."""

    name = "safe"

    def __init__(
        self,
        llm: LLMBackend,
        *,
        max_steps: int = 5,
        max_retries: int = 10,
        num_search_results: int = 3,
    ) -> None:
        self._llm = llm
        self.model = llm.model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.num_search_results = num_search_results
        self.search_backend = "web"
        self.log_dir: Path | None = None
        self._check_atomic_fact = self._load_safe_checker()

        in_price, out_price = CostTracker.pricing_for(self.model)
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
        adapter = SafeModelAdapter(self._llm)

        try:
            final_answer, search_dicts = self._check_atomic_fact(
                atomic_fact=claim,
                rater=adapter,
                max_steps=self.max_steps,
                max_retries=self.max_retries,
            )
        except Exception as exc:
            logger.error("SAFE error for %s: %s", claim_id, exc)
            final_answer = None
            search_dicts = {"google_searches": []}
            reasoning = f"ERROR: {exc}"
        else:
            reasoning = final_answer.response if final_answer else ""

        latency = time.monotonic() - t0
        prompts = adapter.prompts
        searches = search_dicts.get("google_searches", [])
        if final_answer is None:
            predicted = "UNCERTAIN"
        else:
            predicted = normalize_label(final_answer.answer)

        prompt_block = "\n\n".join(
            f"--- prompt {index} ---\n{prompt}"
            for index, prompt in enumerate(prompts, 1)
        )
        search_block = "\n\n".join(
            f"Q: {search.get('query', '')}\nR: {search.get('result', '')[:1200]}"
            for search in searches
        )
        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}", ""),
                ("claim", claim),
                (f"prompts ({len(prompts)})", prompt_block),
                (f"answer: {final_answer.answer if final_answer else None}", ""),
                (f"search_backend: {self.search_backend}", ""),
                (f"searches ({len(searches)})", search_block),
                ("reasoning", reasoning),
            ],
        )

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=0.0,
            reasoning=reasoning[:1000] if reasoning else "",
            evidence=[search.get("query", "") for search in searches],
            input_tokens=adapter.input_tokens,
            output_tokens=adapter.output_tokens,
            cost_usd=(
                adapter.input_tokens / 1_000_000 * self._in_price
                + adapter.output_tokens / 1_000_000 * self._out_price
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )

    def _load_safe_checker(self):
        rate_atomic_fact = import_upstream_module(
            SAFE_REPO_ROOT,
            "eval.safe.rate_atomic_fact",
        )

        # Keep SAFE's upstream loop, but reuse the repo's shared verdict
        # definitions so its prompt taxonomy matches the other baselines.
        rate_atomic_fact._VERDICT_DEFS = verdict_defs_block(indent="  ")
        rate_atomic_fact._FINAL_ANSWER_FORMAT = f"""\\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be one of \"{rate_atomic_fact.SUPPORTED_LABEL}\", \
\"{rate_atomic_fact.NOT_SUPPORTED_LABEL}\", or \"{rate_atomic_fact.UNCERTAIN_LABEL}\". Wrap your final answer in square brackets.
{rate_atomic_fact._VERDICT_DEFS}

KNOWLEDGE:
{rate_atomic_fact._KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{rate_atomic_fact._STATEMENT_PLACEHOLDER}
"""

        def _patched_search(
            search_query: str,
            search_type: str = "web",
            num_searches: int = 3,
            serper_api_key: str = "",
            search_postamble: str = "",
        ) -> str:
            del search_type, serper_api_key, search_postamble
            return do_search(search_query, k=num_searches)

        rate_atomic_fact.call_search = _patched_search
        return rate_atomic_fact.check_atomic_fact