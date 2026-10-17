"""FIRE baseline via the upstream FIRE repository implementation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.llm import LLMBackend
from baselines.shared.label_utils import (
    normalize_label,
)
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.prompt_cache import cached_cost
from baselines.shared.search_utils import retrieved_documents_sink
from baselines.shared.upstream_adapters import (
    FIRE_REPO_ROOT,
    FireModelAdapter,
    do_search,
    name_search_engine,
    import_upstream_module,
)
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)


class FIREBaseline:
    """Verify claims via the original FIRE verification loop."""

    name = "fire"

    def __init__(
        self,
        llm: LLMBackend,
        *,
        max_steps: int = 25,
        max_retries: int = 10,
        num_search_results: int = 5,
        search_backend: str = "web",
    ) -> None:
        from baselines.shared.upstream_adapters import set_search_backend

        self._llm = llm
        self.model = self._llm.model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.num_search_results = num_search_results
        set_search_backend(search_backend)
        self.search_backend = search_backend
        self.log_dir: Path | None = None
        self._verify_atomic_claim = self._load_fire_verifier()

        in_price, out_price = CostTracker.pricing_for(self.model)
        self._in_price = in_price
        self._out_price = out_price

    def verify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        context: dict | None = None,
    ) -> BaselineResult:
        t0 = time.monotonic()
        adapter = FireModelAdapter(self._llm)

        retrieved_documents: list[dict] = []
        token = retrieved_documents_sink.set(retrieved_documents)
        try:
            final_answer, search_dicts, total_usage = self._verify_atomic_claim(
                atomic_claim=claim,
                rater=adapter,
                max_steps=self.max_steps,
                max_retries=self.max_retries,
            )
        except Exception as exc:
            logger.error("FIRE error for %s: %s", claim_id, exc)
            final_answer = None
            search_dicts = {"google_searches": []}
            total_usage = {"input_tokens": 0, "output_tokens": 0}
            reasoning = f"ERROR: {exc}"
        else:
            reasoning = final_answer.response if final_answer else ""
        finally:
            retrieved_documents_sink.reset(token)

        latency = time.monotonic() - t0
        prompts = adapter.prompts
        prompt_block = "\n\n".join(
            f"--- prompt {index} ---\n{prompt}"
            for index, prompt in enumerate(prompts, 1)
        )
        searches = search_dicts.get("google_searches", [])
        search_block = "\n\n".join(
            f"Q: {search.get('query', '')}\nR: {search.get('result', '')[:1200]}"
            for search in searches
        )
        retrieved_block = "\n".join(
            "  [{}] source={} query={!r} ".format(i, d.get("source"), d.get("query"))
            + " ".join(f"{k}={v!r}" for k, v in d.items() if k not in ("query", "source"))
            for i, d in enumerate(retrieved_documents, 1)
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
                (f"prompt cache: read {adapter.cache_read} / write {adapter.cache_write} input tokens", ""),
                (f"searches ({len(searches)})", search_block),
                (f"retrieved documents ({len(retrieved_documents)})", retrieved_block),
                ("reasoning", reasoning),
            ],
        )

        if final_answer is None:
            predicted = "UNCERTAIN"
        else:
            predicted = normalize_label(str(final_answer.answer).strip())

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=0.0,
            reasoning=reasoning[:1000] if reasoning else "",
            evidence=[search.get("query", "") for search in searches],
            input_tokens=total_usage["input_tokens"],
            output_tokens=total_usage["output_tokens"],
            cost_usd=cached_cost(
                total_usage["input_tokens"], total_usage["output_tokens"],
                adapter.cache_read, adapter.cache_write, self._in_price, self._out_price,
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )

    # ── Core FIRE loop ───────────────────────────────────────────────

    def _load_fire_verifier(self):
        verify_atomic_claim = import_upstream_module(
            FIRE_REPO_ROOT,
            "eval.fire.verify_atomic_claim",
        )

        def _patched_sentence_similarity(
            new_sent: str,
            sentences: list[str],
            threshold: float = 0.9,
        ) -> int:
            if not sentences:
                return 0

            embedding_device = verify_atomic_claim.torch.device(verify_atomic_claim.device)
            single_embedding = verify_atomic_claim.sbert_model.encode(
                new_sent,
                convert_to_tensor=True,
            ).to(embedding_device)
            list_embeddings = verify_atomic_claim.sbert_model.encode(
                sentences,
                convert_to_tensor=True,
            ).to(embedding_device)
            similarities = verify_atomic_claim.util.cos_sim(single_embedding, list_embeddings)
            return sum(
                1
                for index in range(len(sentences))
                if similarities[0][index].item() > threshold
            )

        def _patched_search(
            search_query: str,
            search_type: str = "web",
            num_searches: int = 5,
            serper_api_key: str = "",
            search_postamble: str = "",
        ) -> str:
            del search_type, serper_api_key, search_postamble
            return do_search(search_query, k=num_searches)

        verify_atomic_claim._FINAL_ANSWER_OR_NEXT_SEARCH_FORMAT = name_search_engine(
            verify_atomic_claim._FINAL_ANSWER_OR_NEXT_SEARCH_FORMAT, self.search_backend
        )
        verify_atomic_claim.get_sentence_similarity = _patched_sentence_similarity
        verify_atomic_claim.call_search = _patched_search
        return verify_atomic_claim.verify_atomic_claim
