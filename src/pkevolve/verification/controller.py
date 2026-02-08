"""
Metacognitive Control Loop — Algorithm 1.

Backbone: fixed-iteration synchronous loop with heuristic stopping.
Wires together EvidenceState, VerificationTools, Classifier, and Compressor.

Extension points:
  - Subclaim decomposition (_decompose() via LLM)
  - Async execution (async def verify)
  - Parallel processing (asyncio.gather in _process)
  - Gap-directed retrieval (LLM query formulation from gaps)
  - INSUFFICIENT early-exit when no gaps and past halfway
  - Full report generation
  - Hyperparameter tuning (τ, T, B ranges)
"""

import logging
from typing import Tuple

from openai import OpenAI

from pkevolve.search.custom_pubmed import RelevancePubMedSearcher
from pkevolve.verification.classifier import SufficiencyClassifier
from pkevolve.verification.compressor import SufficiencyPreservingCompressor
from pkevolve.verification.evidence_state import EvidenceState
from pkevolve.verification.tools import VerificationTools

logger = logging.getLogger(__name__)


class MetacognitiveController:
    """
    Backbone implementation of Algorithm 1 — the metacognitive control loop.

    Synchronous, fixed-iteration. Uses the heuristic classifier and L1-only
    compressor. Re-queries with the original claim (gap-directed retrieval
    deferred).
    """

    def __init__(
        self,
        llm_client: OpenAI,
        model: str,
        classifier: SufficiencyClassifier,
        compressor: SufficiencyPreservingCompressor,
        pubmed_searcher: RelevancePubMedSearcher,
        threshold: float = 0.7,
        max_iterations: int = 3,
        context_budget: int = 50_000,
    ):
        """
        Args:
            llm_client: OpenAI-compatible client.
            model: Model identifier.
            classifier: Sufficiency classifier (heuristic or trained).
            compressor: Sufficiency-preserving compressor (L1 or full).
            pubmed_searcher: PubMed search client.
            threshold: Confidence threshold for early stopping.
            max_iterations: Maximum verification loop iterations.
            context_budget: Approximate token budget before compression triggers.
        """
        self.llm_client = llm_client
        self.model = model
        self.classifier = classifier
        self.compressor = compressor
        self.searcher = pubmed_searcher
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.context_budget = context_budget

    def verify(self, claim: str) -> Tuple[str, float, str]:
        """
        Run the metacognitive verification loop for a claim.

        Backbone entry point. Synchronous — no async needed yet.

        Args:
            claim: The scientific claim to verify.

        Returns:
            Tuple of (verdict, confidence, report_text) where:
              - verdict: SUFFICIENT_SUPPORT | SUFFICIENT_REFUTE | INSUFFICIENT
              - confidence: float in [0, 1]
              - report_text: human-readable evidence summary
        """
        logger.info("verify: starting for claim=%r", claim)

        # Phase 1: Init (no subclaim decomposition — use claim directly)
        state = EvidenceState(claim, subclaims=[claim])
        tools = VerificationTools(
            state, self.llm_client, self.model, self.searcher
        )

        # Initial retrieval and processing
        tools.retrieve_papers(claim, top_k=5)
        self._process(state, tools)

        # Phase 2: Iterative verification loop
        for t in range(1, self.max_iterations + 1):
            result = self.classifier(state)
            confidence = result["confidence"]
            label = result["label"]

            logger.info(
                "verify: iteration %d/%d — label=%s, confidence=%.2f",
                t, self.max_iterations, label, confidence,
            )

            if label != "INSUFFICIENT" and confidence >= self.threshold:
                logger.info("verify: sufficient at iteration %d", t)
                return label, confidence, self._report(state, label, confidence, t)

            # Re-query with original claim (gap-directed retrieval deferred)
            tools.retrieve_papers(claim, top_k=3)
            self._process(state, tools)

            # Trigger compression if context exceeds budget
            if state.token_count() > self.context_budget:
                logger.info(
                    "verify: compressing (tokens≈%d > budget=%d)",
                    state.token_count(), self.context_budget,
                )
                state = self.compressor.compress(state, claim)
                # Rebind tools to compressed state
                tools = VerificationTools(
                    state, self.llm_client, self.model, self.searcher
                )

        # Phase 3: Timeout — return best available result
        result = self.classifier(state)
        logger.info(
            "verify: timeout after %d iterations — label=%s, confidence=%.2f",
            self.max_iterations, result["label"], result["confidence"],
        )
        return (
            result["label"],
            result["confidence"],
            self._report(state, result["label"], result["confidence"], self.max_iterations),
        )

    def _process(self, state: EvidenceState, tools: VerificationTools) -> None:
        """
        Process new papers: summarize and extract facts.

        Backbone: sequential processing.
        Extension: parallel with asyncio.gather.
        """
        for pmid, paper in list(state.papers.items()):
            if paper.summary is None:
                tools.summarize_paper(pmid)
                tools.extract_facts(pmid)

    def _report(
        self,
        state: EvidenceState,
        label: str,
        confidence: float,
        iterations: int,
    ) -> str:
        """
        Generate a minimal evidence report.

        Backbone: claim, paper count, fact count, verdict, fact listing.
        Extension: subclaim analysis, conflicts, full evidence summary.
        """
        lines = [
            f"Claim: {state.claim}",
            f"Verdict: {label} (confidence: {confidence:.2f})",
            f"Iterations: {iterations}",
            f"Papers examined: {len(state.papers)}",
            f"Facts extracted: {len(state.facts)}",
            "",
            "--- Evidence ---",
        ]
        for fact in state.facts:
            lines.append(f"  [{fact.stance}] {fact.text} (PMID:{fact.source_pmid})")

        return "\n".join(lines)
