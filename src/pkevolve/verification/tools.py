"""
VerificationTools — state-mutating operations for the verification loop.

Backbone: 3 essential tools (retrieve, summarize, extract_facts) + 5 stubs.
Extension points:
  - LLM-powered gap retrieval, coverage, conflict detection, synthesis, gap analysis
  - Async versions of all tools
  - Full-text retrieval (PDF download, PMC)
  - Rate limiting (reuse WebSearchAssistant._rate_limit() pattern)
"""

import json
import logging
from typing import List, Optional

from openai import OpenAI

from pkevolve.search.custom_pubmed import RelevancePubMedSearcher
from pkevolve.verification.data_models import Fact, PaperRecord
from pkevolve.verification.evidence_state import EvidenceState

logger = logging.getLogger(__name__)


class VerificationTools:
    """
    Tools that read/write to an EvidenceState.

    Backbone provides 3 real tools + 5 stubs.  All LLM calls follow the
    project's OpenAI-compatible client pattern.
    """

    def __init__(
        self,
        state: EvidenceState,
        llm_client: OpenAI,
        model: str,
        pubmed_searcher: RelevancePubMedSearcher,
    ):
        self.state = state
        self.client = llm_client
        self.model = model
        self.searcher = pubmed_searcher

    # ------------------------------------------------------------------
    # Essential tools (implemented)
    # ------------------------------------------------------------------

    def retrieve_papers(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search PubMed for papers relevant to *query* and add them to state.

        Returns list of PMIDs added (new papers only — skips duplicates).
        """
        papers = self.searcher.search(query, max_results=top_k)
        added_pmids: List[str] = []
        for paper in papers:
            pmid = paper.paper_id
            if pmid in self.state.papers:
                continue
            record = PaperRecord(
                pmid=pmid,
                title=paper.title or "",
                abstract=paper.abstract or "",
            )
            self.state.add_paper(record)
            added_pmids.append(pmid)
        logger.info(
            "retrieve_papers: query=%r, found=%d, added=%d",
            query, len(papers), len(added_pmids),
        )
        return added_pmids

    def summarize_paper(self, pmid: str) -> Optional[str]:
        """
        Summarize a paper's text via LLM and store the summary in state.

        Returns the summary string, or None if the paper is not in state.
        """
        paper = self.state.papers.get(pmid)
        if paper is None:
            logger.warning("summarize_paper: PMID %s not in state", pmid)
            return None

        text = paper.text_for_summarization()
        if not text:
            logger.warning("summarize_paper: no text for PMID %s", pmid)
            return None

        prompt = (
            f"Summarize the following paper in 3–5 sentences, focusing on the "
            f"key findings and their relevance to the claim: '{self.state.claim}'.\n\n"
            f"Title: {paper.title}\n\n"
            f"Text:\n{text[:8000]}"  # Truncate to avoid hitting context limits
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a scientific paper summarizer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            summary = response.choices[0].message.content.strip()
            paper.summary = summary
            logger.info("summarize_paper: PMID %s summarized (%d chars)", pmid, len(summary))
            return summary
        except Exception as e:
            logger.error("summarize_paper: LLM call failed for PMID %s: %s", pmid, e)
            return None

    def extract_facts(self, pmid: str) -> List[Fact]:
        """
        Extract stance-labeled facts from a paper via LLM.

        Facts are added to state and also returned.
        Returns empty list if the paper is not in state or extraction fails.
        """
        paper = self.state.papers.get(pmid)
        if paper is None:
            logger.warning("extract_facts: PMID %s not in state", pmid)
            return []

        text = paper.summary if paper.summary else paper.abstract
        if not text:
            logger.warning("extract_facts: no text for PMID %s", pmid)
            return []

        prompt = (
            f"Given the claim: '{self.state.claim}'\n\n"
            f"And the following paper text:\n{text}\n\n"
            f"Extract key facts that are relevant to the claim. "
            f"For each fact, provide:\n"
            f"1. A concise statement of the fact\n"
            f"2. Its stance toward the claim: SUPPORT, REFUTE, or NEUTRAL\n\n"
            f"Respond in JSON format as a list of objects with keys "
            f'"text" and "stance". Example:\n'
            f'[{{"text": "p53 binds to the BAX promoter", "stance": "SUPPORT"}}]\n\n'
            f"Return ONLY the JSON array, no other text."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a scientific fact extractor. "
                            "Return ONLY a JSON array of facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            facts_data = json.loads(raw)
            extracted: List[Fact] = []
            for item in facts_data:
                stance = item.get("stance", "NEUTRAL").upper()
                if stance not in ("SUPPORT", "REFUTE", "NEUTRAL"):
                    stance = "NEUTRAL"
                fact = Fact(
                    text=item.get("text", ""),
                    stance=stance,
                    source_pmid=pmid,
                )
                self.state.add_fact(fact)
                extracted.append(fact)
            logger.info(
                "extract_facts: PMID %s → %d facts extracted", pmid, len(extracted)
            )
            return extracted
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("extract_facts: failed to parse LLM output for PMID %s: %s", pmid, e)
            return []
        except Exception as e:
            logger.error("extract_facts: LLM call failed for PMID %s: %s", pmid, e)
            return []

    # ------------------------------------------------------------------
    # Stub tools (deferred to extensions)
    # ------------------------------------------------------------------

    def retrieve_for_gap(self, gap: str, top_k: int = 3) -> List[str]:
        """
        Stub: retrieve papers for a specific evidence gap.

        Backbone: pass-through to retrieve_papers (no LLM query reformulation).
        Extension: LLM-formulated gap queries.
        """
        return self.retrieve_papers(gap, top_k=top_k)

    def compute_coverage(self, subclaim: str) -> float:
        """
        Stub: compute evidence coverage for a subclaim.

        Backbone: returns 0.5 (neutral).
        Extension: LLM-based coverage scoring.
        """
        return 0.5

    def detect_conflicts(self) -> List[dict]:
        """
        Stub: detect conflicting facts in the evidence.

        Backbone: no-op, returns empty list.
        Extension: LLM-based pairwise conflict detection.
        """
        return []

    def synthesize_subclaim(self, subclaim: str) -> str:
        """
        Stub: synthesize evidence for a subclaim.

        Backbone: concatenates facts matching the subclaim.
        Extension: LLM-based synthesis.
        """
        relevant = [
            f for f in self.state.facts
            if subclaim in f.relevant_subclaims or not f.relevant_subclaims
        ]
        if not relevant:
            return f"No evidence found for: {subclaim}"
        lines = [f"[{f.stance}] {f.text}" for f in relevant]
        return "\n".join(lines)

    def analyze_gaps(self) -> List[str]:
        """
        Stub: identify evidence gaps.

        Backbone: returns empty list.
        Extension: LLM-based gap analysis.
        """
        return []
