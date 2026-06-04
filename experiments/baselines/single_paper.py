"""
Single-paper baseline — "in-sandbox" claim verification.

For each claim, fetches the abstract of the **source paper** (the paper
that SIGNOR originally cited when adding the edge) and asks the LLM to
classify the claim based on that single abstract.

This mirrors the "in sandbox" setting where the system is given the
gold evidence document and only needs to perform reading comprehension.

Requirements:
  - The dataset CSV must contain a ``pmid`` column with the source PMID.
  - Claims without a valid PMID are classified as UNCERTAIN.

Cost: 1 PubMed efetch + 1 LLM call per claim.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import normalize_label
from baselines.shared.llm import LLMBackend
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.prompts import (
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_TEMPLATE,
)
from baselines.shared.single_shot import run_single_shot_verdict
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)

_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _fetch_abstract(pmid: str) -> dict | None:
    """Fetch title and abstract for a PMID via PubMed E-utilities.

    Returns a dict with keys ``title``, ``abstract``, ``year``, ``pmid``
    or ``None`` if the fetch fails.
    Implements exponential backoff and retry for HTTP 429 errors.
    """
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    max_retries = 5
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(_EFETCH_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait_time = backoff * (2 ** attempt)
                logger.warning(
                    "PubMed efetch rate-limited (429) for PMID %s, retrying in %.1fs (attempt %d/%d)",
                    pmid, wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("PubMed efetch failed for PMID %s: %s", pmid, exc)
            return None
        break
    else:
        logger.warning("PubMed efetch failed for PMID %s after %d retries (429)", pmid, max_retries)
        return None

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        logger.warning("Malformed XML from PubMed for PMID %s", pmid)
        return None

    article = root.find(".//PubmedArticle")
    if article is None:
        return None

    title_el = article.find(".//ArticleTitle")
    title = title_el.text if title_el is not None and title_el.text else "Untitled"

    # Abstract may have multiple sections (structured abstract)
    abstract_parts: list[str] = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.get("Label", "")
        text = "".join(abstract_text.itertext()).strip()
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts) if abstract_parts else ""

    year_el = article.find(".//PubDate/Year")
    year = year_el.text if year_el is not None else ""

    return {"title": title, "abstract": abstract, "year": year, "pmid": pmid}


def _format_single_passage(paper: dict) -> str:
    """Format a single paper dict into an evidence passage."""
    title = paper.get("title", "Untitled")
    year = paper.get("year", "")
    abstract = paper.get("abstract", "").strip() or "(no abstract available)"
    pmid = paper.get("pmid", "")
    pmid_str = f"  PMID: {pmid}" if pmid else ""
    return f"[1] {title} ({year}){pmid_str}\n{abstract}"


class SinglePaper:
    """Classify claims using only the source paper's abstract."""

    name = "single_paper"

    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm
        self.tracker = CostTracker(model=llm.model)
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

        # Extract source PMID from context (forwarded by the evaluation harness)
        pmid = (context or {}).get("pmid", "") if context else ""

        if not pmid:
            logger.warning(
                "No source PMID for claim %s — defaulting to UNCERTAIN", claim_id
            )
            return BaselineResult(
                claim_id=claim_id,
                claim=claim,
                gold_label=normalize_label(gold_label),
                predicted_label="UNCERTAIN",
                reasoning="No source PMID available for this claim.",
                baseline_name=self.name,
                model=self.llm.model,
                latency_seconds=round(time.monotonic() - t0, 2),
            )

        # 1. Fetch abstract from PubMed
        paper = _fetch_abstract(str(pmid))

        if paper is None or not paper.get("abstract"):
            logger.warning(
                "Could not fetch abstract for PMID %s (claim %s) — defaulting to UNCERTAIN",
                pmid, claim_id,
            )
            return BaselineResult(
                claim_id=claim_id,
                claim=claim,
                gold_label=normalize_label(gold_label),
                predicted_label="UNCERTAIN",
                reasoning=f"Could not retrieve abstract for source PMID {pmid}.",
                baseline_name=self.name,
                model=self.llm.model,
                latency_seconds=round(time.monotonic() - t0, 2),
            )

        # 2. Format as a single evidence passage
        evidence_text = _format_single_passage(paper)

        # 3. Single LLM call with the same retrieval prompt as S2 baseline
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
        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}", ""),
                ("claim", claim),
                ("system prompt", VERIFICATION_SYSTEM_PROMPT),
                ("user prompt", user_msg),
                (f"source PMID: {pmid}", ""),
                (f"title: {paper.get('title', '')}", ""),
                ("abstract", paper.get("abstract", "")),
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
