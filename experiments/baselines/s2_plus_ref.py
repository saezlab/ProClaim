"""
S2 retrieval + reference paper baseline — "in-the-wild with oracle anchor".

Combines the top-k abstracts retrieved from Semantic Scholar (same as
``s2_retrieval``) with the source paper abstract fetched from PubMed
(same as ``single_paper``).  The reference paper is always placed first
in the evidence list and is **not** deduplicated even if S2 happens to
return it, since its position signals importance.

This quantifies how much the source paper anchors the verdict when the
LLM also has access to broader retrieved literature.

Cost: 1 PubMed efetch + 1 S2 API call + 1 LLM call per claim.
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
from baselines.single_paper import _fetch_abstract, _format_single_passage
from baselines.s2_retrieval import _format_passages
from pkevolve.search.semantic_scholar import S2Client, S2RateLimitError

logger = logging.getLogger(__name__)


class S2PlusRef:
    """S2 top-k retrieval + reference paper abstract, single LLM call."""

    name = "s2_plus_ref"

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

        # ── 1. Fetch reference paper from PubMed ──────────────────────────
        pmid = (context or {}).get("pmid", "") if context else ""
        ref_paper = None
        if pmid:
            ref_paper = _fetch_abstract(str(pmid))

        # ── 2. Retrieve from Semantic Scholar ─────────────────────────────
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

        # Deduplicate: remove the reference paper from S2 results if present
        if ref_paper and papers:
            ref_pmid = str(ref_paper["pmid"])
            papers = [
                p for p in papers
                if str((p.get("externalIds") or {}).get("PubMed", "")) != ref_pmid
            ]

        # If we have neither reference nor S2 results, default to UNCERTAIN
        if not ref_paper and not papers:
            logger.warning(
                "No reference paper and no S2 results for claim %s — defaulting to UNCERTAIN",
                claim_id,
            )
            return BaselineResult(
                claim_id=claim_id,
                claim=claim,
                gold_label=normalize_label(gold_label),
                predicted_label="UNCERTAIN",
                reasoning="No reference paper and no S2 results available.",
                baseline_name=self.name,
                model=self.llm.model,
                latency_seconds=round(time.monotonic() - t0, 2),
            )

        # ── 3. Build combined evidence passage ────────────────────────────
        # Reference paper first (if available), then S2 papers
        evidence_parts: list[str] = []
        offset = 0
        if ref_paper and ref_paper.get("abstract"):
            evidence_parts.append(_format_single_passage(ref_paper))
            offset = 1

        if papers:
            # Re-number S2 papers starting after the reference paper
            s2_parts: list[str] = []
            for i, paper in enumerate(papers, offset + 1):
                title = paper.get("title", "Untitled")
                year = paper.get("year", "")
                abstract = (paper.get("abstract") or "").strip()
                if not abstract:
                    abstract = "(no abstract available)"
                ext_ids = paper.get("externalIds") or {}
                s2_pmid = ext_ids.get("PubMed", "")
                pmid_str = f"  PMID: {s2_pmid}" if s2_pmid else ""
                s2_parts.append(f"[{i}] {title} ({year}){pmid_str}\n{abstract}")
            evidence_parts.extend(s2_parts)

        evidence_text = "\n\n".join(evidence_parts)

        # ── 4. Single LLM call ───────────────────────────────────────────
        user_msg = VERIFICATION_USER_TEMPLATE.format(
            claim=claim, evidence=evidence_text
        )
        t_llm = time.monotonic()
        text, in_tok, out_tok = self.llm.complete(
            system=VERIFICATION_SYSTEM_PROMPT,
            user=user_msg,
        )
        self.tracker.record("llm_call", in_tok, out_tok, time.monotonic() - t_llm)

        # ── 5. Parse response ─────────────────────────────────────────────
        parsed = self.llm.parse_json(text)
        raw_label = parsed.get("label", "UNCERTAIN")
        predicted = normalize_label(raw_label)
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning", text[:500] if text else "")
        cited = parsed.get("evidence", [])
        if isinstance(cited, str):
            cited = [cited]

        summary = self.tracker.summary()

        # Build evidence list for output: reference PMID + S2 PMIDs
        all_evidence_pmids = []
        if ref_paper:
            all_evidence_pmids.append(str(ref_paper["pmid"]))
        for p in (papers or []):
            s2_pmid = str((p.get("externalIds") or {}).get("PubMed", ""))
            if s2_pmid:
                all_evidence_pmids.append(s2_pmid)

        # Write per-claim log
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{claim_id}.log"
            with open(log_path, "w") as lf:
                lf.write(f"=== claim_id: {claim_id} ===\n")
                lf.write(f"=== claim ===\n{claim}\n\n")
                if ref_paper:
                    lf.write(f"=== reference paper (PMID: {ref_paper['pmid']}) ===\n")
                    lf.write(f"  {ref_paper.get('title', '')}\n")
                    lf.write(f"  {ref_paper.get('abstract', '')}\n\n")
                lf.write(f"=== S2 papers ({len(papers or [])}) ===\n")
                for p in (papers or []):
                    title = p.get("title", "Untitled")
                    s2_pmid = (p.get("externalIds") or {}).get("PubMed", "")
                    lf.write(f"  [{s2_pmid or 'no-pmid'}] {title}\n")
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
            evidence=all_evidence_pmids,
            input_tokens=summary["input_tokens"],
            output_tokens=summary["output_tokens"],
            cost_usd=summary["cost_usd"],
            latency_seconds=summary["latency_seconds"],
            baseline_name=self.name,
            model=self.llm.model,
        )
