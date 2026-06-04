"""
Evaluation harness for evidence programming baselines.

Runs any baseline on a list of claims and computes metrics:
  - Accuracy
  - Macro F1 (3-class: SUPPORT / REFUTE / UNCERTAIN)
  - Binary F1 (SUPPORT vs. REFUTE, excluding UNCERTAIN rows)
  - Per-class precision / recall / F1

Usage::

    harness = EvaluationHarness(baseline, dataset_name="SIGNOR")
    results = harness.run(claims)   # list of dicts with "claim_id", "claim", "gold_label"
    metrics = harness.metrics(results)
    harness.save(results, Path("results/baselines/random_signor.jsonl"))
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from .label_utils import normalize_label
from .verdict import BaselineResult

logger = logging.getLogger(__name__)


class BaselineProtocol(Protocol):
    """Interface every baseline must satisfy."""

    name: str

    def verify(self, claim_id: str, claim: str, gold_label: str) -> BaselineResult:
        ...


class EvaluationHarness:
    def __init__(self, baseline: BaselineProtocol, dataset_name: str = "") -> None:
        self.baseline = baseline
        self.dataset_name = dataset_name

    # ------------------------------------------------------------------
    # Running

    def run(
        self,
        claims: list[dict[str, Any]],
        resume_path: Path | None = None,
    ) -> list[BaselineResult]:
        """Evaluate baseline on a list of claim dicts.

        Each dict must have ``claim_id``, ``claim``, and ``gold_label`` keys.
        ``gold_label`` is normalized to the canonical taxonomy automatically.

        If the baseline's ``verify()`` accepts a ``context`` keyword argument,
        the full claim dict is forwarded as ``context`` so that richer baselines
        (e.g. OpenScholar) can use pre-retrieved evidence stored in the dict.

        Parameters
        ----------
        resume_path:
            Optional path to a JSONL results file.  When provided, already-
            completed claims (matched by ``claim_id``) are loaded from the file
            and skipped; each new result is streamed to the file immediately
            after it is computed so that a partial run can be resumed after any
            interruption.
        """
        # Load already-completed results for resume support
        completed: dict[str, BaselineResult] = {}
        if resume_path is not None and resume_path.exists():
            for r in self.load(resume_path):
                completed[r.claim_id] = r
            if completed:
                logger.info(
                    "Resume: loaded %d already-completed results from %s",
                    len(completed), resume_path,
                )

        # Open stream file for appending new results
        stream_file = None
        if resume_path is not None:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            stream_file = open(resume_path, "a")

        result_map: dict[str, BaselineResult] = dict(completed)
        n = len(claims)
        _verify_accepts_context = "context" in inspect.signature(
            self.baseline.verify
        ).parameters

        try:
            for i, item in enumerate(claims):
                claim_id = str(item["claim_id"])
                if claim_id in completed:
                    logger.info("[%d/%d] %s  SKIPPED (already done)", i + 1, n, claim_id)
                    continue
                claim = item["claim"]
                gold_raw = item.get("gold_label", "UNCERTAIN")
                gold = normalize_label(gold_raw)

                logger.info("[%d/%d] %s  gold=%s", i + 1, n, claim_id, gold)
                try:
                    if _verify_accepts_context:
                        result = self.baseline.verify(claim_id, claim, gold, context=item)
                    else:
                        result = self.baseline.verify(claim_id, claim, gold)
                except Exception as exc:
                    logger.error("Baseline crashed on %s: %s", claim_id, exc)
                    result = BaselineResult(
                        claim_id=claim_id,
                        claim=claim,
                        gold_label=gold,
                        predicted_label="UNCERTAIN",
                        reasoning=f"ERROR: {exc}",
                        baseline_name=self.baseline.name,
                        dataset=self.dataset_name,
                    )
                result_map[claim_id] = result
                if stream_file is not None:
                    stream_file.write(result.model_dump_json() + "\n")
                    stream_file.flush()
        finally:
            if stream_file is not None:
                stream_file.close()

        # Return results in original claims order
        return [
            result_map[str(item["claim_id"])]
            for item in claims
            if str(item["claim_id"]) in result_map
        ]

    # ------------------------------------------------------------------
    # Metrics

    @staticmethod
    def metrics(results: list[BaselineResult]) -> dict[str, Any]:
        """Compute accuracy, macro-F1, macro/weighted FPR, and macro/weighted FNR from a result list."""
        from collections import defaultdict

        labels = ["SUPPORT", "REFUTE", "UNCERTAIN"]

        # Count TP / FP / FN / TN per class
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)
        tn: dict[str, int] = defaultdict(int)

        correct = 0
        total = len(results)

        for r in results:
            pred = normalize_label(r.predicted_label)
            gold = normalize_label(r.gold_label)
            if pred == gold:
                correct += 1
                tp[gold] += 1
            else:
                fp[pred] += 1
                fn[gold] += 1

        # TN_k = N - TP_k - FP_k - FN_k
        for label in labels:
            tn[label] = total - tp[label] - fp[label] - fn[label]

        def prf(label: str) -> tuple[float, float, float]:
            prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return prec, rec, f1

        per_class: dict[str, dict] = {}
        f1_scores: list[float] = []
        for label in labels:
            p, r, f = prf(label)
            fpr = fp[label] / (fp[label] + tn[label]) if (fp[label] + tn[label]) > 0 else 0.0
            fnr = fn[label] / (fn[label] + tp[label]) if (fn[label] + tp[label]) > 0 else 0.0
            per_class[label] = {
                "precision": round(p, 2),
                "recall": round(r, 2),
                "f1": round(f, 2),
                "fpr": round(fpr, 2),
                "fnr": round(fnr, 2),
                "tp": tp[label],
                "fp": fp[label],
                "fn": fn[label],
                "tn": tn[label],
            }
            f1_scores.append(f)

        macro_f1 = sum(f1_scores) / len(f1_scores)
        macro_fpr = sum(per_class[l]["fpr"] for l in labels) / len(labels)
        macro_fnr = sum(per_class[l]["fnr"] for l in labels) / len(labels)
        accuracy = correct / total if total > 0 else 0.0

        # Weighted FPR / FNR — weight each class by its true support (tp+fn),
        # so rare classes like NEI/UNCERTAIN don't dominate the average.
        total_support = total  # sum of (tp[l]+fn[l]) over all labels == total
        weighted_fpr = sum((tp[l] + fn[l]) * per_class[l]["fpr"] for l in labels) / total_support if total_support > 0 else 0.0
        weighted_fnr = sum((tp[l] + fn[l]) * per_class[l]["fnr"] for l in labels) / total_support if total_support > 0 else 0.0
        weighted_tpr = 1.0 - weighted_fnr
        weighted_tnr = 1.0 - weighted_fpr

        # Cost aggregates
        total_cost = sum(r.cost_usd for r in results)
        avg_cost = total_cost / total if total > 0 else 0.0
        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)

        return {
            "n": total,
            "accuracy": round(accuracy, 2),
            "macro_f1": round(macro_f1, 2),
            "macro_fpr": round(macro_fpr, 2),
            "macro_fnr": round(macro_fnr, 2),
            "weighted_fpr": round(weighted_fpr, 2),
            "weighted_fnr": round(weighted_fnr, 2),
            "weighted_tpr": round(weighted_tpr, 2),
            "weighted_tnr": round(weighted_tnr, 2),
            "per_class": per_class,
            "total_cost_usd": round(total_cost, 3),
            "avg_cost_usd": round(avg_cost, 3),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }

    # ------------------------------------------------------------------
    # Persistence

    @staticmethod
    def save(results: list[BaselineResult], path: Path) -> None:
        """Save results as a JSON lines file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")
        logger.info("Saved %d results to %s", len(results), path)

    @staticmethod
    def load(path: Path) -> list[BaselineResult]:
        """Load results from a JSON lines file."""
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(BaselineResult.model_validate_json(line))
        return results

    @staticmethod
    def save_metrics(metrics: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", path)
