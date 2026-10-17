"""Oracle-context baseline — a plain, non-agentic LLM given ProClaim's own
final evidence state as static prompt context.

Reuses each claim's cached ``evidence_state.json`` (no new retrieval) to test
whether the state's *content* alone reproduces ProClaim's verdict quality
without code execution or an agentic loop. Falling short argues the interactive
process around the state matters beyond its raw content.

Evidence per paper: PMID + title + abstract + the cached NLI stance signal
(``papers[pmid]["nlp"]``); full text is excluded for prompt-size reasons, so
this matches what the other retrieval baselines see. A non-empty ``facts`` list
is appended as a separate section. Verdict prompt and JSON format are shared
with the other retrieval baselines, so only the evidence varies.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from baselines.shared.claim_loader import load_claim_csv
from baselines.shared.cost_tracker import CostTracker
from baselines.shared.evaluate import EvaluationHarness
from baselines.shared.label_utils import normalize_label
from baselines.shared.llm import LLMBackend
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.prompts import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_TEMPLATE
from baselines.shared.single_shot import run_single_shot_verdict
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)

DEFAULT_DATASETS_DIR = Path("datasets")

MAX_ABSTRACT_CHARS = 1500
MAX_FULLTEXT_CHARS = 10_000


def evidence_state_path(cached_run_dir: Path, dataset: str, dir_id: str, flip: bool) -> Path:
    if dataset == "signor":
        flip_dir = "flip_True" if flip else "flip_False"
        return cached_run_dir / dir_id / flip_dir / "rep_1" / "workspace" / "evidence_state.json"
    return cached_run_dir / dir_id / "rep_1" / "workspace" / "evidence_state.json"


def format_evidence(state: dict) -> str:
    """Render a cached ProClaim evidence_state.json as plain-text context."""
    papers = state.get("papers", {}) or {}
    facts = state.get("facts", []) or []
    conflicts = state.get("conflicts", []) or []
    synthesis = state.get("synthesis", {}) or {}

    if not papers:
        return "(No papers were retrieved for this claim.)"

    lines: list[str] = []
    for i, (pmid, p) in enumerate(papers.items(), 1):
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()[:MAX_ABSTRACT_CHARS]
        nlp = p.get("nlp") or {}
        # .get(key, 0) only falls back on a *missing* key — some cached
        # states have the key present with an explicit `null` value, which
        # .get() passes through unchanged and crashes the ":.2f" format spec.
        # `or 0` catches both missing and explicit-None.
        stance_bits = []
        if "nli_entailment" in nlp:
            stance_bits.append(
                f"NLI[entail={nlp.get('nli_entailment') or 0:.2f} "
                f"contradict={nlp.get('nli_contradiction') or 0:.2f} "
                f"neutral={nlp.get('nli_neutral') or 0:.2f}]"
            )
        if "semantic_similarity" in nlp:
            stance_bits.append(f"sim={nlp.get('semantic_similarity') or 0:.2f}")
        stance_str = f" ({', '.join(stance_bits)})" if stance_bits else ""
        lines.append(f"[Paper {i}] PMID:{pmid} — {title}{stance_str}\n{abstract}")

    if facts:
        lines.append("\n--- Extracted facts (from ProClaim's own pipeline) ---")
        for f in facts:
            lines.append(
                f"- ({f.get('stance', '?')}, conf={f.get('confidence') or 0:.2f}, "
                f"PMID:{f.get('source_pmid', '?')}) {f.get('text', '')}"
            )

    if conflicts:
        lines.append("\n--- Detected conflicts ---")
        for c in conflicts:
            lines.append(f"- {json.dumps(c)}")

    if synthesis:
        lines.append("\n--- Synthesis ---")
        lines.append(json.dumps(synthesis))

    return "\n\n".join(lines)


def format_evidence_papers_only(state: dict) -> str:
    """Render *only* the retrieved papers (title/abstract/NLI stance) from a
    cached evidence_state.json — no extracted facts, no conflicts, no
    synthesis.

    Same paper set and rendering as the paper section of `format_evidence`,
    with the structured-extraction layer removed. Isolates *state structure*
    from *content*: comparing this against `format_evidence_facts_only`
    holds "no extraction at all" (raw abstracts) against "extraction only,
    no raw text" (facts), and comparing it against `format_evidence` isolates
    what the facts/conflicts/synthesis layer adds on top of the same papers.
    """
    papers = state.get("papers", {}) or {}
    if not papers:
        return "(No papers were retrieved for this claim.)"

    lines: list[str] = []
    for i, (pmid, p) in enumerate(papers.items(), 1):
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()[:MAX_ABSTRACT_CHARS]
        nlp = p.get("nlp") or {}
        stance_bits = []
        if "nli_entailment" in nlp:
            stance_bits.append(
                f"NLI[entail={nlp.get('nli_entailment') or 0:.2f} "
                f"contradict={nlp.get('nli_contradiction') or 0:.2f} "
                f"neutral={nlp.get('nli_neutral') or 0:.2f}]"
            )
        if "semantic_similarity" in nlp:
            stance_bits.append(f"sim={nlp.get('semantic_similarity') or 0:.2f}")
        stance_str = f" ({', '.join(stance_bits)})" if stance_bits else ""
        lines.append(f"[Paper {i}] PMID:{pmid} — {title}{stance_str}\n{abstract}")

    return "\n\n".join(lines)


def format_evidence_papers_only_fulltext(state: dict) -> str:
    """Same rendering as `format_evidence_papers_only` (title + NLI stance),
    with the one deliberate swap: full article text in place of the
    abstract, truncated to MAX_FULLTEXT_CHARS. Falls back to the abstract
    when a paper has no cached `full_text` (open-access fetch failed or
    wasn't attempted for that paper).

    Isolates abstract-only vs. full-text evidence: same paper set, same NLI
    stance signal, same prompt, only the text swapped.
    """
    papers = state.get("papers", {}) or {}
    if not papers:
        return "(No papers were retrieved for this claim.)"

    lines: list[str] = []
    for i, (pmid, p) in enumerate(papers.items(), 1):
        title = (p.get("title") or "").strip()
        full_text = (p.get("full_text") or "").strip()
        text = full_text[:MAX_FULLTEXT_CHARS] if full_text else (p.get("abstract") or "").strip()[:MAX_ABSTRACT_CHARS]
        nlp = p.get("nlp") or {}
        stance_bits = []
        if "nli_entailment" in nlp:
            stance_bits.append(
                f"NLI[entail={nlp.get('nli_entailment') or 0:.2f} "
                f"contradict={nlp.get('nli_contradiction') or 0:.2f} "
                f"neutral={nlp.get('nli_neutral') or 0:.2f}]"
            )
        if "semantic_similarity" in nlp:
            stance_bits.append(f"sim={nlp.get('semantic_similarity') or 0:.2f}")
        stance_str = f" ({', '.join(stance_bits)})" if stance_bits else ""
        source_tag = "" if full_text else " [no full text cached — abstract used]"
        lines.append(f"[Paper {i}] PMID:{pmid} — {title}{stance_str}{source_tag}\n{text}")

    return "\n\n".join(lines)


def format_evidence_facts_only(state: dict) -> str:
    """Render *only* the extracted facts from a cached evidence_state.json —
    no paper titles/abstracts/NLI signal, no conflicts, no synthesis.

    Isolates whether ProClaim's compact, distilled fact layer alone carries
    the verdict, vs. needing the full paper-level context `format_evidence`
    provides. Claims with an empty facts list get no evidence.
    """
    facts = state.get("facts", []) or []
    if not facts:
        return "(No facts were extracted for this claim.)"

    lines = [
        f"- ({f.get('stance', '?')}, conf={f.get('confidence') or 0:.2f}, "
        f"PMID:{f.get('source_pmid', '?')}) {f.get('text', '')}"
        for f in facts
    ]
    return "\n".join(lines)


# Context modes (all read evidence_state.json).
EVIDENCE_STATE_MODES = {
    "full": format_evidence,
    "facts_only": format_evidence_facts_only,
    "papers_only": format_evidence_papers_only,
    "papers_only_fulltext": format_evidence_papers_only_fulltext,
}

# Baseline name / output subdirectory per context mode.
OUTPUT_BASE_BY_MODE = {
    "full": "oracle_context",
    "facts_only": "oracle_context_facts_only",
    "papers_only": "oracle_context_papers_only",
    "papers_only_fulltext": "oracle_context_papers_only_fulltext",
}

CONTEXT_MODES = list(OUTPUT_BASE_BY_MODE)


class OracleContextBaseline:
    """Verdict from a plain LLM given ProClaim's own cached evidence state as static context."""

    def __init__(self, llm: LLMBackend, cached_run_dir: Path, dataset: str, context_mode: str = "full") -> None:
        self.llm = llm
        self.cached_run_dir = cached_run_dir
        self.dataset = dataset
        self.context_mode = context_mode
        self.name = OUTPUT_BASE_BY_MODE[context_mode]
        self.model = llm.model
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
        dir_id = (context or {}).get("dir_id")
        flip = bool((context or {}).get("flip", False))
        if dir_id is None:
            # The generic runner (run_baselines_datasets.py) passes the loader's
            # claim dict, which has no dir_id; derive it from the claim_id the
            # same way load_claim_csv does (":" and " " -> "_", "_flip" suffix).
            base_id = claim_id[:-5] if claim_id.endswith("_flip") else claim_id
            flip = flip or claim_id.endswith("_flip")
            dir_id = base_id.replace(":", "_").replace(" ", "_")

        state_path = evidence_state_path(self.cached_run_dir, self.dataset, dir_id, flip)
        formatter = EVIDENCE_STATE_MODES[self.context_mode]

        if not state_path.exists():
            return BaselineResult(
                claim_id=claim_id,
                claim=claim,
                gold_label=normalize_label(gold_label),
                predicted_label="UNCERTAIN",
                reasoning=f"ERROR: no cached context file at {state_path}",
                baseline_name=self.name,
                model=self.llm.model,
            )

        state = json.loads(state_path.read_text())
        evidence_text = formatter(state)

        user_msg = VERIFICATION_USER_TEMPLATE.format(claim=claim, evidence=evidence_text)
        t0 = time.monotonic()
        verdict = run_single_shot_verdict(
            llm=self.llm,
            tracker=self.tracker,
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_msg,
        )
        elapsed = time.monotonic() - t0

        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}  context_mode: {self.context_mode}", ""),
                ("claim", claim),
                (f"evidence state: {state_path}", ""),
                (f"evidence ({len(evidence_text)} chars)", evidence_text),
                (f"verdict: {verdict.predicted_label}", verdict.reasoning),
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
            latency_seconds=round(elapsed, 3),
            baseline_name=self.name,
            model=self.llm.model,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="signor",
                        choices=["signor", "connectomedb", "extri", "nubase", "nasa_exoplanet"])
    parser.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--cached-run-dir", required=True,
                        help="A ProClaim run directory; each claim's evidence_state.json is read from it.")
    parser.add_argument(
        "--context-mode", default="full", choices=CONTEXT_MODES,
        help="'full' = papers (title/abstract/NLI stance) + facts + conflicts + synthesis. "
             "'facts_only' = the extracted facts only. 'papers_only' = the same papers as 'full' "
             "without the extraction layer. 'papers_only_fulltext' = 'papers_only' with full "
             "article text in place of the abstract, truncated to MAX_FULLTEXT_CHARS.",
    )
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to results/baselines/<oracle_context variant>/.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    args = parser.parse_args()

    cached_run_dir = Path(args.cached_run_dir)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else PROJECT_ROOT / "results" / "baselines" / OUTPUT_BASE_BY_MODE[args.context_mode])

    claims = load_claim_csv(args.dataset, args.datasets_dir, args.limit)
    if args.context_mode == "facts_only":
        n_empty_facts = 0
        for c in claims:
            p = evidence_state_path(cached_run_dir, args.dataset, c["dir_id"], c["flip"])
            if p.exists() and not (json.loads(p.read_text()).get("facts") or []):
                n_empty_facts += 1
        logger.info(
            "Facts-only mode: %d/%d claims have an EMPTY facts list in this run "
            "(will get essentially no evidence).", n_empty_facts, len(claims),
        )
    logger.info(
        "Loaded %d claims for dataset=%s, context_mode=%s (cached_run_dir=%s).",
        len(claims), args.dataset, args.context_mode, cached_run_dir,
    )

    llm = LLMBackend(model=args.model, temperature=0.0)
    baseline = OracleContextBaseline(llm, cached_run_dir, args.dataset, args.context_mode)
    harness = EvaluationHarness(baseline, dataset_name=args.dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}.jsonl"

    t_start = time.monotonic()
    results = harness.run(claims, resume_path=output_path)
    wall_time = time.monotonic() - t_start

    metrics = harness.metrics(results)
    metrics["wall_time_seconds"] = round(wall_time, 1)
    metrics["total_cost_usd_from_results"] = round(sum(r.cost_usd for r in results), 4)
    EvaluationHarness.save_metrics(metrics, output_dir / f"{args.dataset}_metrics.json")
    logger.info("Wall time %.1f min", wall_time / 60)


if __name__ == "__main__":
    main()
