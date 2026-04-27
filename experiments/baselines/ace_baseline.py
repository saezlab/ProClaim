"""
ACE baseline — Agentic Context Engineering for claim verification.

Reimplements the ACE Generator agent (from the ACE framework) using litellm
for LLM calls.  Runs in-process — no subprocess or separate venv needed.

ACE's Generator receives a playbook of strategies/insights and a question,
then returns a JSON with ``reasoning``, ``bullet_ids``, and ``final_answer``.
For claim verification we supply a built-in playbook that frames the task as
SUPPORT / REFUTE / UNCERTAIN classification.

Prerequisites:
  - An LLM API key recognised by litellm (e.g. ``ANTHROPIC_API_KEY``).

Cost: 1 LLM call per claim.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.llm import LLMBackend
from baselines.shared.label_utils import normalize_label, verdict_defs_block, verdict_names, verdict_options_str
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)

# ── Verdict label helpers (used in playbook and question template) ───

_VERDICT_LIST = ", ".join(verdict_names())            # "SUPPORT, REFUTE, UNCERTAIN"
_VERDICT_OPTIONS = verdict_options_str()               # '"SUPPORT" | "REFUTE" | "UNCERTAIN"'
_VERDICT_DEFS = verdict_defs_block()                   # multi-line label definitions

# ── ACE Generator prompt (from the ACE framework) ───────────────────

_ACE_GENERATOR_PROMPT = """\
You are an analysis expert tasked with answering questions using your knowledge, \
a curated playbook of strategies and insights and a reflection that goes over \
the diagnosis of all previous mistakes made while answering the question.

**Instructions:**
- Read the playbook carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the playbook contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

Your output should be a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that's relevant, helpful for you to answer this question, you should include their bullet_id in this list
- final_answer: your concise final answer


**Playbook:**
{playbook}

**Reflection:**
{reflection}

**Question:**
{question}

**Context:**
{context}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "bullet_ids": ["calc-00001", "fin-00002"],
  "final_answer": "[Your concise final answer here]"
}}

---
"""

# Default playbook for scientific claim verification
_CLAIM_VERIFICATION_PLAYBOOK = f"""\
## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Classify the scientific claim as {_VERDICT_LIST} based on your knowledge.
[str-00002] helpful=0 harmful=0 :: Label definitions:\n{_VERDICT_DEFS}
[str-00003] helpful=0 harmful=0 :: Be conservative: prefer UNCERTAIN when knowledge is insufficient.

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID
[err-00001] helpful=0 harmful=0 :: Do not hallucinate citations or evidence — rely only on your actual knowledge.

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

_CLAIM_QUESTION_TEMPLATE = (
    "Scientific Claim Verification Task:\n\n"
    f"Classify the following scientific claim as one of: {_VERDICT_LIST}.\n\n"
    "Claim: {claim}\n\n"
    f"Your final_answer MUST be exactly one of: {_VERDICT_LIST}."
)


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_answer(response: str) -> str:
    """Extract final_answer from ACE Generator JSON response."""
    try:
        parsed = json.loads(response)
        return str(parsed.get("final_answer", "No final answer found"))
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    # Fallback: regex for "final_answer": "..."
    for pattern in [
        r'"final_answer"\s*:\s*"([^"]*)"',
        r"'final_answer'\s*:\s*'([^']*)'",
        r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)',
    ]:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip().strip("\"'")

    return "No final answer found"


# ── Baseline class ───────────────────────────────────────────────────


class ACEBaseline:
    """Verify claims via the ACE Generator agent (in-process, LLMBackend).

    Parameters
    ----------
    llm:
        Shared ``LLMBackend`` instance.
    playbook:
        Optional playbook text or path to a ``.txt`` playbook file.
        Defaults to the built-in claim verification playbook.
    """

    name = "ace"

    def __init__(
        self,
        llm: LLMBackend,
        *,
        playbook: str | None = None,
    ) -> None:
        self._llm = llm
        self.model = llm.model
        self.log_dir: Path | None = None

        # Resolve playbook
        if playbook is not None:
            p = Path(playbook)
            if p.is_file():
                self._playbook = p.read_text()
            else:
                self._playbook = playbook
        else:
            self._playbook = _CLAIM_VERIFICATION_PLAYBOOK

        # Cost estimation — reuse CostTracker pricing table
        model_key = self.model.split("/", 1)[-1] if "/" in self.model else self.model
        in_price, out_price = CostTracker.DEFAULT_PRICING.get(
            model_key, CostTracker.FALLBACK_PRICING,
        )
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

        question = _CLAIM_QUESTION_TEMPLATE.format(claim=claim)

        # Optionally forward pre-retrieved evidence
        context_text = ""
        if context and context.get("evidence"):
            context_text = f"Evidence: {context['evidence']}"

        prompt = _ACE_GENERATOR_PROMPT.format(
            playbook=self._playbook,
            reflection="(empty)",
            question=question,
            context=context_text,
        )

        gen_response, input_tokens, output_tokens = self._llm.complete_text(
            system="", user=prompt,
        )

        latency = time.monotonic() - t0
        final_answer = _extract_answer(gen_response)
        predicted = normalize_label(final_answer)

        # Write per-claim log
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{claim_id}.log"
            with open(log_path, "w") as lf:
                lf.write(f"=== claim_id: {claim_id} ===\n")
                lf.write(f"=== claim ===\n{claim}\n\n")
                lf.write(f"=== final_answer: {final_answer} ===\n")
                lf.write(f"=== predicted: {predicted} ===\n")
                lf.write(f"=== response ===\n{gen_response}\n")
                lf.write(f"=== prompt_sent ===\n{prompt}\n")
                lf.write(f"=== playbook ===\n{self._playbook}\n")

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=0.0,
            reasoning=gen_response[:1000] if gen_response else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(
                input_tokens / 1_000_000 * self._in_price
                + output_tokens / 1_000_000 * self._out_price
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )
