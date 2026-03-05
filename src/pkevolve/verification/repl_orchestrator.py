"""
Standalone REPL Orchestrator (Mode B) -- OpenAI-compatible client.

Implements the Recursive Language Model paradigm: the LLM generates Python
code that calls evidence_api functions directly inside a persistent Jupyter
kernel. Kernel stdout/stderr is captured and fed back to the LLM as the
next user message.

This is the MCP-free alternative to orchestrator.py (Mode A).  Both modes
share evidence_api, subagents, and kernel_runner.

Usage::

    from pkevolve.verification.repl_orchestrator import verify_claim_repl
    verdict = verify_claim_repl(
        claim="EGFR activates MAPK1 via phosphorylation",
        workspace=Path("workspace/egfr_mapk1"),
        model="glm-4.6",
        base_url="https://api.z.ai/api/paas/v4/",
    )
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from pkevolve.verification.config import VerificationSettings, get_settings
from pkevolve.verification.data_models import VerificationVerdict
from pkevolve.verification.evidence_state import EvidenceState
from pkevolve.verification.kernel_runner import KernelRunner, outputs_to_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (kept as module-level constants for backward compat)
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4/"
DEFAULT_MODEL = "glm-4.6"
MAX_TURNS = 30  # hard ceiling on LLM turns (not sufficiency iterations)
MAX_OUTPUT_CHARS = 12_000  # truncate kernel output fed back to LLM

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an evidence-programming agent that verifies scientific claims.

You work inside a Python REPL.  The kernel already has these imports and
variables ready:

    state          – EvidenceState (mutable; lives in kernel memory)
    workspace      – Path to the workspace directory
    llm            – Callable[[str], str]  (calls the LLM endpoint)

    # Evidence API (operate on `state` in-place)
    search_pubmed(query, state, max_results=5)
    search_pubmed_progressive(claim, state, max_results_per_tier=5)
    find_related_articles(pmid, state, max_results=5)
    get_full_text_article(pmid, state)
    get_paper_text(pmid, state)
    add_facts_from_dicts(facts_data, state)
    extract_and_add_facts(llm, pmid, state) -> int       # PREFERRED for fact extraction
    update_synthesis(subclaim, text, state)
    add_conflict(fact_a_id, fact_b_id, description, severity, state)
    get_evidence_summary(state)
    check_sufficiency(state) -> SufficiencyResult
    compress_evidence(state, target_tokens=40000) -> EvidenceState
    emit_verdict(verdict, confidence, reasoning, key_evidence, gaps_remaining, state, workspace)
    formulate_pubmed_query(claim)
    search_for_gap(gap_description, state, max_results=3)

    # Subagents (require the `llm` callable)
    extract_facts(llm, paper_text, claim, subclaims, source_pmid) -> list[Fact]
    synthesize_subclaim(llm, facts, subclaim) -> str
    detect_conflicts(llm, facts) -> list[dict]
    formulate_gap_queries(llm, gaps) -> list[str]

{schemas}

## Workflow

1. Decompose the claim into subclaims:
       state.subclaims = ["subclaim A", "subclaim B", ...]
2. Search: call search_pubmed_progressive(state.claim, state) for initial retrieval.
3. Extract facts using extract_and_add_facts(llm, pmid, state) for EACH paper.
   This reads the paper and uses the LLM to extract grounded facts.
   Do NOT write fact dicts manually — use extract_and_add_facts.
4. Check sufficiency: result = check_sufficiency(state); print(result).
   This is FREE (no LLM cost). Call it after EVERY retrieval round.
5. If insufficient: read the gaps, call search_for_gap or find_related_articles.
6. Synthesize: call update_synthesis for each subclaim.
7. If state.token_estimate > 40000: state = compress_evidence(state).
8. Repeat 3-7 until sufficient or MAX_ITERATIONS ({max_iterations}) reached.
9. Call emit_verdict to produce your final structured output.

## Rules

- Your replies MUST contain exactly ONE Python code block (```python ... ```).
- Do NOT include explanatory text outside the code block.
- All output is via print(). The kernel stdout is your feedback channel.
- `state` is a live Python object -- mutate it freely.
- State is auto-saved to disk after every mutation.
- When you call emit_verdict, the loop terminates.
- After {max_iterations} sufficiency checks, you MUST call emit_verdict.

## CRITICAL: Grounded Evidence Only

- NEVER fabricate facts from your own knowledge.  Every fact must come from
  a paper retrieved via search_pubmed / search_pubmed_progressive.
- Use extract_and_add_facts(llm, pmid, state) to add facts. This reads the
  actual paper and extracts grounded statements.
- If extract_and_add_facts returns 0 for a paper, try:
      text = get_full_text_article(pmid, state)
      if not text:
          text = get_paper_text(pmid, state)
      facts = extract_facts(llm, text, state.claim, state.subclaims, pmid)
      add_facts_from_dicts([dict(text=f.text, stance=f.stance.value,
          source_pmid=f.source_pmid, relevant_subclaims=f.relevant_subclaims,
          confidence=f.confidence) for f in facts], state)
- NEVER call add_facts_from_dicts with manually written text strings.
- source_pmid must always be a PMID already present in state.papers.
- If no papers contain relevant evidence, say so in the verdict — do NOT
  invent supporting or refuting statements.
"""

# ---------------------------------------------------------------------------
# Code-fence parser
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL,
)


def extract_code(response_text: str) -> Optional[str]:
    """Extract the first Python code block from LLM response text."""
    m = _CODE_FENCE_RE.search(response_text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def verify_claim_repl(
    claim: str,
    workspace: Path,
    model: str = DEFAULT_MODEL,
    subagent_model: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    max_turns: int = MAX_TURNS,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
    temperature: float = 0.2,
    *,
    cfg: Optional[VerificationSettings] = None,
) -> VerificationVerdict:
    """
    Run the evidence programming REPL loop for a single claim.

    Accepts either explicit keyword arguments (backward compatible) or a
    ``cfg`` VerificationSettings instance.  When ``cfg`` is provided, its
    values take priority over the individual kwargs.

    Returns VerificationVerdict.
    """
    # Merge cfg into locals when provided
    if cfg is not None:
        claim = cfg.claim or claim
        workspace = cfg.resolved_workspace
        model = cfg.model
        subagent_model = cfg.subagent_model
        base_url = cfg.subagent_base_url
        api_key = cfg.api_key
        max_turns = cfg.max_turns
        max_iterations = cfg.max_iterations
        sufficiency_threshold = cfg.sufficiency_threshold
        temperature = cfg.temperature

    workspace.mkdir(parents=True, exist_ok=True)

    # Initialize evidence state on disk (for checkpointing)
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Resolve API key
    if api_key is None:
        api_key = get_settings().api_key

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Start kernel and inject prelude (with llm callable)
    runner = KernelRunner(session_id=f"repl:{workspace}")
    if not runner.start():
        raise RuntimeError("Could not start Jupyter kernel.")

    runner.inject_prelude(
        claim=claim,
        workspace=str(workspace),
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=subagent_model or model,
    )

    # Build system prompt with auto-generated schema docs
    from pkevolve.verification.evidence_api import schema_docs
    system = SYSTEM_PROMPT.format(
        max_iterations=max_iterations,
        schemas=schema_docs(),
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Verify the following scientific claim:\n\n"
                f"**{claim}**\n\n"
                f"Workspace: {workspace}\n"
                f"Begin by decomposing the claim and running initial evidence retrieval."
            ),
        },
    ]

    verdict_path = workspace / "verdict.json"

    try:
        for turn in range(max_turns):
            logger.info("REPL turn %d/%d", turn + 1, max_turns)

            # --- LLM call ---
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
            )
            assistant_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})

            # --- Parse code ---
            code = extract_code(assistant_text)
            if code is None:
                # No code block — ask LLM to produce one
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response must contain a Python code block "
                        "(```python ... ```). Please try again."
                    ),
                })
                continue

            # --- Execute in kernel ---
            outputs = runner.execute(code, timeout=120)
            output_text = outputs_to_text(outputs)

            # Truncate if necessary
            if len(output_text) > MAX_OUTPUT_CHARS:
                output_text = (
                    output_text[:MAX_OUTPUT_CHARS]
                    + f"\n\n... [truncated, {len(output_text)} chars total]"
                )

            logger.info(
                "REPL turn %d: code=%d chars, output=%d chars",
                turn + 1, len(code), len(output_text),
            )

            # --- Check if verdict was emitted ---
            if verdict_path.exists():
                logger.info("Verdict file detected — stopping loop.")
                break

            # --- Feed output back ---
            if output_text:
                feedback = f"Kernel output:\n```\n{output_text}\n```"
            else:
                feedback = "Kernel output: (no output)"

            messages.append({"role": "user", "content": feedback})

        else:
            # max_turns exceeded — force verdict
            logger.warning("Max turns (%d) exceeded — forcing verdict.", max_turns)
            force_code = (
                'emit_verdict(\n'
                '    verdict="INSUFFICIENT",\n'
                '    confidence=0.0,\n'
                '    reasoning="Max REPL turns exceeded without verdict.",\n'
                '    key_evidence=[f.text for f in state.facts[:10]],\n'
                '    gaps_remaining=["Max turns exceeded"],\n'
                '    state=state,\n'
                '    workspace=workspace,\n'
                ')'
            )
            runner.execute(force_code)

    finally:
        # Checkpoint state and trace
        try:
            _checkpoint_from_kernel(runner, workspace)
        except Exception as e:
            logger.warning("Checkpoint failed: %s", e)
        runner.shutdown()

    # Save conversation log
    conv_path = workspace / "conversation.json"
    conv_path.write_text(json.dumps(messages, indent=2, ensure_ascii=False))

    # Read verdict
    if verdict_path.exists():
        try:
            return VerificationVerdict.model_validate_json(
                verdict_path.read_text()
            )
        except Exception as e:
            logger.error("Failed to parse verdict: %s", e)

    return VerificationVerdict(
        verdict="INSUFFICIENT",
        confidence=0.0,
        reasoning="REPL loop completed without producing a verdict.",
        key_evidence=[],
        gaps_remaining=["No verdict produced"],
    )


def _checkpoint_from_kernel(runner: KernelRunner, workspace: Path) -> None:
    """Save state and trace from the kernel to disk."""
    code = (
        "state.checkpoint_save(workspace)\n"
        "print('Checkpoint saved.')"
    )
    runner.execute(code, timeout=30)


# ---------------------------------------------------------------------------
# Batch support
# ---------------------------------------------------------------------------


def verify_claims_batch(
    claims: list[dict],
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
) -> list[dict]:
    """
    Run verification on a batch of claims sequentially.

    Each claim gets its own workspace subdirectory.

    Args:
        claims: List of dicts with at least 'id' and 'text' keys.
        output_dir: Base output directory.

    Returns:
        List of result dicts with claim info and verdict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for claim_data in claims:
        claim_id = claim_data["id"]
        claim_text = claim_data["text"]
        ws = output_dir / f"claim_{claim_id}"

        logger.info("Batch: processing %s", claim_id)

        try:
            verdict = verify_claim_repl(
                claim=claim_text,
                workspace=ws,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_iterations=max_iterations,
                sufficiency_threshold=sufficiency_threshold,
            )
        except Exception as e:
            logger.error("Batch: failed %s: %s", claim_id, e)
            verdict = VerificationVerdict(
                verdict="INSUFFICIENT",
                confidence=0.0,
                reasoning=f"Error: {e}",
                key_evidence=[],
                gaps_remaining=[str(e)],
            )

        result = {
            "claim_id": claim_id,
            "claim": claim_text,
            "gold_label": claim_data.get("gold_label"),
            "predicted": verdict.model_dump(),
        }
        result_path = ws / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        results.append(result)

    # Write summary
    summary = {
        "total": len(claims),
        "completed": len(results),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return results
