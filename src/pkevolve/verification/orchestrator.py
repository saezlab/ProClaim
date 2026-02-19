"""
Evidence Programming Orchestrator -- Claude Agent SDK integration.

Agent-driven evidence programming REPL. The agent calls check_sufficiency
(free) after each retrieval round and uses gap feedback to direct subsequent
queries.

Connects directly to GLM's native Anthropic-compatible endpoint at api.z.ai.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)

from pkevolve.verification.data_models import VerificationVerdict
from pkevolve.verification.evidence_state import EvidenceState

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default GLM endpoint configuration
GLM_API_BASE = "https://api.z.ai/api/anthropic"
GLM_DEFAULT_MODEL = "glm-4.6"

SYSTEM_PROMPT = """\
You are an evidence programming agent for scientific claim verification.

You program on evidence the way a coding agent programs on code. Your tools let you
retrieve, extract, synthesize, and evaluate scientific evidence. The sufficiency
classifier (check_sufficiency) is your test suite -- call it frequently to know whether
your evidence gathering is working.

## Your workflow:

1. DECOMPOSE the claim into verifiable subclaims (update the evidence state).
2. Use search_pubmed_progressive for initial evidence retrieval -- it automatically
   broadens from strict gene-pair queries to bridge queries when direct co-mention
   fails. Alternatively, use formulate_pubmed_query + search_pubmed for manual control.
3. For each new paper, try get_full_text_article first for richer content; fall back
   to get_paper_text if full text is unavailable. Then extract facts using the
   add_facts tool (provide a JSON array of fact objects).
4. CHECK SUFFICIENCY -- this is your primary feedback signal. It is free (no LLM cost).
5. If INSUFFICIENT: read the gap types from the classifier output, then formulate
   targeted retrieval queries based on the gap types and call search_for_gap.
6. If keyword searches return few or no results, use find_related_articles on a
   relevant PMID to discover papers through citation links.
7. SYNTHESIZE evidence per subclaim using update_synthesis.
8. If there are conflicting facts, record them with add_conflict.
9. CHECK SUFFICIENCY again after each round of retrieval/extraction.
10. If token_estimate exceeds 40,000, call compress_evidence.
11. Repeat steps 5-10 until sufficient or max iterations reached.
12. Call emit_verdict to produce your final structured output.

## Key principles:

- Call check_sufficiency after EVERY round of retrieval/extraction. It is free.
- The gaps it reports tell you WHAT TYPE of evidence is missing.
- When the classifier reports confidence >= 0.80 and label != INSUFFICIENT, you have
  enough evidence. Call emit_verdict immediately.
- If you hit the iteration limit (8 checks), you must call emit_verdict with your best
  assessment based on available evidence.
- Always produce a structured verdict at the end via emit_verdict.
- Prefer full text over abstracts for fact extraction when available.

## Important:

- The workspace path for all tool calls is: {workspace}
- Always pass the workspace parameter to every MCP tool call.
- Token budget: 50,000 tokens. Compress when token_estimate > 40,000.
- Max iterations: 8 sufficiency checks.
"""


async def verify_claim(
    claim: str,
    workspace: Path,
    model: str = GLM_DEFAULT_MODEL,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
) -> VerificationVerdict:
    """
    Run the evidence programming loop for a single claim.

    1. Initializes workspace with evidence_state.json
    2. Configures Claude Agent SDK with MCP tools + subagents
    3. Sends the claim as a prompt
    4. Streams messages until completion
    5. Reads verdict from workspace/verdict.json

    Returns VerificationVerdict.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    # Initialize evidence state
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Build system prompt with workspace path
    system_prompt = SYSTEM_PROMPT.format(workspace=str(workspace))

    # Build environment for the Claude Agent SDK
    glm_api_key = os.getenv("GLM_API_KEY")
    if not glm_api_key:
        raise RuntimeError(
            "GLM_API_KEY not set. Add it to .env at project root."
        )

    env = {
        **os.environ,
        "API_TIMEOUT_MS": os.getenv("API_TIMEOUT_MS", "3000000"),
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
        "ANTHROPIC_AUTH_TOKEN": glm_api_key,
        "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL", GLM_API_BASE),
    }
    # Remove ANTHROPIC_API_KEY if inherited from os.environ — its presence
    # (even empty) can cause the CLI to attempt Anthropic auth and hang.
    env.pop("ANTHROPIC_API_KEY", None)

    options = ClaudeAgentOptions(
        model=model,
        cwd=str(PROJECT_ROOT),
        env=env,
        setting_sources=["project"],
        allowed_tools=[
            "Task",
            "Read",
            # MCP tools
            "mcp__evidence-tools__formulate_pubmed_query",
            "mcp__evidence-tools__search_pubmed",
            "mcp__evidence-tools__search_pubmed_progressive",
            "mcp__evidence-tools__search_for_gap",
            "mcp__evidence-tools__find_related_articles",
            "mcp__evidence-tools__get_full_text_article",
            "mcp__evidence-tools__get_evidence_summary",
            "mcp__evidence-tools__add_facts",
            "mcp__evidence-tools__get_paper_text",
            "mcp__evidence-tools__update_synthesis",
            "mcp__evidence-tools__add_conflict",
            "mcp__evidence-tools__check_sufficiency",
            "mcp__evidence-tools__compress_evidence",
            "mcp__evidence-tools__emit_verdict",
        ],
        disallowed_tools=[
            "Write",
            "WebSearch",
            "AskUserQuestion",
        ],
        mcp_servers=[{
            "name": "evidence-tools",
            "command": "python",
            "args": ["-m", "pkevolve.verification.mcp_tools"],
            "cwd": str(PROJECT_ROOT),
        }],
    )

    prompt = (
        f"Verify the following scientific claim using evidence programming.\n\n"
        f"Claim: {claim}\n\n"
        f"The evidence state has been initialized at {workspace}/evidence_state.json.\n"
        f"Begin by decomposing the claim into subclaims, then follow the "
        f"evidence programming workflow. Call check_sufficiency after each "
        f"round of evidence gathering. Stop when confidence >= {sufficiency_threshold} "
        f"or after {max_iterations} iterations. "
        f"Always pass workspace=\"{workspace}\" to every MCP tool call."
    )

    # Run the agent
    logger.info("verify_claim: starting for claim=%r, workspace=%s", claim, workspace)

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    logger.info("Agent: %s", block.text[:300])
                if isinstance(block, ToolUseBlock):
                    logger.info("Tool: %s(%s)", block.name,
                               json.dumps(block.input)[:200])
        elif isinstance(message, ResultMessage):
            if hasattr(message, "usage") and message.usage:
                logger.info("Usage: %s", message.usage)

    # Read verdict from disk
    verdict_path = workspace / "verdict.json"
    if verdict_path.exists():
        try:
            return VerificationVerdict.model_validate_json(verdict_path.read_text())
        except Exception as e:
            logger.error("Failed to parse verdict: %s", e)

    # Fallback: construct verdict from final state
    state_path = workspace / "evidence_state.json"
    if state_path.exists():
        state = EvidenceState.load(state_path)
        if state.sufficiency_history:
            latest = state.sufficiency_history[-1]
            return VerificationVerdict(
                verdict=latest.label.replace("SUFFICIENT_", ""),
                confidence=latest.confidence,
                reasoning="Constructed from final evidence state (agent did not call emit_verdict).",
                key_evidence=[f.text for f in state.facts[:10]],
                gaps_remaining=[g.description for g in latest.gaps],
            )

    return VerificationVerdict(
        verdict="INSUFFICIENT",
        confidence=0.0,
        reasoning="Agent did not produce a verdict.",
        key_evidence=[],
        gaps_remaining=["No verdict produced"],
    )


async def verify_claim_batch(
    claims: list[dict],
    output_dir: Path,
    model: str = GLM_DEFAULT_MODEL,
    max_workers: int = 3,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
) -> list[dict]:
    """
    Run evidence programming on a batch of claims with concurrency control.
    Each claim gets its own workspace subdirectory.

    Args:
        claims: List of dicts with at least 'id' and 'text' keys.
        output_dir: Base output directory.
        model: Model identifier (default: glm-4.6).
        max_workers: Maximum concurrent verifications.
        max_iterations: Max sufficiency checks per claim.
        sufficiency_threshold: Confidence threshold for sufficiency.

    Returns:
        List of result dicts with claim info and verdict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_workers)

    async def process_claim(claim_data: dict) -> dict:
        async with semaphore:
            claim_id = claim_data["id"]
            claim_text = claim_data["text"]
            workspace = output_dir / f"claim_{claim_id}"

            logger.info("Batch: processing %s", claim_id)

            try:
                verdict = await verify_claim(
                    claim=claim_text,
                    workspace=workspace,
                    model=model,
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

            result_path = workspace / "result.json"
            result_path.write_text(json.dumps(result, indent=2))

            return result

    tasks = [process_claim(c) for c in claims]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    for r in results:
        if isinstance(r, dict):
            valid_results.append(r)
        elif isinstance(r, Exception):
            logger.error("Batch error: %s", r)

    # Write summary
    summary = {
        "total": len(claims),
        "completed": len(valid_results),
        "errors": len(results) - len(valid_results),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return valid_results
