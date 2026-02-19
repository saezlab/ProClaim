#!/usr/bin/env python3
"""
Notebook-enabled Evidence Verification — produces a Jupyter notebook artifact.

Runs the Claude Agent SDK orchestrator with TWO MCP servers:
  1. evidence-tools  — search, classify, extract (existing)
  2. notebook-tools   — write results into a live Jupyter notebook (new)

The agent documents its evidence programming workflow in the notebook as it
works, producing a rich, reviewable .ipynb file.

Connects directly to GLM's native Anthropic-compatible endpoint at api.z.ai.

Usage:
  uv run python scripts/verification/demo_evidence_programming.py \
      --claim "Does MAPK1 directly activate H3-3A?"

  # Custom model
  uv run python scripts/verification/demo_evidence_programming.py \
      --claim "Does p53 activate BAX?" \
      --model glm-5
"""

import sys
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Path resolution (project convention)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt: evidence programming + notebook documentation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an evidence programming agent for scientific claim verification.

You program on evidence the way a coding agent programs on code. Your tools let you
retrieve, extract, synthesize, and evaluate scientific evidence. The sufficiency
classifier (check_sufficiency) is your test suite -- call it frequently to know whether
your evidence gathering is working.

## Your workflow:

1. Call nb_init to create the notebook, then DECOMPOSE the claim into subclaims.
2. Use search_pubmed_progressive for initial evidence retrieval -- it automatically
   broadens from strict gene-pair queries to bridge queries when direct co-mention
   fails. Alternatively, use formulate_pubmed_query + search_pubmed for manual control.
3. After searching: call nb_render_papers to document the retrieved papers.
4. For each new paper, try get_full_text_article first for richer content; fall back
   to get_paper_text if full text is unavailable. Then extract facts using the
   add_facts tool (provide a JSON array of fact objects).
5. After extracting: call nb_render_facts to show the color-coded fact table.
6. CHECK SUFFICIENCY -- this is your primary feedback signal. It is free (no LLM cost).
7. After checking: call nb_render_sufficiency to visualize coverage and gaps.
8. If INSUFFICIENT: read the gap types from the classifier output, then formulate
   targeted retrieval queries based on the gap types and call search_for_gap.
9. If keyword searches return few or no results, use find_related_articles on a
   relevant PMID to discover papers through citation links.
10. Use nb_markdown between steps to explain your reasoning and decisions.
11. SYNTHESIZE evidence per subclaim using update_synthesis.
12. If there are conflicting facts, record them with add_conflict.
13. CHECK SUFFICIENCY again after each round of retrieval/extraction.
14. If token_estimate exceeds 40,000, call compress_evidence.
15. Repeat steps 8-14 until sufficient or max iterations reached.
16. Call emit_verdict to produce your final structured output.
17. Call nb_render_verdict to display the verdict in the notebook.

## Key principles:

- Call check_sufficiency after EVERY round of retrieval/extraction. It is free.
- The gaps it reports tell you WHAT TYPE of evidence is missing.
- When the classifier reports confidence >= 0.80 and label != INSUFFICIENT, you have
  enough evidence. Call emit_verdict immediately.
- If you hit the iteration limit (8 checks), you must call emit_verdict with your best
  assessment based on available evidence.
- Always produce a structured verdict at the end via emit_verdict.
- Prefer full text over abstracts for fact extraction when available.
- ALWAYS document your work in the notebook using nb_* tools.

## Important:

- The workspace path for all evidence tool calls is: {workspace}
- The notebook path for all notebook tool calls is: {notebook_path}
- Always pass the workspace parameter to every evidence MCP tool call.
- Always pass the notebook_path parameter to every notebook MCP tool call.
- Token budget: 50,000 tokens. Compress when token_estimate > 40,000.
- Max iterations: 8 sufficiency checks.
"""


# ---------------------------------------------------------------------------
# API endpoint configuration
# ---------------------------------------------------------------------------
# Connects directly to GLM's native Anthropic-compatible endpoint.
# ---------------------------------------------------------------------------

GLM_API_BASE = "https://api.z.ai/api/anthropic"
GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_DEFAULT_MODEL = "glm-4.6"


def _build_env() -> dict:
    """Build the env dict for the Claude Agent SDK.

    Uses ANTHROPIC_AUTH_TOKEN with the GLM key and points
    ANTHROPIC_BASE_URL to api.z.ai's native Anthropic endpoint.
    ANTHROPIC_API_KEY is removed to prevent Anthropic auth flows.
    """
    api_key = GLM_API_KEY
    if not api_key:
        raise RuntimeError(
            "GLM_API_KEY not set. Add it to .env at project root.\n"
            "  echo 'GLM_API_KEY=<your-key>' >> .env"
        )

    env = {
        **os.environ,
        "API_TIMEOUT_MS": os.getenv("API_TIMEOUT_MS", "3000000"),
        # Suppress CLI beta headers and telemetry (defense in depth)
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL", GLM_API_BASE),
    }
    # Remove ANTHROPIC_API_KEY if inherited from os.environ — its presence
    # (even empty) can cause the CLI to attempt Anthropic auth and hang.
    env.pop("ANTHROPIC_API_KEY", None)

    return env


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def verify_claim_notebook(
    claim: str,
    workspace: Path,
    notebook_path: Path,
    model: str = GLM_DEFAULT_MODEL,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
) -> Path:
    """
    Run the evidence programming loop, writing results to a Jupyter notebook.

    Returns the path to the generated notebook.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        ToolUseBlock,
        query,
    )
    from pkevolve.verification.data_models import VerificationVerdict
    from pkevolve.verification.evidence_state import EvidenceState

    workspace.mkdir(parents=True, exist_ok=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    # Build env (validates GLM API key)
    env = _build_env()

    # Initialize evidence state
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Build system prompt
    system_prompt = SYSTEM_PROMPT.format(
        workspace=str(workspace),
        notebook_path=str(notebook_path),
    )

    # Stderr callback to surface CLI debug output
    def _on_stderr(line: str) -> None:
        logger.debug("CLI stderr: %s", line.rstrip())

    # Resolve Python from the running environment so MCP servers
    # don't fall back to /usr/bin/python (which lacks pkevolve).
    python_exe = sys.executable

    # Environment
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        cwd=str(PROJECT_ROOT),
        env=env,
        stderr=_on_stderr,
        extra_args={"debug-to-stderr": None},
        setting_sources=[],
        allowed_tools=[
            "Task",
            "Read",
            # Evidence MCP tools
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
            # Notebook MCP tools
            "mcp__notebook-tools__nb_init",
            "mcp__notebook-tools__nb_markdown",
            "mcp__notebook-tools__nb_execute",
            "mcp__notebook-tools__nb_render_papers",
            "mcp__notebook-tools__nb_render_facts",
            "mcp__notebook-tools__nb_render_sufficiency",
            "mcp__notebook-tools__nb_render_verdict",
            "mcp__notebook-tools__nb_save",
        ],
        disallowed_tools=[
            "Write",
            "WebSearch",
            "AskUserQuestion",
        ],
        mcp_servers={
            "evidence-tools": {
                "command": python_exe,
                "args": ["-m", "pkevolve.verification.mcp_tools"],
                "cwd": str(PROJECT_ROOT),
            },
            "notebook-tools": {
                "command": python_exe,
                "args": ["-m", "pkevolve.verification.notebook_mcp"],
                "cwd": str(PROJECT_ROOT),
            },
        }
    )

    prompt = (
        f"Verify the following scientific claim using evidence programming.\n\n"
        f"Claim: {claim}\n\n"
        f"The evidence state has been initialized at {workspace}/evidence_state.json.\n"
        f"The notebook is at {notebook_path}.\n\n"
        f"Start by calling nb_init to set up the notebook, then follow the "
        f"evidence programming workflow. Document every step in the notebook. "
        f"Call check_sufficiency after each round of evidence gathering. "
        f"Stop when confidence >= {sufficiency_threshold} or after "
        f"{max_iterations} iterations. "
        f'Always pass workspace="{workspace}" to every evidence tool call. '
        f'Always pass notebook_path="{notebook_path}" to every notebook tool call.'
    )

    # Run the agent
    logger.info(
        "verify_claim_notebook: starting for claim=%r, workspace=%s, notebook=%s",
        claim, workspace, notebook_path,
    )

    # Use query() — the same function that works in run_signor_qa_glm.py.
    # Set a generous stream-close timeout for long-running agent loops.
    env["CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"] = "300000"  # 5 minutes

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    logger.info("Agent: %s", block.text[:300])
                if isinstance(block, ToolUseBlock):
                    logger.info(
                        "Tool: %s(%s)", block.name,
                        json.dumps(block.input)[:200],
                    )
        elif isinstance(message, ResultMessage):
            if hasattr(message, "usage") and message.usage:
                logger.info("Usage: %s", message.usage)

    # Print verdict if available
    verdict_path = workspace / "verdict.json"
    if verdict_path.exists():
        try:
            verdict = VerificationVerdict.model_validate_json(
                verdict_path.read_text()
            )
            print(f"\nVerdict: {verdict.verdict} (confidence: {verdict.confidence:.2f})")
            print(f"Reasoning: {verdict.reasoning}")
        except Exception as e:
            logger.error("Failed to parse verdict: %s", e)

    return notebook_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evidence Verification with Jupyter Notebook output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--claim", required=True, help="Scientific claim to verify.",
    )
    parser.add_argument(
        "--model", type=str, default=GLM_DEFAULT_MODEL,
        help=f"Model identifier (default: {GLM_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Confidence threshold for early stopping (default: 0.80).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Maximum verification iterations (default: 8).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output workspace directory.",
    )
    parser.add_argument(
        "--notebook-path", type=str, default=None,
        help="Path for the output notebook (default: <output-dir>/evidence_report.ipynb).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Determine paths (needed before logging setup for log file location)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "verification" / "notebook_demo"

    workspace = output_dir / "workspace"

    if args.notebook_path:
        notebook_path = Path(args.notebook_path)
    else:
        notebook_path = output_dir / "evidence_report.ipynb"

    # Configure logging — write to file in the same directory as the notebook
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = notebook_path.parent / f"{notebook_path.stem}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    logger.info("Log file: %s", log_file)

    print(f"Claim: {args.claim}")
    print(f"Model: {args.model}")
    print(f"Backend: GLM direct ({GLM_API_BASE})")
    print(f"Workspace: {workspace}")
    print(f"Notebook: {notebook_path}")
    print()

    result_nb = asyncio.run(verify_claim_notebook(
        claim=args.claim,
        workspace=workspace,
        notebook_path=notebook_path,
        model=args.model,
        max_iterations=args.max_iterations,
        sufficiency_threshold=args.threshold,
    ))

    print(f"\nNotebook saved: {result_nb}")
    print(f"Workspace: {workspace}")
    print("Open the notebook in JupyterLab to view the evidence report.")


if __name__ == "__main__":
    main()
