#!/usr/bin/env python3
"""
Notebook-enabled Evidence Verification.

Runs the Claude Agent SDK orchestrator with ONE MCP server (notebook-tools).
The agent uses nb_execute to run Python code that calls evidence_api functions
directly in a persistent Jupyter kernel.  No evidence-tools MCP server needed.

The Recursive Language Model (RLM) architecture:
  - Claude Agent SDK provides the outer agent loop
  - nb_execute is the primary tool (REPL gateway)
  - Evidence state lives as a Python variable in the kernel
  - The notebook is the audit trail

Connects directly to GLM's native Anthropic-compatible endpoint at api.z.ai.

Usage:
  uv run python -m pkevolve.verification.evidence_programming \\
      --claim "Does MAPK1 directly activate H3-3A?"
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
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt: imported from prompts.py
# ---------------------------------------------------------------------------

from pkevolve.verification.prompts import (
    NOTEBOOK_SYSTEM_PROMPT as SYSTEM_PROMPT,
    NOTEBOOK_USER_PROMPT,
    SUBCLAIM_EXAMPLES,
)


# ---------------------------------------------------------------------------
# Configuration (via pydantic-settings)
# ---------------------------------------------------------------------------

from pkevolve.verification.config import VerificationSettings


# ---------------------------------------------------------------------------
# Mode A: Claude Agent SDK + nb_execute
# ---------------------------------------------------------------------------

async def verify_claim_notebook(
    cfg: VerificationSettings,
) -> Path:
    """Run evidence programming via Claude Agent SDK with nb_execute as primary tool."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        ToolUseBlock,
        query,
    )
    from pkevolve.verification.data_models import VerificationVerdict
    from pkevolve.verification.evidence_state import EvidenceState

    workspace = cfg.resolved_workspace
    notebook_path = cfg.resolved_notebook_path
    claim = cfg.claim

    workspace.mkdir(parents=True, exist_ok=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    env = cfg.build_sdk_env()

    # Initialize evidence state on disk (checkpoint)
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Build system prompt with auto-generated schema docs
    from pkevolve.verification.evidence_api import schema_docs, function_docs
    label_cfg = cfg.labels
    system_prompt = SYSTEM_PROMPT.format(
        workspace=str(workspace),
        notebook_path=str(notebook_path),
        claim=claim,
        max_iterations=cfg.max_iterations,
        sufficiency_threshold=cfg.sufficiency_threshold,
        model=cfg.model,
        schemas=schema_docs(),
        function_docs=function_docs(),
        verdict_names=", ".join(label_cfg.verdict_names()),
        verdict_definitions=label_cfg.verdict_prompt_block(),
        subclaim_examples=SUBCLAIM_EXAMPLES if cfg.include_subclaim_examples else "",
    )

    def _on_stderr(line: str) -> None:
        logger.debug("CLI stderr: %s", line.rstrip())

    python_exe = sys.executable

    options = ClaudeAgentOptions(
        model=cfg.model,
        system_prompt=system_prompt,
        cwd=str(PROJECT_ROOT),
        env=env,
        stderr=_on_stderr,
        extra_args={"debug-to-stderr": None},
        setting_sources=[],
        allowed_tools=[
            "Task",
            "Read",
            # Notebook MCP tools only — evidence API is called via nb_execute
            "mcp__notebook-tools__nb_init",
            "mcp__notebook-tools__nb_markdown",
            "mcp__notebook-tools__nb_execute",
            "mcp__notebook-tools__nb_render_papers",
            "mcp__notebook-tools__nb_render_facts",
            "mcp__notebook-tools__nb_render_sufficiency",
            "mcp__notebook-tools__nb_render_verdict",
            "mcp__notebook-tools__nb_read_output",
            "mcp__notebook-tools__nb_save",
        ],
        disallowed_tools=[
            "Write",
            "WebSearch",
            "AskUserQuestion",
        ],
        mcp_servers={
            "notebook-tools": {
                "command": python_exe,
                "args": ["-m", "pkevolve.verification.notebook_mcp"],
                "cwd": str(PROJECT_ROOT),
            },
        }
    )

    prompt = NOTEBOOK_USER_PROMPT.format(
        claim=claim,
        notebook_path=notebook_path,
        sufficiency_threshold=cfg.sufficiency_threshold,
        max_iterations=cfg.max_iterations,
    )

    logger.info(
        "Mode A: starting claim=%r, workspace=%s, notebook=%s",
        claim, workspace, notebook_path,
    )

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
    cfg = VerificationSettings.from_cli()

    # Resolve paths
    output_dir = cfg.resolved_output_dir
    workspace = cfg.resolved_workspace
    notebook_path = cfg.resolved_notebook_path

    # Configure logging
    log_level = logging.DEBUG if cfg.verbose else logging.INFO
    log_file = output_dir / "run.log"
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

    print(f"Claim: {cfg.claim}")
    print(f"Model: {cfg.model}")
    if cfg.llm.subagent_model:
        print(f"Subagent model: {cfg.subagent_model}")
    print(f"Workspace: {workspace}")
    print(f"Notebook: {notebook_path}")
    print()

    result = asyncio.run(verify_claim_notebook(cfg))
    print(f"\nNotebook saved: {result}")
    print(f"Workspace: {workspace}")


if __name__ == "__main__":
    main()
