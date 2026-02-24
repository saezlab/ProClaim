#!/usr/bin/env python3
"""
Notebook-enabled Evidence Verification — RLM Mode A.

Runs the Claude Agent SDK orchestrator with ONE MCP server (notebook-tools).
The agent uses nb_execute to run Python code that calls evidence_api functions
directly in a persistent Jupyter kernel.  No evidence-tools MCP server needed.

This is Mode A of the Recursive Language Model (RLM) architecture:
  - Claude Agent SDK provides the outer agent loop
  - nb_execute is the primary tool (REPL gateway)
  - Evidence state lives as a Python variable in the kernel
  - The notebook is the audit trail

Connects directly to GLM's native Anthropic-compatible endpoint at api.z.ai.

Usage:
  uv run python scripts/verification/demo_evidence_programming.py \\
      --claim "Does MAPK1 directly activate H3-3A?"

  # Standalone REPL mode (Mode B, no Claude SDK)
  uv run python scripts/verification/demo_evidence_programming.py \\
      --claim "Does p53 activate BAX?" --mode repl
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
# System prompt: REPL-based evidence programming via nb_execute
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an evidence-programming agent that verifies scientific claims.

You work inside a Python REPL accessible via the nb_execute tool.  Every
nb_execute call adds a code cell to the Jupyter notebook AND executes it
in a persistent kernel.  The notebook is your audit trail.

## First step — set up the kernel

Call nb_init to create the notebook.  Then call nb_execute with the
following setup code:

```python
import sys, os
from pathlib import Path

_src = str(Path("{project_root}") / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from pkevolve.verification.evidence_api import (
    search_pubmed, search_pubmed_progressive, find_related_articles,
    get_full_text_article, get_paper_text, add_facts_from_dicts,
    update_synthesis, add_conflict, get_evidence_summary,
    check_sufficiency, compress_evidence, emit_verdict,
    formulate_pubmed_query, search_for_gap, extract_and_add_facts,
)
from pkevolve.verification.subagents import (
    extract_facts, synthesize_subclaim, detect_conflicts, formulate_gap_queries,
)
from pkevolve.verification.evidence_state import EvidenceState
from pkevolve.verification.data_models import Fact, Stance, SufficiencyResult

workspace = Path("{workspace}")
workspace.mkdir(parents=True, exist_ok=True)
state = EvidenceState.init_new(
    claim="{claim}",
    subclaims=["{claim}"],
    workspace=workspace,
)

# Wire LLM callable for subagent functions
from openai import OpenAI as _OpenAI
_llm_client = _OpenAI(
    base_url="{llm_base_url}",
    api_key=os.environ.get("GLM_API_KEY", "EMPTY"),
)
import time as _time
def llm(prompt: str, _retries: int = 3) -> str:
    for _attempt in range(_retries):
        try:
            resp = _llm_client.chat.completions.create(
                model="{model}",
                messages=[{{"role": "user", "content": prompt}}],
                temperature=0.1,
            )
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
            print(f'llm(): empty choices on attempt {{_attempt+1}}/{{_retries}}')
        except Exception as _e:
            print(f'llm(): error on attempt {{_attempt+1}}/{{_retries}}: {{_e}}')
        if _attempt < _retries - 1:
            _time.sleep(2 ** _attempt)
    print('llm(): all retries exhausted, returning empty string')
    return ''

print("Setup complete. State initialized. llm() callable ready.")
```

## Available functions (after setup)

All functions operate on `state` (a live Python object).
State is auto-saved to disk after every mutation.

    search_pubmed(query, state, max_results=5) -> list[str]
    search_pubmed_progressive(claim, state) -> list[str]
    find_related_articles(pmid, state, max_results=5) -> list[str]
    get_full_text_article(pmid, state) -> str
    get_paper_text(pmid, state) -> str
    extract_and_add_facts(llm, pmid, state) -> int     # PREFERRED for fact extraction
    add_facts_from_dicts(facts_data, state) -> int
    update_synthesis(subclaim, text, state)
    add_conflict(fact_a_id, fact_b_id, description, severity, state) -> str
    get_evidence_summary(state) -> str
    check_sufficiency(state) -> SufficiencyResult
    compress_evidence(state, target_tokens=40000) -> EvidenceState
    emit_verdict(verdict, confidence, reasoning, key_evidence, gaps_remaining, state, workspace)
    formulate_pubmed_query(claim) -> str
    search_for_gap(gap_description, state, max_results=3) -> list[str]

{schemas}

## Workflow

1. Call nb_init, then nb_execute with the setup code above.
2. Decompose the claim: state.subclaims = ["subclaim A", ...]
3. Search: call search_pubmed_progressive(state.claim, state) via nb_execute.
4. After searching: call nb_render_papers to show the papers table.
5. Extract facts using extract_and_add_facts(llm, pmid, state) for each paper.
   Do NOT write fact dicts manually — use extract_and_add_facts.
6. After extracting: call nb_render_facts to show the facts table.
7. Check sufficiency: result = check_sufficiency(state); print(result)
8. After checking: call nb_render_sufficiency.
9. If insufficient: read the gaps and do targeted retrieval.
10. Use nb_markdown between steps to explain your reasoning.
11. Repeat until sufficient or {max_iterations} iterations.
12. Call emit_verdict via nb_execute.
13. Call nb_render_verdict.

## Rules

- Use nb_execute for ALL evidence API calls — write Python code directly.
- Use nb_markdown for narrative explanation.
- Use nb_render_* for visualizations (these are separate tools).
- `state` persists across nb_execute calls (same kernel).
- All output from nb_execute is via print().
- When emit_verdict is called, the verification is complete.

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

## Important

- Notebook path: {notebook_path}
- Workspace path: {workspace}
- Max iterations: {max_iterations} sufficiency checks.
"""


# ---------------------------------------------------------------------------
# API endpoint configuration
# ---------------------------------------------------------------------------

GLM_API_BASE = "https://api.z.ai/api/anthropic"
GLM_OPENAI_BASE = "https://api.z.ai/api/openai"
GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_DEFAULT_MODEL = "glm-5"


def _build_env() -> dict:
    """Build the env dict for the Claude Agent SDK."""
    api_key = GLM_API_KEY
    if not api_key:
        raise RuntimeError(
            "GLM_API_KEY not set. Add it to .env at project root.\n"
            "  echo 'GLM_API_KEY=<your-key>' >> .env"
        )

    env = {
        **os.environ,
        "API_TIMEOUT_MS": os.getenv("API_TIMEOUT_MS", "3000000"),
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL", GLM_API_BASE),
    }
    env.pop("ANTHROPIC_API_KEY", None)
    return env


# ---------------------------------------------------------------------------
# Mode A: Claude Agent SDK + nb_execute
# ---------------------------------------------------------------------------

async def verify_claim_notebook(
    claim: str,
    workspace: Path,
    notebook_path: Path,
    model: str = GLM_DEFAULT_MODEL,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
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

    workspace.mkdir(parents=True, exist_ok=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    env = _build_env()

    # Initialize evidence state on disk (checkpoint)
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Build system prompt with auto-generated schema docs
    from pkevolve.verification.evidence_api import schema_docs
    system_prompt = SYSTEM_PROMPT.format(
        project_root=str(PROJECT_ROOT),
        workspace=str(workspace),
        notebook_path=str(notebook_path),
        claim=claim,
        max_iterations=max_iterations,
        model=model,
        llm_base_url=GLM_OPENAI_BASE,
        schemas=schema_docs(),
    )

    def _on_stderr(line: str) -> None:
        logger.debug("CLI stderr: %s", line.rstrip())

    python_exe = sys.executable

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
            # Notebook MCP tools only — evidence API is called via nb_execute
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
        f"Start by calling nb_init to create the notebook at {notebook_path}, "
        f"then run the setup code via nb_execute to import the evidence API. "
        f"Follow the evidence programming workflow. "
        f"Call check_sufficiency after each round. "
        f"Stop when confidence >= {sufficiency_threshold} or after "
        f"{max_iterations} iterations. "
        f'Always pass notebook_path="{notebook_path}" to every notebook tool call.'
    )

    logger.info(
        "Mode A: starting claim=%r, workspace=%s, notebook=%s",
        claim, workspace, notebook_path,
    )

    env["CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"] = "300000"

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
# Mode B: Standalone REPL (no Claude SDK)
# ---------------------------------------------------------------------------

def verify_claim_repl_mode(
    claim: str,
    workspace: Path,
    model: str = GLM_DEFAULT_MODEL,
    max_iterations: int = 8,
    sufficiency_threshold: float = 0.80,
) -> Path:
    """Run evidence programming via standalone REPL orchestrator (Mode B)."""
    from pkevolve.verification.repl_orchestrator import verify_claim_repl

    verdict = verify_claim_repl(
        claim=claim,
        workspace=workspace,
        model=model,
        base_url=GLM_OPENAI_BASE,
        max_iterations=max_iterations,
        sufficiency_threshold=sufficiency_threshold,
    )

    print(f"\nVerdict: {verdict.verdict} (confidence: {verdict.confidence:.2f})")
    print(f"Reasoning: {verdict.reasoning}")

    return workspace / "verdict.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evidence Verification — RLM dual-mode orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--claim", required=True, help="Scientific claim to verify.",
    )
    parser.add_argument(
        "--mode", choices=["sdk", "repl"], default="sdk",
        help="Orchestration mode: sdk (Mode A, Claude Agent SDK + nb_execute) "
             "or repl (Mode B, standalone REPL, no SDK). Default: sdk.",
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
        help="Path for the output notebook (Mode A only).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Determine paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "verification" / "notebook_demo"

    workspace = output_dir / "workspace"

    if args.notebook_path:
        notebook_path = Path(args.notebook_path)
    else:
        notebook_path = output_dir / "evidence_report.ipynb"

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
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

    print(f"Claim: {args.claim}")
    print(f"Mode: {args.mode} ({'Claude Agent SDK + nb_execute' if args.mode == 'sdk' else 'Standalone REPL'})")
    print(f"Model: {args.model}")
    print(f"Workspace: {workspace}")
    if args.mode == "sdk":
        print(f"Notebook: {notebook_path}")
    print()

    if args.mode == "sdk":
        result = asyncio.run(verify_claim_notebook(
            claim=args.claim,
            workspace=workspace,
            notebook_path=notebook_path,
            model=args.model,
            max_iterations=args.max_iterations,
            sufficiency_threshold=args.threshold,
        ))
        print(f"\nNotebook saved: {result}")
    else:
        result = verify_claim_repl_mode(
            claim=args.claim,
            workspace=workspace,
            model=args.model,
            max_iterations=args.max_iterations,
            sufficiency_threshold=args.threshold,
        )
        print(f"\nVerdict saved: {result}")

    print(f"Workspace: {workspace}")


if __name__ == "__main__":
    main()
