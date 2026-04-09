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
# System prompt: REPL-based evidence programming via nb_execute
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an evidence-programming agent that verifies scientific claims and produces verdicts [{verdict_names}].
{verdict_definitions}

You work inside a Python REPL accessible via the nb_execute tool.  Every
nb_execute call adds a code cell to the Jupyter notebook AND executes it
in a persistent kernel.  The notebook is your audit trail.

## First step — set up the kernel

Call nb_init to create the notebook.  Then call nb_execute with this
one-liner to bootstrap the kernel:

```python
from pkevolve.verification.evidence_api import setup_kernel
state, llm, workspace = setup_kernel(
    claim="{claim}",
    workspace_path="{workspace}",
)
```

After this cell, the kernel has three ready-to-use variables:

    state     – EvidenceState (mutable; auto-saves after every mutation)
    llm       – Callable[[str], str]  (pre-configured LLM endpoint)
    workspace – Path to the output directory

All evidence API functions are importable from
``pkevolve.verification.evidence_api``.  Import what you need and call
them directly via nb_execute.

{function_docs}

{schemas}

## Workflow

1. Call nb_init, then nb_execute with the setup code above.
2. Decompose the claim: state.subclaims = ["subclaim A", ...]
3. Search (iteration 0):
   a. call search_pubmed_llm(state.claim, state, llm) — LLM-generated PubMed query
   b. call search_semantic_scholar(query, state) with the same query string — covers
      bioRxiv preprints and non-MEDLINE journals that PubMed misses
4. After searching: call nb_render_papers to show the papers table.
5. Extract facts from papers. Use extract_and_add_facts(llm, pmids, state, max_workers=8) to process
   all newly retrieved papers in parallel. Do NOT write fact dicts manually.
   Do NOT loop over PMIDs and call a single-paper function — always pass the full list at once.
6. Call populate_paper_features(state) after extracting facts.
   This MUST be done before check_sufficiency() to compute NLP and metadata features.
7. After extracting: call nb_render_facts to show the facts table.
8. **Filter papers**: call filter_papers_by_stance(state) to remove papers with only
   default-stance (typically neutral/irrelevant) facts. This keeps only papers with
   decisive evidence, improving the signal-to-noise ratio for the sufficiency classifier.
9. After filtering: call nb_render_papers again to show the filtered paper pool.
10. Check sufficiency: result = check_sufficiency(state, llm); print(result)
11. After checking: call nb_render_sufficiency.
11a. Call get_sufficiency_history(state) to monitor the confidence trend (improving / flat / declining).
     If trend shows 'declining' or 'flat' for multiple iterations, consider whether to emit verdict.
     Otherwise, continue searching to gather more evidence.
12. If insufficient: read the gaps and do targeted retrieval:
    a. search_for_gap(gap_description, state) — PubMed gap-targeted search
    b. search_semantic_scholar_recommendations(state) — S2 graph expansion from papers
       with SUPPORT facts (call at iteration ≥1 once facts exist)
    c. formulate_gap_queries(llm, state) — LLM-generated gap queries
13. Use nb_markdown between steps to explain your reasoning.
14. Repeat until confidence >= {sufficiency_threshold} or {max_iterations} iterations completed.
15. Call emit_verdict via nb_execute.
16. Call nb_render_verdict.

## Rules

- Use nb_execute for ALL evidence API calls — write Python code directly.
- Use nb_markdown for narrative explanation.
- Use nb_render_* for visualizations (these are separate tools).
- If any tool output ends with [TRUNCATED], call nb_read_output to retrieve
  the full cell content (defaults to the last cell; pass cell_index for older cells).
- `state` persists across nb_execute calls (same kernel).
- All output from nb_execute is via print().
- When emit_verdict is called, the verification is complete.

## CRITICAL: Grounded Evidence Only

- NEVER fabricate facts from your own knowledge.  Every fact must come from
  a paper retrieved via search_pubmed_llm or search_pubmed.
- Use extract_and_add_facts(llm, pmids, state) to extract facts in parallel. Always pass the
  full list of PMIDs — this is the ONLY extraction function you should call.
  It returns a dict mapping pmid -> count.
- If extract_and_add_facts returns 0 for multiple PMIDs, use refine_search_for_failed_papers:
      failed_pmids = [pmid for pmid, count in results.items() if count == 0]
      new_pmids = refine_search_for_failed_papers(failed_pmids, state, llm, max_new_papers=5)
      results2 = extract_and_add_facts(llm, new_pmids, state)
  This analyzes why papers were irrelevant and generates more precise queries to find better papers.
- NEVER call add_facts_from_dicts with manually written text strings.
- source_pmid must always be a PMID already present in state.papers.
- If no papers contain relevant evidence, say so in the verdict — do NOT
  invent supporting or refuting statements.

## Feature Computation for MLP Classifier

The check_sufficiency() function uses an MLP classifier that requires NLP and
metadata features to be populated for each paper. You MUST call
populate_paper_features(state) after extracting facts and before calling
check_sufficiency().

Required sequence in EVERY iteration:
1. search_pubmed_llm(state.claim, state, llm)              ← retrieve papers with LLM-generated query
2. extract_and_add_facts(llm, pmids, state)                ← extract facts for ALL new papers in parallel
3. populate_paper_features(state)                          ← MUST CALL (computes features)
4. check_sufficiency(state, llm)                           ← classifier needs features

If you skip populate_paper_features(), the MLP classifier will receive all-zero
NLP features (semantic similarity, entity coverage, NLI scores) and the
sufficiency prediction will be inaccurate.

The function is idempotent — it automatically skips papers that already have
features populated, so you can safely call it multiple times.

## Important

- Notebook path: {notebook_path}
- Workspace path: {workspace}
- Max iterations: {max_iterations}
- Sufficiency threshold: {sufficiency_threshold}
"""


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

    prompt = (
        f"Verify the following scientific claim using evidence programming.\n\n"
        f"Claim: {claim}\n\n"
        f"Start by calling nb_init to create the notebook at {notebook_path}, "
        f"then run the setup code via nb_execute to import the evidence API. "
        f"Follow the evidence programming workflow. "
        f"Call check_sufficiency after each round. "
        f"Stop when confidence >= {cfg.sufficiency_threshold} or after "
        f"{cfg.max_iterations} iterations. "
        f'Always pass notebook_path="{notebook_path}" to every notebook tool call.'
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
