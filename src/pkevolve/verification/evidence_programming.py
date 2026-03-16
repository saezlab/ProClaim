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
  uv run python -m pkevolve.verification.evidence_programming \\
      --claim "Does MAPK1 directly activate H3-3A?"

  # Standalone REPL mode (Mode B, no Claude SDK)
  uv run python -m pkevolve.verification.evidence_programming \\
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
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
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
    populate_paper_features, filter_papers_by_stance,
)
from pkevolve.verification.subagents import (
    extract_facts, synthesize_subclaim, detect_conflicts, formulate_gap_queries,
)
from pkevolve.verification.evidence_state import EvidenceState
from pkevolve.verification.data_models import Fact, Stance, SufficiencyResult
from pkevolve.verification.llm_factory import make_llm

workspace = Path("{workspace}")
workspace.mkdir(parents=True, exist_ok=True)
state = EvidenceState.init_new(
    claim="{claim}",
    subclaims=["{claim}"],
    workspace=workspace,
)

os.environ["MLP_MODEL_DIR"] = "{mlp_model_dir}"

llm = make_llm(
    base_url="{llm_base_url}",
    api_key=os.environ.get("GLM_API_KEY", "EMPTY"),
    model="{subagent_model}",
)

# Pre-warm heavy models (mandatory for all scenarios)
from pkevolve.verification.model_registry import prewarm_all_models
print("Pre-warming ML models (this may take 30-60 seconds)...")
load_times = prewarm_all_models()
for model_name, load_time in load_times.items():
    print(f"  ✓ {{model_name}}: {{load_time:.2f}}s")

print("Setup complete. State initialized. llm() callable ready. Models pre-loaded.")
```

All functions operate on `state` (a live Python object).

**IMPORTANT — auto-save**: `state` auto-saves to disk after every mutation
(add_paper, add_fact, add_conflict, etc.).  You do NOT need to call
`state.save()` or `state.checkpoint_save()` manually — persistence is
handled automatically.  If you truly need an explicit save (rare), use
`state.save()` with no arguments — it writes to the workspace directory.

{function_docs}

{schemas}

## Workflow

1. Call nb_init, then nb_execute with the setup code above.
2. Decompose the claim: state.subclaims = ["subclaim A", ...]
3. Search: call search_pubmed_progressive(state.claim, state) via nb_execute.
4. After searching: call nb_render_papers to show the papers table.
5. Extract facts using extract_and_add_facts(llm, pmid, state) for each paper.
   Do NOT write fact dicts manually — use extract_and_add_facts.
6. Call populate_paper_features(state) after extracting facts.
   This MUST be done before check_sufficiency() to compute NLP and metadata features.
7. After extracting: call nb_render_facts to show the facts table.
8. **Filter papers**: call filter_papers_by_stance(state) to remove papers with only
   NEUTRAL facts. This keeps only papers with SUPPORT or REFUTE evidence, improving
   the signal-to-noise ratio for the sufficiency classifier.
9. After filtering: call nb_render_papers again to show the filtered paper pool.
10. Check sufficiency: result = check_sufficiency(state, llm); print(result)
11. After checking: call nb_render_sufficiency.
12. If insufficient: read the gaps and do targeted retrieval.
13. Use nb_markdown between steps to explain your reasoning.
14. Repeat until sufficient or {max_iterations} iterations.
15. Call emit_verdict via nb_execute.
16. Call nb_render_verdict.

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

## Feature Computation for MLP Classifier

The check_sufficiency() function uses an MLP classifier that requires NLP and
metadata features to be populated for each paper. You MUST call
populate_paper_features(state) after extracting facts and before calling
check_sufficiency().

Required sequence in EVERY iteration:
1. search_pubmed_progressive(...)        ← retrieve papers
2. extract_and_add_facts(llm, pmid, state) for each paper  ← extract facts
3. populate_paper_features(state)        ← MUST CALL (computes features)
4. check_sufficiency(state, llm)         ← classifier needs features

If you skip populate_paper_features(), the MLP classifier will receive all-zero
NLP features (semantic similarity, entity coverage, NLI scores) and the
sufficiency prediction will be inaccurate.

The function is idempotent — it automatically skips papers that already have
features populated, so you can safely call it multiple times.

## Important

- Notebook path: {notebook_path}
- Workspace path: {workspace}
- Max iterations: {max_iterations} sufficiency checks.
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
    system_prompt = SYSTEM_PROMPT.format(
        project_root=str(PROJECT_ROOT),
        workspace=str(workspace),
        notebook_path=str(notebook_path),
        claim=claim,
        max_iterations=cfg.max_iterations,
        model=cfg.model,
        subagent_model=cfg.subagent_model,
        llm_base_url=cfg.subagent_base_url,
        mlp_model_dir=cfg.mlp_model_dir or "results/models/classifier_best",
        schemas=schema_docs(),
        function_docs=function_docs(),
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
# Mode B: Standalone REPL (no Claude SDK)
# ---------------------------------------------------------------------------

def verify_claim_repl_mode(cfg: VerificationSettings) -> Path:
    """Run evidence programming via standalone REPL orchestrator (Mode B)."""
    from pkevolve.verification.repl_orchestrator import verify_claim_repl
    import os
    
    os.environ["MLP_MODEL_DIR"] = cfg.mlp_model_dir or "results/models/classifier_best"

    verdict = verify_claim_repl(
        claim=cfg.claim,
        workspace=cfg.resolved_workspace,
        model=cfg.model,
        subagent_model=cfg.subagent_model,
        base_url=cfg.subagent_base_url,
        api_key=cfg.api_key,
        max_iterations=cfg.max_iterations,
        sufficiency_threshold=cfg.sufficiency_threshold,
    )

    print(f"\nVerdict: {verdict.verdict} (confidence: {verdict.confidence:.2f})")
    print(f"Reasoning: {verdict.reasoning}")

    return cfg.resolved_workspace / "verdict.json"


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

    mode_label = (
        "Claude Agent SDK + nb_execute" if cfg.mode == "sdk" else "Standalone REPL"
    )
    print(f"Claim: {cfg.claim}")
    print(f"Mode: {cfg.mode} ({mode_label})")
    print(f"Model: {cfg.model}")
    if cfg.llm.subagent_model:
        print(f"Subagent model: {cfg.subagent_model}")
    print(f"Workspace: {workspace}")
    if cfg.mode == "sdk":
        print(f"Notebook: {notebook_path}")
    print()

    if cfg.mode == "sdk":
        result = asyncio.run(verify_claim_notebook(cfg))
        print(f"\nNotebook saved: {result}")
    else:
        result = verify_claim_repl_mode(cfg)
        print(f"\nVerdict saved: {result}")

    print(f"Workspace: {workspace}")


if __name__ == "__main__":
    main()
