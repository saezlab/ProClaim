#!/usr/bin/env python3
"""
Evidence Programming — Direct API with per-call context refresh.

Uses LiteLLM for model-agnostic LLM completion, bash for code execution,
and jupytext for post-hoc notebook generation.  No Jupyter kernel, no MCP
server, no Claude Agent SDK.

Architecture:
  - Single flat loop: each LLM call gets [system_prompt, execution_log]
  - Per-call context refresh: messages are rebuilt from the execution log
    before every LLM call — no conversation accumulation
  - Tool results flow through the log, not through message history
  - State persistence: EvidenceState auto-saves to disk after every mutation

Usage:
  uv run python -m pkevolve.verification.evidence_programming_direct \\
      --config experiments/config.yaml \\
      --claim "Does MAPK1 directly activate H3-3A?"
"""

import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = int(os.environ.get("NB_MAX_OUTPUT_CHARS", "12000"))


# ---------------------------------------------------------------------------
# System prompt: imported from prompts.py
# ---------------------------------------------------------------------------

from pkevolve.verification.prompts import DIRECT_SYSTEM_PROMPT as SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format, used by LiteLLM)
# ---------------------------------------------------------------------------

def build_tool_schemas() -> list[dict]:
    """Build tool schemas in OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    "Execute a command in bash. Use for running Python code: "
                    "python3 -c 'code'. Working directory is the workspace."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file at the given path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to read.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Jupytext execution log management
# ---------------------------------------------------------------------------

def init_jupytext_log(path: Path, claim: str) -> None:
    """Create the initial jupytext percent-format .py file."""
    header = textwrap.dedent(f"""\
        # ---
        # jupyter:
        #   jupytext:
        #     text_representation:
        #       format_name: percent
        # ---

        # %% [markdown]
        # # Evidence Report: {claim}
        # Started: {datetime.now().isoformat()}
    """)
    path.write_text(header)


def append_to_jupytext_log(
    path: Path,
    code: str,
    output: str,
    *,
    is_markdown: bool = False,
) -> None:
    """Append a code block and its output to the jupytext log."""
    with open(path, "a") as f:
        if is_markdown:
            f.write("\n# %% [markdown]\n")
            for line in code.splitlines():
                f.write(f"# {line}\n")
        else:
            f.write("\n# %%\n")
            f.write(code)
            f.write("\n")
            if output.strip():
                for line in output.strip().splitlines():
                    f.write(f"# → {line}\n")


def generate_notebook(log_path: Path, notebook_path: Path) -> None:
    """Convert the jupytext percent-format .py log to .ipynb.

    We parse the simple percent format ourselves rather than shelling out to
    ``jupytext``, which (as of 1.19.x) leaks cell-boundary markers into cell
    source and cannot represent captured outputs.  Our parser:

    * Splits on ``# %%`` / ``# %% [markdown]`` boundaries.
    * Keeps multi-line bash commands (``python3 -c "…"``) in a single cell
      instead of splitting at blank lines.
    * Moves ``# → …`` output-comment lines into proper ``stream`` outputs.
    """
    import json as _json
    import re as _re

    text = log_path.read_text()

    # Split into raw cell blocks on the ``# %%`` boundary
    # Each match gives (tag, body) where tag is "" or " [markdown]"
    CELL_RE = _re.compile(r"^# %%( \[markdown\])?\s*$", _re.MULTILINE)
    splits = list(CELL_RE.finditer(text))

    cells: list[dict] = []
    OUTPUT_PREFIX = "# → "

    for idx, m in enumerate(splits):
        is_markdown = m.group(1) is not None
        start = m.end()
        end = splits[idx + 1].start() if idx + 1 < len(splits) else len(text)
        body = text[start:end].strip("\n")
        if not body:
            continue

        if is_markdown:
            # Strip leading "# " from each line to recover markdown
            md_lines: list[str] = []
            for line in body.splitlines():
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [l + "\n" for l in md_lines],
            })
        else:
            # Code cell – separate source from ``# → …`` output lines
            code_lines: list[str] = []
            output_lines: list[str] = []
            for line in body.splitlines():
                if line.startswith(OUTPUT_PREFIX) or line == "# →":
                    out = line[len(OUTPUT_PREFIX):] if line.startswith(OUTPUT_PREFIX) else ""
                    output_lines.append(out + "\n")
                else:
                    code_lines.append(line + "\n")

            # Trim trailing blank lines from code
            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()

            cell: dict = {
                "cell_type": "code",
                "metadata": {},
                "source": code_lines,
                "outputs": [],
                "execution_count": None,
            }
            if output_lines:
                cell["outputs"] = [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output_lines,
                }]
                cell["execution_count"] = 1
            cells.append(cell)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    notebook_path.write_text(_json.dumps(nb, indent=1) + "\n")


def serialize_execution_log(path: Path, max_chars: int = 120_000) -> str:
    """Read the log file for context injection, with optional truncation."""
    text = path.read_text()
    if len(text) <= max_chars:
        return text
    # Keep the header + last portion of the log
    header_end = text.find("\n# %%", 100)  # after jupytext header
    if header_end == -1:
        return text[-max_chars:]
    header = text[:header_end]
    tail = text[-(max_chars - len(header) - 50):]
    return header + "\n\n# [... earlier cells truncated ...]\n" + tail


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

def _smart_truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate text with a note so the agent knows data was clipped."""
    if len(text) <= limit:
        return text
    suffix = (
        f"\n[...TRUNCATED — showing {limit} of {len(text)} chars. "
        f"Use read_file on the output file for full content.]"
    )
    return text[: limit - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def dispatch_tool(
    name: str,
    arguments: dict,
    *,
    workspace: Path,
    log_path: Path,
    env: dict,
) -> str:
    """Dispatch a tool call to the appropriate handler."""
    if name == "bash":
        command = arguments.get("command", "")
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(workspace),
                env=env,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            output = output.strip()
        except subprocess.TimeoutExpired:
            output = "[ERROR: Command timed out after 600 seconds]"
        except Exception as e:
            output = f"[ERROR: {e}]"

        append_to_jupytext_log(log_path, command, output)
        return _smart_truncate(output) if output else "(no output)"

    elif name == "read_file":
        fpath = arguments.get("path", "")
        try:
            content = Path(fpath).read_text()
            truncated = _smart_truncate(content)
            append_to_jupytext_log(log_path, f"cat {fpath}", truncated)
            return truncated
        except Exception as e:
            error_msg = f"[ERROR reading {fpath}: {e}]"
            append_to_jupytext_log(log_path, f"cat {fpath}", error_msg)
            return error_msg

    return f"[ERROR: Unknown tool '{name}']"


# ---------------------------------------------------------------------------
# Stopping condition
# ---------------------------------------------------------------------------

def should_stop(workspace: Path, threshold: float) -> bool:
    """Check if verification should stop."""
    verdict_path = workspace / "verdict.json"
    if verdict_path.exists():
        return True

    state_path = workspace / "evidence_state.json"
    if not state_path.exists():
        return False

    from pkevolve.verification.evidence_state import EvidenceState
    state = EvidenceState.load(state_path)

    if not state.sufficiency_history:
        return False

    last = state.sufficiency_history[-1]
    return last.confidence >= threshold


# ---------------------------------------------------------------------------
# Build environment for subprocess
# ---------------------------------------------------------------------------

def build_subprocess_env(cfg) -> dict:
    """Build environment variables for bash subprocess calls.

    These env vars are read by setup_workspace() inside the subprocess.
    """
    env = {**os.environ}

    # Subagent LLM config (for evidence API calls inside bash)
    env["LLM_BASE_URL"] = cfg.llm.subagent_base_url
    env["LLM_API_KEY"] = cfg.api_key
    env["LLM_MODEL"] = cfg.subagent_model
    env["LLM_TEMPERATURE"] = str(cfg.llm.temperature)
    env["LLM_DISABLE_THINKING"] = "1" if cfg.llm.disable_thinking else "0"
    env["MLP_MODEL_DIR"] = cfg.mlp_model_dir or "results/models/classifier_best"
    env["SUFFICIENCY_BACKEND"] = cfg.sufficiency_backend
    env["MAX_ITERATIONS"] = str(cfg.max_iterations)
    env["LABEL_CONFIG_JSON"] = cfg.labels.model_dump_json()
    env["NB_MAX_OUTPUT_CHARS"] = str(cfg.max_output_chars)

    # Debug mode for evidence API tools
    env["EVIDENCE_DEBUG"] = "1" if cfg.verbose else "0"

    # Ensure PYTHONPATH includes src
    src_dir = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    if src_dir not in existing:
        env["PYTHONPATH"] = f"{src_dir}:{existing}" if existing else src_dir

    return env


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def verify_claim_direct(cfg) -> Path:
    """Run evidence programming via LiteLLM direct API with context refresh."""
    import litellm

    workspace = cfg.resolved_workspace
    output_dir = cfg.resolved_output_dir
    claim = cfg.claim

    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evidence state on disk
    from pkevolve.verification.evidence_state import EvidenceState
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Build system prompt
    from pkevolve.verification.evidence_api import schema_docs, function_docs
    label_cfg = cfg.labels
    system_prompt = SYSTEM_PROMPT.format(
        workspace=str(workspace.resolve()),
        claim=claim,
        max_iterations=cfg.max_iterations,
        sufficiency_threshold=cfg.sufficiency_threshold,
        schemas=schema_docs(),
        function_docs=function_docs(),
        verdict_names=", ".join(label_cfg.verdict_names()),
        verdict_definitions=label_cfg.verdict_prompt_block(),
    )

    # Jupytext execution log
    log_path = workspace / "execution_log.py"
    init_jupytext_log(log_path, claim)

    # Tools and environment
    tools = build_tool_schemas()
    sub_env = build_subprocess_env(cfg)

    # Outer agent model (via LiteLLM)
    agent_model = cfg.llm.model  # e.g. "anthropic/claude-sonnet-4-20250514"

    # Anthropic prompt caching: mark the system message and the last user
    # message with cache_control so repeated turns reuse cached prefixes.
    # LiteLLM passes this through to Anthropic's API.  For non-Anthropic
    # models the extra key is silently ignored.
    _use_cache = agent_model.startswith("anthropic/")

    def _cached_system_msg(text: str) -> dict:
        if _use_cache:
            return {
                "role": "system",
                "content": [{"type": "text", "text": text,
                             "cache_control": {"type": "ephemeral"}}],
            }
        return {"role": "system", "content": text}

    def _cached_user_msg(text: str) -> dict:
        if _use_cache:
            return {
                "role": "user",
                "content": [{"type": "text", "text": text,
                             "cache_control": {"type": "ephemeral"}}],
            }
        return {"role": "user", "content": text}

    # Token usage tracking
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }

    logger.info(
        "Starting direct API verification: claim=%r, model=%s, workspace=%s",
        claim, agent_model, workspace,
    )

    # Single flat loop with per-call context refresh.
    # Each LLM call receives only [system_prompt, execution_log].
    # Tool results flow through the log — no conversation accumulation.
    max_calls = cfg.max_iterations * cfg.max_turns
    call_count = 0

    while call_count < max_calls:
        # *** CONTEXT REFRESH: rebuild messages from log before every call ***
        execution_log = serialize_execution_log(log_path)

        if call_count == 0:
            user_text = (
                f"Verify the following scientific claim using evidence programming.\n\n"
                f"Claim: {claim}\n\n"
                f"Start by calling bash with the setup code to import the evidence API. "
                f"Follow the evidence programming workflow. "
                f"Call check_sufficiency after each round. "
                f"Stop when confidence >= {cfg.sufficiency_threshold} or after "
                f"{cfg.max_iterations} iterations."
            )
        else:
            user_text = (
                f"<execution_log>\n{execution_log}\n</execution_log>\n\n"
                f"Continue evidence verification for claim: {claim}\n"
                f"The execution log above shows all prior work. "
                f"Continue the workflow: run code, check sufficiency, "
                f"address remaining gaps, or emit verdict if ready. "
                f"Stop when confidence >= {cfg.sufficiency_threshold}."
            )

        messages = [
            _cached_system_msg(system_prompt),
            _cached_user_msg(user_text),
        ]

        # --- LLM call ---
        try:
            response = litellm.completion(
                model=agent_model,
                messages=messages,
                tools=tools,
                max_tokens=16384,
            )
        except Exception as e:
            logger.error("LiteLLM API error: %s", e)
            time.sleep(5)
            try:
                response = litellm.completion(
                    model=agent_model,
                    messages=messages,
                    tools=tools,
                    max_tokens=16384,
                )
            except Exception as e2:
                logger.error("LiteLLM retry failed: %s", e2)
                break

        call_count += 1

        # Track usage
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            total_usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            total_usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            total_usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0
            total_usage["cache_read_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
            total_usage["cache_creation_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0

        choice = response.choices[0]
        assistant_msg = choice.message

        # Log agent reasoning to execution log
        if assistant_msg.content:
            logger.info("Agent [call %d]: %s", call_count, assistant_msg.content[:300])
            append_to_jupytext_log(
                log_path,
                assistant_msg.content[:2000],
                "",
                is_markdown=True,
            )

        # No tool calls — agent paused or finished
        if choice.finish_reason == "stop" or not assistant_msg.tool_calls:
            if should_stop(workspace, cfg.sufficiency_threshold):
                logger.info("Stopping: verdict emitted or sufficiency reached.")
                break
            logger.info("Agent stopped without verdict (call %d). Re-prompting.", call_count)
            continue

        # Dispatch tool calls — results are appended to execution log
        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {"command": tc.function.arguments}

            logger.info("Tool [call %d]: %s(%s)", call_count, fn_name, str(fn_args)[:200])

            result = dispatch_tool(
                fn_name,
                fn_args,
                workspace=workspace,
                log_path=log_path,
                env=sub_env,
            )

            logger.info("Tool result: %s", result[:200])

        # Check stopping after tool dispatch
        if should_stop(workspace, cfg.sufficiency_threshold):
            logger.info("Stopping: verdict emitted or sufficiency reached.")
            break

    else:
        logger.info("Reached max calls (%d).", max_calls)

    # Log final usage
    logger.info("Total token usage: %s", json.dumps(total_usage))

    # Save usage stats
    usage_path = output_dir / "token_usage.json"
    usage_path.write_text(json.dumps(total_usage, indent=2))

    # Generate notebook from jupytext log
    notebook_path = output_dir / "evidence_report.ipynb"
    try:
        generate_notebook(log_path, notebook_path)
        logger.info("Notebook generated: %s", notebook_path)
    except Exception as e:
        logger.warning("Failed to generate notebook (jupytext): %s", e)
        notebook_path = log_path  # fall back to .py log

    # Print verdict if available
    verdict_path = workspace / "verdict.json"
    if verdict_path.exists():
        from pkevolve.verification.data_models import VerificationVerdict
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
    from pkevolve.verification.config import VerificationSettings

    cfg = VerificationSettings.from_cli()

    output_dir = cfg.resolved_output_dir
    workspace = cfg.resolved_workspace

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
    print(f"Agent model: {cfg.llm.model}")
    print(f"Subagent model: {cfg.subagent_model}")
    print(f"Workspace: {workspace}")
    print()

    result = verify_claim_direct(cfg)
    print(f"\nOutput: {result}")
    print(f"Workspace: {workspace}")


if __name__ == "__main__":
    main()
