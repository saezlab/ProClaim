#!/usr/bin/env python3
"""
Evidence Programming — Direct API with per-call context refresh.

Uses LiteLLM for model-agnostic LLM completion, bash for code execution,
and jupytext for post-hoc notebook generation.  No Jupyter kernel, no MCP
server, no Claude Agent SDK.

Architecture:
  - Single flat loop: each LLM call gets [system_prompt, workbook + instruction]
  - Per-call context refresh: a compact markdown workbook is regenerated from
    durable EvidenceState before every LLM call — no conversation accumulation
  - Tool results flow into evidence_state.json (and into execution_log.py as
    an audit artifact); the planner reads the workbook, not the raw log
  - State persistence: EvidenceState auto-saves to disk after every mutation

Usage:
  uv run python -m proclaim.verification.evidence_programming_direct \\
      --config experiments/config.yaml \\
      --claim "Does MAPK1 directly activate H3-3A?"
"""

import json
import logging
import os
import shlex
import subprocess
import sys
import textwrap
import threading
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

from proclaim.verification.prompts import DIRECT_SYSTEM_PROMPT as SYSTEM_PROMPT, SUBCLAIM_EXAMPLES


# ---------------------------------------------------------------------------
# Web search helpers (Serper → DuckDuckGo fallback)
# ---------------------------------------------------------------------------

_WEB_SEARCH_LOCK = threading.Lock()  # ddgs hangs on concurrent calls
_SERPER_URL = "https://google.serper.dev"


def _serper_search(query: str, api_key: str, k: int = 5) -> str:
    import requests as _requests
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    resp = _requests.post(
        f"{_SERPER_URL}/search",
        headers=headers,
        params={"q": query, "num": k, "gl": "us", "hl": "en"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    snippets: list[str] = []
    if data.get("answerBox"):
        ab = data["answerBox"]
        for field in ("answer", "snippet"):
            val = ab.get(field)
            if isinstance(val, str):
                snippets.append(val.replace("\n", " "))
    if data.get("knowledgeGraph", {}).get("description"):
        snippets.append(data["knowledgeGraph"]["description"])
    for item in data.get("organic", [])[:k]:
        if "snippet" in item:
            snippets.append(item["snippet"])
    return "\n".join(snippets) if snippets else "No search results."


def _ddg_search(query: str, k: int = 5) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "[ERROR: ddgs not installed. Run: pip install ddgs]"
    snippets: list[str] = []
    try:
        with _WEB_SEARCH_LOCK:
            for result in DDGS().text(query, max_results=k):
                body = result.get("body", "")
                if body:
                    snippets.append(body)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
    return "\n".join(snippets) if snippets else "No search results."


def _do_web_search(query: str, k: int = 5) -> str:
    """Run web search via Serper (if key set) or DuckDuckGo fallback."""
    serper_key = os.environ.get("SERPER_API_KEY", "")
    for attempt in range(3):
        try:
            if serper_key:
                return _serper_search(query, serper_key, k=k)
            return _ddg_search(query, k=k)
        except Exception as exc:
            wait = 2 ** attempt * 2
            logger.warning("Web search error (attempt %d/3), retrying in %ds: %s", attempt + 1, wait, exc)
            time.sleep(wait)
    return "Web search temporarily unavailable."


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format, used by LiteLLM)
# ---------------------------------------------------------------------------

def build_tool_schemas(disable_web_search: bool = False) -> list[dict]:
    """Build tool schemas in OpenAI function calling format."""
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": (
                    "Execute Python code in the workspace. The evidence-API "
                    "functions (setup_workspace, search_pubmed_llm, "
                    "extract_and_add_facts, populate_paper_features, "
                    "check_sufficiency, emit_verdict, ...) are importable "
                    "from proclaim.verification.evidence_api. State does NOT "
                    "persist between calls — re-import and reload state via "
                    "setup_workspace every call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source to execute.",
                        },
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": (
                    "Execute a shell command. Use the python tool to run "
                    "Python code; this is for shell operations like ls, "
                    "cat, find. Working directory is the workspace."
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
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for scientific evidence. Use specific queries "
                    "targeting the entities and relationships in the claim. "
                    "Returns snippets from top results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]
    if disable_web_search:
        schemas = [s for s in schemas if s["function"]["name"] != "web_search"]
    return schemas


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


def _unwrap_python_c_command(command: str) -> str | None:
    """Return embedded Python source from ``python -c`` shell commands."""
    stripped = command.strip()
    if not stripped:
        return None

    try:
        parts = shlex.split(stripped, posix=True)
    except ValueError:
        return None

    if "&&" in parts:
        parts = parts[parts.index("&&") + 1 :]

    if len(parts) < 3 or parts[1] != "-c":
        return None

    executable = Path(parts[0]).name
    if executable != "python" and not executable.startswith("python3"):
        return None

    return parts[2].strip("\n")


def _normalize_jupytext_code(code: str) -> str:
    """Normalize logged code so notebook cells contain Python, not shell wrappers."""
    return _unwrap_python_c_command(code) or code


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
            normalized_code = _normalize_jupytext_code(code)
            f.write("\n# %%\n")
            f.write(normalized_code)
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
        * Unwraps embedded ``python -c`` / ``python3 -c`` commands into notebook
            Python cells.
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

            normalized_code = _normalize_jupytext_code("".join(code_lines)).rstrip("\n")
            code_lines = [line + "\n" for line in normalized_code.splitlines()] if normalized_code else []

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
    if name == "python":
        code = arguments.get("code", "")
        try:
            result = subprocess.run(
                ["python3", "-c", code],
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
            output = "[ERROR: python command timed out after 600 seconds]"
        except Exception as e:
            output = f"[ERROR: {e}]"

        append_to_jupytext_log(log_path, code, output)
        return _smart_truncate(output) if output else "(no output)"

    elif name == "bash":
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

    elif name == "web_search":
        query = arguments.get("query", "")
        k = int(arguments.get("num_results", 5))
        try:
            result = _do_web_search(query, k=k)
        except Exception as e:
            result = f"[ERROR: web search failed: {e}]"
        truncated = _smart_truncate(result)
        append_to_jupytext_log(log_path, f"web_search({query!r})", truncated)
        return truncated

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

    from proclaim.verification.evidence_state import EvidenceState
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
# Forced verdict helper
# ---------------------------------------------------------------------------

def _force_verdict(workspace: Path, claim: str, sub_env: dict) -> None:
    """Force check_sufficiency + LLM-guided emit_verdict when the turn budget is exhausted.

    Mirrors what the agent would do in its final turn: runs check_sufficiency, then
    prompts the subagent LLM to evaluate the evidence and decide SUPPORT/REFUTE/UNCERTAIN,
    then calls emit_verdict.  No hardcoded verdict defaults — the LLM makes the call.
    """
    abs_workspace = str(workspace.resolve())
    script = textwrap.dedent(f"""\
        from proclaim.verification.evidence_api import (
            setup_workspace, populate_paper_features, check_sufficiency, emit_verdict,
            get_evidence_summary,
        )
        from proclaim.verification.config import get_label_config

        state, llm, workspace = setup_workspace(
            claim={claim!r},
            workspace_path={abs_workspace!r},
        )
        print(f"Forced verdict: {{len(state.papers)}} papers, {{len(state.facts)}} facts, iteration={{state.iteration}}")

        populate_paper_features(state)

        # Run or reuse sufficiency check for gaps.
        if state.sufficiency_history:
            suf = state.sufficiency_history[-1]
            print(f"Reusing last sufficiency: {{suf.label}} (confidence={{suf.confidence:.3f}})")
        else:
            suf = check_sufficiency(state, llm)

        gaps = [g.description for g in (suf.gaps if suf else [])]
        suf_confidence = suf.confidence if suf else 0.0

        # Prompt the subagent LLM to evaluate evidence and decide the verdict.
        label_cfg = get_label_config()
        summary = get_evidence_summary(state)
        facts_text = "\\n".join(
            f"  [{{f.stance}}] {{f.text[:200]}}" for f in state.facts[:20]
        ) or "  (none)"

        from proclaim.verification.prompts import FORCE_VERDICT_PROMPT
        gaps_text = chr(10).join(f"  - {{g}}" for g in gaps[:5]) or "  (none)"
        verdict_prompt = FORCE_VERDICT_PROMPT.format(
            claim=state.claim,
            verdict_definitions=label_cfg.verdict_prompt_block(),
            summary=summary,
            facts_text=facts_text,
            gaps_text=gaps_text,
            verdict_names=", ".join(label_cfg.verdict_names()),
        )

        response = llm(verdict_prompt)
        print("LLM verdict response:", response[:600])

        # Parse the LLM's structured response.
        verdict_label = None
        confidence = suf_confidence
        reasoning = None
        key_evidence = []

        for line in response.splitlines():
            line = line.strip()
            if line.startswith("VERDICT:"):
                raw = line.split(":", 1)[1].strip().upper()
                if raw in label_cfg.verdict_names():
                    verdict_label = raw
                else:
                    print(f"WARNING: LLM returned unrecognised verdict label: {{raw!r}}")
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_EVIDENCE:"):
                key_evidence = [e.strip() for e in line.split(":", 1)[1].split("|") if e.strip()]

        if verdict_label is None or reasoning is None:
            raise RuntimeError(
                f"Failed to parse LLM verdict response. Raw response:\\n{{response}}"
            )

        print(f"Parsed verdict: {{verdict_label}} (confidence={{confidence:.3f}})")

        emit_verdict(
            verdict=verdict_label,
            confidence=confidence,
            reasoning=reasoning,
            key_evidence=key_evidence[:5],
            gaps_remaining=gaps[:5],
            state=state,
            workspace=workspace,
        )
    """)

    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            tf.write(script)
            tmp_path = tf.name
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=sub_env,
        )
        output = (result.stdout + ("\n" + result.stderr if result.stderr else "")).strip()
    except Exception as exc:
        output = f"[ERROR: forced verdict script failed: {exc}]"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    logger.info("Forced verdict")


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
    from proclaim.verification.evidence_state import EvidenceState
    from proclaim.verification.workbook import (
        STABLE_VOLATILE_MARKER as WORKBOOK_STABLE_MARKER,
        build_workbook,
        write_workbook,
    )
    from proclaim.verification.action_ledger import (
        record_action as _record_action,
        snapshot_state as _snapshot_state,
        StateSnapshot as _StateSnapshot,
    )
    from proclaim.verification.curation import run_auto_curation
    from proclaim.verification.reflection import (
        detect_stall_signals,
        run_reflection,
    )
    from proclaim.verification.workbook import build_workbook_parts
    EvidenceState.init_new(claim=claim, subclaims=[claim], workspace=workspace)

    # Build system prompt
    from proclaim.verification.evidence_api import schema_docs, function_docs
    label_cfg = cfg.labels
    _web_search_step = (
        ""
        if cfg.disable_web_search
        else (
            "   d. web_search(query) — call this tool directly (NOT via python or bash) to search\n"
            "      the web for evidence not found in PubMed/S2; use when academic databases\n"
            "      return few results or for recent findings not yet indexed."
        )
    )
    system_prompt = SYSTEM_PROMPT.format(
        workspace=str(workspace.resolve()),
        claim=claim,
        max_iterations=cfg.max_iterations,
        sufficiency_threshold=cfg.sufficiency_threshold,
        schemas=schema_docs(),
        function_docs=function_docs(),
        verdict_names=", ".join(label_cfg.verdict_names()),
        verdict_definitions=label_cfg.verdict_prompt_block(),
        subclaim_examples=SUBCLAIM_EXAMPLES if cfg.include_subclaim_examples else "",
        web_search_step=_web_search_step,
    )

    # Jupytext execution log
    log_path = workspace / "execution_log.py"
    init_jupytext_log(log_path, claim)

    # Tools and environment
    tools = build_tool_schemas(disable_web_search=cfg.disable_web_search)
    sub_env = build_subprocess_env(cfg)

    # Outer agent model (via LiteLLM)
    agent_model = cfg.llm.model  # e.g. "anthropic/claude-sonnet-4-20250514"

    # Anthropic extended thinking: pass thinking block when budget_tokens > 0.
    # Requires temperature=1 per Anthropic API requirements.
    _thinking_budget = cfg.llm.thinking_budget_tokens
    _thinking_kwargs: dict = {}
    if _thinking_budget > 0:
        _thinking_kwargs["thinking"] = {"type": "enabled", "budget_tokens": _thinking_budget}
        _thinking_kwargs["temperature"] = 1


    # Anthropic prompt caching: mark the system message and the last user
    # message with cache_control so repeated turns reuse cached prefixes.
    # LiteLLM passes this through to Anthropic's API.  For non-Anthropic
    # models the extra key is silently ignored.
    # Note: prompt caching is disabled when thinking is active (Anthropic restriction).
    _use_cache = agent_model.startswith("anthropic/") and _thinking_budget == 0

    def _cached_system_msg(text: str) -> dict:
        if _use_cache:
            return {
                "role": "system",
                "content": [{"type": "text", "text": text,
                             "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
            }
        return {"role": "system", "content": text}

    def _cached_user_msg(text: str) -> dict:
        """Split the user message at the workbook stable/volatile marker.

        The workbook places invariant sections (Claim Frame + Guardrails)
        before the marker and turn-variable sections (Header, State,
        Action Loop, Recuration, Verdict Readiness, Artifact Index) after.
        A single cache breakpoint at the marker lets every turn reuse the
        stable prefix cache (claim + guardrails + system prompt), while the
        volatile tail is re-sent each turn.
        """
        if not _use_cache:
            return {"role": "user", "content": text}

        marker = WORKBOOK_STABLE_MARKER
        idx = text.find(marker)
        if idx == -1:
            return {"role": "user", "content": text}

        # Cache breakpoint sits just after the marker line so the entire
        # marker + newline are part of the cached prefix.
        split = idx + len(marker) + 1  # +1 for the trailing newline
        if split < 500 or split >= len(text) - 200:
            return {"role": "user", "content": text}

        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text[:split],
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "type": "text",
                    "text": text[split:],
                },
            ],
        }

    # Token usage tracking
    from proclaim.verification.cost_tracker import CostTracker
    tracker = CostTracker(model=agent_model)

    logger.info(
        "Starting direct API verification: claim=%r, model=%s, workspace=%s",
        claim, agent_model, workspace,
    )

    # Single flat loop with per-call context refresh.
    # Each LLM call receives [system_prompt, workbook + instruction].
    # The execution log keeps growing on disk for audit, but the planner now
    # reads the compact workbook regenerated each turn from EvidenceState.
    # ``max_calls`` is the hard kill-switch on total LLM calls.  ``planning_budget``
    # is the realistic per-claim planning horizon that drives verdict-readiness
    # gating (the URGENT warning + workbook status).  Keying gating off max_calls
    # (= max_iterations * max_turns, e.g. 240) meant the "wrap up now" gate never
    # fired until the very last calls; gating off max_turns warns the planner at a
    # realistic point while max_calls still backstops a runaway loop.
    max_calls = cfg.max_iterations * cfg.max_turns
    planning_budget = cfg.max_turns
    call_count = 0
    # Per-iteration turn tracking.  ``call_count`` is the GLOBAL planning-call
    # accumulator (never reset); it only backstops runaway loops via max_calls.
    # The verdict/sufficiency warnings instead key off how many turns have been
    # spent *within the current iteration* — i.e. since the last
    # check_sufficiency, which is what advances ``state.iteration``.
    # ``iteration_turn`` counts turns in the current iteration; ``last_seen_iter``
    # detects the iteration rollover so we can reset it.
    iteration_turn = 0
    last_seen_iter = 0

    while call_count < max_calls:
        turns_remaining = max_calls - call_count  # hard-ceiling log only

        # Detect iteration rollover (a check_sufficiency since last turn bumped
        # state.iteration) and reset the per-iteration turn counter.
        try:
            _cur_iter = EvidenceState.load(workspace / "evidence_state.json").iteration
        except Exception:
            _cur_iter = last_seen_iter
        if _cur_iter != last_seen_iter:
            iteration_turn = 0
            last_seen_iter = _cur_iter

        # Turns remaining within THIS iteration (per-iteration budget).
        iter_turns_remaining = max(planning_budget - iteration_turn, 0)
        # True once we are in the final iteration (0-indexed: check_sufficiency
        # stops once state.iteration reaches max_iterations).
        is_last_iteration = _cur_iter >= cfg.max_iterations - 1

        # Phase 3: refresh the auto-populated curation buckets (gaps from
        # latest sufficiency check, conflicts, zero-fact papers) before the
        # workbook is rendered.  Failures are non-fatal — the workbook will
        # still render with whatever queue is on disk.
        try:
            run_auto_curation(workspace)
        except Exception as exc:
            logger.debug("auto-curation refresh failed: %s", exc)

        # Phase 4: per-turn reflection.  The reflect LLM runs every turn
        # (skipped only on the very first call, before any state exists).
        # It reads the volatile workbook tail and writes Section 9's
        # recommendation to ``reflection.json``.  Stall signals are passed
        # as context so the reflect LLM can switch to diagnostic mode when
        # they fire, but they no longer gate the call.
        state_file = workspace / "evidence_state.json"
        if state_file.exists():
            try:
                signals = detect_stall_signals(workspace)
            except Exception as exc:
                logger.debug("stall-signal detection failed: %s", exc)
                from proclaim.verification.reflection import StallSignals
                signals = StallSignals()

            try:
                current_state = EvidenceState.load(state_file)
                _, volatile_tail = build_workbook_parts(
                    current_state,
                    workspace=workspace,
                    max_turns=planning_budget,
                    turns_used=iteration_turn,
                    sufficiency_threshold=cfg.sufficiency_threshold,
                    max_iterations=cfg.max_iterations,
                    label_cfg=label_cfg,
                )
            except Exception as exc:
                logger.debug("could not render volatile tail for reflect: %s", exc)
                volatile_tail = ""

            _reflect_t0 = time.monotonic()
            try:
                reflect_record, reflect_usage = run_reflection(
                    workspace=workspace,
                    workbook_volatile=volatile_tail,
                    stall_signals=signals,
                    turn=call_count,
                    model=agent_model,
                )
            except Exception as exc:
                logger.warning("Reflect LLM call failed: %s", exc)
                reflect_record, reflect_usage = None, None
            _reflect_latency = time.monotonic() - _reflect_t0

            # Bill the reflect LLM call into the same tracker as the planner.
            # Tokens are recorded under a distinct "reflect_llm" action so the
            # trace stays separable, while the aggregate token/cost totals fold
            # it in automatically.
            if reflect_usage:
                tracker.record(
                    "reflect_llm",
                    input_tokens=reflect_usage.get("prompt_tokens", 0) or 0,
                    output_tokens=reflect_usage.get("completion_tokens", 0) or 0,
                    latency=_reflect_latency,
                    call_number=call_count,
                    cache_read_tokens=reflect_usage.get("cache_read_input_tokens", 0) or 0,
                    cache_write_tokens=reflect_usage.get("cache_creation_input_tokens", 0) or 0,
                )

            if reflect_record is not None:
                logger.info(
                    "Reflection recorded (turn %d): classification=%s next_family=%s override=%s (%.2fs)",
                    call_count,
                    reflect_record.classification.value,
                    reflect_record.proposed_next_family.value,
                    reflect_record.override_invoked or "-",
                    _reflect_latency,
                )
                try:
                    _reflect_snap = _snapshot_state(current_state)
                    _record_action(
                        workspace=workspace,
                        turn=call_count,
                        tool_name="reflect",
                        arguments={
                            "classification": reflect_record.classification.value,
                            "proposed_next_family": reflect_record.proposed_next_family.value,
                            "override_invoked": reflect_record.override_invoked or None,
                        },
                        raw_output=reflect_record.diagnosis,
                        before=_reflect_snap,
                        after=_reflect_snap,
                    )
                except Exception as _refl_exc:
                    logger.debug("Failed to record reflection in ledger: %s", _refl_exc)

        # *** CONTEXT REFRESH: regenerate the workbook from durable state ***
        # The workbook is the planner's primary context.  It is rebuilt from
        # EvidenceState on every turn so the planner sees the current paper /
        # fact / sufficiency snapshot without scrolling the execution log.
        try:
            current_state = EvidenceState.load(workspace / "evidence_state.json")
            workbook_path = write_workbook(
                current_state,
                workspace=workspace,
                max_turns=planning_budget,
                turns_used=iteration_turn,
                sufficiency_threshold=cfg.sufficiency_threshold,
                max_iterations=cfg.max_iterations,
                label_cfg=label_cfg,
            )
            workbook_text = workbook_path.read_text()
        except Exception as exc:
            # Defensive: never let workbook generation kill the loop.  Fall
            # back to a minimal stub so the planner still gets a valid prompt.
            logger.warning("Workbook generation failed: %s", exc)
            workbook_text = (
                f"# ProClaim Workbook\n\n"
                f"(workbook unavailable: {exc})\n\n"
                f"{WORKBOOK_STABLE_MARKER}\n\n"
                f"## 3. Header\n- Claim: {claim}\n- Iteration: {_cur_iter}\n"
                f"- Turns remaining this iteration: {iter_turns_remaining}\n"
            )

        if call_count == 0:
            user_text = (
                f"Verify the following scientific claim using evidence programming.\n\n"
                f"Claim: {claim}\n\n"
                f"<workbook>\n{workbook_text}\n</workbook>\n\n"
                f"Start by calling the python tool with the setup code to import the evidence API. "
                f"Follow the evidence programming workflow. "
                f"Call check_sufficiency after each round. "
                f"Stop when confidence >= {cfg.sufficiency_threshold} or after "
                f"{cfg.max_iterations} iterations."
            )
        elif is_last_iteration and iter_turns_remaining <= 2:
            # Final iteration AND its turn budget is nearly exhausted: wrap up the
            # whole run now — sufficiency check then verdict.
            user_text = (
                f"<workbook>\n{workbook_text}\n</workbook>\n\n"
                f"URGENT — final iteration, only {iter_turns_remaining} turn(s) "
                f"remaining in this iteration.\n"
                f"You MUST call check_sufficiency and then emit_verdict NOW.\n"
                f"Do NOT run any more searches or extractions.\n"
                f"Claim: {claim}"
            )
        elif iter_turns_remaining <= 2:
            # Earlier iteration whose per-iteration turn budget is nearly spent:
            # nudge toward closing out THIS iteration with a sufficiency check.
            # A verdict is not forced — further iterations remain.
            user_text = (
                f"<workbook>\n{workbook_text}\n</workbook>\n\n"
                f"Only {iter_turns_remaining} turn(s) remaining in this iteration "
                f"(iteration {_cur_iter} of {cfg.max_iterations}).\n"
                f"Call check_sufficiency now to close out this iteration before "
                f"the turn budget is exhausted.\n"
                f"Do NOT start new searches or extractions this iteration.\n"
                f"Claim: {claim}"
            )
        else:
            user_text = (
                f"<workbook>\n{workbook_text}\n</workbook>\n\n"
                f"Continue evidence verification for claim: {claim}\n"
                f"The workbook above is the compact planning surface derived "
                f"from durable state. Use it to decide the next single action: "
                f"search, extract, populate features, filter, check sufficiency, "
                f"or emit a verdict if ready. "
                f"State your Expected observation and Abort condition, then make "
                f"exactly ONE tool call. "
                f"Stop when confidence >= {cfg.sufficiency_threshold}."
            )

        messages = [
            _cached_system_msg(system_prompt),
            _cached_user_msg(user_text),
        ]

        # --- LLM call ---
        _t0 = time.monotonic()
        try:
            response = litellm.completion(
                model=agent_model,
                messages=messages,
                tools=tools,
                max_tokens=16384,
                **_thinking_kwargs,
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
                    **_thinking_kwargs,
                )
            except Exception as e2:
                logger.error("LiteLLM retry failed: %s", e2)
                break
        _latency = time.monotonic() - _t0

        call_count += 1
        iteration_turn += 1  # turns spent in the current iteration (reset on rollover)

        choice = response.choices[0]
        assistant_msg = choice.message

        _tool_calls_trace = []
        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                _tool_calls_trace.append({
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        # Track usage
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            tracker.record(
                "llm_call",
                input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
                latency=_latency,
                call_number=call_count,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                system_prompt=system_prompt,
                user_message=user_text,
                assistant_response=assistant_msg.content or "",
                tool_calls=_tool_calls_trace,
            )

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
                logger.info(
                    "Stopping: verdict emitted or sufficiency reached."
                )
                break
            logger.info("Agent stopped without verdict (call %d). Re-prompting.", call_count)
            continue

        # GR6: one next action only.  The planner sometimes batches several
        # tool calls in a single turn; we honour the bounded-ReAct contract by
        # dispatching only the first (its highest-priority intended action) and
        # dropping the rest.  A zero-delta ledger note keeps the audit trail
        # honest and lets the planner see, on its next turn, that extra calls
        # were ignored.  The constraint is on the *number of tool calls*, not
        # the Python statements inside one cell — a single python cell with
        # setup + action is fine.  emit_verdict is a Python evidence-API call
        # inside a python cell, not a top-level tool, so this never blocks it.
        dispatched = assistant_msg.tool_calls[:1]
        dropped = assistant_msg.tool_calls[1:]
        if dropped:
            logger.warning(
                "GR6: planner emitted %d tool calls; dispatching only the first "
                "(%s), dropping %d.",
                len(assistant_msg.tool_calls),
                dispatched[0].function.name,
                len(dropped),
            )
            try:
                state_path = workspace / "evidence_state.json"
                if state_path.exists():
                    _gr6_snap = _snapshot_state(EvidenceState.load(state_path))
                else:
                    _gr6_snap = _StateSnapshot()
                _record_action(
                    workspace=workspace,
                    turn=call_count,
                    tool_name="gr6_enforcement",
                    arguments={"dropped": [tc.function.name for tc in dropped]},
                    raw_output=(
                        f"GR6: dropped {len(dropped)} extra tool call(s); "
                        "one action per turn."
                    ),
                    before=_gr6_snap,
                    after=_gr6_snap,  # zero delta — this is a policy note, not an action
                )
            except Exception as _gr6_exc:
                logger.debug("GR6 ledger note failed: %s", _gr6_exc)

        # Dispatch tool calls — results are appended to execution log
        for tc in dispatched:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {"command": tc.function.arguments}

            logger.info("Tool [call %d]: %s(%s)", call_count, fn_name, str(fn_args)[:200])

            # Pre-action snapshot of durable state.  Reads
            # evidence_state.json from disk so we capture mutations made by
            # earlier tool calls in this same turn.  An empty snapshot is
            # used when the file does not exist yet (very first call).
            state_path = workspace / "evidence_state.json"
            try:
                if state_path.exists():
                    _before_snap = _snapshot_state(EvidenceState.load(state_path))
                else:
                    _before_snap = _StateSnapshot()
            except Exception as _snap_exc:  # never let snapshotting kill the run
                logger.debug("Pre-action snapshot failed: %s", _snap_exc)
                _before_snap = _StateSnapshot()

            result = dispatch_tool(
                fn_name,
                fn_args,
                workspace=workspace,
                log_path=log_path,
                env=sub_env,
            )

            logger.info("Tool result: %s", result[:200])

            # Post-action snapshot + ledger append.  The ledger writer
            # spills large outputs to ``workspace/artifacts/`` and records
            # only a compact summary + delta in the JSONL row.
            try:
                if state_path.exists():
                    _after_snap = _snapshot_state(EvidenceState.load(state_path))
                else:
                    _after_snap = _StateSnapshot()
                _record_action(
                    workspace=workspace,
                    turn=call_count,
                    tool_name=fn_name,
                    arguments=fn_args,
                    raw_output=result or "",
                    before=_before_snap,
                    after=_after_snap,
                )
            except Exception as _led_exc:
                logger.warning("Failed to record action ledger entry: %s", _led_exc)

        # Check stopping after tool dispatch.
        # Only stop on an emitted verdict here — sufficiency alone is not enough
        # because the agent still needs one more turn to call deliver_verdict.
        verdict_path = workspace / "verdict.json"
        if verdict_path.exists():
            logger.info("Stopping: verdict emitted.")
            break

    else:
        logger.info("Reached max calls (%d).", max_calls)

    # If the loop exited without a verdict, force one from the current state.
    verdict_path = workspace / "verdict.json"
    if not verdict_path.exists():
        logger.info("No verdict emitted — forcing check_sufficiency + emit_verdict.")
        _force_verdict(workspace, claim, sub_env)

    # Log final usage
    summary = tracker.summary()
    logger.info("Total token usage: %s", json.dumps(summary))

    # Save usage stats
    usage_path = output_dir / "token_usage.json"
    usage_path.write_text(json.dumps(summary, indent=2))

    trace_path = output_dir / "token_usage_trace.json"
    trace_path.write_text(json.dumps(tracker.trace_as_dicts(), indent=2))

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
        from proclaim.verification.data_models import VerificationVerdict
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
    from proclaim.verification.config import VerificationSettings

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
