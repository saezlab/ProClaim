"""
MCP Tool Server for Jupyter Notebook management.

Provides tools for the orchestrator agent to write evidence programming
results into a Jupyter notebook in real-time. Each tool call adds cells
to the notebook and (optionally) executes code in a persistent kernel.

Kernel lifecycle is managed by KernelRunner. This module handles only
the notebook file (nbformat) layer and MCP tool wrappers.

Launch:  python -m pkevolve.verification.notebook_mcp
Connect: Claude Agent SDK connects via stdio transport.
"""

import logging
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from mcp.server.fastmcp import FastMCP
from pkevolve.verification.kernel_runner import KernelRunner

logger = logging.getLogger(__name__)

mcp = FastMCP("notebook-tools")

# ---------------------------------------------------------------------------
# Global state: in-memory notebooks and kernel runners
# ---------------------------------------------------------------------------

_notebooks: dict[str, nbformat.NotebookNode] = {}
_runners: dict[str, KernelRunner] = {}  # notebook_path -> KernelRunner


def _get_notebook(notebook_path: str) -> nbformat.NotebookNode:
    """Get or create the in-memory notebook for a given path."""
    if notebook_path not in _notebooks:
        path = Path(notebook_path)
        if path.exists():
            _notebooks[notebook_path] = nbformat.read(
                str(path), as_version=4
            )
        else:
            nb = new_notebook()
            nb.metadata.kernelspec = {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
            _notebooks[notebook_path] = nb
    return _notebooks[notebook_path]


def _get_runner(notebook_path: str) -> KernelRunner:
    """Get or create a KernelRunner for a notebook path."""
    if notebook_path not in _runners:
        runner = KernelRunner(session_id=f"nb:{notebook_path}")
        _runners[notebook_path] = runner
    return _runners[notebook_path]


def _save_notebook(notebook_path: str) -> None:
    """Save the in-memory notebook to disk."""
    nb = _get_notebook(notebook_path)
    path = Path(notebook_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(path))


def _runner_outputs_to_nb(outputs: list[dict]) -> list[dict]:
    """Convert KernelRunner output dicts to nbformat output objects."""
    nb_outputs = []
    for out in outputs:
        otype = out.get("type", "")
        if otype == "stream":
            nb_outputs.append(nbformat.v4.new_output(
                output_type="stream",
                name=out.get("name", "stdout"),
                text=out.get("text", ""),
            ))
        elif otype in ("display_data", "execute_result"):
            nb_outputs.append(nbformat.v4.new_output(
                output_type=otype,
                data=out.get("data", {}),
                metadata=out.get("metadata", {}),
            ))
        elif otype == "error":
            nb_outputs.append(nbformat.v4.new_output(
                output_type="error",
                ename=out.get("ename", ""),
                evalue=out.get("evalue", ""),
                traceback=out.get("traceback", []),
            ))
    return nb_outputs


def _execute_code(notebook_path: str, code: str) -> list[dict]:
    """Execute code in the kernel and return nbformat-compatible output dicts.

    Delegates to KernelRunner. Falls back to empty outputs if no kernel.
    """
    runner = _get_runner(notebook_path)
    raw_outputs = runner.execute(code)
    return _runner_outputs_to_nb(raw_outputs)


def _add_code_cell(notebook_path: str, code: str, execute: bool = True) -> str:
    """Add a code cell, optionally execute it, save the notebook."""
    nb = _get_notebook(notebook_path)
    cell = new_code_cell(source=code)

    if execute:
        outputs = _execute_code(notebook_path, code)
        cell.outputs = outputs

    nb.cells.append(cell)
    _save_notebook(notebook_path)

    # Return a text summary of the outputs
    text_parts = []
    for out in cell.outputs:
        if out.get("output_type") == "stream":
            text_parts.append(out.get("text", ""))
        elif out.get("output_type") in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "text/plain" in data:
                text_parts.append(data["text/plain"])
        elif out.get("output_type") == "error":
            text_parts.append(f"ERROR: {out.get('ename')}: {out.get('evalue')}")

    return "\n".join(text_parts) if text_parts else "(no output)"


def _add_markdown_cell(notebook_path: str, content: str) -> None:
    """Add a markdown cell and save."""
    nb = _get_notebook(notebook_path)
    nb.cells.append(new_markdown_cell(source=content))
    _save_notebook(notebook_path)


# ---------------------------------------------------------------------------
# MCP Tools (8)
# ---------------------------------------------------------------------------


@mcp.tool()
def nb_init(title: str, claim: str, notebook_path: str) -> str:
    """Initialize a new Jupyter notebook for evidence programming. Creates the
    notebook file with a title cell. Call this once at the start of
    verification."""
    nb = _get_notebook(notebook_path)

    # Clear any existing cells if re-initializing
    nb.cells = []

    # Title markdown cell
    _add_markdown_cell(notebook_path, f"# {title}\n\n**Claim:** {claim}")

    return f"Notebook initialized at {notebook_path}."


@mcp.tool()
def nb_markdown(content: str, notebook_path: str) -> str:
    """Add a markdown cell to the notebook. Use for narrative text, section
    headers, reasoning explanations, and transition notes between evidence
    programming steps."""
    _add_markdown_cell(notebook_path, content)
    return f"Markdown cell added ({len(content)} chars)."


@mcp.tool()
def nb_execute(code: str, notebook_path: str) -> str:
    """Execute Python code in the notebook's persistent kernel and add the
    cell with its output. Use for custom analysis, visualization, or data
    inspection. The kernel retains state between calls."""
    output = _add_code_cell(notebook_path, code, execute=True)
    return f"Code cell executed. Output:\n{output[:500]}"


@mcp.tool()
def nb_render_papers(workspace: str, notebook_path: str) -> str:
    """Render the currently retrieved papers as a styled HTML table in the
    notebook. Reads from evidence_state.json in the workspace. Call after
    search_pubmed or search_pubmed_progressive."""
    code = (
        "from pkevolve.verification.renderers import render_papers; "
        f"render_papers({workspace!r})"
    )
    output = _add_code_cell(notebook_path, code, execute=True)
    return f"Papers table rendered. {output[:200]}"


@mcp.tool()
def nb_render_facts(workspace: str, notebook_path: str) -> str:
    """Render extracted facts as a color-coded HTML table (green=SUPPORT,
    red=REFUTE, gray=NEUTRAL). Reads from evidence_state.json. Call after
    fact extraction rounds."""
    code = (
        "from pkevolve.verification.renderers import render_facts; "
        f"render_facts({workspace!r})"
    )
    output = _add_code_cell(notebook_path, code, execute=True)
    return f"Facts table rendered. {output[:200]}"


@mcp.tool()
def nb_render_sufficiency(workspace: str, notebook_path: str) -> str:
    """Render the latest sufficiency check results with progress bars for
    confidence and per-subclaim coverage, plus gap analysis. Reads from
    evidence_state.json. Call after check_sufficiency."""
    code = (
        "from pkevolve.verification.renderers import render_sufficiency; "
        f"render_sufficiency({workspace!r})"
    )
    output = _add_code_cell(notebook_path, code, execute=True)
    return f"Sufficiency visualization rendered. {output[:200]}"


@mcp.tool()
def nb_render_verdict(workspace: str, notebook_path: str) -> str:
    """Render the final verification verdict as a styled card with verdict,
    confidence, reasoning, key evidence, and remaining gaps. Reads from
    verdict.json in the workspace. Call after emit_verdict."""
    code = (
        "from pkevolve.verification.renderers import render_verdict; "
        f"render_verdict({workspace!r})"
    )
    output = _add_code_cell(notebook_path, code, execute=True)
    return f"Verdict card rendered. {output[:200]}"


@mcp.tool()
def nb_save(notebook_path: str) -> str:
    """Explicitly save the notebook to disk. The notebook is auto-saved after
    each tool call, but use this if you want to ensure a save at a specific
    point."""
    _save_notebook(notebook_path)
    return f"Notebook saved to {notebook_path}."


if __name__ == "__main__":
    mcp.run()
