"""
Jupyter kernel lifecycle management for REPL-based evidence programming.

Provides a KernelRunner that manages a persistent Python kernel for
executing LLM-generated code. Used by both:
  - notebook_mcp.py  (Mode A: Claude Agent SDK + nb_execute)
  - repl_orchestrator.py  (Mode B: standalone REPL loop)

The kernel acts as the "working memory" in the Recursive Language Model
paradigm: evidence state lives as a Python variable in the kernel,
importable library functions are pre-loaded, and each code cell's output
is captured and fed back to the LLM.
"""

import atexit
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_kernels: dict[str, tuple] = {}  # id -> (KernelManager, KernelClient)
_kernel_available: bool | None = None  # lazy-checked


def _check_kernel_available() -> bool:
    """Check once whether jupyter_client is importable."""
    global _kernel_available
    if _kernel_available is None:
        try:
            import jupyter_client  # noqa: F401
            _kernel_available = True
        except ImportError:
            logger.warning(
                "jupyter_client not available -- kernel execution disabled."
            )
            _kernel_available = False
    return _kernel_available


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class KernelRunner:
    """Manages a persistent Jupyter kernel for code execution.

    Usage::

        runner = KernelRunner("my-session")
        runner.start()
        outputs = runner.execute("print('hello')")
        runner.shutdown()
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._km = None
        self._kc = None

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Start the kernel. Returns True if kernel is ready."""
        if self.session_id in _kernels:
            self._km, self._kc = _kernels[self.session_id]
            return True

        if not _check_kernel_available():
            return False

        from jupyter_client import KernelManager

        try:
            km = KernelManager(kernel_name="python3")
            km.start_kernel()
            kc = km.client()
            kc.start_channels()
            kc.wait_for_ready(timeout=60)
            self._km = km
            self._kc = kc
            _kernels[self.session_id] = (km, kc)
            return True
        except Exception as exc:
            logger.warning("Could not start Jupyter kernel: %s", exc)
            return False

    def shutdown(self) -> None:
        """Shutdown the kernel and clean up."""
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
        _kernels.pop(self.session_id, None)
        self._km = None
        self._kc = None

    @property
    def is_alive(self) -> bool:
        """Check if the kernel is running."""
        return self._km is not None and self._km.is_alive()

    # -- Execution ---------------------------------------------------------

    def execute(self, code: str, timeout: int = 120) -> list[dict]:
        """Execute code in the kernel and return output dicts.

        Each output dict has 'type' and 'content' keys:
          - {"type": "stream", "name": "stdout", "text": "..."}
          - {"type": "execute_result", "data": {...}, "metadata": {...}}
          - {"type": "display_data", "data": {...}, "metadata": {...}}
          - {"type": "error", "ename": "...", "evalue": "...", "traceback": [...]}

        Returns empty list if kernel is not available.
        """
        if self._kc is None:
            if not self.start():
                return []

        outputs: list[dict] = []
        msg_id = self._kc.execute(code)

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except Exception:
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                outputs.append({
                    "type": "stream",
                    "name": content.get("name", "stdout"),
                    "text": content.get("text", ""),
                })
            elif msg_type in ("display_data", "execute_result"):
                outputs.append({
                    "type": msg_type,
                    "data": content.get("data", {}),
                    "metadata": content.get("metadata", {}),
                })
            elif msg_type == "error":
                outputs.append({
                    "type": "error",
                    "ename": content.get("ename", ""),
                    "evalue": content.get("evalue", ""),
                    "traceback": content.get("traceback", []),
                })
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain shell channel
        try:
            self._kc.get_shell_msg(timeout=10)
        except Exception:
            pass

        return outputs

    def execute_silent(self, code: str, timeout: int = 120) -> None:
        """Execute code without capturing outputs (e.g. prelude imports)."""
        self.execute(code, timeout=timeout)

    # -- Prelude -----------------------------------------------------------

    def inject_prelude(
        self,
        claim: str,
        workspace: str,
        extra_code: Optional[str] = None,
        *,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """Inject the standard REPL prelude into the kernel.

        Pre-loads imports, initializes EvidenceState as ``state``, and
        makes the evidence API available for direct use.

        If *llm_base_url*, *llm_api_key*, and *llm_model* are given, a
        default ``llm`` callable is wired up so that ``extract_and_add_facts``
        and the subagent functions work out of the box.
        """
        prelude = f"""\
import sys, os
from pathlib import Path

# Ensure pkevolve is importable
_project_root = Path({workspace!r}).resolve()
while _project_root.name != "grn-llm-correct" and _project_root != _project_root.parent:
    _project_root = _project_root.parent
_src = str(_project_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Core imports
from pkevolve.verification.evidence_api import (
    search_pubmed,
    search_pubmed_progressive,
    find_related_articles,
    get_full_text_article,
    get_paper_text,
    check_sufficiency,
    compress_evidence,
    add_facts_from_dicts,
    update_synthesis,
    add_conflict,
    emit_verdict,
    formulate_pubmed_query,
    get_evidence_summary,
    extract_and_add_facts,
    search_for_gap,
)
from pkevolve.verification.subagents import (
    extract_facts,
    synthesize_subclaim,
    detect_conflicts,
    identify_gaps,
    formulate_gap_queries,
)
from pkevolve.verification.evidence_state import EvidenceState
from pkevolve.verification.data_models import (
    PaperRecord, Fact, Stance, Conflict, Gap, GapType, GapPriority,
    SufficiencyResult, VerificationVerdict,
)
from pkevolve.verification.renderers import (
    render_papers_from_state,
    render_facts_from_state,
    render_sufficiency_from_state,
)
from pkevolve.verification.compressor import SufficiencyPreservingCompressor

# Initialize state
workspace = Path({workspace!r})
workspace.mkdir(parents=True, exist_ok=True)
state = EvidenceState.init_new(
    claim={claim!r},
    subclaims=[{claim!r}],
    workspace=workspace,
)
print(f"State initialized: {{state}}")
"""
        # Wire up the default llm callable if endpoint info is provided
        if llm_base_url and llm_api_key and llm_model:
            llm_code = (
                "\n# LLM callable for subagents and extract_and_add_facts\n"
                "from pkevolve.verification.llm_factory import make_llm\n"
                f"llm = make_llm(\n"
                f"    base_url={llm_base_url!r},\n"
                f"    api_key={llm_api_key!r},\n"
                f"    model={llm_model!r},\n"
                ")\n"
                f'print("llm() wired to {llm_model} at {llm_base_url}")\n'
            )
            prelude += llm_code


        if extra_code:
            prelude += "\n" + extra_code + "\n"

        self.execute_silent(prelude)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def outputs_to_text(outputs: list[dict]) -> str:
    """Convert kernel outputs to plain text for feeding back to the LLM."""
    parts: list[str] = []
    for out in outputs:
        if out["type"] == "stream":
            parts.append(out["text"])
        elif out["type"] in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "text/plain" in data:
                parts.append(data["text/plain"])
        elif out["type"] == "error":
            # Include abbreviated traceback
            tb = out.get("traceback", [])
            # Strip ANSI escapes for LLM readability
            import re
            ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
            clean_tb = [ansi_escape.sub("", line) for line in tb]
            parts.append(
                f"ERROR: {out['ename']}: {out['evalue']}\n"
                + "\n".join(clean_tb[-5:])  # last 5 lines of traceback
            )
    return "\n".join(parts).rstrip()


def _shutdown_all_kernels():
    """Shutdown all managed kernels on exit."""
    for sid, (km, kc) in list(_kernels.items()):
        try:
            kc.stop_channels()
            km.shutdown_kernel(now=True)
        except Exception:
            pass
    _kernels.clear()


atexit.register(_shutdown_all_kernels)
