"""
Jupyter kernel lifecycle management for REPL-based evidence programming.

Provides a KernelRunner that manages a persistent Python kernel for
executing LLM-generated code, used by notebook_mcp.py (Claude Agent
SDK + nb_execute).

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
