"""Opt-in Claude prompt caching for the baselines (``PROCLAIM_PROMPT_CACHE=1``).

The prompt text a model sees is unchanged; only how it is sent changes (split into content
blocks carrying ``cache_control`` markers). Claude on Vertex AI caches per region, so cached
runs should pin ``VERTEXAI_LOCATION`` to one region rather than ``global``.

Pricing (Anthropic multipliers): cache reads 0.1x input, 5-minute cache writes 1.25x input.
LiteLLM's ``prompt_tokens`` includes both, so costs must be recomputed with ``cached_cost``.
"""

from __future__ import annotations

import os
import threading

CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25
_MARK = {"type": "ephemeral"}


def enabled(model: str) -> bool:
    """Caching is on only when requested and the model is Claude."""
    return os.environ.get("PROCLAIM_PROMPT_CACHE") == "1" and "claude" in (model or "").lower()


def usage_cache_tokens(usage) -> tuple[int, int]:
    """(cache_read, cache_write) input tokens from a LiteLLM usage object; 0 when absent."""
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
    read = get("cache_read_input_tokens") or 0
    write = get("cache_creation_input_tokens") or 0
    return int(read), int(write)


def cached_cost(input_tokens: int, output_tokens: int, cache_read: int, cache_write: int,
                in_price: float, out_price: float) -> float:
    """USD cost when ``input_tokens`` (LiteLLM prompt_tokens) includes cache reads and writes;
    prices are per million tokens."""
    uncached = max(input_tokens - cache_read - cache_write, 0)
    billed_in = uncached + CACHE_READ_MULT * cache_read + CACHE_WRITE_MULT * cache_write
    return (billed_in * in_price + output_tokens * out_price) / 1_000_000


class GrowingPrefixBlocks:
    """Content blocks for loops whose prompt is [fixed instructions + evidence so far] + tail,
    where the evidence grows each step and the tail (e.g. ``STATEMENT:``) stays the same.

    Each call returns ``[previous prefix, newly added evidence (marked), tail]`` so step k reads
    step k-1's cache entry (at a block boundary it shares) and writes only the new evidence.
    Loops are told apart by their instructions (text before KNOWLEDGE:) and tail, so
    interleaved loops keep separate chains. Thread-safe.
    """

    def __init__(self, tail_marker: str = "STATEMENT:", head_marker: str = "KNOWLEDGE:") -> None:
        self.tail_marker = tail_marker
        self.head_marker = head_marker
        self._prev: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> list[dict] | str:
        cut = prompt.rfind(self.tail_marker)
        if cut <= 0:
            return prompt
        # End the prefix at the last evidence character: the blank lines before the tail
        # marker would otherwise stop step k's prefix from extending step k-1's.
        prefix = prompt[:cut].rstrip()
        tail = prompt[len(prefix):]
        head_end = prompt.find(self.head_marker)
        key = (prompt[:head_end] if head_end > 0 else prompt[:200], tail)
        with self._lock:
            prev = self._prev.get(key)
            self._prev[key] = prefix
        blocks = []
        if prev and prefix.startswith(prev) and len(prefix) > len(prev):
            blocks.append({"type": "text", "text": prev})
            blocks.append({"type": "text", "text": prefix[len(prev):], "cache_control": _MARK})
        else:
            blocks.append({"type": "text", "text": prefix, "cache_control": _MARK})
        blocks.append({"type": "text", "text": tail})
        return blocks
