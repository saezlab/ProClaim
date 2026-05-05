"""
Cost tracker for evidence programming baselines.

Wraps every LLM call to automatically record token usage and cost.
Attach one ``CostTracker`` per baseline run; reset between claims.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TraceEntry:
    step: int
    action: str  # e.g. "llm_call", "search"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    details: dict = field(default_factory=dict)


class CostTracker:
    """Accumulates token usage and cost across all operations in one claim run."""

    # Pricing per 1M tokens (input, output) in USD.
    # Source: https://cloud.google.com/vertex-ai/generative-ai/pricing (2026-04-02)
    # Keys are model IDs without provider prefix (e.g. "gemini-3.1-pro-preview",
    # "claude-sonnet-4-6"). Provider prefix is stripped before lookup.
    DEFAULT_PRICING: dict[str, tuple[float, float]] = {
        # --- Anthropic ---
        "claude-opus-4-6": (5.00, 25.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-haiku-4-5-20251001": (1.00, 5.00),
        # --- Gemini 3 (standard tier, per 1M tokens) ---
        "gemini-3.1-pro-preview": (2.00, 12.00),
        "gemini-3-pro-preview": (2.00, 12.00),
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (0.3, 2.50),
    }
    FALLBACK_PRICING = (3.00, 15.00)

    @classmethod
    def pricing_for(cls, model: str) -> tuple[float, float]:
        """Return (input_price, output_price) for a model string."""
        model_key = model.split("/", 1)[-1] if "/" in model else model
        return cls.DEFAULT_PRICING.get(model_key, cls.FALLBACK_PRICING)

    def __init__(self, model: str = ""):
        self.model = model
        self._trace: list[TraceEntry] = []
        self._step = 0
        # Strip provider prefix (e.g. "vertex_ai/", "anthropic/") before lookup
        in_price, out_price = self.pricing_for(model)
        self._in_price = in_price
        self._out_price = out_price

    # ------------------------------------------------------------------
    # Accumulation

    def record(
        self,
        action: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency: float = 0.0,
        **details: Any,
    ) -> None:
        cost = (
            (input_tokens / 1_000_000) * self._in_price
            + (output_tokens / 1_000_000) * self._out_price
        )
        self._trace.append(
            TraceEntry(
                step=self._step,
                action=action,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_seconds=latency,
                details=dict(details),
            )
        )
        self._step += 1

    def reset(self) -> None:
        self._trace.clear()
        self._step = 0

    # ------------------------------------------------------------------
    # Aggregates

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self._trace)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens for e in self._trace)

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self._trace)

    @property
    def total_latency_seconds(self) -> float:
        return sum(e.latency_seconds for e in self._trace)

    @property
    def num_llm_calls(self) -> int:
        return sum(1 for e in self._trace if e.action == "llm_call")

    def summary(self) -> dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "latency_seconds": round(self.total_latency_seconds, 2),
            "num_llm_calls": self.num_llm_calls,
        }

    # ------------------------------------------------------------------
    # Decorator / context helper

    def tracked_llm_call(self, fn: Callable[..., tuple[str, int, int]]):
        """Wrap a function that returns (text, input_tokens, output_tokens)."""

        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            latency = time.monotonic() - t0
            text, in_tok, out_tok = result
            self.record("llm_call", in_tok, out_tok, latency)
            return text

        return wrapper
