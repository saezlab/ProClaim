"""Shared construction helpers for baseline runner wiring."""

from __future__ import annotations

import argparse

from baselines.shared.llm import LLMBackend


def build_llm_backend(args: argparse.Namespace) -> LLMBackend:
    """Construct the shared LLM backend from CLI/config args."""
    return LLMBackend(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        thinking_budget=args.thinking_budget,
    )