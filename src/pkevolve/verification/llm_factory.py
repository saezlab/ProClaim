"""
LLM callable factory for the RLM verification system.

Provides ``make_llm``, which constructs a stateless ``llm(prompt: str) -> str``
callable suitable for passing to all subagent functions
(``extract_facts``, ``check_sufficiency``, ``emit_verdict``, etc.).

This module exists so that the kernel setup code injected into the Jupyter
kernel stays minimal and free of hand-rolled retry logic.  Instead of
defining ``llm`` inline as a multi-line string embedded in a system prompt
f-string, the orchestrator simply runs::

    from pkevolve.verification.llm_factory import make_llm
    llm = make_llm(base_url=..., api_key=..., model=...)

Benefits over the previous inline-string approach:
- No curly-brace escaping issues in f-strings.
- Retry and error-handling logic is version-controlled and unit-testable.
- ``make_llm`` raises a clear ``ValueError`` for missing parameters rather
  than silently authenticating with a dummy key.
- Streaming / non-streaming fallback is handled transparently.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Type alias used throughout the verification package
LLMCallable = Callable[[str], str]


def make_llm(
    base_url: str,
    api_key: str,
    model: str,
    *,
    max_tokens: int = 8000,
    temperature: float = 0.7,
    retries: int = 3,
    retry_base_delay: float = 1.0,
    stream: bool = True,
    extra_body: dict | None = None,
) -> LLMCallable:
    """Build an ``llm(prompt) -> str`` callable backed by an OpenAI-compatible endpoint.

    Args:
        base_url: Base URL of the OpenAI-compatible API
                  (e.g. ``"http://localhost:8000/v1/"``).
        api_key:  API key.  Pass ``"EMPTY"`` for unauthenticated local endpoints.
        model:    Model identifier string (e.g. ``"Qwen/Qwen3-8B"``).
        max_tokens:       Maximum tokens to generate per call (default 8000).
        temperature:      Sampling temperature (default 0.1 for determinism).
        retries:          Number of attempts before returning an empty string.
        retry_base_delay: Base delay in seconds for exponential back-off.
        stream:           Use the streaming API.  Falls back to non-streaming
                          automatically if the first streaming attempt returns
                          an empty payload.
        extra_body:       Extra fields forwarded verbatim in the request body.
                          Use this for backend-specific options, e.g. for
                          SGLang + Qwen3 to disable the built-in thinking mode
                          so that token budget is not consumed by reasoning::

                              extra_body={"chat_template_kwargs": {"enable_thinking": False}}

    Returns:
        A callable ``llm(prompt: str) -> str``.

    Raises:
        ValueError: If ``base_url`` or ``model`` are empty strings.

    Example::

        llm = make_llm(
            base_url="http://localhost:8000/v1/",
            api_key="EMPTY",
            model="Qwen/Qwen3-8B",
        )
        print(llm("Summarise the role of MAPK1 in cell signalling."))
    """
    if not base_url:
        raise ValueError("make_llm: base_url must not be empty")
    if not model:
        raise ValueError("make_llm: model must not be empty")

    from openai import OpenAI  # imported lazily so the module is import-safe

    client = OpenAI(base_url=base_url, api_key=api_key)

    def _call_streaming(prompt: str) -> str:
        """Single streaming attempt; returns text or raises."""
        import sys
        import re

        content: list[str] = []
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_body,
        )

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # SGLang/Qwen3 thinking mode: actual answer is in delta.content;
                # reasoning tokens appear in delta.reasoning_content (ignored here).
                if delta.content:
                    content.append(delta.content)

        full_text = "".join(content).strip()

        # Filter out <think>...</think> tags for stdout display
        # but keep them in the returned text
        display_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL)

        # Print a summary to stdout (so Agent sees activity)
        if display_text:
            # Show first 200 chars as a preview
            preview = display_text[:200].replace('\n', ' ')
            if len(display_text) > 200:
                preview += "..."
            print(f"[LLM response: {len(display_text)} chars] {preview}")
        else:
            print("[LLM response received]")

        return full_text

    def _call_blocking(prompt: str) -> str:
        """Single non-streaming attempt; returns text or raises."""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra_body=extra_body,
        )
        return (response.choices[0].message.content or "").strip()

    def llm(prompt: str) -> str:
        """Call the LLM with ``prompt`` and return the response text.

        Retries up to ``retries`` times with exponential back-off.
        Returns an empty string if all attempts fail.
        """
        for attempt in range(retries):
            try:
                if stream:
                    result = _call_streaming(prompt)
                    # If streaming returned nothing, try non-streaming as fallback
                    if not result:
                        logger.debug(
                            "llm(): streaming returned empty on attempt %d; "
                            "falling back to blocking call",
                            attempt + 1,
                        )
                        result = _call_blocking(prompt)
                else:
                    result = _call_blocking(prompt)

                if result:
                    return result

                logger.warning(
                    "llm(): empty response on attempt %d/%d", attempt + 1, retries
                )
            except Exception as exc:
                logger.warning(
                    "llm(): error on attempt %d/%d: %s", attempt + 1, retries, exc
                )

            if attempt < retries - 1:
                delay = retry_base_delay * (2 ** attempt)
                time.sleep(delay)

        logger.error("llm(): all %d retries exhausted; returning empty string", retries)
        return ""

    # Attach metadata so callers can inspect the configuration
    llm.__doc__ = (
        f"LLM callable — model={model!r} endpoint={base_url!r} "
        f"stream={stream} retries={retries}"
    )
    return llm
