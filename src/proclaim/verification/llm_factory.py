"""
LLM callable factory for the RLM verification system.

Provides ``make_llm``, which constructs a stateless ``llm(prompt: str) -> str``
callable suitable for passing to all subagent functions
(``extract_facts``, ``check_sufficiency``, ``emit_verdict``, etc.).

Uses LiteLLM for provider-agnostic routing — supports Anthropic, OpenAI,
vLLM, SGLang, DeepSeek, Gemini, and any other backend via provider-prefixed
model strings (e.g. ``"anthropic/claude-sonnet-4-20250514"``) or raw model
names with a ``base_url`` for local endpoints.
"""

from __future__ import annotations

import logging
import os
import time
import re
from typing import Callable

logger = logging.getLogger(__name__)

_EVIDENCE_DEBUG = os.environ.get("EVIDENCE_DEBUG", "0") == "1"

# Type alias used throughout the verification package
LLMCallable = Callable[[str], str]

# Known cloud providers — LiteLLM routes these automatically via its
# built-in provider map, so we must NOT pass api_base for them.
_CLOUD_PREFIXES = (
    "anthropic/", "openai/", "deepseek/", "gemini/",
    "cohere/", "mistral/", "bedrock/", "vertex_ai/",
)


def make_llm(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 8000,
    temperature: float = 0.7,
    retries: int = 3,
    retry_base_delay: float = 1.0,
    timeout: int = 300,
    extra_body: dict | None = None,
    stream: bool = True,
) -> LLMCallable:
    """Build an ``llm(prompt) -> str`` callable backed by LiteLLM.

    Args:
        model:    LiteLLM model string.  Provider-prefixed strings like
                  ``"anthropic/claude-sonnet-4-20250514"`` are routed automatically.
                  Raw model names (e.g. ``"Qwen/Qwen3-8B"``) require *base_url*.
        api_key:  API key.  For local unauthenticated endpoints pass
                  ``"EMPTY"`` or omit.  For cloud providers, falls back
                  to standard env vars (``ANTHROPIC_API_KEY``, etc.).
        base_url: Base URL for OpenAI-compatible endpoints
                  (e.g. ``"http://localhost:8000/v1/"``).
                  Ignored for cloud provider-prefixed models.
        max_tokens:       Maximum tokens to generate per call (default 8000).
        temperature:      Sampling temperature (default 0.7).
        retries:          Number of attempts before returning an empty string.
        retry_base_delay: Base delay in seconds for exponential back-off.
        timeout:          Connection and read timeout in seconds (default 300).
        extra_body:       Extra fields forwarded verbatim in the request body.
        stream:           Use streaming API (default True). Prevents timeouts
                          during long reasoning phases.

    Returns:
        A callable ``llm(prompt: str) -> str``.
    """
    if not model:
        raise ValueError("make_llm: model must not be empty")

    import litellm as _litellm

    # Cloud providers are routed by LiteLLM automatically
    _is_cloud = any(model.startswith(p) for p in _CLOUD_PREFIXES)
    effective_base_url = base_url
    effective_model = model
    # extra_body is vLLM-specific (e.g. chat_template_kwargs); cloud APIs reject it
    if _is_cloud:
        extra_body = None
    if base_url and _is_cloud:
        effective_base_url = None
    elif base_url and not _is_cloud:
        if not model.startswith("openai/"):
            effective_model = f"openai/{model}"

    def _call_streaming(kwargs: dict) -> str:
        """Single streaming attempt using LiteLLM."""
        content: list[str] = []
        # Add stream=True to kwargs
        stream_kwargs = {**kwargs, "stream": True}
        
        response = _litellm.completion(**stream_kwargs)
        
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # Handle both standard content and reasoning_content (vLLM/SGLang)
                if hasattr(delta, "content") and delta.content:
                    content.append(delta.content)
                elif isinstance(delta, dict) and delta.get("content"):
                    content.append(delta["content"])
        
        return "".join(content).strip()

    def _call_blocking(kwargs: dict) -> str:
        """Single blocking attempt using LiteLLM."""
        response = _litellm.completion(**kwargs)
        # Check for content or reasoning_content
        msg = response.choices[0].message
        result = msg.content or ""
        # If content is empty but reasoning is present, we might have a 
        # model that only produced reasoning or LiteLLM mis-mapped it.
        # But for fact extraction, we only care about final content.
        return result.strip()

    def llm(prompt: str) -> str:
        """Call the LLM with ``prompt`` and return the response text."""
        for attempt in range(retries):
            try:
                kwargs: dict = dict(
                    model=effective_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                if api_key:
                    kwargs["api_key"] = api_key
                if effective_base_url:
                    kwargs["api_base"] = effective_base_url
                if extra_body:
                    kwargs["extra_body"] = extra_body

                if stream:
                    result = _call_streaming(kwargs)
                else:
                    result = _call_blocking(kwargs)

                if result:
                    # Strip reasoning tags for display only
                    display = re.sub(
                        r"<think>.*?</think>", "", result, flags=re.DOTALL
                    )
                    preview = display[:200].replace("\n", " ")
                    if len(display) > 200:
                        preview += "..."
                    if _EVIDENCE_DEBUG:
                        print(f"[LLM response: {len(display)} chars] {preview}")
                    return result

                logger.warning(
                    "llm(): empty response on attempt %d/%d (model=%s)",
                    attempt + 1, retries, model
                )
            except Exception as exc:
                logger.warning(
                    "llm(): error on attempt %d/%d: %s",
                    attempt + 1, retries, exc
                )

            if attempt < retries - 1:
                delay = retry_base_delay * (2 ** attempt)
                time.sleep(delay)

        logger.error("llm(): all %d retries exhausted; returning empty string", retries)
        return ""

    llm.__doc__ = (
        f"LLM callable (litellm) — model={model!r} "
        f"base_url={effective_base_url!r} stream={stream} timeout={timeout}"
    )
    return llm


def make_anthropic_llm(
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    *,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    retries: int = 3,
    retry_base_delay: float = 1.0,
) -> LLMCallable:
    """Build an ``llm(prompt) -> str`` callable backed by the native Anthropic API.

    Designed for Claude models (e.g. claude-haiku-4-5-20251001).
    Uses ``max_tokens=1024`` since sufficiency output is a short JSON block.
    ``temperature=1.0`` follows Anthropic's recommended default.
    """
    if not api_key:
        raise ValueError("make_anthropic_llm: api_key must not be empty")
    if not model:
        raise ValueError("make_anthropic_llm: model must not be empty")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    def llm(prompt: str) -> str:
        """Call Claude via native Anthropic API and return the response text."""
        for attempt in range(retries):
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = (message.content[0].text or "").strip()
                if result:
                    preview = result[:200].replace("\n", " ")
                    if len(result) > 200:
                        preview += "..."
                    logger.debug("[Haiku response: %d chars] %s", len(result), preview)
                    return result

                logger.warning(
                    "make_anthropic_llm: empty response on attempt %d/%d",
                    attempt + 1, retries,
                )
            except Exception as exc:
                logger.warning(
                    "make_anthropic_llm: error on attempt %d/%d: %s",
                    attempt + 1, retries, exc,
                )

            if attempt < retries - 1:
                delay = retry_base_delay * (2 ** attempt)
                time.sleep(delay)

        logger.error(
            "make_anthropic_llm: all %d retries exhausted; returning empty string",
            retries,
        )
        return ""

    llm.__doc__ = (
        f"Anthropic LLM callable — model={model!r} retries={retries}"
    )
    return llm
