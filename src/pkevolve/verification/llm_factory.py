"""
LLM callable factory for the RLM verification system.

Provides ``make_llm``, which constructs a stateless ``llm(prompt: str) -> str``
callable suitable for passing to all subagent functions
(``extract_facts``, ``check_sufficiency``, ``emit_verdict``, etc.).

Uses LiteLLM for provider-agnostic routing — supports Anthropic, OpenAI,
vLLM, SGLang, DeepSeek, Gemini, and any other backend via provider-prefixed
model strings (e.g. ``"anthropic/claude-sonnet-4-20250514"``) or raw model
names with a ``base_url`` for local endpoints.

Example::

    from pkevolve.verification.llm_factory import make_llm
    llm = make_llm(model="anthropic/claude-sonnet-4-20250514", api_key=...)
    llm = make_llm(model="Qwen/Qwen3-8B", base_url="http://localhost:8000/v1/")
"""

from __future__ import annotations

import logging
import os
import time
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
    extra_body: dict | None = None,
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
        extra_body:       Extra fields forwarded verbatim in the request body.
                          Use this for backend-specific options, e.g. for
                          SGLang + Qwen3 to disable the built-in thinking mode::

                              extra_body={"chat_template_kwargs": {"enable_thinking": False}}

    Returns:
        A callable ``llm(prompt: str) -> str``.

    Raises:
        ValueError: If ``model`` is empty.

    Example::

        llm = make_llm(
            model="anthropic/claude-sonnet-4-20250514",
        )
        print(llm("Summarise the role of MAPK1 in cell signalling."))

        # Local vLLM / SGLang endpoint
        llm = make_llm(
            model="Qwen/Qwen3-8B",
            base_url="http://localhost:8000/v1/",
            api_key="EMPTY",
        )
    """
    if not model:
        raise ValueError("make_llm: model must not be empty")

    import litellm as _litellm
    import re as _re

    # Cloud providers are routed by LiteLLM automatically — don't
    # override with base_url which would break routing.
    effective_base_url = base_url
    effective_model = model
    if base_url and any(model.startswith(p) for p in _CLOUD_PREFIXES):
        logger.debug(
            "make_llm: ignoring base_url for cloud model %s", model
        )
        effective_base_url = None
    elif base_url and not any(model.startswith(p) for p in _CLOUD_PREFIXES):
        # Local endpoint (vLLM, SGLang, etc.) — LiteLLM requires a
        # provider prefix to route the request.  Add "openai/" so it
        # uses the OpenAI-compatible chat/completions endpoint.
        if not model.startswith("openai/"):
            effective_model = f"openai/{model}"
            logger.debug(
                "make_llm: auto-prefixed local model as %s", effective_model
            )

    def llm(prompt: str) -> str:
        """Call the LLM with ``prompt`` and return the response text.

        Retries up to *retries* times with exponential back-off.
        Returns an empty string if all attempts fail.
        """
        for attempt in range(retries):
            try:
                kwargs: dict = dict(
                    model=effective_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if api_key:
                    kwargs["api_key"] = api_key
                if effective_base_url:
                    kwargs["api_base"] = effective_base_url
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = _litellm.completion(**kwargs)
                result = (response.choices[0].message.content or "").strip()

                if result:
                    display = _re.sub(
                        r"<think>.*?</think>", "", result, flags=_re.DOTALL
                    )
                    preview = display[:200].replace("\n", " ")
                    if len(display) > 200:
                        preview += "..."
                    if _EVIDENCE_DEBUG:
                        print(f"[LLM response: {len(display)} chars] {preview}")
                    return result

                logger.warning(
                    "llm(): empty response on attempt %d/%d",
                    attempt + 1,
                    retries,
                )
            except Exception as exc:
                logger.warning(
                    "llm(): error on attempt %d/%d: %s",
                    attempt + 1,
                    retries,
                    exc,
                )

            if attempt < retries - 1:
                delay = retry_base_delay * (2 ** attempt)
                time.sleep(delay)

        logger.error("llm(): all %d retries exhausted; returning empty string", retries)
        return ""

    # Attach metadata so callers can inspect the configuration
    llm.__doc__ = (
        f"LLM callable (litellm) — model={model!r} "
        f"base_url={effective_base_url!r} retries={retries}"
    )
    return llm
