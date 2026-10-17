"""
Shared LLM backend for all baselines.

Uses LiteLLM for provider-agnostic model access. Pass the model name with
the LiteLLM provider prefix and set the matching API key in .env — no
base_url required for cloud providers.

Model name examples:
    "zai/glm-4-plus"              → reads ZAI_API_KEY
    "openai/gpt-4o"               → reads OPENAI_API_KEY
    "anthropic/claude-..."        → reads ANTHROPIC_API_KEY
    "vertex_ai/gemini-2.5-flash"   → reads GCP credentials (see below)

Vertex AI credentials (.env):
    VERTEXAI_PROJECT=your-gcp-project-id       # required
    VERTEXAI_LOCATION=us-central1              # optional, defaults to us-central1
    VERTEX_CREDENTIALS='{...}'                 # service account JSON as inline string
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import litellm
from dotenv import load_dotenv

from baselines.shared import prompt_cache

load_dotenv()
litellm.set_verbose = False

logger = logging.getLogger(__name__)

# Some newer models reject a client-supplied `temperature` outright (a real
# API error, not just an unsupported-param warning), and litellm's
# drop_params doesn't catch this since it only drops params its own
# provider table already flags as unsupported. Detected dynamically per
# model below rather than hardcoded, since which models do this has
# already changed across two model generations observed in this repo.
_TEMPERATURE_DEPRECATED_MSG = "temperature` is deprecated for this model"


class LLMBackend:
    """Thin wrapper around LiteLLM for provider-agnostic chat completion.

    Returns (text, input_tokens, output_tokens) from every call so the
    CostTracker can record usage accurately.
    """

    def __init__(
        self,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        retries: int = 3,
        retry_base_delay: float = 1.0,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_base_delay = retry_base_delay
        self.reasoning_effort = reasoning_effort
        self.thinking_budget = thinking_budget
        # Opt-in Claude prompt caching (PROCLAIM_PROMPT_CACHE=1); see shared/prompt_cache.py.
        self.prompt_cache = prompt_cache.enabled(model)

    def provider_kwargs(self) -> dict:
        """Provider-level knobs shared by EVERY call path.

        Single source of truth so an agentic baseline that builds its own client
        cannot drift from ``complete()``. It did drift: ``ReActBaseline`` built a
        ``ChatLiteLLM`` without the ``LLM_EXTRA_BODY`` passthrough, so a thinking
        model served behind a reasoning parser returned its answer in
        ``reasoning_content`` and an EMPTY ``content``. The final-answer parse then
        saw an empty string and every claim silently fell back to UNCERTAIN —
        58/105 on qwen3.5-9b, with no error raised anywhere.
        """
        kw: dict = {}
        if self.thinking_budget is not None:
            kw["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        elif self.reasoning_effort is not None:
            kw["reasoning_effort"] = self.reasoning_effort
        # Opt-in passthrough for OpenAI-compatible servers (e.g. local vLLM):
        # LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
        # No effect unless the env var is set, so cloud-provider runs are unchanged.
        extra_body_env = os.environ.get("LLM_EXTRA_BODY")
        if extra_body_env:
            kw["extra_body"] = json.loads(extra_body_env)
        return kw

    def chat_model(self, **overrides):
        """A LangChain ``ChatLiteLLM`` configured identically to ``complete()``.

        Use this from any LangChain/LangGraph baseline rather than constructing a
        chat model directly, so model, temperature, token budget and provider
        knobs all come from one place.
        """
        from langchain_community.chat_models.litellm import ChatLiteLLM
        kwargs: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        pk = self.provider_kwargs()
        if self.prompt_cache:
            # Mark the latest message each call, so an agent loop reads the previous step's entry.
            pk["cache_control_injection_points"] = [{"location": "message", "index": -1}]
        if pk:
            kwargs["model_kwargs"] = pk
        kwargs.update(overrides)
        return ChatLiteLLM(**kwargs)

    def complete(
        self,
        system: str,
        user: str | list[dict],
        *,
        response_format: str = "json_object",
        debug: dict | None = None,
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens) for a chat completion.

        ``debug``, if passed a dict, is populated in-place with fields not
        otherwise returned — ``finish_reason`` and ``reasoning_content``
        (present on reasoning-parser-backed servers, e.g. vLLM
        ``--reasoning-parser qwen3``: text the parser split OUT of
        ``message.content`` into a separate field, which this method never
        reads into ``text``). Passing a per-call dict rather than storing on
        ``self`` keeps this safe under a ThreadPoolExecutor of concurrent
        calls sharing one ``LLMBackend``. Kept optional and additive so every
        existing 3-value unpack of this method's return stays unaffected.
        ``user`` may be a list of content blocks (prompt caching); ``debug`` then
        also gets ``cache_read`` / ``cache_write`` (included in the returned
        input-token count).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "drop_params": True,
        }
        kwargs.update(self.provider_kwargs())
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.retries):
            try:
                resp = litellm.completion(**kwargs)
                msg = resp.choices[0].message
                text = msg.content or ""
                in_tok = resp.usage.prompt_tokens if resp.usage else 0
                out_tok = resp.usage.completion_tokens if resp.usage else 0
                if debug is not None:
                    debug["cache_read"], debug["cache_write"] = prompt_cache.usage_cache_tokens(resp.usage)
                    debug["finish_reason"] = getattr(resp.choices[0], "finish_reason", None)
                    debug["reasoning_content"] = getattr(msg, "reasoning_content", None) or ""
                    debug["content_len"] = len(text)
                return text, in_tok, out_tok
            except Exception as exc:
                if _TEMPERATURE_DEPRECATED_MSG in str(exc) and "temperature" in kwargs:
                    logger.warning(
                        "Model %s rejects `temperature`; retrying without it.",
                        self.model,
                    )
                    del kwargs["temperature"]
                    continue
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    "LLM call attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, self.retries, exc, delay,
                )
                if attempt < self.retries - 1:
                    time.sleep(delay)
                else:
                    logger.error("All %d LLM call attempts failed.", self.retries)
                    if debug is not None:
                        debug["error"] = str(exc)
                    return "", 0, 0

    def complete_text(self, system: str, user: str | list[dict], *,
                      debug: dict | None = None) -> tuple[str, int, int]:
        """Like ``complete`` but requests plain text rather than JSON."""
        return self.complete(system, user, response_format="text", debug=debug)

    def parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response; fall back to empty dict on error."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
            return {}
