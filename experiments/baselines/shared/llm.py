"""
Shared LLM backend for all baselines.

Uses the OpenAI-compatible client pattern established throughout the project.
All baselines share the same backbone LLM so that differences in accuracy
reflect architecture, not model choice.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Resolve the API key following project priority: GLM_API_KEY > ZAI_API_KEY > OPENAI_API_KEY
def _resolve_api_key() -> str:
    for var in ("GLM_API_KEY", "ZAI_API_KEY", "OPENAI_API_KEY"):
        val = os.getenv(var)
        if val:
            return val
    return "EMPTY"


class LLMBackend:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint.

    Returns structured (text, input_tokens, output_tokens) from every call so
    the CostTracker can record usage accurately.
    """

    def __init__(
        self,
        model: str = "glm-4-plus",
        base_url: str = "https://api.z.ai/api/paas/v4/",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        from openai import OpenAI  # lazy import to avoid hard dep at module level

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_base_delay = retry_base_delay

        resolved_key = api_key or _resolve_api_key()
        self._client = OpenAI(base_url=base_url, api_key=resolved_key)

    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        *,
        response_format: str = "json_object",
    ) -> tuple[str, int, int]:
        """Send a chat completion request and return (text, input_tokens, output_tokens).

        Args:
            system: System prompt.
            user: User message.
            response_format: ``"json_object"`` (default) or ``"text"``.

        Returns:
            Tuple of (response_text, input_token_count, output_token_count).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                in_tok = resp.usage.prompt_tokens if resp.usage else 0
                out_tok = resp.usage.completion_tokens if resp.usage else 0
                return text, in_tok, out_tok
            except Exception as exc:
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    "LLM call attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    self.retries,
                    exc,
                    delay,
                )
                if attempt < self.retries - 1:
                    time.sleep(delay)
                else:
                    logger.error("All %d LLM call attempts failed.", self.retries)
                    return "", 0, 0

    def complete_text(self, system: str, user: str) -> tuple[str, int, int]:
        """Like ``complete`` but requests plain text rather than JSON."""
        return self.complete(system, user, response_format="text")

    def parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response; fall back to empty dict on error."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract a JSON block from the response
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
            return {}
