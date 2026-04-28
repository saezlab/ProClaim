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

load_dotenv()
litellm.set_verbose = False

logger = logging.getLogger(__name__)


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

    def complete(
        self,
        system: str,
        user: str,
        *,
        response_format: str = "json_object",
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens) for a chat completion."""
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
        if self.thinking_budget is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        elif self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.retries):
            try:
                resp = litellm.completion(**kwargs)
                text = resp.choices[0].message.content or ""
                in_tok = resp.usage.prompt_tokens if resp.usage else 0
                out_tok = resp.usage.completion_tokens if resp.usage else 0
                return text, in_tok, out_tok
            except Exception as exc:
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    "LLM call attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, self.retries, exc, delay,
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
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
            return {}
