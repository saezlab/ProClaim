"""
ReAct baseline — unconstrained agentic reasoning with tool use.

Implements the ReAct (Yao et al., ICLR 2023) Thought → Action → Observation
loop for scientific claim verification.  The agent has access to:
  - ``search_web(query)`` — web search via Serper or DuckDuckGo
  - ``finish(verdict, reasoning)`` — emit a final verdict and stop

The agent decides on its own when to stop — there is NO external sufficiency
signal.  This is the "why not just use a ReAct agent?" baseline.

Key comparison points vs Evidence Programming:
  - No structured evidence state (just flat conversation history)
  - No learned sufficiency classifier (LLM decides when to stop)
  - No gap-directed retrieval (LLM generates queries freely)
  - No cross-paper synthesis (implicit in LLM context)

Search backends (checked in order):
  1. If ``SERPER_API_KEY`` is set → Serper API (paid, higher quality).
  2. Otherwise → ``ddgs`` (free, DuckDuckGo search).

Cost: 1–N LLM calls per claim (N ≤ max_steps) + web searches.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm
import requests

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.label_utils import normalize_label
from baselines.shared.verdict import BaselineResult

logger = logging.getLogger(__name__)


# ── Tool definitions for native tool-use API ─────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for scientific evidence relevant to verifying "
                "the claim. Use specific, targeted queries mentioning key "
                "entities and relationships from the claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant evidence.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Emit a final verdict once you have gathered enough evidence. "
                "Call this when you are confident in your assessment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["SUPPORT", "REFUTE", "UNCERTAIN"],
                        "description": (
                            "SUPPORT — evidence corroborates the claim; "
                            "REFUTE — evidence contradicts the claim or no "
                            "evidence substantiates it; "
                            "UNCERTAIN — evidence is ambiguous or conflicting."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "One or two sentences explaining the verdict, "
                            "citing the key evidence found."
                        ),
                    },
                },
                "required": ["verdict", "reasoning"],
            },
        },
    },
]

_SYSTEM_PROMPT = """\
You are a scientific claim verification agent using the ReAct framework.

Your task: determine whether a scientific claim is SUPPORTED, REFUTED, or \
UNCERTAIN based on evidence you retrieve from the web.

Available tools:
- search_web(query): Search the web for scientific evidence. Use specific \
queries targeting the entities and relationships in the claim.
- finish(verdict, reasoning): Emit your final verdict when you have enough \
evidence.

Strategy:
1. Think step-by-step about what evidence you need.
2. Issue targeted search queries to find that evidence.
3. After each search result, assess whether you have enough evidence.
4. When confident, call finish() with your verdict and reasoning.

Rules:
- Do NOT guess — search for evidence before deciding.
- If multiple searches yield no relevant evidence, call finish() with REFUTE \
(no evidence substantiates the claim) or UNCERTAIN (insufficient evidence).
- Cite specific findings from your searches in your reasoning.
- You have a limited number of steps — be efficient with your queries."""

_FORCE_FINISH_PROMPT = """\
You have used all available search steps. Based on all the evidence gathered \
so far, you MUST now call the finish() tool with your final verdict \
(SUPPORT, REFUTE, or UNCERTAIN) and reasoning."""


# ── Web search backends (shared with FIRE) ───────────────────────────

_SERPER_URL = "https://google.serper.dev"


def _serper_search(query: str, api_key: str, k: int = 3) -> str:
    """Query Google via Serper API and return concatenated snippets."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    params = {"q": query, "num": k, "gl": "us", "hl": "en"}
    resp = requests.post(
        f"{_SERPER_URL}/search", headers=headers, params=params, timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    snippets: list[str] = []
    if data.get("answerBox"):
        ab = data["answerBox"]
        for f in ("answer", "snippet", "snippetHighlighted"):
            val = ab.get(f)
            if val and isinstance(val, str):
                snippets.append(val.replace("\n", " "))
    if data.get("knowledgeGraph"):
        kg = data["knowledgeGraph"]
        if kg.get("description"):
            snippets.append(kg["description"])
    for item in data.get("organic", [])[:k]:
        if "snippet" in item:
            snippets.append(item["snippet"])
    return " ".join(snippets) if snippets else "No relevant search results found."


def _ddg_search(query: str, k: int = 3) -> str:
    """Query DuckDuckGo via the ddgs package (free, no API key)."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("ddgs not installed. Run: pip install ddgs")

    snippets: list[str] = []
    try:
        ddgs = DDGS()
        for result in ddgs.text(query, max_results=k):
            body = result.get("body", "")
            if body:
                snippets.append(body)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)

    return " ".join(snippets) if snippets else "No relevant search results found."


# ── Baseline class ───────────────────────────────────────────────────


class ReActBaseline:
    """Verify claims via the ReAct (Thought → Action → Observation) loop.

    Uses the LLM's native tool-use API for structured action selection
    rather than parsing free-form text.

    Parameters
    ----------
    model:
        Any litellm model string, e.g. ``"anthropic/claude-sonnet-4-20250514"``,
        ``"openai/gpt-4o-mini"``.
    max_steps:
        Maximum number of search steps before forcing a final verdict.
    num_search_results:
        Number of search results per query.
    temperature:
        LLM sampling temperature.
    """

    name = "react"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        max_steps: int = 10,
        num_search_results: int = 3,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.num_search_results = num_search_results
        self.temperature = temperature
        self.log_dir: Path | None = None

        self._serper_key = os.environ.get("SERPER_API_KEY", "")
        self._search_backend = "serper" if self._serper_key else "ddg"
        logger.info("ReAct search backend: %s", self._search_backend)

        # Cost estimation
        model_key = model.split("/", 1)[-1] if "/" in model else model
        in_price, out_price = CostTracker.DEFAULT_PRICING.get(
            model_key, CostTracker.FALLBACK_PRICING,
        )
        self._in_price = in_price
        self._out_price = out_price

    # ── Public interface ─────────────────────────────────────────────

    def verify(
        self,
        claim_id: str,
        claim: str,
        gold_label: str,
        *,
        context: dict | None = None,
    ) -> BaselineResult:
        t0 = time.monotonic()
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        search_queries: list[str] = []

        try:
            verdict, reasoning = self._react_loop(
                claim, total_usage, search_queries,
            )
        except Exception as exc:
            logger.error("ReAct error for %s: %s", claim_id, exc)
            verdict, reasoning = "UNCERTAIN", f"ERROR: {exc}"

        latency = time.monotonic() - t0

        # Write per-claim log
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{claim_id}.log"
            with open(log_path, "w") as lf:
                lf.write(f"=== claim_id: {claim_id} ===\n")
                lf.write(f"=== claim ===\n{claim}\n\n")
                lf.write(f"=== verdict: {verdict} ===\n")
                lf.write(f"=== searches ({len(search_queries)}) ===\n")
                for q in search_queries:
                    lf.write(f"  Q: {q}\n")
                lf.write(f"\n=== reasoning ===\n{reasoning}\n")

        predicted = normalize_label(verdict)

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=0.0,
            reasoning=reasoning[:1000] if reasoning else "",
            evidence=search_queries,
            input_tokens=total_usage["input_tokens"],
            output_tokens=total_usage["output_tokens"],
            cost_usd=(
                total_usage["input_tokens"] / 1_000_000 * self._in_price
                + total_usage["output_tokens"] / 1_000_000 * self._out_price
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )

    # ── Core ReAct loop ──────────────────────────────────────────────

    def _react_loop(
        self,
        claim: str,
        usage: dict[str, int],
        search_queries: list[str],
    ) -> tuple[str, str]:
        """Run Thought → Action → Observation loop.  Returns (verdict, reasoning)."""
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Verify this scientific claim:\n\n{claim}"},
        ]

        steps_used = 0

        for _ in range(self.max_steps + 1):  # +1 for the final forced finish
            # Call LLM with tools
            resp = self._llm_call(messages, tools=_TOOLS)
            self._add_usage(usage, resp)

            assistant_msg = resp["message"]
            messages.append(assistant_msg)

            # Check for tool calls
            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # No tool call — try to extract a verdict from the text
                text = assistant_msg.get("content", "") or ""
                verdict, reasoning = self._try_parse_verdict(text)
                if verdict:
                    return verdict, reasoning
                # If no verdict and no tool call, prompt the model to act
                messages.append({
                    "role": "user",
                    "content": (
                        "Please use one of the available tools: search_web() "
                        "to find evidence, or finish() to emit your verdict."
                    ),
                })
                continue

            # Process each tool call
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                if fn_name == "finish":
                    verdict = fn_args.get("verdict", "UNCERTAIN")
                    reasoning = fn_args.get("reasoning", "")
                    return verdict, reasoning

                elif fn_name == "search_web":
                    query = fn_args.get("query", claim)
                    search_queries.append(query)
                    steps_used += 1

                    # Execute search
                    result = self._search(query)

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                    # If we've hit max steps, force finish on next iteration
                    if steps_used >= self.max_steps:
                        messages.append({
                            "role": "user",
                            "content": _FORCE_FINISH_PROMPT,
                        })

        # If we exhaust iterations without a finish call, force one
        return self._force_finish(messages, usage)

    def _force_finish(
        self,
        messages: list[dict],
        usage: dict[str, int],
    ) -> tuple[str, str]:
        """Force the LLM to emit a final verdict."""
        messages.append({
            "role": "user",
            "content": _FORCE_FINISH_PROMPT,
        })

        # Call with only the finish tool available
        finish_tool = [t for t in _TOOLS if t["function"]["name"] == "finish"]
        resp = self._llm_call(messages, tools=finish_tool)
        self._add_usage(usage, resp)

        assistant_msg = resp["message"]
        tool_calls = assistant_msg.get("tool_calls")

        if tool_calls:
            for tc in tool_calls:
                if tc["function"]["name"] == "finish":
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}
                    return (
                        fn_args.get("verdict", "UNCERTAIN"),
                        fn_args.get("reasoning", ""),
                    )

        # Last resort: parse from text
        text = assistant_msg.get("content", "") or ""
        verdict, reasoning = self._try_parse_verdict(text)
        if verdict:
            return verdict, reasoning

        return "UNCERTAIN", "Failed to extract verdict from ReAct loop."

    # ── Search ───────────────────────────────────────────────────────

    def _search(self, query: str) -> str:
        """Execute a web search and return concatenated snippets."""
        if self._serper_key:
            return _serper_search(query, self._serper_key, k=self.num_search_results)
        return _ddg_search(query, k=self.num_search_results)

    # ── LLM call via litellm ─────────────────────────────────────────

    def _llm_call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Single litellm completion with tool use.  Returns parsed response info."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2048,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = litellm.completion(**kwargs)

        msg = resp.choices[0].message

        # Build a serialisable message dict
        message_dict: dict = {"role": "assistant"}
        if msg.content:
            message_dict["content"] = msg.content
        if msg.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return {
            "message": message_dict,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }

    @staticmethod
    def _add_usage(total: dict[str, int], resp: dict) -> None:
        total["input_tokens"] += resp.get("input_tokens", 0)
        total["output_tokens"] += resp.get("output_tokens", 0)

    @staticmethod
    def _try_parse_verdict(text: str) -> tuple[str | None, str]:
        """Try to extract a verdict from plain text (fallback when no tool call)."""
        # Try JSON first
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if "verdict" in parsed:
                    return parsed["verdict"], parsed.get("reasoning", text[:500])
                if "label" in parsed:
                    return parsed["label"], parsed.get("reasoning", text[:500])
            except json.JSONDecodeError:
                pass

        # Try keyword matching
        upper = text.upper()
        for label in ("SUPPORT", "REFUTE", "UNCERTAIN"):
            if label in upper:
                return label, text[:500]

        return None, ""
