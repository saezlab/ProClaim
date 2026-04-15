"""
ReAct baseline — unconstrained agentic reasoning with tool use.

Uses LangGraph's ``create_react_agent`` for the Thought → Action → Observation
loop and ``ChatLiteLLM`` so that any litellm model string works (consistent
with the other baselines).

The agent has access to one search tool (configured at init time):
  - ``search_web(query)`` — web search via Serper or DuckDuckGo
  - ``search_papers(query)`` — Semantic Scholar academic paper search

The agent decides on its own when to stop — there is NO external sufficiency
signal.  This is the "why not just use a ReAct agent?" baseline.

Key comparison points vs Evidence Programming:
  - No structured evidence state (just flat conversation history)
  - No learned sufficiency classifier (LLM decides when to stop)
  - No gap-directed retrieval (LLM generates queries freely)
  - No cross-paper synthesis (implicit in LLM context)

Search backends:
  ``search_backend="web"`` (default, checked in order):
    1. If ``SERPER_API_KEY`` is set → Serper API (paid, higher quality).
    2. Otherwise → ``ddgs`` (free, DuckDuckGo search).
  ``search_backend="s2"``:
    Semantic Scholar relevance search (``S2_API_KEY`` optional but recommended).

Cost: 1–N LLM calls per claim (N ≤ max_steps) + searches.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import warnings
from pathlib import Path

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_litellm import ChatLiteLLM
from langgraph.prebuilt import create_react_agent

from baselines.shared.cost_tracker import CostTracker
from baselines.shared.llm import LLMBackend
from baselines.shared.label_utils import (
    normalize_label,
    validate_verdict,
    verdict_defs_block,
    verdict_names,
    verdict_or_str,
)
from baselines.shared.verdict import BaselineResult
from pkevolve.search.semantic_scholar import S2Client, S2RateLimitError

logger = logging.getLogger(__name__)

# Suppress the LangGraph deprecation warning about create_react_agent
warnings.filterwarnings("ignore", message=".*create_react_agent.*deprecated.*")

# ── Verdict labels (loaded from shared label_utils) ──────────────────

_VERDICT_NAMES: list[str] = verdict_names()
_VERDICT_OPTIONS = verdict_or_str()
_VERDICT_DEFS = verdict_defs_block()

def _build_system_prompt(search_backend: str) -> str:
    """Build the system prompt with the correct tool description."""
    if search_backend == "s2":
        tool_desc = (
            "- search_papers(query): Search Semantic Scholar for academic papers. "
            "Returns titles, abstracts, and metadata. Use specific queries "
            "targeting the entities and relationships in the claim."
        )
        source = "academic literature"
    else:
        tool_desc = (
            "- search_web(query): Search the web for scientific evidence. Use specific "
            "queries targeting the entities and relationships in the claim."
        )
        source = "the web"

    return f"""\
You are a scientific claim verification agent using the ReAct framework.

Your task: determine whether a scientific claim is {_VERDICT_OPTIONS} \
based on evidence you retrieve from {source}.

Verdict definitions:
{_VERDICT_DEFS}

Available tools:
{tool_desc}

Strategy:
1. Think step-by-step about what evidence you need.
2. Issue targeted search queries to find that evidence.
3. After each search result, assess whether you have enough evidence.
4. When confident, respond with your final verdict in the EXACT format:
   VERDICT: <label>
   REASONING: <one or two sentences citing key evidence>
"""


# ── Web search backends ──────────────────────────────────────────────

_SERPER_URL = "https://google.serper.dev"

# Module-level default; updated by ReActBaseline.__init__
_NUM_SEARCH_RESULTS = 3


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


# Serialize DuckDuckGo calls — ddgs hangs when called concurrently
# (LangGraph's ToolNode dispatches parallel tool calls).
_DDG_LOCK = threading.Lock()


def _ddg_search(query: str, k: int = 3) -> str:
    """Query DuckDuckGo via the ddgs package (free, no API key)."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("ddgs not installed. Run: pip install ddgs")

    snippets: list[str] = []
    try:
        with _DDG_LOCK:
            ddgs = DDGS()
            for result in ddgs.text(query, max_results=k):
                body = result.get("body", "")
                if body:
                    snippets.append(body)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)

    return " ".join(snippets) if snippets else "No relevant search results found."


def _do_search(query: str, k: int = 3, _max_retries: int = 3) -> str:
    """Execute a web search with retries on transient errors."""
    serper_key = os.environ.get("SERPER_API_KEY", "")
    last_exc: Exception | None = None
    for attempt in range(_max_retries):
        try:
            if serper_key:
                return _serper_search(query, serper_key, k=k)
            return _ddg_search(query, k=k)
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt * 2
            logger.warning(
                "Search error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, _max_retries, wait, exc,
            )
            time.sleep(wait)
    logger.error("Search failed after %d retries: %s", _max_retries, last_exc)
    return "Search temporarily unavailable. No results found."


# ── Semantic Scholar search backend ──────────────────────────────────

# Module-level client; lazy-initialised on first use.
_s2_client: S2Client | None = None


def _get_s2_client() -> S2Client:
    global _s2_client
    if _s2_client is None:
        _s2_client = S2Client()
    return _s2_client


def _s2_search(query: str, k: int = 3) -> str:
    """Search Semantic Scholar via S2Client and return formatted paper snippets."""
    try:
        papers = _get_s2_client().search(query, limit=k)
    except S2RateLimitError as exc:
        logger.error("S2 rate-limit exhausted: %s", exc)
        return "ERROR: Semantic Scholar rate-limit exhausted after retries. Try again later."
    if not papers:
        return "No relevant papers found on Semantic Scholar."
    return _format_s2_papers(papers)


def _format_s2_papers(papers: list[dict]) -> str:
    """Format S2 paper dicts into numbered evidence passages."""
    parts: list[str] = []
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            abstract = "(no abstract available)"

        authors = paper.get("authors") or []
        author_str = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_str += " et al."

        ext_ids = paper.get("externalIds") or {}
        pmid = ext_ids.get("PubMed", "")
        pmid_str = f"  PMID: {pmid}" if pmid else ""

        cites = paper.get("citationCount", "")
        cite_str = f"  Citations: {cites}" if cites else ""

        parts.append(
            f"[{i}] {title} ({year}) — {author_str}{pmid_str}{cite_str}\n{abstract}"
        )
    return "\n\n".join(parts)


# ── LangChain tool (module-level, used by all instances) ─────────────

@lc_tool
def search_web(query: str) -> str:
    """Search the web for scientific evidence relevant to verifying a claim.

    Use specific, targeted queries mentioning key entities and relationships
    from the claim.
    """
    return _do_search(query, k=_NUM_SEARCH_RESULTS)


@lc_tool
def search_papers(query: str) -> str:
    """Search Semantic Scholar for academic papers relevant to verifying a claim.

    Returns paper titles, authors, abstracts, and metadata. Use specific,
    targeted queries mentioning key entities and relationships from the claim.
    """
    return _s2_search(query, k=_NUM_SEARCH_RESULTS)


# ── Baseline class ───────────────────────────────────────────────────


class ReActBaseline:
    """Verify claims via LangGraph's ReAct agent with LiteLLM backend.

    Parameters
    ----------
    llm:
        Shared ``LLMBackend`` instance.  Model and temperature are read
        from it to configure the underlying ``ChatLiteLLM``.
    max_steps:
        Maximum number of agent steps (LLM calls) before the agent must stop.
    num_search_results:
        Number of search results per query.
    search_backend:
        ``"web"`` (default) for Serper/DuckDuckGo, ``"s2"`` for Semantic
        Scholar.
    """

    def __init__(
        self,
        llm: LLMBackend,
        *,
        max_steps: int = 10,
        num_search_results: int = 3,
        search_backend: str = "web",
    ) -> None:
        if search_backend not in ("web", "s2"):
            raise ValueError(f"search_backend must be 'web' or 's2', got {search_backend!r}")

        self.model = llm.model
        self.max_steps = max_steps
        self.num_search_results = num_search_results
        self.temperature = llm.temperature
        self.search_backend = search_backend
        self.name = "react"
        self.log_dir: Path | None = None

        # Update module-level default for search results count
        global _NUM_SEARCH_RESULTS
        _NUM_SEARCH_RESULTS = num_search_results

        if search_backend == "s2":
            self._search_backend = "s2"
            tools = [search_papers]
        else:
            self._search_backend = "serper" if os.environ.get("SERPER_API_KEY") else "ddg"
            tools = [search_web]
        logger.info("ReAct search backend: %s", self._search_backend)

        # Build LangChain LLM via ChatLiteLLM
        self._llm = ChatLiteLLM(
            model=llm.model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
        )

        # Build the LangGraph ReAct agent
        system_prompt = _build_system_prompt(search_backend)
        self._agent = create_react_agent(
            self._llm,
            tools=tools,
            prompt=SystemMessage(content=system_prompt),
        )

        # Cost estimation
        model_key = self.model.split("/", 1)[-1] if "/" in self.model else self.model
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
        search_queries: list[str] = []
        total_in = 0
        total_out = 0

        try:
            # LangGraph recursion_limit: each "step" uses 2 graph nodes
            # (agent node + tool node), so limit = 2 * max_steps + 2 (extra
            # for the final answer turn).
            result = self._agent.invoke(
                {"messages": [HumanMessage(content=f"Verify this scientific claim:\n\n{claim}")]},
                config={"recursion_limit": 2 * self.max_steps + 2},
            )

            # Extract info from message history
            messages = result["messages"]
            verdict = None
            reasoning = ""

            for msg in messages:
                # Collect token usage from AIMessages
                if isinstance(msg, AIMessage) and msg.usage_metadata:
                    total_in += msg.usage_metadata.get("input_tokens", 0)
                    total_out += msg.usage_metadata.get("output_tokens", 0)

                # Collect search queries from tool calls
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] in ("search_web", "search_papers"):
                            q = tc["args"].get("query", "")
                            if q:
                                search_queries.append(q)

            # Parse verdict from the final AI message
            final_msg = _last_ai_message(messages)
            if final_msg and final_msg.content:
                verdict, reasoning = _parse_verdict(final_msg.content)

            if not verdict:
                verdict = "UNCERTAIN"
                reasoning = reasoning or "Agent did not emit a clear verdict."

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

        predicted = validate_verdict(verdict)

        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=normalize_label(gold_label),
            predicted_label=predicted,
            confidence=0.0,
            reasoning=reasoning[:1000] if reasoning else "",
            evidence=search_queries,
            input_tokens=total_in,
            output_tokens=total_out,
            cost_usd=(
                total_in / 1_000_000 * self._in_price
                + total_out / 1_000_000 * self._out_price
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )


# ── Verdict parsing helpers ──────────────────────────────────────────


def _last_ai_message(messages: list) -> AIMessage | None:
    """Return the last AIMessage without tool calls (i.e. the final answer)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg
    return None


def _parse_verdict(text: str) -> tuple[str | None, str]:
    """Extract verdict and reasoning from agent's final text response."""
    # Try "VERDICT: <label>" format
    m = re.search(r"VERDICT:\s*(\w+)", text, re.IGNORECASE)
    if m:
        label = m.group(1).upper()
        # Extract reasoning
        rm = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        reasoning = rm.group(1).strip() if rm else text[:500]
        return label, reasoning

    # Try keyword matching
    upper = text.upper()
    for label in _VERDICT_NAMES:
        if label in upper:
            return label, text[:500]

    return None, text[:500]
