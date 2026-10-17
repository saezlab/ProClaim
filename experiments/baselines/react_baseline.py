"""ReAct baseline — unconstrained agentic reasoning with tool use.

LangGraph ``create_react_agent`` over ``ChatLiteLLM``. The agent decides on its
own when to stop: no sufficiency signal, no structured evidence state, no
gap-directed retrieval. This is the "why not just use a ReAct agent?" baseline.

Search backends: ``web`` (DuckDuckGo, default), ``s2`` (Semantic Scholar), and
``pubmed_s2`` (PubMed + S2), the last mirroring ProClaim's own dual-source
retrieval so a gap vs. ProClaim can't be blamed on paper access.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import create_react_agent

from baselines.shared.cost_tracker import CostTracker
from baselines.shared import prompt_cache
from baselines.shared.llm import LLMBackend
from baselines.shared.logging_utils import write_claim_log
from baselines.shared.label_utils import (
    normalize_label,
    validate_verdict,
    verdict_defs_block,
    verdict_names,
    verdict_or_str,
)
from baselines.shared.search_utils import (
    ddg_text_search_with_retry,
    format_ddg_body_results,
    format_pubmed_results,
    format_s2_detailed_results,
    record_retrieved,
    retrieved_documents_sink,
)
from baselines.shared.verdict import BaselineResult
from proclaim.search.custom_pubmed import (
    PUBMED_MAX_RETRIES,
    RelevancePubMedSearcher,
    search_with_retry,
)
from proclaim.search.semantic_scholar import S2Client, S2RateLimitError

logger = logging.getLogger(__name__)

# Suppress the LangGraph deprecation warning about create_react_agent
warnings.filterwarnings("ignore", message=".*create_react_agent.*deprecated.*")

# ── Verdict labels (loaded from shared label_utils) ──────────────────

_VERDICT_NAMES: list[str] = verdict_names()
_VERDICT_OPTIONS = verdict_or_str()
_VERDICT_DEFS = verdict_defs_block()

def _build_system_prompt(search_backend: str, reasoning_style: str = "brief") -> str:
    """Build the system prompt with the correct tool description.

    ``reasoning_style`` controls only the REASONING instruction line:
      "brief"   (default, unchanged) — one or two sentences.
      "summary" — a full paragraph summarising the key evidence with
                  citations, so REASONING is itself reusable downstream as an
                  evidence summary rather than a one-line justification.
    """
    if search_backend == "s2":
        tool_desc = (
            "- search_papers(query): Search Semantic Scholar for academic papers. "
            "Returns titles, abstracts, and metadata. Use specific queries "
            "targeting the entities and relationships in the claim."
        )
        source = "academic literature"
    elif search_backend == "pubmed_s2":
        tool_desc = (
            "- search_pubmed(query): Search PubMed for academic papers. This is your "
            "primary source for biomedical literature — use it first. Returns titles, "
            "abstracts, and metadata.\n"
            "- search_papers(query): Search Semantic Scholar for academic papers. Use "
            "this to complement PubMed — it additionally covers preprints (e.g. "
            "bioRxiv) and non-MEDLINE sources PubMed does not index. Returns titles, "
            "abstracts, and metadata."
        )
        source = "academic literature (PubMed + Semantic Scholar)"
    else:
        tool_desc = (
            "- search_web(query): Search the web for scientific evidence. Use specific "
            "queries targeting the entities and relationships in the claim."
        )
        source = "the web"

    reasoning_line = (
        "   REASONING: <summarise the key evidence, with references to the "
        "citation(s) it came from, in one paragraph>"
        if reasoning_style == "summary" else
        "   REASONING: <one or two sentences citing key evidence>"
    )
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
{reasoning_line}
"""


# ── Web search backend ───────────────────────────────────────────────

# Module-level default; updated by ReActBaseline.__init__ from --top-k
_NUM_SEARCH_RESULTS = 5


def _do_search(query: str, k: int = 3, _max_retries: int = 3) -> str:
    """Execute a web search with retries on transient errors."""
    results = ddg_text_search_with_retry(query, k=k, max_retries=_max_retries)
    record_retrieved([
        {
            "query": query,
            "source": "web",
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in results
    ])
    return format_ddg_body_results(
        results,
        empty_message="Search temporarily unavailable. No results found.",
    )


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
    record_retrieved([
        {
            "query": query,
            "source": "s2",
            "title": p.get("title", ""),
            "pmid": (p.get("externalIds") or {}).get("PubMed", ""),
            "doi": (p.get("externalIds") or {}).get("DOI", ""),
            "s2_id": p.get("paperId", ""),
            "year": p.get("year", ""),
        }
        for p in papers
    ])
    if not papers:
        return "No relevant papers found on Semantic Scholar."
    extractor = _fact_extractor.get()
    if extractor is None:
        return format_s2_detailed_results(papers)
    return extractor.render(papers, [
        {
            "paper_id": (p.get("externalIds") or {}).get("PubMed") or f"S2:{p.get('paperId', '')}",
            "doi": (p.get("externalIds") or {}).get("DOI"),
            "title": p.get("title") or "",
            "abstract": p.get("abstract") or "",
        }
        for p in papers
    ], format_s2_detailed_results)


# ── PubMed search backend ────────────────────────────────────────────

# Module-level searcher; lazy-initialised on first use.
_pubmed_searcher: RelevancePubMedSearcher | None = None


def _get_pubmed_searcher() -> RelevancePubMedSearcher:
    global _pubmed_searcher
    if _pubmed_searcher is None:
        _pubmed_searcher = RelevancePubMedSearcher()
    return _pubmed_searcher


def _pubmed_search(query: str, k: int = 3, _max_retries: int = PUBMED_MAX_RETRIES) -> str:
    """Search PubMed via RelevancePubMedSearcher, with retries on transient errors.

    Retry policy is shared with ProClaim's own agent via ``search_with_retry``
    (6 attempts, ~62s total backoff), so neither side can drift from the other.
    """
    papers = search_with_retry(_get_pubmed_searcher(), query, k, max_retries=_max_retries)
    record_retrieved([
        {
            "query": query,
            "source": "pubmed",
            "title": p.title or "",
            "pmid": p.paper_id or "",
            "doi": getattr(p, "doi", "") or "",
            "url": p.url or "",
            "year": p.published_date.year if getattr(p, "published_date", None) else "",
        }
        for p in papers
    ])
    if not papers:
        return "No relevant papers found on PubMed."
    extractor = _fact_extractor.get()
    if extractor is None:
        return format_pubmed_results(papers)
    return extractor.render(papers, [
        {
            "paper_id": p.paper_id or "",
            "doi": getattr(p, "doi", None),
            "title": p.title or "",
            "abstract": p.abstract or "",
        }
        for p in papers
    ], format_pubmed_results)


# ── Per-result fact extraction (ReAct + subclaims + facts - state) ───


class ResultFactExtractor:
    """ProClaim's Qwen fact extraction applied to every search result, with no evidence state.

    Each result's full text comes through ProClaim's fetch chain (``fetch_full_text``: PMC OA,
    Europe PMC, S2 OA PDF, INDRA, Unpaywall, abstract fallback) and goes to ProClaim's
    ``extract_facts`` with the claim and its subclaims. The facts are printed under the
    result's abstract. Nothing is merged or remembered across turns: a paper returned again
    is shown again with the same facts. The caches below only save repeated fetches and calls.
    """

    def __init__(self, llm, max_workers: int = 5) -> None:
        self._llm = llm
        self._max_workers = max_workers
        self._facts: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()
        self.claim = ""
        self.subclaims: list[str] = []
        self.calls = 0
        self.full_text_hits = 0

    def for_claim(self, claim: str, subclaims: list[str] | None) -> "ResultFactExtractor":
        """A per-claim view sharing the model and the full-text cache."""
        view = ResultFactExtractor(self._llm, self._max_workers)
        view.claim, view.subclaims = claim, list(subclaims or [])
        return view

    def _facts_block(self, meta: dict) -> str:
        from proclaim.verification.subagents import extract_facts

        key = (self.claim, meta["paper_id"])
        with self._lock:
            if key in self._facts:
                return self._facts[key]
        text, full = _paper_text(meta["paper_id"], meta["doi"], meta["title"], meta["abstract"])
        if not text:
            block = "   Extracted facts: none (no text available)."
        else:
            try:
                facts = extract_facts(
                    llm=self._llm, paper_text=text, claim=self.claim,
                    subclaims=self.subclaims, source_pmid=meta["paper_id"],
                )
            except Exception as exc:  # one failed paper must not end the claim
                logger.warning("Fact extraction failed for %s: %s", meta["paper_id"], exc)
                facts = []
            source = "from the full text" if full else "from the abstract; full text unavailable"
            if facts:
                block = f"   Extracted facts ({source}):\n" + "\n".join(
                    f"   - [{f.stance}] {f.text} (conf {float(f.confidence):.2f})" for f in facts
                )
            else:
                block = f"   Extracted facts ({source}): none relevant to the claim."
            with self._lock:
                self.calls += 1
                self.full_text_hits += int(full)
        with self._lock:
            self._facts[key] = block
        return block

    def render(self, papers: list, metas: list[dict], formatter) -> str:
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            blocks = list(pool.map(self._facts_block, metas))
        parts = []
        for i, (paper, block) in enumerate(zip(papers, blocks), 1):
            entry = re.sub(r"^\[1\]", f"[{i}]", formatter([paper]))
            parts.append(f"{entry}\n{block}")
        return "\n\n".join(parts)


@functools.lru_cache(maxsize=512)
def _paper_text(paper_id: str, doi: str | None, title: str, abstract: str) -> tuple[str, bool]:
    """Full text via ProClaim's chain, else the abstract; the flag says whether it is full text."""
    from proclaim.verification.full_text import fetch_full_text

    text = None
    if paper_id:
        try:
            text, _ = fetch_full_text(paper_id, doi=doi, title=title, existing_abstract=abstract)
        except Exception as exc:
            logger.warning("Full-text fetch failed for %s: %s", paper_id, exc)
    if text:
        # Same test ProClaim's extraction step logs: the fetch chain's last layer returns an abstract.
        full = not text.startswith("[Abstract only") and len(text) > 2000
        return text, full
    fallback = f"{title}\n\n{abstract}".strip()
    return fallback, False


# Set per claim in ReActBaseline.verify; the search tools read it.
_fact_extractor: ContextVar[ResultFactExtractor | None] = ContextVar("react_fact_extractor", default=None)


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


@lc_tool
def search_pubmed(query: str) -> str:
    """Search PubMed for academic papers relevant to verifying a claim.

    This is the primary biomedical literature source — use it first. Returns
    paper titles, authors, abstracts, and metadata. Use specific, targeted
    queries mentioning key entities and relationships from the claim.
    """
    return _pubmed_search(query, k=_NUM_SEARCH_RESULTS)


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
        Default 25.
    num_search_results:
        Number of search results per query (driven by ``--top-k``).
    search_backend:
        ``"web"`` (default) for DuckDuckGo, ``"s2"`` for Semantic
        Scholar, ``"pubmed_s2"`` for PubMed + Semantic Scholar together.
    reasoning_style:
        ``"brief"`` (default, unchanged) or ``"summary"`` — see
        ``_build_system_prompt``'s docstring.
    subclaims:
        Optional ``{claim_id: [subclaim, ...]}``. When a claim has a non-empty
        entry, the subclaims are listed in the first user message (the
        "ReAct (with decomposed claims)" arm); everything else is unchanged.
    fact_extractor:
        Optional ``ResultFactExtractor``. When given, every search result is
        followed by the Qwen facts extracted from its full text (the "ReAct +
        subclaims + facts - state" arm); no evidence state is kept.
    """

    _VALID_BACKENDS = ("web", "s2", "pubmed_s2")

    def __init__(
        self,
        llm: LLMBackend,
        *,
        max_steps: int = 25,
        num_search_results: int = 5,
        search_backend: str = "web",
        reasoning_style: str = "brief",
        subclaims: dict[str, list[str]] | None = None,
        fact_extractor: ResultFactExtractor | None = None,
    ) -> None:
        if search_backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"search_backend must be one of {self._VALID_BACKENDS}, got {search_backend!r}"
            )

        self.model = llm.model
        self.max_steps = max_steps
        self.num_search_results = num_search_results
        self.temperature = llm.temperature
        self.search_backend = search_backend
        self.name = "react"
        self.log_dir: Path | None = None
        self.subclaims = subclaims
        self.fact_extractor = fact_extractor

        # Update module-level default for search results count
        global _NUM_SEARCH_RESULTS
        _NUM_SEARCH_RESULTS = num_search_results

        if search_backend == "s2":
            self._search_backend = "s2"
            tools = [search_papers]
        elif search_backend == "pubmed_s2":
            self._search_backend = "pubmed_s2"
            tools = [search_pubmed, search_papers]
        else:
            self._search_backend = "ddg"
            tools = [search_web]
        logger.info("ReAct search backend: %s", self._search_backend)

        # Build LangChain LLM via ChatLiteLLM
        # Delegate to LLMBackend so model, temperature, token budget and provider
        # knobs (reasoning effort, LLM_EXTRA_BODY) come from ONE place. Building a
        # chat model here independently is what let ReAct drift from complete().
        self._llm = llm.chat_model()

        # Build the LangGraph ReAct agent
        system_prompt = _build_system_prompt(search_backend, reasoning_style)
        self._system_prompt = system_prompt
        self._agent = create_react_agent(
            self._llm,
            tools=tools,
            prompt=SystemMessage(content=system_prompt),
        )

        # Cost estimation
        in_price, out_price = CostTracker.pricing_for(self.model)
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
        cache_read = 0
        cache_write = 0
        user_prompt = _build_user_prompt(claim, (self.subclaims or {}).get(claim_id))

        retrieved_documents: list[dict] = []
        token = retrieved_documents_sink.set(retrieved_documents)
        extractor = None
        if self.fact_extractor is not None:
            extractor = self.fact_extractor.for_claim(claim, (self.subclaims or {}).get(claim_id))
        fact_token = _fact_extractor.set(extractor)
        messages: list = []
        try:
            # LangGraph recursion_limit: each "step" uses 2 graph nodes
            # (agent node + tool node), so limit = 2 * max_steps + 2 (extra
            # for the final answer turn).
            result = self._agent.invoke(
                {"messages": [HumanMessage(content=user_prompt)]},
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
                    # Prompt caching: usage_metadata drops cache counts; the raw usage keeps them.
                    read, write = prompt_cache.usage_cache_tokens(msg.response_metadata.get("token_usage"))
                    cache_read += read
                    cache_write += write

                # Collect search queries from tool calls
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] in ("search_web", "search_papers", "search_pubmed"):
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
        finally:
            retrieved_documents_sink.reset(token)
            _fact_extractor.reset(fact_token)

        latency = time.monotonic() - t0

        retrieved_block = "\n".join(
            "  [{}] source={} query={!r} ".format(i, d.get("source"), d.get("query"))
            + " ".join(f"{k}={v!r}" for k, v in d.items() if k not in ("query", "source"))
            for i, d in enumerate(retrieved_documents, 1)
        )

        # Write per-claim log
        write_claim_log(
            self.log_dir,
            claim_id,
            [
                (f"claim_id: {claim_id}", ""),
                ("claim", claim),
                ("system prompt", self._system_prompt),
                ("user prompt", user_prompt),
                (f"verdict: {verdict}", ""),
                (f"searches ({len(search_queries)})", "\n".join(f"  Q: {query}" for query in search_queries)),
                (f"retrieved documents ({len(retrieved_documents)})", retrieved_block),
                ("reasoning", reasoning),
            ] + ([
                (f"fact extraction: {extractor.calls} papers read, {extractor.full_text_hits} from full text", ""),
                ("search results shown to the agent", "\n\n".join(
                    str(m.content) for m in messages if isinstance(m, ToolMessage)
                )),
            ] if extractor is not None else []),
        )

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
            cost_usd=prompt_cache.cached_cost(
                total_in, total_out, cache_read, cache_write, self._in_price, self._out_price,
            ),
            latency_seconds=latency,
            baseline_name=self.name,
            model=self.model,
        )


# ── Verdict parsing helpers ──────────────────────────────────────────


def _build_user_prompt(claim: str, subclaims: list[str] | None = None) -> str:
    """First user message; lists the subclaims when given (ReAct with decomposed claims)."""
    prompt = f"Verify this scientific claim:\n\n{claim}"
    if subclaims:
        listed = "\n".join(f"{i}. {s}" for i, s in enumerate(subclaims, 1))
        prompt += (
            "\n\nThe claim has been broken down into the following subclaims:\n"
            f"{listed}\n\n"
            "Use the subclaims to plan your searches. You share one search budget across all\n"
            "of them. Your final VERDICT is for the original claim as a whole, not for each subclaim."
        )
    return prompt


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
