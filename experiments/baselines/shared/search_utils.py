"""Shared search helpers for baseline implementations."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_DDG_LOCK = threading.Lock()


def ddg_text_search(query: str, k: int) -> list[dict[str, str]]:
    """Query DuckDuckGo via the ddgs package and return normalized result dicts."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("ddgs not installed. Run: pip install ddgs") from exc

    results: list[dict[str, str]] = []
    try:
        with _DDG_LOCK:
            ddgs = DDGS()
            for result in ddgs.text(query, max_results=k):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("href", ""),
                        "snippet": result.get("body", ""),
                    }
                )
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for %r: %s", query, exc)

    return results


def ddg_text_search_with_retry(query: str, k: int, max_retries: int = 3) -> list[dict[str, str]]:
    """Execute a DuckDuckGo search with retries on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return ddg_text_search(query, k=k)
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt * 2
            logger.warning(
                "Web search error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1,
                max_retries,
                wait,
                exc,
            )
            time.sleep(wait)
    logger.error("Web search failed after %d retries: %s", max_retries, last_exc)
    return []


def format_ddg_body_results(results: list[dict[str, str]], empty_message: str) -> str:
    """Format DDG results as a single body-only snippet string."""
    snippets = [result.get("snippet", "") for result in results if result.get("snippet", "")]
    return " ".join(snippets) if snippets else empty_message


def format_ddg_passages(results: list[dict[str, str]], empty_message: str) -> str:
    """Format DDG results as numbered title/url/snippet passages."""
    parts: list[str] = []
    for index, item in enumerate(results, 1):
        title = item.get("title", "Untitled")
        url = item.get("link", "")
        snippet = item.get("snippet", "")
        url_str = f"  {url}" if url else ""
        parts.append(f"[{index}] {title}{url_str}\n{snippet}")
    return "\n\n".join(parts) if parts else empty_message


def format_ddg_verbose_lines(results: list[dict[str, str]], empty_message: str) -> str:
    """Format DDG results as one line per result with title, snippet, and URL."""
    lines: list[str] = []
    for result in results:
        parts = [part for part in (result.get("title", ""), result.get("snippet", ""), result.get("link", "")) if part]
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines) if lines else empty_message


def format_s2_basic_results(papers: list[dict], empty_message: str = "No relevant papers found.") -> str:
    """Format Semantic Scholar results as numbered title/year/PMID passages."""
    parts: list[str] = []
    for index, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = (paper.get("abstract") or "").strip() or "(no abstract available)"
        ext_ids = paper.get("externalIds") or {}
        pmid = ext_ids.get("PubMed", "")
        pmid_text = f" PMID:{pmid}" if pmid else ""
        parts.append(f"[{index}] {title} ({year}){pmid_text}\n{abstract}")
    return "\n\n".join(parts) if parts else empty_message


def format_s2_passages(papers: list[dict], empty_message: str = "") -> str:
    """Format Semantic Scholar results as retrieval passages."""
    parts: list[str] = []
    for index, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = (paper.get("abstract") or "").strip() or "(no abstract available)"
        ext_ids = paper.get("externalIds") or {}
        pmid = ext_ids.get("PubMed", "")
        pmid_text = f"  PMID: {pmid}" if pmid else ""
        parts.append(f"[{index}] {title} ({year}){pmid_text}\n{abstract}")
    return "\n\n".join(parts) if parts else empty_message


def format_s2_detailed_results(
    papers: list[dict],
    empty_message: str = "No relevant papers found on Semantic Scholar.",
) -> str:
    """Format S2 paper dicts with authors and citation counts."""
    parts: list[str] = []
    for index, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "")
        abstract = (paper.get("abstract") or "").strip() or "(no abstract available)"
        authors = paper.get("authors") or []
        author_str = ", ".join(author.get("name", "") for author in authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        ext_ids = paper.get("externalIds") or {}
        pmid = ext_ids.get("PubMed", "")
        pmid_text = f"  PMID: {pmid}" if pmid else ""
        citations = paper.get("citationCount", "")
        citation_text = f"  Citations: {citations}" if citations else ""
        parts.append(
            f"[{index}] {title} ({year}) — {author_str}{pmid_text}{citation_text}\n{abstract}"
        )
    return "\n\n".join(parts) if parts else empty_message