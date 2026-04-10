"""
Semantic Scholar API client.

Provides search, recommendations (graph-expansion), and single-paper DOI
lookup.  All three methods return plain dicts matching the S2 response schema.
Callers are responsible for converting dicts to PaperRecord objects.

Rate limiting
─────────────
A global, thread-safe throttle enforces a minimum inter-request interval
across **all** ``S2Client`` instances in the process:

* Without an API key (public tier): 1.1 s between requests.
* With an API key (authenticated tier): 0.15 s between requests.

If a 429 is still returned, the client retries with exponential back-off
(2 s → 4 s → 8 s, up to ``_MAX_RETRIES`` attempts).  After exhausting
retries an ``S2RateLimitError`` is raised so callers can never silently
receive an empty result due to throttling.

Set ``S2_API_KEY`` in the environment to use the authenticated tier.
"""

import logging
import os
import threading
import time

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_REC_URL = "https://api.semanticscholar.org/recommendations/v1/papers/"
_S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

_S2_FIELDS = (
    "paperId,externalIds,title,abstract,authors,year,openAccessPdf,"
    "citationCount,url"
)

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_RETRIES = 3

# Minimum inter-request intervals (seconds)
_INTERVAL_ANON = 1.1   # public / anonymous tier
_INTERVAL_AUTH = 0.15   # authenticated tier (x-api-key set)

# ---------------------------------------------------------------------------
# Global, thread-safe rate limiter shared by ALL S2Client instances
# ---------------------------------------------------------------------------
_global_lock = threading.Lock()
_global_last_request: float = 0.0


class S2RateLimitError(Exception):
    """Raised when S2 returns 429 and all retries are exhausted."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class S2Client:
    """Thin wrapper around the Semantic Scholar Graph API.

    Args:
        api_key: Semantic Scholar API key.  Falls back to the ``S2_API_KEY``
            environment variable.  When absent, unauthenticated requests are
            made and a 1.1 s sleep is enforced between calls.

    Rate limiting is **global and thread-safe**: every ``S2Client`` instance
    in the process shares a single timestamp + lock so concurrent callers
    (different baselines, evidence-programming pipeline, …) can never
    exceed the tier's request rate.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("S2_API_KEY", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"x-api-key": self._api_key}
        return {}

    def _rate_limit(self) -> None:
        """Block until enough time has elapsed since the last global request."""
        global _global_last_request
        min_interval = _INTERVAL_AUTH if self._api_key else _INTERVAL_ANON
        with _global_lock:
            elapsed = time.monotonic() - _global_last_request
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            _global_last_request = time.monotonic()

    def _get(self, url: str, params: dict) -> dict | None:
        """Execute a GET request with rate-limiting, retry, and error handling.

        Raises:
            S2RateLimitError: If HTTP 429 persists after ``_MAX_RETRIES`` attempts.
        """
        for attempt in range(_MAX_RETRIES):
            self._rate_limit()
            try:
                resp = requests.get(
                    url, params=params, headers=self._headers(),
                    timeout=_DEFAULT_TIMEOUT,
                )
                if resp.status_code == 429:
                    backoff = 2 ** attempt * 2
                    logger.warning(
                        "S2 rate-limited (429) GET %s — retry %d/%d in %ds",
                        url, attempt + 1, _MAX_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                logger.warning("S2 GET %s → HTTP %s: %s", url, exc.response.status_code, exc)
                return None
            except requests.RequestException as exc:
                logger.warning("S2 GET %s → request error: %s", url, exc)
                return None
        raise S2RateLimitError(
            f"S2 GET {url} returned 429 after {_MAX_RETRIES} retries"
        )

    def _post(self, url: str, params: dict, body: dict) -> dict | None:
        """Execute a POST request with rate-limiting, retry, and error handling.

        Raises:
            S2RateLimitError: If HTTP 429 persists after ``_MAX_RETRIES`` attempts.
        """
        for attempt in range(_MAX_RETRIES):
            self._rate_limit()
            try:
                resp = requests.post(
                    url, params=params, json=body, headers=self._headers(),
                    timeout=_DEFAULT_TIMEOUT,
                )
                if resp.status_code == 429:
                    backoff = 2 ** attempt * 2
                    logger.warning(
                        "S2 rate-limited (429) POST %s — retry %d/%d in %ds",
                        url, attempt + 1, _MAX_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                logger.warning("S2 POST %s → HTTP %s: %s", url, exc.response.status_code, exc)
                return None
            except requests.RequestException as exc:
                logger.warning("S2 POST %s → request error: %s", url, exc)
                return None
        raise S2RateLimitError(
            f"S2 POST {url} returned 429 after {_MAX_RETRIES} retries"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self, query: str, limit: int = 10, fields: str | None = None,
    ) -> list[dict]:
        """Keyword search across the S2 corpus.

        Args:
            query: Free-text query string (same format as a PubMed query works).
            limit: Maximum number of results to return (S2 cap: 100).
            fields: Comma-separated S2 field names.  Defaults to
                :data:`_S2_FIELDS` which covers the union of fields used by
                the evidence-programming pipeline and the baselines.

        Returns:
            List of S2 paper dicts.  Empty list on error or no results.
        """
        params = {
            "query": query,
            "fields": fields or _S2_FIELDS,
            "limit": min(limit, 100),
        }
        data = self._get(_S2_SEARCH_URL, params)
        if data is None:
            return []
        return data.get("data", [])

    def recommendations(
        self,
        positive_ids: list[str],
        negative_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Fetch paper recommendations given positive and optional negative seeds.

        Seed IDs are expected in the format accepted by S2:
        ``PMID:<id>`` for PubMed papers or a bare S2 paper ID.

        Args:
            positive_ids: Papers the recommendations should be similar to.
            negative_ids: Papers the recommendations should differ from.
            limit: Maximum results (S2 cap: 500).

        Returns:
            List of S2 paper dicts.  Empty list on error or insufficient seeds.
        """
        if not positive_ids:
            logger.warning("S2 recommendations require at least one positive seed.")
            return []

        params = {"fields": _S2_FIELDS, "limit": min(limit, 500)}
        body: dict = {"positivePaperIds": positive_ids}
        if negative_ids:
            body["negativePaperIds"] = negative_ids

        data = self._post(_S2_REC_URL, params, body)
        if data is None:
            return []
        return data.get("recommendedPapers", [])

    def lookup_doi(self, doi: str) -> dict | None:
        """Look up a single paper by DOI.

        Args:
            doi: DOI string (e.g. ``10.1016/j.cell.2020.02.058``).

        Returns:
            S2 paper dict, or ``None`` if not found.
        """
        return self.lookup(f"DOI:{doi}")

    def lookup(
        self, paper_id: str, fields: str | None = None,
    ) -> dict | None:
        """Look up a single paper by any S2-accepted identifier.

        Args:
            paper_id: Identifier string, e.g. ``"DOI:10.1016/…"``,
                ``"PMID:12345678"``, ``"CorpusId:215416146"``, or a bare
                S2 paper ID.
            fields: Comma-separated S2 field names.  Defaults to
                :data:`_S2_FIELDS`.

        Returns:
            S2 paper dict, or ``None`` if not found.
        """
        url = _S2_PAPER_URL.format(paper_id=paper_id)
        params = {"fields": fields or _S2_FIELDS}
        return self._get(url, params)
