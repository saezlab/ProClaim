"""
Semantic Scholar API client.

Provides search, recommendations (graph-expansion), and single-paper DOI
lookup.  All three methods return plain dicts matching the S2 response schema.
Callers are responsible for converting dicts to PaperRecord objects.

Rate limiting
─────────────
Without an API key the S2 public tier allows roughly 5 000 requests/day.
A 1-second sleep is enforced between successive calls when no key is
configured.  Set ``S2_API_KEY`` in the environment to use the authenticated
tier (higher limits, no per-request sleep).
"""

import logging
import os
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
    "paperId,externalIds,title,abstract,authors,year,openAccessPdf"
)

_DEFAULT_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class S2Client:
    """Thin wrapper around the Semantic Scholar Graph API.

    Args:
        api_key: Semantic Scholar API key.  Falls back to the ``S2_API_KEY``
            environment variable.  When absent, unauthenticated requests are
            made and a 1-second sleep is enforced between calls.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("S2_API_KEY", "")
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"x-api-key": self._api_key}
        return {}

    def _rate_limit(self) -> None:
        """Enforce 1 req/s when running without an API key."""
        if self._api_key:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    def _get(self, url: str, params: dict) -> dict | None:
        """Execute a GET request with rate-limiting and error handling."""
        self._rate_limit()
        try:
            resp = requests.get(
                url, params=params, headers=self._headers(),
                timeout=_DEFAULT_TIMEOUT,
            )
            self._last_request_time = time.monotonic()
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            logger.warning("S2 GET %s → HTTP %s: %s", url, exc.response.status_code, exc)
            return None
        except requests.RequestException as exc:
            logger.warning("S2 GET %s → request error: %s", url, exc)
            return None

    def _post(self, url: str, params: dict, body: dict) -> dict | None:
        """Execute a POST request with rate-limiting and error handling."""
        self._rate_limit()
        try:
            resp = requests.post(
                url, params=params, json=body, headers=self._headers(),
                timeout=_DEFAULT_TIMEOUT,
            )
            self._last_request_time = time.monotonic()
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            logger.warning("S2 POST %s → HTTP %s: %s", url, exc.response.status_code, exc)
            return None
        except requests.RequestException as exc:
            logger.warning("S2 POST %s → request error: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Keyword search across the S2 corpus.

        Args:
            query: Free-text query string (same format as a PubMed query works).
            limit: Maximum number of results to return (S2 cap: 100).

        Returns:
            List of S2 paper dicts.  Empty list on error or no results.
        """
        params = {
            "query": query,
            "fields": _S2_FIELDS,
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
        url = _S2_PAPER_URL.format(paper_id=f"DOI:{doi}")
        params = {"fields": _S2_FIELDS}
        return self._get(url, params)
