"""
Layered full-text retrieval for scientific papers.

Provides ``fetch_full_text(pmid, doi=None) -> str | None`` with five
fallback tiers::

    Layer 1   — PMC Open Access XML       (NCBI E-utilities)
    Layer 1b  — Europe PMC REST API       (broader OA coverage)
    Layer 1.5 — Semantic Scholar OA PDF   (S2 openAccessPdf endpoint)
    Layer 2   — INDRA literature          (PMC + Elsevier + REACH readers)
    Layer 3   — Unpaywall + PDF           (OA PDF → pymupdf text extraction)
    Layer 4   — PubMed structured abstract (last-resort; always returns text)

Each layer is tried in order.  If a layer succeeds (returns ≥200 chars of
body text), the remaining layers are skipped.  Layer 4 (structured abstract)
is always attempted if all full-text layers fail and ensures the function
returns usable text rather than None for fact extraction.

Usage::

    from pkevolve.verification.full_text import fetch_full_text
    text = fetch_full_text("35562995", doi="10.1234/example")
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# Minimum length to accept as "real" body text (not a stub/error page)
_MIN_TEXT_LEN = 200

# Maximum chars to return (avoid blowing up context windows)
DEFAULT_MAX_CHARS = 50_000

# E-utilities base
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _title_overlap(title: str, text: str, threshold: float = 0.25) -> bool:
    """Check whether the first 3 000 chars of *text* contain enough
    words from *title* to credibly belong to the same paper."""
    if not title:
        return True  # no title to validate against
    title_words = {w for w in title.lower().split() if len(w) > 3}
    if not title_words:
        return True
    text_lower = text[:3000].lower()
    matches = sum(1 for w in title_words if w in text_lower)
    return (matches / len(title_words)) >= threshold


def _extract_reference_dois(root: ET.Element) -> list[str]:
    """Extract DOIs from JATS ``<back><ref-list>``.

    In JATS XML, references live outside ``<body>`` — they are in
    ``<back><ref-list><ref>...``.  DOIs appear as
    ``<pub-id pub-id-type="doi">10.xxxx/...</pub-id>``.

    Returns a list of DOI strings (e.g. ``["10.1234/foo", ...]``).
    """
    ref_list = root.find(".//back/ref-list")
    if ref_list is None:
        return []
    dois: list[str] = []
    for pub_id in ref_list.iter("pub-id"):
        if pub_id.get("pub-id-type") == "doi" and pub_id.text:
            dois.append(pub_id.text.strip())
    return dois


# ──────────────────────────────────────────────────────────────────────
# Layer 1:  PMC Open-Access XML  (NCBI E-utilities)
# ──────────────────────────────────────────────────────────────────────

def _fetch_pmc(pmid: str, title: str = "") -> tuple[Optional[str], list[str]]:
    """Convert PMID → PMCID via elink, then fetch + parse PMC XML.

    Returns ``(body_text, reference_dois)``.
    """
    try:
        # PMID → PMCID
        link_resp = requests.get(
            f"{_EUTILS_BASE}/elink.fcgi",
            params={"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "xml"},
            timeout=30,
        )
        link_resp.raise_for_status()
        link_root = ET.fromstring(link_resp.content)

        pmcid = None
        for link_set_db in link_root.findall(".//LinkSetDb"):
            link_name_el = link_set_db.find("LinkName")
            if link_name_el is not None and link_name_el.text == "pubmed_pmc":
                link_id_el = link_set_db.find("Link/Id")
                if link_id_el is not None and link_id_el.text:
                    pmcid = link_id_el.text
                break

        if not pmcid:
            logger.debug("No PMCID for PMID %s", pmid)
            return None, []

        # Fetch PMC XML
        fetch_resp = requests.get(
            f"{_EUTILS_BASE}/efetch.fcgi",
            params={"db": "pmc", "id": pmcid, "rettype": "xml", "retmode": "xml"},
            timeout=60,
        )
        fetch_resp.raise_for_status()
        fetch_root = ET.fromstring(fetch_resp.content)

        # Extract body text with structural awareness
        parts: list[str] = []

        # Try structured extraction from <body> first
        body = fetch_root.find(".//body")
        if body is not None:
            for elem in body.iter():
                if elem.tag in ("p", "title", "label"):
                    text = "".join(elem.itertext()).strip()
                    if text:
                        parts.append(text)

        # Fallback: extract from <abstract> + all <p> if body was empty
        if not parts:
            for elem in fetch_root.iter():
                if elem.tag in ("p", "title", "abstract") and elem.text:
                    parts.append(elem.text.strip())

        full_text = "\n\n".join(parts) if parts else ""
        if len(full_text) < _MIN_TEXT_LEN:
            logger.debug("PMC body empty/too short for PMID %s", pmid)
            return None, []

        # Cross-validate title overlap
        if not _title_overlap(title, full_text):
            logger.warning(
                "PMC text for PMID %s (PMC%s) failed title cross-validation; discarding.",
                pmid, pmcid,
            )
            return None, []

        ref_dois = _extract_reference_dois(fetch_root)

        logger.info("Layer 1 (PMC): retrieved %d chars for PMID %s", len(full_text), pmid)
        return full_text, ref_dois

    except Exception as exc:
        logger.debug("Layer 1 (PMC) failed for PMID %s: %s", pmid, exc)
        return None, []


# ──────────────────────────────────────────────────────────────────────
# Layer 1b:  Europe PMC  (REST API — better OA coverage than NCBI PMC)
# ──────────────────────────────────────────────────────────────────────

_EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _fetch_europepmc(pmid: str, title: str = "") -> tuple[Optional[str], list[str]]:
    """Fetch full text from Europe PMC REST API.

    Europe PMC often has OA full text for articles that NCBI PMC does not
    index under the Open Access subset (e.g. author manuscripts, EuropePMC
    grants).  Returns ``(body_text, reference_dois)``.
    """
    try:
        # Step 1: search by PMID to get PMCID and check OA status
        search_resp = requests.get(
            f"{_EUROPEPMC_API}/search",
            params={
                "query": f"EXT_ID:{pmid} AND SRC:MED",
                "resultType": "core",
                "format": "json",
            },
            timeout=15,
        )
        search_resp.raise_for_status()
        data = search_resp.json()
        results = data.get("resultList", {}).get("result", [])

        if not results:
            logger.debug("Europe PMC: no result for PMID %s", pmid)
            return None, []

        hit = results[0]
        pmcid = hit.get("pmcid")
        is_oa = hit.get("isOpenAccess") == "Y"

        if not pmcid:
            logger.debug("Europe PMC: no PMCID for PMID %s", pmid)
            return None, []

        if not is_oa:
            logger.debug("Europe PMC: PMID %s (PMC%s) not open access", pmid, pmcid)
            return None, []

        # Step 2: fetch full text XML
        ft_resp = requests.get(
            f"{_EUROPEPMC_API}/{pmcid}/fullTextXML",
            timeout=60,
        )
        if ft_resp.status_code != 200:
            logger.debug(
                "Europe PMC: fullTextXML returned %d for %s",
                ft_resp.status_code, pmcid,
            )
            return None, []

        root = ET.fromstring(ft_resp.content)

        # Extract body text
        parts: list[str] = []
        body = root.find(".//body")
        if body is not None:
            for elem in body.iter():
                if elem.tag in ("p", "title", "label"):
                    text = "".join(elem.itertext()).strip()
                    if text:
                        parts.append(text)

        full_text = "\n\n".join(parts) if parts else ""
        if len(full_text) < _MIN_TEXT_LEN:
            logger.debug("Europe PMC: body too short for PMID %s", pmid)
            return None, []

        if not _title_overlap(title, full_text):
            logger.warning(
                "Europe PMC text for PMID %s failed title cross-validation; discarding.",
                pmid,
            )
            return None, []

        ref_dois = _extract_reference_dois(root)

        logger.info(
            "Layer 1b (Europe PMC): retrieved %d chars for PMID %s",
            len(full_text), pmid,
        )
        return full_text, ref_dois

    except Exception as exc:
        logger.debug("Layer 1b (Europe PMC) failed for PMID %s: %s", pmid, exc)
        return None, []


# ──────────────────────────────────────────────────────────────────────
# DOI resolution (needed for Unpaywall when DOI is missing)
# ──────────────────────────────────────────────────────────────────────

def _resolve_doi(pmid: str) -> Optional[str]:
    """Resolve a DOI from a PMID via NCBI E-utilities (elink + efetch).

    Falls back to Europe PMC if NCBI doesn't have it.
    """
    # Try NCBI efetch first (ArticleIdList often has DOI)
    try:
        resp = requests.get(
            f"{_EUTILS_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=15,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        # Check ELocationID
        doi_elem = root.find(".//ELocationID[@EIdType='doi']")
        if doi_elem is not None and doi_elem.text:
            return doi_elem.text.strip()

        # Check ArticleIdList
        for aid in root.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                return aid.text.strip()
    except Exception as exc:
        logger.debug("DOI resolution via NCBI failed for PMID %s: %s", pmid, exc)

    # Fallback: Europe PMC
    try:
        resp = requests.get(
            f"{_EUROPEPMC_API}/search",
            params={
                "query": f"EXT_ID:{pmid} AND SRC:MED",
                "resultType": "lite",
                "format": "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
        if results and results[0].get("doi"):
            return results[0]["doi"]
    except Exception as exc:
        logger.debug("DOI resolution via Europe PMC failed for PMID %s: %s", pmid, exc)

    return None


# ──────────────────────────────────────────────────────────────────────
# Layer 1.5:  Semantic Scholar OA PDF  (openAccessPdf endpoint)
# ──────────────────────────────────────────────────────────────────────


def _fetch_semantic_scholar(pmid: str, title: str = "") -> Optional[str]:
    """Fetch full text via Semantic Scholar's ``openAccessPdf`` metadata.

    Looks up the paper by PMID via :class:`~pkevolve.search.semantic_scholar.S2Client`
    (inheriting process-wide rate-limiting and 429 retry), retrieves the
    OA PDF URL, downloads the PDF, and extracts body text with pymupdf.
    """
    from pkevolve.search.semantic_scholar import S2Client, S2RateLimitError

    try:
        client = S2Client()
        data = client.lookup(f"PMID:{pmid}", fields="openAccessPdf,title")
        if data is None:
            logger.debug("S2 returned no data for PMID %s", pmid)
            return None

        oa_pdf = data.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url")
        if not pdf_url:
            logger.debug("No S2 openAccessPdf for PMID %s", pmid)
            return None

        # Download the OA PDF (not an S2 API call — no rate limiting needed)
        pdf_resp = requests.get(
            pdf_url,
            timeout=60,
            headers={"User-Agent": "pkevolve/0.1 (scientific research tool)"},
        )
        if pdf_resp.status_code != 200:
            logger.debug(
                "S2 PDF download failed (%d) for PMID %s url=%s",
                pdf_resp.status_code, pmid, pdf_url,
            )
            return None

        # Extract text with pymupdf
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            import fitz as pymupdf  # type: ignore[no-redef]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_resp.content)
            tmp.flush()
            doc = pymupdf.open(tmp.name)
            pages_text: list[str] = [page.get_text() for page in doc]
            doc.close()

        full_text = "\n\n".join(pages_text)
        if len(full_text) < _MIN_TEXT_LEN:
            logger.debug("S2 PDF text too short (%d chars) for PMID %s", len(full_text), pmid)
            return None

        if not _title_overlap(title, full_text):
            logger.warning(
                "S2 PDF for PMID %s failed title cross-validation; discarding.", pmid
            )
            return None

        logger.info(
            "Layer 1.5 (Semantic Scholar): retrieved %d chars for PMID %s",
            len(full_text), pmid,
        )
        return full_text

    except S2RateLimitError:
        logger.warning("S2 rate-limit exhausted for PMID %s — skipping Layer 1.5", pmid)
        return None
    except Exception as exc:
        logger.debug("Layer 1.5 (Semantic Scholar) failed for PMID %s: %s", pmid, exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Layer 2:  INDRA literature  (pmc_client + elsevier_client)
# ──────────────────────────────────────────────────────────────────────

def _extract_text_from_xml(xml_text: str, content_type: str | None = None) -> str:
    """Extract readable text from XML returned by INDRA's get_full_text.

    Tries pmc_client.extract_text, pmc_client.extract_paragraphs,
    and elsevier_client.extract_paragraphs as fallbacks.
    """
    ctype = (content_type or "").lower()

    def _join_paragraphs(paragraphs: list[str] | None) -> str:
        if not paragraphs:
            return ""
        cleaned = [_clean_ws(p) for p in paragraphs if p and _clean_ws(p)]
        return "\n\n".join(cleaned)

    # If Elsevier content, try Elsevier parser first
    if "elsevier" in ctype:
        try:
            from indra.literature import elsevier_client
            text = _join_paragraphs(elsevier_client.extract_paragraphs(xml_text))
            if text:
                return text
        except Exception:
            pass

    # PMC parser
    try:
        from indra.literature import pmc_client
        text = pmc_client.extract_text(xml_text)
        text = (text or "").strip()
        if text:
            return text
    except Exception:
        pass

    try:
        from indra.literature import pmc_client
        text = _join_paragraphs(pmc_client.extract_paragraphs(xml_text))
        if text:
            return text
    except Exception:
        pass

    # Elsevier as fallback if not tried yet
    if "elsevier" not in ctype:
        try:
            from indra.literature import elsevier_client
            text = _join_paragraphs(elsevier_client.extract_paragraphs(xml_text))
            if text:
                return text
        except Exception:
            pass

    # Last-resort: raw itertext from XML
    if "elsevier" in ctype:
        return ""
    try:
        root = ET.fromstring(xml_text)
        return _clean_ws(" ".join(t for t in root.itertext() if t and t.strip()))
    except Exception:
        return ""


def _fetch_indra(pmid: str, title: str = "") -> Optional[str]:
    """Use INDRA's ``get_full_text`` for broader publisher coverage."""
    try:
        from indra.literature import get_full_text
    except ImportError:
        logger.debug("INDRA not installed — skipping Layer 2.")
        return None

    try:
        content, content_type = get_full_text(pmid, "pmid", preferred_content_type="text/xml")
        content_type = str(content_type or "unknown")
        raw = (
            content.decode("utf-8", errors="replace")
            if isinstance(content, bytes)
            else str(content or "")
        )

        if not raw.strip():
            return None

        # Only parse XML; if it's plain text already, return it
        if content_type.lower().endswith("xml") and content_type.lower() != "abstract":
            full_text = _extract_text_from_xml(raw, content_type=content_type)
        elif "text" in content_type.lower():
            full_text = raw.strip()
        else:
            full_text = raw.strip()

        if not full_text or len(full_text) < _MIN_TEXT_LEN:
            return None

        if not _title_overlap(title, full_text):
            logger.warning("INDRA text for PMID %s failed title cross-validation; discarding.", pmid)
            return None

        logger.info("Layer 2 (INDRA): retrieved %d chars for PMID %s", len(full_text), pmid)
        return full_text

    except Exception as exc:
        logger.debug("Layer 2 (INDRA) failed for PMID %s: %s", pmid, exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Layer 3:  Unpaywall + PDF  (pymupdf)
# ──────────────────────────────────────────────────────────────────────


def _get_unpaywall_email() -> str:
    """Resolve Unpaywall email from config (env / .env / default)."""
    try:
        from pkevolve.verification.config import get_settings
        return get_settings().api.unpaywall_email
    except Exception:
        return os.getenv("UNPAYWALL_EMAIL", "pkevolve@example.com")


def _fetch_unpaywall_pdf(doi: str, title: str = "") -> Optional[str]:
    """Find an open-access PDF via Unpaywall and extract text with pymupdf."""
    if not doi:
        return None

    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": _get_unpaywall_email()},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.debug("Unpaywall returned %d for DOI %s", resp.status_code, doi)
            return None

        data = resp.json()
        # Find best OA location with a PDF URL
        pdf_url = None
        best_oa = data.get("best_oa_location") or {}
        pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")

        if not pdf_url:
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    pdf_url = loc["url_for_pdf"]
                    break

        if not pdf_url:
            logger.debug("No OA PDF URL found via Unpaywall for DOI %s", doi)
            return None

        # Download the PDF
        pdf_resp = requests.get(pdf_url, timeout=60, headers={
            "User-Agent": "pkevolve/0.1 (mailto:pkevolve@example.com)",
        })
        if pdf_resp.status_code != 200:
            logger.debug("PDF download failed (%d) for %s", pdf_resp.status_code, pdf_url)
            return None

        # Extract text with pymupdf
        try:
            import pymupdf  # noqa: F811 (fitz/pymupdf)
        except ImportError:
            import fitz as pymupdf  # type: ignore[no-redef]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_resp.content)
            tmp.flush()
            doc = pymupdf.open(tmp.name)
            pages_text: list[str] = []
            for page in doc:
                pages_text.append(page.get_text())
            doc.close()

        full_text = "\n\n".join(pages_text)
        full_text = _clean_ws(full_text) if len(full_text) < 500 else full_text

        if len(full_text) < _MIN_TEXT_LEN:
            logger.debug("PDF text too short (%d chars) for DOI %s", len(full_text), doi)
            return None

        if not _title_overlap(title, full_text):
            logger.warning("PDF text for DOI %s failed title cross-validation; discarding.", doi)
            return None

        logger.info("Layer 3 (Unpaywall PDF): retrieved %d chars for DOI %s", len(full_text), doi)
        return full_text

    except Exception as exc:
        logger.debug("Layer 3 (Unpaywall PDF) failed for DOI %s: %s", doi, exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Layer 4:  PubMed structured abstract  (last-resort fallback)
# ──────────────────────────────────────────────────────────────────────


def _fetch_pubmed_structured_abstract(pmid: str) -> Optional[str]:
    """Fetch the PubMed structured abstract as a last-resort text source.

    Returns the abstract with section labels (BACKGROUND, METHODS,
    RESULTS, CONCLUSIONS, etc.) when the journal uses IMRAD structure,
    or plain abstract text otherwise.  Also prepends the article title
    so the LLM has maximum context for fact extraction.

    This layer never does network I/O beyond a single lightweight efetch
    call.  It is always attempted when all full-text layers fail so that
    ``fetch_full_text`` returns usable text rather than ``None``.
    """
    try:
        resp = requests.get(
            f"{_EUTILS_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=15,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        parts: list[str] = []

        # Article title
        title_el = root.find(".//ArticleTitle")
        if title_el is not None:
            title_text = "".join(title_el.itertext()).strip()
            if title_text:
                parts.append(f"Title: {title_text}")

        # Abstract — may have multiple <AbstractText> with Label attributes
        abstract_texts = root.findall(".//AbstractText")
        for ab in abstract_texts:
            label = ab.get("Label", "").strip()
            text = "".join(ab.itertext()).strip()
            if not text:
                continue
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

        if not parts or (len(parts) == 1 and parts[0].startswith("Title:")):
            # No abstract found
            return None

        structured = "[Abstract only — full text unavailable]\n\n" + "\n\n".join(parts)
        logger.info(
            "Layer 4 (PubMed abstract): retrieved %d chars for PMID %s",
            len(structured), pmid,
        )
        return structured

    except Exception as exc:
        logger.debug("Layer 4 (PubMed abstract) failed for PMID %s: %s", pmid, exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def fetch_full_text(
    pmid: str,
    *,
    doi: Optional[str] = None,
    title: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[Optional[str], list[str]]:
    """Retrieve full text for a paper through a layered fallback chain.

    Tries in order:
        1.   PMC Open Access XML       (NCBI E-utilities, free)
        1b.  Europe PMC REST API       (broader OA coverage incl. author manuscripts)
        1.5. Semantic Scholar OA PDF   (S2 openAccessPdf endpoint)
        2.   INDRA literature          (broader publisher coverage incl. Elsevier)
        3.   Unpaywall + PDF           (OA PDF download & text extraction)
        4.   PubMed structured abstract (last resort — always returns text)

    If *doi* is not provided and layers 1–2 fail, attempts DOI resolution
    via NCBI/Europe PMC before trying Unpaywall (which requires a DOI).

    Returns ``(text, reference_dois)`` where *text* is the extracted body
    (truncated to *max_chars*) and *reference_dois* is a list of DOI
    strings from the JATS ``<back><ref-list>`` (populated only by
    Layers 1 and 1b which parse JATS XML).

    Args:
        pmid: PubMed identifier.
        doi:  Digital Object Identifier (needed for Unpaywall, Layer 3).
        title: Paper title for cross-validation against retrieved text.
        max_chars: If the retrieved text exceeds this length a warning is
            logged, but the full untruncated text is still returned.
    """
    def _maybe_warn_and_return(
        text: str, ref_dois: list[str]
    ) -> tuple[str, list[str]]:
        if len(text) > max_chars:
            logger.warning(
                "Full text for PMID %s is %d chars, exceeding max_chars=%d; "
                "returning untruncated.",
                pmid, len(text), max_chars,
            )
        return text, ref_dois

    # Layer 1: NCBI PMC
    text, ref_dois = _fetch_pmc(pmid, title=title)
    if text and len(text) >= _MIN_TEXT_LEN:
        return _maybe_warn_and_return(text, ref_dois)

    # Layer 1b: Europe PMC (often has OA full text NCBI doesn't)
    text, ref_dois = _fetch_europepmc(pmid, title=title)
    if text and len(text) >= _MIN_TEXT_LEN:
        return _maybe_warn_and_return(text, ref_dois)

    # Layer 1.5: Semantic Scholar OA PDF
    text = _fetch_semantic_scholar(pmid, title=title)
    if text and len(text) >= _MIN_TEXT_LEN:
        return _maybe_warn_and_return(text, [])

    # Layer 2: INDRA
    text = _fetch_indra(pmid, title=title)
    if text and len(text) >= _MIN_TEXT_LEN:
        return _maybe_warn_and_return(text, [])

    # Resolve DOI if missing (needed for Unpaywall)
    if not doi:
        doi = _resolve_doi(pmid)
        if doi:
            logger.info("Resolved DOI %s for PMID %s", doi, pmid)

    # Layer 3: Unpaywall + PDF
    text = _fetch_unpaywall_pdf(doi or "", title=title)
    if text and len(text) >= _MIN_TEXT_LEN:
        return _maybe_warn_and_return(text, [])

    # Layer 4: PubMed structured abstract (last resort — ensures we never
    # return None and silently discard a paper from fact extraction)
    logger.info(
        "Full-text layers 1–3 failed for PMID %s (DOI: %s); falling back to abstract",
        pmid, doi,
    )
    text = _fetch_pubmed_structured_abstract(pmid)
    if text:
        return _maybe_warn_and_return(text, [])

    logger.warning("All layers including abstract failed for PMID %s", pmid)
    return None, []
