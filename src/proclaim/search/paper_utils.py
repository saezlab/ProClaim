"""
Paper metadata utilities: PubMed identifiers and metadata, journal metrics and citation
counts from OpenAlex. Used by the sufficiency judge's feature extraction.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any
from xml.etree import ElementTree as ET
from datetime import datetime, date
from dataclasses import asdict
from dotenv import load_dotenv
import fitz  # PyMuPDF
from requests.adapters import HTTPAdapter

from urllib3.util.retry import Retry
import re

# ============================================================================
# Text Cleaning Utilities
# ============================================================================

def clean_text(text: str) -> str:
    """
    Standard text cleaning for LLM processing.
    Removes:
    - Metadata lines (Accesses, Citations, Metrics)
    - References/Bibliography sections
    - Markdown artifacts (links, images)
    - Numeric citations [1]
    - Excessive whitespace
    """
    if not text:
        return ""
    
    # Pre-clean: Remove common metadata lines from the top/body before splitting
    # Remove lines like "* 2734 Accesses", "* 5 Citations"
    text = re.sub(r'\n\s*\*\s*\d+\s*(Accesses|Citations|Altmetric|Mentions)[^\n]*', '', text, flags=re.IGNORECASE)
    # Remove "Explore all metrics"
    text = re.sub(r'\n\s*Explore all metrics[^\n]*', '', text, flags=re.IGNORECASE)
    
    # 1. Truncate at References/Bibliography/Citations/End-of-article sections
    stop_phrases = [
        "References", "Bibliography", "LITERATURE CITED", 
        "Acknowledgements", "Declarations", "Rights and permissions"
    ]
    
    for phrase in stop_phrases:
        # Pattern: Newline + optional whitespace/markdown + phrase + optional whitespace + newline/colon/dashes
        pattern = r'\n\s*(?:#{1,6}\s*)?' + re.escape(phrase) + r'\s*(?:\n|$|:)'
        parts = re.split(pattern, text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            text = parts[0]
            break

    # 2. Remove Markdown Links: [Text](URL) -> Text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 3. Remove Markdown Images: !\[...\]\(...\) -> ""
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)

    # 4. Remove numeric citation markers like [1], [1, 2], [1-3]
    text = re.sub(r'\[\s*\d+(?:,\s*\d+|[-–]\d+)*\s*\]', '', text)
    
    # 6. Normalize Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Load environment variables from .env file
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent.parent
load_dotenv(_project_root / ".env")

# ============================================================================
# HTTP Session with Retries
# ============================================================================
_DEFAULT_HTTP_TIMEOUT = float(os.environ.get("PROCLAIM_HTTP_TIMEOUT", "30"))


class _TimeoutSession(requests.Session):
    """Session with a default per-request timeout; callers here passed none,
    so a stalled connection could hang a subprocess until its command limit."""

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", _DEFAULT_HTTP_TIMEOUT)
        return super().request(method, url, **kwargs)


def get_session():
    """
    Return a requests Session with retry logic for 429 and 503 errors.

    Retry-After is ignored: OpenAlex answers 429 with several hundred seconds
    once its per-IP quota is spent, which stalled callers past their command
    limit. Backoff is capped at the 1-16 s schedule below.
    """
    session = _TimeoutSession()
    retry = Retry(
        total=5,
        backoff_factor=1, # 1s, 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        respect_retry_after_header=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ============================================================================
# PubMed Enhanced Search (with Citations & Journal Info)
# ============================================================================
def get_paper_identifiers(pmid: str) -> Dict[str, Optional[str]]:
    """
    Retrieve all identifiers and URLs for a given PubMed ID.

    Args:
        pmid: PubMed ID

    Returns:
        Dict containing:
            - pmid: PubMed ID
            - pmcid: PMC ID if available
            - doi: DOI if available
            - pubmed_url: PubMed web page URL
            - pmc_url: PMC web page URL if PMCID exists
            - doi_url: DOI resolver URL if DOI exists
    """
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml'
    }

    result = {
        'pmid': pmid,
        'pmcid': None,
        'doi': None,
        'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        'pmc_url': None,
        'doi_url': None
    }

    try:
        response = get_session().get(fetch_url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Find article
        article = root.find('.//PubmedArticle')
        if article is not None:
            # Get all article IDs (PMCID, DOI, etc.)
            article_ids = article.find('.//ArticleIdList')
            if article_ids is not None:
                for aid in article_ids.findall('ArticleId'):
                    id_type = aid.get('IdType')
                    if id_type == 'pmc':
                        result['pmcid'] = aid.text
                        result['pmc_url'] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{aid.text}/"
                    elif id_type == 'doi':
                        result['doi'] = aid.text
                        result['doi_url'] = f"https://doi.org/{aid.text}"

        return result

    except (ET.ParseError, requests.exceptions.XMLSyntaxError):
        # Could happen if API returns HTML (500/403) masquerading as success or empty body
        print(f"[WARN] Malformed XML from E-Utils for PMID {pmid}")
        return result
    except Exception as e:
        print(f"[ERROR] Failed to get identifiers for PMID {pmid}: {e}")
        return result


def get_pmcid_from_pmid(pmid: str) -> Optional[str]:
    """
    Retrieve PMCID for a given PubMed ID.

    Args:
        pmid: PubMed ID
    Returns:
        PMCID string (e.g., 'PMC1234567') if found, else None
    """
    identifiers = get_paper_identifiers(pmid)
    return identifiers.get('pmcid')


# ============================================================================
# OpenAlex Integration
# ============================================================================

def fetch_openalex_metrics(issn: str) -> Dict[str, Optional[float]]:
    """
    Fetch journal metrics from OpenAlex using ISSN.

    Args:
        issn: Journal ISSN

    Returns:
        Dict with impact_factor (2yr mean citedness) and h_index
    """
    metrics = {'impact_factor': None, 'h_index': None}
    try:
        oa_url = f"https://api.openalex.org/sources?filter=issn:{issn}"
        oa_res = get_session().get(oa_url)
        if oa_res.status_code == 200:
            data = oa_res.json()
            results = data.get('results', [])
            if results:
                source = results[0]
                if 'summary_stats' in source:
                    stats = source['summary_stats']
                    metrics['impact_factor'] = stats.get('2yr_mean_citedness')
                    metrics['h_index'] = stats.get('h_index')
    except Exception as e:
        print(f"[WARN] OpenAlex query failed for {issn}: {e}")

    return metrics


def get_journal_info_by_pmid(pmid: str) -> Dict[str, Any]:
    """
    Get detailed journal information for a single PubMed ID.

    Args:
        pmid: PubMed ID

    Returns:
        Dict with journal_name, issn, impact_factor, h_index
    """
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml'
    }

    try:
        response = get_session().get(fetch_url, params=params)
        root = ET.fromstring(response.content)

        article = root.find('.//PubmedArticle')
        if article is not None:
            journal = article.find('.//Journal')

            if journal is not None:
                title = journal.find('Title').text if journal.find('Title') is not None else None
                issn = journal.find('ISSN').text if journal.find('ISSN') is not None else None

                # Get OpenAlex metrics if ISSN exists
                metrics = {'impact_factor': None, 'h_index': None}
                if issn:
                    metrics = fetch_openalex_metrics(issn)

                return {
                    'journal_name': title,
                    'issn': issn,
                    'impact_factor': metrics['impact_factor'],
                    'h_index': metrics['h_index']
                }
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Failed to fetch journal info for {pmid}: {e}")
    except ET.ParseError:
        print(f"[WARN] Malformed XML for journal info {pmid}")
    except Exception as e:
        print(f"[ERROR] Failed to get journal info for PMID {pmid}: {e}")

    return {
        'journal_name': None,
        'issn': None,
        'impact_factor': None,
        'h_index': None
    }


# ============================================================================
# Citation Counts
# ============================================================================


# ============================================================================
# PDF Download Functions
# ============================================================================


# ============================================================================
# PMC Full-Text Retrieval
# ============================================================================


# ============================================================================
# Result Saving Utilities
# ============================================================================


# ============================================================================
# Example Usage Functions
# ============================================================================


    
# ============================================================================
# Comprehensive Paper Details Retrieval
# ============================================================================

def get_pubmed_metadata(pmid: str) -> Dict[str, Optional[str]]:
    """
    Retrieve basic metadata (Title, Abstract, Date) from PubMed.
    """
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml'
    }
    
    result = {
        'title': None,
        'abstract': None,
        'published_date': None
    }

    try:
        response = get_session().get(fetch_url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        article = root.find('.//PubmedArticle')
        if article:
            # Title
            title_node = article.find('.//ArticleTitle')
            if title_node is not None:
                result['title'] = title_node.text

            # Abstract
            abstract_texts = []
            for abstract in article.findall('.//Abstract/AbstractText'):
                if abstract.text:
                    label = abstract.get('Label')
                    if label:
                        abstract_texts.append(f"{label}: {abstract.text}")
                    else:
                        abstract_texts.append(abstract.text)
            if abstract_texts:
                result['abstract'] = "\n".join(abstract_texts)

            # Date (PubDate)
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                year = pub_date.find('Year')
                month = pub_date.find('Month')
                day = pub_date.find('Day')
                date_parts = []
                if year is not None: date_parts.append(year.text)
                if month is not None: date_parts.append(month.text)
                if day is not None: date_parts.append(day.text)
                result['published_date'] = "-".join(date_parts)

    except Exception as e:
        print(f"[ERROR] Failed to get PubMed metadata for {pmid}: {e}")

    return result

def get_openalex_citation_count(pmid: str) -> int:
    """
    Get citation count from OpenAlex for a given PMID.
    """
    url = f"https://api.openalex.org/works/pmid:{pmid}"
    try:
        response = get_session().get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('cited_by_count', 0)
    except Exception as e:
        print(f"[WARN] OpenAlex citation query failed for {pmid}: {e}")
    return 0


# ============================================================================
# PDF Text Extraction Functions
# ============================================================================


