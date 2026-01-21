"""
Unified paper search and retrieval utilities.

This module consolidates all useful paper search functions from test scripts:
- PubMed search with citations, journal info, and impact factors
- ArXiv search and download
- PubMed PDF download (via PMC)
- PMC full-text retrieval
- Journal metrics from OpenAlex
"""

import os
import json
import requests
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
load_dotenv()

# ============================================================================
# HTTP Session with Retries
# ============================================================================
def get_session():
    """
    Return a requests Session with retry logic for 429 and 503 errors.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1, # 1s, 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
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

def enrich_pubmed_papers(papers: List[Any]) -> List[Any]:
    """
    Enrich PubMed papers with:
    - Citation counts (PmcRefCount)
    - Journal name, ISSN, PMCID
    - Impact factor and h-index from OpenAlex

    Args:
        papers: List of paper objects from PubMedSearcher

    Returns:
        Enhanced papers with additional metadata attributes
    """
    try:
        ids = [p.paper_id for p in papers]
        if not ids:
            return papers

        # 1. Fetch full metadata from efetch
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_response = get_session().get(fetch_url, params=fetch_params)
        fetch_root = ET.fromstring(fetch_response.content)

        # 2. Fetch citation counts from esummary
        summary_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_response = get_session().get(summary_url, params=summary_params)
        summary_root = ET.fromstring(summary_response.content)

        # Parse citation counts
        citation_map = {}
        for doc in summary_root.findall('.//DocSum'):
            try:
                pmid = doc.find('.//Id').text
                pmc_ref_count = 0
                for item in doc.findall('.//Item'):
                    if item.get('Name') == 'PmcRefCount' and item.text:
                        try:
                            pmc_ref_count = int(item.text)
                        except ValueError:
                            pass
                        break
                citation_map[pmid] = pmc_ref_count
            except:
                pass

        # Parse journal info & PMCID
        journal_map = {}  # pmid -> {issn, title, pmcid}
        for article in fetch_root.findall('.//PubmedArticle'):
            try:
                pmid = article.find('.//PMID').text

                # Journal info
                journal = article.find('.//Journal')
                issn = None
                title = None
                if journal is not None:
                    issn = journal.find('ISSN').text if journal.find('ISSN') is not None else None
                    title = journal.find('Title').text if journal.find('Title') is not None else None

                # PMCID
                pmcid = None
                article_ids = article.find('.//ArticleIdList')
                if article_ids is not None:
                    for aid in article_ids.findall('ArticleId'):
                        if aid.get('IdType') == 'pmc':
                            pmcid = aid.text
                            break

                journal_map[pmid] = {'issn': issn, 'title': title, 'pmcid': pmcid}
            except:
                pass

        # OpenAlex cache to avoid duplicate queries
        issn_cache = {}

        # Enrich papers
        for paper in papers:
            # Add citation count
            paper.citations = citation_map.get(paper.paper_id, 0)

            # Add journal info
            j_info = journal_map.get(paper.paper_id)
            if j_info:
                paper.journal_name = j_info['title']
                paper.issn = j_info['issn']
                paper.pmcid = j_info['pmcid']

                # Fetch OpenAlex metrics if ISSN exists
                if paper.issn:
                    if paper.issn in issn_cache:
                        metrics = issn_cache[paper.issn]
                    else:
                        metrics = fetch_openalex_metrics(paper.issn)
                        issn_cache[paper.issn] = metrics

                    paper.impact_factor = metrics['impact_factor']
                    paper.h_index = metrics['h_index']
                else:
                    paper.impact_factor = None
                    paper.h_index = None
            else:
                paper.journal_name = None
                paper.issn = None
                paper.pmcid = None
                paper.impact_factor = None
                paper.h_index = None

    except Exception as e:
        print(f"[WARN] Error enriching papers: {e}")
        import traceback
        traceback.print_exc()

    return papers


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

def get_citation_counts(pmids: List[str]) -> Dict[str, int]:
    """
    Get citation counts for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs

    Returns:
        Dict mapping PMID to citation count (PmcRefCount)
    """
    if not pmids:
        return {}

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml'
    }

    citation_map = {}
    try:
        response = get_session().get(url, params=params)
        root = ET.fromstring(response.content)

        for doc in root.findall('.//DocSum'):
            try:
                pmid = doc.find('.//Id').text
                pmc_ref_count = 0
                for item in doc.findall('.//Item'):
                    if item.get('Name') == 'PmcRefCount' and item.text:
                        try:
                            pmc_ref_count = int(item.text)
                        except ValueError:
                            pass
                        break
                citation_map[pmid] = pmc_ref_count
            except:
                pass
    except Exception as e:
        print(f"[ERROR] Failed to fetch citations: {e}")

    return citation_map


# ============================================================================
# PDF Download Functions
# ============================================================================

def download_pdf_from_doi(doi: str, save_dir: str = "./pdfs/pubmed", email: str = "changeme@example.com") -> Optional[str]:
    """
    Download PDF using DOI via Unpaywall API (Open Access sources).

    Args:
        doi: DOI of the paper
        save_dir: Directory to save the PDF
        email: Your email for Unpaywall API (required by Unpaywall)

    Returns:
        Path to downloaded PDF if successful, None otherwise

    Note:
        Please set a valid email address when using this function.
        You can set it via environment variable UNPAYWALL_EMAIL or pass it directly.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if file already exists
        safe_doi = doi.replace('/', '_').replace('\\', '_')
        pdf_path = os.path.join(save_dir, f"{safe_doi}.pdf")
        
        if os.path.exists(pdf_path):
            # print(f"[INFO] Found existing PDF: {pdf_path}")
            return pdf_path

        # Get email from environment or parameter
        if email == "changeme@example.com":
            email = os.environ.get("UNPAYWALL_EMAIL", email)
            if email == "changeme@example.com":
                print(f"[WARN] Using placeholder email. Set UNPAYWALL_EMAIL env var for better results.")

        # Try Unpaywall API (legal, open access only)
        # print(f"[INFO] Checking Unpaywall for open access PDF...")
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        response = get_session().get(unpaywall_url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Check if there's an open access PDF
            if data.get('is_oa'):
                pdf_url = data.get('best_oa_location', {}).get('url_for_pdf')

                if pdf_url:
                    # print(f"[INFO] Found open access PDF: {pdf_url}")

                    # Download the PDF
                    pdf_response = get_session().get(pdf_url, stream=True, timeout=30)
                    pdf_response.raise_for_status()

                    with open(pdf_path, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"[INFO] Downloaded PDF: {pdf_path}")
                    return pdf_path
                else:
                    # print(f"[WARN] Open access available but no PDF URL found")
                    pass
            else:
                # print(f"[WARN] Paper is not open access according to Unpaywall")
                pass

        return None

        return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print(f"[WARN] 403 Forbidden downloading DOI {doi}. Likely behind paywall.")
        else:
            print(f"[ERROR] HTTP Error downloading DOI {doi}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to download via DOI: {e}")
        return None


async def download_arxiv_pdf(arxiv_id: str, save_dir: str = "./pdfs/arxiv") -> str:
    """
    Download PDF for an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., '2106.12345')
        save_dir: Directory to save the PDF

    Returns:
        Path to downloaded PDF or error message
    """
    try:
        from paper_search_mcp.server import download_arxiv

        os.makedirs(save_dir, exist_ok=True)
        pdf_path = await download_arxiv(arxiv_id, save_dir)
        return f"Downloaded to: {pdf_path}"
    except Exception as e:
        return f"Failed to download PDF for arXiv {arxiv_id}: {e}"


def download_pubmed_pdfs_batch(pmids: List[str], save_dir: str = "./pdfs/pubmed") -> Dict[str, str]:
    """
    Download PDFs for multiple PubMed papers.

    Args:
        pmids: List of PubMed IDs
        save_dir: Directory to save PDFs

    Returns:
        Dict mapping PMID to download status/result
    """
    import asyncio

    async def download_all():
        results = {}
        for pmid in pmids:
            result = await download_pubmed_pdf(pmid, save_dir)
            results[pmid] = result
            print(f"[INFO] PMID {pmid}: {result}")
        return results

    return asyncio.run(download_all())


def download_arxiv_pdfs_batch(arxiv_ids: List[str], save_dir: str = "./pdfs/arxiv") -> Dict[str, str]:
    """
    Download PDFs for multiple arXiv papers.

    Args:
        arxiv_ids: List of arXiv IDs
        save_dir: Directory to save PDFs

    Returns:
        Dict mapping arXiv ID to download status/result
    """
    import asyncio

    async def download_all():
        results = {}
        for arxiv_id in arxiv_ids:
            result = await download_arxiv_pdf(arxiv_id, save_dir)
            results[arxiv_id] = result
            print(f"[INFO] arXiv {arxiv_id}: {result}")
        return results

    return asyncio.run(download_all())


# ============================================================================
# PMC Full-Text Retrieval
# ============================================================================

def get_pmc_fulltext(pmcid: str) -> Optional[str]:
    """
    Retrieve full text from PubMed Central using BioC API.

    Args:
        pmcid: PMC ID (e.g., 'PMC1234567')

    Returns:
        Full text as string, or None if failed
    """
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

    try:
        response = get_session().get(url)
        response.raise_for_status()

        # Check for HTML error page (BioC API returns 200 OK even when paper not found/not OA)
        if "No result can be found" in response.text or "<!DOCTYPE html>" in response.text:
            # print(f"[INFO] PMC full text not available for {pmcid} (likely not Open Access).")
            return None

        data = response.json()

        # BioC JSON is a list of collections
        if isinstance(data, list):
            data = data[0]

        # Extract text from all passages
        full_text_parts = []
        for document in data.get('documents', []):
            for passage in document.get('passages', []):
                text_part = passage.get('text', '')
                if text_part:
                    full_text_parts.append(text_part)

        return "\n\n".join(full_text_parts)

    except json.JSONDecodeError:
        print(f"[WARN] PMC API returned invalid JSON for {pmcid}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch PMC text for {pmcid}: {e}")
        return None


def save_pmc_fulltext(pmcid: str, output_path: str) -> bool:
    """
    Download and save PMC full text to a file.

    Args:
        pmcid: PMC ID
        output_path: Path to save the text file

    Returns:
        True if successful, False otherwise
    """
    text = get_pmc_fulltext(pmcid)
    if text:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"[INFO] Saved PMC text to {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save PMC text: {e}")
    return False


# ============================================================================
# Result Saving Utilities
# ============================================================================

def save_papers_to_json(papers: List[Any], filename: str,
                        include_dynamic_attrs: bool = True) -> str:
    """
    Save papers to JSON file with proper serialization.

    Args:
        papers: List of paper objects
        filename: Output filename (will add timestamp if not present)
        include_dynamic_attrs: Whether to include dynamically added attributes

    Returns:
        Absolute path to saved file
    """
    # Add timestamp if not in filename
    if not any(x in filename for x in ['_202', '_203']):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{timestamp}{ext}"

    # Convert to dicts
    results_list = []
    for p in papers:
        p_dict = asdict(p) if hasattr(p, '__dataclass_fields__') else p.__dict__

        # Add dynamic attributes if requested
        if include_dynamic_attrs:
            for attr in ['journal_name', 'issn', 'impact_factor', 'h_index',
                        'pmcid', 'citations']:
                if hasattr(p, attr):
                    p_dict[attr] = getattr(p, attr)

        results_list.append(p_dict)

    # JSON serialization helper
    def json_serial(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return str(obj)

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, default=json_serial, indent=2)

    abs_path = os.path.abspath(filename)
    print(f"[INFO] Saved {len(results_list)} papers to {abs_path}")
    return abs_path


# ============================================================================
# Example Usage Functions
# ============================================================================

def example_pubmed_search():
    """Example: Search PubMed with full enrichment."""
    from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher

    # Search
    searcher = PubMedSearcher()
    query = "BCL2L1 BAD interaction"
    papers = searcher.search(query, max_results=10)

    # Enrich with citations and journal info
    papers = enrich_pubmed_papers(papers)

    # Save results
    save_papers_to_json(papers, "pubmed_results.json")

    # Print summary
    for i, p in enumerate(papers, 1):
        print(f"\n[{i}] {p.title[:80]}")
        print(f"    PMID: {p.paper_id}")
        print(f"    Citations: {getattr(p, 'citations', 'N/A')}")
        print(f"    Journal: {getattr(p, 'journal_name', 'N/A')}")
        print(f"    Impact Factor: {getattr(p, 'impact_factor', 'N/A')}")
        print(f"    PMCID: {getattr(p, 'pmcid', 'N/A')}")


def example_get_identifiers():
    """Example: Get all identifiers for a PMID."""
    pmids = ["36194155", "38954691"]

    print("\n=== Getting Paper Identifiers ===")
    for pmid in pmids:
        print(f"\nPMID: {pmid}")
        identifiers = get_paper_identifiers(pmid)
        print(f"  PMCID: {identifiers['pmcid']}")
        print(f"  DOI: {identifiers['doi']}")
        print(f"  PubMed URL: {identifiers['pubmed_url']}")
        print(f"  PMC URL: {identifiers['pmc_url']}")
        print(f"  DOI URL: {identifiers['doi_url']}")


def example_doi_download():
    """Example: Download PDF using DOI via Unpaywall."""
    # Example: Get DOI from PMID first, then download
    # Using PMID 25356929 - a PLOS ONE paper that should have PDF available
    pmid = "25356929"

    print("\n=== DOI-based PDF Download Test ===")

    # Step 1: Get DOI from PMID
    identifiers = get_paper_identifiers(pmid)
    print(f"\nPMID: {pmid}")
    print(f"DOI: {identifiers['doi']}")

    # Step 2: Download PDF using DOI
    if identifiers['doi']:
        pdf_path = download_pdf_from_doi(
            doi=identifiers['doi'],
            save_dir="./pdfs/test"
        )

        if pdf_path:
            print(f"\n✓ Successfully downloaded to: {pdf_path}")
        else:
            print(f"\n✗ Could not download PDF")
            print(f"Try manually: {identifiers['doi_url']}")
    else:
        print("\n✗ No DOI found for this paper")


def example_pmc_fulltext():
    """Example: Retrieve PMC full text."""
    pmcid = "PMC1790863"

    # Get full text
    text = get_pmc_fulltext(pmcid)
    if text:
        print(f"Retrieved {len(text)} characters from {pmcid}")
        print(f"Preview:\n{text[:500]}...")

        # Save to file
        save_pmc_fulltext(pmcid, f"{pmcid}_fulltext.txt")


def example_pdf_download():
    """Example: Download PDFs from PubMed and arXiv."""
    # Download single PubMed PDF
    pmids = ["36194155", "38954691"]
    print("[INFO] Downloading PubMed PDFs...")
    results = download_pubmed_pdfs_batch(pmids, save_dir="./pdfs/pubmed")

    # Download single arXiv PDF
    arxiv_ids = ["2106.12345"]
    print("\n[INFO] Downloading arXiv PDFs...")
    arxiv_results = download_arxiv_pdfs_batch(arxiv_ids, save_dir="./pdfs/arxiv")


if __name__ == "__main__":
    print("=" * 70)
    print("Paper Utilities - Example Usage")
    print("=" * 70)

    # Uncomment to run examples:
    # example_pubmed_search()
    # example_pmc_fulltext()
    # example_pdf_download()
    # example_get_identifiers()
    example_doi_download()

    print("\nImport this module to use the utility functions.")
    
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

def get_fulltext_from_jina(url: str) -> Optional[str]:
    """
    Fetch full text using Jina AI (r.jina.ai).
    """
    if not url:
        return None
        
    jina_url = f"https://r.jina.ai/{url}"
    try:
        response = get_session().get(jina_url)
        if response.status_code == 200:
            text = response.text
            # Check for common blocking/error messages
            if any(x in text for x in ["Just a moment...", "403: Forbidden", "Access to this page has been denied", "Error - Cookies Turned Off"]):
                # print(f"[WARN] Jina AI blocked or failed for {url}")
                return None
            return text
    except Exception as e:
        # print(f"[WARN] Jina AI query failed for {url}: {e}")
        pass
    return None

def get_paper_details(pmid: str, json_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve comprehensive details for a paper including full text if available.

    Args:
        pmid: PubMed ID
        json_path: Optional path to JSON file. If provided and a corresponding PDF exists,
                   will attempt to extract text from PDF instead of using Jina.

    Returns:
        Dict with keys: pmid, pmcid, doi, title, abstract, published_date,
        journal_name, issn, impact_factor, h_index, citations, url,
        full_text
    """
    # 1. Identifiers
    ids = get_paper_identifiers(pmid)

    # 2. Basic Metadata (Title, Abstract, Date)
    metadata = get_pubmed_metadata(pmid)

    # 3. Journal Info & Metrics
    journal_info = get_journal_info_by_pmid(pmid)

    # 4. Citations (OpenAlex)
    citations = get_openalex_citation_count(pmid)

    # 5. Full Text (PDF -> PMC -> Jina Fallback)
    full_text = None

    # Try PDF first if json_path is provided
    if json_path:
        pdf_path = json_path.replace('.json', '.pdf')
        if os.path.exists(pdf_path):
            # print(f"[INFO] Found PDF at {pdf_path}, extracting text...")
            full_text = extract_text_from_pdf(pdf_path)

    # Try PMC if no PDF text
    if not full_text and ids.get('pmcid'):
        pmcid = ids['pmcid']
        # print(f"[INFO] No PDF found. Trying PMC full text for {pmcid}")
        full_text = get_pmc_fulltext(pmcid)

    # Fallback to Jina if no full text
    if not full_text:
        target_url = ids.get('doi_url') or ids.get('pubmed_url')
        if target_url:
            # print(f"[INFO] No PDF or PMC full text. Trying Jina AI with {target_url}")
            full_text = get_fulltext_from_jina(target_url)

    return {
        'pmid': pmid,
        'pmcid': ids.get('pmcid'),
        'doi': ids.get('doi'),
        'title': metadata.get('title'),
        'abstract': metadata.get('abstract'),
        'published_date': metadata.get('published_date'),
        'journal_name': journal_info.get('journal_name'),
        'issn': journal_info.get('issn'),
        'impact_factor': journal_info.get('impact_factor'),
        'h_index': journal_info.get('h_index'),
        'citations': citations,
        'url': ids.get('pubmed_url'),
        'doi_url': ids.get('doi_url'),
        'full_text': full_text
    }


# ============================================================================
# PDF Text Extraction Functions
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as string
    """
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def update_json_with_pdf_text(json_path: str, pdf_path: Optional[str] = None) -> bool:
    """
    Extract text from PDF and update the 'full_text' field in a JSON file.

    If pdf_path is not provided, will look for a PDF with the same name as the JSON file.

    Args:
        json_path: Path to the JSON file to update
        pdf_path: Optional path to the PDF file. If not provided, will use json_path.replace('.json', '.pdf')

    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine PDF path
        if pdf_path is None:
            pdf_path = json_path.replace('.json', '.pdf')

        # Check if PDF exists
        if not os.path.exists(pdf_path):
            print(f"[ERROR] PDF file not found at {pdf_path}")
            return False

        # Extract text from PDF
        print(f"[INFO] Extracting text from {pdf_path}")
        full_text = extract_text_from_pdf(pdf_path)
        print(f"[INFO] Extracted {len(full_text)} characters")

        # Load existing JSON
        if not os.path.exists(json_path):
            print(f"[ERROR] JSON file not found at {json_path}")
            return False

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Update the full_text field
        data['full_text'] = full_text

        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"[INFO] Updated {json_path} with PDF text ({len(full_text)} chars)")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to update JSON with PDF text: {e}")
        return False

