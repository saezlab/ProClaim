from typing import List
import logging
import requests
import time
from xml.etree import ElementTree as ET
from datetime import datetime
from paper_search_mcp.paper import Paper
from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher
import os

logger = logging.getLogger(__name__)

# NCBI's anonymous tier rate-limits by shared institutional egress IP, so a
# transient failure is common and worth waiting out. Unlike S2Client, this
# searcher has no internal retry, so the policy lives here and every caller
# (ProClaim's evidence_api and the baselines alike) uses this one function —
# an earlier split had ProClaim retrying 3x/~3s against the baselines' 6x/~62s.
PUBMED_MAX_RETRIES = 6


def search_with_retry(
    searcher: "RelevancePubMedSearcher",
    query: str,
    max_results: int,
    max_retries: int = PUBMED_MAX_RETRIES,
) -> List[Paper]:
    """Run a PubMed search, retrying transient failures with exponential backoff.

    Sleeps 2, 4, 8, 16, 32s between the 6 default attempts (~62s total).
    Returns an empty list once retries are exhausted — callers cannot
    distinguish that from a genuine no-hit query, so the failure is logged.
    """
    for attempt in range(max_retries):
        try:
            return searcher.search(query, max_results=max_results)
        except Exception as exc:
            if attempt + 1 == max_retries:
                logger.warning(
                    "PubMed search failed after %d attempts for %r: %s",
                    max_retries, query, exc,
                )
                return []
            backoff = 2 ** attempt * 2
            logger.warning(
                "PubMed search error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, max_retries, backoff, exc,
            )
            time.sleep(backoff)
    return []

class RelevancePubMedSearcher(PubMedSearcher):
    """
    Custom PubMed searcher that sorts results by relevance (Best Match).
    Supports NCBI API key for higher rate limits (10 req/s instead of 3 req/s).
    """
    def __init__(self, api_key: str = None, email: str = None):
        """
        Initialize searcher with optional API key and email.

        Args:
            api_key: NCBI API key (or reads from PUBMED_API_KEY env var)
            email: Email for NCBI (or reads from PUBMED_EMAIL env var)
        """
        super().__init__()
        self.api_key = api_key or os.environ.get('PUBMED_API_KEY')
        self.email = email or os.environ.get('PUBMED_EMAIL')

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        # Use usehistory=y + sort=relevance to get relevance-sorted results
        # Note: sort=relevance alone causes API bug (returns 0 results for some queries)
        # but usehistory=y + sort=relevance works correctly
        # See: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'usehistory': 'y',  # <--- Required to make sort=relevance work
            'sort': 'relevance',  # <--- Sort by Best Match (relevance)
        }

        # Add API key and email if available (increases rate limit to 10 req/s)
        if self.api_key:
            search_params['api_key'] = self.api_key
        if self.email:
            search_params['email'] = self.email

        # We need to re-implement the rest because the original method
        # does not allow injecting extra params easily without rewriting.

        search_response = requests.get(self.SEARCH_URL, params=search_params, timeout=30)
        search_response.raise_for_status()  # NCBI returns a JSON error body (not XML) on 429/5xx
        search_root = ET.fromstring(search_response.content)
        ids = [id.text for id in search_root.findall('.//Id')]

        if not ids:
            return []

        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }

        # Add API key and email to fetch request too
        if self.api_key:
            fetch_params['api_key'] = self.api_key
        if self.email:
            fetch_params['email'] = self.email

        fetch_response = requests.get(self.FETCH_URL, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        fetch_root = ET.fromstring(fetch_response.content)
        
        papers = []
        for article in fetch_root.findall('.//PubmedArticle'):
            try:
                pmid = article.find('.//PMID').text
                title = article.find('.//ArticleTitle').text
                
                authors_elem = article.findall('.//Author')
                authors = []
                for author in authors_elem:
                    try:
                        last = author.find('LastName').text if author.find('LastName') is not None else ''
                        initials = author.find('Initials').text if author.find('Initials') is not None else ''
                        authors.append(f"{last} {initials}".strip())
                    except:
                        pass
                
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ''
                
                pub_date_elem = article.find('.//PubDate/Year')
                if pub_date_elem is not None:
                    pub_date = pub_date_elem.text
                    try:
                        published = datetime.strptime(pub_date, '%Y')
                    except:
                        published = datetime.now() # Fallback
                else:
                    published = datetime.now() # Fallback

                doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
                doi = doi_elem.text if doi_elem is not None else ''
                if not doi:
                    doi_aid = article.find('.//ArticleIdList/ArticleId[@IdType="doi"]')
                    doi = doi_aid.text if doi_aid is not None else ''
                
                papers.append(Paper(
                    paper_id=pmid,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    pdf_url='',
                    published_date=published,
                    updated_date=published,
                    source='pubmed',
                    categories=[],
                    keywords=[],
                    doi=doi
                ))
            except Exception as e:
                print(f"[WARN] Error parsing PubMed article (PMID: {pmid if 'pmid' in locals() else 'unknown'}): {e}")
                
        return papers
