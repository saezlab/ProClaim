from typing import List
import requests
from xml.etree import ElementTree as ET
from datetime import datetime
from paper_search_mcp.paper import Paper
from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher

class RelevancePubMedSearcher(PubMedSearcher):
    """
    Custom PubMed searcher that sorts results by relevance (Best Match).
    """
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        # Add 'sort': 'relevance' to the search parameters
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'sort': 'relevance'  # <--- Added this line
        }
        
        # We need to re-implement the rest because the original method
        # does not allow injecting extra params easily without rewriting.
        
        search_response = requests.get(self.SEARCH_URL, params=search_params)
        search_root = ET.fromstring(search_response.content)
        ids = [id.text for id in search_root.findall('.//Id')]
        
        if not ids:
            return []

        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }
        fetch_response = requests.get(self.FETCH_URL, params=fetch_params)
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
