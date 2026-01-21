import requests
from xml.etree import ElementTree as ET

def test_journal_info():
    # Example PMIDs
    ids = ["36194155", "38954691"]
    
    # 1. Try efetch (more detailed)
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(ids),
        'retmode': 'xml'
    }
    
    print(f"Fetching full metadata (efetch) for IDs: {ids}")
    response = requests.get(fetch_url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return

    root = ET.fromstring(response.content)
    
    for article in root.findall('.//PubmedArticle'):
        pmid = article.find('.//PMID').text
        journal = article.find('.//Journal')
        
        print(f"\n--- Paper {pmid} ---")
        if journal is not None:
            title = journal.find('Title').text if journal.find('Title') is not None else "N/A"
            iso_abbrev = journal.find('ISOAbbreviation').text if journal.find('ISOAbbreviation') is not None else "N/A"
            issn = journal.find('ISSN').text if journal.find('ISSN') is not None else "N/A"
            print(f"Journal Title: {title}")
            print(f"ISO Abbreviation: {iso_abbrev}")
            print(f"ISSN: {issn}")
            
            # 2. Try OpenAlex for metrics (using ISSN)
            if issn != "N/A":
                try:
                    # Use sources endpoint with filter
                    openalex_url = f"https://api.openalex.org/sources?filter=issn:{issn}"
                    print(f"  Querying OpenAlex for ISSN {issn}...")
                    oa_response = requests.get(openalex_url)
                    if oa_response.status_code == 200:
                        data = oa_response.json()
                        results = data.get('results', [])
                        if results:
                            source = results[0]
                            print(f"  OpenAlex Data Found: {source.get('display_name')}")
                            print(f"  - Works Count: {source.get('works_count')}")
                            print(f"  - Cited By Count: {source.get('cited_by_count')}")
                            
                            if 'summary_stats' in source:
                                stats = source['summary_stats']
                                print(f"  - 2yr Mean Citedness (IF proxy): {stats.get('2yr_mean_citedness')}")
                                print(f"  - h-index: {stats.get('h_index')}")
                        else:
                            print("  No source found in OpenAlex for this ISSN")
                    else:
                        print(f"  OpenAlex query failed: {oa_response.status_code}")
                except Exception as e:
                    print(f"  Error querying OpenAlex: {e}")

        else:
            print("No Journal info found")

if __name__ == "__main__":
    test_journal_info()
