import requests
from xml.etree import ElementTree as ET

def test_pubmed_citations():
    # Example PMIDs (SARS-CoV-2 R0 papers found previously)
    ids = ["40752024", "36194155", "35994497", "38954691"]
    
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(ids),
        'retmode': 'xml'
    }
    
    print(f"Fetching summary for IDs: {ids}")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return

    root = ET.fromstring(response.content)
    
    for doc in root.findall('.//DocSum'):
        id_elem = doc.find('.//Id')
        pmid = id_elem.text if id_elem is not None else "Unknown"
        
        # Look for PmcRefCount
        pmc_ref_count = "0"
        for item in doc.findall('.//Item'):
            if item.get('Name') == 'PmcRefCount':
                pmc_ref_count = item.text
                break
        
        print(f"PMID: {pmid}, PmcRefCount: {pmc_ref_count}")

if __name__ == "__main__":
    test_pubmed_citations()
