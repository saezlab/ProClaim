import requests

def get_pmc_text(pmcid):
    # URL format: https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{ID}/unicode
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        try:
            data = response.json()
        except ValueError:
            return f"Error: Response is not valid JSON. Content:\n{response.text[:500]}"
        
        # BioC JSON is a list of collections, usually just one
        if isinstance(data, list):
            data = data[0]
            
        # Combine full text
        full_text_parts = []
        
        # BioC structure: documents -> passages -> text
        for document in data.get('documents', []):
            for passage in document.get('passages', []):
                # passage['infons']['type'] indicates section type (title, abstract, paragraph, etc.)
                text_part = passage.get('text', '')
                full_text_parts.append(text_part)
        
        return "\n\n".join(full_text_parts)

    except Exception as e:
        return f"Error fetching PMC text: {e}"

# Usage example
pmcid = "PMC1790863" # Replace with a valid PMCID
text = get_pmc_text(pmcid)
print(f"Type of text: {type(text)}")
print(text[:500]) # Print first 500 characters

# Save to file
with open("pmc_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Text saved to pmc_text.txt")