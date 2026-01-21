import requests
import json

def inspect_openalex_history():
    # PLoS One ISSN
    issn = "1932-6203"
    url = f"https://api.openalex.org/sources?filter=issn:{issn}"
    
    print(f"Querying OpenAlex for ISSN {issn}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        if results:
            source = results[0]
            # Print keys to see available fields
            print("\nTop-level keys:", source.keys())
            
            # Check for year-by-year stats
            if 'counts_by_year' in source:
                print("\ncounts_by_year (first 3):")
                print(json.dumps(source['counts_by_year'][:3], indent=2))
            
            # Check summary_stats structure again
            if 'summary_stats' in source:
                print("\nsummary_stats:")
                print(json.dumps(source['summary_stats'], indent=2))
                
            # Save full JSON to file for manual inspection if needed
            with open("openalex_source_dump.json", "w") as f:
                json.dump(source, f, indent=2)
            print("\nFull response saved to openalex_source_dump.json")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    inspect_openalex_history()
