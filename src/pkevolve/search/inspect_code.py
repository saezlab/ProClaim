import inspect
from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher

print("=== PubMedSearcher.download_pdf source ===")
try:
    print(inspect.getsource(PubMedSearcher.download_pdf))
except Exception as e:
    print(f"Could not get source: {e}")
