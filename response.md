# Explanation
1. **Why didn't the demo script find evidence? Did it even search PubMed?**
Yes, it did perform a live web search!
The script originally used a regex (`re.findall(r"PMID:\s*(\d+)", content)`) to find PMIDs, which failed because the output of `mcp-simple-pubmed` is actually JSON (like `{"pmid": "123"}`). I fixed this with JSON parsing, and the updated script successfully found 13 papers. It extracted 0 facts simply because the GLM model didn't find explicit statements about "p53 activates BAX" in those specific texts.

2. **Where do `orchestrator.py` and `evidence-tools` come from?**
The Verification Agent (`orchestrator.py`) runs a custom local MCP Server (`src/pkevolve/verification/mcp_tools.py`) to handle its tools (like `search_pubmed`). 
**Best Practice**: For the production agent, you should modify the tools in `mcp_tools.py` to call the `mcp-simple-pubmed` client under the hood. For quick local testing, using `mcp-simple-pubmed` directly in a standalone script (like the demo) is fine.
