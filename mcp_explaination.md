# Answering your questions

Hi there,

### 1. Why didn't the demo script finding evidence? Did it even search PubMed?
**Yes, it did perform a live web search!**
The reason the script failed to extract any papers previously was because `mcp-simple-pubmed` returns the search results in JSON format (which contains the PMIDs), but the original demo script (`demo_evidence_programming_with_paper_features.py`) was trying to use a simple string regex (`re.findall(r"PMID:\s*(\d+)", content)`) to find them. Since the JSON uses the key `"pmid": "123"` instead of `"PMID: 123"`, the regex failed, resulting in 0 papers being passed to the extraction loop.
*I have already fixed this parsing logic in the script (by trying `json.loads` first).* After the fix, the script successfully retrieved all 13 papers and ran them through the `BiomedicalEntityExtractor`! The reason it extracted 0 facts is simply because the GLM model didn't find any facts explicitly stating "p53 activates BAX" in the abstracts/text it pulled.

### 2. Where does `orchestrator.py` and the `evidence-tools` MCP come from, and where should I make changes?
When the main Verification Agent (`orchestrator.py` or `run_verification.py`) runs, it spins up a custom local Python MCP Server to act as its toolset. That server is defined in `src/pkevolve/verification/mcp_tools.py`!

You can see this in `orchestrator.py`:
```python
        mcp_servers=[{
            "name": "evidence-tools",
            "command": "python",
            "args": ["-m", "pkevolve.verification.mcp_tools"],
        }]
```
This means the Verification Agent uses this custom server (`mcp_tools.py`) to handle searching for papers, filling gaps, and extracting facts.

**Best Practice for Development:**
1. **For the Production Agent (`orchestrator.py`)**: If you want to improve the paper-fetching logic for the main agent, you should modify the tools inside `src/pkevolve/verification/mcp_tools.py` (like `search_pubmed`). The ideal setup is to have the tools in `mcp_tools.py` call the `mcp-simple-pubmed` client under the hood. This ensures `mcp_tools.py` remains the "State Master" that correctly updates `evidence_state.json`.
2. **For the current script (`evidence_programming...`)**: This is just a standalone script with a hardcoded loop. It does not launch the full orchestrator; instead, it manually starts the `mcp-simple-pubmed` server and steps through the verification process itself.

**Conclusion:**
- For local testing and quick experiments: using `mcp-simple-pubmed` directly in standalone scripts is fine.
- **For integration into the real Verification Agent**: we need to go into `src/pkevolve/verification/mcp_tools.py` and update its Web Search tools to use the `mcp-simple-pubmed` client.

Would you like me to guide you through updating the Web Search logic in `src/pkevolve/verification/mcp_tools.py` to use `mcp-simple-pubmed`?
