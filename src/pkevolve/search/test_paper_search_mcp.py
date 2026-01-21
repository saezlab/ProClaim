import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from paper_search_mcp.server import search_arxiv, search_pubmed, download_arxiv, download_pubmed
from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher
from paper_search_mcp.academic_platforms.arxiv import ArxivSearcher
import requests
from xml.etree import ElementTree as ET

# Monkey-patch PubMedSearcher to include citations
original_search = PubMedSearcher.search

def search_with_citations(self, query: str, max_results: int = 20):
    # Call original search to get papers
    papers = original_search(self, query, max_results)
    
    # Enrich with citations and journal info
    try:
        ids = [p.paper_id for p in papers]
        if ids:
            # 1. Fetch Citations & Journal ISSN from PubMed (efetch)
            # We use efetch instead of esummary to get Journal ISSN
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(ids),
                'retmode': 'xml'
            }
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_root = ET.fromstring(fetch_response.content)
            
            # Also fetch citation counts (PmcRefCount) from esummary (efetch doesn't have it easily)
            summary_params = {
                'db': 'pubmed',
                'id': ','.join(ids),
                'retmode': 'xml'
            }
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_response = requests.get(summary_url, params=summary_params)
            summary_root = ET.fromstring(summary_response.content)
            
            # Parse Citation Counts
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

            # Parse Journal Info & ISSN & PMCID
            journal_map = {} # pmid -> {issn, title, pmcid}
            for article in fetch_root.findall('.//PubmedArticle'):
                try:
                    pmid = article.find('.//PMID').text
                    
                    # Journal Info
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

            # OpenAlex Cache to avoid duplicate queries for same journal
            issn_cache = {} 

            # Update papers
            for paper in papers:
                # Add Citation Count
                paper.citations = citation_map.get(paper.paper_id, 0)
                
                # Add Journal Info & PMCID
                j_info = journal_map.get(paper.paper_id)
                if j_info:
                    # Dynamically add attributes to the paper object
                    paper.journal_name = j_info['title']
                    paper.issn = j_info['issn']
                    paper.pmcid = j_info['pmcid']
                    
                    # Fetch OpenAlex Metrics if ISSN exists
                    if paper.issn:
                        if paper.issn in issn_cache:
                            metrics = issn_cache[paper.issn]
                        else:
                            metrics = {'impact_factor': None, 'h_index': None}
                            try:
                                oa_url = f"https://api.openalex.org/sources?filter=issn:{paper.issn}"
                                oa_res = requests.get(oa_url)
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
                                print(f"[WARN] OpenAlex query failed for {paper.issn}: {e}")
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
            
            # Debug print to verify attributes are added
            if papers:
                p = papers[0]
                print(f"[DEBUG] First paper journal info: {getattr(p, 'journal_name', 'N/A')}, IF: {getattr(p, 'impact_factor', 'N/A')}, PMCID: {getattr(p, 'pmcid', 'N/A')}")
                
    except Exception as e:
        print(f"[WARN] Error fetching metadata: {e}")
        import traceback
        traceback.print_exc()
        
    return papers

PubMedSearcher.search = search_with_citations

import asyncio

# 1. Wrap paper-search-mcp functions as LangChain tools
@tool
async def search_arxiv_papers(query: str) -> str:
    """Search for papers on arXiv.
    
    Args:
        query: The search query.
    """
    try:
        print(f"[DEBUG] Searching arXiv for: {query}")
        results = await search_arxiv(query)
        print(f"[DEBUG] arXiv results length: {len(str(results))}")
        return str(results)
    except Exception as e:
        return f"Error searching arXiv: {e}"

@tool
async def search_pubmed_papers(query: str) -> str:
    """Search for papers on PubMed.

    Args:
        query: The search query.
    """
    try:
        results = await search_pubmed(query)
        return str(results)
    except Exception as e:
        return f"Error searching PubMed: {e}"

@tool
async def download_arxiv_pdf(paper_id: str, save_path: str = "./pdfs") -> str:
    """Download PDF of an arXiv paper.

    Args:
        paper_id: arXiv paper ID (e.g., '2106.12345').
        save_path: Directory to save the PDF (default: './pdfs').
    """
    try:
        result = await download_arxiv(paper_id, save_path)
        return f"Downloaded arXiv PDF: {result}"
    except Exception as e:
        return f"Error downloading arXiv PDF: {e}"

@tool
async def download_pubmed_pdf(paper_id: str, save_path: str = "./pdfs") -> str:
    """Download PDF of a PubMed paper.

    Args:
        paper_id: PubMed ID (PMID).
        save_path: Directory to save the PDF (default: './pdfs').
    """
    try:
        result = await download_pubmed(paper_id, save_path)
        return f"PubMed PDF download result: {result}"
    except Exception as e:
        return f"Error downloading PubMed PDF: {e}"

tools = [search_arxiv_papers, search_pubmed_papers, download_arxiv_pdf, download_pubmed_pdf]

from langgraph.graph.message import add_messages

# 2. Define the State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Define the Agent/LLM
# Using SGLang server as requested
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="gpt-oss-120b",
    temperature=0
)

# Bind tools to the LLM
model_with_tools = llm.bind_tools(tools)

# 4. Define Nodes
async def agent(state: AgentState):
    messages = state["messages"]
    # Debug: print messages to see what's being sent
    print(f"\n[DEBUG] Sending messages to LLM: {len(messages)}")
    for m in messages:
        print(f"  - Role: {m.type}, Content: {m.content[:50]}...")
        if hasattr(m, 'tool_calls'):
            print(f"    Tool Calls: {m.tool_calls}")
            
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

from langchain_core.messages import ToolMessage
import json

from dataclasses import asdict
from datetime import date
import re
from pathlib import Path

async def tools_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    outputs = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"[DEBUG] Executing tool: {tool_name} with args: {tool_args}")

        if tool_name == "search_arxiv_papers":
            # Get structured data for arXiv papers
            try:
                searcher = ArxivSearcher()
                papers = searcher.search(tool_args["query"])

                # Download PDFs for arXiv papers
                pdf_dir = "pdfs/arxiv"
                os.makedirs(pdf_dir, exist_ok=True)

                for paper in papers:
                    try:
                        # Extract arXiv ID from paper_id or URL
                        arxiv_id = paper.paper_id
                        print(f"[INFO] Downloading arXiv PDF for: {arxiv_id}")
                        pdf_path = await download_arxiv(arxiv_id, pdf_dir)
                        print(f"[INFO] Saved to: {pdf_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to download PDF for {arxiv_id}: {e}")

                # Convert to dicts for JSON serialization
                results_list = [asdict(p) for p in papers]

                # Save metadata to JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"arxiv_results_{timestamp}.json"

                def json_serial(obj):
                    if isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    return str(obj)

                with open(json_filename, 'w') as f:
                    json.dump(results_list, f, default=json_serial, indent=2)
                print(f"[INFO] Saved arXiv metadata to {os.path.abspath(json_filename)}")

                result = str(results_list)

            except Exception as e:
                print(f"[ERROR] Failed to execute arXiv search/download: {e}")
                import traceback
                traceback.print_exc()
                result = f"Error searching arXiv: {e}"

        elif tool_name == "search_pubmed_papers":
            # We want to capture the structured data before it gets stringified
            # So we manually call the searcher here
            try:
                searcher = PubMedSearcher()
                # Monkey-patch the search method
                searcher.search = search_with_citations.__get__(searcher, PubMedSearcher)
                papers = searcher.search(tool_args["query"])

                # Download PDFs for PubMed papers (if available)
                pdf_dir = "pdfs/pubmed"
                os.makedirs(pdf_dir, exist_ok=True)

                for paper in papers:
                    try:
                        pmid = paper.paper_id
                        print(f"[INFO] Attempting to download PubMed PDF for PMID: {pmid}")
                        result_msg = await download_pubmed(pmid, pdf_dir)
                        print(f"[INFO] {result_msg}")
                    except Exception as e:
                        print(f"[WARN] Failed to download PDF for PMID {pmid}: {e}")

                # Convert to dicts and include dynamic attributes
                results_list = []
                for p in papers:
                    p_dict = asdict(p)
                    # Manually add dynamic attributes if they exist
                    if hasattr(p, 'journal_name'):
                        p_dict['journal_name'] = p.journal_name
                    if hasattr(p, 'issn'):
                        p_dict['issn'] = p.issn
                    if hasattr(p, 'impact_factor'):
                        p_dict['impact_factor'] = p.impact_factor
                    if hasattr(p, 'h_index'):
                        p_dict['h_index'] = p.h_index
                    if hasattr(p, 'pmcid'):
                        p_dict['pmcid'] = p.pmcid
                    results_list.append(p_dict)

                # Save to JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"pubmed_results_{timestamp}.json"

                def json_serial(obj):
                    if isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    return str(obj)

                with open(json_filename, 'w') as f:
                    json.dump(results_list, f, default=json_serial, indent=2)
                print(f"[INFO] Saved PubMed metadata to {os.path.abspath(json_filename)}")

                # Return stringified result for the agent
                result = str(results_list)

            except Exception as e:
                print(f"[ERROR] Failed to execute/save PubMed search: {e}")
                import traceback
                traceback.print_exc()
                result = f"Error searching PubMed: {e}"

        elif tool_name == "download_arxiv_pdf":
            result = await download_arxiv_pdf.ainvoke(tool_args)

        elif tool_name == "download_pubmed_pdf":
            result = await download_pubmed_pdf.ainvoke(tool_args)

        else:
            result = f"Error: Tool {tool_name} not found"

        outputs.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": outputs}

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
workflow.add_node("tools", tools_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

import sys
import os
from datetime import datetime

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# 6. Run the Test
async def main():
    # Setup logging to a new file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_output_{timestamp}.txt"
    log_file = open(log_filename, 'w')
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect stdout and stderr to both console and file
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    try:
        print(f"Starting LangGraph Paper Search Test...")
        print(f"Logging output to: {os.path.abspath(log_filename)}")
        
        query = '''Search PubMed for: (BCL2L1[Title/Abstract] OR Bcl-xL[Title/Abstract] OR Bcl-X[Title/Abstract]) AND (BAD[Title/Abstract] OR "Bcl-2-associated death promoter"[Title/Abstract]) AND (binding[Title/Abstract] OR dimerize*[Title/Abstract] OR interaction[Title/Abstract] OR "14-3-3"[Title/Abstract] OR phosphorylat*[Title/Abstract])'''
        print(f"\nQuery: {query}")
        
        initial_state = {"messages": [HumanMessage(content=query)]}
        
        config = {"recursion_limit": 5}
        
        # Store collected papers to save later
        collected_papers = []
        
        async for event in app.astream(initial_state, config=config):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                # print(value)
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    print(f"Content: {last_msg.content}")
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"Tool Calls: {last_msg.tool_calls}")
                    
                    # Capture tool output if this is the tools node
                    if key == "tools" and isinstance(last_msg, ToolMessage):
                        # The content of ToolMessage is a stringified list of dicts (usually)
                        # But in our custom tools_node, we might be returning a list of Paper objects or dicts
                        # Let's inspect how tools_node returns data. 
                        # It returns {"messages": [ToolMessage(content=str(results), ...)]}
                        try:
                            # We need to parse the content back to JSON if it's a string
                            # However, the 'results' in tools_node was a list of Paper objects converted to dicts/strings
                            # Let's look at tools_node implementation again.
                            # It calls str(results) which might be messy to parse back.
                            # Better approach: Modify tools_node to save the raw results to a global or pass them differently?
                            # Or just try to parse the string if it's valid JSON-like representation.
                            pass
                        except:
                            pass

        # Since parsing the stringified tool output is error-prone, 
        # let's modify the tools_node to save the results to a file directly or return them in a structured way.
        # But we can't easily change the graph signature.
        # A simpler way for this test script is to use a global variable or a side-effect in the tool wrapper.
        
    except Exception as e:
        print(f"\nError running graph: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout/stderr and close file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"\nTest finished. Output saved to: {log_filename}")

if __name__ == "__main__":
    asyncio.run(main())
