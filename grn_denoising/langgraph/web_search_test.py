import requests
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()


class GraphState(TypedDict):
    """State for the agent graph."""
    city: str
    search_results: str
    best_url: str
    page_content: str
    temperature: str


def search_node(state: GraphState) -> GraphState:
    """Search for temperature information using Tavily."""
    city = state["city"]
    query = f"current temperature in {city}"

    print(f"[Tavily] Searching for: {query}")

    search_tool = TavilySearchResults(max_results=2)
    results = search_tool.invoke({"query": query})

    # Get the first result URL
    best_url = results[0].get('url', '') if results else ''

    # Combine search results
    search_text = "\n\n".join([
        result.get('content', '')
        for result in results
    ])

    return {
        "search_results": search_text,
        "best_url": best_url
    }


def searxng_node(state: GraphState) -> GraphState:
    """Search for temperature information using SearXNG."""
    city = state["city"]
    query = f"current temperature in {city}"

    print(f"[SearXNG] Searching for: {query}")

    # SearXNG instance URL
    searxng_url = "http://localhost:8888"
    search_endpoint = f"{searxng_url}/search?q={query}&format=json&categories=general"

    try:
        response = requests.get(search_endpoint, timeout=50)
        response.raise_for_status()

        data = response.json()
        results = data.get('results', [])

        # Take first 2 results and combine their content
        search_text = "\n\n".join([
            f"{result.get('title', '')}: {result.get('content', '')}"
            for result in results[:2]
        ])

        # Get the first result URL
        best_url = results[0].get('url', '') if results else ''

        print(f"[SearXNG] Found {len(results)} results")
        return {
            "search_results": search_text,
            "best_url": best_url
        }

    except Exception as e:
        print(f"[SearXNG] Search failed: {e}")
        return {
            "search_results": f"Error: {str(e)}",
            "best_url": ""
        }


def jina_node(state: GraphState) -> GraphState:
    """Retrieve full page content using Jina Reader API."""
    best_url = state["best_url"]

    if not best_url:
        print("[Jina] No URL to retrieve")
        return {"page_content": "No URL available"}

    print(f"[Jina] Retrieving content from: {best_url}")

    try:
        jina_url = f"https://r.jina.ai/{best_url}"
        response = requests.get(jina_url, timeout=50, headers={'Accept': 'text/plain'})
        response.raise_for_status()

        content = response.text.strip()

        # Limit content size
        if len(content) > 10000:
            content = content[:10000] + "\n... (content truncated)"

        print(f"[Jina] Retrieved {len(content)} characters")
        return {"page_content": content}

    except Exception as e:
        print(f"[Jina] Failed: {e}")
        return {"page_content": f"Error: {str(e)}"}


def extract_node(state: GraphState) -> GraphState:
    """Extract temperature from page content using LLM."""
    city = state["city"]
    page_content = state["page_content"]

    prompt = f"""<|start|>system<|message|>Extract the temperature from the text. Reply with ONLY the temperature value and unit (e.g., "15°C" or "59°F").<|end|>
<|start|>user<|message|>City: {city}

Page content:
{page_content}

Temperature:<|end|>
<|start|>assistant<|message|>"""

    print("[Extract] Extracting temperature from page content...")

    payload = {
        "prompt": prompt,
        "temperature": 0.1,
        "n_predict": -1,
        "stream": False,
    }

    response = requests.post("http://localhost:8080/completion", json=payload, timeout=300)
    response.raise_for_status()

    temperature = response.json().get("content", "").strip()

    return {"temperature": temperature}


def build_graph(use_searxng: bool = False) -> StateGraph:
    """Build simple graph: search -> jina -> extract -> end."""
    workflow = StateGraph(GraphState)

    # Add nodes - choose search method
    if use_searxng:
        workflow.add_node("search", searxng_node)
    else:
        workflow.add_node("search", search_node)

    workflow.add_node("jina", jina_node)
    workflow.add_node("extract", extract_node)

    # Define flow
    workflow.set_entry_point("search")
    workflow.add_edge("search", "jina")
    workflow.add_edge("jina", "extract")
    workflow.add_edge("extract", END)

    return workflow.compile()


def main():
    city = "Cambridge"
    use_searxng = True  # Set to True to use SearXNG, False to use Tavily

    print(f"Using: {'SearXNG' if use_searxng else 'Tavily'}\n")

    # Build and run graph
    app = build_graph(use_searxng=use_searxng)

    result = app.invoke({
        "city": city,
        "search_results": "",
        "best_url": "",
        "page_content": "",
        "temperature": ""
    })

    print("\n" + "="*60)
    print(f"Temperature in {city}: {result['temperature']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()