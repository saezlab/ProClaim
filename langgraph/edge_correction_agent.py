"""LangGraph-based edge correction agent for gene regulation validation."""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import from modular components
from utils import load_edges, get_interaction_prompt
from llm_reasoning import StructuredOutcome, ask_gene_regulation_question
from confidence_evaluation import evaluate_confidence, update_outcome_with_confidence
from relevance_check import evaluate_relevance, save_attempt_result

load_dotenv()


class GraphState(TypedDict):
    """State definition for the LangGraph workflow."""
    messages: Annotated[list[str], operator.add]
    response: str
    structured_outcome: StructuredOutcome | None
    retry_count: int
    confidence_threshold: float
    is_relevant: bool  # True if reasoning is relevant, False otherwise
    original_question: str  # Track the original question for relevance evaluation
    max_retries: int  # Maximum number of retry attempts
    search_results: str  # Web search results to provide context for the LLM
    use_search: bool  # Whether to use web search or not

# Global LLM instances (will be set in __main__)
reasoning_llm = None
logprob_llm = None


def llm_node(state: GraphState) -> GraphState:
    """Node that calls the LLM and processes the response."""
    # Extract gene information from the first message
    # Expected format: "source_gene|target_gene|relationship"
    parts = state["messages"][0].split("|") if state["messages"] else []
    if len(parts) == 3:
        source_gene, target_gene, relationship = parts
    else:
        raise ValueError("Invalid message format. Expected 'source_gene|target_gene|relationship'.")

    # Get search results from state
    search_context = state.get("search_results", "")

    # Call the LLM with search results as context
    result = ask_gene_regulation_question(
        source_gene=source_gene,
        target_gene=target_gene,
        relationship=relationship,
        search_context=search_context,
        llm=reasoning_llm
    )

    return {
        "messages": [f"Attempt {state['retry_count'] + 1}: {source_gene} {relationship} {target_gene}"],
        "response": result.answer_text,
        "structured_outcome": result,
        "retry_count": state["retry_count"] + 1
    }

def search_node(state: GraphState) -> GraphState:
    """Search for relevant publications using Tavily Search."""
    source_gene = state["messages"][0].split("|")[0]
    target_gene = state["messages"][0].split("|")[1]
    relationship = state["messages"][0].split("|")[2]

    # Use natural language prompt for search query
    interaction_prompt = get_interaction_prompt(source_gene, target_gene, relationship)
    query = f"Does {interaction_prompt}?"

    search_tool = TavilySearchResults(max_results=2, search_depth="advanced", start_date="2018-01-01")
    results = search_tool.invoke({"query": query})

    # Combine search results
    search_text = "\n\n".join([
        result.get('content', '')
        for result in results
    ])

    # Save raw results to a file for debugging
    # with open(f"search_results_{state['retry_count'] + 1}.txt", 'w') as f:
    #     f.write(search_text)

    return {
        "search_results": search_text
    }

def confidence_node(state: GraphState) -> GraphState:
    """Evaluate confidence based on structured outcome using logprob_llm."""
    structured_outcome = state.get("structured_outcome")

    # Evaluate confidence using the modular function
    confidence, logprob = evaluate_confidence(structured_outcome, logprob_llm)

    # Update the structured outcome with the confidence value
    updated_outcome = update_outcome_with_confidence(
        structured_outcome,
        confidence,
        logprob
    )

    return {
        "structured_outcome": updated_outcome
    }

def reflect_node(state: GraphState) -> GraphState:
    """Node that reflects on the LLM response quality and reasoning relevance."""
    if state["structured_outcome"] is None:
        return {"is_relevant": False}

    reasoning = state["structured_outcome"].reasoning
    original_question = state["original_question"]

    # Evaluate relevance using the modular function
    is_relevant = evaluate_relevance(reasoning, original_question, reasoning_llm)

    # Save this attempt to a separate JSON file
    save_attempt_result(
        attempt_number=state["retry_count"],
        original_question=original_question,
        structured_outcome=state["structured_outcome"],
        is_relevant=is_relevant,
        use_search=state.get("use_search", False)
    )

    return {"is_relevant": is_relevant}

def should_retry(state: GraphState) -> str:
    """Conditional edge function that decides whether to retry or end."""
    max_retries = state.get("max_retries", 3)  # Get from state, default to 3

    # Check if we have a structured outcome
    if state["structured_outcome"] is None:
        return "llm"

    # Check if max retries reached
    if state["retry_count"] >= max_retries:
        print(f"Max retries ({max_retries}) reached, ending...")
        return END

    # Check confidence threshold
    confidence = state["structured_outcome"].confidence
    print("confidence", confidence)
    if confidence < state["confidence_threshold"]:
        print(f"Low confidence: {confidence:.3f} < {state['confidence_threshold']}, retrying...")
        return "llm"

    # Check relevance
    is_relevant = state.get("is_relevant", False)  # Default to False if not present

    # If not relevant (either failed or not relevant), retry
    if not is_relevant:
        print(f"Reasoning not relevant (is_relevant={is_relevant}), retrying...")
        return "llm"

    return END

def build_graph(use_search: bool = False) -> StateGraph:
    """Build the LangGraph with optional search node.

    Args:
        use_search: If True, flow is search -> llm -> confidence -> reflect -> (retry or end)
                   If False, flow is llm -> confidence -> reflect -> (retry or end)
    """
    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("llm", llm_node)
    workflow.add_node("confidence", confidence_node)
    workflow.add_node("reflect", reflect_node)

    if use_search:
        workflow.add_node("search", search_node)
        # Set entry point to search
        workflow.set_entry_point("search")
        # Add edge from search to llm
        workflow.add_edge("search", "llm")
    else:
        # Set entry point to llm
        workflow.set_entry_point("llm")

    # Add edge from llm to confidence
    workflow.add_edge("llm", "confidence")

    # Add edge from confidence to reflect
    workflow.add_edge("confidence", "reflect")

    # Add conditional edge from reflect (retry or end)
    workflow.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "llm": "llm",
            END: END
        }
    )

    # Compile the graph
    return workflow.compile()

def run_edge_evaluation(
    edges_df: pd.DataFrame,
    app: StateGraph,
    output_path: str,
    run_number: int,
    use_search: bool,
    confidence_threshold: float = 0.6,
    max_retries: int = 5
):
    """Run edge evaluation for a single run.

    Args:
        edges_df: DataFrame of edges to evaluate
        app: Compiled LangGraph application
        output_path: Base output path for results
        run_number: Current run number
        use_search: Whether to use web search
        confidence_threshold: Minimum confidence threshold
        max_retries: Maximum retry attempts
    """
    # Create results directory for this run
    results_dir = Path(f"{output_path}/negative_edges_run_{run_number}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Loop through all edges
    for idx, edge in edges_df.iterrows():
        source_gene = edge['source_gene']
        target_gene = edge['target_gene']
        relationship = edge['relationship']

        # Generate natural language question
        interaction_prompt = get_interaction_prompt(source_gene, target_gene, relationship)
        original_question = f"Does {interaction_prompt}?"

        initial_state = {
            "messages": [f"{source_gene}|{target_gene}|{relationship}"],
            "response": "",
            "structured_outcome": None,
            "retry_count": 0,
            "confidence_threshold": confidence_threshold,
            "is_relevant": False,
            "original_question": original_question,
            "max_retries": max_retries,
            "search_results": "",
            "use_search": use_search
        }

        # Run the graph
        result = app.invoke(initial_state)

        # Save result to JSON file
        search_suffix = "_with_search" if use_search else "_no_search"
        output_file = results_dir / f"gene_regulation_result_{source_gene}_{target_gene}{search_suffix}.json"

        if result["structured_outcome"]:
            result_data = result["structured_outcome"].model_dump()
            result_data["is_relevant"] = result.get("is_relevant", False)
            result_data["used_search"] = use_search
            result_data["edge_index"] = int(idx)
            result_data["source_gene"] = source_gene
            result_data["target_gene"] = target_gene
            result_data["relationship"] = relationship
            result_data["run_number"] = run_number

            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    # Initialize global LLM instances
    reasoning_llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-dummy",
        model="gpt-oss",
        temperature=1,
        max_tokens=2048
    )

    logprob_llm = ChatOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="sk-dummy",
        model="qwen3-4b",
        temperature=1,
        max_tokens=10,
        model_kwargs={
            "logprobs": True,
            "top_logprobs": 5,
        }
    )

    # Configuration
    USE_SEARCH = False
    NUM_REPETITIONS = 3
    input_path = "../data/all_removed_edges_with_sources.csv"
    output_path = "./results"

    # Load edges from CSV
    removed_edges_df = load_edges(csv_path=input_path)

    # Build the graph
    app = build_graph(use_search=USE_SEARCH)

    # Repeat the simulation NUM_REPETITIONS times
    for run_number in tqdm(range(NUM_REPETITIONS)):
        run_edge_evaluation(
            edges_df=removed_edges_df,
            app=app,
            output_path=output_path,
            run_number=run_number,
            use_search=USE_SEARCH,
            confidence_threshold=0.6,
            max_retries=5
        )

