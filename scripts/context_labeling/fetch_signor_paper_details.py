"""
Fetch SIGNOR paper details and save to JSON files.

Usage:
    # For true_negative_edges (default):
    python fetch_signor_paper_details.py

    # For true_positive_edges:
    python fetch_signor_paper_details.py data/signor/true_positive_edges.csv
"""

import os
import sys
import json
import asyncio
from typing import Annotated, Literal, TypedDict, Dict, Any
from datetime import datetime, date

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pkevolve.utils.signor_utils import load_signor_data
from pkevolve.search.paper_utils import get_paper_details

# ============================================================================
# Global Configuration
# ============================================================================
SAVE_DIR = "../data/papers/signor/true_negative_edges"  # Will be updated by main()

# ============================================================================
# Tools
# ============================================================================

@tool
def retrieve_paper_details(pmid: str, entity_a: str, entity_b: str, effect: str) -> str:
    """
    Retrieve comprehensive details for a paper (metadata, full text, etc.) and save to JSON.

    Args:
        pmid: PubMed ID of the paper.
        entity_a: First entity name (used for filename).
        entity_b: Second entity name (used for filename).
        effect: Effect description (used for filename).
    """
    try:
        # Construct filename first to check for existence
        # Sanitize filename components
        def sanitize(s):
            return "".join(c for c in s if c.isalnum() or c in ('_', '-')).strip()

        safe_a = sanitize(entity_a)
        safe_b = sanitize(entity_b)
        safe_effect = sanitize(effect)

        filename = f"{safe_a}_{safe_b}_{safe_effect}.json"
        save_dir = SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        # Check if file exists and has valid full text
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                full_text = existing_data.get('full_text')
                # Check validity: not None, not empty, not an error message
                if full_text and len(full_text.strip()) > 50 and "Just a moment..." not in full_text and "Error - Cookies Turned Off" not in full_text:
                    print(f"[INFO] File {save_path} exists with valid full text. Skipping retrieval.")
                    return f"Paper details already exist in {save_path} (skipped retrieval)."
                else:
                    print(f"[INFO] File {save_path} exists but full text is missing or invalid. Re-retrieving...")
            except Exception as e:
                print(f"[WARN] Error reading existing file {save_path}: {e}. Proceeding with retrieval.")

        print(f"[INFO] Retrieving details for PMID: {pmid}")
        # Pass json_path to enable PDF extraction if available
        details = get_paper_details(pmid, json_path=save_path)

        # JSON serialization helper
        def json_serial(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            return str(obj)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, default=json_serial, indent=2)

        return f"Successfully saved paper details to {save_path}"

    except Exception as e:
        return f"Error retrieving/saving paper details: {e}"

# ============================================================================
# Agent Setup
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="gpt-oss-120b",
    temperature=0
)

tools = [retrieve_paper_details]
model_with_tools = llm.bind_tools(tools)

async def agent_node(state: AgentState):
    messages = state["messages"]
    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# ============================================================================
# Main Execution
# ============================================================================

async def main():
    global SAVE_DIR

    print("Starting Sufficient Context Agent with SIGNOR Data...")

    # 1. Parse command-line arguments
    base_path = "/nfs/research/saezrodriguez/rain/workspace/grn-llm-correct"

    if len(sys.argv) > 1:
        # Use provided CSV path
        csv_path = sys.argv[1]
        # If relative path, make it absolute
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(base_path, csv_path)
    else:
        # Default to true_negative_edges
        csv_path = os.path.join(base_path, "data/signor/true_negative_edges.csv")

    # Determine save directory based on CSV filename
    if "true_positive_edges" in csv_path:
        SAVE_DIR = "../data/papers/signor/true_positive_edges"
    else:
        SAVE_DIR = "../data/papers/signor/true_negative_edges"

    print(f"Loading data from: {csv_path}")
    print(f"Saving papers to: {SAVE_DIR}")

    required_columns = ['ENTITYA', 'ENTITYB', 'EFFECT', 'PMID']
    df = load_signor_data(csv_path, columns=required_columns)
    for idx, row in df.iterrows():  
        input_data = {
            "pmid": str(row['PMID']),
            "entity_a": str(row['ENTITYA']),
            "entity_b": str(row['ENTITYB']),
            "effect": str(row['EFFECT'])
        }
        
        print(f"\nProcessing first row: {input_data}")
        
        prompt = f"""
        Please retrieve the details for the paper with PMID {input_data['pmid']}.
        The entities involved are {input_data['entity_a']} and {input_data['entity_b']}, and the effect is {input_data['effect']}.
        Save the results using the provided entity and effect names.
        """
        
        initial_state = {"messages": [HumanMessage(content=prompt)]}
        
        async for event in app.astream(initial_state):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if hasattr(last_msg, 'content') and last_msg.content:
                        print(f"Content: {last_msg.content}")
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"Tool Calls: {last_msg.tool_calls}")

if __name__ == "__main__":
    asyncio.run(main())