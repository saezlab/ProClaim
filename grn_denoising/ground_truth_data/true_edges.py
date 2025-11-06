import pandas as pd
import numpy as np
import random
from typing import Set, Tuple, Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import requests
import json
import omnipath
from omnipath.interactions import OmniPath
from omnipath.interactions import AllInteractions
import sys

# Add parent directory to path to import from langgraph
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from edge_correction_agent
from langgraph.edge_correction_agent import (
    query_llm,
    get_interaction_prompt,
    ask_gene_regulation_question,
    build_graph,
    StructuredOutcome,
    GraphState,
    Alternative
)

def load_edges(input_file: str = "../all_removed_edges_with_sources.csv") -> pd.DataFrame:
    """Load the edges from CSV file.

    Args:
        csv_path: Path to the all_edges_with_sources.csv file

    Returns:
        DataFrame with columns including ENTITYA (source_gene), ENTITYB (target_gene), EFFECT (relationship)
    """
    df = pd.read_csv(input_file)

    # Filter to keep only protein-protein interactions
    df_ppi = df[(df['TYPEA'] == 'protein') & (df['TYPEB'] == 'protein')].copy()

    # Rename columns for clarity
    df_ppi.rename(columns={
        'ENTITYA': 'source_gene',
        'ENTITYB': 'target_gene',
        'EFFECT': 'relationship'
    }, inplace=True)

    return df_ppi

if __name__ == "__main__":
    NUM_REPETITIONS = 5
    input_file = Path("./candidate_true_edges.csv")
    output_path = Path("../langgraph/true_edges_results")

    # Load data
    df = load_edges(input_file=input_file)

    # Build graph
    app = build_graph(use_search=False)
    
    TARGET_TRUE_EDGE_NUM = 50
    TRUE_EDGE_COUNT = 0
    true_edges = []
    while TRUE_EDGE_COUNT < TARGET_TRUE_EDGE_NUM:
        # Select a random row from the dataframe with no duplicates
        random_row = df.sample(n=1)
        # Get the source and target genes
        source = random_row["source_gene"].values[0]
        target = random_row["target_gene"].values[0]
        relationship = random_row["relationship"].values[0]
        interaction_prompt = get_interaction_prompt(source, target, relationship)
        query = f"Does {interaction_prompt}?"

        initial_state = {
            "messages": [f"{source}|{target}|{relationship}"],
            "response": "",
            "structured_outcome": None,
            "retry_count": 0,
            "probability_threshold": 0.6,
            "is_relevant": False,  # Initialize to False
            "original_question": query,
            "max_retries": 5,  # Maximum number of retry attempts
            "search_results": "",  # Will be populated by search_node if USE_SEARCH is True
            "use_search": False
        }
        found_valid_edge = False
        for run_number in range(NUM_REPETITIONS):
            # Create results directory for this run
            results_dir = Path(f"{output_path}/true_edges_run_{run_number}")
            results_dir.mkdir(parents=True, exist_ok=True)
            # Run the graph
            result = app.invoke(initial_state)

            # Save result to JSON file with edge index in filename
            search_suffix = "_no_search"
            output_file = results_dir / f"gene_regulation_result_{source}_{target}{search_suffix}.json"
            if result["structured_outcome"]:
                result_data = result["structured_outcome"].model_dump()
                result_data["is_relevant"] = result.get("is_relevant", False)
                result_data["used_search"] = False
                result_data["edge_index"] = int(random_row.index[0])
                result_data["source_gene"] = source
                result_data["target_gene"] = target
                result_data["relationship"] = relationship
                result_data["run_number"] = run_number

                with open(output_file, 'w') as f:
                    json.dump(result_data, f, indent=2)

                if result_data["answer"] == False and result_data["probability"] != -1.0:
                    true_edges.append(random_row)
                    TRUE_EDGE_COUNT += 1
                    found_valid_edge = True
                    # Remove this edge from df to avoid resampling
                    df = df.drop(random_row.index)
                    break

        # If no valid edge found after all repetitions, remove this edge and try another
        if not found_valid_edge:
            df = df.drop(random_row.index)

    # Save true_edges to CSV
    if true_edges:
        true_edges_df = pd.concat(true_edges, ignore_index=True)
        output_csv = Path("./true_edges.csv")
        true_edges_df.to_csv(output_csv, index=False)