import pandas as pd
import numpy as np
import random
import ollama
from typing import Set, Tuple, Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import re
import json
from pydantic import BaseModel, Field
from graph_utils import trace_downstream_network, convert_edge_list_to_text

# Define Pydantic models for structured output
class EdgeRemoval(BaseModel):
    reasoning: str = Field(description="Biological reasoning for removal")
    source: str = Field(description="Source gene name")
    target: str = Field(description="Target gene name") 
    interaction: int = Field(description="Interaction value (1 or -1)")

class NetworkAnalysis(BaseModel):
    thinking: str = Field(description="Thinking process for identifying the best edges to remove from the GRN")
    edge_removals: List[EdgeRemoval] = Field(description="List of edges to remove with reasoning")

def random_modify_network_edges(G: nx.DiGraph, 
                                add_num: int=1) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Randomly modify the edges of a directed graph."""
    modified_G = G.copy()
    edges_to_add = add_num

    modification_log = {
        'removed': [],
        'added': [],
        'changed': []
    }

    # Add edges
    if edges_to_add > 0:
        all_nodes = list(modified_G.nodes())
        count = 0
        while count < edges_to_add:
            source = random.choice(all_nodes)
            target = random.choice(all_nodes)
            if source != target and not modified_G.has_edge(source, target):
                interaction = random.choice([1, -1])
                modified_G.add_edge(source, target, interaction=interaction)
                modification_log['added'].append([source, target, interaction])
                count += 1

    return modified_G, modification_log
    
def ICL_example(G: nx.DiGraph, modification_log, ICL_size: int = 10):
    valid_example = []
    invalid_example = []
    
    # Convert modification_log to set for O(1) lookup
    added_edges = set((edges[0], edges[1]) for edges in modification_log['added'])
    
    all_nodes = list(G.nodes())
    existing_edges = set(G.edges())
    all_edges_with_data = list(G.edges(data=True))
    
    # Early return if no edges exist
    if not all_edges_with_data:
        return valid_example, invalid_example
    
    # Pre-sample valid edges (without replacement)
    try:
        sampled_valid_edges = random.sample(all_edges_with_data, ICL_size)
    except ValueError:
        print(f"ICL_size ({ICL_size}) bigger than total edge number ({len(all_edges_with_data)}).")
        print(f"Using all available edges instead.")
        sampled_valid_edges = all_edges_with_data.copy()

    valid_edge_iter = iter(sampled_valid_edges)  # Create iterator
    
    # Track generated invalid edges to prevent duplicates
    generated_invalid_edges = set()
    
    attempts = 0
    max_attempts = ICL_size * 1000  # Increased for safety
    
    while len(invalid_example) < ICL_size and attempts < max_attempts:
        attempts += 1
        
        # Generate random edge
        source = random.choice(all_nodes)
        target = random.choice(all_nodes)
        interaction = random.choice([1, -1])
        
        # Check if this would be a valid invalid example AND not already generated
        if (source != target and 
            (source, target) not in existing_edges and 
            (source, target) not in added_edges and
            (source, target) not in generated_invalid_edges):  # NEW: Check for duplicates
            
            # Add to tracking set
            generated_invalid_edges.add((source, target))
            
            # Add invalid example
            invalid_example.append([source, target, interaction])

            # Add next unique valid example
            try:
                random_valid_edge = next(valid_edge_iter)
                valid_example.append([
                    random_valid_edge[0], 
                    random_valid_edge[1], 
                    random_valid_edge[2].get('interaction', 1)
                ])
            except StopIteration:
                # No more unique edges available
                print("No more unique edges available for valid examples")
                break
    
    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts ({max_attempts}). Generated {len(invalid_example)} invalid examples.")

    return valid_example, invalid_example

def nx_graph_to_layered_structure(G: nx.DiGraph, start_gene: str, max_layer: int = 3) -> Dict:
    """
    Convert a directed graph to a layered structure starting from a given gene.
    This matches the behavior of trace_downstream_network function.
    """
    layered_structure = {}
    
    # Get first layer targets
    first_layer_nodes = set()
    first_layer_interactions = {}
    
    if start_gene in G:
        targets = []
        for target in G.successors(start_gene):
            interaction = G[start_gene][target].get('interaction')
            targets.append((target, interaction))
            first_layer_nodes.add(target)
   
        if targets:
            first_layer_interactions[start_gene] = targets
    
    # Initialize first layer
    if first_layer_nodes:
        layered_structure[1] = {
            'genes': first_layer_nodes.copy(),
            'interactions': first_layer_interactions
        }
    
    current_layer_nodes = first_layer_nodes
    # Continue with subsequent layers
    for layer in range(2, max_layer + 1):
        if not current_layer_nodes:
            break
            
        next_layer_nodes = set()
        layer_interactions = {}
        for source in current_layer_nodes:
            if source in G:
                targets = []
                for target in G.successors(source):
                    interaction = G[source][target].get('interaction')
                    targets.append((target, interaction))
                    next_layer_nodes.add(target)
                
                if targets:
                    layer_interactions[source] = targets
        
        if next_layer_nodes:
            layered_structure[layer] = {
                'genes': next_layer_nodes.copy(),
                'interactions': layer_interactions
            }
        
        current_layer_nodes = next_layer_nodes

    return layered_structure

def convert_network_to_text(layered_structure: Dict[int, Dict[str, Any]], start_gene: str) -> str:
    """
    Convert the layered structure of the network to a text format.
    """
    text_representation = f"Start gene: {start_gene}\n"
    
    for layer, data in layered_structure.items():
        text_representation += f"Layer {layer}:\n"
        text_representation += "Source genes, Target genes, Interactions:\n"
        
        for source, targets in data['interactions'].items():
            for target, interaction in targets:
                text_representation += f"{source}, {target}, {interaction}\n"
        
        text_representation += "\n"
    
    return text_representation

def parse_structured_response(response_content: str) -> List[List]:
    """
    Parse the structured JSON response and extract edge removals in the original format.
    """
    try:
        # Parse the JSON response
        analysis = NetworkAnalysis.model_validate_json(response_content)
        
        # Convert to the original format expected by the rest of the code
        results = []
        for edge_removal in analysis.edge_removals:
            results.append([edge_removal.source, edge_removal.target, edge_removal.interaction])
        
        return results
    except Exception as e:
        print(f"Error parsing structured response: {e}")
        # Fallback to original regex parsing
        return llm_result_to_edges(response_content)

def llm_result_to_edges(llm_result: str):
    # Find all lines that match the pattern [REMOVE, gene1, gene2, number]
    pattern = r'\[REMOVE,\s*([^,]+),\s*([^,]+),\s*(\d+)\]'
    matches = re.findall(pattern, llm_result)
    
    # Reconstruct the full commands
    results = []
    for match in matches:
        results.append([match[0].strip(), match[1].strip(), int(match[2])])
    
    return results

# def evaluate(gt_edges, pred_edges):
#     true_positives = len(gt_edges.intersection(pred_edges))
#     false_positives = len(pred_edges - gt_edges)
#     false_negatives = len(gt_edges - pred_edges)

#     # Calculate Precision and Recall
#     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     return precision, recall, f1

def calculate_mrr(gt_edges, pred_edges_ranked):
    """
    Calculate Mean Reciprocal Rank
    
    Args:
        gt_edges: Set of ground truth edges
        pred_edges_ranked: List of predicted edges in ranking order (best predictions first)
    
    Returns:
        float: MRR score between 0 and 1
    """
    if len(gt_edges) == 0:
        return 0.0
    
    reciprocal_ranks = []
    
    for gt_edge in gt_edges:
        # Find the rank (1-indexed) of this ground truth edge in predictions
        try:
            rank = pred_edges_ranked.index(gt_edge) + 1  # +1 for 1-indexed ranking
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # Ground truth edge not found in predictions
            reciprocal_ranks.append(0.0)
    
    mrr = sum(reciprocal_ranks) / len(gt_edges)
    return mrr

# Enhanced evaluation function that includes MRR
def evaluate(gt_edges, pred_edges_ranked):
    """
    Comprehensive evaluation including MRR
    
    Args:
        gt_edges: Set of ground truth edges
        pred_edges_ranked: List of predicted edges in ranking order
    
    Returns:
        tuple: (precision, recall, f1, mrr)
    """
    pred_edges = set(pred_edges_ranked)  # Convert to set for other metrics
    
    true_positives = len(gt_edges.intersection(pred_edges))
    false_positives = len(pred_edges - gt_edges)
    false_negatives = len(gt_edges - pred_edges)

    # Calculate Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate MRR
    mrr = calculate_mrr(gt_edges, pred_edges_ranked)
    
    return precision, recall, f1, mrr

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load Biomni corrected PKN. This PKN has a correct MOON score patter.
    pkn_file = Path("./data/corrected_PKN_with_BRAF_fixes.csv")
    if not pkn_file.exists():
        raise FileNotFoundError(f"PKN file {pkn_file} does not exist.")
    pkn = pd.read_csv(pkn_file)

    # Set the model
    model = 'qwen3:8b'
    # model = 'qwen3:30b'
    print(f"Using model: {model}")
    result_path = Path(f"./results/llm/recover_edge_ICL/{model.replace(':', '_')}")
    print(f"Results will be saved to: {result_path}")
    if not result_path.exists():
        result_path.mkdir(parents=True)
    
    # Get downstream network of the target gene
    start_gene = "BRAF"
    max_layers = [3]
    repeat = 100
    add_nums = [1, 2, 4, 8, 16]
    ICL_sizes = [0, 10]
    
    for max_layer in max_layers:
        downstream_network = trace_downstream_network(pkn, start_gene, max_layer=max_layer)
        # Convert the downstream network to networkx graph
        G = nx.DiGraph()
        for layer, data in downstream_network.items():
            for source, targets in data['interactions'].items():
                for target, interaction in targets:
                    G.add_edge(source, target, interaction=interaction)

        for add_num in add_nums:
            for ICL_size in ICL_sizes:
                for i in tqdm(range(repeat)):               
                    G_modify, modification_log = random_modify_network_edges(G, add_num=add_num)
                    valid_example, invalid_example = ICL_example(G, modification_log, ICL_size=ICL_size)
                    valid_example_text = convert_edge_list_to_text(valid_example)
                    # valid_example_text = "\nSource genes, Target genes, Interactions:\nMAPK3, TP53, 1\nMAPK1, MYC, 1\nMAPK3, GSK3B, -1\nMAPK1, FOXO3, -1\nMAPK3, BRAF, -1\nMAPK1, APC_AXIN1_GSK3B, -1\nMAPK3, BCL2, 1\nMAPK1, RPS6KB1, 1\nMAPK3, HIF1A, 1\nMAPK1, PPARG, -1\n"
                    invalid_example_text = convert_edge_list_to_text(invalid_example)

                    layer_struct_G_modify = nx_graph_to_layered_structure(G_modify, start_gene, max_layer=max_layer+1)

                    downstream_network_text = convert_network_to_text(layer_struct_G_modify, start_gene)

                    # Load prompt
                    prompt_file = Path(f"./prompts/prompt_recover_edges_ICL.txt")
                    if not prompt_file.exists():
                        raise FileNotFoundError(f"Prompt file {prompt_file} does not exist.")
                    with open(prompt_file, 'r') as f:
                        prompt = f.read()
                    # Replace placeholders in the prompt with actual values
                    prompt = prompt.format(start_gene=start_gene,
                                        valid_example_text=valid_example_text,
                                        invalid_example_text=invalid_example_text,
                                        downstream_network_text=downstream_network_text)
                    # save the prompt
                    prompt_file = Path(f"prompt_used_add_num_{add_num}_max_layer_{max_layer}_example_size_{ICL_size}_{i}.txt")
                    with open(result_path/prompt_file, 'w') as f:
                        f.write(prompt)
                    
                    
                    # Use structured output with ollama.generate
                    result = ollama.generate(
                        model=model, 
                        prompt=prompt,
                        format=NetworkAnalysis.model_json_schema(),
                        options={"temperature": 0.7},
                        stream=False
                    )
                    
                    with open(result_path / f"llm_response_add_num_{add_num}_max_layer_{max_layer}_example_size_{ICL_size}_{i}.txt", 'w') as f:
                        f.write(result['response'])

                    ground_truth = modification_log['added']
                    llm_result_edges = parse_structured_response(result['response'])

                    gt_edges = set((edge[0], edge[1]) for edge in ground_truth)
                    pred_edges = [(edge[0], edge[1]) for edge in llm_result_edges]
     
                    precision, recall, f1, mrr = evaluate(gt_edges, pred_edges)

                    # Save the ground truth and predicted edges to dictionary and save to file
                    eval_dict = {
                        'original_num_edges': G.number_of_edges(),
                        'original_num_nodes': G.number_of_nodes(),
                        'modified_num_edges': G_modify.number_of_edges(),
                        'modified_num_nodes': G_modify.number_of_nodes(),
                        'ground_truth': list(gt_edges),
                        'predicted': list(pred_edges),
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'mrr': mrr
                    }
                    with open(result_path / f"evaluation_results_add_num_{add_num}_max_layer_{max_layer}_example_size_{ICL_size}_{i}.json", "w") as f:
                        json.dump(eval_dict, f, indent=4)