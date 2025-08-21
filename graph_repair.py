import pandas as pd
import numpy as np
import random
import ollama
from typing import Set, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import re
import json
from graph_utils import trace_downstream_network

def random_modify_network_edges(G: nx.DiGraph, 
                                remove_rate: float=0.1,
                                add_rate: float=0.0,
                                change_rate: float=0.0) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Randomly modify the edges of a directed graph."""
    modified_G = G.copy()
    total_edges = modified_G.number_of_edges()
    edges_to_remove = int(total_edges * remove_rate)
    edges_to_add = int(total_edges * add_rate)
    edges_to_change = int(total_edges * change_rate)

    modification_log = {
        'removed': [],
        'added': [],
        'changed': []
    }

    # # Remove edges
    # if edges_to_remove > 0 and modified_G.number_of_edges() > 0:
    #     all_edges = list(modified_G.edges(data=True))
    #     edges_to_remove_list = random.sample(all_edges, min(edges_to_remove, len(all_edges)))
        
    #     for u, v, data in edges_to_remove_list:
    #         modification_log['removed'].append({
    #             'source': u,
    #             'target': v,
    #             'interaction': data.get('interaction')
    #         })
    #         modified_G.remove_edge(u, v)
        
    #     print(f"Removed {len(edges_to_remove_list)} edges")

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
        print(f"Added {count} edges")


    return modified_G, modification_log
    

def nx_graph_to_layered_structure(G: nx.DiGraph, start_gene: str, max_layers: int = 3) -> Dict:
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
    for layer in range(2, max_layers + 1):
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

def llm_result_to_edges(llm_result: str):
    # Find all lines that match the pattern [REMOVE, gene1, gene2, number]
    pattern = r'\[REMOVE,\s*([^,]+),\s*([^,]+),\s*(\d+)\]'
    matches = re.findall(pattern, llm_result)
    
    # Reconstruct the full commands
    results = []
    for match in matches:
        results.append([match[0].strip(), match[1].strip(), int(match[2])])
    
    return results

def evaluate(gt_edges, pred_edges):
    true_positives = len(gt_edges.intersection(pred_edges))
    false_positives = len(pred_edges - gt_edges)
    false_negatives = len(gt_edges - pred_edges)

    # Calculate Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load Biomni corrected PKN. This PKN has a correct MOON score patter.
    pkn_file = Path("./data/corrected_PKN_with_BRAF_fixes.csv")
    if not pkn_file.exists():
        raise FileNotFoundError(f"PKN file {pkn_file} does not exist.")
    pkn = pd.read_csv(pkn_file)

    # # Convert the PKN to a networkx graph
    # G = nx.from_pandas_edgelist(pkn, source='source', target='target', 
    #                              edge_attr='interaction', create_using=nx.DiGraph)
    # # Check if G has cycles
    # print(nx.is_directed_acyclic_graph(G))

    # Get downstream network of the target gene
    start_gene = "BRAF"
    for max_layers in [3]:
        downstream_network = trace_downstream_network(pkn, start_gene, max_layers=max_layers)
        print('original: ', downstream_network)
        # Convert the downstream network to networkx graph
        G = nx.DiGraph()
        for layer, data in downstream_network.items():
            for source, targets in data['interactions'].items():
                for target, interaction in targets:
                    G.add_edge(source, target, interaction=interaction)
        
        # # print(f"Original graph has {G.number_of_edges()} edges.")
        # Load prompt
        prompt_file = Path(f"./prompts/prompt_recover_edges.txt")
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file {prompt_file} does not exist.")
        with open(prompt_file, 'r') as f:
            prompt = f.read()

        if max_layers == 2:
            add_rates = [0.2, 0.3, 0.4]
        else:
            add_rates = [0.01, 0.02, 0.03]

        for add_rate in add_rates:
            repeat = 10
            for i in tqdm(range(repeat)):
                G_modify, modification_log = random_modify_network_edges(G, remove_rate=0.0, 
                                                                        add_rate=add_rate)
                # print(modification_log)
                # # Plot the modified graph
                # print(f"Modified graph has {G_modify.number_of_edges()} edges.")
                # plt.figure(figsize=(12, 8))
                # pos = nx.spring_layout(G_modify, seed=42)
                # nx.draw(G_modify, pos, with_labels=True, node_size=700, node_color='lightblue', 
                #         font_size=10, font_color='black', arrows=True)
                # edge_labels = nx.get_edge_attributes(G_modify, 'interaction')
                # nx.draw_networkx_edge_labels(G_modify, pos, edge_labels=edge_labels, font_color='red')
                # plt.title("Modified PKN with Random Edge Modifications")
                # plt.savefig("./results/modified_pkn_graph.png")

                # print(modification_log)
                # nx graph to text format {source}, {target}, {interaction}\n
                layer_struct_G_modify = nx_graph_to_layered_structure(G_modify, start_gene, max_layers=max_layers+1) # max_layers+1 becaude there can be a added edge from the last layer connect back to the previous layers' nodes
                # print('modified: ', layer_struct_G_modify)

                downstream_network_text = convert_network_to_text(layer_struct_G_modify, start_gene)
                # print(downstream_network_text)

                
                # Replace placeholders in the prompt with actual values
                prompt = prompt.format(start_gene=start_gene,
                                downstream_network_text=downstream_network_text)
                # Set the model and save the prompt
                model = 'qwen3:8b'
                print(f"Using model: {model}")
                result_path = Path(f"./results/llm/recover_edge/{model.replace(':', '_')}")
                print(f"Results will be saved to: {result_path}")
                if not result_path.exists():
                    result_path.mkdir(parents=True)
                prompt_file = Path(f"prompt_used_{i}_add_rate_{add_rate}_max_layers_{max_layers}.txt")
                with open(result_path/prompt_file, 'w') as f:
                    f.write(prompt)
                
                
                # llm response
                result = ollama.generate(model=model, prompt=prompt,
                                    options={"temperature": 0.7},
                                    stream=False)
                with open(result_path / f"llm_response_{i}_add_rate_{add_rate}_max_layers_{max_layers}.txt", 'w') as f:
                    f.write(result['response'])

                ground_truth = modification_log['added']
                llm_result_edges = llm_result_to_edges(result['response'])


                gt_edges = set((edge[0], edge[1]) for edge in ground_truth)
                pred_edges = set((edge[0], edge[1]) for edge in llm_result_edges)
                
                precision, recall = evaluate(gt_edges, pred_edges)

                # Save the ground truth and predicted edges to dictionary and save to file
                eval_dict = {
                    'ground_truth': list(gt_edges),
                    'predicted': list(pred_edges),
                    'precision': precision,
                    'recall': recall,
                }
                with open(result_path / f"evaluation_results_{i}_add_rate_{add_rate}_max_layers_{max_layers}.json", "w") as f:
                    json.dump(eval_dict, f, indent=4)
            