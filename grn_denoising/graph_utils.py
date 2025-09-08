from typing import Dict, Set, List, Tuple, Any
import pandas as pd

def trace_downstream_network(pkn: pd.DataFrame, start_gene: str, max_layer: int=3) -> Dict[int, Dict[str, Any]]:
    """Trace downstream network layers from a starting gene"""
    layers = {}
    current_layer = {start_gene}
    
    for layer_num in range(max_layer):
        if not current_layer:
            break
            
        # Find all targets of genes in current layer
        next_layer = set()
        layer_interactions = {}
  
        for gene in current_layer:
            targets = pkn[pkn['source'] == gene]
            for _, row in targets.iterrows():
                next_layer.add(row['target'])
                if gene not in layer_interactions:
                    layer_interactions[gene] = []
                layer_interactions[gene].append((row['target'], row['interaction']))
        
        if next_layer:
            layers[layer_num + 1] = {
                'genes': next_layer,
                'interactions': layer_interactions
            }
            current_layer = next_layer
        else:
            break
    
    return layers

def convert_edge_list_to_text(edge_list: list):
    text_output = ""
    # text_output += "Source genes, Target genes, Interactions:\n"
    text_output += "Source genes, Target genes:\n"
    for row in edge_list:
        # text_output += f"{row[0]}, {row[1]}, {row[2]}\n"
        text_output += f"{row[0]}, {row[1]}\n"

    return text_output