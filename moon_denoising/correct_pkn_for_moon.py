import pandas as pd
import numpy as np
import random
import ollama
from pathlib import Path
from tqdm import tqdm
from utils.graph_utils import trace_downstream_network



if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    remove_edge_num = [1, 2, 4, 8]
    for remove_num in remove_edge_num:
        print(f"Removing {remove_num} edges from the PKN for correction.")
        # Load the initial PKN
        initial_pkn_file = Path("./data/clean_omnipath_PKN.csv")
        initial_pkn = pd.read_csv(initial_pkn_file)
        print(f"Number of edges in the initial PKN: {len(initial_pkn)}")

        # Get downstream network of the target gene
        start_gene = "BRAF"
        start_gene_targets = initial_pkn[initial_pkn['source'] == start_gene]
        print(f"Number of edges in the downstream network of {start_gene}: \
            {len(start_gene_targets)}")
        # convert to a hashmap format with setted max_layers
        downstream_network = trace_downstream_network(initial_pkn, start_gene, 
                                                    max_layers=2)
        # Convert the downstream network to a text format for the prompt
        downstream_network_text = ""
        for layer, data in downstream_network.items():
            downstream_network_text += f"Layer {layer}:\n"
            downstream_network_text += f"Source genes, Target genes, Interactions:\n"
            for source, targets in data['interactions'].items():
                for target, interaction in targets:
                    downstream_network_text += f"{source}, {target}, {interaction}\n"
            downstream_network_text += "\n"

        # Load prompt
        prompt_file = Path(f"./prompts/prompt_remove_{remove_num}_edge.txt")
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file {prompt_file} does not exist.")
        with open(prompt_file, 'r') as f:
            prompt = f.read()
        # Replace placeholders in the prompt with actual values
        prompt = prompt.format(start_gene=start_gene,
                        downstream_network_text=downstream_network_text)

        # Set the model and save the prompt
        model = 'qwen3:8b'
        print(f"Using model: {model}")
        result_path = Path(f"./results/llm/correct_{remove_num}_edge/{model.replace(':', '_')}")
        print(f"Results will be saved to: {result_path}")
        if not result_path.exists():
            result_path.mkdir(parents=True)
        prompt_file = Path(f"prompt_used.txt")
        with open(result_path/prompt_file, 'w') as f:
            f.write(prompt)
        
        # llm
        repeat = 10
        for i in tqdm(range(repeat)):
            result = ollama.generate(model=model, prompt=prompt,
                                options={"temperature": 0.7},
                                stream=False)
            with open(result_path / f"result_{i}.txt", 'w') as f:
                f.write(result['response'])