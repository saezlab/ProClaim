import pandas as pd
import numpy as np
import random
import ollama
from typing import Set, Tuple
from pathlib import Path
from tqdm import tqdm

def llm_result_to_corrected_pkn(pkn: pd.DataFrame, llm_result: str, remove_num=1) -> pd.DataFrame:
    """
    Process LLM result to correct a PKN (Prior Knowledge Network) DataFrame.
    
    Args:
        pkn: DataFrame with columns ['source', 'target', 'interaction']
        llm_result: String containing LLM response with corrections
        remove_num: Number of lines to process from the end of llm_result
    
    Returns:
        Corrected PKN DataFrame
    """
    lines = llm_result.strip().split('\n')
    corrected_pkn = pkn.copy()
    
    # Process lines from the end backwards
    for i in range(remove_num):
        # Check if we have enough lines
        if i >= len(lines):
            print(f"Warning: Not enough lines in LLM result. Requested {remove_num} lines, but only {len(lines)} available.")
            break
            
        # The response should be in the last line with the answer between []
        result_line = lines[-(i+1)].strip()
        
        # Skip empty lines
        if not result_line:
            continue
        
        # Find the index of the first '[' and last ']'
        try:
            start_index = result_line.find('[')
            end_index = result_line.rfind(']')
            
            # Save the content between the brackets into a list
            if start_index == -1 or end_index == -1 or start_index >= end_index:
                raise ValueError(f"The result format is incorrect. Expected format: [ACTION, SOURCE, TARGET, INTERACTION]. Got: {result_line}")
            
            result_content = result_line[start_index + 1:end_index].strip()
            
            # Split the content by commas and strip whitespace
            result_parts = [part.strip() for part in result_content.split(',')]
            
            # Handle different formats based on action
            if len(result_parts) != 4:
                raise ValueError(f"The result format is incorrect. Expected 4 parts: [ACTION, SOURCE, TARGET, INTERACTION]. Got {len(result_parts)} parts: {result_parts}")
            
            action, source, target, interaction = result_parts
            
            print(f"Processing: Action: {action}, Source: {source}, Target: {target}, Interaction: {interaction}")
            
            if action == 'ADD':
                # First check if the edge already exists
                edge_mask = (corrected_pkn['source'] == source) & (corrected_pkn['target'] == target)
                edge_exists = edge_mask.any()
                
                if not edge_exists:
                    # If the edge does not exist, add it
                    new_edge = pd.DataFrame([[source, target, int(interaction)]], columns=corrected_pkn.columns)
                    corrected_pkn = pd.concat([corrected_pkn, new_edge], ignore_index=True)
                    print(f"Edge {source} -> {target} with interaction {interaction} added.")
                else:
                    # If the edge exists but with a different interaction, change it
                    old_interaction = corrected_pkn.loc[edge_mask, 'interaction'].iloc[0]
                    corrected_pkn.loc[edge_mask, 'interaction'] = int(interaction)
                    print(f"Edge {source} -> {target} already exists. Changed interaction from {old_interaction} to {interaction}.")
                    
            elif action == 'REMOVE':
                # Remove the edge if it exists
                edge_mask = (corrected_pkn['source'] == source) & (corrected_pkn['target'] == target)
                if edge_mask.any():
                    corrected_pkn = corrected_pkn[~edge_mask].reset_index(drop=True)
                    print(f"Edge {source} -> {target} removed.")
                else:
                    print(f"Action: REMOVE, but edge {source} -> {target} does not exist. No action taken.")
                    
            elif action == 'CHANGE':
                # Change the interaction of the edge if it exists
                edge_mask = (corrected_pkn['source'] == source) & (corrected_pkn['target'] == target)
                if edge_mask.any():
                    old_interaction = corrected_pkn.loc[edge_mask, 'interaction'].iloc[0]
                    corrected_pkn.loc[edge_mask, 'interaction'] = int(interaction)
                    print(f"Edge {source} -> {target} interaction changed from {old_interaction} to {interaction}.")
                else:
                    print(f"Action: CHANGE, but edge {source} -> {target} does not exist. No action taken.")
            else:
                raise ValueError(f"Unknown action: {action}. Expected one of [ADD, REMOVE, CHANGE].")
                
        except ValueError as ve:
            print(f"Validation error processing line '{result_line}': {ve}")
            continue  # Continue processing other lines instead of returning early
        except Exception as e:
            print(f"Unexpected error processing line '{result_line}': {e}")
            continue  # Continue processing other lines instead of returning early
    
    return corrected_pkn

if __name__ == "__main__":
    initial_pkn_file = Path("./data/clean_omnipath_PKN.csv")
    initial_pkn = pd.read_csv(initial_pkn_file)

    remove_edge_num = 8
    model = 'qwen3:8b'
    # model = 'claude_sonnet_4'
    print(f"Using model: {model}")
    result_path = Path(f"./results/moon_score/correct_{remove_edge_num}_edge/{model.replace(':', '_')}")

    repeat = 10
    for i in range(repeat):
        print(f"\nProcessing result {i+1}/{repeat}...")
        llm_result_file = result_path / f"result_{i}.txt"
        with open(llm_result_file, 'r') as f:
            llm_result = f.read()
        corrected_pkn = llm_result_to_corrected_pkn(initial_pkn, llm_result, remove_num=remove_edge_num)
        # Save the corrected PKN to a CSV file
        if initial_pkn.equals(corrected_pkn):
            print(f"No changes made in result {i+1}.")
            continue
        corrected_pkn_file = result_path/Path(f"corrected_PKN_{i}.csv")
        corrected_pkn.to_csv(corrected_pkn_file, index=False)
