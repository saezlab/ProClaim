import pandas as pd
import numpy as np
import random
from typing import Set, Tuple, Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import requests
import json


# LLM interaction functions
def query_llm(
    prompt: str,
    api_url: str = "http://localhost:8080",
    **kwargs
) -> dict:
    """Query the LLM API with the given prompt and parameters."""
    url = f"{api_url}/completion"

    payload = {
        "prompt": prompt,
        "temperature": kwargs.get("temperature", 0.1),
        "n_predict": kwargs.get("max_tokens", -1),
        "stream": False,
        "n_probs": kwargs.get("n_probs", 0),
        "post_sampling_probs": True,
        "grammar": kwargs.get("grammar", ""),
        "json_schema": kwargs.get("json_schema", None),
    }

    for key, value in kwargs.items():
        if key not in ["temperature", "max_tokens", "n_probs", "grammar", "json_schema"]:
            payload[key] = value

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")


def extract_json_from_gpt_oss(text):
    """Extract JSON from GPT-OSS-120B response with special tokens."""
    if '{"answer"' in text:
        start = text.rfind('{"answer"')
        end = text.find('}', start) + 1
        if end > start:
            return text[start:end]
    return text


def get_interaction_prompt(source: str, target: str, interaction: str) -> str:
    """Convert interaction type to natural language prompt."""
    if interaction == 'down-regulates':
        return f"{source} down-regulate {target}"
    elif interaction == 'down-regulates activity':
        return f"{source} inhibit the activity of {target}"
    elif interaction == 'form complex':
        return f"{source} form a complex with {target}"
    elif interaction == 'up-regulates':
        return f"{source} up-regulate {target}"
    elif interaction == 'up-regulates activity':
        return f"{source} activate {target}"
    elif interaction == 'up-regulates quantity':
        return f"{source} increase {target} expression?"
    elif interaction == 'up-regulates quantity by expression':
        return f"{source} increase {target} expression?"
    else:  # handles 'unknown' and any other unexpected interactions
        return f"{source} interact with {target}"


# JSON schema for structured output
json_schema = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["Yes", "No"]
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": ["answer", "reasoning"]
}

# System prompt for LLM
system_prompt = """You are a molecular biologist expert in biological interactions.
You need to determine whether the following sentence accurately describes the specified interaction between two entities.
You MUST respond with ONLY valid JSON in this exact format:
{"answer":"Yes","reasoning":"your detailed reasoning here"}

Rules:
- answer must be exactly "Yes" or "No"
- reasoning must be a single line string (escape quotes with \\")
- Do NOT include any text before or after the JSON
- Do NOT use markdown code blocks"""


def verify_positive_edges_with_llm(
    data_subset: pd.DataFrame,
    output_file: str = 'positive_edges_with_llm_reasoning_full.csv'
) -> pd.DataFrame:
    """
    Query LLM to verify all edges in the dataset.

    Args:
        data_subset: DataFrame containing edges to verify
        output_file: Path to save results

    Returns:
        DataFrame with llm_answer and llm_reasoning columns added
    """
    results_list = []

    for index in tqdm(range(len(data_subset))):
        source = data_subset['ENTITYA'].iloc[index]
        target = data_subset['ENTITYB'].iloc[index]
        interaction = data_subset['EFFECT'].iloc[index]
        sentence = data_subset['SENTENCE'].iloc[index]

        # Get interaction prompt
        interaction_prompt = get_interaction_prompt(source, target, interaction)

        # Build prompt and query LLM
        user_prompt = f"""Sentence: {sentence}
Interaction: {interaction_prompt}
Respond with JSON only:"""
        prompt = f"{system_prompt}\n\n{user_prompt}"

        # Query LLM
        try:
            result = query_llm(prompt, json_schema=json_schema, max_tokens=-1)
            response_text = result['content'].strip()
            cleaned = extract_json_from_gpt_oss(response_text)

            # Parse response
            data = json.loads(cleaned)
            answer = data['answer']
            reasoning = data['reasoning']

            # Append answer and reasoning to the row
            edge_dict = data_subset.iloc[index].to_dict()
            edge_dict['llm_answer'] = answer
            edge_dict['llm_reasoning'] = reasoning
            results_list.append(edge_dict)

            if answer == 'Yes':
                print(f"Index {index}: Yes ({len([r for r in results_list if r['llm_answer'] == 'Yes'])}/{len(results_list)})")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"✗ Error at index {index}: {e}. Raw: {response_text[:200]}")
            # Still append the row with error information
            edge_dict = data_subset.iloc[index].to_dict()
            edge_dict['llm_answer'] = 'Error'
            edge_dict['llm_reasoning'] = f"Parse error: {str(e)}"
            results_list.append(edge_dict)

    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(results_df)} edges to '{output_file}'")
    print(f"Total 'Yes' answers: {len(results_df[results_df['llm_answer'] == 'Yes'])}")
    print(f"Total 'No' answers: {len(results_df[results_df['llm_answer'] == 'No'])}")
    print(f"Total errors: {len(results_df[results_df['llm_answer'] == 'Error'])}")

    return results_df


def main():
    """Main execution function to process SIGNOR data and verify positive edges."""

    # Set up paths
    data_path = Path("../signor")
    data_jul_2025 = pd.read_csv(data_path / 'Jul2025_release.txt', sep='\t')

    # Load the removed edges CSV (should already exist from notebook)
    all_removed_edges_df = pd.read_csv('all_removed_edges_with_sources.csv')

    # Filter for protein-protein interactions only
    data_jul_2025_ppi = data_jul_2025[
        (data_jul_2025['TYPEA'] == 'protein') &
        (data_jul_2025['TYPEB'] == 'protein')
    ]

    # Get candidate source nodes from removed edges
    candidate_source_nodes = set()
    for row in all_removed_edges_df.itertuples():
        if row.TYPEA != 'protein' or row.TYPEB != 'protein':
            continue
        source_entity = row.ENTITYA
        target_entity = row.ENTITYB
        candidate_source_nodes.add(source_entity)
        candidate_source_nodes.add(target_entity)

    # Find positive edges containing these nodes
    index_list = []
    for node in list(candidate_source_nodes):
        indices = data_jul_2025_ppi[
            (data_jul_2025_ppi['ENTITYA'] == node) |
            (data_jul_2025_ppi['ENTITYB'] == node)
        ]
        index_list.extend(indices.index.tolist())

    # Create subset and remove duplicates
    data_jul_2025_ppi_subset = data_jul_2025_ppi.loc[index_list].drop_duplicates()

    # Reset index for easier iteration
    data_jul_2025_ppi_subset = data_jul_2025_ppi_subset.reset_index(drop=True)

    results_df = verify_positive_edges_with_llm(
        data_jul_2025_ppi_subset,
        output_file='positive_edges_with_llm_reasoning_full.csv'
    )

    return results_df


if __name__ == "__main__":
    results = main()
