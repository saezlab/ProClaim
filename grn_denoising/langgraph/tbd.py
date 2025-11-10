from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import requests
from pydantic import BaseModel, Field
import math
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

load_dotenv()

def query_llm(
    prompt: str,
    api_url: str = "http://localhost:8080",
    **kwargs
) -> dict:
    """Query the LLM API with the given prompt and parameters."""
    url = f"{api_url}/completion"

    payload = {
        "prompt": prompt,
        "temperature": kwargs.get("temperature", 1),
        "n_predict": kwargs.get("max_tokens", -1),
        "stream": False,
        "n_probs": kwargs.get("n_probs", 5),
        "post_sampling_probs": False,
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
    
if __name__ == "__main__":
    # Loop through all reasoning files in the specified directories
    base_dir = Path("results")
    # search_types = ['no_search', 'with_search']
    search_types = ['with_search']
    for search_type in search_types:
        combined_results = {}
        print(f"Processing search type: {search_type}")
        for run_dir in tqdm([f"negative_edges_run_{i}" for i in range(13)]):
            reasoning_dir = base_dir / run_dir
            for reasoning_file in reasoning_dir.glob(f"*{search_type}*.json"):
                # print(f"Processing file: {reasoning_file}")
                with open(reasoning_file, "r") as f:
                    reasoning_data = json.load(f)
                
                # Extract source and target from the file name
                file_name = reasoning_file.stem  # Get the file name without extension
                parts = file_name.split("_")
                source = parts[-4]  # Assuming the source is the fourth last part
                target = parts[-3]  # Assuming the target is the third last part

                reasoning = reasoning_data["reasoning"]
                ptompt = "Now I will rate my confidence in the proposed answer as either 0 or 1.  Proposed confidence: ("
                prompt = reasoning + "\n" + ptompt
                # Use qwen2.5-14b to rate confidence
                response = query_llm(
                    prompt=prompt,
                    temperature=0.02,
                    max_tokens=8,
                    n_probs=3
                )

                # Extract logprob value of "1" in the first token's top_logprobs
                completion_probs = response.get("completion_probabilities", [])
                logprob_of_1 = None
                if completion_probs and isinstance(completion_probs[0], dict):
                    top_logprobs = completion_probs[0].get("top_logprobs", [])
                    if isinstance(top_logprobs, list):
                        for entry in top_logprobs:
                            if entry.get("token") == "1":
                                logprob_of_1 = entry.get("logprob")
                                break

                # Store the logprob value in the combined dictionary as a list
                if (source, target) not in combined_results:
                    combined_results[(source, target)] = []
                combined_results[(source, target)].append(np.exp(logprob_of_1))

        # Save all results to a single JSON file
        output_file = base_dir / f"llm_confidence_combined_results_{search_type}.json"
        # Convert tuple keys to strings for JSON serialization
        json_serializable_results = {f"{key[0]}_{key[1]}": value for key, value in combined_results.items()}
        with open(output_file, "w") as f:
            json.dump(json_serializable_results, f, indent=4)