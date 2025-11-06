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
    # prompt = """Reasoning: Low. Randomly pick Yes or No. In the new line, return either "ANSWER: [Yes]" or "ANSWER: [No]"."""
    prompt = "You are a molecular biologist expert in biological interactions. Focus on evidence from 2018 onwards. Answer with ONLY a SINGLE Yes or No.\n\nDoes p53 up-regulate BAX? Yes\nDoes insulin inhibit the activity of glucagon? No\nDoes TNF-alpha up-regulate apoptosis? Yes\nDoes AMPK activate mTOR? No\nDoes FER activate CTTN?"
    response = query_llm(
        prompt=prompt,
        temperature=1.0,
        max_tokens=8,
        n_probs=3
    )
    with open("tbd_output.json", "w") as f:
        json.dump(response, f, indent=4)