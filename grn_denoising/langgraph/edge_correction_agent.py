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


def load_removed_edges(csv_path: str = "../all_removed_edges_with_sources.csv") -> pd.DataFrame:
    """Load the removed edges from CSV file.

    Args:
        csv_path: Path to the all_removed_edges_with_sources.csv file

    Returns:
        DataFrame with columns including ENTITYA (source_gene), ENTITYB (target_gene), EFFECT (relationship)
    """
    csv_file = Path(__file__).parent / csv_path
    df = pd.read_csv(csv_file)

    # Filter to keep only protein-protein interactions
    df_ppi = df[(df['TYPEA'] == 'protein') & (df['TYPEB'] == 'protein')].copy()

    # Rename columns for clarity
    df_ppi.rename(columns={
        'ENTITYA': 'source_gene',
        'ENTITYB': 'target_gene',
        'EFFECT': 'relationship'
    }, inplace=True)

    return df_ppi


class Alternative(BaseModel):
    token: str = Field(description="The alternative token")
    logprob: float = Field(default=0.0)


class StructuredOutcome(BaseModel):
    reasoning: str = Field(description="LLM's reasoning")
    answer_text: str = Field(description="Raw text answer from LLM")
    answer: bool = Field(description="True/False answer")
    probability: float = Field(description="Probability of the answer")
    logprob: float = Field(default=-float('inf'), description="Log probability of the answer")
    alternatives: list[Alternative] = Field(default_factory=list, description="Alternative answers")


class GraphState(TypedDict):
    messages: Annotated[list[str], operator.add]
    response: str
    structured_outcome: StructuredOutcome | None
    retry_count: int
    probability_threshold: float
    is_relevant: bool  # True if reasoning is relevant, False otherwise
    original_question: str  # Track the original question for relevance evaluation
    max_retries: int  # Maximum number of retry attempts
    search_results: str  # Web search results to provide context for the LLM
    use_search: bool  # Whether to use web search or not


def query_llm(
    prompt: str,
    api_url: str = "http://localhost:8080",
    **kwargs
) -> dict:
    url = f"{api_url}/completion"

    payload = {
        "prompt": prompt,
        "temperature": kwargs.get("temperature", 1),
        "n_predict": kwargs.get("max_tokens", -1),
        "stream": False,
        "n_probs": kwargs.get("n_probs", 0),
        "post_sampling_probs": False,
    }

    # Handle logit_bias separately to ensure proper formatting
    # if "logit_bias" in kwargs:
    #     payload["logit_bias"] = kwargs["logit_bias"]

    for key, value in kwargs.items():
        if key not in ["temperature", "max_tokens", "n_probs"]:
            payload[key] = value

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")


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
        return f"{source} increase {target} expression"
    elif interaction == 'up-regulates quantity by expression':
        return f"{source} increase {target} expression"
    else:  # handles 'unknown' and any other unexpected interactions
        return f"{source} interact with {target}"


def ask_gene_regulation_question(
    source_gene: str,
    target_gene: str,
    relationship: str = "activate",
    search_context: str = "",
    api_url: str = "http://localhost:8080",
    n_probs: int = 5
) -> StructuredOutcome:
    
    base_system = """You are a molecular biologist expert in biological interactions. Focus on evidence from 2018 onwards. Answer with ONLY a SINGLE Yes or No.

"""

    examples = """Does p53 up-regulate BAX? Yes
Does insulin inhibit the activity of glucagon? No
Does TNF-alpha up-regulate apoptosis? Yes
Does AMPK activate mTOR? No
"""

    # Get natural language prompt based on relationship type
    interaction_prompt = get_interaction_prompt(source_gene, target_gene, relationship)

    if search_context:
        query = f"""Scientific context: {search_context}

Does {interaction_prompt}?"""
    else:
        query = f"""Does {interaction_prompt}?"""

    prompt = base_system + examples + query

    # Prepare kwargs for query_llm
    query_kwargs = {
        "temperature": 1.0,
        "max_tokens": 51200,
        "n_probs": n_probs,
        "repeat_penalty": 1.0,
        "repeat_last_n": 64
    }

    # Add logit_bias if provided
    # if logit_bias is not None:
    #     query_kwargs["logit_bias"] = logit_bias

    response = query_llm(
        prompt=prompt,
        api_url=api_url,
        **query_kwargs
    )

    # Save the raw response for debugging
    with open(f"llm_response_{source_gene}_{target_gene}.json", 'w') as f:
        json.dump(response, f, indent=2)

    full_content = response.get("content", "").strip()
    completion_probs = response.get("completion_probabilities", [])

    # Find the answer line and corresponding token
    lines = full_content.split('\n')
    reasoning = full_content
    answer_text = "No"

    # Look for the special marker: <|start|>assistant<|channel|>final<|message|>
    # The answer (Yes/No) comes after this marker
    marker = "<|start|>assistant<|channel|>final<|message|>"
    marker_pos = full_content.find(marker)
    # Find the yes/no token after the marker
    answer_value = False
    probability = -1.0
    answer_logprob = -float('inf')
    alternatives = []

    if completion_probs and marker_pos >= 0:
        # Calculate character position where to start looking for yes/no token
        # Start looking right after the marker
        char_pos_before_answer = marker_pos + len(marker)
        # Look for the answer in the next 100 characters (should be enough for "Yes" or "No")
        char_pos_after_answer = min(char_pos_before_answer + 100, len(full_content))

        # Extract the answer text for debugging
        answer_text = full_content[char_pos_before_answer:char_pos_after_answer].strip()
        # Find the first line with yes/no for better display
        answer_lines = answer_text.split('\n')
        for line in answer_lines:
            if 'yes' in line.lower() or 'no' in line.lower():
                answer_text = line.strip()
                break
        # print(f"Answer line: ", full_content[char_pos_before_answer:char_pos_after_answer])
        # Find token that corresponds to yes/no in the answer line (search backwards)
        answer_token_index = -1

        # Build position mapping for all tokens
        token_positions = []
        current_pos = 0
        for i, token_data in enumerate(completion_probs):
            token_str = token_data.get("token", "")
            token_positions.append((i, current_pos, current_pos + len(token_str)))
            current_pos += len(token_str)

        # Search from the marker position forward for the first yes/no token
        # We look for tokens that appear after the marker position
        for i in range(len(token_positions)):
            token_idx, start_pos, end_pos = token_positions[i]
            token_data = completion_probs[token_idx]
            token_str = token_data.get("token", "")
            token_lower = token_str.strip().lower()

            # Check if token is after the marker and is yes/no
            if start_pos >= char_pos_before_answer and token_lower in ["yes", "no"]:
                answer_token_index = token_idx
                print(f"[DEBUG] Found answer token '{token_str}' at position {start_pos} (token index {token_idx})")
                break

        # If we found the token, extract probabilities
        if answer_token_index >= 0:
            answer_token = completion_probs[answer_token_index]
            top_probs_list = answer_token.get("top_logprobs", [])

            for alt_data in top_probs_list:
                if isinstance(alt_data, dict):
                    token = alt_data.get("token", "")
                    logprob = alt_data.get("logprob", -float('inf'))
                    # Convert logprob to probability using exp
                    prob = math.exp(logprob) if logprob != -float('inf') else -1.0

                    alternatives.append(Alternative(
                        token=token.strip(),
                        logprob=logprob
                    ))

            generated = answer_token.get("token", "")
            # Get logprob and convert to probability using exp
            answer_logprob = answer_token.get("logprob", -float('inf'))
            probability = math.exp(answer_logprob) if answer_logprob != -float('inf') else -1.0

            # print(f"[DEBUG] Answer token: '{generated}', logprob: {answer_logprob}, probability: {probability}")
            # print(f"[DEBUG] Top alternatives: {[(alt.token, alt.logprob, math.exp(alt.logprob)) for alt in alternatives[:3]]}")

            if "yes" in generated.lower():
                answer_value = True
            elif "no" in generated.lower():
                answer_value = False
        else:
            print(f"[WARNING] Could not find yes/no token after marker")
            print(f"[DEBUG] Marker position: {marker_pos}")
            print(f"[DEBUG] Answer text: {answer_text}")
            print(f"[DEBUG] Total tokens: {len(completion_probs)}")

    return StructuredOutcome(
        reasoning=reasoning,
        answer_text=answer_text,
        answer=answer_value,
        probability=probability,
        logprob=answer_logprob,
        alternatives=alternatives
    )

def llm_node(state: GraphState) -> GraphState:
    """Node that calls the LLM and processes the response."""
    # Extract gene information from the first message
    # Expected format: "source_gene|target_gene|relationship"
    parts = state["messages"][0].split("|") if state["messages"] else []
    if len(parts) == 3:
        source_gene, target_gene, relationship = parts
    else:
        source_gene, target_gene, relationship = "HCK", "BCR", "activate"

    # Get search results from state
    search_context = state.get("search_results", "")

    # Call the LLM with search results as context
    result = ask_gene_regulation_question(
        source_gene=source_gene,
        target_gene=target_gene,
        relationship=relationship,
        search_context=search_context,
        n_probs=10
    )

    return {
        "messages": [f"Attempt {state['retry_count'] + 1}: {source_gene} {relationship} {target_gene}"],
        "response": result.answer_text,
        "structured_outcome": result,
        "retry_count": state["retry_count"] + 1
    }

def search_node(state: GraphState) -> GraphState:
    """Search for relevant publications using Tavily Search."""
    source_gene = state["messages"][0].split("|")[0]
    target_gene = state["messages"][0].split("|")[1]
    relationship = state["messages"][0].split("|")[2]

    # Use natural language prompt for search query
    interaction_prompt = get_interaction_prompt(source_gene, target_gene, relationship)
    query = f"Does {interaction_prompt}?"

    search_tool = TavilySearchResults(max_results=2, search_depth="advanced", start_date="2018-01-01")
    results = search_tool.invoke({"query": query})

    # Combine search results
    search_text = "\n\n".join([
        result.get('content', '')
        for result in results
    ])

    # Save raw results to a file for debugging
    # with open(f"search_results_{state['retry_count'] + 1}.txt", 'w') as f:
    #     f.write(search_text)

    return {
        "search_results": search_text
    }


def clean_reasoning(reasoning: str) -> str:
    """Remove prompt formatting artifacts from the reasoning text.

    This function removes special tokens used by the LLM while preserving
    the original content, spacing, and formatting.

    Args:
        reasoning: Raw reasoning text containing special tokens

    Returns:
        Cleaned reasoning text with only special tokens removed
    """
    # Remove all special tokens while preserving original content
    cleaned = reasoning

    # Remove channel-specific markers
    cleaned = cleaned.replace("<|start|>assistant<|channel|>analysis<|message|>", "")
    cleaned = cleaned.replace("<|start|>assistant<|channel|>final<|message|>", "")
    cleaned = cleaned.replace("<|start|>assistant<|channel|>commentary<|message|>", "")

    # Remove generic special tokens
    cleaned = cleaned.replace("<|start|>", "")
    cleaned = cleaned.replace("<|end|>", "")
    cleaned = cleaned.replace("<|message|>", "")
    cleaned = cleaned.replace("<|channel|>", "")

    return cleaned.strip()

def reflect_node(state: GraphState) -> GraphState:
    """Node that reflects on the LLM response quality and reasoning relevance."""
    if state["structured_outcome"] is None:
        return {"is_relevant": False}

    reasoning = state["structured_outcome"].reasoning
    cleand_reasoning = clean_reasoning(reasoning)
    original_question = state["original_question"]

    # JSON grammar to ensure structured output
    json_grammar = r'''
root ::= object
object ::= "{" ws "\"relevant\"" ws ":" ws boolean ws "}"
boolean ::= "true" | "false"
ws ::= [ \t\n]*
'''

    # Evaluate relevance with clear instructions
    relevance_prompt = f"""You are evaluating reasoning relevance. You must respond with ONLY a JSON object with a "relevant" field.
Original Question: "{original_question}" Reasoning provided: "{cleand_reasoning}" Is the reasoning relevant and focused on answering the original question? Respond with JSON only, example format: {{"relevant": true}} or {{"relevant": false}}"""

    try:
        response = query_llm(
            prompt=relevance_prompt,
            temperature=1,
            max_tokens=10,
            grammar=json_grammar
        )

        content = response.get("content", "").strip()

        # Parse JSON response
        parsed = json.loads(content)
        is_relevant = parsed.get("relevant", False)

        # Ensure it's a boolean
        if not isinstance(is_relevant, bool):
            is_relevant = False
            print(f"Warning: Invalid relevance type, setting to False")

        print(f"Relevance evaluation - Raw: '{content}' -> Relevant: {is_relevant}")

    except (json.JSONDecodeError, ValueError, KeyError, RuntimeError) as e:
        # Set to False to indicate error/failure
        is_relevant = False
        print(f"Error in relevance evaluation: {e}, setting to False")

    # Create results directory if it doesn't exist
    results_dir = Path("./results/negative_edges")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save this attempt to a separate JSON file
    attempt_number = state["retry_count"]
    search_suffix = "_with_search" if state.get("use_search", False) else "_no_search"
    output_file = results_dir / f"gene_regulation_attempt_{attempt_number}{search_suffix}.json"

    attempt_data = {
        "attempt_number": attempt_number,
        "question": original_question,
        "structured_outcome": state["structured_outcome"].model_dump(),
        "is_relevant": is_relevant,
        "used_search": state.get("use_search", False)
    }

    with open(output_file, 'w') as f:
        json.dump(attempt_data, f, indent=2)

    # print(f"Saved attempt {attempt_number} to {output_file}")

    return {"is_relevant": is_relevant}


def should_retry(state: GraphState) -> str:
    """Conditional edge function that decides whether to retry or end."""
    max_retries = state.get("max_retries", 3)  # Get from state, default to 3

    # Check if we have a structured outcome
    if state["structured_outcome"] is None:
        return "llm"

    # Check if max retries reached
    if state["retry_count"] >= max_retries:
        print(f"Max retries ({max_retries}) reached, ending...")
        return END

    # Check probability threshold
    probability = state["structured_outcome"].probability
    if probability < state["probability_threshold"]:
        print(f"Low probability: {probability:.3f} < {state['probability_threshold']}, retrying...")
        return "llm"

    # Check relevance
    is_relevant = state.get("is_relevant", False)  # Default to False if not present

    # If not relevant (either failed or not relevant), retry
    if not is_relevant:
        print(f"Reasoning not relevant (is_relevant={is_relevant}), retrying...")
        return "llm"

    return END


def build_graph(use_search: bool = False) -> StateGraph:
    """Build the LangGraph with optional search node.

    Args:
        use_search: If True, flow is search -> llm -> reflect -> (retry or end)
                   If False, flow is llm -> reflect -> (retry or end)
    """
    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("llm", llm_node)
    workflow.add_node("reflect", reflect_node)

    if use_search:
        workflow.add_node("search", search_node)
        # Set entry point to search
        workflow.set_entry_point("search")
        # Add edge from search to llm
        workflow.add_edge("search", "llm")
    else:
        # Set entry point to llm
        workflow.set_entry_point("llm")

    # Add edge from llm to reflect
    workflow.add_edge("llm", "reflect")

    # Add conditional edge from reflect (retry or end)
    workflow.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "llm": "llm",
            END: END
        }
    )

    # Compile the graph
    return workflow.compile()


if __name__ == "__main__":
    # Configuration
    USE_SEARCH = False  # Set to True to use web search, False to skip search
    NUM_REPETITIONS = 15  # Number of times to repeat the simulation

    # Load removed edges from CSV
    removed_edges_df = load_removed_edges()
    # print(f"Loaded {len(removed_edges_df)} removed protein-protein edges")
    # print(f"\nFirst few edges:")
    # print(removed_edges_df[['source_gene', 'target_gene', 'relationship']].head())

    # Build the graph
    app = build_graph(use_search=USE_SEARCH)

    # Repeat the simulation NUM_REPETITIONS times
    for run_number in tqdm(range(NUM_REPETITIONS)):
        # print(f"\n{'#'*80}")
        # print(f"# STARTING RUN {run_number}/{NUM_REPETITIONS - 1}")
        # print(f"{'#'*80}\n")

        # Create results directory for this run
        results_dir = Path(f"./results/negative_edges_run_{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Loop through all removed edges
        # removed_edges_df = removed_edges_df[0:1]
        for idx, edge in removed_edges_df.iterrows():
            source_gene = edge['source_gene']
            target_gene = edge['target_gene']
            relationship = edge['relationship']

            # print(f"\n{'='*60}")
            # print(f"Run {run_number} - Edge: {source_gene} -> {target_gene} ({relationship})")
            # print(f"{'='*60}")

            # Generate natural language question
            interaction_prompt = get_interaction_prompt(source_gene, target_gene, relationship)
            original_question = f"Does {interaction_prompt}?"

            initial_state = {
                "messages": [f"{source_gene}|{target_gene}|{relationship}"],
                "response": "",
                "structured_outcome": None,
                "retry_count": 0,
                "probability_threshold": 0.6,
                "is_relevant": False,  # Initialize to False
                "original_question": original_question,
                "max_retries": 5,  # Maximum number of retry attempts
                "search_results": "",  # Will be populated by search_node if USE_SEARCH is True
                "use_search": USE_SEARCH
            }

            # Run the graph
            result = app.invoke(initial_state)

            # Save result to JSON file with edge index in filename
            search_suffix = "_with_search" if USE_SEARCH else "_no_search"
            output_file = results_dir / f"gene_regulation_result_{source_gene}_{target_gene}{search_suffix}.json"

            if result["structured_outcome"]:
                result_data = result["structured_outcome"].model_dump()
                result_data["is_relevant"] = result.get("is_relevant", False)
                result_data["used_search"] = USE_SEARCH
                result_data["edge_index"] = int(idx)
                result_data["source_gene"] = source_gene
                result_data["target_gene"] = target_gene
                result_data["relationship"] = relationship
                result_data["run_number"] = run_number

                with open(output_file, 'w') as f:
                    json.dump(result_data, f, indent=2)

    #             print(f"\nFinal Results for edge {idx + 1}:")
    #             print(f"Question: {result['original_question']}")
    #             print(f"Answer: {result['structured_outcome'].answer}")
    #             print(f"Answer probability: {result['structured_outcome'].probability:.3f}")
    #             print(f"Reasoning relevant: {result.get('is_relevant', False)}")
    #             print(f"Total attempts: {result['retry_count']}")
    #             print(f"Results saved to: {output_file}")

    #     print(f"\n{'='*60}")
    #     print(f"Run {run_number} completed! Total edges processed: {len(removed_edges_df)}")
    #     print(f"Results saved in: {results_dir}")
    #     print(f"{'='*60}\n")

    # print(f"\n{'#'*80}")
    # print(f"# ALL {NUM_REPETITIONS} RUNS COMPLETED!")
    # print(f"{'#'*80}\n")

    # print(app.get_graph().draw_mermaid())
    # app.get_graph().print_ascii()