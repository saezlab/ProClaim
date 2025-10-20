from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import requests
from pydantic import BaseModel, Field
import math
import json


class Alternative(BaseModel):
    token: str = Field(description="The alternative token")
    probability: float = Field(default=0.0, description="Probability of the token")
    logprob: float = Field(default=0.0)


class StructuredOutcome(BaseModel):
    reasoning: str = Field(description="LLM's reasoning")
    answer_text: str = Field(description="Raw text answer from LLM")
    answer: bool = Field(description="True/False answer")
    probability: float = Field(description="Probability of the answer")
    alternatives: list[Alternative] = Field(default_factory=list, description="Alternative answers")


class GraphState(TypedDict):
    messages: Annotated[list[str], operator.add]
    response: str
    structured_outcome: StructuredOutcome | None
    retry_count: int
    probability_threshold: float
    relevance_score: float  # -1 indicates error, 0.0-1.0 indicates relevance
    original_question: str  # Track the original question for relevance evaluation
    max_retries: int  # Maximum number of retry attempts
    relevance_threshold: float  # Minimum relevance score required


def query_llm(
    prompt: str,
    api_url: str = "http://localhost:8080",
    **kwargs
) -> dict:
    url = f"{api_url}/completion"

    payload = {
        "prompt": prompt,
        "temperature": kwargs.get("temperature", 0.1),
        "n_predict": kwargs.get("max_tokens", -1),
        "stream": False,
        "n_probs": kwargs.get("n_probs", 5),
        "post_sampling_probs": True,
    }

    for key, value in kwargs.items():
        if key not in ["temperature", "max_tokens", "n_probs"]:
            payload[key] = value

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")


def ask_gene_regulation_question(
    source_gene: str,
    target_gene: str,
    relationship: str = "activate",
    api_url: str = "http://localhost:8080",
    n_probs: int = 5
) -> StructuredOutcome:
    prompt = f"""<|start|>system<|message|>You are a molecular biologist expert in biological interaction.
You must analyze the question and provide reasoning, then answer with ONLY 'Yes' or 'No'.<|end|>
<|start|>user<|message|>Question: Does {source_gene} {relationship} {target_gene}?

First provide your reasoning, then on a new line write "Answer: Yes" or "Answer: No".

Reasoning:<|end|>
<|start|>assistant<|message|>"""

    response = query_llm(
        prompt=prompt,
        api_url=api_url,
        temperature=0.7,
        max_tokens=-1,
        n_probs=n_probs,
        repeat_penalty=1.2,
        repeat_last_n=64,
        stop=["<|end|>", "<|start|>"]
    )

    full_content = response.get("content", "").strip()
    completion_probs = response.get("completion_probabilities", [])

    # Find the answer line and corresponding token
    lines = full_content.split('\n')
    reasoning = full_content
    answer_text = "No"
    answer_line_index = -1

    # Look for answer line with keywords (thus, answer, etc.) containing yes/no
    # Search from the beginning to find the first conclusive answer
    for i in range(len(lines) - 1, -1, -1):
        line_lower = lines[i].strip().lower()
        if (("answer" in line_lower or "thus" in line_lower or "conclusion" in line_lower or "final" in line_lower)
            and ("yes" in line_lower or "no" in line_lower)):
            answer_text = lines[i].strip()
            answer_line_index = i
            break

    # If no answer line found with keywords, fall back to last line with yes/no
    if answer_line_index == -1:
        for i in range(len(lines) - 1, -1, -1):
            line_lower = lines[i].strip().lower()
            if "yes" in line_lower or "no" in line_lower:
                answer_text = lines[i].strip()
                answer_line_index = i

                break

    # Find the yes/no token in the answer line by reconstructing position in full text
    answer_value = False
    probability = -1.0
    alternatives = []

    if completion_probs and answer_line_index >= 0:
        # Calculate character position where answer line starts and ends
        char_pos_before_answer = sum(len(lines[i]) + 1 for i in range(answer_line_index))  # +1 for \n
        char_pos_after_answer = char_pos_before_answer + len(lines[answer_line_index])
        print(f"Answer line: ", full_content[char_pos_before_answer:char_pos_after_answer])
        # Find token that corresponds to yes/no in the answer line (search backwards)
        answer_token_index = -1

        # Build position mapping for all tokens
        token_positions = []
        current_pos = 0
        for i, token_data in enumerate(completion_probs):
            token_str = token_data.get("token", "")
            token_positions.append((i, current_pos, current_pos + len(token_str)))
            current_pos += len(token_str)

        # Search from the end for yes/no token within answer line region
        for i in range(len(token_positions) - 1, -1, -1):
            token_idx, start_pos, _ = token_positions[i]
            token_data = completion_probs[token_idx]
            token_str = token_data.get("token", "")
            token_lower = token_str.strip().lower()

            # Check if token is in the answer line region and is yes/no
            if start_pos >= char_pos_before_answer and start_pos < char_pos_after_answer and token_lower in ["yes", "no"]:
                answer_token_index = token_idx
                break

        # If we found the token, extract probabilities
        if answer_token_index >= 0:
            answer_token = completion_probs[answer_token_index]
            top_probs_list = answer_token.get("top_probs", [])

            for alt_data in top_probs_list:
                if isinstance(alt_data, dict):
                    token = alt_data.get("token", "")
                    prob = alt_data.get("prob", 0.0)
                    logprob = math.log(prob) if prob > 0 else -float('inf')

                    alternatives.append(Alternative(
                        token=token.strip(),
                        probability=prob,
                        logprob=logprob
                    ))

            generated = answer_token.get("token", "")
            probability = answer_token.get("prob", 0.0)

            if "yes" in generated.lower():
                answer_value = True
            elif "no" in generated.lower():
                answer_value = False

    return StructuredOutcome(
        reasoning=reasoning,
        answer_text=answer_text,
        answer=answer_value,
        probability=probability,
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

    # Call the LLM
    result = ask_gene_regulation_question(
        source_gene=source_gene,
        target_gene=target_gene,
        relationship=relationship,
        n_probs=10
    )

    return {
        "messages": [f"Attempt {state['retry_count'] + 1}: {source_gene} {relationship} {target_gene}"],
        "response": result.answer_text,
        "structured_outcome": result,
        "retry_count": state["retry_count"] + 1
    }


def reflect_node(state: GraphState) -> GraphState:
    """Node that reflects on the LLM response quality and reasoning relevance."""
    if state["structured_outcome"] is None:
        return {"relevance_score": -1.0}

    reasoning = state["structured_outcome"].reasoning
    original_question = state["original_question"]

    # JSON grammar to ensure structured output
    json_grammar = r'''
root ::= object
object ::= "{" ws "\"score\"" ws ":" ws number ws "}"
number ::= "0." [0-9]+
ws ::= [ \t\n]*
'''

    # Evaluate relevance with clear instructions
    relevance_prompt = f"""<|start|>system<|message|>You are evaluating reasoning relevance.
You must respond with ONLY a JSON object with a "score" field.<|end|>
<|start|>user<|message|>Original Question: {original_question}

Reasoning provided: {reasoning}

Rate how relevant and focused the reasoning is to answering the original question.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Respond with JSON only, example format: {{"score": <your_number_here>}}<|end|>
<|start|>assistant<|message|>"""

    try:
        response = query_llm(
            prompt=relevance_prompt,
            temperature=0.1,
            max_tokens=30,
            grammar=json_grammar
        )

        content = response.get("content", "").strip()

        # Parse JSON response
        parsed = json.loads(content)
        relevance_score = float(parsed.get("score", -1.0))

        # Clamp to valid range (0.0-1.0)
        if relevance_score < 0.0 or relevance_score > 1.0:
            relevance_score = -1.0
            print(f"Warning: Invalid relevance score {relevance_score}, setting to -1")

        print(f"Relevance evaluation - Raw: '{content}' -> Score: {relevance_score}")

    except (json.JSONDecodeError, ValueError, KeyError, RuntimeError) as e:
        # Set to -1 to indicate error
        relevance_score = -1.0
        print(f"Error in relevance evaluation: {e}, setting score to -1")

    # Save this attempt to a separate JSON file
    attempt_number = state["retry_count"]
    output_file = f"gene_regulation_attempt_{attempt_number}.json"

    attempt_data = {
        "attempt_number": attempt_number,
        "question": original_question,
        "structured_outcome": state["structured_outcome"].model_dump(),
        "relevance_score": relevance_score
    }

    with open(output_file, 'w') as f:
        json.dump(attempt_data, f, indent=2)

    print(f"Saved attempt {attempt_number} to {output_file}")

    return {"relevance_score": relevance_score}


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

    # Check relevance score
    relevance = state.get("relevance_score", -1.0)  # Default to -1 (error) if not present
    relevance_threshold = state.get("relevance_threshold", 0.7)  # Get from state, default to 0.7

    # If relevance evaluation failed (score = -1), we should retry
    if relevance < 0.0:
        print(f"Relevance evaluation failed (score={relevance}), retrying...")
        return "llm"

    # If relevance is too low, retry
    if relevance < relevance_threshold:
        print(f"Low relevance score: {relevance} < {relevance_threshold}, retrying...")
        return "llm"

    return END


def build_graph() -> StateGraph:
    """Build the LangGraph with start -> llm -> reflect -> (retry or end) structure."""
    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("llm", llm_node)
    workflow.add_node("reflect", reflect_node)

    # Set entry point
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
    # Build the graph
    app = build_graph()

    # Example: Does HCK activate BCR?
    # Format: "source_gene|target_gene|relationship"
    source_gene = "HCK"
    target_gene = "BCR"
    relationship = "activate"

    initial_state = {
        "messages": [f"{source_gene}|{target_gene}|{relationship}"],
        "response": "",
        "structured_outcome": None,
        "retry_count": 0,
        "probability_threshold": 0.9,
        "relevance_score": -1.0,  # Initialize to error state
        "original_question": f"Does {source_gene} {relationship} {target_gene}?",
        "max_retries": 10,  # Maximum number of retry attempts
        "relevance_threshold": 0.7  # Minimum relevance score required
    }

    # Run the graph
    result = app.invoke(initial_state)

    # Save result to JSON file
    output_file = "gene_regulation_result.json"
    if result["structured_outcome"]:
        result_data = result["structured_outcome"].model_dump()
        result_data["relevance_score"] = result.get("relevance_score", -1.0)

        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"{'='*60}")
        print(f"Question: {result['original_question']}")
        print(f"Answer: {result['structured_outcome'].answer}")
        print(f"Answer probability: {result['structured_outcome'].probability:.3f}")
        print(f"Relevance score: {result.get('relevance_score', -1.0):.3f}")
        print(f"Total attempts: {result['retry_count']}")
        print(f"{'='*60}\n")

    # print(app.get_graph().draw_mermaid())
    # app.get_graph().print_ascii()