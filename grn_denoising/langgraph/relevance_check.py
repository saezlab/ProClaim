"""Relevance checking module for evaluating reasoning quality."""

import json
from pathlib import Path
from langchain_openai import ChatOpenAI


def get_relevance_grammar() -> str:
    """Get the JSON grammar for structured relevance output.

    Returns:
        Grammar string for xgrammar-based decoding
    """
    return r'''
root ::= object
object ::= "{" ws "\"relevant\"" ws ":" ws boolean ws "}"
boolean ::= "true" | "false"
ws ::= [ \t\n]*
'''


def evaluate_relevance(
    reasoning: str,
    original_question: str,
    reasoning_llm: ChatOpenAI
) -> bool:
    """Evaluate whether reasoning is relevant to the original question.

    Args:
        reasoning: The reasoning text to evaluate
        original_question: The original question asked
        reasoning_llm: ChatOpenAI instance for evaluation

    Returns:
        True if reasoning is relevant, False otherwise
    """
    json_grammar = get_relevance_grammar()

    # Evaluate relevance with clear instructions
    relevance_prompt = f"""You are evaluating reasoning relevance. You must respond with ONLY a JSON object with a "relevant" field.
Original Question: "{original_question}" Reasoning provided: "{reasoning}" Is the reasoning relevant and focused on answering the original question? Respond with JSON only, example format: {{"relevant": true}} or {{"relevant": false}}"""

    try:
        response = reasoning_llm.invoke(
            relevance_prompt,
            temperature=1,
            max_tokens=10,
            extra_body={"guided_decoding_backend": "xgrammar", "guided_grammar": json_grammar}
        )

        content = response.content.strip()

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

    return is_relevant


def save_attempt_result(
    attempt_number: int,
    original_question: str,
    structured_outcome,
    is_relevant: bool,
    use_search: bool,
    output_dir: str = "./results/negative_edges"
) -> Path:
    """Save attempt result to JSON file.

    Args:
        attempt_number: The attempt number
        original_question: The original question
        structured_outcome: The structured outcome object
        is_relevant: Whether reasoning is relevant
        use_search: Whether search was used
        output_dir: Directory to save results

    Returns:
        Path to saved file
    """
    # Create results directory if it doesn't exist
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save this attempt to a separate JSON file
    search_suffix = "_with_search" if use_search else "_no_search"
    output_file = results_dir / f"gene_regulation_attempt_{attempt_number}{search_suffix}.json"

    attempt_data = {
        "attempt_number": attempt_number,
        "question": original_question,
        "structured_outcome": structured_outcome.model_dump(),
        "is_relevant": is_relevant,
        "used_search": use_search
    }

    with open(output_file, 'w') as f:
        json.dump(attempt_data, f, indent=2)

    return output_file
