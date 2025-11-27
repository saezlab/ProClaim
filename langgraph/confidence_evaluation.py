"""Confidence evaluation module using logprob analysis."""

import json
import math
from langchain_openai import ChatOpenAI
from llm_reasoning import StructuredOutcome


def evaluate_confidence(
    structured_outcome: StructuredOutcome,
    logprob_llm: ChatOpenAI
) -> tuple[float, float]:
    """Evaluate confidence in the reasoning and answer using logprobs.

    Args:
        structured_outcome: The structured outcome to evaluate
        logprob_llm: ChatOpenAI instance configured for logprob extraction

    Returns:
        Tuple of (confidence, logprob_1) where confidence is exp(logprob_1)
    """
    if structured_outcome is None:
        raise ValueError("Structured outcome is missing for confidence evaluation.")

    reasoning = structured_outcome.reasoning
    answer_text = structured_outcome.answer_text
    full_result = reasoning + "\n" + answer_text

    prompt = f"""Below is a reasoning process and final answer that needs confidence evaluation:

=== CONTENT TO EVALUATE ===
{full_result}
=== END OF CONTENT ===

Task: Rate your confidence in the above answer.
Output format: Provide only a single token - either 0 (low confidence) or 1 (high confidence).
Proposed confidence: ("""

    print(f"Confidence prompt: {prompt}")
    response = logprob_llm.invoke(prompt)

    # Save the raw response for debugging
    with open(f"llm_logprob_response.json", 'w') as f:
        response_dict = {
            "content": response.content,
            "response_metadata": response.response_metadata if hasattr(response, 'response_metadata') else {}
        }
        json.dump(response_dict, f, indent=2)

    # Access logprobs from the response_metadata
    logprob_1 = -math.inf
    confidence = 0.0

    if hasattr(response, 'response_metadata') and 'logprobs' in response.response_metadata:
        logprobs_data = response.response_metadata['logprobs']
        if 'content' in logprobs_data and len(logprobs_data['content']) > 0:
            first_token = logprobs_data['content'][0]
            if 'top_logprobs' in first_token:
                # Look for "1" in top_logprobs
                for logprob_item in first_token['top_logprobs']:
                    if logprob_item['token'] == '1':
                        logprob_1 = logprob_item['logprob']
                        confidence = math.exp(logprob_1)
                        break

    return confidence, logprob_1


def update_outcome_with_confidence(
    structured_outcome: StructuredOutcome,
    confidence: float,
    logprob: float
) -> StructuredOutcome:
    """Create new StructuredOutcome with updated confidence values.

    Args:
        structured_outcome: Original structured outcome
        confidence: Confidence score to set
        logprob: Log probability to set

    Returns:
        New StructuredOutcome with updated confidence values
    """
    return StructuredOutcome(
        reasoning=structured_outcome.reasoning,
        answer_text=structured_outcome.answer_text,
        answer=structured_outcome.answer,
        confidence=confidence,
        logprob=logprob,
        alternatives=structured_outcome.alternatives
    )
