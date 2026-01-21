#!/usr/bin/env python3
"""
Sufficient Context Analysis Script

This script will analyze papers and determine if they provide sufficient context
for the claimed biological interactions.
uv run python sufficient_context.py --data_dir ../data/papers/qa_search
(time TQDM_DISABLE=1 uv run python sufficient_context.py --data_dir ../data/papers/qa_search) > execution.log 2>&1
"""

import json
import re
from pathlib import Path
from typing import Tuple, Optional
import os
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from pkevolve.utils.signor_utils import construct_signor_question

# Define the directory containing the JSON files (parent directory for both positive and negative)
# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# DATA_DIR = PROJECT_ROOT / "data/papers/signor"
# DATA_DIR = PROJECT_ROOT / "data/papers/qa_search"


def extract_sufficient_support_score(response_text: str) -> Optional[int]:
    """
    Extract the 'Sufficient Support' score (0 or 1) from the LLM response.

    Args:
        response_text: The full LLM response containing the evaluation

    Returns:
        The score (0 or 1), or None if not found
    """
    # Try to find JSON block with "Sufficient Support"
    # Pattern matches both {"Sufficient Support": 0} and variations
    json_pattern = r'\{[^}]*"Sufficient Support"\s*:\s*([01])[^}]*\}'
    match = re.search(json_pattern, response_text)

    if match:
        return int(match.group(1))

    # Fallback: try to find the score in plain text
    text_pattern = r'Sufficient Support["\s:]+([01])'
    match = re.search(text_pattern, response_text, re.IGNORECASE)

    if match:
        return int(match.group(1))

    return None



def create_evaluation_prompt(question: str, reference: str) -> str:
    """
    Create the evaluation prompt for the LLM.

    Args:
        question: The question to evaluate
        reference: The reference text to evaluate against

    Returns:
        The formatted prompt
    """
    return f"""
You are an expert LLM evaluator that excels at evaluating a YES/NO QUESTION against a single source of information: the REFERENCE.
Consider the following criteria:
Sufficient Support:
1 IF the REFERENCE provides direct evidence, explicit statements, or data that allows you to definitively answer EITHER "Yes" OR "No" to the question. This includes cases where the reference explicitly refutes the claim or supports a conflicting fact that proves the answer is "No".
0 IF the REFERENCE does not mention the specific entities, does not address the specific relationship, or is too vague to infer a definitive answer in either direction.
First, output a list of step-by-step questions that would be used to arrive at a label for the criteria.
Make sure to include questions that check if the specific Subject (Entity A) and Object (Entity B) mentioned in the QUESTION are present in the text of the REFERENCE.
Make sure to include questions that check if the specific interaction/action (the verb) mentioned in the QUESTION is explicitly tested or observed.
Include questions about any statistical significance, quantitative measurements (e.g., fold change, p-values), or arithmetic required to interpret the results.
Next, answer each of the questions. Make sure to work step by step to verify the claims in the reference.
Finally, use these answers to evaluate the criteria. Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON).
EXAMPLE:
### QUESTION
Does treatment with Compound X inhibit the expression of the MYC gene?
### REFERENCE
To determine the mechanism of action, we treated HeLa cells with 10 µM Compound X for 24 hours. qPCR analysis revealed that MYC mRNA levels decreased by 3.5-fold compared to control (p < 0.01). Western blot analysis confirmed a corresponding decrease in MYC protein levels.
### EXPLANATION
To determine if the Reference supports the answer, we must verify the presence of entities and the direction of the interaction.
Is the Subject (Compound X) present? Yes, the reference explicitly mentions treating cells with "Compound X".
Is the Object (MYC gene/protein) present? Yes, the reference measures "MYC mRNA" and "MYC protein".
Is the Interaction (Inhibition/Decrease) observed? Yes. The text states mRNA levels "decreased by 3.5-fold" and protein levels showed a "corresponding decrease."
Is there quantitative evidence? Yes, a 3.5-fold decrease with a p-value < 0.01 indicates a statistically significant inhibition.
Conclusion: The text provides direct, quantitative evidence affirming that Compound X inhibits MYC expression.
### JSON
{{"Sufficient Support": 1}}
Remember the instructions: You are an expert LLM evaluator. First, output a list of step-by-step questions. Next, answer each of the questions. Finally, Output the ### EXPLANATION (Text).
Then, use the EXPLANATION to output the ### EVALUATION (JSON).
### QUESTION
{question}
### REFERENCE
{reference}
"""

def evaluate_sufficient_support(llm, question: str, reference: str) -> Tuple[Optional[int], str]:
    """
    Evaluate whether a reference provides sufficient support for a question.

    Args:
        llm: The LLM instance
        question: The question to evaluate
        reference: The reference text

    Returns:
        Tuple containing:
        - Score (0 or 1), or None if extraction failed
        - Full response text (reasoning)
    """
    prompt = create_evaluation_prompt(question, reference)
    response = llm.invoke(prompt)
    return extract_sufficient_support_score(response.content), response.content

# Interaction mapping from filename format to SIGNOR format
INTERACTION_MAPPING = {
    'down-regulates': 'down-regulates',
    'down-regulatesactivity': 'down-regulates activity',
    'up-regulates': 'up-regulates',
    'up-regulatesactivity': 'up-regulates activity',
    'up-regulatesquantity': 'up-regulates quantity',
    'up-regulatesquantitybyexpression': 'up-regulates quantity by expression',
    'formcomplex': 'form complex',
    'unknown': 'unknown'
}

# Initialize LLM once
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="gpt-oss-120b",
    temperature=0
)

import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sufficient Context Analysis Script")
    parser.add_argument("--data_dir", type=str, default=None, 
                        help="Directory containing JSON files (default: relative to script)")
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        DATA_DIR = Path(args.data_dir).resolve()
    else:
        # Default fallback
        DATA_DIR = PROJECT_ROOT / "data/papers/signor"

    # Process each JSON file in the directory and subdirectories
    # Ensure directory exists before globbing
    if not DATA_DIR.exists():
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    # Use recursive glob for both true_positive_edges and true_negative_edges
    json_files = list(DATA_DIR.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    for json_file in tqdm(json_files, desc="Processing papers"):
        try:
            # Load data first to check for existing 'question'
            with open(json_file, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            question = paper_data.get('question')
            
            # If question is not in JSON, try to construct it from filename
            if not question:
                # Parse filename: source_target_interaction.json
                stem = json_file.stem
                parts = stem.split('_')
                
                # Heuristic: standard SIGNOR format usually has 3 parts
                # QA search might have index suffix (4 parts), but we prefer 'question' field.
                # If we are here, 'question' field is missing, so we must rely on filename.
                
                if len(parts) >= 3: 
                    # Try to parse source, target, interaction based on standard assumption
                    # CAUTION: If filename is complex (e.g. 4 parts), this might be ambiguous without 'question' field.
                    # For now, we keep backward compatibility for 3-part filenames.
                    if len(parts) == 3:
                        source = parts[0]
                        target = parts[1]
                        interaction_raw = parts[2]
                        interaction = INTERACTION_MAPPING.get(interaction_raw, interaction_raw)
                        question = construct_signor_question(source, target, interaction)
                    else:
                         tqdm.write(f"Skipping {json_file.name}: Missing 'question' field and filename has {len(parts)} parts (expected 3 for auto-generation).")
                         continue
                else:
                    tqdm.write(f"Skipping {json_file.name}: Filename format not recognized and 'question' field missing.")
                    continue
            
            # Use the question (either from JSON or constructed)
            # tqdm.write(f"Question: {question}")

            # Start with title + abstract
            # tqdm.write("  Evaluating title + abstract...")
            title = paper_data.get('title') or ''
            abstract = paper_data.get('abstract') or ''
            title_abstract = f"{title}\n\n{abstract}"
            score_abstract, reasoning_abstract = evaluate_sufficient_support(llm, question, title_abstract)
            paper_data['title_abstract_sufficient_support'] = score_abstract
            paper_data['title_abstract_reasoning'] = reasoning_abstract
            # tqdm.write(f"  Score: {score_abstract}")
            
            # Evaluate full text
            # tqdm.write("  Evaluating full text...")
            full_text = paper_data.get('full_text') or ''
            full_info = title_abstract + "\n\n" + full_text
            score_full, reasoning_full = evaluate_sufficient_support(llm, question, full_info)
            paper_data['full_text_sufficient_support'] = score_full
            paper_data['full_text_reasoning'] = reasoning_full
            # tqdm.write(f"  Score: {score_full}")
                
            # Save results only if successful
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
            # tqdm.write(f"  Saved updates to {json_file.name}")
            
        except Exception as e:
            tqdm.write(f"Error processing {json_file.name}: {e}")

if __name__ == "__main__":
    main()