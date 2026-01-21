#!/usr/bin/env python3
"""
Sufficient Context Analysis Script for comparing two paper full-texts.

This script will analyze two paper full-texts and determine which one provides
more sufficient context for the claimed biological interactions.
"""

import json
import re
from pathlib import Path
from itertools import combinations
from langchain_openai import ChatOpenAI

# Prompt Template
PROMPT_TEMPLATE = """You are an expert LLM evaluator that excels at evaluating a YES/NO QUESTION against two distinct sources of information: PAPER A and PAPER B.
Consider the following criteria:
Best Support: Identify which paper provides stronger, more direct, or more empirically supported evidence to answer the Yes/No question regarding the interaction between two entities.
* Select "Paper A" if Paper A contains the answer while Paper B does not, or if Paper A provides stronger evidence or reasoning than Paper B.
* Select "Paper B" if Paper B contains the answer while Paper A does not, or if Paper B provides stronger evidence or reasoning than Paper A.
First, output a list of step-by-step questions that would be used to arrive at a decision.
* Make sure to include questions that check if the specific Subject (Entity A) and Object (Entity B) mentioned in the QUESTION are present in the text of PAPER A.
* Make sure to include questions that check if the specific interaction/action (the verb) mentioned in the QUESTION is explicitly tested or observed in PAPER A.
* Repeat the above checks for PAPER B.
* Include questions about any statistical significance, quantitative measurements (e.g., fold change, p-values), or arithmetic required to interpret the results.
Next, answer each of the questions. Make sure to work step by step to verify the claims in the papers.
Finally, use these answers to evaluate the criteria. Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON).
EXAMPLE:
### QUESTION
Does treatment with Compound X inhibit the expression of the MYC gene?
### PAPER A
We performed a screening of 50 compounds to identify potential inhibitors of cell growth. Compound X was included in the screen. We observed that cells treated with Compound X showed a 40% reduction in proliferation. The discussion section hypothesizes that this might be due to interference with the MYC pathway, but no direct measurement of MYC gene expression levels (mRNA or protein) was performed.
### PAPER B
To determine the mechanism of action, we treated HeLa cells with 10 µM Compound X for 24 hours. qPCR analysis revealed that MYC mRNA levels decreased by 3.5-fold compared to control (p < 0.01). Western blot analysis confirmed a corresponding decrease in MYC protein levels.
### EXPLANATION
To answer the question "Does Compound X inhibit MYC expression?", we need direct evidence of a change in MYC levels.
1. Does Paper A answer it? Paper A measures cell proliferation (growth), not MYC expression. It only hypothesizes a link to the MYC pathway. It lacks direct evidence of the interaction.
2. Does Paper B answer it? Yes. Paper B explicitly performs qPCR and Western blots targeting MYC.
3. Quantitative Check: Paper B reports a 3.5-fold decrease with statistical significance (p < 0.01), which confirms inhibition.
4. Comparison: Paper B provides direct, quantitative evidence of the specific interaction asked in the question. Paper A only infers it loosely.
Therefore, Paper B is the better source.
### JSON
{{"Better Paper": "Paper B"}}
Remember the instructions:
You are an expert LLM evaluator.
First, output a list of step-by-step questions.
Next, answer each of the questions.
Finally, Output the ### EXPLANATION (Text) and ### EVALUATION (JSON) where "Better Paper" must be either "Paper A" or "Paper B".
### QUESTION
<question>
{question}
### PAPER A
{full_text_A}
### PAPER B
{full_text_B}
"""

def extract_winner(response_text: str) -> str:
    """Extract 'Paper A' or 'Paper B' from the response."""
    # Look for JSON first
    json_match = re.search(r'\{[^}]*"Better Paper"\s*:\s*"?(Paper [AB])"?[^}]*\}', response_text, re.IGNORECASE)
    if json_match:
        return json_match.group(1)
    
    # Fallback to text search
    if '"Better Paper": "Paper A"' in response_text or "'Better Paper': 'Paper A'" in response_text:
        return "Paper A"
    if '"Better Paper": "Paper B"' in response_text or "'Better Paper': 'Paper B'" in response_text:
        return "Paper B"
        
    return "Unknown"

def get_paper_content(filepath: Path) -> dict:
    """Reads the JSON file and constructs a text representation of the paper."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    content = []
    if data.get('title'):
        content.append(f"Title: {data['title']}")
    if data.get('abstract'):
        content.append(f"Abstract: {data['abstract']}")
    if data.get('full_text'):
        content.append(f"Full Text: {data['full_text']}")
        
    return {
        'id': filepath.name,
        'content': "\n\n".join(content),
        'question': data.get('question', '')
    }

def main():
    # Initialize LLM
    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="gpt-oss-120b",
        temperature=0
    )

    # Define data directory relative to project root or absolute
    # Assuming script is in workspace/scripts
    # DATA_DIR = Path(__file__).resolve().parent.parent / "data/papers/qa_search"
    DATA_DIR = Path("/nfs/research/saezrodriguez/rain/workspace/grn-llm-correct/data/papers/qa_search")
    
    if not DATA_DIR.exists():
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    # Select files 0 to 4
    file_indices = range(5)
    files = [
        DATA_DIR / f"GNAO1_ADCY1_down-regulate_{i}.json"
        for i in file_indices
    ]
    
    papers = []
    print("Loading papers...")
    for f in files:
        if not f.exists():
            print(f"Warning: {f} does not exist.")
            continue
        try:
            papers.append(get_paper_content(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
        
    if len(papers) < 2:
        print("Not enough papers to compare.")
        return

    # Scores
    scores = {p['id']: 0 for p in papers}
    
    # Generate all pairwise combinations (10 pairs for 5 papers)
    comparisons = list(combinations(range(len(papers)), 2))
    print(f"Starting {len(comparisons)} comparisons for {len(papers)} papers...")
    
    # Load existing results if available
    results_file = Path("rater_results.json")
    results = {}
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} cached comparisons from {results_file.resolve()}.")
        except Exception as e:
            print(f"Error loading results file: {e}")

    for idx_a, idx_b in comparisons:
        paper_a = papers[idx_a]
        paper_b = papers[idx_b]
        
        # specific key for this pair - ensure consistent ordering via sorting IDs
        # We sort by ID string to ensure A vs B is same as B vs A key
        p1_id, p2_id = sorted([paper_a['id'], paper_b['id']])
        key = f"{p1_id}_vs_{p2_id}"
        
        winner = None

        if key in results:
            print(f"Skipping {key} (cached)")
            cached_entry = results[key]
            if "winner" in cached_entry:
               winner = cached_entry["winner"]
            else:
               # Handle old format or incomplete entry if any
               pass
        
        if winner is None:
            # Determine question
            question = paper_a['question']
            if not question:
                 question = paper_b['question']
            if not question:
                 # Fallback question if missing in both
                 question = "Does GNAO1 down-regulate, inhibit, or destabilize ADCY1?"

            print(f"\nComparing {paper_a['id']} (A) vs {paper_b['id']} (B)...")
            
            prompt = PROMPT_TEMPLATE.format(
                question=question,
                full_text_A=paper_a['content'],
                full_text_B=paper_b['content']
            )
            
            try:
                response = llm.invoke(prompt)
                winner = extract_winner(response.content)
                
                print(f"Winner: {winner}")
                
                # Store result
                results[key] = {
                    "winner": winner,
                    "paper_a": paper_a['id'],
                    "paper_b": paper_b['id'],
                    "reasoning": response.content
                }
                
                # Save immediately
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                print(f"Error during comparison: {e}")
                continue

        # Update scores based on winner
        # winner is "Paper A" or "Paper B" relative to the prompt context.
        # In the prompt, PAPER A is paper_a, PAPER B is paper_b.
        if winner == "Paper A":
            scores[paper_a['id']] += 1
        elif winner == "Paper B":
            scores[paper_b['id']] += 1
        else:
            print(f"No clear winner for {key} (Winner string: {winner})")

    print("\nFinal Rankings:")
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for pid, score in sorted_scores:
        print(f"{pid}: {score}")

if __name__ == "__main__":
    main()