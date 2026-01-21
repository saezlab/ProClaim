import re
import itertools
from operator import itemgetter
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

class PaperRater:
    """
    Evaluator for comparing two paper full-texts to determine which provides more sufficient context.
    """
    
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

    def __init__(self, client: OpenAI, model: str = "gpt-oss-120b", temperature: float = 0):
        self.client = client
        self.model = model
        self.temperature = temperature

    def extract_winner(self, response_text: str) -> str:
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

    def compare_papers(self, paper_a: Dict, paper_b: Dict, question: str) -> Dict:
        """
        Compare two papers to determine which provides better support.
        
        Args:
            paper_a: Dictionary with 'content' (full text) and 'id'
            paper_b: Dictionary with 'content' (full text) and 'id'
            question: The question to evaluate against
            
        Returns:
            Dictionary with evaluation result (winner, reason, etc.)
        """
        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            full_text_A=paper_a.get('content', ''),
            full_text_B=paper_b.get('content', '')
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=4096
            )
            
            response_text = response.choices[0].message.content
            winner = self.extract_winner(response_text)
            
            return {
                "winner": winner,
                "reasoning": response_text
            }
            
        except Exception as e:
            return {
                "winner": "Error",
                "reasoning": str(e)
            }

    def rank_papers(self, papers: List[Dict], question: str) -> Tuple[List[Dict], Dict]:
        """
        Rank a list of papers by performing round-robin comparisons.
        
        Args:
            papers: List of paper dicts. Each must have 'content' and 'id'.
            question: The question to evaluate.
            
        Returns:
            Tuple of (sorted_papers, debug_details)
        """
        if len(papers) < 2:
            return papers, {}

        scores = {p['id']: 0 for p in papers}
        comparisons_log = []
        
        # All pairwise combinations
        combinations = list(itertools.combinations(papers, 2))
        
        for p1, p2 in combinations:
            result = self.compare_papers(p1, p2, question)
            winner = result['winner']
            
            comparisons_log.append({
                "paper_a": p1['id'],
                "paper_b": p2['id'],
                "winner": winner,
                "reasoning": result.get('reasoning')
            })
            
            if winner == "Paper A":
                scores[p1['id']] += 1
            elif winner == "Paper B":
                scores[p2['id']] += 1
            # If Unknown/Error, no points awarded
            
        # Sort papers by score descending
        # Add score to paper dict (temporarily or permanently? let's not mutate input if possible, but sorting needs key)
        # We'll create a lookup for sorting
        
        sorted_papers = sorted(papers, key=lambda p: scores.get(p['id'], 0), reverse=True)
        
        debug_info = {
            "scores": scores,
            "comparisons": comparisons_log
        }
        
        return sorted_papers, debug_info
