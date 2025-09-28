import ollama
import json
import time
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict

def load_config():
    """Load configuration with defaults"""
    return {
        'model': 'gpt-oss:20b',
        'max_retries': 3,
        'enable_colors': True
    }

CONFIG = load_config()

class SimpleBiologyAssistant:
    def __init__(self, verbose: bool = False):
        """
        Initialize SimpleBiologyAssistant
        
        Args:
            verbose: Whether to print status messages
        """
        self.verbose = verbose
        
        if self.verbose:
            print(f"SimpleBiologyAssistant initialized - Model: {CONFIG['model']}")

    def model_response(self, model: str, message: str, max_retries: int = None) -> Optional[str]:
        """
        Get response from Ollama model with error handling
        
        Args:
            model: Ollama model name
            message: Prompt/message to send to model
            max_retries: Maximum retry attempts
            
        Returns:
            Model response text or None if failed
        """
        if max_retries is None:
            max_retries = CONFIG.get('max_retries', 3)
            
        for attempt in range(max_retries):
            try:
                response = ollama.chat(model=model, messages=[
                {
                    'role': 'system',
                    'content': 'You are a molecular biologist expert in biological interactions.',
                },
                {
                    'role': 'user',
                    'content': message,
                }])
                return response['message']['content']
                
            except Exception as e:
                if self.verbose:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Exponential backoff for retries
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    if self.verbose:
                        print(f"Failed to get model response after {max_retries} attempts")
                    return None

    def generate_reasoning_and_binary_answer(self, question: str) -> Tuple[Optional[str], Optional[bool]]:
        """
        Generate comprehensive reasoning and extract binary answer using only the model's knowledge
        
        Args:
            question: Original user question about biological interaction
            
        Returns:
            Tuple of (reasoning, binary_answer) where:
            - reasoning: Full reasoning text or None if failed
            - binary_answer: True/False or None if couldn't extract
        """
        prompt = f"""You are a molecular biology expert analyzing biological interactions based on your scientific knowledge.

Question: {question}

Instructions:
1. Provide a comprehensive analysis of the biological interaction described in the question
2. Use your knowledge of molecular mechanisms, pathways, and experimental evidence
3. Consider any conflicting evidence or uncertainty you're aware of
4. Be thorough in your scientific reasoning
5. Base your answer on established scientific literature and knowledge

After your detailed analysis, provide your final assessment in this exact format:
CONCLUSION: [True/False]

Where True means there is strong scientific evidence supporting the interaction, and False means there is insufficient evidence or evidence against the interaction."""

        if self.verbose:
            print("Generating reasoning and binary answer...")
            
        response = self.model_response(CONFIG['model'], prompt)
        
        if not response:
            return None, None
        
        # Extract binary answer from the response
        binary_answer = self._extract_binary_conclusion(response)
        
        return response, binary_answer
    
    def _extract_binary_conclusion(self, text: str) -> Optional[bool]:
        """
        Extract binary conclusion from model response
        
        Args:
            text: Full model response text
            
        Returns:
            True/False or None if couldn't extract
        """
        if not text:
            return None
            
        # Look for CONCLUSION: True/False with optional markdown formatting
        conclusion_pattern = r'CONCLUSION:\s*\*?\*?(True|False)\*?\*?'
        match = re.search(conclusion_pattern, text, re.IGNORECASE)
        
        if match:
            conclusion_text = match.group(1).lower()
            return conclusion_text == 'true'
            
        # Fallback: look for final True/False at end of text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1].lower()
            if last_line == 'true':
                return True
            elif last_line == 'false':
                return False
        
        # Additional fallback: look for "answer: true/false" pattern
        answer_pattern = r'answer:\s*(true|false)'
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower() == 'true'
        
        return None

    def analyze_question(self, question: str) -> Dict:
        """
        Complete analysis workflow - from question to reasoning and binary answer
        
        Args:
            question: User's question to analyze
            
        Returns:
            Dictionary containing:
            - success: bool - Whether the analysis was successful
            - question: str - Original question
            - reasoning: str - Full reasoning text (if successful)
            - binary_answer: bool - True/False conclusion (if successful)
            - source_title: str - Always "Ollama Model Knowledge"
            - source_url: str - Always "N/A"
            - error: str - Error message (if failed)
            - timestamp: str - When the analysis was performed
        """
        result = {
            'success': False,
            'question': question,
            'reasoning': None,
            'binary_answer': None,
            'source_title': "Ollama Model Knowledge",
            'source_url': "N/A",
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if self.verbose:
                print(f"Starting analysis for: {question}")
            
            # Generate reasoning and binary answer using model knowledge only
            reasoning, binary_answer = self.generate_reasoning_and_binary_answer(question)
            if not reasoning:
                result['error'] = "Could not generate reasoning"
                return result
                
            result['reasoning'] = reasoning
            result['binary_answer'] = binary_answer
            result['success'] = True
            
            if self.verbose:
                print("Analysis completed successfully")
                if binary_answer is not None:
                    print(f"Binary conclusion: {binary_answer}")
                else:
                    print("Warning: Could not extract binary conclusion")
                
            return result
            
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            if self.verbose:
                print(f"Analysis failed: {str(e)}")
            return result


if __name__ == "__main__":
    # Initialize assistant
    assistant = SimpleBiologyAssistant(verbose=False)

    # Load signor negative edges csv file
    data_path = Path('../all_removed_edges_with_sources.csv')
    data = pd.read_csv(data_path)
    sources = data['ENTITYA'].tolist()
    targets = data['ENTITYB'].tolist()
    interactions = data['EFFECT'].tolist()

    repeat = 50
    for r in tqdm(range(repeat)):
        results = []
        for i in range(len(sources)):
            source = sources[i]
            target = targets[i]
            interaction = interactions[i]

            if interaction == 'down-regulates':
                question = f"Does {source} down-regulate {target}?"
            elif interaction == 'down-regulates activity':
                question = f"Does {source} inhibit the activity of {target}?"
            elif interaction == 'form complex':
                question = f"Does {source} form a complex with {target}?"
            elif interaction == 'up-regulates':
                question = f"Does {source} up-regulate {target}?"
            elif interaction == 'up-regulates activity':
                question = f"Does {source} activate {target}?"
            elif interaction == 'up-regulates quantity':
                question = f"Does {source} increase {target} expression?"
            elif interaction == 'up-regulates quantity by expression':
                question = f"Does {source} increase {target} expression?"
            else:  # handles 'unknown' and any other unexpected interactions
                question = f"Does {source} interact with {target}?"

            result = assistant.analyze_question(question)
            results.append(result)
            
            # Optional: Print results (uncomment if needed)
            # if result['success']:
            #     print(f"✅ Success: {result['question']}")
            #     print(f"🔬 Binary Answer: {result['binary_answer']}")
            #     print(f"💡 Reasoning Preview: {result['reasoning'][:150]}...")
            # else:
            #     print(f"❌ Failed: {result['error']}")
            # print("-" * 80)
            
        # Save all results to file after each iteration
        with open(f'./results/without_search/batch_results_{r}.json', 'w') as f:
            json.dump(results, f, indent=2)