import requests
import json
import math
import sys
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

class LogprobsAnalyzer:
    """Logprobs analyzer for llama.cpp API"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
    
    def query_llm(
        self,
        prompt: str,
        n_probs: int = 2,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = -1,
        repeat_penalty: float = 1.2,
        repeat_last_n: int = 64,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """Query LLM and get logprobs with anti-repetition controls

        Args:
            prompt: The prompt to send to the LLM
            n_probs: Number of top probabilities to return
            stream: Whether to stream the response
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (-1 = unlimited)
            repeat_penalty: Penalty for repeating tokens (default: 1.2)
            repeat_last_n: Number of last tokens to consider for repeat penalty (default: 64)
            stop: List of stop sequences to end generation
        """
        url = f"{self.api_url}/completion"

        # Default stop sequences if not provided
        if stop is None:
            stop = [
                "\n\nThus answer:",
                "\nThus answer:",
                "Thus the answer:",
                "<|end|>",
                "<|start|>",
                "<|message|>"
            ]

        payload = {
            "prompt": prompt,
            "stream": stream,
            "n_probs": n_probs,
            "temperature": temperature,
            "n_predict": max_tokens,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "stop": stop
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: API request failed: {e}")
            sys.exit(1)
    
    @staticmethod
    def logprob_to_prob(logprob: float) -> float:
        """Convert logprob to probability"""
        return math.exp(logprob)
    
    def analyze_true_false(self, result: Dict) -> Dict:
        """Analyze True/False probabilities from result"""
        # Handle both full result and summary result
        if 'last_5_tokens' in result:
            tokens = result['last_5_tokens']
            answer = result.get('answer', '')
        else:
            completion_probs = result.get('completion_probabilities', [])
            tokens = completion_probs[-10:] if completion_probs else []
            answer = result.get('content', '')
        
        analysis = {
            'full_answer': answer,
            'true_false_found': False,
            'selected_answer': None,
            'confidence': None,
            'probability': None,
            'alternatives': [],
            'position': None
        }
        
        # Search for True or False token
        for i, token_data in enumerate(reversed(tokens)):
            token = token_data.get('token', '').strip()
            token_lower = token.lower()
            
            if 'true' in token_lower or 'false' in token_lower:
                analysis['true_false_found'] = True
                
                # Determine answer
                if 'true' in token_lower:
                    analysis['selected_answer'] = 'True'
                elif 'false' in token_lower:
                    analysis['selected_answer'] = 'False'
                
                # Calculate probability
                logprob = token_data.get('logprob', 0)
                prob = self.logprob_to_prob(logprob) * 100
                analysis['probability'] = round(prob, 2) / 100
                
                # Determine confidence
                if prob > 80:
                    analysis['confidence'] = 'high'
                elif prob > 50:
                    analysis['confidence'] = 'medium'
                else:
                    analysis['confidence'] = 'low'
                
                analysis['position'] = i + 1
                
                # Extract alternatives
                top_logprobs = token_data.get('top_logprobs', [])
                for alt in top_logprobs[:5]:
                    alt_token = alt.get('token', '')
                    alt_logprob = alt.get('logprob', 0)
                    alt_prob = self.logprob_to_prob(alt_logprob) * 100
                    
                    analysis['alternatives'].append({
                        'token': alt_token,
                        'probability': round(alt_prob, 2) / 100,
                        'logprob': round(alt_logprob, 4)
                    })
                
                break
        
        return analysis
    
    def print_analysis(self, analysis: Dict, detailed: bool = True):
        """Print analysis results"""
        print("=" * 70)
        print("True/False Probability Analysis")
        print("=" * 70)
        print()
        
        if not analysis['true_false_found']:
            print("No True/False token found")
            print("\nSuggestions:")
            print("  1. Check if prompt explicitly requests True/False")
            print("  2. Try: 'Return True or False in the end'")
            return
        
        answer = analysis['selected_answer']
        prob = analysis['probability']
        confidence = analysis['confidence']
        
        print(f"Selected answer: {answer}")
        print(f"Probability: {prob}")
        print(f"Confidence: {confidence}")
        
        if detailed:
            print(f"Position: {analysis['position']} tokens from end")
            print()
            print("Top 5 alternatives:")
            for i, alt in enumerate(analysis['alternatives'], 1):
                marker = "*" if i == 1 else " "
                token_display = alt['token'].replace('\n', '\\n').replace('\t', '\\t')
                print(f"{marker} {i}. '{token_display}': {alt['probability']}")
        
        print()
        print("=" * 70)
    
    def save_results(self, data: Dict, output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    
    def batch_query(
        self,
        questions: List[str],
        output_dir: str = "results",
        n_probs: int = 10,
        temperature: float = 0.7,
        max_tokens: int = -1,
        repeat_penalty: float = 1.2,
        repeat_last_n: int = 64
    ):
        """Batch query multiple questions with anti-repetition controls"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results_summary = []

        print(f"\nStarting batch query for {len(questions)} questions...\n")

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Query: {question[:60]}...")

            # Query with anti-repetition parameters
            result = self.query_llm(
                question,
                n_probs=n_probs,
                temperature=temperature,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n
            )
            
            # Analyze
            analysis = self.analyze_true_false(result)
            
            # Save full result
            output_file = os.path.join(output_dir, f"result_{i:03d}.json")
            self.save_results(result, output_file)
            
            # Save analysis
            analysis_file = os.path.join(output_dir, f"analysis_{i:03d}.json")
            self.save_results(analysis, analysis_file)
            
            # Brief display
            if analysis['true_false_found']:
                answer = analysis['selected_answer']
                prob = analysis['probability']
                print(f"   Result: {answer} ({prob:.2f})")
            else:
                print(f"   Warning: No True/False found")
            
            # Add to summary
            results_summary.append({
                'question': question,
                'answer': analysis.get('selected_answer', 'Unknown'),
                'probability': analysis.get('probability', 0),
                'confidence': analysis.get('confidence', 'N/A'),
                'result_file': output_file,
                'analysis_file': analysis_file
            })
            
            print()
        
        # Save summary
        summary_file = os.path.join(output_dir, "summary.json")
        self.save_results(results_summary, summary_file)
        
        print(f"\nBatch query completed: {len(questions)} questions")
        print(f"Results saved in: {output_dir}/")
        
        # Statistics
        true_count = sum(1 for r in results_summary if r['answer'] == 'True')
        false_count = sum(1 for r in results_summary if r['answer'] == 'False')
        unknown_count = len(results_summary) - true_count - false_count
        
        print(f"\nStatistics:")
        print(f"  True:    {true_count}")
        print(f"  False:   {false_count}")
        print(f"  Unknown: {unknown_count}")

def main():
    parser = argparse.ArgumentParser(
        description='Llama.cpp Logprobs Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query single question
  python3 analyze_logprobs.py --prompt "Does TP53 inhibit BCL2L1?"

  # Analyze existing result
  python3 analyze_logprobs.py --analyze result.json

  # Batch queries from file
  python3 analyze_logprobs.py --batch questions.txt --output-dir results/

  # Batch with custom anti-repetition settings
  python3 analyze_logprobs.py --batch questions.txt --n-probs 5 --output-dir ./ --repeat-penalty 1.5 --temperature 0.3

  # Single query with strict anti-repetition
  python3 analyze_logprobs.py --prompt "Does TP53 activate MDM2?" --repeat-penalty 1.5 --max-tokens 100
        """
    )
    
    parser.add_argument('--prompt', type=str, help='Question prompt')
    parser.add_argument('--analyze', type=str, help='Analyze existing JSON file')
    parser.add_argument('--batch', type=str, help='Batch query from text file (one question per line)')
    parser.add_argument('--output', type=str, default='result.json', help='Output file (default: result.json)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for batch (default: results)')
    parser.add_argument('--n-probs', type=int, default=10, help='Number of top probabilities (default: 10)')
    parser.add_argument('--api-url', type=str, default='http://localhost:8080', help='API URL (default: http://localhost:8080)')
    parser.add_argument('--simple', action='store_true', help='Simple output without alternatives')

    # Anti-repetition parameters
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=-1, help='Maximum tokens to generate, -1 = unlimited (default: -1)')
    parser.add_argument('--repeat-penalty', type=float, default=1.2, help='Penalty for repeating tokens (default: 1.2)')
    parser.add_argument('--repeat-last-n', type=int, default=64, help='Number of last tokens to consider for repeat penalty (default: 64)')
    
    args = parser.parse_args()
    
    analyzer = LogprobsAnalyzer(api_url=args.api_url)
    
    # Mode 1: Analyze existing file
    if args.analyze:
        print(f"Analyzing: {args.analyze}\n")
        with open(args.analyze, 'r') as f:
            result = json.load(f)
        
        analysis = analyzer.analyze_true_false(result)
        analyzer.print_analysis(analysis, detailed=not args.simple)
        
        # Save analysis
        analysis_file = args.analyze.replace('.json', '_analyzed.json')
        analyzer.save_results(analysis, analysis_file)
    
    # Mode 2: Batch query
    elif args.batch:
        with open(args.batch, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]

        analyzer.batch_query(
            questions,
            output_dir=args.output_dir,
            n_probs=args.n_probs,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repeat_penalty=args.repeat_penalty,
            repeat_last_n=args.repeat_last_n
        )
    
    # Mode 3: Single query
    elif args.prompt:
        print(f"Query: {args.prompt}\n")

        result = analyzer.query_llm(
            args.prompt,
            n_probs=args.n_probs,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repeat_penalty=args.repeat_penalty,
            repeat_last_n=args.repeat_last_n
        )
        
        # Save full result
        analyzer.save_results(result, args.output)
        
        # Analyze and print
        analysis = analyzer.analyze_true_false(result)
        print()
        analyzer.print_analysis(analysis, detailed=not args.simple)
        
        # Save analysis
        analysis_file = args.output.replace('.json', '_analyzed.json')
        analyzer.save_results(analysis, analysis_file)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()