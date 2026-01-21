# (time uv run qa_comprehensive.py --scenario full_text --ordering rater) 2>&1 | tee qa_execution.log

import sys
import argparse
from pathlib import Path
import json
import random
import tiktoken
from openai import OpenAI
from tqdm import tqdm
from pkevolve.llm.evaluator import GeneInteractionEvaluator, StructuredOutcome

# Adjust path to import from src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.pkevolve.utils.signor_utils import load_signor_data, construct_signor_question
from src.pkevolve.search.paper_utils import clean_text
from src.pkevolve.llm.rater import PaperRater

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SIGNOR_BASE = PROJECT_ROOT / "data/signor"  # CSV files location
DATA_SEARCH_DIR = PROJECT_ROOT / "data/papers/qa_search"
RESULTS_BASE_DIR = PROJECT_ROOT / "results/qa_comprehensive"


def get_supporting_files(entity_a: str, entity_b: str, effect: str, search_dir: Path, target_count: int = 5):
    """
    Find supporting files in search_dir that match the edge.
    Filename pattern: {entity_a}_{entity_b}_{effect}_*.json
    Prioritize files with full_text.
    """
    
    # Sanitize filename components to match how they were saved
    def sanitize(s):
        return "".join(c for c in s if c.isalnum() or c in ('_', '-')).strip()
    
    safe_a = sanitize(entity_a)
    safe_b = sanitize(entity_b)
    safe_effect = sanitize(effect)
    
    stem = f"{safe_a}_{safe_b}_{safe_effect}"
    pattern = f"{stem}_*.json"
    files = list(search_dir.glob(pattern))
    
    supported_papers = []
    
    # First pass: look for full text
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fr:
                data = json.load(fr)
                if data.get('full_text'):
                    data['filename'] = f.name
                    supported_papers.append(data)
        except Exception:
            continue
            
    # Return top N full text papers
    # Based on requirement: "skip files if the full-text if none, until it finds the 5 full-text file"
    if len(supported_papers) > 0:
        return supported_papers[:target_count]
        
    return []

def construct_context(papers, include_full_text: bool):
    context_parts = []
    # Estimated char limit per paper to fit 5 papers in ~60k tokens. 
    # 60k tokens ~ 240k chars. / 5 = 48k chars.
    # To be safe, use 30k chars per paper.
    # MAX_CHARS_PER_PAPER = 30000 
    
    for i, p in enumerate(papers):
        # Clean up title/abstract if None
        title = p.get('title') or "N/A"
        abstract = p.get('abstract') or "N/A"
        
        part = f"Paper {i+1}:\nTitle: {title}\nAbstract: {abstract}"
        if include_full_text:
            ft = p.get('full_text') or "N/A"
            ft = clean_text(ft)
            part += f"\nFull Text: {ft}"
        context_parts.append(part)
    return "\n\n".join(context_parts)

def process_edge(row, search_dir: Path, results_dir: Path, client: OpenAI, evaluator: GeneInteractionEvaluator, scenario: str, shuffle: bool = False, use_rater: bool = False):
    """Process a single edge row and save results for the selected scenario."""
    
    entity_a = str(row['ENTITYA'])
    entity_b = str(row['ENTITYB'])
    effect = str(row['EFFECT'])
    
    # Construct filename for saving matches the pattern
    def sanitize(s):
        return "".join(c for c in s if c.isalnum() or c in ('_', '-')).strip()
        
    safe_a = sanitize(entity_a)
    safe_b = sanitize(entity_b)
    safe_effect = sanitize(effect)
    output_filename = f"{safe_a}_{safe_b}_{safe_effect}.json"

    try:
        # Get supporting papers
        supporting_papers = get_supporting_files(entity_a, entity_b, effect, search_dir, target_count=5)
        
        # Determine question for rater and prompt
        question = construct_signor_question(entity_a, entity_b, effect)

        rater_debug_info = {}
        if use_rater and len(supporting_papers) > 1:
            print("  > Running Rater to rank papers...")
            # Prepare papers for Rater (needs 'id' and 'content')
            rater_papers = []
            for p in supporting_papers:
                # Construct text representation for rater
                content_parts = []
                if p.get('title'): content_parts.append(f"Title: {p['title']}")
                if p.get('abstract'): content_parts.append(f"Abstract: {p['abstract']}")
                if p.get('full_text'): content_parts.append(f"Full Text: {p['full_text']}")
                
                rater_papers.append({
                    'id': p.get('filename'),
                    'content': "\n\n".join(content_parts),
                    'original_data': p 
                })
            
            # Initialize Rater
            rater = PaperRater(client=client, model="gpt-oss-120b")
            sorted_rater_papers, debug_info = rater.rank_papers(rater_papers, question)
            
            # Restore sorted order to supporting_papers
            supporting_papers = [p['original_data'] for p in sorted_rater_papers]
            rater_debug_info = debug_info
            
        elif shuffle:
            random.shuffle(supporting_papers)
        
        if len(supporting_papers) == 0:
             return False, "No supporting full-text papers found."

        # Construct question
        question = construct_signor_question(entity_a, entity_b, effect)

        # Prepare context based on scenario
        include_full_text = (scenario == 'full_text')
        search_context = construct_context(supporting_papers, include_full_text=include_full_text)
        
        # Evaluate
        # Token estimation
        prompt = evaluator.construct_prompt(entity_a, entity_b, effect, search_context)
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(prompt))
        print(f"  > Token usage estimate: {token_count}")

        result = evaluator.evaluate(
            source_gene=entity_a,
            target_gene=entity_b,
            relationship=effect,
            search_context=search_context,
            save_raw_path=None
        )

        # Prepare output
        output_data = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "effect": effect,
            "question": question,
            "supporting_papers_count": len(supporting_papers),
            "scenario": scenario,
            "scenario": scenario,
            "shuffle": shuffle,
            "use_rater": use_rater,
            "paper_order": [p.get('filename') for p in supporting_papers],
            "result": {
                "search_context_type": "title + abstract + full_text" if include_full_text else "title + abstract",
                "token_usage": result.usage['prompt_tokens'] if result.usage else token_count,
                 # Keep estimate as fallback or for comparison if needed
                "token_usage_estimate": token_count, 
                "token_usage_real": result.usage,
                "prediction": result.answer,
                "reasoning": result.reasoning,
                "full_output": result.model_dump(),
                "rater_debug": rater_debug_info
            }
        }

        # Save result
        output_path = results_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return True, f"Success (Papers: {len(supporting_papers)}, Prediction: {result.answer})"

    except Exception as e:
        import traceback
        error_msg = str(e)
        context_limit_hit = "exceeds the maximum allowed length" in error_msg
        
        # Attempt to save partial result
        try:
            output_data = {
                "entity_a": entity_a,
                "entity_b": entity_b,
                "effect": effect,
                "question": locals().get('question', "N/A"),
                "supporting_papers_count": len(locals().get('supporting_papers', [])),
                "scenario": scenario,
                "result": {
                    "search_context_type": "title + abstract + full_text" if scenario == 'full_text' else "title + abstract",
                    "prediction": "ERROR",
                    "reasoning": f"Error during evaluation: {error_msg}",
                    "context_limit_exceeded": context_limit_hit,
                    "token_usage_estimate": locals().get('token_count', -1)
                }
            }
            
            output_path = results_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            return False, f"Error (Saved): {error_msg}"
        except Exception as save_error:
             return False, f"Error: {error_msg} | Failed to save error JSON: {str(save_error)}"


def main():
    parser = argparse.ArgumentParser(description="Run QA evaluation on Signor edges using searched papers.")
    parser.add_argument('--scenario', type=str, choices=['title_abstract', 'full_text'], required=True,
                        help="Scenario to run: 'title_abstract' or 'full_text'")
    parser.add_argument('--ordering', type=str, choices=['shuffle', 'rater'], default='shuffle',
                        help="Method to order papers: 'shuffle' (random) or 'rater' (LLM ranked). Default: shuffle.")
    parser.add_argument('--model', type=str, default='gpt-oss-120b',
                        help="Model name to use for evaluation. Default: gpt-oss-120b")
    args = parser.parse_args()
    
    # Initialize Client and Evaluator once
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    evaluator = GeneInteractionEvaluator(client=client, model=args.model)

    # Define edge files and their corresponding search/result folders
    configs = [
        {
            "name": "true_positive",
            "csv_file": "true_positive_edges.csv",
            "search_folder": "true_positive",
            "result_subfolder": "true_positive"
        },
        {
            "name": "true_negative",
            "csv_file": "true_negative_edges.csv",
            "search_folder": "true_negative",
            "result_subfolder": "true_negative"
        }
    ]

    for config in configs:
        csv_path = DATA_SIGNOR_BASE / config["csv_file"]
        search_dir = DATA_SEARCH_DIR / config["search_folder"]
        
        # Output directory structure: results/qa_comprehensive/{model}/{scenario}/{ordering}/{group}/
        
        results_dir = RESULTS_BASE_DIR / args.model / args.scenario
        if args.ordering == 'rater':
            results_dir = results_dir / "rater"
        elif args.ordering == 'shuffle':
            results_dir = results_dir / "shuffled"
        
        results_dir = results_dir / config["result_subfolder"]
        
        print(f"\n{'='*80}")
        print(f"Processing {config['name']} from {csv_path}")
        print(f"Scenario: {args.scenario}")
        print(f"Ordering: {args.ordering}")
        print(f"{'='*80}")

        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}")
            continue
            
        # Ensure results directory exists
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df = load_signor_data(str(csv_path), columns=['ENTITYA', 'ENTITYB', 'EFFECT'])
        
        if df is None or df.empty:
            print("DataFrame is empty or failed to load.")
            continue

        print(f"Found {len(df)} edges to process")
        
        # Test Limit: 1 edge per group
        # print("TEST MODE: Limiting to 1 edge per group.")
        # df = df.head(1)

        success_count = 0
        error_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {config['name']}"):
            success, message = process_edge(
                row, 
                search_dir, 
                results_dir, 
                client, 
                evaluator, 
                args.scenario, 
                shuffle=(args.ordering == 'shuffle'), 
                use_rater=(args.ordering == 'rater')
            )
            if success:
                success_count += 1
            else:
                error_count += 1
                tqdm.write(f"✗ {row['ENTITYA']} -> {row['ENTITYB']}: {message}")

        # Summary
        print(f"\n{'-'*80}")
        print(f"Summary for {config['name']} ({args.scenario}):")
        print(f"  Total edges: {len(df)}")
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")
        print(f"  Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
