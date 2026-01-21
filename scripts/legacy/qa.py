'''
uv run python qa.py --mode nosearch --model gpt-oss-120b
'''


from pathlib import Path
import json
from openai import OpenAI
from tqdm import tqdm
from pkevolve.llm.evaluator import GeneInteractionEvaluator, StructuredOutcome
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

DATA_BASE_DIR = PROJECT_ROOT / "data/papers/signor"
RESULTS_BASE_DIR = PROJECT_ROOT / "results/qa"

from src.pkevolve.utils.signor_utils import construct_signor_question

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

def process_file(json_file: Path, results_dir: Path, client: OpenAI, evaluator: GeneInteractionEvaluator):
    """Process a single JSON file and save results for both scenarios."""
    # Parse filename
    stem = json_file.stem
    parts = stem.split('_')

    if len(parts) < 3:
        return False, f"Invalid filename format"

    source = parts[0]
    target = parts[1]
    interaction_raw = parts[2]

    # Map interaction
    interaction = INTERACTION_MAPPING.get(interaction_raw, interaction_raw)

    try:
        # Load paper data for context
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)

        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        full_text = paper_data.get('full_text', '')

        # Scenario 1: Title + Abstract only
        search_context_abstract = f"Title: {title}\n\nAbstract: {abstract}"

        result_abstract = evaluator.evaluate(
            source_gene=source,
            target_gene=target,
            relationship=interaction,
            search_context=search_context_abstract,
            save_raw_path=None
        )

        # Scenario 2: Title + Abstract + Full Text
        search_context_full = f"Title: {title}\n\nAbstract: {abstract}\n\nFull Text: {full_text}"

        result_full = evaluator.evaluate(
            source_gene=source,
            target_gene=target,
            relationship=interaction,
            search_context=search_context_full,
            save_raw_path=None
        )

        # Prepare output with both scenarios
        output_data = {
            "original_file": json_file.name,
            "question": construct_signor_question(source, target, interaction),
            "scenario_title_abstract": {
                "search_context_type": "title + abstract",
                "search_context_length": len(search_context_abstract),
                "prediction": result_abstract.answer,
                "reasoning": result_abstract.reasoning,
                "reasoning_length": len(result_abstract.reasoning),
                "full_output": result_abstract.model_dump()
            },
            "scenario_title_abstract_fulltext": {
                "search_context_type": "title + abstract + full_text",
                "search_context_length": len(search_context_full),
                "prediction": result_full.answer,
                "reasoning": result_full.reasoning,
                "reasoning_length": len(result_full.reasoning),
                "full_output": result_full.model_dump()
            }
        }

        # Save result
        output_path = results_dir / json_file.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return True, f"Success (Abstract: {result_abstract.answer}, Full: {result_full.answer})"

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        traceback.print_exc()
        return False, error_msg


import argparse

def main():
    parser = argparse.ArgumentParser(description="Run QA on gene interactions.")
    parser.add_argument("--mode", choices=['nosearch', 'signor_ref'], default='nosearch', help="Execution mode")
    parser.add_argument("--model", default="gpt-oss-120b", help="Model name")
    args = parser.parse_args()

    # Initialize Client and Evaluator once for all files
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    evaluator = GeneInteractionEvaluator(client=client, model=args.model)

    # Process both true_negative_edges and true_positive_edges
    edge_types = ['true_negative_edges', 'true_positive_edges']

    for edge_type in edge_types:
        print(f"\n{'='*80}")
        print(f"Processing {edge_type} | Mode: {args.mode} | Model: {args.model}")
        print(f"{'='*80}")

        # Setup directories
        data_dir = DATA_BASE_DIR / edge_type
        
        if args.mode == 'nosearch':
            results_dir = RESULTS_BASE_DIR / "nosearch" / args.model / edge_type
        else:
            results_dir = RESULTS_BASE_DIR / "signor_ref" / args.model / edge_type

        # Ensure results directory exists
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get all JSON files
        json_files = list(data_dir.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {data_dir}")
            continue

        print(f"Found {len(json_files)} JSON files to process\n")

        # Process each file with progress bar
        success_count = 0
        error_count = 0

        for json_file in tqdm(json_files, desc=f"Processing {edge_type}"):
            success, message = process_file_with_mode(json_file, results_dir, client, evaluator, args.mode)
            if success:
                success_count += 1
            else:
                error_count += 1
                tqdm.write(f"✗ {json_file.name}: {message}")

        # Summary
        print(f"\n{'-'*80}")
        print(f"Summary for {edge_type}:")
        print(f"  Total files: {len(json_files)}")
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")
        print(f"  Results saved to: {results_dir}")


def process_file_with_mode(json_file: Path, results_dir: Path, client: OpenAI, evaluator: GeneInteractionEvaluator, mode: str):
    """Process a single JSON file and save results based on mode."""
    # Parse filename
    stem = json_file.stem
    parts = stem.split('_')

    if len(parts) < 3:
        return False, f"Invalid filename format"

    source = parts[0]
    target = parts[1]
    interaction_raw = parts[2]

    # Map interaction
    interaction = INTERACTION_MAPPING.get(interaction_raw, interaction_raw)

    try:
        # Load paper data for context
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)

        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        full_text = paper_data.get('full_text', '')

        output_data = {
            "original_file": json_file.name,
            "question": construct_signor_question(source, target, interaction),
        }

        if mode == 'nosearch':
             # Scenario: Without Search
            result_nosearch = evaluator.evaluate(
                source_gene=source,
                target_gene=target,
                relationship=interaction,
                search_context="",
                save_raw_path=None
            )
            output_data["scenario_without_search"] = {
                "search_context_type": "without_search",
                "search_context_length": 0,
                "prediction": result_nosearch.answer,
                "reasoning": result_nosearch.reasoning,
                "reasoning_length": len(result_nosearch.reasoning),
                "full_output": result_nosearch.model_dump()
            }
            log_msg = f"Success (NoSearch: {result_nosearch.answer})"

        else: # signor_ref
            # Scenario 1: Title + Abstract only
            search_context_abstract = f"Title: {title}\n\nAbstract: {abstract}"
            result_abstract = evaluator.evaluate(
                source_gene=source,
                target_gene=target,
                relationship=interaction,
                search_context=search_context_abstract,
                save_raw_path=None
            )

            # Scenario 2: Title + Abstract + Full Text
            search_context_full = f"Title: {title}\n\nAbstract: {abstract}\n\nFull Text: {full_text}"
            result_full = evaluator.evaluate(
                source_gene=source,
                target_gene=target,
                relationship=interaction,
                search_context=search_context_full,
                save_raw_path=None
            )
            
            output_data["scenario_title_abstract"] = {
                "search_context_type": "title + abstract",
                "search_context_length": len(search_context_abstract),
                "prediction": result_abstract.answer,
                "reasoning": result_abstract.reasoning,
                "reasoning_length": len(result_abstract.reasoning),
                "full_output": result_abstract.model_dump()
            }
            output_data["scenario_title_abstract_fulltext"] = {
                "search_context_type": "title + abstract + full_text",
                "search_context_length": len(search_context_full),
                "prediction": result_full.answer,
                "reasoning": result_full.reasoning,
                "reasoning_length": len(result_full.reasoning),
                "full_output": result_full.model_dump()
            }
            log_msg = f"Success (Abstract: {result_abstract.answer}, Full: {result_full.answer})"

        # Save result
        output_path = results_dir / json_file.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return True, log_msg

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        traceback.print_exc()
        return False, error_msg

if __name__ == "__main__":
    main()
