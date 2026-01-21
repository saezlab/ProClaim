#!/usr/bin/env python3
"""
Unified QA Execution Script

Modes:
1. nosearch: Evaluate using LLM parametric knowledge only.
2. specify_paper: Evaluate using specific paper(s) provided in input JSONs (Title/Abstract/FullText).
3. search: Search for papers based on edge info, rank/shuffle them, and evaluate.

Usage:
  uv run python scripts/run_qa.py --mode nosearch --model gpt-oss-120b
  uv run python scripts/run_qa.py --mode specify_paper --input_dir data/papers/signor/true_positive_edges
  uv run python scripts/run_qa.py --mode search --scenario full_text --ordering rater
"""

import sys
import argparse
from pathlib import Path
import json
import random
import tiktoken
from openai import OpenAI
from tqdm import tqdm
import pandas as pd

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Adjust path to import from src
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from pkevolve.llm.evaluator import GeneInteractionEvaluator
from pkevolve.utils.signor_utils import load_signor_data, construct_signor_question
from pkevolve.search.paper_utils import clean_text
from pkevolve.llm.rater import PaperRater

# Constants
DATA_SIGNOR_BASE = PROJECT_ROOT / "data/signor"
DATA_SEARCH_DIR = PROJECT_ROOT / "data/papers/qa_search"
RESULTS_BASE_DIR = PROJECT_ROOT / "results/qa_unified"

# Interaction Mapping (for specific_paper/legacy)
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

def setup_client_and_evaluator(model_name: str):
    if model_name == "glm-4.6":
        api_key = os.environ.get("GLM_API_KEY")
        if not api_key:
            raise ValueError("GLM_API_KEY not found in environment variables. Please check your .env file.")
        base_url = "https://open.bigmodel.cn/api/paas/v4/"
        print(f"Using GLM-4.6 with official API")
    else:
        # Default to local vLLM or compatible server
        api_key = "EMPTY"
        base_url = "http://localhost:8000/v1"
        print(f"Using {model_name} with local/compatible server at {base_url}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    evaluator = GeneInteractionEvaluator(client=client, model=model_name)
    return client, evaluator

def parse_filename_interaction(filename: str):
    """Parse SOURCE_TARGET_INTERACTION from filename."""
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None, None
    source = parts[0]
    target = parts[1]
    interaction_raw = parts[2]
    interaction = INTERACTION_MAPPING.get(interaction_raw, interaction_raw)
    return source, target, interaction

# ==============================================================================
# Mode: No Search & Specify Paper (Legacy qa.py)
# ==============================================================================

# ==============================================================================
# Mode: No Search & Specify Paper (Legacy qa.py)
# ==============================================================================

def process_file_static(json_file: Path, results_dir: Path, evaluator: GeneInteractionEvaluator, mode: str, row_data: dict = None):
    """
    Process a single JSON file for nosearch or specify_paper modes.
    If row_data is provided, use it for explicit Entity/Interaction information.
    """
    if row_data is not None:
        source = str(row_data['ENTITYA'])
        target = str(row_data['ENTITYB'])
        interaction = str(row_data['EFFECT'])
    else:
        # Fallback to filename parsing
        source, target, interaction = parse_filename_interaction(json_file.name)
        if not source:
            return False, "Invalid filename format"

    try:
        # For nosearch mode, we might not strictly need the file content if source/target/interaction are known
        # But usually we check if the file exists as a validity check of the dataset.
        
        paper_data = {}
        if json_file and json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
        elif mode == 'specify_paper':
             return False, f"JSON file required for specify_paper but not found: {json_file}"

        question = construct_signor_question(source, target, interaction)
        output_data = {
            "original_file": json_file.name if json_file else "N/A",
            "question": question,
            "mode": mode,
            "source": source,
            "target": target,
            "interaction": interaction
        }

        if mode == 'nosearch':
            result = evaluator.evaluate(source, target, interaction, search_context="")
            output_data["scenario_without_search"] = {
                "search_context_type": "without_search",
                "prediction": result.answer,
                "reasoning": result.reasoning,
                "full_output": result.model_dump()
            }
            log_msg = f"Success (NoSearch: {result.answer})"

        elif mode == 'specify_paper':
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            full_text = paper_data.get('full_text', '')

            # Scenario 1: Abstract
            ctx_abs = f"Title: {title}\n\nAbstract: {abstract}"
            res_abs = evaluator.evaluate(source, target, interaction, search_context=ctx_abs)
            
            # Scenario 2: Full Text
            ctx_full = f"Title: {title}\n\nAbstract: {abstract}\n\nFull Text: {full_text}"
            res_full = evaluator.evaluate(source, target, interaction, search_context=ctx_full)

            output_data["scenario_title_abstract"] = {
                "search_context_type": "title + abstract",
                "prediction": res_abs.answer,
                "reasoning": res_abs.reasoning,
                "full_output": res_abs.model_dump()
            }
            output_data["scenario_title_abstract_fulltext"] = {
                "search_context_type": "title + abstract + full_text",
                "prediction": res_full.answer,
                "reasoning": res_full.reasoning,
                "full_output": res_full.model_dump()
            }
            log_msg = f"Success (Abstract: {res_abs.answer}, Full: {res_full.answer})"

        # Save
        filename = json_file.name if json_file else f"{source}_{target}_{interaction.replace(' ', '')}.json"
        output_path = results_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        return True, log_msg

    except Exception as e:
        return False, f"Error: {str(e)}"

def run_static_mode(args, evaluator):
    """Run nosearch or specify_paper logic."""
    
    # Check if we should use CSV iteration or Directory iteration
    # If args.input_dir is explicit, we might fall back to directory iteration unless a CSV is found there.
    # To keep it simple and powerful: try to find the standard CSVs matching 'true_positive' / 'true_negative' buckets.
    
    configs = [
        {"name": "true_positive_edges", "csv": "true_positive_edges.csv"},
        {"name": "true_negative_edges", "csv": "true_negative_edges.csv"}
    ]
    
    # Base directory where JSONs are stored (for specify_paper content or file verification)
    base_json_dir = PROJECT_ROOT / "data/papers/signor"

    for config in configs:
        edge_type = config['name']
        csv_file = config['csv']
        csv_path = DATA_SIGNOR_BASE / csv_file
        
        # Results Directory
        mode_dir_name = "nosearch" if args.mode == "nosearch" else "signor_ref"
        results_dir = RESULTS_BASE_DIR / mode_dir_name / args.model / edge_type
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {edge_type} | Mode: {args.mode}")

        # Strategy: CSV Driven
        if csv_path.exists() and not args.input_dir:
            print(f"Using CSV source: {csv_path}")
            df = load_signor_data(str(csv_path), columns=['ENTITYA', 'ENTITYB', 'EFFECT'])
            
            # The directory containing the JSONs for this edge type
            json_source_dir = base_json_dir / edge_type
            
            success_count = 0
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {edge_type}"):
                # Construct expected filename stem logic to find the file
                # Filename logic usually: SOURCE_TARGET_INTERACTION(sanitized).json
                # We need to replicate the sanitization or glob matching
                
                # Sanitize from row data
                def sanitize(s): return "".join(c for c in s if c.isalnum() or c in ('_', '-')).strip()
                # Legacy filenames used a specific mapping for interactions which removed spaces/dashes differently sometimes
                # But typically: up-regulates activity -> up-regulatesactivity or similar
                # Let's try to find the file by globbing: SOURCE_TARGET_*.json and matching interaction?
                # Or simpler: parse valid JSONs in the dir and match them to the ROW (reverse lookup)
                # But reverse lookup is slow O(N*M).
                
                # Let's try to construct the filename if possible, or Glob efficiently?
                # Actually, the legacy code relied on filenames.
                # Let's try generous globbing.
                
                s_safe = sanitize(str(row['ENTITYA']))
                t_safe = sanitize(str(row['ENTITYB']))
                # Interaction in filename usually has spaces removed or replaced
                
                # Find matching file in json_source_dir
                # Pattern: s_safe + "_" + t_safe + "_" + "*"
                candidates = list(json_source_dir.glob(f"{s_safe}_{t_safe}_*.json"))
                
                target_file = None
                if len(candidates) == 1:
                    target_file = candidates[0]
                elif len(candidates) > 1:
                    # Ambiguity? Pick first or try to match interaction more closely?
                    target_file = candidates[0] 
                
                if not target_file and args.mode == 'specify_paper':
                    # Skip if file not found in specify_paper mode
                    tqdm.write(f"Skipping {row['ENTITYA']}->{row['ENTITYB']}: No matching JSON file found.")
                    continue
                
                # If nosearch, we can proceed even without file if we want, but let's stick to file existence for consistency
                if not target_file:
                     # fallback or skip
                     # For nosearch, we can proceed without file! 
                     if args.mode == 'nosearch':
                        pass # target_file remains None
                     else:
                        continue
                        
                success, msg = process_file_static(target_file, results_dir, evaluator, args.mode, row_data=row)
                if success: success_count += 1
                else: tqdm.write(f"Failed: {msg}")
                
            print(f"Completed {edge_type}. Success: {success_count}/{len(df)}")

        else:
            # Fallback: Directory Iteration (Legacy behavior or Custom Input Dir)
            target_input_dir = Path(args.input_dir) if args.input_dir else (base_json_dir / edge_type)
            
            if not target_input_dir.exists():
                print(f"Skipping {target_input_dir}, does not exist.")
                continue

            json_files = list(target_input_dir.glob("*.json"))
            print(f"Found {len(json_files)} files in {target_input_dir} (Directory Iteration)")

            success_count = 0
            for json_file in tqdm(json_files, desc=f"Processing {edge_type}"):
                # No row_data, relying on filename parsing
                success, msg = process_file_static(json_file, results_dir, evaluator, args.mode, row_data=None)
                if success: success_count += 1
                else: tqdm.write(f"Failed {json_file.name}: {msg}")
                
            print(f"Completed {edge_type}. Success: {success_count}/{len(json_files)}")


# ==============================================================================
# Mode: Search (Legacy qa_comprehensive.py)
# ==============================================================================

def get_supporting_files(entity_a, entity_b, effect, search_dir, target_count=5):
    """Find supporting files matching the edge."""
    def sanitize(s): return "".join(c for c in s if c.isalnum() or c in ('_', '-')).strip()
    stem = f"{sanitize(str(entity_a))}_{sanitize(str(entity_b))}_{sanitize(str(effect))}"
    files = list(search_dir.glob(f"{stem}_*.json"))
    
    supported_papers = []
    for f in files:
        try:
            with open(f, 'r') as fr:
                data = json.load(fr)
                if data.get('full_text'):
                    data['filename'] = f.name
                    supported_papers.append(data)
        except: continue
        
    return supported_papers[:target_count]

def construct_context(papers, include_full_text):
    parts = []
    for i, p in enumerate(papers):
        title = p.get('title') or "N/A"
        abstract = p.get('abstract') or "N/A"
        part = f"Paper {i+1}:\nTitle: {title}\nAbstract: {abstract}"
        if include_full_text:
            ft = clean_text(p.get('full_text') or "N/A")
            part += f"\nFull Text: {ft}"
        parts.append(part)
    return "\n\n".join(parts)

def process_edge_search(row, search_dir, results_dir, client, evaluator, args):
    entity_a = str(row['ENTITYA'])
    entity_b = str(row['ENTITYB'])
    effect = str(row['EFFECT'])
    
    def sanitize(s): return "".join(c for c in s if c.isalnum() or c in ('_', '-')).strip()
    output_filename = f"{sanitize(entity_a)}_{sanitize(entity_b)}_{sanitize(effect)}.json"

    try:
        supporting_papers = get_supporting_files(entity_a, entity_b, effect, search_dir)
        rater_info = {}

        if args.ordering == 'rater' and len(supporting_papers) > 1:
            # Rater logic
            q_rater = construct_signor_question(entity_a, entity_b, effect)
            rater_papers = []
            for p in supporting_papers:
                content = f"Title: {p.get('title')}\nAbstract: {p.get('abstract')}\nFull Text: {p.get('full_text')}"
                rater_papers.append({'id': p.get('filename'), 'content': content, 'original': p})
            
            rater = PaperRater(client=client, model="gpt-oss-120b") # Analyzer model for ranking could be fixed or same
            sorted_papers, rater_info = rater.rank_papers(rater_papers, q_rater)
            supporting_papers = [p['original'] for p in sorted_papers]
            
        elif args.ordering == 'shuffle':
            random.shuffle(supporting_papers)
            
        if not supporting_papers:
            return False, "No full-text papers found"

        # Evaluate
        include_ft = (args.scenario == 'full_text')
        context = construct_context(supporting_papers, include_full_text=include_ft)
        
        result = evaluator.evaluate(entity_a, entity_b, effect, search_context=context)
        
        output_data = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "effect": effect,
            "question": construct_signor_question(entity_a, entity_b, effect),
            "supporting_papers_count": len(supporting_papers),
            "scenario": args.scenario,
            "ordering": args.ordering,
            "paper_order": [p.get('filename') for p in supporting_papers],
            "result": {
                "prediction": result.answer,
                "reasoning": result.reasoning,
                "full_output": result.model_dump(),
                "rater_debug": rater_info
            }
        }
        
        with open(results_dir / output_filename, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        return True, f"Success (Ans: {result.answer})"

    except Exception as e:
        return False, str(e)


def run_search_mode(args, client, evaluator):
    """Run search mode logic."""
    configs = [
        {"name": "true_positive", "csv": "true_positive_edges.csv", "search_sub": "true_positive"},
        {"name": "true_negative", "csv": "true_negative_edges.csv", "search_sub": "true_negative"}
    ]
    
    for config in configs:
        csv_path = DATA_SIGNOR_BASE / config['csv']
        search_dir = DATA_SEARCH_DIR / config['search_sub']
        
        # Path: results/qa_unified/search/{model}/{scenario}/{ordering}/{group}
        results_dir = RESULTS_BASE_DIR / "search" / args.model / args.scenario / args.ordering / config['search_sub']
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if not csv_path.exists():
            print(f"Skipping {config['name']}, CSV missing.")
            continue
            
        df = load_signor_data(str(csv_path), columns=['ENTITYA', 'ENTITYB', 'EFFECT'])
        print(f"\nProcessing {config['name']} ({len(df)} edges) | Scenario: {args.scenario} | Ordering: {args.ordering}")
        
        success_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            success, msg = process_edge_search(row, search_dir, results_dir, client, evaluator, args)
            if success: success_count += 1
            else: tqdm.write(f"Failed: {msg}")
            
        print(f"Completed {config['name']}. Success: {success_count}/{len(df)}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified QA Runner")
    parser.add_argument("--mode", required=True, choices=['nosearch', 'specify_paper', 'search'], help="Execution mode")
    parser.add_argument("--model", default="gpt-oss-120b", help="LLM model name")
    parser.add_argument("--input_dir", help="Input directory (for nosearch/specify_paper)")
    # Search mode specific
    parser.add_argument("--scenario", choices=['title_abstract', 'full_text'], help="Scenario for search mode")
    parser.add_argument("--ordering", choices=['shuffle', 'rater'], default='shuffle', help="Ordering for search mode")

    args = parser.parse_args()
    
    client, evaluator = setup_client_and_evaluator(args.model)
    
    if args.mode in ['nosearch', 'specify_paper']:
        run_static_mode(args, evaluator)
    elif args.mode == 'search':
        if not args.scenario:
            parser.error("--scenario is required for search mode")
        run_search_mode(args, client, evaluator)

if __name__ == "__main__":
    main()
