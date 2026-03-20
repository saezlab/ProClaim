#!/usr/bin/env python3
"""
End-to-end evaluation script for the SIGNOR dataset using the RLM framework.

This script iterates through the SIGNOR ground truth CSV, generates both
forward and flipped claims, and sequentially runs `evidence_programming.py`
for multiple repetitions per claim. Results are appended incrementally to an
output CSV. Token usage and cost estimates are parsed from the verdict artifacts.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = "/hps/nobackup/saezrodriguez/shared_datasets/signor*/ground_truth.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "signor_eval_results.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "example_config.yaml"

# Cost estimates (as per Claude Sonnet 4 pricing, arbitrary placeholder)
# Update these if exact pricing tracking is critical.
COST_PER_1M_INPUT_TOKENS = 3.00
COST_PER_1M_OUTPUT_TOKENS = 15.00

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("signor_eval")

# ---------------------------------------------------------------------------
# Claim Generation Logic
# ---------------------------------------------------------------------------

def construct_signor_claim(source: str, target: str, interaction: str, flip: bool = False) -> Tuple[str, str]:
    """
    Constructs the natural language claim from the entities and interaction.
    Also returns the expected flipped label mapping.
    """
    # Group definitions based on signor_utils.py
    is_positive = interaction in [
        'up-regulates', 
        'up-regulates activity', 
        'up-regulates quantity', 
        'up-regulates quantity by expression'
    ]
    is_negative = interaction in [
        'down-regulates', 
        'down-regulates activity', 
        'down-regulates quantity by destabilization'
    ]
    
    # Apply flip logic to the interaction type
    if flip:
        if is_positive:
            is_positive = False
            is_negative = True
        elif is_negative:
            is_negative = False
            is_positive = True
            
    # Formulate claim sentence
    if is_positive:
        claim_str = f"{source} directly activates {target} (either through activation or increase of expression)."
    elif is_negative:
        claim_str = f"{source} directly inhibits {target} (either through inhibition or destabilization)."
    else:
        # Complex, unknown, or other interactions
        if flip:
            claim_str = f"{source} does not physically interact with {target}."
        else:
            claim_str = f"{source} physically interacts with {target}."
            
    return claim_str

def get_flipped_label(original_label: str, flip: bool) -> str:
    """ Maps the ground truth label depending on flip status. """
    if not flip:
        return original_label
        
    mapped_label = original_label.upper()
    if mapped_label == "SUPPORTED":
        return "WRONG"  # REFUTED
    elif mapped_label == "WRONG":
        return "SUPPORTED"
    elif mapped_label == "UNCERTAIN":
        return "UNCERTAIN"
    return mapped_label

# ---------------------------------------------------------------------------
# Execution and Result Parsing
# ---------------------------------------------------------------------------

def parse_verdict_file(verdict_path: Path) -> Dict[str, Any]:
    """ Safely parse a verdict.json file to extract required fields. """
    try:
        data = json.loads(verdict_path.read_text())
        return {
            "verdict": data.get("verdict", "ERROR"),
            "confidence": data.get("confidence", 0.0),
            "reasoning": data.get("reasoning", "")
        }
    except Exception as e:
        logger.error(f"Failed to parse verdict at {verdict_path}: {e}")
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "reasoning": str(e)
        }

def run_evaluation(
    claim: str, 
    output_dir: Path, 
    config_path: Path, 
    env_vars: Dict[str, str] = None
) -> Dict[str, Any]:
    """ Runs evidence_programming.py via subprocess to evaluate the given claim. """
    
    verdict_path = output_dir / "workspace" / "verdict.json"
    
    if verdict_path.exists():
        logger.info(f"Verdict already exists, skipping run: {output_dir}")
        stats = parse_verdict_file(verdict_path)
        # Placeholder for token usage since we don't currently save it in verdict.json
        # A more advanced parser would search output_dir/workspace/trace.json or run.log
        stats["input_tokens"] = 0
        stats["output_tokens"] = 0
        stats["cost_estimate"] = 0.0
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "evidence_report.ipynb"
    
    # Build subprocess command
    cmd = [
        "uv", "run", "python", "-m", "pkevolve.verification.evidence_programming",
        "--config", str(config_path),
        "--claim", claim,
        "--output-dir", str(output_dir),
        "--notebook-path", str(notebook_path)
    ]
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
        
    logger.info(f"Running LLM Agent verifying claim: '{claim}' -> {output_dir.name}")
    start_time = time.time()
    
    process = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if process.returncode != 0:
        logger.error(f"Run failed for {output_dir.name} with exit code {process.returncode}")
        logger.error(process.stderr[-1000:])
        return {
            "verdict": "FAIL",
            "confidence": 0.0,
            "reasoning": "Subprocess crashed or timed out.",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_estimate": 0.0
        }
        
    logger.info(f"Run finished in {time.time() - start_time:.1f}s")
    
    # Try parsing token usage from stderr logger output
    in_tok = 0
    out_tok = 0
    for line in process.stderr.splitlines():
        if "Usage: " in line and "input_tokens=" in line:
            # Example text: Usage: Usage(input_tokens=1500, output_tokens=300)
            try:
                parts = line.split("input_tokens=")[1]
                in_tok_str = parts.split(",")[0]
                out_tok_str = parts.split("output_tokens=")[1].split(")")[0]
                in_tok += int(in_tok_str)
                out_tok += int(out_tok_str)
            except Exception:
                pass
                
    cost = (in_tok / 1_000_000) * COST_PER_1M_INPUT_TOKENS + (out_tok / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS
    
    stats = parse_verdict_file(verdict_path)
    stats["input_tokens"] = in_tok
    stats["output_tokens"] = out_tok
    stats["cost_estimate"] = round(cost, 4)
    
    return stats


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation of SIGNOR Dataset")
    parser.add_argument("--input-csv", type=str, help="Path to SIGNOR ground_truth.csv")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Path to write results.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Base VerificationSettings YAML.")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per claim variation.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of input CSV rows to process (0=all).")
    
    args = parser.parse_args()
    
    input_path = args.input_csv
    if not input_path:
        import glob
        matches = glob.glob(DEFAULT_INPUT_CSV)
        if matches:
            input_path = matches[0]
        else:
            raise FileNotFoundError(f"Could not find default target: {DEFAULT_INPUT_CSV}")
            
    output_path = Path(args.output_csv)
    config_path = Path(args.config)
    
    logger.info(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    if args.limit > 0:
        df = df.head(args.limit)
    
    # Define columns for the output CSV
    out_cols = [
        "SIGNOR_ID", "ENTITYA", "ENTITYB", "Original_Label", "Flipped_Label", 
        "Claim_String", "Is_Flipped", "Repetition", "Agent_Verdict", 
        "Agent_Confidence", "Reasoning_Snippet", "Output_Directory",
        "Input_Tokens", "Output_Tokens", "Cost_Estimate"
    ]
    
    # Initialize output CSV if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_cols)
    else:
        logger.info(f"Resuming with existing output CSV: {output_path}")

    # Process each row
    total_rows = len(df)
    for idx, row in df.iterrows():
        sid = row.get("SIGNOR_ID", f"ROW_{idx}")
        entity_a = row.get("ENTITYA", "UnknownA")
        entity_b = row.get("ENTITYB", "UnknownB")
        effect = row.get("EFFECT", "unknown")
        orig_label = row.get("Label", "UNCERTAIN")
        
        logger.info(f"--- Processing [{idx+1}/{total_rows}] {sid} : {entity_a} -> {entity_b} ---")
        
        for flip in [False, True]:
             # Create string formats and labels
             claim_str = construct_signor_claim(entity_a, entity_b, effect, flip=flip)
             expected_label = get_flipped_label(orig_label, flip=flip)
             
             for rep in range(1, args.reps + 1):
                 run_dir_name = f"{sid}/flip_{flip}/rep_{rep}"
                 run_dir = PROJECT_ROOT / "results" / "signor_eval" / run_dir_name
                 
                 logger.info(f"Running Repetition {rep}/{args.reps} (Flipped: {flip})")
                 stats = run_evaluation(claim_str, run_dir, config_path)
                 
                 # Append straight to CSV safely
                 row_dict = {
                     "SIGNOR_ID": sid,
                     "ENTITYA": entity_a,
                     "ENTITYB": entity_b,
                     "Original_Label": orig_label,
                     "Flipped_Label": expected_label,
                     "Claim_String": claim_str,
                     "Is_Flipped": flip,
                     "Repetition": rep,
                     "Agent_Verdict": stats.get("verdict"),
                     "Agent_Confidence": stats.get("confidence"),
                     "Reasoning_Snippet": str(stats.get("reasoning"))[:500],
                     "Output_Directory": str(run_dir.relative_to(PROJECT_ROOT)),
                     "Input_Tokens": stats.get("input_tokens", 0),
                     "Output_Tokens": stats.get("output_tokens", 0),
                     "Cost_Estimate": stats.get("cost_estimate", 0.0)
                 }
                 
                 with open(output_path, "a", newline="") as f:
                     writer = csv.DictWriter(f, fieldnames=out_cols)
                     writer.writerow(row_dict)

if __name__ == "__main__":
    main()
