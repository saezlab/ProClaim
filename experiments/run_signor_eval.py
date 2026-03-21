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
import yaml

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = "/hps/nobackup/saezrodriguez/shared_datasets/signor*/ground_truth.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "signor_eval_results.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "example_config.yaml"

# Token pricing per 1M tokens (input, output) in USD
CLAUDE_PRICING = {
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
# Fallback to Sonnet pricing if model is not recognized
DEFAULT_PRICING = (3.00, 15.00)

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

def get_model_from_config(config_path: Path) -> str:
    """Extracts the LLM model name from the given YAML config."""
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("llm", {}).get("model", "unknown-model")
    except Exception as e:
        logger.warning(f"Failed to load model from {config_path}: {e}")
    return "unknown-model"

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
    in_price: float = DEFAULT_PRICING[0],
    out_price: float = DEFAULT_PRICING[1],
    env_vars: Dict[str, str] = None
) -> Dict[str, Any]:
    """ Runs evidence_programming.py via subprocess to evaluate the given claim. """
    
    verdict_path = output_dir / "workspace" / "verdict.json"
    
    if verdict_path.exists():
        logger.info(f"Verdict already exists, skipping run: {output_dir}")
        stats = parse_verdict_file(verdict_path)

        # Try to parse token usage from existing run.log
        in_tok = 0
        out_tok = 0
        cache_creation_tok = 0
        cache_read_tok = 0

        run_log_path = output_dir / "run.log"
        if run_log_path.exists():
            try:
                log_content = run_log_path.read_text()
                for line in log_content.splitlines():
                    if "INFO: Usage:" in line and "input_tokens" in line:
                        try:
                            usage_str = line.split("INFO: Usage: ")[1].replace("'", '"')
                            usage_dict = json.loads(usage_str)
                            in_tok += usage_dict.get("input_tokens", 0)
                            out_tok += usage_dict.get("output_tokens", 0)
                            cache_creation_tok += usage_dict.get("cache_creation_input_tokens", 0)
                            cache_read_tok += usage_dict.get("cache_read_input_tokens", 0)
                        except Exception:
                            pass
            except Exception:
                pass

        # Calculate cost with correct Anthropic prompt caching pricing
        cost_input = (in_tok / 1_000_000) * in_price
        cost_cache_write = (cache_creation_tok / 1_000_000) * in_price * 1.25
        cost_cache_read = (cache_read_tok / 1_000_000) * in_price * 0.1
        cost_output = (out_tok / 1_000_000) * out_price
        cost = cost_input + cost_cache_write + cost_cache_read + cost_output

        stats["input_tokens"] = in_tok
        stats["output_tokens"] = out_tok
        stats["cache_creation_tokens"] = cache_creation_tok
        stats["cache_read_tokens"] = cache_read_tok
        stats["total_input_tokens"] = in_tok + cache_creation_tok + cache_read_tok
        stats["cost_estimate"] = round(cost, 4)
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

    # Parse token usage from run.log (not stderr, since logging goes to file)
    in_tok = 0
    out_tok = 0
    cache_creation_tok = 0
    cache_read_tok = 0

    run_log_path = output_dir / "run.log"
    if run_log_path.exists():
        try:
            log_content = run_log_path.read_text()
            for line in log_content.splitlines():
                if "INFO: Usage:" in line and "input_tokens" in line:
                    # Example: INFO: Usage: {'input_tokens': 38, 'cache_creation_input_tokens': 23359, ...}
                    try:
                        # Extract the dict portion after "Usage: "
                        usage_str = line.split("INFO: Usage: ")[1]
                        # Convert single quotes to double quotes for JSON parsing
                        usage_str = usage_str.replace("'", '"')
                        usage_dict = json.loads(usage_str)

                        in_tok += usage_dict.get("input_tokens", 0)
                        out_tok += usage_dict.get("output_tokens", 0)
                        cache_creation_tok += usage_dict.get("cache_creation_input_tokens", 0)
                        cache_read_tok += usage_dict.get("cache_read_input_tokens", 0)
                    except Exception as e:
                        logger.debug(f"Failed to parse usage line: {e}")
                        pass
        except Exception as e:
            logger.warning(f"Could not read run.log: {e}")

    # Calculate cost with Anthropic prompt caching pricing
    # Regular input tokens = normal input price
    # Cache writes (creation) = 125% of input price (25% premium to write to cache)
    # Cache reads = 10% of input price (90% discount)
    cost_input = (in_tok / 1_000_000) * in_price
    cost_cache_write = (cache_creation_tok / 1_000_000) * in_price * 1.25
    cost_cache_read = (cache_read_tok / 1_000_000) * in_price * 0.1
    cost_output = (out_tok / 1_000_000) * out_price

    cost = cost_input + cost_cache_write + cost_cache_read + cost_output
    
    stats = parse_verdict_file(verdict_path)
    stats["input_tokens"] = in_tok
    stats["output_tokens"] = out_tok
    stats["cache_creation_tokens"] = cache_creation_tok
    stats["cache_read_tokens"] = cache_read_tok
    stats["total_input_tokens"] = in_tok + cache_creation_tok + cache_read_tok
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
    
    # Determine model and pricing
    model_name = get_model_from_config(config_path)
    in_price, out_price = CLAUDE_PRICING.get(model_name, DEFAULT_PRICING)
    logger.info(f"Using pricing for model '{model_name}': ${in_price:.2f}/1M input, ${out_price:.2f}/1M output")

    # Prepare environment variables to pass to subprocesses
    # This ensures LLM_BASE_URL from run_signor_batch.sh is propagated to notebook kernels
    env_vars = {}
    if "LLM_BASE_URL" in os.environ:
        env_vars["LLM_BASE_URL"] = os.environ["LLM_BASE_URL"]
        logger.info(f"Propagating LLM_BASE_URL={os.environ['LLM_BASE_URL']}")

    logger.info(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    if args.limit > 0:
        df = df.head(args.limit)
    
    # Define columns for the output CSV
    out_cols = [
        "SIGNOR_ID", "ENTITYA", "ENTITYB", "Original_Label", "Flipped_Label",
        "Claim_String", "Is_Flipped", "Repetition", "Agent_Verdict",
        "Agent_Confidence", "Reasoning_Snippet", "Output_Directory",
        "Input_Tokens", "Output_Tokens", "Cache_Creation_Tokens", "Cache_Read_Tokens",
        "Total_Input_Tokens", "Cost_Estimate"
    ]
    
    # Initialize output CSV if it doesn't exist
    existing_runs = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_cols)
    else:
        logger.info(f"Resuming with existing output CSV: {output_path}")
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sid_val = row.get("SIGNOR_ID")
                    flip_val = row.get("Is_Flipped") == "True"
                    rep_val = int(row.get("Repetition", 1))
                    existing_runs.add((sid_val, flip_val, rep_val))
                except Exception:
                    pass

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
                 if (sid, flip, rep) in existing_runs:
                     continue
                 
                 run_dir_name = f"{sid}/flip_{flip}/rep_{rep}"
                 run_dir = PROJECT_ROOT / "results" / "signor_eval" / run_dir_name
                 
                 logger.info(f"Running Repetition {rep}/{args.reps} (Flipped: {flip})")
                 stats = run_evaluation(claim_str, run_dir, config_path, in_price, out_price, env_vars=env_vars)
                 
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
                     "Cache_Creation_Tokens": stats.get("cache_creation_tokens", 0),
                     "Cache_Read_Tokens": stats.get("cache_read_tokens", 0),
                     "Total_Input_Tokens": stats.get("total_input_tokens", 0),
                     "Cost_Estimate": stats.get("cost_estimate", 0.0)
                 }
                 
                 with open(output_path, "a", newline="") as f:
                     writer = csv.DictWriter(f, fieldnames=out_cols)
                     writer.writerow(row_dict)

if __name__ == "__main__":
    main()
