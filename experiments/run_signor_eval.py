#!/usr/bin/env python3
"""
End-to-end evaluation script for the SIGNOR dataset using the RLM framework.

This script iterates through the claim-level SIGNOR dataset, where each row
already contains a concrete claim and label, and sequentially runs
`evidence_programming.py` for multiple repetitions per claim. Results are
appended incrementally to an output CSV. Token usage and cost estimates are
parsed from the verdict artifacts.

Example command:
    time uv run python -m proclaim.verification.evidence_programming_direct --config experiments/configs/test_config.yaml \
        --claim "SRC directly inhibits CTTN." \
        --output-dir results/test_direct_1

    time uv run python -m proclaim.verification.evidence_programming_direct --config experiments/configs/test_config.yaml \
        --claim "GNAS directly activates ADCY1 (either through post-translational modification, complex formation, or direct regulation of expression)." \
        --output-dir results/test_direct_1

Claim-level modified variant (directness constraint baked in, for testing without ICL):
    time uv run python -m proclaim.verification.evidence_programming_direct --config experiments/configs/test_config.yaml \
        --claim "CRTC2 directly activates AKT1 (direct physical interaction, not through intermediate proteins; either through post-translational modification, complex formation, or direct regulation of expression)." \
        --output-dir results/test_direct_1_no_intermediary
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
DEFAULT_INPUT_CSV = PROJECT_ROOT / "datasets" / "signor.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "signor_eval_results.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "signor_eval"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "signor_eval_config.yaml"
REQUIRED_INPUT_COLUMNS = {
    "id",
    "flip",
    "claim",
    "label",
    "original_label",
    "entity_a",
    "entity_b",
}

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
# Input validation
# ---------------------------------------------------------------------------

def validate_input_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "SIGNOR input CSV must use the claim-level dataset schema. "
            f"Missing columns: {', '.join(missing)}"
        )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)

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


def get_mode_from_config(config_path: Path) -> str:
    """Extracts the orchestration mode from the given YAML config."""
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("mode", "sdk")
    except Exception as e:
        logger.warning(f"Failed to load mode from {config_path}: {e}")
    return "sdk"

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

def _parse_tokens_from_run_log(output_dir: Path) -> Dict[str, int]:
    """Parse SDK-mode token usage from run.log (INFO: Usage: {...} lines)."""
    in_tok = out_tok = cache_creation_tok = cache_read_tok = 0
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
                    except Exception as e:
                        logger.debug(f"Failed to parse usage line: {e}")
        except Exception as e:
            logger.warning(f"Could not read run.log: {e}")
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_creation_tokens": cache_creation_tok,
        "cache_read_tokens": cache_read_tok,
    }


def _parse_tokens_from_usage_json(output_dir: Path) -> Dict[str, int]:
    """Parse direct-mode token usage from token_usage.json (LiteLLM format)."""
    usage_path = output_dir / "token_usage.json"
    if usage_path.exists():
        try:
            data = json.loads(usage_path.read_text())
            return {
                "input_tokens": data.get("input_tokens", data.get("prompt_tokens", 0)),
                "output_tokens": data.get("output_tokens", data.get("completion_tokens", 0)),
                "cache_creation_tokens": data.get("cache_creation_tokens", 0),
                "cache_read_tokens": data.get("cache_read_tokens", 0),
            }
        except Exception as e:
            logger.warning(f"Could not read token_usage.json: {e}")
    return {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0}


def _compute_cost(tokens: Dict[str, int], in_price: float, out_price: float) -> float:
    in_tok = tokens["input_tokens"]
    out_tok = tokens["output_tokens"]
    cache_creation_tok = tokens["cache_creation_tokens"]
    cache_read_tok = tokens["cache_read_tokens"]
    cost = (
        (in_tok / 1_000_000) * in_price
        + (cache_creation_tok / 1_000_000) * in_price * 1.25
        + (cache_read_tok / 1_000_000) * in_price * 0.1
        + (out_tok / 1_000_000) * out_price
    )
    return round(cost, 4)


def run_evaluation(
    claim: str,
    output_dir: Path,
    config_path: Path,
    in_price: float = DEFAULT_PRICING[0],
    out_price: float = DEFAULT_PRICING[1],
    env_vars: Dict[str, str] = None,
    mode: str = "sdk",
) -> Dict[str, Any]:
    """Runs evidence_programming (sdk) or evidence_programming_direct (direct)
    via subprocess to evaluate the given claim."""

    verdict_path = output_dir / "workspace" / "verdict.json"

    if verdict_path.exists():
        logger.info(f"Verdict already exists, skipping run: {output_dir}")
        stats = parse_verdict_file(verdict_path)
        if mode == "direct":
            tokens = _parse_tokens_from_usage_json(output_dir)
        else:
            tokens = _parse_tokens_from_run_log(output_dir)
        stats.update(tokens)
        stats["total_input_tokens"] = (
            tokens["input_tokens"] + tokens["cache_creation_tokens"] + tokens["cache_read_tokens"]
        )
        stats["cost_estimate"] = _compute_cost(tokens, in_price, out_price)
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "direct":
        cmd = [
            "uv", "run", "python", "-m", "proclaim.verification.evidence_programming_direct",
            "--config", str(config_path),
            "--claim", claim,
            "--output-dir", str(output_dir),
        ]
    else:
        notebook_path = output_dir / "evidence_report.ipynb"
        cmd = [
            "uv", "run", "python", "-m", "proclaim.verification.evidence_programming",
            "--config", str(config_path),
            "--claim", claim,
            "--output-dir", str(output_dir),
            "--notebook-path", str(notebook_path),
        ]

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    logger.info(f"Running [{mode}] verifying claim: '{claim}' -> {output_dir.name}")
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
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "total_input_tokens": 0,
            "cost_estimate": 0.0,
        }

    logger.info(f"Run finished in {time.time() - start_time:.1f}s")

    if mode == "direct":
        tokens = _parse_tokens_from_usage_json(output_dir)
    else:
        tokens = _parse_tokens_from_run_log(output_dir)

    stats = parse_verdict_file(verdict_path)
    stats.update(tokens)
    stats["total_input_tokens"] = (
        tokens["input_tokens"] + tokens["cache_creation_tokens"] + tokens["cache_read_tokens"]
    )
    stats["cost_estimate"] = _compute_cost(tokens, in_price, out_price)
    return stats


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation of SIGNOR Dataset")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV), help="Path to the claim-level SIGNOR dataset CSV.")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag for namespaced output directory (e.g. 20260330_142500).")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to write results (derived from --run-tag if omitted).")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Base VerificationSettings YAML.")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per claim variation.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of input CSV rows to process (0=all).")
    parser.add_argument("--row-start", type=int, default=0, help="Start row index for parallel chunking (default: 0).")

    args = parser.parse_args()
    
    input_path = args.input_csv

    config_path = Path(args.config)

    # Determine mode, model, and pricing
    mode = get_mode_from_config(config_path)

    # Resolve output paths (namespaced by run-tag when provided)
    if args.run_tag:
        results_dir = PROJECT_ROOT / "results" / f"signor_{mode}_eval_{args.run_tag}"
        default_csv = results_dir / "results.csv"
    else:
        results_dir = DEFAULT_RESULTS_DIR
        default_csv = DEFAULT_OUTPUT_CSV

    output_path = Path(args.output_csv) if args.output_csv else default_csv
    logger.info(f"Orchestration mode: {mode}")
    model_name = get_model_from_config(config_path)
    # Strip LiteLLM provider prefix for pricing lookup (e.g. "anthropic/claude-sonnet-4-20250514")
    model_key = model_name.split("/")[-1] if "/" in model_name else model_name
    in_price, out_price = CLAUDE_PRICING.get(model_key, DEFAULT_PRICING)
    logger.info(f"Using pricing for model '{model_name}': ${in_price:.2f}/1M input, ${out_price:.2f}/1M output")

    # Prepare environment variables to pass to subprocesses
    # This ensures LLM_BASE_URL from run_signor_batch.sh is propagated to notebook kernels
    env_vars = {}
    if "LLM_BASE_URL" in os.environ:
        env_vars["LLM_BASE_URL"] = os.environ["LLM_BASE_URL"]
        logger.info(f"Propagating LLM_BASE_URL={os.environ['LLM_BASE_URL']}")

    logger.info(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    validate_input_schema(df)
    
    if args.limit > 0:
        df = df.iloc[args.row_start:args.row_start + args.limit]
    elif args.row_start > 0:
        df = df.iloc[args.row_start:]
    
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
        sid = row["id"]
        entity_a = row["entity_a"]
        entity_b = row["entity_b"]
        orig_label = row["original_label"]
        claim_str = row["claim"]
        flip = _to_bool(row["flip"])
        expected_label = row["label"]

        logger.info(f"--- Processing [{idx+1}/{total_rows}] {sid} : {entity_a} -> {entity_b} ---")

        for rep in range(1, args.reps + 1):
            if (sid, flip, rep) in existing_runs:
                continue

            run_dir_name = f"{sid}/flip_{flip}/rep_{rep}"
            run_dir = results_dir / run_dir_name

            logger.info(f"Running Repetition {rep}/{args.reps} (Flipped: {flip})")
            stats = run_evaluation(claim_str, run_dir, config_path, in_price, out_price, env_vars=env_vars, mode=mode)

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
