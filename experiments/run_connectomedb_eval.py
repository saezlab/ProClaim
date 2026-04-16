#!/usr/bin/env python3
"""
End-to-end evaluation script for the ConnectomeDB dataset using the RLM framework.

This script iterates through the ConnectomeDB CSV, reads the pre-formed claim
string from each row, and sequentially runs `evidence_programming.py` for
multiple repetitions per claim. Results are appended incrementally to an
output CSV. Token usage and cost estimates are parsed from the verdict artifacts.

No flip variants are generated — each claim is run as-is.

To test a single claim (e.g., AFDN EPHA7) directly from the terminal, you can use:
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/configs/test_config.yaml \
    --claim "In the context of protein-protein interactions, AFDN as ligand directly interacts with EPHA7 as receptor." \
    --output-dir results/test_AFDN_EPHA7 \
    --notebook-path results/test_AFDN_EPHA7/evidence_report.ipynb
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/configs/test_config.yaml \
    --claim "In the context of protein-protein interactions, ITGB2 as ligand directly interacts with THY1 as receptor." \
    --output-dir results/test_ITGB2_THY1 \
    --notebook-path results/test_ITGB2_THY1/evidence_report.ipynb
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/configs/test_config.yaml \
    --claim "In the context of protein-protein interactions, LY86 as ligand directly interacts with CD180 as receptor." \
    --output-dir results/test_LY86_CD180 \
    --notebook-path results/test_LY86_CD180/evidence_report.ipynb
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/configs/test_config.yaml \
    --claim "In the context of protein-protein interactions, B2M as ligand directly interacts with CD1A as receptor." \
    --output-dir results/test_B2M_CD1A \
    --notebook-path results/test_B2M_CD1A/evidence_report.ipynb
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/configs/test_config.yaml \
    --claim "B2M as ligand directly interacts with CD1A as receptor." \
    --output-dir results/test_B2M_CD1A_no_context \
    --notebook-path results/test_B2M_CD1A_no_context/evidence_report.ipynb
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/configs/test_config.yaml \
    --claim "AANAT as ligand directly interacts with MTNR1A as receptor." \
    --output-dir results/test_AANAT_MTNR1A_no_context \
    --notebook-path results/test_AANAT_MTNR1A_no_context/evidence_report.ipynb
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import yaml

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = "/hps/nobackup/saezrodriguez/shared_datasets/claims/datasets/connectomedb.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "connectomedb_eval_results.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "connectomedb_eval"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "connectomedb_eval_config.yaml"

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
logger = logging.getLogger("connectomedb_eval")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_id(cdb_id: str) -> str:
    """Convert a CDB ID to a filesystem-safe directory name."""
    return cdb_id.replace(":", "_").replace(" ", "_")


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
    """Safely parse a verdict.json file to extract required fields."""
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
    """Runs evidence_programming.py via subprocess to evaluate the given claim."""

    verdict_path = output_dir / "workspace" / "verdict.json"

    if verdict_path.exists():
        logger.info(f"Verdict already exists, skipping run: {output_dir}")
        stats = parse_verdict_file(verdict_path)

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
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "total_input_tokens": 0,
            "cost_estimate": 0.0
        }

    logger.info(f"Run finished in {time.time() - start_time:.1f}s")

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
                        usage_str = line.split("INFO: Usage: ")[1]
                        usage_str = usage_str.replace("'", '"')
                        usage_dict = json.loads(usage_str)
                        in_tok += usage_dict.get("input_tokens", 0)
                        out_tok += usage_dict.get("output_tokens", 0)
                        cache_creation_tok += usage_dict.get("cache_creation_input_tokens", 0)
                        cache_read_tok += usage_dict.get("cache_read_input_tokens", 0)
                    except Exception as e:
                        logger.debug(f"Failed to parse usage line: {e}")
        except Exception as e:
            logger.warning(f"Could not read run.log: {e}")

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
    parser = argparse.ArgumentParser(description="End-to-End Evaluation of ConnectomeDB Dataset")
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV, help="Path to connectomedb.csv")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag for namespaced output directory (e.g. 20260330_142500).")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to write results (derived from --run-tag if omitted).")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Base VerificationSettings YAML.")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per claim.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of input CSV rows to process (0=all).")
    parser.add_argument("--row-start", type=int, default=0, help="Start row index for parallel chunking (default: 0).")

    args = parser.parse_args()

    # Resolve output paths (namespaced by run-tag when provided)
    if args.run_tag:
        results_dir = PROJECT_ROOT / "results" / f"connectomedb_eval_{args.run_tag}"
        default_csv = results_dir / "results.csv"
    else:
        results_dir = DEFAULT_RESULTS_DIR
        default_csv = DEFAULT_OUTPUT_CSV

    output_path = Path(args.output_csv) if args.output_csv else default_csv
    config_path = Path(args.config)

    # Determine model and pricing
    model_name = get_model_from_config(config_path)
    in_price, out_price = CLAUDE_PRICING.get(model_name, DEFAULT_PRICING)
    logger.info(f"Using pricing for model '{model_name}': ${in_price:.2f}/1M input, ${out_price:.2f}/1M output")

    # Propagate LLM_BASE_URL to subprocesses (notebook kernels)
    env_vars = {}
    if "LLM_BASE_URL" in os.environ:
        env_vars["LLM_BASE_URL"] = os.environ["LLM_BASE_URL"]
        logger.info(f"Propagating LLM_BASE_URL={os.environ['LLM_BASE_URL']}")

    logger.info(f"Reading dataset: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if args.limit > 0:
        df = df.iloc[args.row_start:args.row_start + args.limit]
    elif args.row_start > 0:
        df = df.iloc[args.row_start:]

    # Define columns for the output CSV
    out_cols = [
        "CDB_ID", "Ligand", "Receptor", "Label",
        "Claim_String", "Repetition", "Agent_Verdict",
        "Agent_Confidence", "Reasoning_Snippet", "Output_Directory",
        "Input_Tokens", "Output_Tokens", "Cache_Creation_Tokens", "Cache_Read_Tokens",
        "Total_Input_Tokens", "Cost_Estimate"
    ]

    # Initialize output CSV; build resume set from existing rows
    existing_runs: set = set()
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
                    existing_runs.add((row.get("CDB_ID"), int(row.get("Repetition", 1))))
                except Exception:
                    pass

    # Process each row
    total_rows = len(df)
    for idx, row in df.iterrows():
        cdb_id = str(row.get("id", f"ROW_{idx}"))
        ligand = row.get("ligand", "")
        receptor = row.get("receptor", "")
        label = row.get("label", "UNCERTAIN")
        claim_str = str(row["claim"])

        logger.info(f"--- Processing [{idx+1}/{total_rows}] {cdb_id} : {ligand} -> {receptor} ---")

        for rep in range(1, args.reps + 1):
            if (cdb_id, rep) in existing_runs:
                logger.info(f"  Skipping rep {rep} (already done)")
                continue

            safe_id = sanitize_id(cdb_id)
            run_dir = results_dir / safe_id / f"rep_{rep}"

            logger.info(f"Running Repetition {rep}/{args.reps}")
            stats = run_evaluation(claim_str, run_dir, config_path, in_price, out_price, env_vars=env_vars)

            row_dict = {
                "CDB_ID": cdb_id,
                "Ligand": ligand,
                "Receptor": receptor,
                "Label": label,
                "Claim_String": claim_str,
                "Repetition": rep,
                "Agent_Verdict": stats.get("verdict"),
                "Agent_Confidence": stats.get("confidence"),
                "Reasoning_Snippet": str(stats.get("reasoning", ""))[:500],
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
