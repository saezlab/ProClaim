#!/usr/bin/env python3
"""
End-to-end evaluation script for the ConnectomeDB dataset using the RLM framework.

This script iterates through the ConnectomeDB CSV, reads the pre-formed claim
string from each row, and sequentially runs evidence_programming_direct.py for
multiple repetitions per claim. Results are appended incrementally to an
output CSV. Token usage and cost estimates are parsed from the verdict artifacts.

No flip variants are generated — each claim is run as-is.

To test a single claim (e.g., AFDN EPHA7) directly from the terminal, you can use:
Supported interaction:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "App as ligand directly interacts extracellularly with Cntn3 as receptor." \
    --output-dir results/test_App_Cntn3_supported

Intracellular interaction:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "AFDN as ligand directly interacts with EPHA7 as receptor." \
    --output-dir results/test_AFDN_EPHA7

Wrong direction:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "ITGB2 as ligand directly interacts with THY1 as receptor." \
    --output-dir results/test_ITGB2_THY1

In-cis interaction:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "LY86 as ligand directly interacts with CD180 as receptor." \
    --output-dir results/test_LY86_CD180

Not a ligand-receptor pair:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "B2M as ligand directly interacts with CD1A as receptor." \
    --output-dir results/test_B2M_CD1A_no_context

Not a protein-protein interaction:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "AANAT as ligand directly interacts with MTNR1A as receptor." \
    --output-dir results/test_AANAT_MTNR1A_no_context

--- Claim-level modified variants (extracellular constraint baked in, for testing without ICL) ---
Intracellular interaction (should REFUTE — AFDN-EPHA7 only interacts after endocytosis):
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "AFDN as ligand directly interacts extracellularly with EPHA7 as receptor." \
    --output-dir results/test_AFDN_EPHA7_extracellular

Wrong direction:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "ITGB2 as ligand directly interacts extracellularly with THY1 as receptor." \
    --output-dir results/test_ITGB2_THY1_extracellular

In-cis interaction (should REFUTE — LY86-CD180 interact on the same cell surface):
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "LY86 as ligand directly interacts extracellularly with CD180 as receptor." \
    --output-dir results/test_LY86_CD180_extracellular

Not a ligand-receptor pair:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "B2M as ligand directly interacts extracellularly with CD1A as receptor." \
    --output-dir results/test_B2M_CD1A_extracellular

Not a protein-protein interaction:
uv run python -m proclaim.verification.evidence_programming_direct \
    --config experiments/configs/test_config.yaml \
    --claim "AANAT as ligand directly interacts extracellularly with MTNR1A as receptor." \
    --output-dir results/test_AANAT_MTNR1A_extracellular
"""

import argparse
import csv
import json
import logging
import os
import queue
import subprocess
import threading
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
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Could not read run.log: {e}")
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_creation_tokens": cache_creation_tok,
        "cache_read_tokens": cache_read_tok,
    }


def _parse_tokens_from_usage_json(output_dir: Path) -> Dict[str, int]:
    """Parse direct-mode token usage from token_usage.json (CostTracker format).

    CostTracker summary fields:
      - input_tokens            = litellm prompt_tokens (non-cached + write + read)
      - non_cached_input_tokens = true non-cached tokens (preferred)
      - cache_write_tokens      = tokens written to cache (= cache_creation)
      - cache_read_tokens       = tokens served from cache
    """
    usage_path = output_dir / "token_usage.json"
    if usage_path.exists():
        try:
            data = json.loads(usage_path.read_text())
            cache_write = data.get("cache_write_tokens", data.get("cache_creation_tokens", 0))
            cache_read = data.get("cache_read_tokens", 0)
            total_input = data.get("input_tokens", data.get("prompt_tokens", 0))
            non_cached = data.get("non_cached_input_tokens",
                                  max(0, total_input - cache_write - cache_read))
            return {
                "input_tokens": non_cached,
                "output_tokens": data.get("output_tokens", data.get("completion_tokens", 0)),
                "cache_creation_tokens": cache_write,
                "cache_read_tokens": cache_read,
            }
        except Exception as e:
            logger.warning(f"Could not read token_usage.json: {e}")
    return {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0}


def _parse_tokens(output_dir: Path, mode: str) -> Dict[str, int]:
    if mode == "direct":
        return _parse_tokens_from_usage_json(output_dir)
    return _parse_tokens_from_run_log(output_dir)


def _compute_cost(tokens: Dict[str, int], in_price: float, out_price: float) -> float:
    cost = (
        (tokens["input_tokens"] / 1_000_000) * in_price
        + (tokens["cache_creation_tokens"] / 1_000_000) * in_price * 1.25
        + (tokens["cache_read_tokens"] / 1_000_000) * in_price * 0.1
        + (tokens["output_tokens"] / 1_000_000) * out_price
    )
    return round(cost, 4)



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


def init_output_csv(csv_path: Path, fieldnames: list[str]) -> None:
    """Create a CSV with a header if it does not already exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_output_row(csv_path: Path, fieldnames: list[str], row_dict: Dict[str, Any]) -> None:
    """Append a single row to a CSV, creating the file if needed."""
    init_output_csv(csv_path, fieldnames)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row_dict)


def collect_existing_runs(csv_paths: list[Path]) -> set[tuple[str | None, int]]:
    """Build the resume set from one or more result CSVs."""
    existing_runs: set[tuple[str | None, int]] = set()
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        logger.info("Resuming with existing output CSV: %s", csv_path)
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing_runs.add((row.get("CDB_ID"), int(row.get("Repetition", 1))))
                except Exception:
                    pass
    return existing_runs


def get_worker_shard_paths(output_path: Path, workers: int) -> list[Path]:
    """Return deterministic per-worker shard paths for one batch output."""
    return [
        output_path.with_name(f"{output_path.stem}.worker{worker_id + 1:02d}{output_path.suffix}")
        for worker_id in range(workers)
    ]


def merge_result_csvs(output_path: Path, source_paths: list[Path], fieldnames: list[str]) -> int:
    """Merge one or more result CSVs into the canonical output file, de-duplicating rows."""
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    seen_keys: set[tuple[str | None, str | None]] = set()
    merged_rows = 0

    with open(temp_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for source_path in source_paths:
            if not source_path.exists():
                continue
            with open(source_path, "r", newline="") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    row_key = (row.get("CDB_ID"), row.get("Repetition"))
                    if row_key in seen_keys:
                        continue
                    seen_keys.add(row_key)
                    writer.writerow({field: row.get(field, "") for field in fieldnames})
                    merged_rows += 1

    os.replace(temp_path, output_path)
    return merged_rows


def write_timing_report(timing_path: Path, report: Dict[str, Any]) -> None:
    """Persist a lightweight timing report for benchmark comparisons."""
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(report, indent=2))


def run_evaluation(
    claim: str,
    output_dir: Path,
    config_path: Path,
    in_price: float = DEFAULT_PRICING[0],
    out_price: float = DEFAULT_PRICING[1],
    env_vars: Dict[str, str] = None,
    mode: str = "sdk",
) -> Dict[str, Any]:
    """Runs evidence_programming (sdk) or evidence_programming_direct (direct) via subprocess."""

    verdict_path = output_dir / "workspace" / "verdict.json"

    if verdict_path.exists():
        logger.info(f"Verdict already exists, skipping run: {output_dir}")
        stats = parse_verdict_file(verdict_path)
        tokens = _parse_tokens(output_dir, mode)
        stats.update(tokens)
        stats["total_input_tokens"] = tokens["input_tokens"] + tokens["cache_creation_tokens"] + tokens["cache_read_tokens"]
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

    tokens = _parse_tokens(output_dir, mode)
    stats = parse_verdict_file(verdict_path)
    stats.update(tokens)
    stats["total_input_tokens"] = tokens["input_tokens"] + tokens["cache_creation_tokens"] + tokens["cache_read_tokens"]
    stats["cost_estimate"] = _compute_cost(tokens, in_price, out_price)
    return stats


def build_row_result(
    idx: int,
    total_rows: int,
    row: pd.Series,
    rep: int,
    reps: int,
    results_dir: Path,
    config_path: Path,
    in_price: float,
    out_price: float,
    env_vars: Dict[str, str],
    mode: str,
) -> Dict[str, Any]:
    """Run one claim repetition and return timing plus a CSV-ready row payload."""
    started_at = time.time()
    cdb_id = str(row.get("id", f"ROW_{idx}"))
    ligand = row.get("ligand", "")
    receptor = row.get("receptor", "")
    label = row.get("label", "UNCERTAIN")
    claim_str = str(row["claim"])
    safe_id = sanitize_id(cdb_id)
    run_dir = results_dir / safe_id / f"rep_{rep}"

    logger.info(
        "--- Processing [%d/%d] %s : %s -> %s (rep %d/%d) ---",
        idx + 1,
        total_rows,
        cdb_id,
        ligand,
        receptor,
        rep,
        reps,
    )
    stats = run_evaluation(
        claim_str,
        run_dir,
        config_path,
        in_price,
        out_price,
        env_vars=env_vars,
        mode=mode,
    )

    return {
        "cdb_id": cdb_id,
        "rep": rep,
        "job_seconds": round(time.time() - started_at, 2),
        "row": {
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
            "Cost_Estimate": stats.get("cost_estimate", 0.0),
        },
    }


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent claim repetitions to run against the same LLM endpoint.",
    )
    parser.add_argument(
        "--worker-shards",
        action="store_true",
        help="Write one intermediate result CSV per worker and merge them into the final output.",
    )

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

    # Determine mode, model, and pricing
    mode = get_mode_from_config(config_path)
    logger.info(f"Orchestration mode: {mode}")
    model_name = get_model_from_config(config_path)
    model_key = model_name.split("/")[-1] if "/" in model_name else model_name
    in_price, out_price = CLAUDE_PRICING.get(model_key, DEFAULT_PRICING)
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

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    # Define columns for the output CSV
    out_cols = [
        "CDB_ID", "Ligand", "Receptor", "Label",
        "Claim_String", "Repetition", "Agent_Verdict",
        "Agent_Confidence", "Reasoning_Snippet", "Output_Directory",
        "Input_Tokens", "Output_Tokens", "Cache_Creation_Tokens", "Cache_Read_Tokens",
        "Total_Input_Tokens", "Cost_Estimate"
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    init_output_csv(output_path, out_cols)
    shard_paths = get_worker_shard_paths(output_path, args.workers) if args.worker_shards else []
    if args.worker_shards:
        for shard_path in shard_paths:
            init_output_csv(shard_path, out_cols)
    existing_runs = collect_existing_runs([output_path, *shard_paths])

    total_rows = len(df)
    pending_jobs = []
    for idx, row in df.iterrows():
        cdb_id = str(row.get("id", f"ROW_{idx}"))
        for rep in range(1, args.reps + 1):
            if (cdb_id, rep) in existing_runs:
                logger.info("Skipping %s rep %d (already done)", cdb_id, rep)
                continue
            pending_jobs.append((idx, row.copy(), rep))

    logger.info(
        "Prepared %d pending claim repetitions across %d rows with workers=%d%s",
        len(pending_jobs),
        total_rows,
        args.workers,
        " (worker shards enabled)" if args.worker_shards else "",
    )

    if not pending_jobs:
        if args.worker_shards:
            merged_rows = merge_result_csvs(output_path, [output_path, *shard_paths], out_cols)
            logger.info("Merged %d rows into %s", merged_rows, output_path)
        logger.info("No pending claim repetitions to run.")
        return

    overall_started_at = time.time()
    timing_path = output_path.with_name(f"{output_path.stem}_timing.json")
    jobs_queue: queue.Queue[tuple[int, pd.Series, int] | None] = queue.Queue()
    progress_lock = threading.Lock()
    write_lock = threading.Lock()
    worker_stats: dict[int, Dict[str, Any]] = {}
    worker_errors: list[str] = []
    progress_state = {"completed": 0}

    for job in pending_jobs:
        jobs_queue.put(job)
    for _ in range(args.workers):
        jobs_queue.put(None)

    def worker_loop(worker_id: int, shard_path: Path | None) -> None:
        worker_started_at = time.time()
        jobs_completed = 0
        total_job_seconds = 0.0
        destination_path = shard_path or output_path

        logger.info(
            "Worker %d started (%s)",
            worker_id,
            f"shard={destination_path.name}" if shard_path else f"output={destination_path.name}",
        )

        while True:
            job = jobs_queue.get()
            try:
                if job is None:
                    break

                idx, row, rep = job
                result = build_row_result(
                    idx,
                    total_rows,
                    row,
                    rep,
                    args.reps,
                    results_dir,
                    config_path,
                    in_price,
                    out_price,
                    env_vars,
                    mode,
                )

                if shard_path is None:
                    with write_lock:
                        append_output_row(destination_path, out_cols, result["row"])
                else:
                    append_output_row(destination_path, out_cols, result["row"])

                jobs_completed += 1
                total_job_seconds += result["job_seconds"]

                with progress_lock:
                    progress_state["completed"] += 1
                    completed = progress_state["completed"]

                logger.info(
                    "Worker %d finished %s rep %d in %.1fs [%d/%d done]",
                    worker_id,
                    result["cdb_id"],
                    result["rep"],
                    result["job_seconds"],
                    completed,
                    len(pending_jobs),
                )
            except Exception as exc:
                logger.exception("Worker %d failed on a claim repetition", worker_id)
                with progress_lock:
                    worker_errors.append(f"worker {worker_id}: {exc}")
            finally:
                jobs_queue.task_done()

        worker_stats[worker_id] = {
            "worker_id": worker_id,
            "output_path": str(destination_path),
            "jobs_completed": jobs_completed,
            "job_seconds_total": round(total_job_seconds, 2),
            "job_seconds_avg": round(total_job_seconds / jobs_completed, 2) if jobs_completed else 0.0,
            "worker_wall_seconds": round(time.time() - worker_started_at, 2),
        }
        logger.info(
            "Worker %d done: jobs=%d total_job_seconds=%.1f wall=%.1fs",
            worker_id,
            jobs_completed,
            total_job_seconds,
            time.time() - worker_started_at,
        )

    threads = []
    for worker_index in range(args.workers):
        shard_path = shard_paths[worker_index] if args.worker_shards else None
        thread = threading.Thread(
            target=worker_loop,
            args=(worker_index + 1, shard_path),
            name=f"connectomedb-worker-{worker_index + 1}",
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if args.worker_shards:
        merged_rows = merge_result_csvs(output_path, [output_path, *shard_paths], out_cols)
        logger.info("Merged %d rows into %s", merged_rows, output_path)
    else:
        merged_rows = len(existing_runs) + progress_state["completed"]

    timing_report = {
        "output_csv": str(output_path),
        "timing_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": args.workers,
        "worker_shards": args.worker_shards,
        "scheduled_jobs": len(pending_jobs),
        "completed_jobs": progress_state["completed"],
        "pending_jobs": max(0, len(pending_jobs) - progress_state["completed"]),
        "total_wall_seconds": round(time.time() - overall_started_at, 2),
        "merged_rows": merged_rows,
        "worker_stats": [worker_stats[key] for key in sorted(worker_stats)],
    }
    if args.worker_shards:
        timing_report["worker_shard_paths"] = [str(path) for path in shard_paths]
    write_timing_report(timing_path, timing_report)
    logger.info("Wrote timing report to %s", timing_path)

    if worker_errors:
        raise RuntimeError(f"Encountered {len(worker_errors)} worker errors. First error: {worker_errors[0]}")


if __name__ == "__main__":
    main()
