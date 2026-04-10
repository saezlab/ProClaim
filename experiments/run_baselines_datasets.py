#!/usr/bin/env python3
"""
Run baselines on pre-processed dataset CSVs from the datasets directory.

Expects CSVs with at minimum: id, claim, label columns.

For stochastic baselines (random), multiple repeats are run and metrics are
aggregated as mean ± std across repeats. Each repeat uses seed + repeat_index
so results are reproducible.

Usage
-----
# Random baseline, 10 repeats, seed=100 (defaults):
uv run python experiments/run_baselines_datasets.py \\
    --datasets-dir /path_to/connectomeDB_data/datasets \\
    --datasets signor connectomedb \\
    --baseline random

# Custom seed / repeats:
uv run python experiments/run_baselines_datasets.py \\
    --datasets-dir /path_to/connectomeDB_data/datasets \\
    --datasets signor connectomedb \\
    --baseline random --seed 42 --repeats 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path

import yaml

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))  # for experiments/baselines/

from baselines.shared.evaluate import EvaluationHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baselines_datasets")


def load_claims(csv_path: Path) -> list[dict]:
    """Load a pre-processed dataset CSV into a list of claim dicts."""
    df = pd.read_csv(csv_path)
    # Extra columns forwarded to richer baselines (e.g. OpenScholar RAG)
    _EXTRA_COLS = ("evidence", "pmid", "entity_a", "entity_b", "effect")
    claims = []
    for _, row in df.iterrows():
        base_id = str(row["id"])
        # Flip variants share the same base id in the SIGNOR dataset; disambiguate
        # them so the evaluation harness can store and resume them independently.
        flip_val = row.get("flip", None)
        if flip_val is not None and str(flip_val).strip().lower() in ("true", "1"):
            claim_id = base_id + "_flip"
        else:
            claim_id = base_id
        item: dict = {
            "claim_id": claim_id,
            "claim": str(row["claim"]),
            "gold_label": str(row["label"]),
            "dataset": str(row.get("dataset", csv_path.stem)),
        }
        for col in _EXTRA_COLS:
            if col in row.index and pd.notna(row[col]):
                item[col] = str(row[col])
        claims.append(item)
    return claims


def build_baseline(name: str, args: argparse.Namespace, seed: int | None = None):
    if name == "random":
        from baselines.random_baseline import RandomBaseline
        return RandomBaseline(seed=seed if seed is not None else args.seed)
    elif name == "llm_only":
        from baselines.llm_only import LLMOnly
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=args.model,
            temperature=args.temperature,
        )
        return LLMOnly(llm=llm)
    elif name == "s2_retrieval":
        from baselines.s2_retrieval import S2Retrieval
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=getattr(args, "s2_model", "anthropic/claude-sonnet-4-6"),
            temperature=args.temperature,
        )
        return S2Retrieval(llm=llm, top_k=getattr(args, "s2_top_k", 5))
    elif name == "open_scholar":
        from baselines.open_scholar_baseline import OpenScholarBaseline
        raw_max = getattr(args, "os_max_tokens", None)
        return OpenScholarBaseline(
            model=getattr(args, "os_model", "claude-sonnet-4-6"),
            api=getattr(args, "os_api", "anthropic"),
            top_n=getattr(args, "os_top_n", 5),
            max_tokens=raw_max if raw_max and raw_max > 0 else None,
            use_retrieval=getattr(args, "os_retrieval", False),
            oracle_context=getattr(args, "os_oracle_context", False),
            reranker=getattr(args, "os_reranker", None) or None,
            task_name=getattr(args, "os_task_name", "claim_verdict_question"),
        )
    elif name == "fire":
        from baselines.fire_baseline import FIREBaseline
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=getattr(args, "fire_model", "openai/gpt-4o-mini"),
            temperature=args.temperature,
            max_tokens=2048,
        )
        return FIREBaseline(
            llm=llm,
            max_steps=getattr(args, "fire_max_steps", 5),
        )
    elif name == "ace":
        from baselines.ace_baseline import ACEBaseline
        playbook = getattr(args, "ace_playbook", None)
        return ACEBaseline(
            model=getattr(args, "ace_model", "openai/gpt-4o-mini"),
            max_tokens=getattr(args, "ace_max_tokens", 4096),
            playbook=playbook if playbook else None,
            temperature=args.temperature,
        )
    elif name == "react":
        from baselines.react_baseline import ReActBaseline
        return ReActBaseline(
            model=getattr(args, "react_model", "openai/gpt-4o-mini"),
            max_steps=getattr(args, "react_max_steps", 10),
            temperature=args.temperature,
            search_backend=getattr(args, "react_search_backend", "web"),
        )
    else:
        raise ValueError(f"Unknown baseline: {name!r}. Supported: random, llm_only, s2_retrieval, open_scholar, fire, ace, react")


def _mean_std(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {"mean": round(mean, 4), "std": round(math.sqrt(variance), 4)}


def aggregate_metrics(all_runs: list[dict]) -> dict:
    """Aggregate a list of per-repeat metric dicts into mean ± std."""
    scalar_keys = ["accuracy", "macro_f1", "macro_fpr", "macro_fnr", "weighted_fpr", "weighted_fnr"]
    per_class_labels = ["SUPPORT", "REFUTE", "UNCERTAIN"]
    per_class_subkeys = ["precision", "recall", "f1"]

    agg: dict = {
        "n": all_runs[0]["n"],
        "n_repeats": len(all_runs),
    }

    for key in scalar_keys:
        agg[key] = _mean_std([r[key] for r in all_runs])

    agg["per_class"] = {}
    for label in per_class_labels:
        agg["per_class"][label] = {}
        for sub in per_class_subkeys:
            agg["per_class"][label][sub] = _mean_std(
                [r["per_class"][label][sub] for r in all_runs]
            )

    agg["total_cost_usd"] = _mean_std([r["total_cost_usd"] for r in all_runs])
    agg["avg_cost_usd"] = _mean_std([r["avg_cost_usd"] for r in all_runs])
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="Path to a YAML config file. CLI flags override values from the config.")
    p.add_argument(
        "--datasets-dir",
        default="/path_to/connectomeDB_data/datasets",
        help="Directory containing dataset CSV files.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["signor", "connectomedb"],
        help="Dataset names (without .csv extension).",
    )
    p.add_argument("--baseline", default="random", choices=["random", "llm_only", "s2_retrieval", "open_scholar", "fire", "ace", "react"], help="Baseline to run.")
    p.add_argument("--seed", type=int, default=100, help="Base random seed. Each repeat i uses seed+i.")
    p.add_argument("--repeats", type=int, default=10, help="Number of independent repeats. Each repeat i uses seed+i.")
    p.add_argument("--model", default="zai/glm-4-plus", help="LiteLLM model string for llm_only, e.g. 'zai/glm-4-plus' or 'openai/gpt-4o'.")
    # S2 Retrieval-specific arguments
    p.add_argument("--s2-model", dest="s2_model", default="anthropic/claude-sonnet-4-6", help="LiteLLM model string for S2 retrieval baseline.")
    p.add_argument("--s2-top-k", dest="s2_top_k", type=int, default=5, help="Number of S2 abstracts to retrieve (e.g. 5, 10).")
    # OpenScholar-specific arguments
    p.add_argument("--os-model", dest="os_model", default="claude-sonnet-4-6", help="Model name for OpenScholar (--model_name in run.py).")
    p.add_argument("--os-api", dest="os_api", default="anthropic", help="API provider for OpenScholar (e.g. anthropic, gemini).")
    p.add_argument("--os-top-n", dest="os_top_n", type=int, default=5, help="Number of passages for OpenScholar (--top_n).")
    p.add_argument("--os-max-tokens", dest="os_max_tokens", type=int, default=3000, help="Max generation tokens for OpenScholar (0 = no constraint, use OpenScholar default of 3000).")
    p.add_argument("--os-oracle-context", dest="os_oracle_context", action="store_true", help="Inject CSV evidence as oracle context in OpenScholar (default: False; use for oracle-leakage experiments).")
    p.add_argument("--os-retrieval", dest="os_retrieval", action="store_true", help="Enable S2 retrieval + feedback in OpenScholar (--ss_retriever --feedback).")
    p.add_argument("--os-reranker", dest="os_reranker", default="OpenScholar/OpenScholar_Reranker", help="Reranker model for OpenScholar (--ranking_ce --reranker). Set to empty string to disable. Default: OpenScholar/OpenScholar_Reranker.")
    p.add_argument("--os-task-name", dest="os_task_name", default="claim_verdict_question", help="OpenScholar task name passed to --task_name (e.g. claim_verdict_question, claim_verdict).")
    # FIRE-specific arguments
    p.add_argument("--fire-model", dest="fire_model", default="openai/gpt-4o-mini", help="LiteLLM model string for FIRE (e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4-20250514).")
    p.add_argument("--fire-max-steps", dest="fire_max_steps", type=int, default=5, help="Maximum iterative search steps for FIRE.")
    # ACE-specific arguments
    p.add_argument("--ace-model", dest="ace_model", default="openai/gpt-4o-mini", help="LiteLLM model string for ACE Generator (e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4-20250514).")
    p.add_argument("--ace-max-tokens", dest="ace_max_tokens", type=int, default=4096, help="Max generation tokens for ACE.")
    p.add_argument("--ace-playbook", dest="ace_playbook", default=None, help="Path to a pre-trained ACE playbook .txt file (optional; uses built-in claim verification playbook if omitted).")
    # ReAct-specific arguments
    p.add_argument("--react-model", dest="react_model", default="openai/gpt-4o-mini", help="LiteLLM model string for ReAct (e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4-20250514).")
    p.add_argument("--react-max-steps", dest="react_max_steps", type=int, default=10, help="Maximum number of search steps for ReAct.")
    p.add_argument("--react-search-backend", dest="react_search_backend", default="web", choices=["web", "s2"], help="Search backend for ReAct: 'web' (Serper/DuckDuckGo) or 's2' (Semantic Scholar).")
    p.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature. Overrides each baseline's default (llm_only/s2/react: 0.0, fire: 0.5, ace: 0.0). Anthropic supports 0.0–1.0.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of claims per dataset (0 = all).")
    p.add_argument(
        "--output-dir",
        default="results/baselines",
        help="Directory to save results.",
    )

    # Pre-parse --config so we can apply YAML defaults before the full parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        p.set_defaults(**cfg)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets_dir = Path(args.datasets_dir)
    output_dir = PROJECT_ROOT / args.output_dir

    # For llm_only / open_scholar, append a sanitised model name so runs for
    # different models don't overwrite each other.
    baseline_subdir = args.baseline
    if args.baseline == "llm_only":
        model_slug = args.model.replace("/", "--")
        baseline_subdir = f"{args.baseline}/{model_slug}"
    elif args.baseline == "s2_retrieval":
        model_slug = args.s2_model.replace("/", "--")
        baseline_subdir = f"{args.baseline}/{model_slug}/top{args.s2_top_k}"
    elif args.baseline == "open_scholar":
        model_slug = args.os_model.replace("/", "--")
        baseline_subdir = f"{args.baseline}/{model_slug}"
    elif args.baseline == "fire":
        model_slug = args.fire_model.replace("/", "--")
        baseline_subdir = f"{args.baseline}/{model_slug}"
    elif args.baseline == "ace":
        model_slug = args.ace_model.replace("/", "--")
        baseline_subdir = f"{args.baseline}/{model_slug}"
    elif args.baseline == "react":
        model_slug = args.react_model.replace("/", "--")
        search_be = getattr(args, "react_search_backend", "web")
        baseline_subdir = f"react/{search_be}/{model_slug}"

    logger.info("Baseline: %s  repeats=%d  base_seed=%d", args.baseline, args.repeats, args.seed)

    # Save run parameters to a log file in the output directory.
    run_log_dir = output_dir / baseline_subdir
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_params = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv,
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    run_params_path = run_log_dir / "run_params.json"
    with open(run_params_path, "w") as f:
        json.dump(run_params, f, indent=2)
    logger.info("Run parameters saved to %s", run_params_path)

    all_metrics: dict[str, dict] = {}

    for dataset_name in args.datasets:
        csv_path = datasets_dir / f"{dataset_name}.csv"
        if not csv_path.exists():
            logger.warning("CSV not found, skipping: %s", csv_path)
            continue

        logger.info("=== Dataset: %s ===", dataset_name)
        claims = load_claims(csv_path)
        if args.limit:
            claims = claims[: args.limit]
            logger.info("  Limited to %d claims", len(claims))
        else:
            logger.info("  Loaded %d claims", len(claims))

        n_repeats = args.repeats
        repeat_metrics: list[dict] = []

        for rep in range(n_repeats):
            seed = args.seed + rep
            baseline = build_baseline(args.baseline, args, seed=seed)
            out_path = output_dir / baseline_subdir / f"{dataset_name}_seed{seed}.jsonl"
            if hasattr(baseline, "log_dir"):
                baseline.log_dir = out_path.parent / f"{dataset_name}_seed{seed}_logs"
            harness = EvaluationHarness(baseline, dataset_name=dataset_name)
            results = harness.run(claims, resume_path=out_path)
            m = EvaluationHarness.metrics(results)
            repeat_metrics.append(m)
            logger.info(
                "  [repeat %d/%d seed=%d]  accuracy=%.4f  macro_f1=%.4f  w_fpr=%.4f  w_fnr=%.4f",
                rep + 1, n_repeats, seed,
                m["accuracy"], m["macro_f1"], m["weighted_fpr"], m["weighted_fnr"],
            )

            # Rewrite the JSONL in canonical (claims-order) form after full completion
            EvaluationHarness.save(results, out_path)

        agg = aggregate_metrics(repeat_metrics)

        # Save aggregated metrics
        metrics_path = output_dir / baseline_subdir / f"{dataset_name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(agg, f, indent=2)

        logger.info(
            "  AGGREGATED (%d repeats): accuracy=%.4f±%.4f  macro_f1=%.4f±%.4f  w_fpr=%.4f±%.4f  w_fnr=%.4f±%.4f",
            n_repeats,
            agg["accuracy"]["mean"], agg["accuracy"]["std"],
            agg["macro_f1"]["mean"], agg["macro_f1"]["std"],
            agg["weighted_fpr"]["mean"], agg["weighted_fpr"]["std"],
            agg["weighted_fnr"]["mean"], agg["weighted_fnr"]["std"],
        )
        logger.info("  Metrics saved to %s", metrics_path)

        all_metrics[dataset_name] = agg

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"\n{'Dataset':<16} {'Macro F1':>12} {'W-FPR':>12} {'W-FNR':>12} {'Cost (USD)':>12}")
    print("-" * 64)
    for dataset_name, agg in all_metrics.items():
        mf1  = agg["macro_f1"]
        fpr  = agg["weighted_fpr"]
        fnr  = agg["weighted_fnr"]
        cost = agg["total_cost_usd"]
        print(
            f"{dataset_name.upper():<16}"
            f" {mf1['mean']:.3f}±{mf1['std']:.3f}"
            f" {fpr['mean']:.3f}±{fpr['std']:.3f}"
            f" {fnr['mean']:.3f}±{fnr['std']:.3f}"
            f" {cost['mean']:.3f}±{cost['std']:.3f}"
        )


if __name__ == "__main__":
    main()
