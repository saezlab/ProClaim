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

# LLM-only with a specific model:
uv run python experiments/run_baselines_datasets.py \\
    --baseline llm_only --model anthropic/claude-sonnet-4-6

# S2 retrieval with top-k=10:
uv run python experiments/run_baselines_datasets.py \\
    --baseline s2_retrieval --model anthropic/claude-sonnet-4-6 --top-k 10

# FIRE with custom max-steps:
uv run python experiments/run_baselines_datasets.py \\
    --baseline fire --model anthropic/claude-sonnet-4-6 --max-steps 5
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
    _EXTRA_COLS = ("evidence", "pmid", "entity_a", "entity_b", "effect", "ligand", "receptor")
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


def _split_model_provider(model: str) -> tuple[str, str]:
    """Split a LiteLLM model string 'provider/model-id' into (api, model_name).

    Falls back to ('openai', model) if no provider prefix is present.
    """
    if "/" in model:
        api, model_name = model.split("/", 1)
        return api, model_name
    return "openai", model


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
            max_tokens=args.max_tokens,
        )
        return LLMOnly(llm=llm)
    elif name == "single_paper":
        from baselines.single_paper import SinglePaper
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return SinglePaper(llm=llm)
    elif name == "s2_retrieval":
        from baselines.s2_retrieval import S2Retrieval
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return S2Retrieval(llm=llm, top_k=args.top_k)
    elif name == "open_scholar":
        from baselines.open_scholar_baseline import OpenScholarBaseline
        api, model_name = _split_model_provider(args.model)
        raw_max = args.max_tokens
        return OpenScholarBaseline(
            model=model_name,
            api=api,
            top_n=args.top_k,
            max_tokens=raw_max if raw_max and raw_max > 0 else None,
            use_retrieval=args.retrieval,
            oracle_context=args.oracle_context,
            reranker=args.reranker or None,
            task_name=args.task_name,
        )
    elif name == "fire":
        from baselines.fire_baseline import FIREBaseline
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return FIREBaseline(
            llm=llm,
            max_steps=args.max_steps,
        )
    elif name == "ace":
        from baselines.ace_baseline import ACEBaseline
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return ACEBaseline(
            llm=llm,
            playbook=args.playbook if args.playbook else None,
        )
    elif name == "react":
        from baselines.react_baseline import ReActBaseline
        from baselines.shared.llm import LLMBackend
        llm = LLMBackend(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return ReActBaseline(
            llm=llm,
            max_steps=args.max_steps,
            search_backend=args.search_backend,
        )
    else:
        raise ValueError(f"Unknown baseline: {name!r}. Supported: random, llm_only, single_paper, s2_retrieval, open_scholar, fire, ace, react")


def _mean_std(values: list[float], decimals: int = 2) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {"mean": round(mean, decimals), "std": round(math.sqrt(variance), decimals)}


def aggregate_metrics(all_runs: list[dict]) -> dict:
    """Aggregate a list of per-repeat metric dicts into mean ± std."""
    scalar_keys = ["accuracy", "macro_f1", "macro_fpr", "macro_fnr", "weighted_fpr", "weighted_fnr", "weighted_tpr", "weighted_tnr"]
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

    agg["total_cost_usd"] = _mean_std([r["total_cost_usd"] for r in all_runs], decimals=3)
    agg["avg_cost_usd"] = _mean_std([r["avg_cost_usd"] for r in all_runs], decimals=3)
    agg["total_input_tokens"] = _mean_std([r["total_input_tokens"] for r in all_runs], decimals=3)
    agg["total_output_tokens"] = _mean_std([r["total_output_tokens"] for r in all_runs], decimals=3)

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
    p.add_argument("--baseline", default="random", choices=["random", "llm_only", "single_paper", "s2_retrieval", "open_scholar", "fire", "ace", "react"], help="Baseline to run.")
    p.add_argument("--seed", type=int, default=100, help="Base random seed. Each repeat i uses seed+i.")
    p.add_argument("--repeats", type=int, default=10, help="Number of independent repeats. Each repeat i uses seed+i.")
    # ── Shared LLM arguments (apply to all LLM-backed baselines) ─────
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6", help="LiteLLM model string, e.g. 'anthropic/claude-sonnet-4-6' or 'openai/gpt-4o'. For OpenScholar the provider prefix is split into --model_name / --api automatically.")
    p.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature (0.0–1.0).")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096, help="Max generation tokens per LLM call.")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=10, help="Maximum iterative search/reasoning steps (used by fire, react).")
    p.add_argument("--top-k", dest="top_k", type=int, default=5, help="Number of retrieved items (S2 abstracts for s2_retrieval; passages for open_scholar).")
    # ── S2 Retrieval-specific ─────────────────────────────────────────
    p.add_argument("--no-strip-query", dest="strip_query", action="store_false", default=True,
                    help="Disable stripping dataset-specific boilerplate from claims before S2 search.")
    # ── OpenScholar-specific ──────────────────────────────────────────
    p.add_argument("--oracle-context", dest="oracle_context", action="store_true", help="Inject CSV evidence as oracle context in OpenScholar (default: False; for ablation only).")
    p.add_argument("--retrieval", dest="retrieval", action="store_true", help="Enable S2 retrieval + feedback in OpenScholar (--ss_retriever --feedback).")
    p.add_argument("--reranker", default="OpenScholar/OpenScholar_Reranker", help="Reranker model for OpenScholar. Set to empty string to disable.")
    p.add_argument("--task-name", dest="task_name", default="claim_verdict_question", help="OpenScholar task name (e.g. claim_verdict_question, claim_verdict).")
    # ── ACE-specific ──────────────────────────────────────────────────
    p.add_argument("--playbook", default=None, help="Path to a pre-trained ACE playbook .txt file (optional).")
    # ── ReAct-specific ────────────────────────────────────────────────
    p.add_argument("--search-backend", dest="search_backend", default="web", choices=["web", "s2"], help="Search backend for ReAct: 'web' (Serper/DuckDuckGo) or 's2' (Semantic Scholar).")
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
    output_dir = Path(args.output_dir)

    # Append a sanitised model name so runs for different models don't
    # overwrite each other.  All non-random baselines include the model slug.
    model_slug = args.model.replace("/", "--")
    baseline_subdir = args.baseline
    if args.baseline == "random":
        pass  # no model
    elif args.baseline == "s2_retrieval":
        baseline_subdir = f"{args.baseline}/{model_slug}/top{args.top_k}"
    elif args.baseline == "react":
        baseline_subdir = f"react/{args.search_backend}/{model_slug}"
    else:
        baseline_subdir = f"{args.baseline}/{model_slug}"

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
            # Set per-dataset query processor when stripping is enabled
            if hasattr(baseline, "query_process") and getattr(args, "strip_query", True):
                from baselines.s2_retrieval import QUERY_PROCESSORS
                baseline.query_process = QUERY_PROCESSORS.get(dataset_name)
            out_path = output_dir / baseline_subdir / f"{dataset_name}_seed{seed}.jsonl"
            if hasattr(baseline, "log_dir"):
                baseline.log_dir = out_path.parent / f"{dataset_name}_seed{seed}_logs"
            harness = EvaluationHarness(baseline, dataset_name=dataset_name)
            results = harness.run(claims, resume_path=out_path)
            m = EvaluationHarness.metrics(results)
            repeat_metrics.append(m)
            logger.info(
                "  [repeat %d/%d seed=%d]  accuracy=%.2f  macro_f1=%.2f  w_fpr=%.2f  w_fnr=%.2f  w_tpr=%.2f  w_tnr=%.2f",
                rep + 1, n_repeats, seed,
                m["accuracy"], m["macro_f1"], m["weighted_fpr"], m["weighted_fnr"],
                m["weighted_tpr"], m["weighted_tnr"],
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
            "  AGGREGATED (%d repeats): accuracy=%.2f±%.2f  macro_f1=%.2f±%.2f  w_fpr=%.2f±%.2f  w_fnr=%.2f±%.2f  w_tpr=%.2f±%.2f  w_tnr=%.2f±%.2f",
            n_repeats,
            agg["accuracy"]["mean"], agg["accuracy"]["std"],
            agg["macro_f1"]["mean"], agg["macro_f1"]["std"],
            agg["weighted_fpr"]["mean"], agg["weighted_fpr"]["std"],
            agg["weighted_fnr"]["mean"], agg["weighted_fnr"]["std"],
            agg["weighted_tpr"]["mean"], agg["weighted_tpr"]["std"],
            agg["weighted_tnr"]["mean"], agg["weighted_tnr"]["std"],
        )
        logger.info("  Metrics saved to %s", metrics_path)

        all_metrics[dataset_name] = agg

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"\n{'Dataset':<16} {'Macro F1':>12} {'W-FPR':>12} {'W-FNR':>12} {'W-TPR':>12} {'W-TNR':>12} {'Cost (USD)':>14}")
    print("-" * 90)
    for dataset_name, agg in all_metrics.items():
        mf1  = agg["macro_f1"]
        fpr  = agg["weighted_fpr"]
        fnr  = agg["weighted_fnr"]
        tpr  = agg["weighted_tpr"]
        tnr  = agg["weighted_tnr"]
        cost = agg["total_cost_usd"]
        print(
            f"{dataset_name.upper():<16}"
            f" {mf1['mean']:.2f}±{mf1['std']:.2f}"
            f" {fpr['mean']:.2f}±{fpr['std']:.2f}"
            f" {fnr['mean']:.2f}±{fnr['std']:.2f}"
            f" {tpr['mean']:.2f}±{tpr['std']:.2f}"
            f" {tnr['mean']:.2f}±{tnr['std']:.2f}"
            f" {cost['mean']:.3f}±{cost['std']:.3f}"
        )


if __name__ == "__main__":
    main()
