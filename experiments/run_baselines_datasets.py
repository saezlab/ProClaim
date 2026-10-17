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

# Static retrieval (S2 backend, top-k=10):
uv run python experiments/run_baselines_datasets.py \\
    --baseline retrieval --search-backend s2 --model anthropic/claude-sonnet-4-6 --top-k 10

# FIRE with custom max-steps:
uv run python experiments/run_baselines_datasets.py \\
    --baseline fire --model anthropic/claude-sonnet-4-6 --max-steps 5
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path

import yaml

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))  # for experiments/baselines/

from baselines.shared.evaluate import EvaluationHarness
from baselines.shared.builder import build_llm_backend

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
            # The *_judge40.csv samples already carry the suffix in their id.
            claim_id = base_id if base_id.endswith("_flip") else base_id + "_flip"
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
        llm = build_llm_backend(args)
        return LLMOnly(llm=llm)
    elif name == "retrieval":
        from baselines.retrieval_baseline import RetrievalBaseline
        llm = build_llm_backend(args)
        return RetrievalBaseline(llm=llm, top_k=args.top_k, search_backend=args.search_backend)
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
        llm = build_llm_backend(args)
        return FIREBaseline(
            llm=llm,
            max_steps=args.max_steps,
            num_search_results=args.top_k,
            search_backend=args.search_backend,
        )
    elif name == "react":
        from baselines.react_baseline import ReActBaseline
        llm = build_llm_backend(args)
        fact_extractor = None
        if getattr(args, "extract_facts", False):
            from baselines.react_baseline import ResultFactExtractor
            from proclaim.verification.llm_factory import make_llm
            # ProClaim's extraction subagent settings (build_subprocess_env): local vLLM Qwen,
            # config temperature (0.0), thinking off.
            fact_extractor = ResultFactExtractor(make_llm(
                model=os.environ.get("LLM_MODEL", "qwen3.5-9b"),
                api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                base_url=os.environ["LLM_BASE_URL"],
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=int(os.environ.get("LLM_TIMEOUT", "300")),
            ))
        return ReActBaseline(
            llm=llm,
            max_steps=args.max_steps,
            num_search_results=args.top_k,
            search_backend=args.search_backend,
            fact_extractor=fact_extractor,
        )
    elif name == "claude_code":
        from baselines.claude_code_baseline import ClaudeCodeBaseline
        return ClaudeCodeBaseline(
            model=args.harness_model,
            evidence_source=args.evidence_source,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
            scratch_dir=args.scratch_dir,
        )
    elif name == "codex":
        from baselines.codex_baseline import CodexBaseline
        return CodexBaseline(
            evidence_source=args.evidence_source,
            timeout_seconds=args.timeout_seconds,
            bridge_port=args.bridge_port,
            scratch_dir=args.scratch_dir,
        )
    elif name == "oracle_context":
        from baselines.oracle_context_baseline import OracleContextBaseline
        if not args.cached_run_dir:
            raise ValueError("oracle_context requires --cached-run-dir (a ProClaim run directory)")
        llm = build_llm_backend(args)
        # One dataset per baseline instance: the cached run dir is dataset-specific.
        return OracleContextBaseline(llm, Path(args.cached_run_dir), args.datasets[0], args.context_mode)
    else:
        raise ValueError(f"Unknown baseline: {name!r}. Supported: random, llm_only, retrieval, open_scholar, fire, react, claude_code, codex, oracle_context")


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
        default="datasets",
        help="Directory containing dataset CSV files.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["signor", "connectomedb"],
        help="Dataset names (without .csv extension).",
    )
    p.add_argument("--baseline", default="random", choices=["random", "llm_only", "retrieval", "open_scholar", "fire", "react", "claude_code", "codex", "oracle_context"], help="Baseline to run.")
    p.add_argument("--seed", type=int, default=100, help="Base random seed. Each repeat i uses seed+i.")
    p.add_argument("--repeats", type=int, default=10, help="Number of independent repeats. Each repeat i uses seed+i.")
    # ── Shared LLM arguments (apply to all LLM-backed baselines) ─────
    p.add_argument("--model", default="anthropic/claude-sonnet-4-6", help="LiteLLM model string, e.g. 'anthropic/claude-sonnet-4-6' or 'openai/gpt-4o'. For OpenScholar the provider prefix is split into --model_name / --api automatically.")
    p.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature (0.0–1.0).")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096, help="Max generation tokens per LLM call.")
    p.add_argument("--reasoning-effort", dest="reasoning_effort", default=None,
                    choices=["none", "disable", "low", "medium", "high"],
                    help="LiteLLM reasoning_effort: controls thinking/CoT budget. 'none' minimises hidden thinking (recommended for fair baselines). Ignored by models that don't support it (via drop_params).")
    p.add_argument("--thinking-budget", dest="thinking_budget", type=int, default=None,
                    help="Explicit thinking budget in tokens.")
    p.add_argument("--extra-body", dest="extra_body", default=None,
                    help="Extra request-body fields as JSON (or a mapping in the YAML config), exported as "
                         "LLM_EXTRA_BODY for every LLMBackend call path, e.g. "
                         "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}' to turn GLM 5 reasoning off.")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None, help="Maximum iterative search/reasoning steps (used by fire, react). Default: 25.")
    p.add_argument("--top-k", dest="top_k", type=int, default=5, help="Number of retrieved items (S2 abstracts for retrieval; passages for open_scholar).")
    # ── Retrieval-specific ─────────────────────────────────────────────
    p.add_argument("--search-backend", dest="search_backend", default=None, choices=["web", "s2", "pubmed_s2"],
                    help="Search backend for retrieval, react and fire baselines: 'web' (DuckDuckGo), "
                         "'s2' (Semantic Scholar), or 'pubmed_s2'. Default: s2 (web for fire, its "
                         "upstream default).")
    p.add_argument("--no-strip-query", dest="strip_query", action="store_false", default=True,
                    help="Disable stripping dataset-specific boilerplate from claims before S2 search.")
    # ── OpenScholar-specific ──────────────────────────────────────────
    p.add_argument("--oracle-context", dest="oracle_context", action="store_true", help="Inject CSV evidence as oracle context in OpenScholar (default: False; for ablation only).")
    p.add_argument("--retrieval", dest="retrieval", action="store_true", help="Enable S2 retrieval + feedback in OpenScholar (--ss_retriever --feedback).")
    p.add_argument("--reranker", default="OpenScholar/OpenScholar_Reranker", help="Reranker model for OpenScholar. Set to empty string to disable.")
    p.add_argument("--task-name", dest="task_name", default="claim_verdict_question", help="OpenScholar task name (e.g. claim_verdict_question, claim_verdict).")
    # ── Black-box harness baselines (claude_code, codex) ──
    # These drive an external agent product, so they take no --model/--temperature;
    # each also has a standalone driver with its own concurrency and output paths.
    p.add_argument("--harness-model", dest="harness_model", default=None,
                    help="Backbone model for claude_code (e.g. claude-opus-4-8). Unset = the harness's own default.")
    p.add_argument("--evidence-source", dest="evidence_source", default="web", choices=["web", "pubmed_s2"],
                    help="Evidence access for claude_code/codex: native web search, or ProClaim's PubMed+S2 backend.")
    p.add_argument("--scratch-dir", dest="scratch_dir", default=None,
                    help="Scratch dir for isolated harness session state (default: a fresh temp dir).")
    p.add_argument("--max-turns", dest="max_turns", type=int, default=25, help="Per-claim turn cap for claude_code.")
    p.add_argument("--max-budget-usd", dest="max_budget_usd", type=float, default=0.75, help="Per-claim budget cap for claude_code.")
    p.add_argument("--timeout-seconds", dest="timeout_seconds", type=float, default=900, help="Per-claim wall-clock timeout for codex.")
    p.add_argument("--bridge-port", dest="bridge_port", type=int, default=4180, help="Port for codex's sonnet_bridge.py.")
    # ── Oracle-context-specific ───────────────────────────────────────
    p.add_argument("--cached-run-dir", dest="cached_run_dir", default=None, help="oracle_context: ProClaim run directory whose evidence states are read.")
    p.add_argument("--context-mode", dest="context_mode", default="full",
                    choices=["full", "facts_only", "papers_only", "papers_only_fulltext"],
                    help="How much of the cached evidence state to put in the prompt.")
    # ── ReAct (with decomposed claims) ────────────────────────────────
    p.add_argument("--subclaims-dir", dest="subclaims_dir", default=None,
                    help="react: directory of {dataset}_seed{seed}.json subclaim files "
                         "(scripts/analysis/extract_proclaim_subclaims.py); lists each claim's "
                         "subclaims in ReAct's first user message.")
    p.add_argument("--extract-facts", dest="extract_facts", action="store_true", default=False,
                   help="react: follow each search result with Qwen facts extracted from its full text "
                        "(ProClaim's fetch chain and EXTRACT_FACTS prompt; vLLM at LLM_BASE_URL).")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Claims run in parallel (worker threads via EvaluationHarness.arun).")
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

    args = p.parse_args()
    if args.max_steps is None:
        # Shared 25-step default for react/fire, matching the classes'
        # own defaults.
        args.max_steps = 25
    if args.search_backend is None:
        # FIRE keeps its upstream web-search default unless a config
        # or the CLI sets a backend explicitly; everything else defaults to S2.
        args.search_backend = "web" if args.baseline in ("fire",) else "s2"
    if args.extra_body:
        if isinstance(args.extra_body, str):
            args.extra_body = json.loads(args.extra_body)
        os.environ["LLM_EXTRA_BODY"] = json.dumps(args.extra_body)
    return args


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
    elif args.baseline == "retrieval":
        baseline_subdir = f"retrieval/{args.search_backend}/{model_slug}/top{args.top_k}"
    elif args.baseline == "react":
        # max_steps=10 was the original shared default across all react search
        # backends (react_web/react_s2/react_pubmed_s2) and is left out of the
        # path so existing 10-step outputs keep their location; any other
        # max_steps (including the current default of 25) gets its own path so
        # step-budget variants of the same backend never collide.
        backend_slug = args.search_backend
        if args.max_steps != 10:
            backend_slug = f"{backend_slug}_max{args.max_steps}"
        if args.subclaims_dir:
            backend_slug = f"{backend_slug}_proclaim_subclaims"
        if args.extract_facts:
            backend_slug = f"{backend_slug}_facts"
        baseline_subdir = f"react/{backend_slug}/{model_slug}"
    elif args.baseline in ("fire",) and args.search_backend != "web":
        # Paper-search variants of FIRE get their own path.
        baseline_subdir = f"{args.baseline}/{args.search_backend}_max{args.max_steps}/{model_slug}"
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
            if args.subclaims_dir:
                # Seed 100 + k reads ProClaim repeat k's subclaims; a missing file is an error.
                sub_path = Path(args.subclaims_dir) / f"{dataset_name}_seed{seed}.json"
                with open(sub_path) as fh:
                    baseline.subclaims = json.load(fh)
                logger.info("  Subclaims for %d claims from %s", len(baseline.subclaims), sub_path)
            # Set per-dataset query processor when stripping is enabled
            if hasattr(baseline, "query_process") and getattr(args, "strip_query", True):
                from baselines.retrieval_baseline import QUERY_PROCESSORS
                baseline.query_process = QUERY_PROCESSORS.get(dataset_name)
            out_path = output_dir / baseline_subdir / f"{dataset_name}_seed{seed}.jsonl"
            if hasattr(baseline, "log_dir"):
                baseline.log_dir = out_path.parent / f"{dataset_name}_seed{seed}_logs"
            harness = EvaluationHarness(baseline, dataset_name=dataset_name)
            if args.concurrency > 1:
                results = asyncio.run(harness.arun(claims, resume_path=out_path, concurrency=args.concurrency))
            else:
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
