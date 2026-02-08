#!/usr/bin/env python3
"""
Metacognitive Verification CLI — end-to-end claim verification.

Usage:
  uv run python scripts/verification/run_verification.py --claim "Does p53 activate BAX?"
  uv run python scripts/verification/run_verification.py --claim "Does EGFR up-regulate MAPK?" --model gpt-oss-120b
  uv run python scripts/verification/run_verification.py --signor --edge-type positive --max-edges 5
"""

import sys
import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
import os

# Path resolution (project convention)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

from pkevolve.search.custom_pubmed import RelevancePubMedSearcher
from pkevolve.utils.signor_utils import load_signor_data, construct_signor_question
from pkevolve.verification.classifier import SufficiencyClassifier
from pkevolve.verification.compressor import SufficiencyPreservingCompressor
from pkevolve.verification.controller import MetacognitiveController


def setup_client(model_name: str) -> OpenAI:
    """Create an OpenAI-compatible client following project conventions."""
    if model_name == "glm-4.6":
        api_key = os.environ.get("GLM_API_KEY")
        if not api_key:
            raise ValueError("GLM_API_KEY not found. Check your .env file.")
        base_url = "https://open.bigmodel.cn/api/paas/v4/"
    else:
        api_key = "EMPTY"
        base_url = "http://localhost:8000/v1"
    print(f"Using {model_name} at {base_url}")
    return OpenAI(base_url=base_url, api_key=api_key)


def build_controller(
    client: OpenAI,
    model: str,
    threshold: float,
    max_iterations: int,
    context_budget: int,
) -> MetacognitiveController:
    """Assemble the full verification pipeline from backbone components."""
    classifier = SufficiencyClassifier()
    compressor = SufficiencyPreservingCompressor(
        llm_client=client, classifier=classifier
    )
    searcher = RelevancePubMedSearcher()
    return MetacognitiveController(
        llm_client=client,
        model=model,
        classifier=classifier,
        compressor=compressor,
        pubmed_searcher=searcher,
        threshold=threshold,
        max_iterations=max_iterations,
        context_budget=context_budget,
    )


def verify_single_claim(controller: MetacognitiveController, claim: str) -> dict:
    """Run verification on a single claim and return structured result."""
    label, confidence, report = controller.verify(claim)
    return {
        "claim": claim,
        "verdict": label,
        "confidence": confidence,
        "report": report,
    }


def run_signor_edges(
    controller: MetacognitiveController,
    edge_type: str,
    max_edges: int,
    output_dir: Path,
) -> None:
    """Run verification on SIGNOR edges and save results."""
    filename = (
        "true_positive_edges.csv" if edge_type == "positive"
        else "true_negative_edges.csv"
    )
    data_path = PROJECT_ROOT / "data" / "signor" / filename
    df = load_signor_data(str(data_path))
    if df is None:
        print(f"ERROR: Could not load {data_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_edges = min(max_edges, len(df)) if max_edges > 0 else len(df)
    print(f"Processing {n_edges} {edge_type} edges from {filename}")

    for idx in range(n_edges):
        row = df.iloc[idx]
        source = str(row["ENTITYA"])
        target = str(row["ENTITYB"])
        interaction = str(row["EFFECT"])

        claim = construct_signor_question(source, target, interaction)
        print(f"\n[{idx + 1}/{n_edges}] {source} → {target} ({interaction})")
        print(f"  Claim: {claim}")

        result = verify_single_claim(controller, claim)
        result["source"] = source
        result["target"] = target
        result["interaction"] = interaction
        result["edge_type"] = edge_type

        # Save result
        safe_interaction = interaction.replace(" ", "_").replace("/", "_")
        out_file = output_dir / f"{source}_{target}_{safe_interaction}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  Verdict: {result['verdict']} (confidence: {result['confidence']:.2f})")
        print(f"  Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Metacognitive Evidence Verification for scientific claims.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a single claim
  uv run python scripts/verification/run_verification.py \\
      --claim "Does p53 activate BAX?"

  # Verify SIGNOR edges
  uv run python scripts/verification/run_verification.py \\
      --signor --edge-type positive --max-edges 5

  # Custom model and parameters
  uv run python scripts/verification/run_verification.py \\
      --claim "Does EGFR phosphorylate MAPK?" \\
      --model gpt-oss-120b --max-iterations 5 --threshold 0.8
        """,
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--claim", type=str, help="A single scientific claim to verify."
    )
    input_group.add_argument(
        "--signor",
        action="store_true",
        help="Run verification on SIGNOR edges.",
    )

    # SIGNOR options
    parser.add_argument(
        "--edge-type",
        choices=["positive", "negative"],
        default="positive",
        help="SIGNOR edge type (default: positive).",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=5,
        help="Max number of SIGNOR edges to process (0 = all). Default: 5.",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        help="Model identifier (default: gpt-oss-120b).",
    )

    # Controller hyperparameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for early stopping (default: 0.7).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum verification iterations (default: 3).",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=50000,
        help="Approximate token budget before compression (default: 50000).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Default: results/verification/{model}/",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build pipeline
    client = setup_client(args.model)
    controller = build_controller(
        client=client,
        model=args.model,
        threshold=args.threshold,
        max_iterations=args.max_iterations,
        context_budget=args.context_budget,
    )

    if args.claim:
        # Single claim mode
        result = verify_single_claim(controller, args.claim)
        print("\n" + "=" * 60)
        print(result["report"])
        print("=" * 60)
        print(f"\nVerdict: {result['verdict']} (confidence: {result['confidence']:.2f})")

        # Save if output dir specified
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "result.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {out_file}")

    elif args.signor:
        # SIGNOR edges mode
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = (
                PROJECT_ROOT
                / "results"
                / "verification"
                / args.model
                / f"true_{args.edge_type}_edges"
            )
        run_signor_edges(controller, args.edge_type, args.max_edges, out_dir)


if __name__ == "__main__":
    main()
