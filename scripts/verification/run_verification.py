#!/usr/bin/env python3
"""
Metacognitive Verification CLI — end-to-end claim verification.

Uses the Claude Agent SDK with MCP tools for evidence programming.

Usage:
  # Single claim
  uv run python scripts/verification/run_verification.py \
      --claim "Does p53 activate BAX?"

  # Batch SIGNOR edges
  uv run python scripts/verification/run_verification.py \
      --dataset signor --max-edges 5

  # Batch SciFact claims
  uv run python scripts/verification/run_verification.py \
      --dataset scifact --scifact-path data/scifact/claims.json --max-edges 10
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

# Path resolution (project convention)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_single(claim: str, model: str, max_iterations: int,
               threshold: float, output_dir: Path) -> None:
    """Run a single claim via Claude Agent SDK."""
    from pkevolve.verification.orchestrator import verify_claim

    workspace = output_dir / "single_claim"
    verdict = asyncio.run(verify_claim(
        claim=claim,
        workspace=workspace,
        model=model,
        max_iterations=max_iterations,
        sufficiency_threshold=threshold,
    ))

    print("\n" + "=" * 60)
    print(f"Verdict: {verdict.verdict} (confidence: {verdict.confidence:.2f})")
    print(f"Reasoning: {verdict.reasoning}")
    print(f"Key evidence: {verdict.key_evidence}")
    print(f"Gaps remaining: {verdict.gaps_remaining}")
    print("=" * 60)
    print(f"Workspace: {workspace}")


def run_batch(dataset: str, model: str, max_edges: int,
              max_iterations: int, threshold: float,
              output_dir: Path, scifact_path: str = None,
              label: str = "both") -> None:
    """Run a batch of claims via Claude Agent SDK."""
    from pkevolve.verification.orchestrator import verify_claim_batch

    if dataset == "signor":
        from pkevolve.verification.adapters import SignorAdapter
        data_dir = PROJECT_ROOT / "data" / "signor"
        adapter = SignorAdapter(data_dir, label=label)
    elif dataset == "scifact":
        from pkevolve.verification.adapters import SciFactAdapter
        if not scifact_path:
            print("ERROR: --scifact-path required when --dataset scifact")
            sys.exit(1)
        adapter = SciFactAdapter(Path(scifact_path))
    else:
        print(f"ERROR: Unknown dataset: {dataset}")
        sys.exit(1)

    claims_list = list(adapter.iter_claims(max_claims=max_edges))
    print(f"Processing {len(claims_list)} claims from {dataset}")

    # Convert Claim dataclasses to dicts for the batch runner
    claims_dicts = [
        {"id": c.id, "text": c.text, "gold_label": c.gold_label}
        for c in claims_list
    ]

    results = asyncio.run(verify_claim_batch(
        claims=claims_dicts,
        output_dir=output_dir,
        model=model,
        max_iterations=max_iterations,
        sufficiency_threshold=threshold,
    ))

    # Summary
    correct = sum(
        1 for r in results
        if r.get("gold_label") and r["predicted"]["verdict"] == r["gold_label"]
    )
    total_with_gold = sum(1 for r in results if r.get("gold_label"))
    print(f"\nCompleted: {len(results)} claims")
    if total_with_gold > 0:
        print(f"Accuracy: {correct}/{total_with_gold} ({correct/total_with_gold:.1%})")
    print(f"Results: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Metacognitive Evidence Verification for scientific claims.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a single claim
  uv run python scripts/verification/run_verification.py \\
      --claim "Does p53 activate BAX?"

  # Batch SIGNOR edges
  uv run python scripts/verification/run_verification.py \\
      --dataset signor --max-edges 5

  # Batch SciFact claims
  uv run python scripts/verification/run_verification.py \\
      --dataset scifact --scifact-path data/scifact/claims.json
        """,
    )

    # Input mode
    parser.add_argument("--claim", type=str, help="A single scientific claim to verify.")
    parser.add_argument(
        "--dataset",
        choices=["signor", "scifact"],
        default="signor",
        help="Dataset to process in batch mode (default: signor).",
    )
    parser.add_argument(
        "--scifact-path", type=str, default=None,
        help="Path to SciFact claims JSON (required when --dataset scifact).",
    )

    # SIGNOR options
    parser.add_argument(
        "--label",
        choices=["true_positive", "true_negative", "both"],
        default="both",
        help="SIGNOR label filter (default: both).",
    )
    parser.add_argument(
        "--max-edges",
        type=int, default=5,
        help="Max number of edges/claims to process (0 = all). Default: 5.",
    )

    # Model options
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-5-20250929",
        help="Claude model identifier (default: claude-sonnet-4-5-20250929).",
    )

    # Hyperparameters
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Confidence threshold for early stopping (default: 0.80).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Maximum verification iterations (default: 8).",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results.",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "verification" / args.model

    if args.claim:
        run_single(
            claim=args.claim, model=args.model,
            max_iterations=args.max_iterations, threshold=args.threshold,
            output_dir=output_dir,
        )
    else:
        run_batch(
            dataset=args.dataset, model=args.model,
            max_edges=args.max_edges,
            max_iterations=args.max_iterations, threshold=args.threshold,
            output_dir=output_dir,
            scifact_path=args.scifact_path,
            label=args.label,
        )


if __name__ == "__main__":
    main()
