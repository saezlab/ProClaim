#!/usr/bin/env python3
"""
Analyze all Claude API costs from run.log files across all experiments.
"""

import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Claude Sonnet 4.6 pricing
CLAUDE_SONNET_INPUT_PRICE = 3.0  # $ per 1M tokens
CLAUDE_SONNET_OUTPUT_PRICE = 15.0  # $ per 1M tokens


def parse_run_log(log_path: Path) -> Dict[str, int]:
    """Parse token usage from a single run.log file."""
    in_tok = 0
    out_tok = 0
    cache_creation_tok = 0
    cache_read_tok = 0

    try:
        log_content = log_path.read_text()
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
                    pass
    except Exception:
        pass

    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cache_creation_tokens": cache_creation_tok,
        "cache_read_tokens": cache_read_tok,
        "total_input_tokens": in_tok + cache_creation_tok + cache_read_tok,
    }


def calculate_cost(token_usage: Dict[str, int]) -> float:
    """Calculate cost with Anthropic prompt caching pricing."""
    cost_input = (token_usage["input_tokens"] / 1_000_000) * CLAUDE_SONNET_INPUT_PRICE
    cost_cache_write = (token_usage["cache_creation_tokens"] / 1_000_000) * CLAUDE_SONNET_INPUT_PRICE * 1.25
    cost_cache_read = (token_usage["cache_read_tokens"] / 1_000_000) * CLAUDE_SONNET_INPUT_PRICE * 0.1
    cost_output = (token_usage["output_tokens"] / 1_000_000) * CLAUDE_SONNET_OUTPUT_PRICE
    return cost_input + cost_cache_write + cost_cache_read + cost_output


def main():
    """Main function to analyze all Claude API costs."""
    # Find all run.log files with Claude usage
    all_logs = list(RESULTS_DIR.glob("**/run.log"))
    claude_logs = []

    for log_path in all_logs:
        # Check if it has Claude usage
        try:
            if "INFO: Usage:" in log_path.read_text():
                claude_logs.append(log_path)
        except Exception:
            pass

    print(f"Found {len(claude_logs)} run.log files with Claude API usage\n")
    print("=" * 80)

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation = 0
    total_cache_reads = 0

    results = []

    for log_path in sorted(claude_logs):
        usage = parse_run_log(log_path)
        cost = calculate_cost(usage)

        # Get relative path for display
        rel_path = log_path.relative_to(RESULTS_DIR)
        experiment_name = "/".join(rel_path.parts[:-1])  # Remove "run.log"

        results.append({
            "experiment": experiment_name,
            "path": str(rel_path),
            **usage,
            "cost": cost
        })

        print(f"\nExperiment: {experiment_name}")
        print(f"  Input tokens:         {usage['input_tokens']:>12,}")
        print(f"  Cache creation:       {usage['cache_creation_tokens']:>12,}")
        print(f"  Cache reads:          {usage['cache_read_tokens']:>12,}")
        print(f"  Total input:          {usage['total_input_tokens']:>12,}")
        print(f"  Output tokens:        {usage['output_tokens']:>12,}")
        print(f"  Cost:                 ${cost:>11.4f}")

        total_cost += cost
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        total_cache_creation += usage["cache_creation_tokens"]
        total_cache_reads += usage["cache_read_tokens"]

    print("\n" + "=" * 80)
    print("\nGRAND TOTAL:")
    print(f"  Experiments:          {len(results):>12}")
    print(f"  Input tokens:         {total_input_tokens:>12,}")
    print(f"  Cache creation:       {total_cache_creation:>12,}")
    print(f"  Cache reads:          {total_cache_reads:>12,}")
    print(f"  Total input:          {total_input_tokens + total_cache_creation + total_cache_reads:>12,}")
    print(f"  Output tokens:        {total_output_tokens:>12,}")
    print(f"  TOTAL COST:           ${total_cost:>11.2f}")
    print("\n" + "=" * 80)

    # Cost breakdown
    cost_breakdown_input = (total_input_tokens / 1_000_000) * CLAUDE_SONNET_INPUT_PRICE
    cost_breakdown_cache_write = (total_cache_creation / 1_000_000) * CLAUDE_SONNET_INPUT_PRICE * 1.25
    cost_breakdown_cache_read = (total_cache_reads / 1_000_000) * CLAUDE_SONNET_INPUT_PRICE * 0.1
    cost_breakdown_output = (total_output_tokens / 1_000_000) * CLAUDE_SONNET_OUTPUT_PRICE

    print("\nCost Breakdown:")
    print(f"  Regular input:        ${cost_breakdown_input:>11.4f}")
    print(f"  Cache writes (+25%):  ${cost_breakdown_cache_write:>11.4f}")
    print(f"  Cache reads (90% off): ${cost_breakdown_cache_read:>11.4f}")
    print(f"  Output:               ${cost_breakdown_output:>11.4f}")
    print(f"  TOTAL:                ${total_cost:>11.2f}")


if __name__ == "__main__":
    main()
