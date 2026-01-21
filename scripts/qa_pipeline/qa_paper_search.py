"""
Retrieve paper details based on a natural language question.

Usage:
    uv run python QA_paper_search.py "Question here" [--output_base optional_base_name]

Example:
    uv run python QA_paper_search.py "Does GNAO1 down-regulate, inhibit, or destabilize ADCY1?" --output_base GNAO1_ADCY1_down-regulate --auto
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in path if not installed as package
# (Optional if running with uv/pip install -e ., but good for standalone script usage)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from pkevolve.search.paper_search_agent import run_paper_search_agent

def main():
    parser = argparse.ArgumentParser(description="QA Paper Search")
    parser.add_argument("question", help="Natural language question", nargs='?', default="Does CRTC2 up-regulates AKT1?")
    parser.add_argument("--output_base", help="Base name for output files (e.g. EntityA_EntityB_interaction)", default=None)
    parser.add_argument("--auto", action="store_true", help="Run in auto mode without interactive prompts")
    
    args = parser.parse_args()
    print(f"[DEBUG] Arguments parsed: question='{args.question}', output_base='{args.output_base}'")
    
    # Determine output base name
    if args.output_base:
        output_base = args.output_base
    else:
        # Sanitize question for filename
        output_base = "".join(c for c in args.question if c.isalnum() or c in ('_', '-')).strip()
        if not output_base:
            output_base = "search_results"

    # Search (and save incrementally)
    # Using the agentic loop
    results = run_paper_search_agent(args.question, output_base, target_full_text_count=5, auto=args.auto)
    
    if not results:
        print("[WARN] No results found or retrieved.")
        return

if __name__ == "__main__":
    main()
