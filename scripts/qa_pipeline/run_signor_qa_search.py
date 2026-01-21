"""echo "(time uv run run_signor_qa_search.py) > execution.log 2>&1"""
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

# Adjust path to import from scripts and src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Updated import to use the src module directly
from src.pkevolve.search.paper_search_agent import run_paper_search_agent
from src.pkevolve.utils.signor_utils import construct_signor_question, load_signor_data

def process_edge(row: pd.Series, category: str, output_base: Optional[str] = None):
    """
    Process a single edge: construct question, define save directory, and run search agent.
    """
    entity_a = str(row['ENTITYA'])
    entity_b = str(row['ENTITYB'])
    effect = str(row['EFFECT'])
    
    # Construct question using the utility
    question = construct_signor_question(entity_a, entity_b, effect)
    
    tqdm.write(f"Processing Edge: {entity_a} -> {entity_b} ({effect})")
    
    # Define save directory
    save_dir = project_root / f"data/papers/qa_search/{category}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Output base name for files
    if not output_base:
        # Sanitize filename components
        def sanitize(s):
            return "".join(c for c in s if c.isalnum() or c in ('_', '-')).strip()
        safe_a = sanitize(entity_a)
        safe_b = sanitize(entity_b)
        safe_effect = sanitize(effect)
        output_base = f"{safe_a}_{safe_b}_{safe_effect}"

    # Run search agent in auto mode
    run_paper_search_agent(
        question=question, 
        output_base=output_base, 
        target_full_text_count=5, 
        max_attempts=10, 
        auto=True,
        save_dir=save_dir
    )

def main():
    print("Starting SIGNOR QA Search...")
    
    # 1. Process True Positive Edges
    tp_path = project_root / "data/signor/true_positive_edges.csv"
    if tp_path.exists():
        print(f"\nLOADING True Positive Edges from {tp_path}")
        df_tp = load_signor_data(str(tp_path))
        if df_tp is not None and not df_tp.empty:
            print(f"Found {len(df_tp)} edges. Processing...")
            for idx, row in tqdm(df_tp.iterrows(), total=len(df_tp), desc="True Positive Edges"):
                try:
                    process_edge(row, "true_positive")
                except KeyboardInterrupt:
                    print("\n[INFO] Interrupted by user. Exiting...")
                    sys.exit(0)
                except Exception as e:
                    print(f"[ERROR] Failed to process edge {idx}: {e}")
        else:
            print("[WARN] True Positive data empty or failed to load.")
    else:
        print(f"[ERROR] True Positive file not found: {tp_path}")

    # 2. Process True Negative Edges
    tn_path = project_root / "data/signor/true_negative_edges.csv"
    if tn_path.exists():
        print(f"\nLOADING True Negative Edges from {tn_path}")
        df_tn = load_signor_data(str(tn_path))
        if df_tn is not None and not df_tn.empty:
            print(f"Found {len(df_tn)} edges. Processing...")
            for idx, row in tqdm(df_tn.iterrows(), total=len(df_tn), desc="True Negative Edges"):
                try:
                    process_edge(row, "true_negative")
                except KeyboardInterrupt:
                    print("\n[INFO] Interrupted by user. Exiting...")
                    sys.exit(0)
                except Exception as e:
                    print(f"[ERROR] Failed to process edge {idx}: {e}")
        else:
            print("[WARN] True Negative data empty or failed to load.")
    else:
        print(f"[ERROR] True Negative file not found: {tn_path}")

if __name__ == "__main__":
    main()
