import pandas as pd
import os
from typing import List, Optional

def load_signor_data(file_path: str, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Loads data from a SIGNOR CSV file.

    Args:
        file_path: Path to the CSV file.
        columns: List of columns to load. If None, loads all columns.

    Returns:
        DataFrame containing the loaded data, or None if an error occurs.
    """
    try:
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        
        if columns:
            # Check if columns exist
            if not all(col in df.columns for col in columns):
                missing = [col for col in columns if col not in df.columns]
                raise ValueError(f"Missing columns in CSV: {missing}")
            return df[columns]
            
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def construct_signor_question(source: str, target: str, interaction: str) -> str:
    """
    Constructs an inclusive natural language question to cover both general effects
    and specific mechanisms, maximizing the chance of identifying the correct edge sign.
    """
    
    # --- Group 1: Positive Interactions (A -> B) ---
    # Includes: up-regulates, up-regulates activity, up-regulates quantity, up-regulates quantity by expression
    if interaction in [
        'up-regulates', 
        'up-regulates activity', 
        'up-regulates quantity', 
        'up-regulates quantity by expression'
    ]:
        return f"Does {source} directly activate {target} (either through activation or increase of expression)?"

    # --- Group 2: Negative Interactions (A -| B) ---
    # Includes: down-regulates, down-regulates activity, down-regulates quantity by destabilization
    elif interaction in [
        'down-regulates', 
        'down-regulates activity', 
        'down-regulates quantity by destabilization'
    ]:
        return f"Does {source} directly inhibit {target} (either through inhibition or destabilization)?"

    # # --- Group 3: Complex / Other ---
    # elif interaction == 'form complex':
    #     return f"Does {source} form a complex with {target}?"
        
    # elif interaction == 'unknown':
    #     return f"Is the interaction between {source} and {target} unknown?"
        
    else:
        # Fallback for any missed types
        return f"Does {source} physically interact with {target}?"
