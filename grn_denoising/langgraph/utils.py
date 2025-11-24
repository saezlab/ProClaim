"""General utility functions for gene regulation analysis."""

import pandas as pd
from pathlib import Path


def load_edges(csv_path: str = "../all_removed_edges_with_sources.csv") -> pd.DataFrame:
    """Load the removed edges from CSV file.

    Args:
        csv_path: Path to the data file

    Returns:
        DataFrame with columns including ENTITYA (source_gene), ENTITYB (target_gene), EFFECT (relationship)
    """
    csv_file = Path(__file__).parent / csv_path
    df = pd.read_csv(csv_file)

    # Filter to keep only protein-protein interactions
    df_ppi = df[(df['TYPEA'] == 'protein') & (df['TYPEB'] == 'protein')].copy()

    # Rename columns for clarity
    df_ppi.rename(columns={
        'ENTITYA': 'source_gene',
        'ENTITYB': 'target_gene',
        'EFFECT': 'relationship'
    }, inplace=True)

    return df_ppi


def get_interaction_prompt(source: str, target: str, interaction: str) -> str:
    """Convert interaction type to natural language prompt.

    Args:
        source: Source gene name
        target: Target gene name
        interaction: Interaction type (e.g., 'up-regulates', 'down-regulates')

    Returns:
        Natural language description of the interaction
    """
    if interaction == 'down-regulates':
        return f"{source} down-regulate {target}"
    elif interaction == 'down-regulates activity':
        return f"{source} inhibit the activity of {target}"
    elif interaction == 'form complex':
        return f"{source} form a complex with {target}"
    elif interaction == 'up-regulates':
        return f"{source} up-regulate {target}"
    elif interaction == 'up-regulates activity':
        return f"{source} activate {target}"
    elif interaction == 'up-regulates quantity':
        return f"{source} increase {target} expression"
    elif interaction == 'up-regulates quantity by expression':
        return f"{source} increase {target} expression"
    else:  # handles 'unknown' and any other unexpected interactions
        return f"{source} interact with {target}"
