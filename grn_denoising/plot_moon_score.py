import pandas as pd
import numpy as np
import random
import ollama
from typing import Set, Tuple
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn
plt.style.use("./rw_visualization.mplstyle")
current_palette = seaborn.color_palette()

if __name__ == "__main__":
    remove_edge_num = 8
    model = 'qwen3:8b'
    model = model.replace(':', '_')
    path = Path(f"./results/moon_score/correct_{remove_edge_num}_edge")
    
    for i in range(10):
        file = path / Path(f"qwen3_8b_{i}.csv")
        df = pd.read_csv(file)
        moon_score_df = df[df['variable'] == 'moon_score']
        expression_zscore_df = df[df['variable'] == 'expression_zscore']
        
        # Smooth the moon score
        x = moon_score_df['conc'].values
        y = moon_score_df['value'].values
        smoothed = lowess(y, x, frac=0.5)
        x_smooth = smoothed[:, 0]
        y_smooth = smoothed[:, 1]
        
        # Smooth the expression z-score
        x_expression = expression_zscore_df['conc'].values
        y_expression = expression_zscore_df['value'].values
        smoothed_expression = lowess(y_expression, x_expression, frac=0.5)
        x_expression_smooth = smoothed_expression[:, 0]
        y_expression_smooth = smoothed_expression[:, 1]

        # Plot without labels (except for first iteration)
        label_moon = 'Smoothed Moon Score' if i == 0 else None
        label_expr = 'Smoothed Expression Z-Score' if i == 0 else None
        
        plt.plot(x_smooth, y_smooth, color=current_palette[2], alpha=0.2, 
                linewidth=2, label=label_moon)
        plt.plot(x_expression_smooth, y_expression_smooth, color=current_palette[0], 
                alpha=0.4, linewidth=2, label=label_expr)
    
    # Add labels and legend after the loop
    plt.xlabel('Concentration')
    plt.ylabel('Value')
    plt.title(f'BRAF Dabrafenib, {model}, remove {remove_edge_num} edge, 10 repeats')
    plt.legend()
    plt.savefig(path / Path("moon_score_plot.png"))
