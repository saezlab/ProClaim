import pandas as pd
import numpy as np
import random
import ollama
from typing import Set, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("./rw_visualization.mplstyle")

def plot_violin_plot(data_dict, ax=None):
    data = pd.DataFrame.from_dict(data_dict)
    if ax is None:
        ax = plt.gca()
    return sns.violinplot(data=data, ax=ax, inner=None, density_norm='width', cut=0)

def plot_violin_plot_point(data_dict, ax=None):
    data = pd.DataFrame.from_dict(data_dict)
    if ax is None:
        ax = plt.gca()
    return sns.swarmplot(data=data, color='black', alpha=0.5, size=4, ax=ax)

def plot_mean_median_dots(data_dict, ax=None):
    """Add mean and median dots to violin plot"""
    data = pd.DataFrame.from_dict(data_dict)
    if ax is None:
        ax = plt.gca()
    # Calculate means and medians for each column
    means = data.mean()
    medians = data.median()
    
    # Get x-positions for each column
    x_positions = range(len(data.columns))
    
    # Plot mean dots (smaller, black)
    ax.scatter(x_positions, means, color='black', s=20, zorder=10, 
               marker='o', label='Mean')
    
    # Plot median dots (white with black edge)
    ax.scatter(x_positions, medians, color='white', s=40, zorder=10, 
               marker='o', label='Median', edgecolors='black', linewidth=1.5)
    
    return ax

def plot_complete_violin(data_dict, ax=None, show_points=True, show_stats=True, mean_size=20, median_size=40):
    """Complete violin plot with optional points and statistics"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot violin
    plot_violin_plot(data_dict, ax=ax)
    
    # Add swarm points if requested
    if show_points:
        plot_violin_plot_point(data_dict, ax=ax)
    
    # Add mean and median dots if requested
    if show_stats:
        plot_mean_median_dots(data_dict, ax=ax)
        ax.legend(scatterpoints=1, loc='upper right')
    
    return ax

if __name__ == "__main__":
    result_path = Path("./results/llm/recover_edge")
    max_layers = [2, 3]
    repeat = 30
    add_nums = [1, 2, 4, 8, 16]
    # max_layers = [3]
    # repeat = 10
    # add_nums = [1]
    model = 'qwen3:8b'.replace(':', '_')
    plt_path = result_path / Path(model) / Path("pr_violin_plot")
    if not plt_path.exists():
        plt_path.mkdir(parents=True, exist_ok=True)
    # precisions = []
    # recalls = []
    # f1s = []
    eval_precisions = {}
    eval_recalls = {}
    eval_f1s = {}
    for max_layer in max_layers:
        for add_num in add_nums:
            # plt.figure()
            precisions = []
            recalls = []
            f1s = []
            for i in range(repeat):
                file = result_path / Path(model) / Path(f"evaluation_results_add_num_{add_num}_max_layer_{max_layer}_{i}.json")
                with open(file, 'r') as f:
                    eval_dict = json.load(f)
                precisions.append(eval_dict['precision'])
                recalls.append(eval_dict['recall'])
                f1s.append(eval_dict['f1'])
            eval_precisions[add_num] = precisions
            eval_recalls[add_num] = recalls
            eval_f1s[add_num] = f1s
        # Precision plot
        plot_complete_violin(eval_precisions)
        plt.xlabel('Number of edges added')
        plt.ylabel('Precision')
        plt.savefig(plt_path / Path(f'precision_violin_plot_{model}_max_layer_{max_layer}_repeat_{repeat}.png'))
        # Recall plot
        plot_complete_violin(eval_recalls)
        plt.xlabel('Number of edges added')
        plt.ylabel('Recall')
        plt.savefig(plt_path / Path(f'recall_violin_plot_{model}_max_layer_{max_layer}_repeat_{repeat}.png'))
        # # F1 plot
        plot_complete_violin(eval_f1s)
        plt.xlabel('Number of edges added')
        plt.ylabel('F1 Score')
        plt.savefig(plt_path / Path(f'f1_violin_plot_{model}_max_layer_{max_layer}_repeat_{repeat}.png'))