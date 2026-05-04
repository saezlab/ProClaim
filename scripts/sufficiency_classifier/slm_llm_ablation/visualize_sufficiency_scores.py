"""
Visualize and compare sufficiency scores from LLM, SLM, and MLP backends
across SIGNOR interactions in the slm_llm_ablation results.
uv run python scripts/sufficiency_classifier/slm_llm_ablation/visualize_sufficiency_scores.py
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BACKENDS = ["llm", "slm", "mlp"]
BACKEND_COLORS = {"llm": "#2196F3", "slm": "#4CAF50", "mlp": "#FF9800"}
BACKEND_LABELS = {"llm": "LLM (Claude Haiku)", "slm": "SLM (Qwen 3.5-9B)", "mlp": "MLP"}


def load_results(results_dir: Path) -> pd.DataFrame:
    """Recursively collect sufficiency scores from all JSON result files."""
    records = []

    for json_path in results_dir.rglob("*_sufficiency_result.json"):
        backend = json_path.stem.replace("_sufficiency_result", "")
        if backend not in BACKENDS:
            continue

        # Parse path from the right: workspace / rep_X / flip_X / SIGNOR-X / ...
        parts = json_path.parts
        # find the last "workspace" occurrence (the result workspace, not the project root)
        workspace_indices = [i for i, p in enumerate(parts) if p == "workspace"]
        if not workspace_indices:
            continue
        workspace_idx = workspace_indices[-1]

        try:
            rep_part    = parts[workspace_idx - 1]   # e.g. "rep_1"
            flip_part   = parts[workspace_idx - 2]   # e.g. "flip_False"
            signor_id   = parts[workspace_idx - 3]   # e.g. "SIGNOR-38079"
        except IndexError:
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)
            score = data.get("sufficient_support_score")
            if score is None:
                continue
            records.append({
                "signor_id": signor_id,
                "flip": flip_part.replace("flip_", "") == "True",
                "rep": int(rep_part.replace("rep_", "")),
                "backend": backend,
                "score": float(score),
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def plot_distributions(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Violin + strip plot of score distributions per backend."""
    plot_df = df[["backend", "score"]].copy()
    plot_df["Backend"] = plot_df["backend"].map(BACKEND_LABELS)

    order = [BACKEND_LABELS[b] for b in BACKENDS if b in df["backend"].unique()]
    palette = {BACKEND_LABELS[b]: BACKEND_COLORS[b] for b in BACKENDS}

    sns.violinplot(
        data=plot_df, x="Backend", y="score",
        order=order, palette=palette, hue="Backend", legend=False,
        inner=None, alpha=0.6, cut=0, ax=ax,
    )
    sns.stripplot(
        data=plot_df, x="Backend", y="score",
        order=order, palette=palette, hue="Backend", legend=False,
        size=4, alpha=0.5, jitter=True, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Sufficiency Score")
    ax.set_title("Sufficiency Score Distribution by Backend")
    ax.set_ylim(-0.05, 1.12)

    for i, backend in enumerate(BACKENDS):
        if backend not in df["backend"].unique():
            continue
        vals = df.loc[df["backend"] == backend, "score"]
        ax.text(i, 1.05, f"median={vals.median():.2f}\nn={len(vals)}", ha="center", va="bottom", fontsize=9)


def plot_lines(df: pd.DataFrame, ax: plt.Axes, backends: list[str]) -> None:
    """One line per backend; x = claim index (original order)."""
    wide = df.pivot_table(index=["signor_id", "flip"], columns="backend", values="score").reset_index()
    wide.columns.name = None
    wide = wide.reset_index(drop=True)

    x = range(len(wide))
    for backend in backends:
        if backend not in wide.columns:
            continue
        ax.plot(x, wide[backend], marker="o", markersize=4, label=BACKEND_LABELS[backend],
                color=BACKEND_COLORS[backend], linewidth=1.5, alpha=0.85)

    signor_nums = [sid.replace("SIGNOR-", "") for sid in wide["signor_id"]]
    ax.set_xlabel("SIGNOR ID")
    ax.set_ylabel("Sufficiency Score")
    ax.set_title("Per-Claim Sufficiency Scores: " + " vs ".join(BACKEND_LABELS[b] for b in backends if b in wide.columns))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(list(x))
    ax.set_xticklabels(signor_nums, fontsize=7, rotation=90)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def make_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide format: one row per (signor_id, flip, rep)."""
    wide = df.pivot_table(
        index=["signor_id", "flip", "rep"],
        columns="backend",
        values="score",
    ).reset_index()
    wide.columns.name = None
    # readable label for hover / annotation
    wide["label"] = wide["signor_id"] + "\nflip=" + wide["flip"].astype(str)
    return wide


def plot_scatter(wide: pd.DataFrame, x_backend: str, y_backend: str, ax: plt.Axes) -> None:
    """Scatter plot comparing two backends; each point = one claim, no flip delineation or index labels."""
    from scipy import stats

    if x_backend not in wide.columns or y_backend not in wide.columns:
        ax.set_visible(False)
        return

    data = wide.dropna(subset=[x_backend, y_backend])
    
    # Use a single neutral color instead of coloring by flip status
    ax.scatter(data[x_backend], data[y_backend], c="#607D8B", alpha=0.6, s=30, edgecolors="white", linewidths=0.4)

    # diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)

    r, p = stats.pearsonr(data[x_backend], data[y_backend])
    ax.set_xlabel(f"{BACKEND_LABELS[x_backend]} Score", fontsize=10)
    ax.set_ylabel(f"{BACKEND_LABELS[y_backend]} Score", fontsize=10)
    ax.set_title(
        f"{BACKEND_LABELS[x_backend]} vs {BACKEND_LABELS[y_backend]}\n"
        f"r={r:.3f}  p={p:.2e}  n={len(data)}",
        fontsize=10,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser(description="Visualize LLM/SLM/MLP sufficiency scores")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(__file__).parents[3] / "results" / "slm_llm_ablation",
        help="Root directory containing ablation results",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (default: <results_dir>/sufficiency_comparison.png)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_path = args.out or results_dir / "sufficiency_comparison.png"

    print(f"Loading results from {results_dir} ...")
    df = load_results(results_dir)
    if df.empty:
        raise RuntimeError(f"No result JSON files found under {results_dir}")

    wide = make_wide(df)

    print(f"  {df['signor_id'].nunique()} interactions, {df['backend'].nunique()} backends")
    print(df.groupby("backend")["score"].describe().round(3))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    plot_distributions(df, ax1)
    fig1.tight_layout()
    out1 = out_path.with_stem(out_path.stem + "_violin_combined")
    fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure → {out1}")

    fig_s1, ax_s1 = plt.subplots(figsize=(6, 6))
    plot_scatter(wide, "slm", "llm", ax_s1)
    out_s1 = out_path.with_stem(out_path.stem + "_scatter_slm_vs_llm")
    fig_s1.savefig(out_s1, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure → {out_s1}")

    fig_s2, ax_s2 = plt.subplots(figsize=(6, 6))
    plot_scatter(wide, "mlp", "slm", ax_s2)
    out_s2 = out_path.with_stem(out_path.stem + "_scatter_mlp_vs_slm")
    fig_s2.savefig(out_s2, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure → {out_s2}")
    
    fig_s3, ax_s3 = plt.subplots(figsize=(6, 6))
    plot_scatter(wide, "mlp", "llm", ax_s3)
    out_s3 = out_path.with_stem(out_path.stem + "_scatter_mlp_vs_llm")
    fig_s3.savefig(out_s3, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure → {out_s3}")


if __name__ == "__main__":
    main()
