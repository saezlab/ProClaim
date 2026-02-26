"""
Feature analysis for the Sufficiency Classifier training data.

Includes:
  1. Cohen's d (feature separability between y=0 and y=1)
  2. Pearson correlation matrix (multicollinearity detection)
  3. PCA (variance explained & biplot)

Usage:
  uv run scripts/sufficiency_classifier/analyze_classifier_features.py \
    --data_path data/classifier_train_data.json \
    --output_dir results/models/classifier/analysis
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_features(filepath: str):
    """Load classifier_train_data.json and return X, y, feature_names."""
    with open(filepath) as f:
        data = json.load(f)

    feature_names = list(data[0]["features"].keys())
    X = np.array([[d["features"].get(f, 0.0) for f in feature_names] for d in data])
    y = np.array([d["target_y"] for d in data])
    pool_types = [d.get("pool_type", "unknown") for d in data]

    logging.info(f"Loaded {len(X)} samples, {len(feature_names)} features")
    logging.info(f"Label distribution: y=1: {int(y.sum())}/{len(y)} ({y.mean()*100:.1f}%)")
    return X, y, feature_names, pool_types


# ──────────────────────────────────────────────
# 1. Cohen's d
# ──────────────────────────────────────────────
def cohens_d_analysis(X, y, feature_names, output_dir: Path):
    """Compute Cohen's d for each feature (y=0 vs y=1) and plot."""
    results = []
    for i, fname in enumerate(feature_names):
        vals_0 = X[y == 0, i]
        vals_1 = X[y == 1, i]
        pooled_std = np.sqrt((vals_0.std() ** 2 + vals_1.std() ** 2) / 2)
        d = abs(vals_0.mean() - vals_1.mean()) / pooled_std if pooled_std > 0 else 0.0
        results.append((fname, d, vals_0.mean(), vals_1.mean()))

    # Sort by descending d
    results.sort(key=lambda x: x[1], reverse=True)

    logging.info("\n=== Cohen's d (Feature Separability: y=0 vs y=1) ===")
    for fname, d, m0, m1 in results:
        tag = "⭐" if d >= 0.8 else ("▪" if d >= 0.5 else "·")
        logging.info(f"  {tag} {fname:40s}  d={d:.3f}  (y0={m0:.3f}, y1={m1:.3f})")

    # Bar plot
    names = [r[0] for r in results]
    ds = [r[1] for r in results]
    colors = ["#2ecc71" if d >= 0.8 else "#f39c12" if d >= 0.5 else "#e74c3c" for d in ds]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), ds, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Feature Separability (Cohen's d): y=0 vs y=1")
    ax.axvline(0.8, color="green", linestyle="--", alpha=0.5, label="Large (0.8)")
    ax.axvline(0.5, color="orange", linestyle="--", alpha=0.5, label="Medium (0.5)")
    ax.axvline(0.2, color="red", linestyle="--", alpha=0.5, label="Small (0.2)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / "cohens_d.png", dpi=150)
    plt.close(fig)
    logging.info(f"Saved Cohen's d plot to {output_dir / 'cohens_d.png'}")

    return results


# ──────────────────────────────────────────────
# 2. Correlation matrix (multicollinearity)
# ──────────────────────────────────────────────
def correlation_analysis(X, feature_names, output_dir: Path, threshold: float = 0.9):
    """Compute Pearson correlation matrix and flag highly correlated pairs."""
    corr = np.corrcoef(X.T)

    # Log pairs with |r| > threshold
    logging.info(f"\n=== Highly Correlated Feature Pairs (|r| > {threshold}) ===")
    pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = corr[i, j]
            if abs(r) > threshold:
                pairs.append((feature_names[i], feature_names[j], r))
                logging.info(f"  {feature_names[i]:40s} <-> {feature_names[j]:40s}  r={r:.3f}")

    if not pairs:
        logging.info("  (No pairs above threshold)")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask,
        xticklabels=feature_names, yticklabels=feature_names,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Pearson)")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    logging.info(f"Saved correlation heatmap to {output_dir / 'correlation_matrix.png'}")

    return pairs


# ──────────────────────────────────────────────
# 3. PCA
# ──────────────────────────────────────────────
def pca_analysis(X, y, feature_names, pool_types, output_dir: Path):
    """Run PCA: variance explained curve + 2D scatter biplot."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(len(feature_names), len(X))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Log variance explained
    logging.info("\n=== PCA Variance Explained ===")
    for i, (ev, cum) in enumerate(zip(explained, cumulative)):
        logging.info(f"  PC{i+1}: {ev*100:.2f}%  (cumulative: {cum*100:.2f}%)")
        if cum >= 0.99:
            break

    n_95 = np.argmax(cumulative >= 0.95) + 1
    n_90 = np.argmax(cumulative >= 0.90) + 1
    logging.info(f"  Components for 90% variance: {n_90}")
    logging.info(f"  Components for 95% variance: {n_95}")

    # --- Plot 1: Scree plot (variance explained) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(range(1, len(explained) + 1), explained * 100, alpha=0.7, label="Individual")
    ax.step(range(1, len(cumulative) + 1), cumulative * 100, where="mid", color="red", label="Cumulative")
    ax.axhline(95, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(90, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Scree Plot")
    ax.legend()

    # --- Plot 2: 2D scatter (PC1 vs PC2) colored by label ---
    ax = axes[1]
    colors_map = {0: "#e74c3c", 1: "#2ecc71"}
    for label in [0, 1]:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_map[label],
                   alpha=0.6, s=30, label=f"y={label}")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA: PC1 vs PC2 (by label)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "pca_overview.png", dpi=150)
    plt.close(fig)
    logging.info(f"Saved PCA overview to {output_dir / 'pca_overview.png'}")

    # --- Plot 3: Biplot (PC1 vs PC2 with feature loading arrows) ---
    fig, ax = plt.subplots(figsize=(10, 10))
    for label in [0, 1]:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_map[label],
                   alpha=0.4, s=20, label=f"y={label}")

    # Feature loading arrows
    loadings = pca.components_[:2].T  # shape (n_features, 2)
    scale = float(np.abs(X_pca[:, :2]).max()) / float(max(np.abs(loadings).max(), 1e-8)) * 0.8
    for i, fname in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                 head_width=0.15, head_length=0.1, fc="navy", ec="navy", alpha=0.6)
        ax.text(loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1,
                fname, fontsize=7, ha="center", color="navy")

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA Biplot (Feature Loadings)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "pca_biplot.png", dpi=150)
    plt.close(fig)
    logging.info(f"Saved PCA biplot to {output_dir / 'pca_biplot.png'}")

    # --- Plot 4: 2D scatter colored by pool_type ---
    fig, ax = plt.subplots(figsize=(8, 6))
    pool_colors = {
        "positive_support": "#2ecc71",
        "positive_contradict": "#27ae60",
        "negative_conflict": "#e74c3c",
        "negative_noise": "#e67e22",
    }
    for pt, color in pool_colors.items():
        mask = np.array([p == pt for p in pool_types])
        if mask.sum() > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color,
                       alpha=0.5, s=25, label=pt)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA: PC1 vs PC2 (by pool_type)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "pca_by_pool_type.png", dpi=150)
    plt.close(fig)
    logging.info(f"Saved PCA by pool_type to {output_dir / 'pca_by_pool_type.png'}")

    return pca, X_pca


def main():
    parser = argparse.ArgumentParser(description="Feature analysis for classifier training data")
    parser.add_argument("--data_path", type=str, default="data/classifier_train_data.json")
    parser.add_argument("--output_dir", type=str, default="results/models/classifier/analysis")
    parser.add_argument("--corr_threshold", type=float, default=0.9,
                        help="Threshold for flagging correlated feature pairs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names, pool_types = load_features(args.data_path)

    # 1. Cohen's d
    cohens_d_analysis(X, y, feature_names, output_dir)

    # 2. Correlation matrix
    correlation_analysis(X, feature_names, output_dir, threshold=args.corr_threshold)

    # 3. PCA
    pca_analysis(X, y, feature_names, pool_types, output_dir)

    logging.info(f"\nAll analysis outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
