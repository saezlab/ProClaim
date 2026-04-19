"""
Visualise verdict transitions between Setting 1 (single-paper, in-sandbox)
and Setting 2 (S2 retrieval top-5, in-the-wild) on SIGNOR-Fact.

The focus is on how verdicts *change* when additional evidence is introduced,
not on accuracy against ground truth.

Produces three panels:
  1. Sankey / alluvial diagram of verdict flows (Setting 1 → Setting 2)
  2. Grouped bar chart of verdict distributions per setting
  3. Transition heatmap with counts and percentages

Usage::

    uv run python scripts/analysis/plot_sandbox_vs_wild_transitions.py

    # Custom paths
    uv run python scripts/analysis/plot_sandbox_vs_wild_transitions.py \
        --setting1 results/baselines/single_paper/.../signor_seed100.jsonl \
        --setting2 results/baselines/s2_retrieval/.../signor_seed100.jsonl \
        --output figs/sandbox_vs_wild.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

LABELS = ["UNCERTAIN", "REFUTE", "SUPPORT"]
COLORS = {"SUPPORT": "#4CAF50", "REFUTE": "#E53935", "UNCERTAIN": "#91A0AF"}

DEFAULT_S1 = PROJECT_ROOT / "results/baselines/single_paper/anthropic--claude-sonnet-4-6/signor_seed100.jsonl"
DEFAULT_S2 = PROJECT_ROOT / "results/baselines/s2_retrieval/processed_s2_search/anthropic--claude-sonnet-4-6/top5/signor_seed100.jsonl"
DEFAULT_S3 = PROJECT_ROOT / "results/baselines/s2_plus_ref/anthropic--claude-sonnet-4-6/top5/signor_seed100.jsonl"


def load_jsonl(path: Path) -> dict[str, dict]:
    items = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            items[d["claim_id"]] = d
    return items


def compute_transitions(s1: dict, s2: dict) -> tuple[Counter, int]:
    transitions = Counter()
    for cid in s1:
        v1 = s1[cid]["predicted_label"]
        v2 = s2[cid]["predicted_label"]
        transitions[(v1, v2)] += 1
    return transitions, len(s1)


# ── Panel 1: Alluvial / Sankey ───────────────────────────────────────────────

def _draw_alluvial_pair(ax, transitions: Counter, n: int,
                        x_left: float, x_right: float,
                        left_positions: dict, right_positions: dict,
                        left_totals: dict, right_totals: dict,
                        bar_width: float,
                        show_left_labels: bool = True,
                        show_right_labels: bool = True):
    """Draw bars and flows between two columns of an alluvial diagram."""
    # Draw bars
    for l in LABELS:
        y0, h = left_positions[l]
        ax.barh(y0 + h / 2, bar_width, height=h, left=x_left - bar_width / 2,
                color=COLORS[l], edgecolor="white", linewidth=0.5, zorder=3)
        if show_left_labels:
            ax.text(x_left - bar_width / 2 - 0.02, y0 + h / 2,
                    f"{l}\n({left_totals[l]})", ha="right", va="center", fontsize=8, fontweight="bold")

        y0, h = right_positions[l]
        ax.barh(y0 + h / 2, bar_width, height=h, left=x_right - bar_width / 2,
                color=COLORS[l], edgecolor="white", linewidth=0.5, zorder=3)
        if show_right_labels:
            ax.text(x_right + bar_width / 2 + 0.02, y0 + h / 2,
                    f"{l}\n({right_totals[l]})", ha="left", va="center", fontsize=8, fontweight="bold")

    # Draw flows
    left_consumed = {l: 0.0 for l in LABELS}
    right_consumed = {l: 0.0 for l in LABELS}

    for src in LABELS:
        for dst in LABELS:
            count = transitions.get((src, dst), 0)
            if count == 0:
                continue
            frac = count / n

            sy0 = left_positions[src][0] + left_consumed[src]
            left_consumed[src] += frac

            dy0 = right_positions[dst][0] + right_consumed[dst]
            right_consumed[dst] += frac

            if src == dst:
                color = COLORS[src]
                alpha = 0.25
            else:
                color = COLORS[dst]
                alpha = 0.45

            xs = np.linspace(x_left + bar_width / 2, x_right - bar_width / 2, 100)
            t = (xs - xs[0]) / (xs[-1] - xs[0])
            curve = t * t * (3 - 2 * t)
            y_top = sy0 + frac + (dy0 + frac - sy0 - frac) * curve
            y_bot = sy0 + (dy0 - sy0) * curve

            ax.fill_between(xs, y_bot, y_top, color=color, alpha=alpha, zorder=2,
                            edgecolor="none")


def _compute_positions(totals: dict, n: int, gap: float = 0.02) -> dict:
    """Compute y positions for stacked bars given totals per label."""
    positions = {}
    y = 0
    for l in LABELS:
        h = totals[l] / n
        positions[l] = (y, h)
        y += h + gap
    return positions


def draw_alluvial(ax, transitions: Counter, n: int, *,
                  title: str = "Effect of context for verification",
                  left_label: str = "Limited context\n(reference abstract)",
                  right_label: str = "Context from 5\nretrieved abstracts"):
    """Draw a two-column alluvial diagram."""
    left_totals = {l: sum(transitions.get((l, r), 0) for r in LABELS) for l in LABELS}
    right_totals = {l: sum(transitions.get((src, l), 0) for src in LABELS) for l in LABELS}

    bar_width = 0.12
    gap = 0.02
    x_left, x_right = 0.0, 1.0

    left_positions = _compute_positions(left_totals, n, gap)
    right_positions = _compute_positions(right_totals, n, gap)

    _draw_alluvial_pair(ax, transitions, n, x_left, x_right,
                        left_positions, right_positions,
                        left_totals, right_totals, bar_width)

    ax.set_xlim(-0.35, 1.35)
    y_max = max(
        sum(h + gap for _, h in left_positions.values()),
        sum(h + gap for _, h in right_positions.values()),
    )
    ax.set_ylim(-0.03, y_max + 0.05)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(x_left, -0.025, left_label, ha="center", va="top", fontsize=8, style="italic")
    ax.text(x_right, -0.025, right_label, ha="center", va="top", fontsize=8, style="italic")
    ax.axis("off")


def draw_alluvial_three(ax, trans_12: Counter, trans_23: Counter, n: int):
    """Draw a three-column alluvial: Setting 1 → Setting 2 → Setting 3."""
    # Totals per column
    col1_totals = {l: sum(trans_12.get((l, r), 0) for r in LABELS) for l in LABELS}
    col2_totals = {l: sum(trans_12.get((src, l), 0) for src in LABELS) for l in LABELS}
    col3_totals = {l: sum(trans_23.get((src, l), 0) for src in LABELS) for l in LABELS}

    bar_width = 0.10
    gap = 0.02
    x1, x2, x3 = 0.0, 0.5, 1.0

    pos1 = _compute_positions(col1_totals, n, gap)
    pos2 = _compute_positions(col2_totals, n, gap)
    pos3 = _compute_positions(col3_totals, n, gap)

    # Draw Setting 1 → Setting 2
    _draw_alluvial_pair(ax, trans_12, n, x1, x2,
                        pos1, pos2,
                        col1_totals, col2_totals, bar_width,
                        show_left_labels=True, show_right_labels=False)
    # Draw count labels on centre column (between the two flow pairs)
    for l in LABELS:
        y0, h = pos2[l]
        ax.text(x2, y0 + h / 2,
                f"{l}\n({col2_totals[l]})", ha="center", va="center",
                fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Draw Setting 2 → Setting 3
    _draw_alluvial_pair(ax, trans_23, n, x2, x3,
                        pos2, pos3,
                        col2_totals, col3_totals, bar_width,
                        show_left_labels=False, show_right_labels=True)

    y_max = max(
        sum(h + gap for _, h in pos1.values()),
        sum(h + gap for _, h in pos2.values()),
        sum(h + gap for _, h in pos3.values()),
    )
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.06, y_max + 0.05)
    ax.set_title("Effect of context for verification", fontsize=10, fontweight="bold")
    ax.text(x1, -0.025, "Reference\nabstract only", ha="center", va="top", fontsize=8, style="italic")
    ax.text(x2, -0.025, "5 retrieved\nabstracts", ha="center", va="top", fontsize=8, style="italic")
    ax.text(x3, -0.025, "5 retrieved +\nreference abstract", ha="center", va="top", fontsize=8, style="italic")
    ax.axis("off")


# ── Panel 2: Grouped bar chart ──────────────────────────────────────────────

def draw_distribution_bars(ax, transitions: Counter, n: int):
    """Grouped bar chart comparing verdict distributions."""
    s1_counts = [sum(transitions.get((l, r), 0) for r in LABELS) for l in LABELS]
    s2_counts = [sum(transitions.get((src, l), 0) for src in LABELS) for l in LABELS]

    x = np.arange(len(LABELS))
    width = 0.32

    bars1 = ax.bar(x - width / 2, s1_counts, width, label="Setting 1 (source paper)",
                   color=[COLORS[l] for l in LABELS], edgecolor="white", linewidth=0.8, alpha=0.6)
    bars2 = ax.bar(x + width / 2, s2_counts, width, label="Setting 2 (S2 top-5)",
                   color=[COLORS[l] for l in LABELS], edgecolor="black", linewidth=0.8, alpha=1.0,
                   hatch="//")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

    # Delta annotations
    for i, l in enumerate(LABELS):
        delta = s2_counts[i] - s1_counts[i]
        sign = "+" if delta > 0 else ""
        color = "#2E7D32" if delta > 0 else "#C62828" if delta < 0 else "gray"
        y_pos = max(s1_counts[i], s2_counts[i]) + 6
        ax.text(x[i], y_pos, f"{sign}{delta}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("Number of claims", fontsize=9)
    ax.set_title("Verdict distribution shift", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, max(max(s1_counts), max(s2_counts)) + 15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel 3: Transition heatmap ──────────────────────────────────────────────

def draw_heatmap(ax, transitions: Counter, n: int):
    """Transition matrix as a heatmap."""
    matrix = np.array([[transitions.get((src, dst), 0) for dst in LABELS] for src in LABELS])

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    # Annotate cells
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            count = matrix[i, j]
            pct = count / n * 100
            text_color = "white" if count > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=9, fontweight="bold" if i != j and count > 0 else "normal",
                    color=text_color)

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("Setting 2 verdict (S2 top-5)", fontsize=9)
    ax.set_ylabel("Setting 1 verdict (source paper)", fontsize=9)
    ax.set_title("Verdict transition matrix", fontsize=10, fontweight="bold")

    # Highlight diagonal
    for i in range(len(LABELS)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="black", linewidth=2))

    # Summary stats as text below
    changed = sum(transitions[(s, d)] for s, d in transitions if s != d)
    unchanged = sum(transitions[(s, d)] for s, d in transitions if s == d)
    ax.text(1.0, 3.4, f"Unchanged: {unchanged}/{n} ({unchanged/n*100:.0f}%)  ·  "
            f"Changed: {changed}/{n} ({changed/n*100:.0f}%)",
            ha="center", va="top", fontsize=8, style="italic",
            transform=ax.transData)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise verdict transitions between sandbox and wild settings."
    )
    parser.add_argument("--setting1", type=Path, default=DEFAULT_S1,
                        help="Path to Setting 1 (single paper) JSONL results.")
    parser.add_argument("--setting2", type=Path, default=DEFAULT_S2,
                        help="Path to Setting 2 (S2 retrieval) JSONL results.")
    parser.add_argument("--setting3", type=Path, default=DEFAULT_S3,
                        help="Path to Setting 3 (S2 top-5 + reference paper) JSONL results.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for figures (default: results/analysis/)")
    args = parser.parse_args()

    s1 = load_jsonl(args.setting1)
    s2 = load_jsonl(args.setting2)
    transitions_12, n = compute_transitions(s1, s2)

    print(f"Loaded {n} claims. Verdicts changed (S1→S2) in "
          f"{sum(v for (a,b),v in transitions_12.items() if a != b)}/{n} claims.")

    # Load Setting 3 if available
    has_s3 = args.setting3.exists()
    if has_s3:
        s3 = load_jsonl(args.setting3)
        transitions_23, _ = compute_transitions(s2, s3)
        transitions_13, _ = compute_transitions(s1, s3)
        print(f"Verdicts changed (S2→S3) in "
              f"{sum(v for (a,b),v in transitions_23.items() if a != b)}/{n} claims.")
        print(f"Verdicts changed (S1→S3) in "
              f"{sum(v for (a,b),v in transitions_13.items() if a != b)}/{n} claims.")

    out_dir = (args.output or PROJECT_ROOT / "results/analysis")
    if args.output and args.output.suffix:
        out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1a: Two-column alluvial (Setting 1 → Setting 2)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    draw_alluvial(ax1, transitions_12, n)
    fig1.tight_layout()
    p1 = out_dir / "sandbox_vs_wild_alluvial.pdf"
    fig1.savefig(p1, bbox_inches="tight", dpi=200)
    fig1.savefig(p1.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {p1}")

    # Plot 1b: Three-column alluvial (Setting 1 → Setting 2 → Setting 3)
    if has_s3:
        fig1b, ax1b = plt.subplots(figsize=(10, 5))
        draw_alluvial_three(ax1b, transitions_12, transitions_23, n)
        fig1b.tight_layout()
        p1b = out_dir / "sandbox_vs_wild_alluvial_three.pdf"
        fig1b.savefig(p1b, bbox_inches="tight", dpi=200)
        fig1b.savefig(p1b.with_suffix(".png"), bbox_inches="tight", dpi=150)
        print(f"Saved {p1b}")

    # Plot 1c: Two-column alluvial (Setting 1 → Setting 3)
    if has_s3:
        fig1c, ax1c = plt.subplots(figsize=(7, 5))
        draw_alluvial(ax1c, transitions_13, n,
                      title="Reference only vs. retrieved + reference",
                      left_label="Reference\nabstract only",
                      right_label="5 retrieved +\nreference abstract")
        fig1c.tight_layout()
        p1c = out_dir / "sandbox_vs_s2plusref_alluvial.pdf"
        fig1c.savefig(p1c, bbox_inches="tight", dpi=200)
        fig1c.savefig(p1c.with_suffix(".png"), bbox_inches="tight", dpi=150)
        print(f"Saved {p1c}")

    # Plot 2: Distribution bars
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    draw_distribution_bars(ax2, transitions_12, n)
    fig2.tight_layout()
    p2 = out_dir / "sandbox_vs_wild_distribution.pdf"
    fig2.savefig(p2, bbox_inches="tight", dpi=200)
    fig2.savefig(p2.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {p2}")

    # Plot 3: Heatmap
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    draw_heatmap(ax3, transitions_12, n)
    fig3.tight_layout()
    p3 = out_dir / "sandbox_vs_wild_heatmap.pdf"
    fig3.savefig(p3, bbox_inches="tight", dpi=200)
    fig3.savefig(p3.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
