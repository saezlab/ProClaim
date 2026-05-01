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
DEFAULT_S2 = PROJECT_ROOT / "results/baselines/retrieval/s2/anthropic--claude-sonnet-4-6/top5/signor_seed100.jsonl"
DEFAULT_S3 = PROJECT_ROOT / "results/baselines/s2_plus_ref/anthropic--claude-sonnet-4-6/signor_seed100.jsonl"


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


def compute_accuracy(data: dict[str, dict]) -> float:
    correct = sum(1 for row in data.values() if row["predicted_label"] == row["gold_label"])
    return correct / len(data)


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
            ax.text(x_left - bar_width / 2 - 0.012, y0 + h / 2,
                f"{l}\n({left_totals[l]})", ha="right", va="center", fontsize=15, fontweight="bold")

        y0, h = right_positions[l]
        ax.barh(y0 + h / 2, bar_width, height=h, left=x_right - bar_width / 2,
                color=COLORS[l], edgecolor="white", linewidth=0.5, zorder=3)
        if show_right_labels:
            ax.text(x_right + bar_width / 2 + 0.012, y0 + h / 2,
                f"{l}\n({right_totals[l]})", ha="left", va="center", fontsize=15, fontweight="bold")

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
                  right_label: str = "Context from 5\nretrieved abstracts",
                  left_accuracy: float | None = None,
                  right_accuracy: float | None = None):
    """Draw a two-column alluvial diagram."""
    left_totals = {l: sum(transitions.get((l, r), 0) for r in LABELS) for l in LABELS}
    right_totals = {l: sum(transitions.get((src, l), 0) for src in LABELS) for l in LABELS}

    bar_width = 0.12
    gap = 0.02
    x_left, x_right = 0.12, 0.88

    left_positions = _compute_positions(left_totals, n, gap)
    right_positions = _compute_positions(right_totals, n, gap)

    _draw_alluvial_pair(ax, transitions, n, x_left, x_right,
                        left_positions, right_positions,
                        left_totals, right_totals, bar_width)

    ax.set_xlim(0.0, 1.0)
    y_max = max(
        sum(h + gap for _, h in left_positions.values()),
        sum(h + gap for _, h in right_positions.values()),
    )
    ax.set_ylim(-0.06, y_max + 0.02)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
    if left_accuracy is not None:
        left_label = f"{left_label}\nAcc={left_accuracy:.3f}"
    if right_accuracy is not None:
        right_label = f"{right_label}\nAcc={right_accuracy:.3f}"
    ax.text(x_left, -0.03, left_label, ha="center", va="top", fontsize=15, style="italic")
    ax.text(x_right, -0.03, right_label, ha="center", va="top", fontsize=15, style="italic")
    ax.axis("off")


def draw_alluvial_three(ax, trans_12: Counter, trans_23: Counter, n: int,
                        acc1: float | None = None,
                        acc2: float | None = None,
                        acc3: float | None = None):
    """Draw a three-column alluvial: Setting 1 → Setting 2 → Setting 3."""
    # Totals per column
    col1_totals = {l: sum(trans_12.get((l, r), 0) for r in LABELS) for l in LABELS}
    col2_totals = {l: sum(trans_12.get((src, l), 0) for src in LABELS) for l in LABELS}
    col3_totals = {l: sum(trans_23.get((src, l), 0) for src in LABELS) for l in LABELS}

    bar_width = 0.10
    gap = 0.02
    x1, x2, x3 = 0.20, 0.50, 0.80

    pos1 = _compute_positions(col1_totals, n, gap)
    pos2 = _compute_positions(col2_totals, n, gap)
    pos3 = _compute_positions(col3_totals, n, gap)

    # Draw Setting 1 → Setting 2
    _draw_alluvial_pair(ax, trans_12, n, x1, x2,
                        pos1, pos2,
                        col1_totals, col2_totals, bar_width,
                        show_left_labels=True, show_right_labels=False)

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
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.06, y_max + 0.02)
    ax.set_title("Effect of evidence sources", fontsize=20, fontweight="bold", pad=8)
    label1 = "Reference\nabstract only"
    label2 = "5 retrieved\nabstracts"
    label3 = "5 retrieved +\nreference abstract"
    if acc1 is not None:
        label1 = f"{label1}\nAcc={acc1:.3f}"
    if acc2 is not None:
        label2 = f"{label2}\nAcc={acc2:.3f}"
    if acc3 is not None:
        label3 = f"{label3}\nAcc={acc3:.3f}"
    ax.text(x1, -0.03, label1, ha="center", va="top", fontsize=15, style="italic")
    ax.text(x2, -0.03, label2, ha="center", va="top", fontsize=15, style="italic")
    ax.text(x3, -0.03, label3, ha="center", va="top", fontsize=15, style="italic")
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


# ── Panel 4: Pie chart ───────────────────────────────────────────────────────

def draw_pie_chart(ax, data: dict[str, dict]):
    """Pie chart of ground-truth (gold) label distribution.

    Parameters
    ----------
    ax : matplotlib Axes
        Single axis to draw on.
    data : dict[str, dict]
        Loaded JSONL data (claim_id → row); uses ``gold_label``.
    """
    counts = Counter(row["gold_label"] for row in data.values())
    total = sum(counts.values())
    sizes = [counts.get(l, 0) for l in LABELS]
    colors = [COLORS[l] for l in LABELS]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=LABELS,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100))})",
        startangle=90,
        textprops={"fontsize": 11},
        pctdistance=0.65,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title("Ground-truth\nlabel distribution", fontsize=14, fontweight="bold", pad=10)


# ── Panel 5: Correction vs regression bar ────────────────────────────────────

def draw_correction_regression(ax, buckets: dict[str, list[dict]]):
    """Stacked / grouped bar showing S2-corrected vs S2-regressed claims.

    Bars are further broken down by the gold label of each claim so the
    reader can see *which* verdict classes gained or lost accuracy.
    """
    corrected = buckets["s2_corrected"]
    regressed = buckets["s2_regressed"]

    # Break down by gold label
    corr_by_gold = Counter(r["gold"] for r in corrected)
    regr_by_gold = Counter(r["gold"] for r in regressed)

    x = np.arange(len(LABELS))
    width = 0.35

    corr_vals = [corr_by_gold.get(l, 0) for l in LABELS]
    regr_vals = [regr_by_gold.get(l, 0) for l in LABELS]

    bars_c = ax.bar(x - width / 2, corr_vals, width,
                    color=[COLORS[l] for l in LABELS], edgecolor="white",
                    linewidth=0.8, alpha=0.85, label=f"Corrected ({len(corrected)})")
    bars_r = ax.bar(x + width / 2, regr_vals, width,
                    color=[COLORS[l] for l in LABELS], edgecolor="black",
                    linewidth=0.8, alpha=0.85, hatch="//",
                    label=f"Regressed ({len(regressed)})")

    # Value labels
    for bar in list(bars_c) + list(bars_r):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("Claims", fontsize=10)
    ax.set_title("Verdict changes", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    y_max = max(max(corr_vals, default=0), max(regr_vals, default=0))
    ax.set_ylim(0, y_max + 4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Changed-claim analysis ───────────────────────────────────────────────────

def find_changed_claims(
    s1: dict[str, dict],
    s2: dict[str, dict],
    s3: dict[str, dict],
) -> dict[str, list[dict]]:
    """Identify claims whose predicted label changed across the three settings.

    Returns a dict with the following keys:

    * ``"any_change"`` – label differs in at least one setting pair.
    * ``"s1_ne_s2"`` – Setting 1 → Setting 2 changed.
    * ``"s2_ne_s3"`` – Setting 2 → Setting 3 changed.
    * ``"s1_ne_s3"`` – Setting 1 → Setting 3 changed.
    * ``"all_differ"`` – all three settings produced different labels.
    * ``"s2_corrected"`` – S2 fixed an S1 error (S1 wrong, S2 matches gold).
    * ``"s2_regressed"`` – S2 broke an S1 correct (S1 matches gold, S2 wrong).
    * ``"s3_corrected"`` – S3 fixed an S1 error (S1 wrong, S3 matches gold).
    * ``"s3_regressed"`` – S3 broke an S1 correct (S1 matches gold, S3 wrong).

    Each list entry is a dict with ``claim_id``, ``gold``, ``s1``, ``s2``,
    ``s3``, and the original claim text (if present).
    """
    buckets: dict[str, list[dict]] = {
        "any_change": [],
        "s1_ne_s2": [],
        "s2_ne_s3": [],
        "s1_ne_s3": [],
        "all_differ": [],
        "s2_corrected": [],
        "s2_regressed": [],
        "s3_corrected": [],
        "s3_regressed": [],
    }

    for cid in s1:
        p1 = s1[cid]["predicted_label"]
        p2 = s2[cid]["predicted_label"]
        p3 = s3[cid]["predicted_label"]
        gold = s1[cid]["gold_label"]
        claim_text = s1[cid].get("claim", s1[cid].get("claim_text", ""))

        row = {
            "claim_id": cid,
            "gold": gold,
            "s1": p1,
            "s2": p2,
            "s3": p3,
            "claim": claim_text,
        }

        changed_12 = p1 != p2
        changed_23 = p2 != p3
        changed_13 = p1 != p3

        if not (changed_12 or changed_23 or changed_13):
            continue

        buckets["any_change"].append(row)

        if changed_12:
            buckets["s1_ne_s2"].append(row)
        if changed_23:
            buckets["s2_ne_s3"].append(row)
        if changed_13:
            buckets["s1_ne_s3"].append(row)
        if p1 != p2 and p2 != p3 and p1 != p3:
            buckets["all_differ"].append(row)

        # Correction / regression relative to gold
        if p1 != gold and p2 == gold:
            buckets["s2_corrected"].append(row)
        if p1 == gold and p2 != gold:
            buckets["s2_regressed"].append(row)
        if p1 != gold and p3 == gold:
            buckets["s3_corrected"].append(row)
        if p1 == gold and p3 != gold:
            buckets["s3_regressed"].append(row)

    return buckets


def save_changed_claims(buckets: dict[str, list[dict]], out_dir: Path) -> None:
    """Print a summary and save each bucket to a JSONL file."""
    print("\n── Changed-claim analysis ──")
    for name, rows in buckets.items():
        print(f"  {name:20s}: {len(rows)} claims")

    changed_dir = out_dir / "changed_claims"
    changed_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        path = changed_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"  → Saved to {changed_dir}/")


# ── Panel: Triple confusion-matrix (baseline + two delta matrices) ────────────

def _compute_cm(data: dict[str, dict]) -> np.ndarray:
    """Return a (len(LABELS) × len(LABELS)) confusion matrix.

    Rows = gold label, columns = predicted label.
    """
    idx = {l: i for i, l in enumerate(LABELS)}
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for row in data.values():
        g = row["gold_label"]
        p = row["predicted_label"]
        if g in idx and p in idx:
            mat[idx[g], idx[p]] += 1
    return mat


def _improvement_color(delta: int, is_diagonal: bool) -> str:
    """Return hex colour based on whether a cell delta is an improvement."""
    if delta == 0:
        return "#FFFFFF"
    good = (delta > 0) if is_diagonal else (delta < 0)
    if good:
        # green scale: light → dark depending on magnitude
        return "#C8E6C9" if abs(delta) == 1 else "#66BB6A" if abs(delta) <= 3 else "#2E7D32"
    else:
        return "#FFCDD2" if abs(delta) == 1 else "#EF9A9A" if abs(delta) <= 3 else "#C62828"


def draw_confusion_matrix_triple(
    axes,
    s1: dict[str, dict],
    s2: dict[str, dict],
    s3: dict[str, dict] | None,
    acc1: float,
    acc2: float,
    acc3: float | None,
    titles: tuple[str, str, str] = (
        "Reference abstract only",
        "5 retrieved abstracts",
        "5 retrieved +\nreference abstract",
    ),
):
    """Draw three confusion matrices side by side.

    * Left  : absolute counts for Setting 1 (baseline).
    * Centre: delta Setting 2 − Setting 1 with improvement-coded colours.
    * Right : delta Setting 3 − Setting 1 with improvement-coded colours.

    Parameters
    ----------
    axes : sequence of 3 Axes
    s1, s2, s3 : loaded JSONL dicts
    acc1, acc2, acc3 : accuracy values for subtitles
    titles : column titles
    """
    cm1 = _compute_cm(s1)
    cm2 = _compute_cm(s2)
    cm3 = _compute_cm(s3) if s3 is not None else None

    n = len(LABELS)
    subtitles = [
        f"{titles[0]}\nAcc = {acc1:.3f}",
        f"{titles[1]}\nAcc = {acc2:.3f}",
        f"{titles[2]}\nAcc = {acc3:.3f}" if acc3 is not None else titles[2],
    ]

    # ── Axis 0: baseline absolute ─────────────────────────────────────────────
    ax = axes[0]
    max_val = cm1.max() if cm1.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            val = cm1[i, j]
            bg = COLORS[LABELS[j]]
            intensity = 0.15 + 0.75 * (val / max_val)
            # blend white with the verdict colour
            import matplotlib.colors as mcolors
            base = np.array(mcolors.to_rgb(bg))
            cell_color = 1 - intensity * (1 - base)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=cell_color, edgecolor="white", linewidth=1.5))
            text_color = "white" if intensity > 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_color)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Gold", fontsize=11)
    ax.set_title(subtitles[0], fontsize=12, fontweight="bold", pad=6)

    # ── Axes 1 & 2: delta matrices ────────────────────────────────────────────
    for k, (ax, cm_new, subtitle) in enumerate(
        zip(axes[1:], [cm2, cm3], subtitles[1:])
    ):
        if cm_new is None:
            ax.axis("off")
            continue

        delta = cm_new - cm1

        for i in range(n):
            for j in range(n):
                d = int(delta[i, j])
                cell_color = _improvement_color(d, i == j)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=cell_color, edgecolor="white", linewidth=1.5))
                # Show base count + signed delta
                base_count = cm1[i, j]
                sign = "+" if d >= 0 else ""
                label = f"{base_count}\n({sign}{d})"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=10, fontweight="bold" if d != 0 else "normal",
                        color="black")

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(LABELS, fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(LABELS, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Gold", fontsize=11)
        ax.set_title(subtitle, fontsize=12, fontweight="bold", pad=6)

    # Shared legend for the delta panels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E7D32", label="Improvement"),
        Patch(facecolor="#C62828", label="Regression"),
        Patch(facecolor="#FFFFFF", edgecolor="gray", label="No change"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right",
                    fontsize=9, framealpha=0.85, title="vs. baseline",
                    title_fontsize=9)


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
    acc1 = compute_accuracy(s1)
    acc2 = compute_accuracy(s2)
    transitions_12, n = compute_transitions(s1, s2)

    print(f"Loaded {n} claims. Verdicts changed (S1→S2) in "
          f"{sum(v for (a,b),v in transitions_12.items() if a != b)}/{n} claims.")
    print(f"Accuracy (S1): {acc1:.3f}")
    print(f"Accuracy (S2): {acc2:.3f}")

    # Load Setting 3 if available
    has_s3 = args.setting3.exists()
    if has_s3:
        s3 = load_jsonl(args.setting3)
        acc3 = compute_accuracy(s3)
        transitions_23, _ = compute_transitions(s2, s3)
        transitions_13, _ = compute_transitions(s1, s3)
        print(f"Verdicts changed (S2→S3) in "
              f"{sum(v for (a,b),v in transitions_23.items() if a != b)}/{n} claims.")
        print(f"Verdicts changed (S1→S3) in "
              f"{sum(v for (a,b),v in transitions_13.items() if a != b)}/{n} claims.")
        print(f"Accuracy (S3): {acc3:.3f}")

        # Analyse claims that changed across the three settings
        buckets = find_changed_claims(s1, s2, s3)

    out_dir = (args.output or PROJECT_ROOT / "results/analysis")
    if args.output and args.output.suffix:
        out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_s3:
        save_changed_claims(buckets, out_dir)

    # Plot 1a: Two-column alluvial (Setting 1 → Setting 2)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    draw_alluvial(
        ax1,
        transitions_12,
        n,
        title=f"Effect of context for verification\nAccuracy: {acc1:.3f} → {acc2:.3f}",
        left_accuracy=acc1,
        right_accuracy=acc2,
    )
    fig1.tight_layout()
    p1 = out_dir / "sandbox_vs_wild_alluvial.pdf"
    fig1.savefig(p1, bbox_inches="tight", dpi=200)
    fig1.savefig(p1.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {p1}")

    # Plot 1b: Three-column alluvial + ground-truth pie + correction/regression
    if has_s3:
        fig1b = plt.figure(figsize=(15, 6.4))
        gs = fig1b.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1],
                                wspace=0.25, hspace=0.35)
        ax_alluvial = fig1b.add_subplot(gs[:, 0])   # left column, both rows
        ax_pie = fig1b.add_subplot(gs[0, 1])         # top-right
        ax_bar = fig1b.add_subplot(gs[1, 1])         # bottom-right
        draw_alluvial_three(ax_alluvial, transitions_12, transitions_23, n, acc1=acc1, acc2=acc2, acc3=acc3)
        draw_pie_chart(ax_pie, s1)
        draw_correction_regression(ax_bar, buckets)
        fig1b.tight_layout()
        p1b = out_dir / "sandbox_vs_wild_alluvial_three.pdf"
        fig1b.savefig(p1b, bbox_inches="tight", dpi=200)
        fig1b.savefig(p1b.with_suffix(".png"), bbox_inches="tight", dpi=150)
        print(f"Saved {p1b}")

    # Plot 1c: Two-column alluvial (Setting 1 → Setting 3)
    if has_s3:
        fig1c, ax1c = plt.subplots(figsize=(7, 5))
        draw_alluvial(ax1c, transitions_13, n,
                      title=f"Reference only vs. retrieved + reference\nAccuracy: {acc1:.3f} → {acc3:.3f}",
                      left_label="Reference\nabstract only",
                      right_label="5 retrieved +\nreference abstract",
                      left_accuracy=acc1,
                      right_accuracy=acc3)
        fig1c.tight_layout()
        p1c = out_dir / "sandbox_vs_s2plusref_alluvial.pdf"
        fig1c.savefig(p1c, bbox_inches="tight", dpi=200)
        fig1c.savefig(p1c.with_suffix(".png"), bbox_inches="tight", dpi=150)
        print(f"Saved {p1c}")

    # Plot 1d: Triple confusion-matrix (Figure 3 replacement)
    fig1d, axes_cm = plt.subplots(1, 3, figsize=(13, 4.2))
    draw_confusion_matrix_triple(
        axes_cm,
        s1,
        s2,
        s3 if has_s3 else None,
        acc1,
        acc2,
        acc3 if has_s3 else None,
    )
    fig1d.suptitle("Effect of evidence sources on SIGNOR-Fact",
                   fontsize=14, fontweight="bold", y=1.02)
    fig1d.tight_layout()
    p1d = out_dir / "sandbox_vs_wild_confusion_matrices.pdf"
    fig1d.savefig(p1d, bbox_inches="tight", dpi=200)
    fig1d.savefig(p1d.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {p1d}")

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