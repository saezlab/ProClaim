"""
Visualise Analyses A and B for the sandbox-vs-wild motivation section.

Panel A compares settings against gold labels.
Panel B breaks changed verdicts into fixes vs. breaks.

Usage::

    uv run python scripts/analysis/plot_sandbox_vs_wild_gold_summary.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

LABELS = ["SUPPORT", "REFUTE", "UNCERTAIN"]
SETTING_COLORS = {
    "S1": "#4E79A7",
    "S2": "#E15759",
    "S3": "#59A14F",
}
TRANSITION_COLORS = {
    "fixed": "#59A14F",
    "broke": "#E15759",
    "stayed_wrong": "#9C755F",
}

DEFAULT_S1 = PROJECT_ROOT / "results/baselines/single_paper/anthropic--claude-sonnet-4-6/signor_seed100.jsonl"
DEFAULT_S2 = PROJECT_ROOT / "results/baselines/s2_retrieval/processed_s2_search/anthropic--claude-sonnet-4-6/top5/signor_seed100.jsonl"
DEFAULT_S3 = PROJECT_ROOT / "results/baselines/s2_plus_ref/anthropic--claude-sonnet-4-6/top5/signor_seed100.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/analysis/sandbox_vs_wild_gold_summary.pdf"


def load_jsonl(path: Path) -> dict[str, dict]:
    items = {}
    with open(path) as file:
        for line in file:
            row = json.loads(line)
            items[row["claim_id"]] = row
    return items


def compute_metrics(data: dict[str, dict]) -> dict:
    y_true = [row["gold_label"] for row in data.values()]
    y_pred = [row["predicted_label"] for row in data.values()]

    per_class = {}
    for label in LABELS:
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred) / len(y_true)
    macro_f1 = sum(values["f1"] for values in per_class.values()) / len(LABELS)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def compute_transition_breakdown(s1: dict[str, dict], s2: dict[str, dict]) -> tuple[Counter, dict[tuple[str, str], dict[str, int]]]:
    transitions = Counter()
    details: dict[tuple[str, str], dict[str, int]] = {}

    for claim_id, row1 in s1.items():
        row2 = s2[claim_id]
        verdict1 = row1["predicted_label"]
        verdict2 = row2["predicted_label"]
        gold = row1["gold_label"]
        key = (verdict1, verdict2)
        transitions[key] += 1

        if key not in details:
            details[key] = {"fixed": 0, "broke": 0, "stayed_wrong": 0}

        s1_correct = verdict1 == gold
        s2_correct = verdict2 == gold
        if not s1_correct and s2_correct:
            details[key]["fixed"] += 1
        elif s1_correct and not s2_correct:
            details[key]["broke"] += 1
        elif not s1_correct and not s2_correct:
            details[key]["stayed_wrong"] += 1

    return transitions, details


def draw_metric_panel(ax: plt.Axes, metrics: dict[str, dict]) -> None:
    metric_names = ["Accuracy", "Macro F1", "SUPPORT recall"]
    setting_names = ["S1", "S2", "S3"]
    values = np.array([
        [metrics[setting]["accuracy"] for setting in setting_names],
        [metrics[setting]["macro_f1"] for setting in setting_names],
        [metrics[setting]["per_class"]["SUPPORT"]["recall"] for setting in setting_names],
    ])

    x = np.arange(len(metric_names))
    width = 0.22

    for index, setting in enumerate(setting_names):
        bars = ax.bar(
            x + (index - 1) * width,
            values[:, index],
            width,
            label=setting,
            color=SETTING_COLORS[setting],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("A. Against gold labels", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_transition_panel(ax: plt.Axes, transitions: Counter, details: dict[tuple[str, str], dict[str, int]]) -> None:
    rows = [
        ("SUPPORT→REFUTE", ("SUPPORT", "REFUTE")),
        ("UNCERTAIN→REFUTE", ("UNCERTAIN", "REFUTE")),
        ("All changed", None),
    ]

    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    parts = {"fixed": [], "broke": [], "stayed_wrong": []}

    for _, key in rows:
        if key is None:
            changed_keys = [pair for pair in transitions if pair[0] != pair[1]]
            parts["fixed"].append(sum(details[pair]["fixed"] for pair in changed_keys))
            parts["broke"].append(sum(details[pair]["broke"] for pair in changed_keys))
            parts["stayed_wrong"].append(sum(details[pair]["stayed_wrong"] for pair in changed_keys))
        else:
            row = details[key]
            parts["fixed"].append(row["fixed"])
            parts["broke"].append(row["broke"])
            parts["stayed_wrong"].append(row["stayed_wrong"])

    for name in ["fixed", "broke", "stayed_wrong"]:
        values = np.array(parts[name])
        bars = ax.barh(
            y,
            values,
            left=left,
            color=TRANSITION_COLORS[name],
            edgecolor="white",
            linewidth=0.8,
            label=name.replace("_", " ").title(),
        )
        for bar, value in zip(bars, values):
            if value == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                str(int(value)),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 4 else "black",
                fontweight="bold",
            )
        left += values

    counts = [transitions[("SUPPORT", "REFUTE")], transitions[("UNCERTAIN", "REFUTE")], int(left[-1])]
    for index, total in enumerate(counts):
        ax.text(total + 0.6, y[index], f"n={total}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of claims", fontsize=9)
    ax.set_title("B. What changed when moving to S2", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    s1 = load_jsonl(DEFAULT_S1)
    s2 = load_jsonl(DEFAULT_S2)
    s3 = load_jsonl(DEFAULT_S3)

    metrics = {
        "S1": compute_metrics(s1),
        "S2": compute_metrics(s2),
        "S3": compute_metrics(s3),
    }
    transitions, details = compute_transition_breakdown(s1, s2)

    output_path = DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    draw_metric_panel(axes[0], metrics)
    draw_transition_panel(axes[1], transitions, details)

    fig.suptitle(
        "Why broader retrieval alone does not solve in-the-wild verification",
        fontsize=12,
        fontweight="bold",
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()