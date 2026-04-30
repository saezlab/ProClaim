"""
Analysis A & B: Compare sandbox vs. wild settings against gold labels.

Analysis A — Accuracy and per-class metrics for both settings against gold.
Analysis B — Verdict transition correctness: for each transition, was it a fix or a break?

Usage::

    uv run python scripts/analysis/sandbox_vs_wild_gold_analysis.py

    # With S3 included
    uv run python scripts/analysis/sandbox_vs_wild_gold_analysis.py --include-s3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

LABELS = ["SUPPORT", "REFUTE", "UNCERTAIN"]

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


def compute_metrics(data: dict[str, dict]) -> dict:
    """Compute accuracy, macro-F1, per-class P/R/F1/FPR/FNR."""
    y_true = [r["gold_label"] for r in data.values()]
    y_pred = [r["predicted_label"] for r in data.values()]

    per_class = {}
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != label and p != label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        per_class[label] = {
            "prec": prec, "rec": rec, "f1": f1,
            "fpr": fpr, "fnr": fnr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / len(LABELS)
    macro_fpr = sum(pc["fpr"] for pc in per_class.values()) / len(LABELS)
    macro_fnr = sum(pc["fnr"] for pc in per_class.values()) / len(LABELS)

    return {
        "n": n, "acc": acc,
        "macro_f1": macro_f1, "macro_fpr": macro_fpr, "macro_fnr": macro_fnr,
        "per_class": per_class,
        "gold_dist": Counter(y_true),
        "pred_dist": Counter(y_pred),
    }


def confusion_matrix(data: dict[str, dict]) -> np.ndarray:
    """Return confusion matrix (rows=gold, cols=predicted)."""
    y_true = [r["gold_label"] for r in data.values()]
    y_pred = [r["predicted_label"] for r in data.values()]
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[LABELS.index(t)][LABELS.index(p)] += 1
    return matrix


def transition_analysis(s1: dict, s2: dict) -> dict:
    """For each (v1→v2) transition, count how many are fixes, breaks, etc."""
    transitions = Counter()
    details = {}

    for cid in s1:
        v1 = s1[cid]["predicted_label"]
        v2 = s2[cid]["predicted_label"]
        gold = s1[cid]["gold_label"]

        key = (v1, v2)
        transitions[key] += 1

        if key not in details:
            details[key] = {
                "correct_to_correct": 0,
                "correct_to_wrong": 0,   # BROKE
                "wrong_to_correct": 0,   # FIXED
                "wrong_to_wrong": 0,
            }

        s1_ok = v1 == gold
        s2_ok = v2 == gold

        if s1_ok and s2_ok:
            details[key]["correct_to_correct"] += 1
        elif s1_ok and not s2_ok:
            details[key]["correct_to_wrong"] += 1
        elif not s1_ok and s2_ok:
            details[key]["wrong_to_correct"] += 1
        else:
            details[key]["wrong_to_wrong"] += 1

    return {"transitions": transitions, "details": details}


def print_analysis_a(name: str, metrics: dict):
    """Print Analysis A: metrics vs. gold."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  n = {metrics['n']}")
    print(f"  Gold distribution:      {dict(metrics['gold_dist'])}")
    print(f"  Predicted distribution:  {dict(metrics['pred_dist'])}")
    print(f"  Accuracy:    {metrics['acc']:.3f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.3f}")
    print(f"  Macro FPR:   {metrics['macro_fpr']:.3f}")
    print(f"  Macro FNR:   {metrics['macro_fnr']:.3f}")
    print()
    print(f"  {'Class':>12s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  {'FPR':>6s}  {'FNR':>6s}")
    print(f"  {'-' * 50}")
    for label in LABELS:
        c = metrics["per_class"][label]
        print(f"  {label:>12s}  {c['prec']:6.3f}  {c['rec']:6.3f}  {c['f1']:6.3f}  {c['fpr']:6.3f}  {c['fnr']:6.3f}")


def print_confusion(name: str, matrix: np.ndarray):
    """Print confusion matrix."""
    print(f"\n  Confusion matrix ({name}, rows=gold, cols=predicted):")
    print(f"  {'':>12s}  {'SUPPORT':>10s}  {'REFUTE':>10s}  {'UNCERTAIN':>10s}")
    for i, label in enumerate(LABELS):
        print(f"  {label:>12s}  {matrix[i][0]:>10d}  {matrix[i][1]:>10d}  {matrix[i][2]:>10d}")


def print_analysis_b(result: dict, n: int):
    """Print Analysis B: transition correctness."""
    transitions = result["transitions"]
    details = result["details"]

    print(f"\n{'=' * 60}")
    print(f"  Analysis B: Verdict Transition Correctness")
    print(f"{'=' * 60}")

    for src in LABELS:
        for dst in LABELS:
            key = (src, dst)
            count = transitions.get(key, 0)
            if count == 0:
                continue
            d = details[key]
            change_type = "STAY" if src == dst else "CHANGE"
            print(f"\n  {src} → {dst}: {count} claims [{change_type}]")
            print(f"    ✓→✓ (stayed correct):   {d['correct_to_correct']}")
            print(f"    ✓→✗ (broke):            {d['correct_to_wrong']}")
            print(f"    ✗→✓ (fixed):            {d['wrong_to_correct']}")
            print(f"    ✗→✗ (stayed wrong):     {d['wrong_to_wrong']}")

    # Summary
    changed_keys = {k for k in transitions if k[0] != k[1]}
    total_changed = sum(transitions[k] for k in changed_keys)
    total_broke = sum(details[k]["correct_to_wrong"] for k in changed_keys)
    total_fixed = sum(details[k]["wrong_to_correct"] for k in changed_keys)
    total_stayed_wrong = sum(details[k]["wrong_to_wrong"] for k in changed_keys)

    print(f"\n  {'─' * 50}")
    print(f"  Summary of {total_changed}/{n} changed verdicts:")
    print(f"    Broke (was correct, now wrong):       {total_broke}")
    print(f"    Fixed (was wrong, now correct):        {total_fixed}")
    print(f"    Stayed wrong (different wrong):        {total_stayed_wrong}")
    print(f"    Net accuracy change:                   {total_fixed - total_broke:+d} claims")
    print(f"  {'─' * 50}")

    # Key transitions
    sup_ref = ("SUPPORT", "REFUTE")
    unc_ref = ("UNCERTAIN", "REFUTE")
    if sup_ref in details:
        d = details[sup_ref]
        print(f"\n  Key transition: SUPPORT → REFUTE ({transitions[sup_ref]} claims)")
        print(f"    Broke: {d['correct_to_wrong']},  Fixed: {d['wrong_to_correct']}")
    if unc_ref in details:
        d = details[unc_ref]
        print(f"  Key transition: UNCERTAIN → REFUTE ({transitions[unc_ref]} claims)")
        print(f"    Broke: {d['correct_to_wrong']},  Fixed: {d['wrong_to_correct']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse sandbox vs. wild settings against gold labels."
    )
    parser.add_argument("--setting1", type=Path, default=DEFAULT_S1)
    parser.add_argument("--setting2", type=Path, default=DEFAULT_S2)
    parser.add_argument("--setting3", type=Path, default=DEFAULT_S3)
    parser.add_argument("--include-s3", action="store_true",
                        help="Include Setting 3 (S2 + reference) in the analysis.")
    args = parser.parse_args()

    s1 = load_jsonl(args.setting1)
    s2 = load_jsonl(args.setting2)

    # ── Analysis A ──
    print("\n" + "█" * 60)
    print("  ANALYSIS A: Settings vs. Gold Labels")
    print("█" * 60)

    m1 = compute_metrics(s1)
    m2 = compute_metrics(s2)

    print_analysis_a("S1: Sandbox (source paper only)", m1)
    print_confusion("S1", confusion_matrix(s1))

    print_analysis_a("S2: Naive retrieval (top-5 S2)", m2)
    print_confusion("S2", confusion_matrix(s2))

    if args.include_s3 and args.setting3.exists():
        s3 = load_jsonl(args.setting3)
        m3 = compute_metrics(s3)
        print_analysis_a("S3: S2 + reference paper", m3)
        print_confusion("S3", confusion_matrix(s3))

    # Key comparison
    print(f"\n  Accuracy: S1={m1['acc']:.3f} → S2={m2['acc']:.3f} (Δ={m2['acc'] - m1['acc']:+.3f})")
    print(f"  Macro F1: S1={m1['macro_f1']:.3f} → S2={m2['macro_f1']:.3f} (Δ={m2['macro_f1'] - m1['macro_f1']:+.3f})")
    s1_sup_rec = m1["per_class"]["SUPPORT"]["rec"]
    s2_sup_rec = m2["per_class"]["SUPPORT"]["rec"]
    print(f"  SUPPORT recall: {s1_sup_rec:.3f} → {s2_sup_rec:.3f} (Δ={s2_sup_rec - s1_sup_rec:+.3f}, "
          f"{abs(s2_sup_rec - s1_sup_rec) / s1_sup_rec * 100:.0f}% relative drop)")

    s2_false_refutes = sum(1 for r in s2.values()
                           if r["predicted_label"] == "REFUTE" and r["gold_label"] != "REFUTE")
    s2_total_refutes = sum(1 for r in s2.values() if r["predicted_label"] == "REFUTE")
    print(f"  S2 false REFUTE: {s2_false_refutes}/{s2_total_refutes} "
          f"({s2_false_refutes / s2_total_refutes * 100:.0f}% of REFUTE predictions are wrong)")

    # ── Analysis B ──
    print("\n" + "█" * 60)
    print("  ANALYSIS B: Verdict Transition Correctness (S1 → S2)")
    print("█" * 60)

    result = transition_analysis(s1, s2)
    print_analysis_b(result, m1["n"])


if __name__ == "__main__":
    main()
