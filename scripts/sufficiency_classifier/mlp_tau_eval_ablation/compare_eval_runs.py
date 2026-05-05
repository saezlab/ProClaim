"""
Compare dynamic evaluation runs across different MLP sufficiency classifier tau thresholds.

Usage:
    uv run python scripts/sufficiency_classifier/mlp_tau_eval_ablation/compare_eval_runs.py \
        --tau000_dir results/signor_direct_eval_20260429_200315 \
        --tau025_dir results/signor_direct_eval_20260430_094006 \
        --tau050_dir results/signor_direct_eval_20260427_221617 \
        --out results/mlp_tau_eval_ablation/signor_direct_tau_ablation_plots

signor_direct_eval_20260501_154319, this is tau 0.5 with no web search
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, str] = {
    "SUPPORT": "SUPPORT", "REFUTE": "REFUTE", "UNCERTAIN": "UNCERTAIN",
    "SUPPORTED": "SUPPORT", "WRONG": "REFUTE",
    "CONTRADICT": "REFUTE", "SUPPORTS": "SUPPORT", "REFUTES": "REFUTE",
    "REFUTED": "REFUTE", "NEI": "UNCERTAIN",
    "NOT ENOUGH INFORMATION": "UNCERTAIN", "NOT_ENOUGH_INFORMATION": "UNCERTAIN",
    "ERROR": "UNCERTAIN", "FAIL": "UNCERTAIN",
}

def normalize_label(s: str) -> str:
    return LABEL_MAP.get(str(s).upper().strip(), "UNCERTAIN")


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def extract_accuracy_data(results_dir: Path, model_name: str) -> pd.DataFrame:
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        print(f"  [warn] No results.csv in {results_dir}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df = df[df["Repetition"] == 1].copy()

    df["gold"] = df["Flipped_Label"].apply(normalize_label)
    df["pred"] = df["Agent_Verdict"].apply(normalize_label)
    df["correct"] = df["gold"] == df["pred"]
    df["model"] = model_name
    df["claim_key"] = df["SIGNOR_ID"] + "_flip_" + df["Is_Flipped"].astype(str)
    df["cost_usd"] = pd.to_numeric(df.get("Cost_Estimate", 0), errors="coerce").fillna(0)

    return df[["claim_key", "model", "gold", "pred", "correct", "cost_usd"]]


def compute_metrics(df: pd.DataFrame) -> dict:
    from collections import defaultdict

    labels = ["SUPPORT", "REFUTE", "UNCERTAIN"]
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    total = len(df)
    correct = int(df["correct"].sum())

    for _, row in df.iterrows():
        pred = row["pred"]
        gold = row["gold"]
        if pred == gold:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    tn: dict[str, int] = {}
    for label in labels:
        tn[label] = total - tp[label] - fp[label] - fn[label]

    per_class: dict[str, dict] = {}
    f1_scores, fprs, fnrs = [], [], []
    for label in labels:
        prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        rec  = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr  = fp[label] / (fp[label] + tn[label]) if (fp[label] + tn[label]) > 0 else 0.0
        fnr  = fn[label] / (fn[label] + tp[label]) if (fn[label] + tp[label]) > 0 else 0.0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4),
                            "f1": round(f1, 4), "fpr": round(fpr, 4), "fnr": round(fnr, 4),
                            "tp": tp[label], "fp": fp[label], "fn": fn[label], "tn": tn[label]}
        f1_scores.append(f1)
        fprs.append(fpr)
        fnrs.append(fnr)

    return {
        "n": total,
        "accuracy": round(correct / total if total > 0 else 0.0, 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "macro_fpr": round(sum(fprs) / len(fprs), 4),
        "macro_fnr": round(sum(fnrs) / len(fnrs), 4),
        "avg_cost_usd": round(df["cost_usd"].mean(), 4),
        "per_class": per_class,
    }


def extract_evaluation_data(results_dir: Path, model_name: str) -> pd.DataFrame:
    records = []

    for signor_dir in results_dir.glob("SIGNOR-*"):
        signor_id = signor_dir.name

        for flip in ["flip_False", "flip_True"]:
            evidence_file = signor_dir / flip / "rep_1" / "workspace" / "evidence_state.json"
            if not evidence_file.exists():
                continue

            try:
                with open(evidence_file, "r") as f:
                    data = json.load(f)

                final_iter = data.get("iteration", 0)
                history = data.get("sufficiency_history", [])

                record = {
                    "signor_id": signor_id,
                    "flip": flip,
                    "model": model_name,
                    "iterations_taken": final_iter,
                }

                for i in range(4):
                    if i < len(history):
                        record[f"score_iter_{i+1}"] = history[i].get("confidence", 0.0)
                    else:
                        if len(history) > 0:
                            record[f"score_iter_{i+1}"] = history[-1].get("confidence", 0.0)
                        else:
                            record[f"score_iter_{i+1}"] = np.nan

                records.append(record)
            except Exception:
                pass

    return pd.DataFrame(records)


def _bold(val: float, best: float) -> str:
    s = f"{val:.2f}"
    return rf"\textbf{{{s}}}" if val == best else s


def write_latex_tables(
    metrics_by_model: dict,
    present_models: list,
    out_dir: Path,
    mean_iters: dict[str, float] | None = None,
) -> None:
    classes = ["SUPPORT", "REFUTE", "UNCERTAIN"]

    overall_keys  = ["accuracy", "macro_f1", "macro_fpr", "macro_fnr"]
    higher_better = {"accuracy", "macro_f1"}

    if mean_iters:
        overall_keys.append("mean_iters")
        for m in present_models:
            metrics_by_model[m]["mean_iters"] = mean_iters.get(m, float("nan"))

    best_overall: dict[str, float] = {}
    for k in overall_keys:
        vals = [metrics_by_model[m][k] for m in present_models]
        best_overall[k] = max(vals) if k in higher_better else min(vals)

    iter_col = r" & Mean Iter." if mean_iters else ""
    col_header = rf"Model & Accuracy & Macro F1 & Macro FPR & Macro FNR{iter_col} \\"
    n_t1_cols  = 4 + (1 if mean_iters else 0)
    col_spec_t1 = "l" + "r" * n_t1_cols

    rows_t1 = []
    for model in present_models:
        m = metrics_by_model[model]
        cells = [
            _bold(m["accuracy"],  best_overall["accuracy"]),
            _bold(m["macro_f1"],  best_overall["macro_f1"]),
            _bold(m["macro_fpr"], best_overall["macro_fpr"]),
            _bold(m["macro_fnr"], best_overall["macro_fnr"]),
        ]
        if mean_iters:
            cells.append(_bold(m["mean_iters"], best_overall["mean_iters"]))
        rows_t1.append(f"{model} & " + " & ".join(cells) + r" \\")

    table1 = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Agent performance across different MLP sufficiency classifier tau thresholds. Metrics include Accuracy, Macro-averaged F1, False Positive Rate (FPR), False Negative Rate (FNR), and Mean Iterations to convergence. Results are based on shared evaluation claims.}",
        r"\label{tab:tau-ablation-performance}",
        rf"\begin{{tabular}}{{{col_spec_t1}}}",
        r"\toprule",
        col_header,
        r"\midrule",
        *rows_t1,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    n_models   = len(present_models)
    metrics_t2 = ["precision", "recall", "f1"]
    higher_t2  = {"precision", "recall", "f1"}
    col_hdr_t2 = r"P & R & F1"

    col_spec_t2 = "l" + ("rrr" * n_models)
    model_multicolumns = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in present_models
    )
    sub_header = " & ".join(["Class"] + [col_hdr_t2] * n_models) + r" \\"
    cmidrule_str = " ".join(
        rf"\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}" for i in range(n_models)
    )

    rows_t2 = []
    for cls in classes:
        best: dict[str, float] = {}
        for metric in metrics_t2:
            vals = [metrics_by_model[m]["per_class"][cls][metric] for m in present_models]
            best[metric] = max(vals) if metric in higher_t2 else min(vals)

        cells = []
        for model in present_models:
            pc = metrics_by_model[model]["per_class"][cls]
            cells += [_bold(pc[metric], best[metric]) for metric in metrics_t2]
        rows_t2.append(rf"\textsc{{{cls.capitalize()}}} & " + " & ".join(cells) + r" \\")

    table2 = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-class precision, recall, and F1 metrics for each tau threshold.}",
        r"\label{tab:tau-ablation-per-class}",
        rf"\begin{{tabular}}{{{col_spec_t2}}}",
        r"\toprule",
        rf"& {model_multicolumns} \\",
        cmidrule_str,
        sub_header,
        r"\midrule",
        *rows_t2,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = out_dir / "accuracy_tables.tex"
    out_path.write_text(table1 + "\n\n" + table2 + "\n")
    print(f"Saved LaTeX tables -> {out_path}")


def main():
    results_root = Path(__file__).parents[3] / "results"

    parser = argparse.ArgumentParser(description="Compare dynamic evaluation runs across MLP tau thresholds")
    parser.add_argument("--tau000_dir", type=Path, default=results_root / "signor_direct_eval_20260429_200315")
    parser.add_argument("--tau025_dir", type=Path, default=results_root / "signor_direct_eval_20260430_094006")
    parser.add_argument("--tau050_dir", type=Path, default=results_root / "signor_direct_eval_20260427_221617")
    parser.add_argument("--out", type=Path, default=results_root / "mlp_tau_eval_ablation" / "signor_direct_tau_ablation_plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    tau000_label = r"$\tau=0.00$"
    tau025_label = r"$\tau=0.25$"
    tau050_label = r"$\tau=0.50$"

    print(f"Extracting {tau000_label} data...")
    df_000 = extract_evaluation_data(args.tau000_dir, tau000_label)
    print(f"Extracting {tau025_label} data...")
    df_025 = extract_evaluation_data(args.tau025_dir, tau025_label)
    print(f"Extracting {tau050_label} data...")
    df_050 = extract_evaluation_data(args.tau050_dir, tau050_label)

    if df_000.empty or df_025.empty or df_050.empty:
        print("No data found in one or more directories!")
        return

    for df in [df_000, df_025, df_050]:
        df["claim_key"] = df["signor_id"] + "_" + df["flip"]

    common_claims = (
        set(df_000["claim_key"])
        .intersection(set(df_025["claim_key"]))
        .intersection(set(df_050["claim_key"]))
    )
    df_000 = df_000[df_000["claim_key"].isin(common_claims)]
    df_025 = df_025[df_025["claim_key"].isin(common_claims)]
    df_050 = df_050[df_050["claim_key"].isin(common_claims)]

    print(f"Analyzing {len(common_claims)} shared claims.")

    df_all = pd.concat([df_000, df_025, df_050], ignore_index=True)

    palette = {
        tau000_label: "#E53935",
        tau025_label: "#FF9800",
        tau050_label: "#43A047",
    }
    model_order = [tau000_label, tau025_label, tau050_label]

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"]  = ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]

    def save_fig(fig, base_name):
        png_path = args.out / f"{base_name}.png"
        pdf_path = args.out / f"{base_name}.pdf"
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved figures -> {png_path}, {pdf_path}")

    # 1. Stopping Iteration Distribution (Bar Chart)
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    iter_counts = df_all.groupby(["iterations_taken", "model"]).size().reset_index(name="count")
    total_per_model = df_all.groupby("model").size()
    iter_counts["percentage"] = iter_counts.apply(
        lambda row: (row["count"] / total_per_model[row["model"]]) * 100, axis=1
    )

    sns.barplot(data=iter_counts, x="iterations_taken", y="percentage", hue="model",
                hue_order=model_order, palette=palette, ax=ax1)
    ax1.set_xlabel("Iterations to Converge")
    ax1.set_ylabel("Claims (%)")
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
    ax1.legend(title=None)
    fig1.tight_layout()
    save_fig(fig1, "iterations_distribution")

    # 2. Sufficiency Score Trajectory (Line Plot)
    melted = df_all.melt(
        id_vars=["claim_key", "model"],
        value_vars=["score_iter_1", "score_iter_2", "score_iter_3", "score_iter_4"],
        var_name="iteration", value_name="score",
    )
    melted["iteration"] = melted["iteration"].str.extract(r'(\d+)').astype(int)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=melted, x="iteration", y="score", hue="model", marker="o",
                 hue_order=model_order, palette=palette,
                 err_style="bars", err_kws={'capsize': 5}, ax=ax2)
    ax2.set_xlabel("Search Iteration")
    ax2.set_ylabel("Average Sufficiency Score")
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_ylim(-0.05, 1.05)
    fig2.tight_layout()
    save_fig(fig2, "sufficiency_trajectory")

    # 3. Iteration Difference Profile vs tau=0.50 baseline (Violin + Strip)
    merge_df = pd.merge(
        df_050[["claim_key", "iterations_taken"]],
        df_000[["claim_key", "iterations_taken"]],
        on="claim_key", suffixes=("_050", "_000"),
    )
    merge_df = pd.merge(
        merge_df,
        df_025[["claim_key", "iterations_taken"]].rename(columns={"iterations_taken": "iterations_taken_025"}),
        on="claim_key",
    )

    diff_label_000 = rf"{tau050_label} $-$ {tau000_label}"
    diff_label_025 = rf"{tau050_label} $-$ {tau025_label}"
    merge_df[diff_label_000] = merge_df["iterations_taken_050"] - merge_df["iterations_taken_000"]
    merge_df[diff_label_025] = merge_df["iterations_taken_050"] - merge_df["iterations_taken_025"]

    t_stat_000, p_val_000 = stats.ttest_rel(merge_df["iterations_taken_050"], merge_df["iterations_taken_000"])
    t_stat_025, p_val_025 = stats.ttest_rel(merge_df["iterations_taken_050"], merge_df["iterations_taken_025"])
    print(f"Paired t-test {tau050_label} vs {tau000_label}: t={t_stat_000:.3f}, p={p_val_000:.3e}")
    print(f"Paired t-test {tau050_label} vs {tau025_label}: t={t_stat_025:.3f}, p={p_val_025:.3e}")
    for label, col in [(tau000_label, "iterations_taken_000"), (tau025_label, "iterations_taken_025"), (tau050_label, "iterations_taken_050")]:
        print(f"{label} Mean Iterations: {merge_df[col].mean():.3f}")

    diff_melted = merge_df[[diff_label_000, diff_label_025]].melt(var_name="comparison", value_name="iteration_diff")

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    diff_palette = {diff_label_000: "#E53935", diff_label_025: "#FF9800"}
    sns.violinplot(data=diff_melted, x="comparison", y="iteration_diff", inner="quartiles",
                   hue="comparison", palette=diff_palette, legend=False, cut=0, ax=ax3)
    sns.stripplot(data=diff_melted, x="comparison", y="iteration_diff",
                  color="black", alpha=0.3, jitter=True, size=4, ax=ax3)
    ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax3.set_ylabel(r"$\Delta$ Iterations ($\tau=0.50$ $-$ other)")
    fig3.tight_layout()
    save_fig(fig3, "iteration_difference_profile")

    # 4. Sufficiency Score Trajectory with 95% Percentile Interval
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    for model_name in model_order:
        model_melted = melted[melted["model"] == model_name]
        iters, means, ci_lo, ci_hi = [], [], [], []
        for it in [1, 2, 3, 4]:
            vals = model_melted[model_melted["iteration"] == it]["score"].dropna().values
            iters.append(it)
            means.append(vals.mean())
            ci_lo.append(np.percentile(vals, 2.5))
            ci_hi.append(np.percentile(vals, 97.5))

        iters = np.array(iters)
        means = np.array(means)
        lo_err = means - np.array(ci_lo)
        hi_err = np.array(ci_hi) - means

        color = palette[model_name]
        ax4.plot(iters, means, marker="o", color=color, label=model_name, linewidth=2)
        ax4.errorbar(iters, means, yerr=[lo_err, hi_err],
                     fmt="none", color=color, capsize=5, linewidth=1.5)

    ax4.set_xlabel("Search Iteration")
    ax4.set_ylabel("Sufficiency Score")
    ax4.set_xticks([1, 2, 3, 4])
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(title=None)
    fig4.tight_layout()
    save_fig(fig4, "sufficiency_trajectory_95ci")

    # 5. Sufficiency Score Trajectory — Box Plot per iteration × tau
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted, x="iteration", y="score", hue="model",
                hue_order=model_order, palette=palette,
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
                linewidth=0.8, ax=ax5)
    ax5.set_xlabel("Search Iteration")
    ax5.set_ylabel("Sufficiency Score")
    ax5.set_ylim(-0.05, 1.05)
    fig5.tight_layout()
    save_fig(fig5, "sufficiency_trajectory_boxplot")

    # 6. Final converged score distribution
    def get_final_score(row):
        it = int(row["iterations_taken"])
        it = max(1, min(it, 4))
        return row[f"score_iter_{it}"]

    final_scores = df_all.copy()
    final_scores["final_score"] = final_scores.apply(get_final_score, axis=1)

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=final_scores, x="model", y="final_score", order=model_order,
                   hue="model", palette=palette, inner="quartile", cut=0, alpha=0.7, ax=ax6, legend=False)
    sns.stripplot(data=final_scores, x="model", y="final_score", order=model_order,
                  hue="model", palette=palette, alpha=0.4, jitter=True, size=4, ax=ax6, legend=False)
    ax6.set_xlabel("")
    ax6.set_ylabel("Sufficiency Score at Convergence")
    ax6.set_ylim(-0.05, 1.05)
    fig6.tight_layout()
    save_fig(fig6, "final_score_distribution")

    # Combined figure: (a) final score distribution  |  (b) iterations to converge
    fig_c, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))
    LABEL_FS  = 16
    TICK_FS   = 14
    LEGEND_FS = 14
    PANEL_FS  = 18

    sns.violinplot(data=final_scores, x="model", y="final_score", order=model_order,
                   hue="model", palette=palette, inner="quartile", cut=0, alpha=0.7, ax=ax_a, legend=False)
    sns.stripplot(data=final_scores, x="model", y="final_score", order=model_order,
                  hue="model", palette=palette, alpha=0.4, jitter=True, size=4, ax=ax_a, legend=False)
    ax_a.set_xlabel("", fontsize=LABEL_FS)
    ax_a.set_ylabel("Sufficiency Score at Convergence", fontsize=LABEL_FS)
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.tick_params(labelsize=TICK_FS)
    ax_a.text(-0.07, 0.98, "(a)", transform=ax_a.transAxes,
              fontsize=PANEL_FS, fontweight="bold", va="bottom", ha="right")

    sns.barplot(data=iter_counts, x="iterations_taken", y="percentage", hue="model",
                hue_order=model_order, palette=palette, ax=ax_b)
    ax_b.set_xlabel("Iterations to Converge", fontsize=LABEL_FS)
    ax_b.set_ylabel("Claims (%)", fontsize=LABEL_FS)
    ax_b.tick_params(labelsize=TICK_FS)
    ax_b.legend(title=None, fontsize=LEGEND_FS)
    ax_b.text(-0.07, 0.98, "(b)", transform=ax_b.transAxes,
              fontsize=PANEL_FS, fontweight="bold", va="bottom", ha="right")

    fig_c.tight_layout()
    save_fig(fig_c, "combined_score_iterations")

    # -----------------------------------------------------------------------
    # Accuracy / F1 comparison
    # -----------------------------------------------------------------------
    print("\nExtracting accuracy data from results.csv ...")
    df_acc_000 = extract_accuracy_data(args.tau000_dir, tau000_label)
    df_acc_025 = extract_accuracy_data(args.tau025_dir, tau025_label)
    df_acc_050 = extract_accuracy_data(args.tau050_dir, tau050_label)

    acc_dfs = {k: v for k, v in [
        (tau000_label, df_acc_000),
        (tau025_label, df_acc_025),
        (tau050_label, df_acc_050),
    ] if not v.empty}

    if len(acc_dfs) < 2:
        print("Not enough accuracy data to compare — skipping accuracy plots.")
        return

    common_acc = set.intersection(*[set(df["claim_key"]) for df in acc_dfs.values()])
    print(f"Accuracy analysis: {len(common_acc)} shared claims across {list(acc_dfs.keys())}.")

    metrics_by_model: dict[str, dict] = {}
    for model_name, df_acc in acc_dfs.items():
        df_sub = df_acc[df_acc["claim_key"].isin(common_acc)]
        metrics_by_model[model_name] = compute_metrics(df_sub)

    print("\n── MLP Tau Threshold Accuracy Comparison ──")
    header = f"{'Model':<15}  {'N':>5}  {'Acc':>6}  {'MacroF1':>8}  {'MacroFPR':>9}  {'MacroFNR':>9}  {'AvgCost':>8}"
    print(header)
    print("─" * len(header))
    for model_name in model_order:
        if model_name not in metrics_by_model:
            continue
        m = metrics_by_model[model_name]
        print(f"{model_name:<15}  {m['n']:>5}  {m['accuracy']:>6.4f}  {m['macro_f1']:>8.4f}"
              f"  {m['macro_fpr']:>9.4f}  {m['macro_fnr']:>9.4f}  {m['avg_cost_usd']:>8.4f}")
    print()

    present_models = [m for m in model_order if m in metrics_by_model]

    # 7. Grouped bar chart: Performance Metrics
    metric_keys   = ["accuracy", "macro_f1", "macro_fpr", "macro_fnr"]
    metric_labels = ["Accuracy", "Macro F1", "Macro FPR", "Macro FNR"]
    x = np.arange(len(metric_keys))
    width = 0.25
    offsets = np.linspace(-(len(present_models) - 1) / 2, (len(present_models) - 1) / 2, len(present_models)) * width

    fig7, ax7 = plt.subplots(figsize=(10, 5))
    for offset, model_name in zip(offsets, present_models):
        vals = [metrics_by_model[model_name][k] for k in metric_keys]
        bars = ax7.bar(x + offset, vals, width, label=model_name, color=palette[model_name], alpha=0.85)
        ax7.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax7.set_xticks(x)
    ax7.set_xticklabels(metric_labels)
    ax7.set_ylabel("Score")
    ax7.set_ylim(0, 1.15)
    ax7.legend(title=None)
    fig7.tight_layout()
    save_fig(fig7, "accuracy_metrics")

    # 8. Per-class F1 heatmap
    classes = ["SUPPORT", "REFUTE", "UNCERTAIN"]
    f1_matrix = np.array([
        [metrics_by_model[m]["per_class"][c]["f1"] for m in present_models]
        for c in classes
    ])
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    im = ax8.imshow(f1_matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax8.set_xticks(range(len(present_models)))
    ax8.set_xticklabels(present_models)
    ax8.set_yticks(range(len(classes)))
    ax8.set_yticklabels(classes)
    plt.colorbar(im, ax=ax8, label="F1 Score")
    for i, cls in enumerate(classes):
        for j, model_name in enumerate(present_models):
            val = f1_matrix[i, j]
            ax8.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=11, color="black" if val < 0.7 else "white")
    fig8.tight_layout()
    save_fig(fig8, "per_class_f1_heatmap")

    # 9. Quality-Efficiency Landscape (Mean Iterations vs Macro-F1)
    summary_data = []
    for model_name in present_models:
        m_acc = metrics_by_model[model_name]
        mean_iter = df_all[df_all["model"] == model_name]["iterations_taken"].mean()
        summary_data.append({
            "Model": model_name,
            "Mean Iterations": mean_iter,
            "Macro-F1": m_acc["macro_f1"],
            "Accuracy": m_acc["accuracy"],
        })
    df_summary = pd.DataFrame(summary_data)

    fig9, ax9 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_summary, x="Mean Iterations", y="Macro-F1", hue="Model",
                    style="Model", s=200, palette=palette, ax=ax9)

    for i in range(df_summary.shape[0]):
        ax9.text(df_summary["Mean Iterations"][i] + 0.02, df_summary["Macro-F1"][i],
                 f"{df_summary['Model'][i]}\n(Acc: {df_summary['Accuracy'][i]:.3f})",
                 fontsize=10, verticalalignment='center')

    ax9.set_xlabel("Mean Search Iterations (Efficiency)")
    ax9.set_ylabel("Macro-F1 Score (Quality)")
    ax9.set_xlim(df_summary["Mean Iterations"].min() - 0.2, df_summary["Mean Iterations"].max() + 0.6)
    ax9.set_ylim(df_summary["Macro-F1"].min() - 0.02, df_summary["Macro-F1"].max() + 0.05)
    ax9.get_legend().remove()
    fig9.tight_layout()
    save_fig(fig9, "synthesis_efficiency_vs_quality")

    # 10. Initial Confidence Bias (Score at Iteration 1)
    iter1_scores = melted[melted["iteration"] == 1].copy()

    fig10, ax10 = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=iter1_scores, x="model", y="score", order=model_order,
                   hue="model", palette=palette, inner="quartile", cut=0, ax=ax10, legend=False)
    sns.stripplot(data=iter1_scores, x="model", y="score", order=model_order,
                  hue="model", palette=palette, alpha=0.3, jitter=True, size=3, ax=ax10, legend=False)
    ax10.set_xlabel("")
    ax10.set_ylabel("Sufficiency Score at Iteration 1")
    ax10.set_ylim(-0.05, 1.05)
    fig10.tight_layout()
    save_fig(fig10, "synthesis_initial_confidence_bias")

    # LaTeX tables
    mean_iters = {
        row["Model"]: row["Mean Iterations"]
        for _, row in df_summary.iterrows()
        if row["Model"] in present_models
    }
    write_latex_tables(metrics_by_model, present_models, args.out, mean_iters=mean_iters)

    # 5b. Sufficiency Score Trajectory (Box Plot) — SUPPORT ground truth only
    gold_lookup = next(iter(acc_dfs.values()))[["claim_key", "gold"]].drop_duplicates("claim_key")
    melted_with_gold = melted.merge(gold_lookup, on="claim_key", how="inner")
    melted_supported = melted_with_gold[melted_with_gold["gold"] == "SUPPORT"]
    print(f"  sufficiency_trajectory_boxplot_supported_only: {len(melted_supported['claim_key'].unique())} claims")

    fig5b, ax5b = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted_supported, x="iteration", y="score", hue="model",
                hue_order=model_order, palette=palette,
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
                linewidth=0.8, ax=ax5b)
    ax5b.set_xlabel("Search Iteration")
    ax5b.set_ylabel("Sufficiency Score")
    ax5b.set_ylim(-0.05, 1.05)
    fig5b.tight_layout()
    save_fig(fig5b, "sufficiency_trajectory_boxplot_supported_only")

    # 5c. Per-claim spaghetti plot — SUPPORT ground truth only
    fig5c, axes5c = plt.subplots(1, len(model_order), figsize=(5 * len(model_order), 5), sharey=True)
    for ax, model_name in zip(axes5c, model_order):
        color = palette[model_name]
        model_data = melted_supported[melted_supported["model"] == model_name]
        for ck, grp in model_data.groupby("claim_key"):
            grp_sorted = grp.sort_values("iteration")
            ax.plot(grp_sorted["iteration"], grp_sorted["score"],
                    color=color, alpha=0.2, linewidth=0.8)
        mean_line = model_data.groupby("iteration")["score"].mean()
        ax.plot(mean_line.index, mean_line.values,
                color=color, linewidth=2.5, marker="o", label="Mean")
        ax.set_title(model_name)
        ax.set_xlabel("Search Iteration")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(-0.05, 1.05)
    axes5c[0].set_ylabel("Sufficiency Score")
    mean_handle = mlines.Line2D([], [], color="black", linewidth=2.5, marker="o", label="Mean")
    fig5c.legend(handles=[mean_handle], loc="lower right", frameon=True)
    fig5c.tight_layout()
    save_fig(fig5c, "sufficiency_trajectory_spaghetti_supported_only")


if __name__ == "__main__":
    main()
