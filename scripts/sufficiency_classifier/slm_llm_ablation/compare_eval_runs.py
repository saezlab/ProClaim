import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def extract_evaluation_data(results_dir: Path, model_name: str) -> pd.DataFrame:
    records = []
    
    # Iterate through all SIGNOR directories
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
                    "iterations_taken": final_iter
                }
                
                # Extract sufficiency score at each iteration
                # Note: max iterations is usually 4
                for i in range(4):
                    if i < len(history):
                        record[f"score_iter_{i+1}"] = history[i].get("confidence", 0.0)
                    else:
                        # If the iteration wasn't reached, it means it terminated early
                        # Typically we carry forward the last score or mark as NaN. Let's carry forward.
                        if len(history) > 0:
                            record[f"score_iter_{i+1}"] = history[-1].get("confidence", 0.0)
                        else:
                            record[f"score_iter_{i+1}"] = np.nan
                            
                records.append(record)
            except Exception as e:
                pass
                
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Compare dynamic evaluation runs (MLP vs LLM)")
    parser.add_argument("--mlp_dir", type=Path, default=Path(__file__).parents[3] / "results" / "signor_eval_20260330_205939")
    parser.add_argument("--llm_dir", type=Path, default=Path(__file__).parents[3] / "results" / "signor_eval_20260402_214154")
    parser.add_argument("--out", type=Path, default=Path(__file__).parents[3] / "results" / "slm_llm_ablation" / "signor_eval_ablation_plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("Extracting MLP data...")
    df_mlp = extract_evaluation_data(args.mlp_dir, "MLP")
    print("Extracting LLM data...")
    df_llm = extract_evaluation_data(args.llm_dir, "LLM")
    
    # Only keep overlapping claims
    if df_mlp.empty or df_llm.empty:
        print("No data found!")
        return
        
    df_mlp["claim_key"] = df_mlp["signor_id"] + "_" + df_mlp["flip"]
    df_llm["claim_key"] = df_llm["signor_id"] + "_" + df_llm["flip"]
    
    common_claims = set(df_mlp["claim_key"]).intersection(set(df_llm["claim_key"]))
    df_mlp = df_mlp[df_mlp["claim_key"].isin(common_claims)]
    df_llm = df_llm[df_llm["claim_key"].isin(common_claims)]
    
    print(f"Analyzing {len(common_claims)} shared claims.")
    
    df_all = pd.concat([df_mlp, df_llm], ignore_index=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Stopping Iteration Distribution (Bar Chart)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    iter_counts = df_all.groupby(["iterations_taken", "model"]).size().reset_index(name="count")
    total_per_model = df_all.groupby("model").size()
    iter_counts["percentage"] = iter_counts.apply(lambda row: (row["count"] / total_per_model[row["model"]]) * 100, axis=1)
    
    sns.barplot(data=iter_counts, x="iterations_taken", y="percentage", hue="model", palette={"MLP": "#FF9800", "LLM": "#2196F3"}, ax=ax1)
    ax1.set_xlabel("Iterations to Convergence")
    ax1.set_ylabel("Percentage of Claims (%)")
    ax1.set_title("Iteration Distribution: MLP vs LLM")
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
    fig1.tight_layout()
    out1 = args.out / "iterations_distribution.png"
    fig1.savefig(out1, dpi=args.dpi)
    print(f"Saved figure -> {out1}")
    
    # 2. Sufficiency Score Evolution (Line Plot)
    # Melt the dataframe so we have iteration matching
    melted = df_all.melt(id_vars=["claim_key", "model"], value_vars=["score_iter_1", "score_iter_2", "score_iter_3", "score_iter_4"], 
                         var_name="iteration", value_name="score")
    melted["iteration"] = melted["iteration"].str.extract(r'(\d+)').astype(int)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=melted, x="iteration", y="score", hue="model", marker="o", 
                 palette={"MLP": "#FF9800", "LLM": "#2196F3"}, err_style="bars", err_kws={'capsize': 5}, ax=ax2)
    ax2.set_xlabel("Search Iteration")
    ax2.set_ylabel("Average Sufficiency Score")
    ax2.set_title("Sufficiency Score Trajectory")
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_ylim(-0.05, 1.05)
    fig2.tight_layout()
    out2 = args.out / "sufficiency_trajectory.png"
    fig2.savefig(out2, dpi=args.dpi)
    print(f"Saved figure -> {out2}")
    
    # 3. Iteration Difference Profile (Violin Plot)
    merge_df = pd.merge(df_mlp[["claim_key", "iterations_taken"]], df_llm[["claim_key", "iterations_taken"]], on="claim_key", suffixes=("_mlp", "_llm"))
    merge_df["iteration_diff"] = merge_df["iterations_taken_mlp"] - merge_df["iterations_taken_llm"]
    
    # Statistical test
    t_stat, p_val = stats.ttest_rel(merge_df["iterations_taken_mlp"], merge_df["iterations_taken_llm"])
    print(f"Paired t-test for iterations: t={t_stat:.3f}, p={p_val:.3e}")
    print(f"MLP Mean Iterations: {merge_df['iterations_taken_mlp'].mean():.3f}")
    print(f"LLM Mean Iterations: {merge_df['iterations_taken_llm'].mean():.3f}")
    
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    sns.violinplot(data=merge_df, y="iteration_diff", inner="quartiles", color="#B0BEC5", cut=0, ax=ax3)
    sns.stripplot(data=merge_df, y="iteration_diff", color="black", alpha=0.3, jitter=True, size=4, ax=ax3)
    ax3.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax3.set_ylabel("Δ Iterations (MLP - LLM)")
    ax3.set_title(f"Per-Claim Iteration Difference\n(Positive means LLM converged faster)\nPaired t-test: p={p_val:.2e}")
    fig3.tight_layout()
    out3 = args.out / "iteration_difference_profile.png"
    fig3.savefig(out3, dpi=args.dpi)
    print(f"Saved figure -> {out3}")

    # 4. Sufficiency Score Trajectory with 95% Percentile Interval
    # Directly use 2.5th/97.5th percentiles of the raw data — no distributional
    # assumptions needed. Shows where 95% of individual scores fall.
    palette = {"MLP": "#FF9800", "LLM": "#2196F3"}
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    for model_name in ["MLP", "LLM"]:
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
        lo_err = means - np.array(ci_lo)   # distance below mean
        hi_err = np.array(ci_hi) - means   # distance above mean

        color = palette[model_name]
        ax4.plot(iters, means, marker="o", color=color, label=model_name, linewidth=2)
        ax4.errorbar(iters, means, yerr=[lo_err, hi_err],
                     fmt="none", color=color, capsize=5, linewidth=1.5)

    ax4.set_xlabel("Search Iteration")
    ax4.set_ylabel("Sufficiency Score")
    ax4.set_title("Sufficiency Score Trajectory (95% Percentile Interval)")
    ax4.set_xticks([1, 2, 3, 4])
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(title="model")
    fig4.tight_layout()
    out4 = args.out / "sufficiency_trajectory_95ci.png"
    fig4.savefig(out4, dpi=args.dpi)
    print(f"Saved figure -> {out4}")

if __name__ == "__main__":
    main()

