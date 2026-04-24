"""
Analyze the distribution of agent turn/step counts in evaluation results,
broken down by label (SUPPORT, REFUTE, UNCERTAIN), to inform max_turns setting.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # repo root


def count_agent_calls(run_log_path: Path) -> int | None:
    """Count the number of 'Agent [call N]' entries in a run.log."""
    if not run_log_path.exists():
        return None
    text = run_log_path.read_text(errors="replace")
    return len(re.findall(r"Agent \[call \d+\]", text))


def load_dataset(csv_path: Path, label_col: str = "Agent_Verdict") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    steps = []
    for _, row in df.iterrows():
        out_dir = BASE_DIR / row["Output_Directory"]
        n = count_agent_calls(out_dir / "run.log")
        steps.append(n)
    df["steps"] = steps
    df = df[df["steps"].notna() & (df[label_col] != "FAIL")].copy()
    df["steps"] = df["steps"].astype(int)
    df["label"] = df[label_col].str.upper()
    return df


def print_stats(df: pd.DataFrame, name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total samples: {len(df)}")
    print(f"  Overall  — mean={df['steps'].mean():.1f}, "
          f"median={df['steps'].median():.0f}, "
          f"p90={df['steps'].quantile(0.9):.0f}, "
          f"p95={df['steps'].quantile(0.95):.0f}, "
          f"max={df['steps'].max()}")
    for label, g in df.groupby("label"):
        print(f"  {label:10s} (n={len(g):4d}) — "
              f"mean={g['steps'].mean():.1f}, "
              f"median={g['steps'].median():.0f}, "
              f"p90={g['steps'].quantile(0.9):.0f}, "
              f"p95={g['steps'].quantile(0.95):.0f}, "
              f"max={g['steps'].max()}")


def plot_distributions(cdb: pd.DataFrame, sig: pd.DataFrame, out_path: Path):
    labels = ["SUPPORT", "REFUTE", "UNCERTAIN"]
    colors = {"SUPPORT": "#2ecc71", "REFUTE": "#e74c3c", "UNCERTAIN": "#f39c12"}
    datasets = [("ConnectomeDB", cdb), ("SIGNOR Direct", sig)]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, len(labels) + 1, figure=fig, hspace=0.45, wspace=0.35)

    for row_i, (ds_name, df) in enumerate(datasets):
        # Overall histogram (first column)
        ax_all = fig.add_subplot(gs[row_i, 0])
        ax_all.hist(df["steps"], bins=range(1, df["steps"].max() + 2), color="#3498db",
                    edgecolor="white", linewidth=0.5)
        p90 = df["steps"].quantile(0.9)
        p95 = df["steps"].quantile(0.95)
        ax_all.axvline(p90, color="orange", linestyle="--", linewidth=1.5, label=f"p90={p90:.0f}")
        ax_all.axvline(p95, color="red", linestyle="--", linewidth=1.5, label=f"p95={p95:.0f}")
        ax_all.set_title(f"{ds_name}\nAll labels (n={len(df)})", fontsize=9, fontweight="bold")
        ax_all.set_xlabel("Steps (agent calls)")
        ax_all.set_ylabel("Count")
        ax_all.legend(fontsize=7)

        # Per-label histograms
        for col_i, label in enumerate(labels):
            ax = fig.add_subplot(gs[row_i, col_i + 1])
            sub = df[df["label"] == label]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{label} (n=0)", fontsize=9)
                continue
            max_steps = sub["steps"].max()
            ax.hist(sub["steps"], bins=range(1, max_steps + 2),
                    color=colors.get(label, "grey"), edgecolor="white", linewidth=0.5)
            p90 = sub["steps"].quantile(0.9)
            p95 = sub["steps"].quantile(0.95)
            ax.axvline(p90, color="orange", linestyle="--", linewidth=1.5, label=f"p90={p90:.0f}")
            ax.axvline(p95, color="red", linestyle="--", linewidth=1.5, label=f"p95={p95:.0f}")
            ax.set_title(f"{label} (n={len(sub)})", fontsize=9, fontweight="bold",
                         color=colors.get(label, "black"))
            ax.set_xlabel("Steps (agent calls)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

    fig.suptitle("Agent Step Count Distribution by Label\n(dashed lines: p90=orange, p95=red)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


def suggest_max_turns(df: pd.DataFrame, name: str):
    p95 = int(df["steps"].quantile(0.95)) + 1
    p90 = int(df["steps"].quantile(0.9)) + 1
    print(f"\n  [{name}] Suggested max_turns:")
    print(f"    Conservative (covers p95): {p95}")
    print(f"    Balanced    (covers p90): {p90}")


def main():
    cdb_csv = BASE_DIR / "results/connectomedb_eval_20260420_012842/results.csv"
    sig_csv = BASE_DIR / "results/signor_direct_eval_20260419_135117/results.csv"

    print("Loading ConnectomeDB results...")
    cdb = load_dataset(cdb_csv)
    print("Loading SIGNOR Direct results...")
    sig = load_dataset(sig_csv)

    print_stats(cdb, "ConnectomeDB  (connectomedb_eval_20260420_012842)")
    print_stats(sig, "SIGNOR Direct (signor_direct_eval_20260419_135117)")

    suggest_max_turns(cdb, "ConnectomeDB")
    suggest_max_turns(sig, "SIGNOR Direct")

    out_path = BASE_DIR / "experiments/step_distribution.png"
    plot_distributions(cdb, sig, out_path)


if __name__ == "__main__":
    main()
