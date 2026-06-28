#!/usr/bin/env python3
"""Analyze consistency results for Paper 3 (Consistency).

Loads the CSV produced by generate_synthetic.py or run_consistency.py and
generates:
  1. Bar chart:   entropy by condition, grouped by domain
  2. Line plot:   self-consistency vs team size (cost on secondary y-axis)
  3. Scatter:     flip_rate seed vs paraphrase variation
  4. Heatmap:     improvement over single-LLM by (topology_family, scale)

Also runs statistical tests (paired t-test + bootstrap) for team vs single-LLM
and prints LaTeX-ready summary tables.

Usage:
    python analyze_results.py                                  # defaults
    python analyze_results.py --input ../data/consistency_synthetic.csv
    python analyze_results.py --input ../data/consistency_synthetic.csv --fig_dir ../figures/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {path}")


def _condition_label(row: pd.Series) -> str:
    """Human-readable label for a condition + scale combination."""
    cond = row["condition"]
    n = row["n_agents"]
    if cond == "single_llm":
        return "Single LLM"
    elif cond == "ensemble":
        return f"Ensemble (n={n})"
    else:
        return f"Team (n={n})"


# ---------------------------------------------------------------------------
# Figure 1: Bar chart -- entropy by condition, grouped by domain
# ---------------------------------------------------------------------------

def fig_entropy_by_condition(df: pd.DataFrame, fig_dir: Path) -> None:
    """Bar chart of answer entropy across conditions, grouped by domain."""
    # Aggregate: mean entropy per (domain, condition, n_agents)
    agg = (
        df.groupby(["domain", "condition", "n_agents"])["answer_entropy"]
        .mean()
        .reset_index()
    )
    agg["label"] = agg.apply(_condition_label, axis=1)

    # Canonical ordering
    order = ["Single LLM"]
    for n in sorted(df[df["condition"] == "ensemble"]["n_agents"].unique()):
        order.append(f"Ensemble (n={n})")
    for n in sorted(df[df["condition"] == "team"]["n_agents"].unique()):
        order.append(f"Team (n={n})")
    # Keep only labels that exist
    order = [o for o in order if o in agg["label"].values]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=agg, x="label", y="answer_entropy", hue="domain",
        order=order, ax=ax, palette="Set2", edgecolor="black", linewidth=0.5,
    )
    ax.set_xlabel("Condition")
    ax.set_ylabel("Answer Entropy (nats)")
    ax.set_title("Answer Entropy by Condition and Domain")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Domain")
    sns.despine()
    _save_fig(fig, fig_dir / "entropy_by_condition.pdf")
    _save_fig(fig, fig_dir / "entropy_by_condition.png")


# ---------------------------------------------------------------------------
# Figure 2: Line plot -- self-consistency vs team size (stability-vs-cost)
# ---------------------------------------------------------------------------

def fig_consistency_vs_size(df: pd.DataFrame, fig_dir: Path) -> None:
    """Self-consistency vs team size with cost on secondary y-axis."""
    teams = df[df["condition"] == "team"].copy()
    if teams.empty:
        print("  [WARN] No team data for consistency-vs-size plot; skipping.")
        return

    agg = (
        teams.groupby("n_agents")
        .agg(
            sc_mean=("self_consistency", "mean"),
            sc_std=("self_consistency", "std"),
            cost_mean=("total_cost", "mean"),
        )
        .reset_index()
    )

    # Add single-LLM baseline
    single = df[df["condition"] == "single_llm"]
    baseline_sc = single["self_consistency"].mean()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    color_sc = "#2c7bb6"
    color_cost = "#d7191c"

    # Self-consistency line
    ax1.errorbar(
        agg["n_agents"], agg["sc_mean"], yerr=agg["sc_std"],
        marker="o", color=color_sc, linewidth=2, capsize=4, label="Self-consistency",
    )
    ax1.axhline(baseline_sc, color=color_sc, linestyle="--", alpha=0.5,
                label=f"Single-LLM baseline ({baseline_sc:.2f})")
    ax1.set_xlabel("Team Size (n agents)")
    ax1.set_ylabel("Self-Consistency (modal share)", color=color_sc)
    ax1.tick_params(axis="y", labelcolor=color_sc)
    ax1.set_ylim(0.4, 1.0)

    # Cost on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        agg["n_agents"], agg["cost_mean"],
        marker="s", color=color_cost, linewidth=2, linestyle="--", label="Mean cost (USD)",
    )
    ax2.set_ylabel("Mean Cost per Prompt (USD)", color=color_cost)
    ax2.tick_params(axis="y", labelcolor=color_cost)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Stability-vs-Cost Law: Self-Consistency and Cost by Team Size")
    fig.tight_layout()
    _save_fig(fig, fig_dir / "consistency_vs_size.pdf")
    _save_fig(fig, fig_dir / "consistency_vs_size.png")


# ---------------------------------------------------------------------------
# Figure 3: Scatter -- flip_rate seed vs paraphrase
# ---------------------------------------------------------------------------

def fig_flip_rate_scatter(df: pd.DataFrame, fig_dir: Path) -> None:
    """Scatter plot comparing flip rates under seed vs paraphrase variation."""
    seed_df = (
        df[df["variation_type"] == "seed"]
        .groupby(["domain", "condition", "n_agents", "prompt_id"])["flip_rate"]
        .mean()
        .reset_index()
        .rename(columns={"flip_rate": "flip_seed"})
    )
    para_df = (
        df[df["variation_type"] == "paraphrase"]
        .groupby(["domain", "condition", "n_agents", "prompt_id"])["flip_rate"]
        .mean()
        .reset_index()
        .rename(columns={"flip_rate": "flip_para"})
    )
    merged = seed_df.merge(para_df, on=["domain", "condition", "n_agents", "prompt_id"])

    fig, ax = plt.subplots(figsize=(6, 6))
    palette = {"single_llm": "#e41a1c", "ensemble": "#377eb8", "team": "#4daf4a"}
    for cond, color in palette.items():
        subset = merged[merged["condition"] == cond]
        ax.scatter(
            subset["flip_seed"], subset["flip_para"],
            alpha=0.3, s=15, color=color, label=cond.replace("_", " ").title(),
        )

    # Diagonal reference line
    lims = [0, max(merged["flip_seed"].max(), merged["flip_para"].max()) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Flip Rate (seed variation)")
    ax.set_ylabel("Flip Rate (paraphrase variation)")
    ax.set_title("Flip Rate: Seed vs. Paraphrase Variation")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    sns.despine()
    _save_fig(fig, fig_dir / "flip_rate_scatter.pdf")
    _save_fig(fig, fig_dir / "flip_rate_scatter.png")


# ---------------------------------------------------------------------------
# Figure 4: Heatmap -- improvement over single-LLM by (topology, scale)
# ---------------------------------------------------------------------------

def fig_improvement_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """Heatmap of self-consistency improvement over single-LLM baseline."""
    single_baseline = (
        df[df["condition"] == "single_llm"]
        .groupby("domain")["self_consistency"]
        .mean()
    )

    teams = df[df["condition"] == "team"].copy()
    if teams.empty:
        print("  [WARN] No team data for improvement heatmap; skipping.")
        return

    team_agg = (
        teams.groupby(["topology_family", "n_agents", "domain"])["self_consistency"]
        .mean()
        .reset_index()
    )
    # Compute improvement (percentage points)
    team_agg["improvement"] = team_agg.apply(
        lambda r: r["self_consistency"] - single_baseline.get(r["domain"], 0), axis=1
    )

    # Average improvement across domains
    pivot_data = (
        team_agg.groupby(["topology_family", "n_agents"])["improvement"]
        .mean()
        .reset_index()
    )
    pivot = pivot_data.pivot(index="topology_family", columns="n_agents", values="improvement")
    pivot = pivot.reindex(["chain", "star", "hierarchy", "full"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGn", linewidths=0.5,
        ax=ax, cbar_kws={"label": "Self-consistency improvement (pp)"},
    )
    ax.set_xlabel("Team Size (n agents)")
    ax.set_ylabel("Topology Family")
    ax.set_title("Self-Consistency Improvement over Single LLM\n(by Topology and Scale)")
    fig.tight_layout()
    _save_fig(fig, fig_dir / "improvement_heatmap.pdf")
    _save_fig(fig, fig_dir / "improvement_heatmap.png")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42
) -> dict:
    """Bootstrap confidence interval for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.integers(0, len(a), size=len(a))
        idx_b = rng.integers(0, len(b), size=len(b))
        diffs[i] = a[idx_a].mean() - b[idx_b].mean()
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    return {"mean_diff": diffs.mean(), "ci_lo": lo, "ci_hi": hi, "p_value_approx": (diffs <= 0).mean()}


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests and bootstrap for team vs single-LLM on self-consistency."""
    single = df[df["condition"] == "single_llm"]
    teams = df[df["condition"] == "team"]

    # Align by (domain, prompt_id, variation_type)
    single_agg = (
        single.groupby(["domain", "prompt_id", "variation_type"])["self_consistency"]
        .mean()
        .reset_index()
        .rename(columns={"self_consistency": "sc_single"})
    )

    results = []
    for n_ag in sorted(teams["n_agents"].unique()):
        team_sub = teams[teams["n_agents"] == n_ag]
        team_agg = (
            team_sub.groupby(["domain", "prompt_id", "variation_type"])["self_consistency"]
            .mean()
            .reset_index()
            .rename(columns={"self_consistency": "sc_team"})
        )
        merged = single_agg.merge(team_agg, on=["domain", "prompt_id", "variation_type"])
        if merged.empty:
            continue

        a = merged["sc_team"].values
        b = merged["sc_single"].values

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(a, b)
        # Bootstrap
        boot = bootstrap_ci(a, b)

        results.append({
            "n_agents": n_ag,
            "mean_sc_team": a.mean(),
            "mean_sc_single": b.mean(),
            "mean_diff": a.mean() - b.mean(),
            "t_stat": t_stat,
            "p_value_ttest": p_val,
            "boot_ci_lo": boot["ci_lo"],
            "boot_ci_hi": boot["ci_hi"],
            "boot_p_approx": boot["p_value_approx"],
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def print_latex_summary(df: pd.DataFrame) -> None:
    """Print LaTeX-ready summary tables to stdout."""
    print("\n" + "=" * 70)
    print("LATEX TABLE: Mean metrics by condition")
    print("=" * 70)

    agg = (
        df.groupby(["condition", "n_agents"])[
            ["answer_entropy", "flip_rate", "self_consistency", "total_cost"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    # Flatten multi-level columns
    agg.columns = ["_".join(col).strip("_") for col in agg.columns]
    agg = agg.reset_index()

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Consistency metrics by condition and scale (mean $\pm$ std).}")
    print(r"\label{tab:consistency_summary}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"Condition & $n$ & Entropy & Flip Rate & Self-Consistency & Cost (USD) \\")
    print(r"\midrule")

    for _, row in agg.iterrows():
        cond = row["condition"].replace("_", r"\_")
        n = int(row["n_agents"])
        ent = f"${row['answer_entropy_mean']:.3f} \\pm {row['answer_entropy_std']:.3f}$"
        fr = f"${row['flip_rate_mean']:.3f} \\pm {row['flip_rate_std']:.3f}$"
        sc = f"${row['self_consistency_mean']:.3f} \\pm {row['self_consistency_std']:.3f}$"
        cost = f"${row['total_cost_mean']:.4f} \\pm {row['total_cost_std']:.4f}$"
        print(f"{cond} & {n} & {ent} & {fr} & {sc} & {cost} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print()


def print_latex_tests(test_df: pd.DataFrame) -> None:
    """Print LaTeX table of statistical test results."""
    print("\n" + "=" * 70)
    print("LATEX TABLE: Statistical tests (team vs single-LLM)")
    print("=" * 70)

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Paired t-test and bootstrap CI for self-consistency improvement (team vs.\ single LLM).}")
    print(r"\label{tab:statistical_tests}")
    print(r"\begin{tabular}{ccccccc}")
    print(r"\toprule")
    print(r"$n$ & $\overline{\Delta}$ & $t$ & $p$ (t-test) & Boot 95\% CI & Boot $p$ \\")
    print(r"\midrule")

    for _, row in test_df.iterrows():
        n = int(row["n_agents"])
        diff = f"{row['mean_diff']:.4f}"
        t = f"{row['t_stat']:.2f}"
        p_t = f"{row['p_value_ttest']:.2e}" if row["p_value_ttest"] < 0.001 else f"{row['p_value_ttest']:.4f}"
        ci = f"[{row['boot_ci_lo']:.4f}, {row['boot_ci_hi']:.4f}]"
        p_b = f"{row['boot_p_approx']:.4f}"
        print(f"{n} & {diff} & {t} & {p_t} & {ci} & {p_b} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Paper 3 consistency results and produce figures + tables."
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "consistency_synthetic.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--fig_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "figures"),
        help="Directory for output figures.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    fig_dir = Path(args.fig_dir)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        print("Run generate_synthetic.py or run_consistency.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Domains: {sorted(df['domain'].unique())}")
    print(f"Scales: {sorted(df['n_agents'].unique())}")
    print()

    # --- Generate figures ---
    print("Generating figures...")
    fig_entropy_by_condition(df, fig_dir)
    fig_consistency_vs_size(df, fig_dir)
    fig_flip_rate_scatter(df, fig_dir)
    fig_improvement_heatmap(df, fig_dir)
    print()

    # --- Statistical tests ---
    print("Running statistical tests (team vs single-LLM)...")
    test_df = run_statistical_tests(df)
    if not test_df.empty:
        print(test_df.to_string(index=False))
        print()
    else:
        print("  No aligned data for statistical tests.")

    # --- LaTeX tables ---
    print_latex_summary(df)
    if not test_df.empty:
        print_latex_tests(test_df)

    # --- Save test results ---
    test_out = fig_dir.parent / "data" / "statistical_tests.csv"
    test_out.parent.mkdir(parents=True, exist_ok=True)
    if not test_df.empty:
        test_df.to_csv(test_out, index=False)
        print(f"Statistical test results saved -> {test_out}")


if __name__ == "__main__":
    main()
