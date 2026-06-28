#!/usr/bin/env python3
"""
Analysis pipeline for Paper 1 (Games / Exhaustive).

Loads synthetic (or real) CSV data and produces:
  1. Ranking table (sorted by win_rate) -- printed and LaTeX-formatted
  2. Figures:
     - Bar chart: top 10 topologies by win rate
     - Scatter: lambda_2 vs win_rate (with regression line)
     - Heatmap: (W, V) split performance
     - Box plot: score distribution by topology family
  3. Correlation analysis between spectral features and performance

Usage
-----
    python analyze_results.py
    python analyze_results.py --data_dir ../data/ --figures_dir ../figures/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results_csv(path: str) -> List[Dict[str, Any]]:
    """Load the aggregate results CSV into a list of dicts.

    Numeric fields are automatically converted to float/int.
    """
    numeric_fields = {
        "W", "V", "n_edges", "diameter",
        "avg_llm_calls", "avg_tokens",
    }
    float_fields = {
        "density", "lambda_2", "eigen_ratio", "avg_degree",
        "clustering_coeff", "win_rate", "avg_score", "std_score",
        "avg_latency", "robustness_drop",
    }

    rows: List[Dict[str, Any]] = []
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if k in numeric_fields:
                    parsed[k] = int(float(v))
                elif k in float_fields:
                    parsed[k] = float(v)
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_per_match_jsonl(matches_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load per-match JSONL files from a directory.

    Returns a dict mapping topology_id -> list of match records.
    """
    per_match: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(matches_dir):
        return per_match

    for fname in sorted(os.listdir(matches_dir)):
        if not fname.endswith(".jsonl"):
            continue
        topo_id = fname.replace(".jsonl", "")
        records: List[Dict[str, Any]] = []
        with open(os.path.join(matches_dir, fname), "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        per_match[topo_id] = records

    return per_match


# ---------------------------------------------------------------------------
# Ranking and correlation analysis
# ---------------------------------------------------------------------------

def compute_ranking(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort results by win_rate (descending) and add rank."""
    ranked = sorted(results, key=lambda r: r["win_rate"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked


def compute_correlations(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute Pearson correlation between spectral features and win_rate."""
    feature_keys = [
        "lambda_2", "eigen_ratio", "avg_degree", "density",
        "n_edges", "diameter", "clustering_coeff",
        "avg_llm_calls", "avg_latency", "robustness_drop",
    ]

    win_rates = np.array([r["win_rate"] for r in results])
    correlations: Dict[str, float] = {}

    for key in feature_keys:
        values = np.array([float(r[key]) for r in results])
        if np.std(values) < 1e-10 or np.std(win_rates) < 1e-10:
            correlations[key] = 0.0
        else:
            corr = np.corrcoef(values, win_rates)[0, 1]
            correlations[key] = float(corr)

    return correlations


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def print_latex_table(
    ranked: List[Dict[str, Any]],
    top_n: int = 15,
) -> str:
    """Print a LaTeX-formatted ranking table and return it as a string."""
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Top %d topology shapes ranked by win rate (four-penguin, exhaustive).}" % top_n)
    lines.append(r"\label{tab:topology_ranking}")
    lines.append(r"\begin{tabular}{r l c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Rank & Topology ID & $W$ & $V$ & $|E|$ & Win Rate & "
        r"Avg Score & $\lambda_2$ & Family \\"
    )
    lines.append(r"\midrule")

    for r in ranked[:top_n]:
        tid_short = r["topology_id"][:20]
        lines.append(
            f"  {r['rank']} & \\texttt{{{tid_short}}} & "
            f"{r['W']} & {r['V']} & {r['n_edges']} & "
            f"{r['win_rate']:.3f} & {r['avg_score']:.1f} & "
            f"{r['lambda_2']:.2f} & {r['family']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)
    print("\n" + table_str)
    return table_str


def print_correlation_table(
    correlations: Dict[str, float],
) -> str:
    """Print a LaTeX-formatted correlation table."""
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pearson correlation between topology features and win rate.}")
    lines.append(r"\label{tab:correlations}")
    lines.append(r"\begin{tabular}{l r}")
    lines.append(r"\toprule")
    lines.append(r"Feature & $r$ (Pearson) \\")
    lines.append(r"\midrule")

    for key, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        feature_name = key.replace("_", r"\_")
        lines.append(f"  {feature_name} & {corr:+.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)
    print("\n" + table_str)
    return table_str


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })
    return plt


def plot_top10_bar(
    ranked: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Bar chart of top 10 topologies by win rate."""
    plt = _setup_matplotlib()

    top10 = ranked[:10]
    labels = [f"#{r['rank']}\n(V={r['V']}, E={r['n_edges']})" for r in top10]
    win_rates = [r["win_rate"] for r in top10]
    families = [r["family"] for r in top10]

    family_colors = {
        "star": "#2ecc71",
        "dense": "#3498db",
        "moderate": "#f39c12",
        "sparse": "#e74c3c",
    }
    colors = [family_colors.get(f, "#95a5a6") for f in families]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(top10)), win_rates, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Win Rate")
    ax.set_title("Top 10 Topology Shapes by Win Rate (Four-Penguin Exhaustive)")
    ax.set_ylim(0, min(1.0, max(win_rates) * 1.2))

    # Legend for families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="white", label=f)
        for f, c in family_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right", title="Family")

    # Value labels on bars
    for bar, wr in zip(bars, win_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{wr:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_lambda2_vs_winrate(
    results: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Scatter plot of lambda_2 (algebraic connectivity) vs win_rate."""
    plt = _setup_matplotlib()

    lambda2 = np.array([r["lambda_2"] for r in results])
    win_rates = np.array([r["win_rate"] for r in results])
    families = [r["family"] for r in results]
    v_values = [r["V"] for r in results]

    family_colors = {
        "star": "#2ecc71",
        "dense": "#3498db",
        "moderate": "#f39c12",
        "sparse": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot by family
    for family, color in family_colors.items():
        mask = [f == family for f in families]
        if not any(mask):
            continue
        l2 = lambda2[mask]
        wr = win_rates[mask]
        vs = [v for v, m in zip(v_values, mask) if m]
        # Marker size by V
        sizes = [30 + 30 * v for v in vs]
        ax.scatter(
            l2, wr, c=color, s=sizes, alpha=0.75,
            edgecolors="white", linewidth=0.5, label=family, zorder=3,
        )

    # Regression line
    if len(lambda2) > 2:
        coeffs = np.polyfit(lambda2, win_rates, 1)
        x_line = np.linspace(lambda2.min(), lambda2.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(
            x_line, y_line, "--", color="gray", alpha=0.7, linewidth=1.5,
            label=f"Linear fit (r={np.corrcoef(lambda2, win_rates)[0,1]:.3f})",
        )

    ax.set_xlabel(r"$\lambda_2$ (Algebraic Connectivity)")
    ax.set_ylabel("Win Rate")
    ax.set_title(r"Algebraic Connectivity ($\lambda_2$) vs Win Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_wv_heatmap(
    results: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Heatmap showing average win rate split by (W, V)."""
    plt = _setup_matplotlib()

    # Aggregate by (W, V)
    wv_scores: Dict[Tuple[int, int], List[float]] = {}
    wv_counts: Dict[Tuple[int, int], int] = {}
    for r in results:
        key = (r["W"], r["V"])
        wv_scores.setdefault(key, []).append(r["win_rate"])
        wv_counts[key] = wv_counts.get(key, 0) + 1

    # For W=4, V=1..3
    v_vals = sorted(set(r["V"] for r in results))
    w_vals = sorted(set(r["W"] for r in results))

    # Build matrix
    matrix = np.zeros((len(w_vals), len(v_vals)))
    count_matrix = np.zeros((len(w_vals), len(v_vals)), dtype=int)
    for i, w in enumerate(w_vals):
        for j, v in enumerate(v_vals):
            key = (w, v)
            if key in wv_scores:
                matrix[i, j] = np.mean(wv_scores[key])
                count_matrix[i, j] = wv_counts[key]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.3, vmax=0.8)

    # Labels
    ax.set_xticks(range(len(v_vals)))
    ax.set_xticklabels([f"V={v}" for v in v_vals])
    ax.set_yticks(range(len(w_vals)))
    ax.set_yticklabels([f"W={w}" for w in w_vals])
    ax.set_xlabel("Number of Validators (V)")
    ax.set_ylabel("Number of Workers (W)")
    ax.set_title("Mean Win Rate by Team Composition (W, V)")

    # Annotate cells
    for i in range(len(w_vals)):
        for j in range(len(v_vals)):
            if count_matrix[i, j] > 0:
                text = f"{matrix[i, j]:.3f}\n(n={count_matrix[i, j]})"
                ax.text(
                    j, i, text, ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if matrix[i, j] > 0.55 else "black",
                )

    fig.colorbar(im, ax=ax, label="Mean Win Rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_family_boxplot(
    results: List[Dict[str, Any]],
    per_match: Dict[str, List[Dict[str, Any]]],
    save_path: str,
) -> None:
    """Box plot of score distribution by topology family.

    If per-match data is available, uses individual match scores.
    Otherwise uses the aggregate avg_score with simulated distributions.
    """
    plt = _setup_matplotlib()

    # Group by family
    family_order = ["star", "dense", "moderate", "sparse"]
    family_scores: Dict[str, List[float]] = {f: [] for f in family_order}

    for r in results:
        family = r["family"]
        if family not in family_scores:
            continue
        tid = r["topology_id"]

        if tid in per_match and per_match[tid]:
            # Use actual per-match scores
            scores = [m.get("score_team_a", r["avg_score"]) for m in per_match[tid]]
            family_scores[family].extend(scores)
        else:
            # Simulate from aggregate stats
            rng = np.random.default_rng(hash(tid) % (2**31))
            simulated = rng.normal(r["avg_score"], r["std_score"], size=50)
            family_scores[family].extend(simulated.tolist())

    # Filter out empty families
    plot_families = [f for f in family_order if family_scores[f]]
    plot_data = [family_scores[f] for f in plot_families]

    family_colors = {
        "star": "#2ecc71",
        "dense": "#3498db",
        "moderate": "#f39c12",
        "sparse": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        plot_data,
        tick_labels=[f.capitalize() for f in plot_families],
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )

    for patch, family in zip(bp["boxes"], plot_families):
        patch.set_facecolor(family_colors.get(family, "#95a5a6"))
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel("Team A Score")
    ax.set_xlabel("Topology Family")
    ax.set_title("Score Distribution by Topology Family")
    ax.grid(True, axis="y", alpha=0.3)

    # Add sample sizes
    for i, family in enumerate(plot_families):
        n = len(family_scores[family])
        ax.text(
            i + 1, ax.get_ylim()[0] + 0.5,
            f"n={n}", ha="center", va="bottom", fontsize=8, color="gray",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cost_performance_scatter(
    results: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Scatter: avg_llm_calls (cost proxy) vs win_rate, colored by V."""
    plt = _setup_matplotlib()

    calls = np.array([r["avg_llm_calls"] for r in results])
    win_rates = np.array([r["win_rate"] for r in results])
    v_values = np.array([r["V"] for r in results])

    fig, ax = plt.subplots(figsize=(8, 6))

    v_colors = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71"}
    for v in sorted(set(v_values)):
        mask = v_values == v
        ax.scatter(
            calls[mask], win_rates[mask],
            c=v_colors.get(v, "#95a5a6"),
            s=60, alpha=0.7,
            edgecolors="white", linewidth=0.5,
            label=f"V={v}",
            zorder=3,
        )

    ax.set_xlabel("Average LLM Calls per Match (Cost Proxy)")
    ax.set_ylabel("Win Rate")
    ax.set_title("Performance vs Communication Cost")
    ax.legend(loc="lower right", title="Validators")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    results: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    correlations: Dict[str, float],
) -> None:
    """Print a concise summary to stdout."""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY -- Paper 1 (Games / Exhaustive)")
    print("=" * 70)

    print(f"\nTotal topologies analyzed: {len(results)}")

    # By (W,V) split
    for V in sorted(set(r["V"] for r in results)):
        subset = [r for r in results if r["V"] == V]
        mean_wr = np.mean([r["win_rate"] for r in subset])
        print(f"  W=4, V={V}: {len(subset)} shapes, mean win_rate={mean_wr:.3f}")

    # Top 10
    print(f"\nTop 10 topologies by win rate:")
    print(f"  {'Rank':>4}  {'Topology':>22}  {'V':>2}  {'E':>2}  "
          f"{'WinRate':>8}  {'Score':>7}  {'Calls':>6}  {'Family':>8}")
    print(f"  {'-'*4}  {'-'*22}  {'-'*2}  {'-'*2}  "
          f"{'-'*8}  {'-'*7}  {'-'*6}  {'-'*8}")
    for r in ranked[:10]:
        print(
            f"  {r['rank']:>4}  {r['topology_id']:>22}  "
            f"{r['V']:>2}  {r['n_edges']:>2}  "
            f"{r['win_rate']:>8.3f}  {r['avg_score']:>7.1f}  "
            f"{r['avg_llm_calls']:>6}  {r['family']:>8}"
        )

    # Correlations
    print(f"\nFeature correlations with win_rate (|r|):")
    for key, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "#" * int(abs(corr) * 40)
        sign = "+" if corr > 0 else "-"
        print(f"  {key:>20}: {sign}{abs(corr):.3f}  {bar}")

    # Family analysis
    print(f"\nPerformance by topology family:")
    families = sorted(set(r["family"] for r in results))
    for family in families:
        subset = [r for r in results if r["family"] == family]
        wr = [r["win_rate"] for r in subset]
        calls = [r["avg_llm_calls"] for r in subset]
        print(
            f"  {family:>10}: n={len(subset):>3}, "
            f"win_rate={np.mean(wr):.3f} +/- {np.std(wr):.3f}, "
            f"avg_calls={np.mean(calls):.0f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze exhaustive topology sweep results for Paper 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data/",
        help="Directory containing results CSV and match JSONL files.",
    )
    parser.add_argument(
        "--figures_dir", type=str, default="../figures/",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--csv_name", type=str, default="synthetic_results.csv",
        help="Name of the aggregate results CSV file.",
    )
    parser.add_argument(
        "--top_n", type=int, default=15,
        help="Number of top topologies to show in LaTeX table.",
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(
            os.path.join(_SCRIPT_DIR, args.data_dir)
        )
    if not os.path.isabs(args.figures_dir):
        args.figures_dir = os.path.abspath(
            os.path.join(_SCRIPT_DIR, args.figures_dir)
        )

    csv_path = os.path.join(args.data_dir, args.csv_name)
    matches_dir = os.path.join(args.data_dir, "matches")

    # Check data exists
    if not os.path.isfile(csv_path):
        print(f"ERROR: Results CSV not found at {csv_path}")
        print("Run generate_synthetic.py first to create synthetic data.")
        sys.exit(1)

    os.makedirs(args.figures_dir, exist_ok=True)

    print("=" * 60)
    print("Analysis Pipeline -- Paper 1 (Games / Exhaustive)")
    print("=" * 60)

    # Step 1: Load data
    print(f"\n[1/5] Loading data from {csv_path}")
    results = load_results_csv(csv_path)
    print(f"  Loaded {len(results)} topology records")

    per_match = load_per_match_jsonl(matches_dir)
    if per_match:
        total_matches = sum(len(v) for v in per_match.values())
        print(f"  Loaded per-match data: {total_matches} match records across {len(per_match)} topologies")
    else:
        print("  No per-match data found (will use aggregate stats for box plots)")

    # Step 2: Compute ranking and correlations
    print(f"\n[2/5] Computing rankings and correlations...")
    ranked = compute_ranking(results)
    correlations = compute_correlations(results)

    # Step 3: Print summary
    print_summary(results, ranked, correlations)

    # Step 4: Generate figures
    print(f"\n[4/5] Generating figures in {args.figures_dir}")

    plot_top10_bar(
        ranked,
        os.path.join(args.figures_dir, "top10_win_rate.png"),
    )

    plot_lambda2_vs_winrate(
        results,
        os.path.join(args.figures_dir, "lambda2_vs_winrate.png"),
    )

    plot_wv_heatmap(
        results,
        os.path.join(args.figures_dir, "wv_heatmap.png"),
    )

    plot_family_boxplot(
        results,
        per_match,
        os.path.join(args.figures_dir, "family_boxplot.png"),
    )

    plot_cost_performance_scatter(
        results,
        os.path.join(args.figures_dir, "cost_vs_performance.png"),
    )

    # Step 5: LaTeX tables
    print(f"\n[5/5] LaTeX tables:")
    print_latex_table(ranked, top_n=args.top_n)
    print_correlation_table(correlations)

    print(f"\nAnalysis complete!")
    print(f"  Figures:  {args.figures_dir}")
    print(f"  Data:     {args.data_dir}")


if __name__ == "__main__":
    main()
