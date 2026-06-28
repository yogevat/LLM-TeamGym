#!/usr/bin/env python3
"""
Analyze BabyVision scaling experiment results.

Loads the synthetic (or real) CSV data and produces:

  1. **Line plot**: accuracy vs n_agents for each subtask, with single-MLLM
     and human baselines as horizontal reference lines.
  2. **Scatter plot**: spectral cluster ID vs accuracy (do clusters predict
     performance?).
  3. **Heatmap**: subtask x topology_family accuracy matrix.
  4. **Pareto front**: cost-accuracy trade-off with dominated / non-dominated
     markers.

Also prints summary statistics and a LaTeX-formatted table suitable for
direct inclusion in the paper.

Usage
-----
    python analyze_results.py                           # default paths
    python analyze_results.py --input ../data/synthetic_results.csv
    python analyze_results.py --input ../data/synthetic_results.csv --figures_dir ../figures/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root for optional imports
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PAPER_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.abspath(os.path.join(_PAPER_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===================================================================
# Data loading
# ===================================================================


def load_csv(path: str) -> List[Dict[str, Any]]:
    """Load the results CSV into a list of dicts.

    Numeric fields are cast to int/float as appropriate.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, val in row.items():
                # Try int then float then keep as string
                try:
                    parsed[key] = int(val)
                except (ValueError, TypeError):
                    try:
                        parsed[key] = float(val)
                    except (ValueError, TypeError):
                        parsed[key] = val
            rows.append(parsed)
    return rows


# ===================================================================
# Summary statistics
# ===================================================================


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from the loaded data."""
    subtasks = sorted(set(r["subtask"] for r in rows))
    topologies = sorted(set(r["topology_family"] for r in rows))
    n_agents_vals = sorted(set(r["n_agents"] for r in rows))

    # Per-subtask accuracy (multi-agent only)
    multi_rows = [r for r in rows if r["n_agents"] > 1]
    subtask_acc: Dict[str, List[float]] = defaultdict(list)
    for r in multi_rows:
        subtask_acc[r["subtask"]].append(r["accuracy"])

    # Per-topology accuracy (multi-agent only)
    topo_acc: Dict[str, List[float]] = defaultdict(list)
    for r in multi_rows:
        topo_acc[r["topology_family"]].append(r["accuracy"])

    # Per-n_agents accuracy
    scale_acc: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        scale_acc[r["n_agents"]].append(r["accuracy"])

    # Best config per subtask
    best_per_subtask: Dict[str, Dict[str, Any]] = {}
    for st in subtasks:
        st_rows = [r for r in multi_rows if r["subtask"] == st]
        if st_rows:
            best = max(st_rows, key=lambda r: r["accuracy"])
            best_per_subtask[st] = best

    return {
        "n_rows": len(rows),
        "subtasks": subtasks,
        "topologies": topologies,
        "n_agents_values": n_agents_vals,
        "subtask_accuracy": {
            st: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
            for st, vals in subtask_acc.items()
        },
        "topology_accuracy": {
            t: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
            for t, vals in topo_acc.items()
        },
        "scale_accuracy": {
            n: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
            for n, vals in scale_acc.items()
        },
        "best_per_subtask": best_per_subtask,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print summary statistics to stdout."""
    print("\n" + "=" * 72)
    print("BABYVISION SCALING ANALYSIS -- SUMMARY STATISTICS")
    print("=" * 72)

    print(f"\nTotal data rows: {summary['n_rows']}")
    print(f"Agent scales: {summary['n_agents_values']}")
    print(f"Subtasks: {summary['subtasks']}")
    print(f"Topologies: {summary['topologies']}")

    print("\n--- Accuracy by Subtask (multi-agent only) ---")
    print(f"  {'Subtask':<35}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for st, stats in summary["subtask_accuracy"].items():
        print(
            f"  {st:<35}  {stats['mean']:>6.4f}  {stats['std']:>6.4f}  "
            f"{stats['min']:>6.4f}  {stats['max']:>6.4f}"
        )

    print("\n--- Accuracy by Topology Family (multi-agent only) ---")
    print(f"  {'Topology':<12}  {'Mean':>6}  {'Std':>6}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}")
    for t, stats in summary["topology_accuracy"].items():
        print(f"  {t:<12}  {stats['mean']:>6.4f}  {stats['std']:>6.4f}")

    print("\n--- Accuracy by Scale ---")
    print(f"  {'N':>6}  {'Mean Acc':>8}  {'Std':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*6}")
    for n in summary["n_agents_values"]:
        s = summary["scale_accuracy"][n]
        print(f"  {n:>6}  {s['mean']:>8.4f}  {s['std']:>6.4f}")

    print("\n--- Best Configuration per Subtask ---")
    for st, best in summary["best_per_subtask"].items():
        print(
            f"  {st}: N={best['n_agents']}, topo={best['topology_family']}, "
            f"acc={best['accuracy']:.4f}"
        )


# ===================================================================
# LaTeX table generation
# ===================================================================


def generate_latex_table(
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    """Generate a LaTeX table of accuracy by (subtask, topology_family).

    Returns the complete LaTeX string (tabular environment).
    """
    subtasks = summary["subtasks"]
    topologies = [t for t in summary["topologies"] if t != "single"]

    # Build accuracy matrix: subtask x topology (average over all N)
    acc_matrix: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r["n_agents"] > 1:
            acc_matrix[r["subtask"]][r["topology_family"]].append(r["accuracy"])

    # Column spec
    n_cols = len(topologies) + 1
    col_spec = "l" + "c" * len(topologies)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean accuracy (\%) by subtask and topology family (multi-agent, all scales).}",
        r"\label{tab:accuracy_matrix}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header row
    header = "Subtask"
    for t in topologies:
        header += f" & {t.capitalize()}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for st in subtasks:
        # Shorten subtask name for the table
        short_name = st.replace("_", " ").title()
        row_str = short_name
        for t in topologies:
            vals = acc_matrix[st][t]
            if vals:
                mean_pct = np.mean(vals) * 100
                row_str += f" & {mean_pct:.1f}"
            else:
                row_str += " & --"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\midrule")

    # Average row
    avg_row = r"\textbf{Average}"
    for t in topologies:
        all_vals = []
        for st in subtasks:
            all_vals.extend(acc_matrix[st][t])
        if all_vals:
            avg_row += f" & \\textbf{{{np.mean(all_vals) * 100:.1f}}}"
        else:
            avg_row += " & --"
    avg_row += r" \\"
    lines.append(avg_row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ===================================================================
# Plotting
# ===================================================================


def _setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def plot_accuracy_vs_agents(
    rows: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Line plot: accuracy vs n_agents for each subtask.

    Includes single-MLLM and human baselines as horizontal dashed lines.
    Averages over all topology families for the main curves.
    """
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(9, 6))

    subtasks = sorted(set(r["subtask"] for r in rows))
    colors = plt.cm.Set2(np.linspace(0, 1, len(subtasks)))

    # Get agent scales (multi-agent only)
    n_agents_vals = sorted(set(r["n_agents"] for r in rows if r["n_agents"] > 1))

    for i, subtask in enumerate(subtasks):
        st_rows = [r for r in rows if r["subtask"] == subtask]

        # Human and single-MLLM baselines (constant across N)
        human_acc = st_rows[0]["human_accuracy"]
        single_mllm_acc = st_rows[0]["single_mllm_accuracy"]

        # Compute mean accuracy at each N (averaged over topologies)
        means = []
        stds = []
        valid_ns = []
        for n in n_agents_vals:
            vals = [r["accuracy"] for r in st_rows if r["n_agents"] == n]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_ns.append(n)

        # Short label for legend
        short_label = subtask.replace("_", " ").title()

        # Main accuracy curve with error band
        ax.plot(
            valid_ns, means,
            marker="o", markersize=5,
            color=colors[i], linewidth=2,
            label=short_label,
        )
        ax.fill_between(
            valid_ns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=colors[i], alpha=0.12,
        )

        # Human baseline (only draw once per subtask at top)
        ax.axhline(
            y=human_acc,
            color=colors[i], linestyle="--", alpha=0.4, linewidth=1,
        )
        # Single-MLLM baseline
        ax.axhline(
            y=single_mllm_acc,
            color=colors[i], linestyle=":", alpha=0.4, linewidth=1,
        )

    # Add labeled reference lines
    ax.axhline(y=0, color="none")  # dummy for spacing
    ax.text(
        max(n_agents_vals) * 1.02, 0.92,
        "Human\nbaseline", fontsize=8, color="gray", va="center",
    )
    ax.text(
        max(n_agents_vals) * 1.02, 0.50,
        "Single-MLLM\nbaseline", fontsize=8, color="gray", va="center",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Agents (N)")
    ax.set_ylabel("Accuracy")
    ax.set_title("BabyVision Accuracy vs. Team Size")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0.30, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(n_agents_vals)
    ax.set_xticklabels([str(n) for n in n_agents_vals])

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [1/4] Accuracy vs agents -> {save_path}")


def plot_cluster_vs_accuracy(
    rows: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Scatter: spectral cluster ID vs accuracy.

    Assigns cluster IDs based on topology family (as a proxy for spectral
    clustering, since the synthetic data does not carry actual cluster IDs).
    """
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Map topology families to cluster IDs
    multi_rows = [r for r in rows if r["n_agents"] > 1]
    families = sorted(set(r["topology_family"] for r in multi_rows))
    family_to_cluster = {f: i for i, f in enumerate(families)}

    cluster_ids = [family_to_cluster[r["topology_family"]] for r in multi_rows]
    accuracies = [r["accuracy"] for r in multi_rows]

    # Color by n_agents to add information density
    n_vals = [r["n_agents"] for r in multi_rows]
    scatter = ax.scatter(
        cluster_ids, accuracies,
        c=np.log2(n_vals), cmap="viridis",
        alpha=0.6, s=40, edgecolors="gray", linewidths=0.3,
    )
    cbar = fig.colorbar(scatter, ax=ax, label="log2(N agents)")

    # Overlay cluster means
    for family, cid in family_to_cluster.items():
        fam_accs = [r["accuracy"] for r in multi_rows
                     if r["topology_family"] == family]
        mean_acc = np.mean(fam_accs)
        ax.plot(
            cid, mean_acc,
            marker="D", markersize=12,
            color="red", markeredgecolor="black", markeredgewidth=1.5,
            zorder=10,
        )

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.capitalize() for f in families], rotation=30)
    ax.set_xlabel("Spectral Cluster (Topology Family)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Do Spectral Clusters Predict Performance?")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [2/4] Cluster vs accuracy -> {save_path}")


def plot_accuracy_heatmap(
    rows: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Heatmap: subtask x topology_family accuracy matrix.

    Shows mean accuracy (averaged over all N > 1) for each combination.
    """
    plt = _setup_matplotlib()

    multi_rows = [r for r in rows if r["n_agents"] > 1]
    subtasks = sorted(set(r["subtask"] for r in multi_rows))
    topologies = sorted(set(r["topology_family"] for r in multi_rows))

    # Build matrix
    matrix = np.zeros((len(subtasks), len(topologies)))
    for i, st in enumerate(subtasks):
        for j, topo in enumerate(topologies):
            vals = [r["accuracy"] for r in multi_rows
                     if r["subtask"] == st and r["topology_family"] == topo]
            matrix[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.4, vmax=0.95)

    # Labels
    short_subtasks = [s.replace("_", "\n").title() for s in subtasks]
    ax.set_xticks(range(len(topologies)))
    ax.set_xticklabels([t.capitalize() for t in topologies], rotation=30)
    ax.set_yticks(range(len(subtasks)))
    ax.set_yticklabels(short_subtasks)

    # Annotate cells
    for i in range(len(subtasks)):
        for j in range(len(topologies)):
            val = matrix[i, j]
            text_color = "white" if val > 0.75 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=text_color,
            )

    ax.set_title("Accuracy: Subtask x Topology Family")
    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [3/4] Accuracy heatmap -> {save_path}")


def plot_pareto_front(
    rows: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """Cost-accuracy Pareto front plot.

    Each point is an (n_agents, topology_family) configuration averaged
    over subtasks.  Pareto-optimal configs are highlighted.
    """
    plt = _setup_matplotlib()

    multi_rows = [r for r in rows if r["n_agents"] > 1]

    # Aggregate by (n_agents, topology_family)
    config_data: Dict[Tuple[int, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"accuracy": [], "cost": []}
    )
    for r in multi_rows:
        key = (r["n_agents"], r["topology_family"])
        config_data[key]["accuracy"].append(r["accuracy"])
        config_data[key]["cost"].append(r["total_cost"])

    configs = []
    for (n, topo), data in config_data.items():
        configs.append({
            "n_agents": n,
            "topology": topo,
            "accuracy": float(np.mean(data["accuracy"])),
            "cost": float(np.mean(data["cost"])),
        })

    # Identify Pareto front (maximize accuracy, minimize cost)
    # -> maximize accuracy, maximize -cost
    accs = np.array([c["accuracy"] for c in configs])
    costs = np.array([c["cost"] for c in configs])
    objectives = np.column_stack([accs, -costs])

    n_pts = len(configs)
    is_dominated = np.zeros(n_pts, dtype=bool)
    for i in range(n_pts):
        for j in range(n_pts):
            if i == j:
                continue
            # j dominates i if j is at least as good in all objectives
            # and strictly better in at least one
            if (np.all(objectives[j] >= objectives[i]) and
                    np.any(objectives[j] > objectives[i])):
                is_dominated[i] = True
                break

    pareto_idx = [i for i in range(n_pts) if not is_dominated[i]]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Map topologies to markers
    topo_markers = {
        "star": "o", "chain": "s", "ring": "^",
        "sparse": "v", "dense": "D",
    }

    # Plot dominated points
    for i in range(n_pts):
        if is_dominated[i]:
            c = configs[i]
            marker = topo_markers.get(c["topology"], "x")
            ax.scatter(
                c["cost"], c["accuracy"],
                marker=marker, s=50, alpha=0.35,
                c="gray", edgecolors="gray", linewidths=0.5,
            )

    # Plot Pareto-optimal points
    for i in pareto_idx:
        c = configs[i]
        marker = topo_markers.get(c["topology"], "x")
        ax.scatter(
            c["cost"], c["accuracy"],
            marker=marker, s=120, alpha=0.9,
            c="crimson", edgecolors="black", linewidths=1.0,
            zorder=10,
        )
        ax.annotate(
            f"N={c['n_agents']}\n{c['topology']}",
            (c["cost"], c["accuracy"]),
            textcoords="offset points",
            xytext=(8, 5), fontsize=7,
            color="darkred",
        )

    # Draw Pareto front line
    pareto_configs = [configs[i] for i in pareto_idx]
    pareto_configs.sort(key=lambda c: c["cost"])
    if len(pareto_configs) > 1:
        ax.plot(
            [c["cost"] for c in pareto_configs],
            [c["accuracy"] for c in pareto_configs],
            "--", color="crimson", alpha=0.5, linewidth=1.5,
            label="Pareto front",
        )

    # Legend for topology markers
    for topo, marker in topo_markers.items():
        ax.scatter(
            [], [], marker=marker, c="gray", s=50,
            label=topo.capitalize(),
        )

    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost-Accuracy Pareto Front")
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.25)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [4/4] Pareto front -> {save_path}")


# ===================================================================
# Main
# ===================================================================


def parse_args() -> argparse.Namespace:
    default_input = os.path.join(_PAPER_DIR, "data", "synthetic_results.csv")
    default_figures = os.path.join(_PAPER_DIR, "figures")

    parser = argparse.ArgumentParser(
        description="Analyze BabyVision scaling results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=default_input,
        help=f"Input CSV path (default: {default_input})",
    )
    parser.add_argument(
        "--figures_dir", "-f",
        type=str,
        default=default_figures,
        help=f"Output directory for figures (default: {default_figures})",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip figure generation (print stats only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    print(f"Loading data from: {args.input}")
    rows = load_csv(args.input)

    if not rows:
        print("ERROR: No data loaded. Run generate_synthetic.py first.")
        sys.exit(1)

    # Compute and print summary
    summary = compute_summary(rows)
    print_summary(summary)

    # Generate LaTeX table
    latex = generate_latex_table(rows, summary)
    print("\n" + "=" * 72)
    print("LATEX TABLE")
    print("=" * 72)
    print(latex)

    # Save LaTeX table to file
    latex_path = os.path.join(_PAPER_DIR, "data", "accuracy_table.tex")
    os.makedirs(os.path.dirname(latex_path), exist_ok=True)
    with open(latex_path, "w") as fh:
        fh.write(latex)
    print(f"\nLaTeX table saved to: {latex_path}")

    # Generate figures
    if not args.no_plots:
        os.makedirs(args.figures_dir, exist_ok=True)
        print(f"\nGenerating figures in: {args.figures_dir}")

        plot_accuracy_vs_agents(
            rows,
            os.path.join(args.figures_dir, "accuracy_vs_agents.png"),
        )
        plot_cluster_vs_accuracy(
            rows,
            os.path.join(args.figures_dir, "cluster_vs_accuracy.png"),
        )
        plot_accuracy_heatmap(
            rows,
            os.path.join(args.figures_dir, "accuracy_heatmap.png"),
        )
        plot_pareto_front(
            rows,
            os.path.join(args.figures_dir, "pareto_front.png"),
        )
        print("\nAll figures generated.")


if __name__ == "__main__":
    main()
