#!/usr/bin/env python3
"""
Generate realistic synthetic BabyVision results across scales.

Produces a CSV with columns:
    n_agents, topology_family, subtask, accuracy, human_accuracy,
    single_mllm_accuracy, consistency, flip_rate, total_tokens, total_cost

Design principles for the synthetic curves:
  - Single-MLLM baseline: 40--60 % depending on subtask.
  - Multi-agent teams close the gap to human performance (85--95 %) as N
    increases, but with diminishing returns (log-like saturation).
  - Topology families have distinct scaling profiles:
      * star   -- strong early gains, saturates quickly.
      * chain  -- slow start, steady improvement.
      * ring   -- balanced, similar to chain but slightly better.
      * sparse -- weakest overall.
      * dense  -- nearly as good as star.
  - Cost scales roughly linearly with N (more agents = more tokens).
  - Consistency improves then degrades slightly at very large N
    (coordination overhead).
  - Flip rate decreases with N (ensemble smoothing).

Usage
-----
    python generate_synthetic.py                           # default output
    python generate_synthetic.py --output ../data/synthetic_results.csv
    python generate_synthetic.py --seed 42 --n_agents 2,4,8,16,32,64,128,256
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBTASKS = [
    "fine_grained_discrimination",
    "visual_tracking",
    "spatial_perception",
    "pattern_recognition",
]

# Per-subtask difficulty parameters:
#   (human_accuracy, single_mllm_accuracy, max_team_accuracy, difficulty)
# difficulty controls how quickly multi-agent teams approach ceiling.
SUBTASK_PARAMS: Dict[str, Dict[str, float]] = {
    "fine_grained_discrimination": {
        "human_accuracy": 0.92,
        "single_mllm_accuracy": 0.42,
        "max_team_accuracy": 0.89,
        "difficulty": 1.2,   # hardest -- slow scaling
    },
    "visual_tracking": {
        "human_accuracy": 0.88,
        "single_mllm_accuracy": 0.48,
        "max_team_accuracy": 0.86,
        "difficulty": 0.9,
    },
    "spatial_perception": {
        "human_accuracy": 0.95,
        "single_mllm_accuracy": 0.55,
        "max_team_accuracy": 0.92,
        "difficulty": 0.7,   # easiest -- fastest scaling
    },
    "pattern_recognition": {
        "human_accuracy": 0.90,
        "single_mllm_accuracy": 0.58,
        "max_team_accuracy": 0.88,
        "difficulty": 0.8,
    },
}

TOPOLOGY_FAMILIES = ["star", "chain", "ring", "sparse", "dense"]

# Topology-specific scaling modifiers:
#   (rate_multiplier, accuracy_offset, consistency_base)
# rate_multiplier: how quickly this topology scales (higher = faster).
# accuracy_offset: additive bonus/penalty (at large N).
# consistency_base: baseline consistency before N-dependent adjustment.
TOPOLOGY_MODIFIERS: Dict[str, Dict[str, float]] = {
    "star": {
        "rate_multiplier": 1.3,
        "accuracy_offset": 0.03,
        "consistency_base": 0.82,
    },
    "chain": {
        "rate_multiplier": 0.8,
        "accuracy_offset": -0.02,
        "consistency_base": 0.78,
    },
    "ring": {
        "rate_multiplier": 0.9,
        "accuracy_offset": 0.0,
        "consistency_base": 0.80,
    },
    "sparse": {
        "rate_multiplier": 0.6,
        "accuracy_offset": -0.05,
        "consistency_base": 0.70,
    },
    "dense": {
        "rate_multiplier": 1.2,
        "accuracy_offset": 0.02,
        "consistency_base": 0.84,
    },
}

# Cost model: tokens and USD per agent per evaluation
TOKENS_PER_AGENT = 1500       # average tokens per agent per trial
COST_PER_1K_TOKENS = 0.003    # USD (GPT-4o-class pricing)

DEFAULT_N_AGENTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Number of repeated trials per (n_agents, topology, subtask) configuration
N_TRIALS = 5


# ===================================================================
# Synthetic data generation
# ===================================================================


def _accuracy_curve(
    n_agents: int,
    single_mllm: float,
    max_team: float,
    difficulty: float,
    rate_multiplier: float,
    accuracy_offset: float,
) -> float:
    """Compute expected accuracy for a multi-agent team.

    Uses a saturating logarithmic model:
        acc = single_mllm + (max_team - single_mllm) * (1 - exp(-rate * ln(N) / difficulty))
    clamped to [single_mllm, max_team + offset].

    For N=1 (single agent), returns single_mllm directly.
    """
    if n_agents <= 1:
        return single_mllm

    gap = max_team - single_mllm
    rate = rate_multiplier * 0.5
    progress = 1.0 - np.exp(-rate * np.log(n_agents) / difficulty)
    acc = single_mllm + gap * progress + accuracy_offset * progress
    return float(np.clip(acc, single_mllm, max_team + accuracy_offset))


def _consistency_curve(
    n_agents: int,
    consistency_base: float,
) -> float:
    """Compute consistency (0-1).

    Consistency rises quickly with N (ensemble effect), peaks around N=32-64,
    then decreases slightly at very large N (coordination overhead).
    """
    if n_agents <= 1:
        return 0.65

    # Rise: 1 - 1/(1 + log2(N))
    rise = 1.0 - 1.0 / (1.0 + np.log2(n_agents))
    # Coordination penalty at large N
    penalty = max(0.0, 0.02 * np.log2(max(n_agents / 64, 1.0)))
    cons = consistency_base * rise - penalty
    return float(np.clip(cons, 0.3, 0.98))


def _flip_rate_curve(n_agents: int) -> float:
    """Compute flip rate (0-1).

    Flip rate is high for single agents (~0.25) and decreases with ensemble
    size, approaching ~0.03 for very large teams.
    """
    if n_agents <= 1:
        return 0.25
    return float(0.03 + 0.22 * np.exp(-0.15 * n_agents))


def _compute_cost(n_agents: int) -> Tuple[int, float]:
    """Compute total tokens and cost for an evaluation at scale N.

    Returns (total_tokens, total_cost_usd).
    """
    # Communication overhead grows slightly super-linearly
    overhead_factor = 1.0 + 0.1 * np.log2(max(n_agents, 1))
    total_tokens = int(n_agents * TOKENS_PER_AGENT * overhead_factor)
    total_cost = total_tokens * COST_PER_1K_TOKENS / 1000.0
    return total_tokens, round(total_cost, 6)


def generate_synthetic_data(
    n_agents_list: List[int],
    seed: int = 42,
    n_trials: int = N_TRIALS,
) -> List[Dict[str, object]]:
    """Generate the full synthetic dataset.

    Parameters
    ----------
    n_agents_list : list of int
        Agent counts to generate data for.
    seed : int
        Random seed for reproducibility.
    n_trials : int
        Number of noisy trials per configuration.

    Returns
    -------
    rows : list of dict
        Each dict corresponds to one CSV row.
    """
    rng = np.random.RandomState(seed)
    rows: List[Dict[str, object]] = []

    for n in sorted(n_agents_list):
        for subtask in SUBTASKS:
            sp = SUBTASK_PARAMS[subtask]
            human_acc = sp["human_accuracy"]
            single_mllm = sp["single_mllm_accuracy"]
            max_team = sp["max_team_accuracy"]
            difficulty = sp["difficulty"]

            # For N=1 (single MLLM baseline), use only one "topology"
            topo_list = ["single"] if n <= 1 else TOPOLOGY_FAMILIES

            for topo in topo_list:
                if topo == "single":
                    tm = {"rate_multiplier": 1.0, "accuracy_offset": 0.0,
                          "consistency_base": 0.65}
                else:
                    tm = TOPOLOGY_MODIFIERS[topo]

                # Compute expected (noiseless) metrics
                expected_acc = _accuracy_curve(
                    n, single_mllm, max_team, difficulty,
                    tm["rate_multiplier"], tm["accuracy_offset"],
                )
                expected_cons = _consistency_curve(n, tm["consistency_base"])
                expected_flip = _flip_rate_curve(n)
                total_tokens, total_cost = _compute_cost(n)

                # Generate multiple noisy trials and average
                trial_accs = []
                trial_cons = []
                trial_flips = []

                for _ in range(n_trials):
                    # Add Gaussian noise scaled by 1/sqrt(n_agents)
                    noise_scale = 0.04 / max(np.sqrt(n), 1.0)
                    acc_noise = rng.normal(0, noise_scale)
                    cons_noise = rng.normal(0, noise_scale * 0.5)
                    flip_noise = rng.normal(0, noise_scale * 0.3)

                    trial_accs.append(
                        np.clip(expected_acc + acc_noise, 0.0, 1.0)
                    )
                    trial_cons.append(
                        np.clip(expected_cons + cons_noise, 0.0, 1.0)
                    )
                    trial_flips.append(
                        np.clip(expected_flip + flip_noise, 0.0, 1.0)
                    )

                # Record the trial-averaged result
                avg_acc = float(np.mean(trial_accs))
                avg_cons = float(np.mean(trial_cons))
                avg_flip = float(np.mean(trial_flips))

                rows.append({
                    "n_agents": n,
                    "topology_family": topo if n > 1 else "single",
                    "subtask": subtask,
                    "accuracy": round(avg_acc, 4),
                    "human_accuracy": round(human_acc, 4),
                    "single_mllm_accuracy": round(single_mllm, 4),
                    "consistency": round(avg_cons, 4),
                    "flip_rate": round(avg_flip, 4),
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                })

    return rows


def save_csv(rows: List[Dict[str, object]], path: str) -> None:
    """Write rows to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fieldnames = [
        "n_agents",
        "topology_family",
        "subtask",
        "accuracy",
        "human_accuracy",
        "single_mllm_accuracy",
        "consistency",
        "flip_rate",
        "total_tokens",
        "total_cost",
    ]

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PAPER_DIR = os.path.dirname(_THIS_DIR)
    default_output = os.path.join(_PAPER_DIR, "data", "synthetic_results.csv")

    parser = argparse.ArgumentParser(
        description="Generate synthetic BabyVision scaling results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=default_output,
        help=f"Output CSV path (default: {default_output})",
    )
    parser.add_argument(
        "--n_agents",
        type=str,
        default=",".join(str(n) for n in DEFAULT_N_AGENTS),
        help="Comma-separated agent counts (default: 1,2,4,8,16,32,64,128,256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=N_TRIALS,
        help=f"Trials per configuration (default: {N_TRIALS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_agents_list = [int(x.strip()) for x in args.n_agents.split(",")]

    print(f"Generating synthetic BabyVision data (seed={args.seed})")
    print(f"  Agent counts: {n_agents_list}")
    print(f"  Subtasks: {SUBTASKS}")
    print(f"  Topology families: {TOPOLOGY_FAMILIES}")
    print(f"  Trials per config: {args.n_trials}")

    rows = generate_synthetic_data(
        n_agents_list=n_agents_list,
        seed=args.seed,
        n_trials=args.n_trials,
    )

    save_csv(rows, args.output)

    # Print summary
    n_configs = len(rows)
    n_with_multi = sum(1 for r in rows if r["n_agents"] > 1)
    accuracies = [r["accuracy"] for r in rows]

    print(f"\nGenerated {n_configs} rows ({n_with_multi} multi-agent configs)")
    print(f"  Accuracy range: [{min(accuracies):.4f}, {max(accuracies):.4f}]")
    print(f"  Saved to: {args.output}")

    # Preview first few rows
    print(f"\nPreview (first 5 rows):")
    print(f"  {'n_agents':>8}  {'topology':>8}  {'subtask':<35}  {'acc':>6}  {'cost':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*35}  {'-'*6}  {'-'*8}")
    for row in rows[:5]:
        print(
            f"  {row['n_agents']:>8}  {row['topology_family']:>8}  "
            f"{row['subtask']:<35}  {row['accuracy']:>6.4f}  "
            f"${row['total_cost']:>7.4f}"
        )
    if len(rows) > 5:
        print(f"  ... ({len(rows) - 5} more rows)")


if __name__ == "__main__":
    main()
