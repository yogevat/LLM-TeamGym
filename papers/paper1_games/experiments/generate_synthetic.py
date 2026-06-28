#!/usr/bin/env python3
"""
Generate realistic synthetic experiment results for all 61 four-penguin
topologies, for Paper 1 (Games / Exhaustive).

Produces:
  1. CSV summary: one row per topology with aggregate metrics
  2. JSONL per-match data: 200 matches per topology with per-match stats

The synthetic data model captures realistic trends:
  - Star (complete bipartite) topologies score well: high consensus,
    moderate cost from full connectivity.
  - Denser topologies have higher communication cost (more LLM calls,
    tokens, latency) but potentially better coordination.
  - Sparse topologies are cheap but may miss critical validation.
  - Intermediate connectivity often yields the best performance-cost
    trade-off.
  - Results include realistic variance: outcome noise, score jitter,
    and topology-dependent robustness.

Usage
-----
    python generate_synthetic.py
    python generate_synthetic.py --output_dir ../data/ --n_matches 200 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_team_gym.core.topology import TopologyGraph
from llm_team_gym.topology.enumeration import (
    enumerate_shapes,
    shape_to_topology,
)
from llm_team_gym.topology.spectral import SpectralAnalyzer


# ---------------------------------------------------------------------------
# Shape hash (same as run_exhaustive.py)
# ---------------------------------------------------------------------------

def _shape_hash(B: np.ndarray) -> str:
    return hashlib.sha256(B.tobytes()).hexdigest()[:12]


def _topology_id(W: int, V: int, idx: int, shape_hash: str) -> str:
    return f"W{W}_V{V}_{idx:03d}_{shape_hash}"


# ---------------------------------------------------------------------------
# Enumerate all 61 shapes with spectral features
# ---------------------------------------------------------------------------

def enumerate_all_shapes() -> List[Dict[str, Any]]:
    """Enumerate all 61 four-penguin shapes and compute their features."""
    all_shapes: List[Dict[str, Any]] = []
    global_idx = 0

    for V in range(1, 4):
        shapes = enumerate_shapes(W=4, V=V, relaxed=True)
        for local_idx, B in enumerate(shapes):
            sh = _shape_hash(B)
            topo_id = _topology_id(4, V, global_idx, sh)

            # Full adjacency for spectral analysis (W=4 workers + V validators + 1 thinker)
            tg = TopologyGraph.from_matrix(B, n_thinkers=1)
            adj = tg.to_adjacency_matrix()
            sa = SpectralAnalyzer(adj)
            features = sa.feature_vector()

            desc = shape_to_topology(B)
            n_edges = int(B.sum())
            max_edges = 4 * V  # complete bipartite would have W*V edges
            density = desc["density"]

            all_shapes.append({
                "W": 4,
                "V": V,
                "idx": global_idx,
                "topology_id": topo_id,
                "shape_hash": sh,
                "shape_matrix": B.tolist(),
                "n_edges": n_edges,
                "max_edges": max_edges,
                "density": density,
                "worker_degrees": desc["worker_degrees"],
                "validator_degrees": desc["validator_degrees"],
                "lambda_2": float(features[0]),
                "eigen_ratio": float(features[1]),
                "avg_degree": float(features[2]),
                "max_degree": float(features[3]),
                "diameter": int(features[4]),
                "clustering_coeff": float(features[5]),
            })
            global_idx += 1

    assert len(all_shapes) == 61, f"Expected 61 shapes, got {len(all_shapes)}"
    return all_shapes


# ---------------------------------------------------------------------------
# Synthetic data generation model
# ---------------------------------------------------------------------------

def _classify_topology(desc: Dict[str, Any]) -> str:
    """Classify a topology into a structural family for analysis."""
    W, V = desc["W"], desc["V"]
    n_edges = desc["n_edges"]
    max_edges = desc["max_edges"]
    density = desc["density"]

    if density >= 0.99:
        return "star"  # complete bipartite (star-like)
    elif density >= 0.7:
        return "dense"
    elif density <= 0.3:
        return "sparse"
    else:
        return "moderate"


def generate_synthetic_data(
    all_shapes: List[Dict[str, Any]],
    n_matches: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate synthetic per-topology aggregate results.

    The model captures these empirical trends observed in multi-agent
    coordination research:

    1. Win rate: positively correlated with lambda_2 (connectivity) but
       with diminishing returns. Star topologies benefit from full
       consensus. Very sparse topologies suffer from coordination
       failure.

    2. Score: correlated with win rate but has independent noise.
       Higher connectivity enables better strategic play.

    3. LLM calls: directly proportional to n_edges (each edge =
       one Worker-Validator interaction per round). Thinker adds 1 call.
       Multiple rounds per match step.

    4. Latency: proportional to LLM calls but with sequential
       bottlenecks for deeper topologies.

    5. Robustness: how much performance drops when one agent is
       degraded. Star topologies are robust (redundancy). Minimal
       spanning topologies are fragile.
    """
    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []

    for desc in all_shapes:
        n_edges = desc["n_edges"]
        max_edges = desc["max_edges"]
        density = desc["density"]
        lambda_2 = desc["lambda_2"]
        W = desc["W"]
        V = desc["V"]
        family = _classify_topology(desc)

        # --- Win rate model ---
        # Base win rate from connectivity: sigmoid of lambda_2
        # lambda_2 range for these topologies: ~0.4 to ~5.0
        connectivity_bonus = 1.0 / (1.0 + np.exp(-0.8 * (lambda_2 - 2.0)))
        # Density penalty: very dense = more calls but marginally better
        density_adjustment = 0.05 * (density - 0.5)
        # V-count bonus: more validators = better validation coverage
        v_bonus = 0.03 * (V - 1)
        # Base rate
        base_win = 0.35 + 0.30 * connectivity_bonus + density_adjustment + v_bonus
        # Clamp and add noise
        win_rate = float(np.clip(base_win + rng.normal(0, 0.03), 0.15, 0.85))

        # --- Score model ---
        # Score correlates with win rate; range ~5-25 (team_fish style)
        base_score = 8.0 + 12.0 * win_rate
        # Star topologies get a coordination bonus
        if family == "star":
            base_score += 2.0
        score_noise = rng.normal(0, 1.5)
        avg_score = float(np.clip(base_score + score_noise, 3.0, 30.0))
        std_score = float(np.clip(2.0 + 1.5 * (1 - win_rate), 1.0, 6.0))

        # --- LLM calls model ---
        # Per match step: W proposals + n_edges validations + 1 thinker = W + n_edges + 1
        calls_per_step = W + n_edges + 1
        # Average match length: ~15-30 steps depending on game
        avg_match_steps = 20.0 + rng.normal(0, 3)
        avg_llm_calls = int(np.clip(
            calls_per_step * avg_match_steps + rng.normal(0, 10),
            50, 600
        ))

        # --- Token count model ---
        # ~300 tokens per LLM call on average (prompt + completion)
        avg_tokens = int(avg_llm_calls * (280 + rng.normal(0, 30)))

        # --- Latency model ---
        # Sequential bottleneck: deeper topologies have more serial
        # rounds within each step. ~0.5s per LLM call for real models.
        sequential_depth = max(desc["worker_degrees"]) if desc["worker_degrees"] else 1
        # Mock model: ~0.001s per call; real model: ~0.5s per call
        avg_latency = float(np.clip(
            0.001 * avg_llm_calls * (1 + 0.1 * sequential_depth) + rng.normal(0, 0.05),
            0.05, 2.0
        ))

        # --- Robustness model ---
        # Robustness = how much win_rate drops if one worker is degraded
        # Sparse topologies are fragile (removing one edge can disconnect)
        # Dense topologies are robust (many redundant paths)
        redundancy = density * V  # crude measure of path redundancy
        robustness_drop = float(np.clip(
            0.25 * (1 - density) + 0.1 / max(V, 1) + rng.normal(0, 0.03),
            0.01, 0.40
        ))

        results.append({
            "topology_id": desc["topology_id"],
            "W": W,
            "V": V,
            "n_edges": n_edges,
            "shape_hash": desc["shape_hash"],
            "density": density,
            "lambda_2": lambda_2,
            "eigen_ratio": desc["eigen_ratio"],
            "avg_degree": desc["avg_degree"],
            "diameter": desc["diameter"],
            "clustering_coeff": desc["clustering_coeff"],
            "family": family,
            "win_rate": win_rate,
            "avg_score": avg_score,
            "std_score": std_score,
            "avg_llm_calls": avg_llm_calls,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_latency,
            "robustness_drop": robustness_drop,
        })

    return results


def generate_per_match_data(
    results: List[Dict[str, Any]],
    n_matches: int = 200,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate per-match synthetic data for each topology.

    Returns a dict mapping topology_id -> list of match records.
    """
    rng = np.random.default_rng(seed + 1000)
    per_match: Dict[str, List[Dict[str, Any]]] = {}

    for topo_result in results:
        tid = topo_result["topology_id"]
        win_rate = topo_result["win_rate"]
        avg_score = topo_result["avg_score"]
        std_score = topo_result["std_score"]
        avg_calls = topo_result["avg_llm_calls"]
        avg_latency = topo_result["avg_latency"]

        matches: List[Dict[str, Any]] = []
        for mi in range(n_matches):
            # Win/loss for this match
            won = bool(rng.random() < win_rate)

            # Score for Team A
            score_a = float(np.clip(
                rng.normal(avg_score, std_score), 0.0, 40.0
            ))
            # Score for Team B (opponent): inversely correlated with Team A
            score_b = float(np.clip(
                rng.normal(20.0 - avg_score * 0.5, std_score * 1.2), 0.0, 40.0
            ))

            if won:
                score_a = max(score_a, score_b + rng.uniform(0.5, 3.0))
            else:
                score_b = max(score_b, score_a + rng.uniform(0.5, 3.0))

            # Match-level LLM calls with variance
            match_calls = int(np.clip(
                rng.normal(avg_calls, avg_calls * 0.15), 20, 800
            ))

            # Match-level tokens
            match_tokens = int(match_calls * rng.normal(280, 40))

            # Match-level latency
            match_latency = float(np.clip(
                rng.normal(avg_latency, avg_latency * 0.2), 0.01, 5.0
            ))

            # Steps
            match_steps = int(np.clip(rng.normal(22, 5), 8, 50))

            matches.append({
                "match_idx": mi,
                "topology_id": tid,
                "won": won,
                "score_team_a": round(score_a, 2),
                "score_team_b": round(score_b, 2),
                "total_steps": match_steps,
                "llm_calls": match_calls,
                "tokens": match_tokens,
                "latency_s": round(match_latency, 4),
                "winner": "team_A" if won else "team_B",
            })

        per_match[tid] = matches

    return per_match


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_summary_csv(
    results: List[Dict[str, Any]],
    path: str,
) -> None:
    """Save aggregate results as CSV (one row per topology)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "topology_id", "W", "V", "n_edges", "shape_hash",
        "density", "lambda_2", "eigen_ratio", "avg_degree",
        "diameter", "clustering_coeff", "family",
        "win_rate", "avg_score", "std_score",
        "avg_llm_calls", "avg_tokens", "avg_latency", "robustness_drop",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Write only the fields in fieldnames
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  Summary CSV saved to {path}  ({len(results)} rows)")


def save_per_match_jsonl(
    per_match: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> None:
    """Save per-match data as JSONL files (one file per topology)."""
    matches_dir = os.path.join(output_dir, "matches")
    os.makedirs(matches_dir, exist_ok=True)

    for topo_id, matches in per_match.items():
        path = os.path.join(matches_dir, f"{topo_id}.jsonl")
        with open(path, "w") as fh:
            for record in matches:
                fh.write(json.dumps(record, default=str) + "\n")

    print(f"  Per-match JSONL saved to {matches_dir}/  ({len(per_match)} files)")


def save_topology_catalog(
    all_shapes: List[Dict[str, Any]],
    path: str,
) -> None:
    """Save the topology catalog as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_shapes, f, indent=2, default=str)
    print(f"  Topology catalog saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic experiment results for all 61 four-penguin "
            "topologies (Paper 1)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--n_matches", type=int, default=200,
        help="Number of per-match records per topology.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Resolve output_dir relative to script location
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(
            os.path.join(_SCRIPT_DIR, args.output_dir)
        )

    print("=" * 60)
    print("Synthetic Data Generator -- Paper 1 (Games / Exhaustive)")
    print("=" * 60)

    # Step 1: Enumerate all 61 shapes
    print("\n[1/4] Enumerating all four-penguin shapes...")
    all_shapes = enumerate_all_shapes()
    print(f"  Found {len(all_shapes)} shapes")
    for V in range(1, 4):
        count = sum(1 for s in all_shapes if s["V"] == V)
        print(f"    V={V}: {count} shapes")

    # Step 2: Generate aggregate synthetic results
    print(f"\n[2/4] Generating synthetic aggregate results (seed={args.seed})...")
    results = generate_synthetic_data(all_shapes, n_matches=args.n_matches, seed=args.seed)

    # Print summary statistics
    win_rates = [r["win_rate"] for r in results]
    print(f"  Win rate range: [{min(win_rates):.3f}, {max(win_rates):.3f}]")
    print(f"  Win rate mean:  {np.mean(win_rates):.3f}")

    # Print top 5
    ranked = sorted(results, key=lambda r: r["win_rate"], reverse=True)
    print(f"\n  Top 5 topologies by win rate:")
    for i, r in enumerate(ranked[:5]):
        print(
            f"    {i+1}. {r['topology_id']}  "
            f"win={r['win_rate']:.3f}  "
            f"score={r['avg_score']:.1f}  "
            f"calls={r['avg_llm_calls']}  "
            f"family={r['family']}"
        )

    # Step 3: Generate per-match data
    print(f"\n[3/4] Generating per-match data ({args.n_matches} matches/topology)...")
    per_match = generate_per_match_data(results, n_matches=args.n_matches, seed=args.seed)
    total_matches = sum(len(v) for v in per_match.values())
    print(f"  Total match records: {total_matches}")

    # Step 4: Save everything
    print(f"\n[4/4] Saving results to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    save_summary_csv(results, os.path.join(args.output_dir, "synthetic_results.csv"))
    save_per_match_jsonl(per_match, args.output_dir)
    save_topology_catalog(all_shapes, os.path.join(args.output_dir, "topology_catalog.json"))

    print(f"\nDone! All synthetic data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
