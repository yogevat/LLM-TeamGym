#!/usr/bin/env python3
"""
Patch-granularity scaling sweep for BabyVision / Paper 2.

Sweeps agent count from 1 to hundreds, selecting the appropriate topology
generation method at each scale:

  * N <= 8:   Exhaustive canonical-form enumeration (exact).
  * 9 <= N <= 99: Geometric-spectral generation + spectral clustering.
  * N >= 100: Controller-guided learned composition (LLM or random fallback).

CLI
---
    python run_scaling.py --n_agents 4,8,16,32,64,128 --model mock --output_dir ../data/

Each scale produces a JSON results file in ``output_dir/scaling_N<n>.json``
containing topology descriptors, spectral fingerprints, and (when available)
mock evaluation scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so imports work when run from the experiments/ dir.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PAPER_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.abspath(os.path.join(_PAPER_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_team_gym.topology.enumeration import (
    enumerate_shapes,
    shape_to_topology,
)
from llm_team_gym.topology.geometric import (
    GeometricGenerator,
    TopologyDescriptor,
)
from llm_team_gym.topology.spectral import (
    SpectralAnalyzer,
    TopologyClusterer,
)
from llm_team_gym.topology.learned import (
    TopologyController,
    TopologySpec,
    _compute_fingerprint,
)
from llm_team_gym.core.topology import (
    TopologyGraph,
    random_topology,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topology family labels used across the paper
# ---------------------------------------------------------------------------
TOPOLOGY_FAMILIES = [
    "star",       # complete bipartite (all W connect to all V)
    "chain",      # linear bipartite path
    "ring",       # cyclic bipartite
    "sparse",     # low-density random
    "dense",      # high-density random
    "clustered",  # geometric-spectral cluster exemplars
    "learned",    # controller-proposed topologies
]


# ===================================================================
# Scale-specific generators
# ===================================================================


def _run_exhaustive(n_agents: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Exhaustive enumeration for small teams (N <= 8).

    Iterates over all valid (W, V) splits with W + V = N - 1 (one Thinker),
    enumerates canonical shapes, and records topology descriptors.

    Returns a list of result dicts (one per shape).
    """
    results: List[Dict[str, Any]] = []
    rng = np.random.RandomState(seed)

    for W in range(1, n_agents - 1):
        V = n_agents - 1 - W
        if V < 1:
            continue

        shapes = enumerate_shapes(W, V, relaxed=True)
        for idx, shape_matrix in enumerate(shapes):
            desc = shape_to_topology(shape_matrix)

            # Build a full adjacency to compute spectral features
            topo = TopologyGraph.from_matrix(shape_matrix)
            full_adj = topo.to_adjacency_matrix()
            sa = SpectralAnalyzer(full_adj)
            spectral_features = sa.feature_vector().tolist()

            # Classify into a topology family heuristically
            density = desc["density"]
            if density >= 0.95:
                family = "star"
            elif density <= 0.2:
                family = "sparse"
            elif density >= 0.7:
                family = "dense"
            else:
                family = "chain"

            results.append({
                "n_agents": n_agents,
                "method": "exhaustive",
                "W": W,
                "V": V,
                "shape_index": idx,
                "topology_family": family,
                "density": density,
                "n_edges": len(desc["edges"]),
                "worker_degrees": desc["worker_degrees"],
                "validator_degrees": desc["validator_degrees"],
                "spectral_features": spectral_features,
                "mock_score": float(rng.uniform(0.3, 0.9)),
            })

    logger.info(
        "Exhaustive enumeration for N=%d: %d shapes across all (W,V) splits.",
        n_agents, len(results),
    )
    return results


def _run_geometric_spectral(
    n_agents: int,
    n_topologies: int = 50,
    n_clusters: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Geometric generation + spectral clustering for medium teams (9-99).

    1. Generate ``n_topologies`` topologies via GeometricGenerator.
    2. Compute spectral feature vectors.
    3. Cluster with TopologyClusterer.
    4. Report cluster-level statistics.

    Returns a list of result dicts (one per topology).
    """
    gen = GeometricGenerator(n_agents=n_agents, seed=seed)
    batch = gen.generate_batch(n_topologies)

    # Compute spectral features for every topology
    feature_matrix = np.zeros((len(batch), 6))
    results: List[Dict[str, Any]] = []

    for i, (agents, adj) in enumerate(batch):
        sa = SpectralAnalyzer(adj)
        fv = sa.feature_vector()
        feature_matrix[i] = fv

        td = TopologyDescriptor(adj, agents)
        desc_vec = td.feature_vector()

        # Classify family based on density and spectral ratio
        density = float(np.sum(adj > 0)) / max(adj.shape[0] ** 2, 1)
        eigen_ratio = fv[1]

        if density >= 0.6:
            family = "dense"
        elif density <= 0.15:
            family = "sparse"
        elif eigen_ratio > 0.5:
            family = "ring"
        else:
            family = "chain"

        results.append({
            "n_agents": n_agents,
            "method": "geometric_spectral",
            "topology_index": i,
            "topology_family": family,
            "density": round(density, 4),
            "n_edges": int(np.sum(adj > 0)),
            "spectral_features": fv.tolist(),
            "descriptor_features": desc_vec.tolist(),
            "lambda_2": float(fv[0]),
            "eigen_ratio": float(fv[1]),
            "diameter": int(fv[4]),
            "clustering_coeff": float(fv[5]),
        })

    # Cluster the topologies
    clusterer = TopologyClusterer(
        n_clusters=min(n_clusters, len(batch)),
        random_state=seed,
    )
    labels = clusterer.fit(feature_matrix)
    sil_score = clusterer.silhouette_score(feature_matrix, labels)

    for i, res in enumerate(results):
        res["cluster_id"] = int(labels[i])

    logger.info(
        "Geometric-spectral for N=%d: %d topologies in %d clusters "
        "(silhouette=%.3f).",
        n_agents, len(batch), len(np.unique(labels)), sil_score,
    )

    return results


def _run_learned_composition(
    n_agents: int,
    n_candidates: int = 20,
    n_iterations: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Controller-guided topology search for large teams (N >= 100).

    Uses TopologyController in random-fallback mode (no LLM backend)
    to propose and fingerprint topologies.  In production, supply an
    LLM agent for guided generation.

    Returns a list of result dicts (one per candidate).
    """
    # Compute role split: ~50% workers, ~40% validators, ~10% thinkers
    n_thinkers = max(1, n_agents // 10)
    remaining = n_agents - n_thinkers
    n_workers = max(1, int(remaining * 0.55))
    n_validators = max(1, remaining - n_workers)
    # Adjust to match exact total
    diff = n_agents - (n_workers + n_validators + n_thinkers)
    n_workers += diff

    spec = TopologySpec(
        n_agents=n_agents,
        n_workers=n_workers,
        n_validators=n_validators,
        n_thinkers=n_thinkers,
        sparsity_target=0.3,
        task_description="BabyVision patch-level visual reasoning",
    )

    controller = TopologyController(llm_agent=None, seed=seed)
    rng = np.random.RandomState(seed)

    results: List[Dict[str, Any]] = []
    best_score = -1.0

    for iteration in range(n_iterations):
        for c in range(n_candidates):
            proposal = controller.propose(spec)
            adj = proposal["adjacency"]
            fp = _compute_fingerprint(adj)

            # Mock evaluation score (in production, run actual benchmark)
            mock_score = float(rng.beta(3, 2))

            # Update cluster context based on best performers
            if mock_score > best_score:
                best_score = mock_score
                context = (
                    f"Best topology so far (score={best_score:.3f}) has "
                    f"density={fp['density']:.3f}, "
                    f"degree_mean={fp['degree_mean']:.2f}. "
                    f"Generate similar structures."
                )
                controller.set_cluster_context(context)

            # Classify family
            if fp["density"] >= 0.6:
                family = "dense"
            elif fp["density"] <= 0.15:
                family = "sparse"
            elif fp["degree_std"] < 0.5:
                family = "ring"
            else:
                family = "learned"

            results.append({
                "n_agents": n_agents,
                "method": "learned_composition",
                "iteration": iteration,
                "candidate": c,
                "topology_family": family,
                "source": proposal["source"],
                "fingerprint": fp,
                "density": fp["density"],
                "n_edges": fp["n_edges"],
                "degree_mean": fp["degree_mean"],
                "eigenvalues_top3": fp["eigenvalues_top3"],
                "mock_score": mock_score,
            })

    logger.info(
        "Learned composition for N=%d: %d candidates across %d iterations "
        "(best_score=%.3f).",
        n_agents, len(results), n_iterations, best_score,
    )

    return results


# ===================================================================
# Main dispatcher
# ===================================================================


def run_scaling_sweep(
    n_agents_list: List[int],
    model: str = "mock",
    output_dir: str = "../data/",
    seed: int = 42,
) -> Dict[int, List[Dict[str, Any]]]:
    """Run the full scaling sweep across all requested agent counts.

    Parameters
    ----------
    n_agents_list : list of int
        Agent counts to sweep (e.g. [4, 8, 16, 32, 64, 128]).
    model : str
        Model identifier.  ``"mock"`` uses random evaluation scores.
    output_dir : str
        Directory for JSON output files.
    seed : int
        Base random seed.

    Returns
    -------
    all_results : dict mapping n_agents -> list of result dicts
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[int, List[Dict[str, Any]]] = {}

    for n in sorted(n_agents_list):
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Scaling sweep: N = %d agents", n)

        if n <= 1:
            logger.warning("N=%d is too small for multi-agent; skipping.", n)
            continue

        # Select generation method by scale
        if n <= 8:
            logger.info("  Method: exhaustive enumeration (N <= 8)")
            results = _run_exhaustive(n, seed=seed)
        elif n <= 99:
            logger.info("  Method: geometric-spectral generation (9 <= N <= 99)")
            n_topos = min(50, n * 3)
            results = _run_geometric_spectral(
                n, n_topologies=n_topos, seed=seed,
            )
        else:
            logger.info("  Method: controller-guided learned composition (N >= 100)")
            results = _run_learned_composition(n, seed=seed)

        elapsed = time.time() - t0
        logger.info(
            "  Completed N=%d: %d topologies in %.2fs",
            n, len(results), elapsed,
        )

        all_results[n] = results

        # Save per-scale JSON
        out_path = os.path.join(output_dir, f"scaling_N{n}.json")
        with open(out_path, "w") as fh:
            json.dump(
                {
                    "n_agents": n,
                    "model": model,
                    "method": results[0]["method"] if results else "none",
                    "n_topologies": len(results),
                    "elapsed_s": round(elapsed, 3),
                    "seed": seed,
                    "topologies": results,
                },
                fh,
                indent=2,
                default=str,
            )
        logger.info("  Saved -> %s", out_path)

    # Save consolidated manifest
    manifest_path = os.path.join(output_dir, "scaling_manifest.json")
    manifest = {
        "n_agents_list": sorted(n_agents_list),
        "model": model,
        "seed": seed,
        "scales": {
            n: {
                "n_topologies": len(res),
                "method": res[0]["method"] if res else "none",
            }
            for n, res in all_results.items()
        },
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest saved -> %s", manifest_path)

    return all_results


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BabyVision patch-granularity scaling sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_scaling.py --n_agents 4,8,16,32,64,128 "
            "--model mock --output_dir ../data/\n"
            "  python run_scaling.py --n_agents 3,4,5,6,7,8 "
            "--model mock --output_dir ../data/small/\n"
        ),
    )
    parser.add_argument(
        "--n_agents",
        type=str,
        default="4,8,16,32,64,128",
        help="Comma-separated list of agent counts to sweep (default: 4,8,16,32,64,128)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        help="Model identifier (default: mock)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_PAPER_DIR, "data"),
        help="Output directory for results (default: ../data/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    n_agents_list = [int(x.strip()) for x in args.n_agents.split(",")]

    logger.info("BabyVision Scaling Sweep")
    logger.info("  Agent counts: %s", n_agents_list)
    logger.info("  Model: %s", args.model)
    logger.info("  Output: %s", args.output_dir)
    logger.info("  Seed: %d", args.seed)

    results = run_scaling_sweep(
        n_agents_list=n_agents_list,
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("SCALING SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'N':>6}  {'Method':<25}  {'Topologies':>10}  {'Families':>8}")
    print("-" * 70)
    for n in sorted(results.keys()):
        entries = results[n]
        method = entries[0]["method"] if entries else "none"
        families = set(e.get("topology_family", "?") for e in entries)
        print(f"{n:>6}  {method:<25}  {len(entries):>10}  {len(families):>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
