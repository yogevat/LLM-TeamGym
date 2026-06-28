#!/usr/bin/env python3
"""Generate realistic synthetic consistency data for Paper 3 (Consistency).

Produces a CSV with aggregated consistency metrics across domains, scales,
topologies, and conditions (single_llm / ensemble / role-structured team).
Each row represents one prompt's aggregated statistics over R=30 repeats.

Usage:
    python generate_synthetic.py                        # defaults
    python generate_synthetic.py --output ../data/consistency_synthetic.csv
    python generate_synthetic.py --n_prompts 200 --seed 99
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Baseline profiles (mean, std) for each metric under each condition
# ---------------------------------------------------------------------------
CONDITION_PROFILES = {
    "single_llm": {
        "answer_entropy": (1.20, 0.15),
        "flip_rate":      (0.35, 0.06),
        "self_consistency": (0.55, 0.07),
    },
    "ensemble": {
        "answer_entropy": (0.90, 0.12),
        "flip_rate":      (0.25, 0.05),
        "self_consistency": (0.65, 0.06),
    },
    "team": {
        # Base for n_agents=4; scaled by _team_scale() for larger teams.
        "answer_entropy": (0.70, 0.10),
        "flip_rate":      (0.20, 0.04),
        "self_consistency": (0.75, 0.05),
    },
}

# Topology families available for teams
TOPOLOGY_FAMILIES = ["chain", "star", "full", "hierarchy"]

# Per-agent token cost model (tokens per prompt)
TOKENS_PER_AGENT = 800
COST_PER_1K_TOKENS = 0.003  # USD


def _team_scale(n_agents: int) -> dict[str, float]:
    """Return multiplicative adjustments for team metrics as team size grows.

    Consistency improves with more agents but with diminishing returns
    (logarithmic scaling).  Cost grows linearly.
    """
    # Diminishing improvement factor: log(n)/log(4) so that n=4 -> 1.0
    improvement = np.log(n_agents) / np.log(4)
    return {
        "entropy_mult":    1.0 / improvement,          # lower is better
        "flip_mult":       1.0 / improvement,          # lower is better
        "sc_add":          0.05 * (improvement - 1.0),  # higher is better
    }


def _paraphrase_penalty() -> dict[str, float]:
    """Extra noise/shift for paraphrase variation vs seed variation."""
    return {
        "entropy_add": 0.08,
        "flip_add":    0.06,
        "sc_add":     -0.04,
    }


def generate(
    n_prompts: int = 100,
    n_agents_list: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the full synthetic dataset and return a DataFrame."""
    if n_agents_list is None:
        n_agents_list = [4, 6, 8]

    rng = np.random.default_rng(seed)
    domains = ["games", "babyvision"]
    variation_types = ["seed", "paraphrase"]

    rows: list[dict] = []

    for domain in domains:
        # Slight domain offset so games and babyvision are not identical
        domain_offset = 0.0 if domain == "games" else 0.05

        for prompt_id in range(1, n_prompts + 1):
            for var_type in variation_types:
                para_pen = _paraphrase_penalty() if var_type == "paraphrase" else {
                    "entropy_add": 0.0, "flip_add": 0.0, "sc_add": 0.0
                }

                # --- single_llm ---
                prof = CONDITION_PROFILES["single_llm"]
                entropy = np.clip(
                    rng.normal(prof["answer_entropy"][0] + domain_offset + para_pen["entropy_add"],
                               prof["answer_entropy"][1]),
                    0.0, None,
                )
                flip = np.clip(
                    rng.normal(prof["flip_rate"][0] + domain_offset * 0.3 + para_pen["flip_add"],
                               prof["flip_rate"][1]),
                    0.0, 1.0,
                )
                sc = np.clip(
                    rng.normal(prof["self_consistency"][0] - domain_offset * 0.3 + para_pen["sc_add"],
                               prof["self_consistency"][1]),
                    0.0, 1.0,
                )
                tokens = int(rng.normal(TOKENS_PER_AGENT, 80))
                rows.append({
                    "domain": domain,
                    "n_agents": 1,
                    "topology_family": "none",
                    "condition": "single_llm",
                    "prompt_id": prompt_id,
                    "variation_type": var_type,
                    "answer_entropy": round(entropy, 4),
                    "flip_rate": round(flip, 4),
                    "self_consistency": round(sc, 4),
                    "total_tokens": tokens,
                    "total_cost": round(tokens * COST_PER_1K_TOKENS / 1000, 6),
                })

                # --- equal-cost ensemble ---
                for n_ag in n_agents_list:
                    prof_e = CONDITION_PROFILES["ensemble"]
                    # Ensemble gets marginal benefit from more copies
                    ens_scale = 1.0 / (1.0 + 0.05 * (n_ag - 4))
                    entropy_e = np.clip(
                        rng.normal(prof_e["answer_entropy"][0] * ens_scale + domain_offset + para_pen["entropy_add"],
                                   prof_e["answer_entropy"][1]),
                        0.0, None,
                    )
                    flip_e = np.clip(
                        rng.normal(prof_e["flip_rate"][0] * ens_scale + domain_offset * 0.2 + para_pen["flip_add"],
                                   prof_e["flip_rate"][1]),
                        0.0, 1.0,
                    )
                    sc_e = np.clip(
                        rng.normal(prof_e["self_consistency"][0] + 0.02 * (n_ag - 4) + para_pen["sc_add"],
                                   prof_e["self_consistency"][1]),
                        0.0, 1.0,
                    )
                    tokens_e = int(rng.normal(TOKENS_PER_AGENT * n_ag, 80 * n_ag))
                    rows.append({
                        "domain": domain,
                        "n_agents": n_ag,
                        "topology_family": "none",
                        "condition": "ensemble",
                        "prompt_id": prompt_id,
                        "variation_type": var_type,
                        "answer_entropy": round(entropy_e, 4),
                        "flip_rate": round(flip_e, 4),
                        "self_consistency": round(sc_e, 4),
                        "total_tokens": tokens_e,
                        "total_cost": round(tokens_e * COST_PER_1K_TOKENS / 1000, 6),
                    })

                # --- role-structured teams ---
                for n_ag in n_agents_list:
                    scale = _team_scale(n_ag)
                    for topo in TOPOLOGY_FAMILIES:
                        prof_t = CONDITION_PROFILES["team"]
                        # Topology-specific slight offsets
                        topo_offsets = {
                            "chain":     {"e": 0.03, "f": 0.02, "s": -0.01},
                            "star":      {"e": -0.02, "f": -0.01, "s": 0.02},
                            "full":      {"e": -0.05, "f": -0.03, "s": 0.03},
                            "hierarchy": {"e": 0.00, "f": 0.00, "s": 0.01},
                        }
                        t_off = topo_offsets[topo]

                        entropy_t = np.clip(
                            rng.normal(
                                prof_t["answer_entropy"][0] * scale["entropy_mult"]
                                + domain_offset + para_pen["entropy_add"] + t_off["e"],
                                prof_t["answer_entropy"][1],
                            ),
                            0.0, None,
                        )
                        flip_t = np.clip(
                            rng.normal(
                                prof_t["flip_rate"][0] * scale["flip_mult"]
                                + domain_offset * 0.15 + para_pen["flip_add"] + t_off["f"],
                                prof_t["flip_rate"][1],
                            ),
                            0.0, 1.0,
                        )
                        sc_t = np.clip(
                            rng.normal(
                                prof_t["self_consistency"][0] + scale["sc_add"]
                                + para_pen["sc_add"] + t_off["s"],
                                prof_t["self_consistency"][1],
                            ),
                            0.0, 1.0,
                        )
                        tokens_t = int(rng.normal(TOKENS_PER_AGENT * n_ag, 80 * n_ag))
                        rows.append({
                            "domain": domain,
                            "n_agents": n_ag,
                            "topology_family": topo,
                            "condition": "team",
                            "prompt_id": prompt_id,
                            "variation_type": var_type,
                            "answer_entropy": round(entropy_t, 4),
                            "flip_rate": round(flip_t, 4),
                            "self_consistency": round(sc_t, 4),
                            "total_tokens": tokens_t,
                            "total_cost": round(tokens_t * COST_PER_1K_TOKENS / 1000, 6),
                        })

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic consistency data for Paper 3."
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "consistency_synthetic.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--n_prompts", type=int, default=100, help="Prompts per condition.")
    parser.add_argument("--n_agents", type=str, default="4,6,8",
                        help="Comma-separated team sizes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_agents_list = [int(x) for x in args.n_agents.split(",")]
    df = generate(n_prompts=args.n_prompts, n_agents_list=n_agents_list, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path.resolve()}")
    print(f"Domains:    {sorted(df['domain'].unique())}")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Scales:     {sorted(df['n_agents'].unique())}")
    print(f"Topologies: {sorted(df['topology_family'].unique())}")
    print()

    # Quick sanity summary
    summary = (
        df.groupby(["condition", "variation_type"])[
            ["answer_entropy", "flip_rate", "self_consistency"]
        ]
        .mean()
        .round(3)
    )
    print(summary)


if __name__ == "__main__":
    main()
