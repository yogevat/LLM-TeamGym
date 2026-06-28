#!/usr/bin/env python3
"""Run the consistency battery for Paper 3 (Consistency).

Queries each prompt R times under two variation conditions (seed variation
and paraphrase variation) and computes:
  - Answer entropy  H = -sum(p * log(p))
  - Flip rate       fraction of run-pairs with different answers
  - Self-consistency  modal answer share (majority vote fraction)

Test conditions:
  - single_llm:     one LLM instance
  - ensemble:       equal-cost ensemble (N independent copies, majority vote)
  - team:           role-structured team with communication topology

Usage:
    python run_consistency.py --domain games --n_agents 4,6,8 --model mock --R 30 --output_dir ../data/
    python run_consistency.py --domain babyvision --model gpt-4o-mini --R 30 --output_dir ../data/
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Prompt banks (small built-in set; extend or load from file for production)
# ---------------------------------------------------------------------------
PROMPT_BANKS: dict[str, list[dict[str, str]]] = {
    "games": [
        {
            "base": "What is the optimal opening move in chess?",
            "paraphrase": "Which first move is strongest when starting a chess game?",
        },
        {
            "base": "How does the minimax algorithm work in tic-tac-toe?",
            "paraphrase": "Explain minimax search applied to noughts and crosses.",
        },
        {
            "base": "What is the Nash equilibrium in rock-paper-scissors?",
            "paraphrase": "Describe the equilibrium strategy for rock-paper-scissors.",
        },
        {
            "base": "Why is Go harder for AI than chess?",
            "paraphrase": "What makes the game of Go more challenging for artificial intelligence compared to chess?",
        },
        {
            "base": "What is the expected value of a fair six-sided die roll?",
            "paraphrase": "Compute the mean outcome when rolling a standard die.",
        },
    ],
    "babyvision": [
        {
            "base": "At what age do infants develop depth perception?",
            "paraphrase": "When do babies start perceiving depth?",
        },
        {
            "base": "What is the visual acuity of a newborn?",
            "paraphrase": "How well can a newborn baby see?",
        },
        {
            "base": "How does face recognition develop in infants?",
            "paraphrase": "Describe the developmental trajectory of infant face perception.",
        },
        {
            "base": "What role does contrast sensitivity play in early vision?",
            "paraphrase": "Why is contrast important for the developing visual system?",
        },
        {
            "base": "When do infants begin to track moving objects?",
            "paraphrase": "At what developmental stage can babies follow objects with their eyes?",
        },
    ],
}

# Role templates for team conditions
ROLE_TEMPLATES: dict[str, list[str]] = {
    "chain": ["analyst", "critic", "synthesizer", "presenter"],
    "star":  ["coordinator", "expert_a", "expert_b", "expert_c"],
    "full":  ["peer_1", "peer_2", "peer_3", "peer_4"],
    "hierarchy": ["lead", "senior", "junior_a", "junior_b"],
}

# ---------------------------------------------------------------------------
# Mock LLM backend (deterministic-ish for testing without API calls)
# ---------------------------------------------------------------------------

class MockLLM:
    """Deterministic mock that returns one of several plausible answers
    with controlled randomness, simulating real LLM stochasticity."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._call_count = 0

    def query(self, prompt: str, role: str | None = None) -> dict[str, Any]:
        """Return a mock answer and token count."""
        self._call_count += 1
        # Hash prompt to get a stable but varied answer set
        prompt_hash = hash(prompt) % 997
        n_options = 4
        options = [f"answer_{(prompt_hash + i) % 26}" for i in range(n_options)]

        # Role-structured agents are more likely to converge
        if role is not None:
            # Higher probability on the modal answer
            weights = np.array([0.55, 0.25, 0.12, 0.08])
        else:
            # Single LLM / ensemble: flatter distribution
            weights = np.array([0.40, 0.25, 0.20, 0.15])

        weights = weights / weights.sum()
        answer = self.rng.choice(options, p=weights)
        tokens = int(self.rng.normal(800, 80))
        return {"answer": answer, "tokens": max(tokens, 100)}


def _get_llm(model: str, seed: int) -> MockLLM:
    """Factory for LLM backends. Currently only mock is implemented."""
    if model == "mock":
        return MockLLM(seed=seed)
    raise NotImplementedError(
        f"Model '{model}' not implemented. Use --model mock for synthetic runs, "
        "or implement an API backend."
    )


# ---------------------------------------------------------------------------
# Consistency metrics
# ---------------------------------------------------------------------------

def answer_entropy(answers: list[str]) -> float:
    """Shannon entropy of the empirical answer distribution (natural log)."""
    counts = Counter(answers)
    n = len(answers)
    if n == 0:
        return 0.0
    probs = [c / n for c in counts.values()]
    return -sum(p * math.log(p) for p in probs if p > 0)


def flip_rate(answers: list[str]) -> float:
    """Fraction of consecutive run-pairs with different answers."""
    if len(answers) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(answers[:-1], answers[1:]) if a != b)
    return flips / (len(answers) - 1)


def self_consistency(answers: list[str]) -> float:
    """Modal answer share (majority-vote fraction)."""
    if not answers:
        return 0.0
    counts = Counter(answers)
    return counts.most_common(1)[0][1] / len(answers)


# ---------------------------------------------------------------------------
# Team execution logic
# ---------------------------------------------------------------------------

def _extend_roles(base_roles: list[str], n_agents: int) -> list[str]:
    """Extend a role template to fit the requested team size."""
    if n_agents <= len(base_roles):
        return base_roles[:n_agents]
    roles = list(base_roles)
    extra = n_agents - len(base_roles)
    for i in range(extra):
        roles.append(f"{base_roles[i % len(base_roles)]}_extra_{i}")
    return roles


def run_team(llm: MockLLM, prompt: str, roles: list[str]) -> dict[str, Any]:
    """Simulate a role-structured team processing one prompt.

    Each agent produces an answer; the team answer is the majority vote.
    """
    agent_answers = []
    total_tokens = 0
    for role in roles:
        result = llm.query(prompt, role=role)
        agent_answers.append(result["answer"])
        total_tokens += result["tokens"]

    # Majority vote
    counts = Counter(agent_answers)
    team_answer = counts.most_common(1)[0][1]
    final_answer = counts.most_common(1)[0][0]
    return {"answer": final_answer, "tokens": total_tokens}


def run_ensemble(llm: MockLLM, prompt: str, n_copies: int) -> dict[str, Any]:
    """Simulate an equal-cost ensemble (N independent LLM copies, majority vote)."""
    answers = []
    total_tokens = 0
    for _ in range(n_copies):
        result = llm.query(prompt, role=None)
        answers.append(result["answer"])
        total_tokens += result["tokens"]

    counts = Counter(answers)
    final_answer = counts.most_common(1)[0][0]
    return {"answer": final_answer, "tokens": total_tokens}


def run_single(llm: MockLLM, prompt: str) -> dict[str, Any]:
    """Single LLM query."""
    return llm.query(prompt, role=None)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    domain: str,
    n_agents_list: list[int],
    model: str,
    R: int,
    topologies: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full consistency battery for one domain."""
    if topologies is None:
        topologies = list(ROLE_TEMPLATES.keys())

    prompts = PROMPT_BANKS.get(domain)
    if prompts is None:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(PROMPT_BANKS.keys())}")

    rows: list[dict] = []

    for prompt_idx, prompt_pair in enumerate(prompts):
        prompt_id = prompt_idx + 1

        for var_type in ["seed", "paraphrase"]:
            prompt_text = prompt_pair["base"] if var_type == "seed" else prompt_pair["paraphrase"]

            # --- Single LLM ---
            answers_single: list[str] = []
            tokens_single = 0
            for r in range(R):
                llm = _get_llm(model, seed=prompt_id * 1000 + r)
                res = run_single(llm, prompt_text)
                answers_single.append(res["answer"])
                tokens_single += res["tokens"]

            rows.append({
                "domain": domain,
                "n_agents": 1,
                "topology_family": "none",
                "condition": "single_llm",
                "prompt_id": prompt_id,
                "variation_type": var_type,
                "answer_entropy": round(answer_entropy(answers_single), 4),
                "flip_rate": round(flip_rate(answers_single), 4),
                "self_consistency": round(self_consistency(answers_single), 4),
                "total_tokens": tokens_single,
                "total_cost": round(tokens_single * 0.003 / 1000, 6),
            })

            # --- Equal-cost ensemble & Role-structured teams ---
            for n_ag in n_agents_list:
                # Ensemble
                answers_ens: list[str] = []
                tokens_ens = 0
                for r in range(R):
                    llm = _get_llm(model, seed=prompt_id * 1000 + r + 500)
                    res = run_ensemble(llm, prompt_text, n_ag)
                    answers_ens.append(res["answer"])
                    tokens_ens += res["tokens"]

                rows.append({
                    "domain": domain,
                    "n_agents": n_ag,
                    "topology_family": "none",
                    "condition": "ensemble",
                    "prompt_id": prompt_id,
                    "variation_type": var_type,
                    "answer_entropy": round(answer_entropy(answers_ens), 4),
                    "flip_rate": round(flip_rate(answers_ens), 4),
                    "self_consistency": round(self_consistency(answers_ens), 4),
                    "total_tokens": tokens_ens,
                    "total_cost": round(tokens_ens * 0.003 / 1000, 6),
                })

                # Teams per topology
                for topo in topologies:
                    base_roles = ROLE_TEMPLATES.get(topo, ["agent"])
                    roles = _extend_roles(base_roles, n_ag)

                    answers_team: list[str] = []
                    tokens_team = 0
                    for r in range(R):
                        llm = _get_llm(model, seed=prompt_id * 1000 + r + 1000)
                        res = run_team(llm, prompt_text, roles)
                        answers_team.append(res["answer"])
                        tokens_team += res["tokens"]

                    rows.append({
                        "domain": domain,
                        "n_agents": n_ag,
                        "topology_family": topo,
                        "condition": "team",
                        "prompt_id": prompt_id,
                        "variation_type": var_type,
                        "answer_entropy": round(answer_entropy(answers_team), 4),
                        "flip_rate": round(flip_rate(answers_team), 4),
                        "self_consistency": round(self_consistency(answers_team), 4),
                        "total_tokens": tokens_team,
                        "total_cost": round(tokens_team * 0.003 / 1000, 6),
                    })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Paper 3 consistency battery."
    )
    parser.add_argument("--domain", type=str, default="games",
                        choices=list(PROMPT_BANKS.keys()),
                        help="Evaluation domain.")
    parser.add_argument("--n_agents", type=str, default="4,6,8",
                        help="Comma-separated team sizes.")
    parser.add_argument("--model", type=str, default="mock",
                        help="LLM backend (mock for synthetic).")
    parser.add_argument("--R", type=int, default=30,
                        help="Number of repeats per prompt.")
    parser.add_argument("--topologies", type=str, default=None,
                        help="Comma-separated topology families (default: all).")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data"),
                        help="Output directory for CSV results.")
    args = parser.parse_args()

    n_agents_list = [int(x) for x in args.n_agents.split(",")]
    topologies = args.topologies.split(",") if args.topologies else None

    print(f"Running consistency battery:")
    print(f"  Domain:     {args.domain}")
    print(f"  Scales:     {n_agents_list}")
    print(f"  Model:      {args.model}")
    print(f"  R:          {args.R}")
    print(f"  Topologies: {topologies or 'all'}")
    print()

    df = run_experiment(
        domain=args.domain,
        n_agents_list=n_agents_list,
        model=args.model,
        R=args.R,
        topologies=topologies,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"consistency_{args.domain}_{args.model}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows -> {out_path.resolve()}")
    print()

    # Summary
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
