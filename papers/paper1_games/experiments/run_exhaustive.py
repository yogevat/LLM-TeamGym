#!/usr/bin/env python3
"""
Exhaustive topology sweep for Paper 1 (Games / Exhaustive).

Enumerates all 61 four-penguin topology shapes (W=4, V=1..3, relaxed)
and runs N matches of a specified game per topology, recording results
as JSONL.

Usage
-----
    python run_exhaustive.py --game team_fish --n_matches 50 --model mock --output_dir ../data/
    python run_exhaustive.py --game team_fish --n_matches 20 --model gpt-4o --output_dir ../data/

The script supports:
  --model mock   : fast synthetic random agents (no API calls)
  --model <name> : real LLM-backed agents via llm_team_gym.core.llm_agent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so llm_team_gym imports resolve
# regardless of the working directory.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_team_gym.core.base_agent import BaseAgent, RandomAgent
from llm_team_gym.core.base_game import Action, AgentID, BaseGame, Observation
from llm_team_gym.core.roles import (
    Role,
    RoleAssignment,
    TeamOrchestrator,
    Topology as RolesTopo,
    WorkerAgent,
    ValidatorAgent,
    ThinkerAgent,
)
from llm_team_gym.core.topology import TopologyGraph
from llm_team_gym.topology.enumeration import (
    enumerate_shapes,
    shape_to_topology,
)
from llm_team_gym.topology.spectral import SpectralAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GAME_REGISTRY: Dict[str, str] = {
    "team_fish": "llm_team_gym.games.team_fish",
    "connect_four": "llm_team_gym.games.connect_four",
    "dice_race": "llm_team_gym.games.dice_race",
    "hanabi": "llm_team_gym.games.hanabi",
    "mancala": "llm_team_gym.games.mancala",
    "othello": "llm_team_gym.games.othello",
    "extended_rps": "llm_team_gym.games.extended_rps",
}


def _shape_hash(B: np.ndarray) -> str:
    """Stable hex hash of a canonical shape matrix."""
    raw = B.tobytes()
    return hashlib.sha256(raw).hexdigest()[:12]


def _topology_id(W: int, V: int, idx: int, shape_hash: str) -> str:
    """Human-readable topology identifier."""
    return f"W{W}_V{V}_{idx:03d}_{shape_hash}"


def _enumerate_all_four_penguin_shapes() -> List[Dict[str, Any]]:
    """Enumerate all 61 four-penguin (W=4) shapes for V=1,2,3.

    Returns a list of dicts with keys:
        W, V, idx, shape_matrix, topology_id, shape_hash, n_edges,
        spectral_features
    """
    all_shapes: List[Dict[str, Any]] = []
    global_idx = 0

    for V in range(1, 4):
        shapes = enumerate_shapes(W=4, V=V, relaxed=True)
        for local_idx, B in enumerate(shapes):
            sh = _shape_hash(B)
            topo_id = _topology_id(4, V, global_idx, sh)

            # Build full adjacency for spectral analysis
            tg = TopologyGraph.from_matrix(B, n_thinkers=1)
            adj = tg.to_adjacency_matrix()
            sa = SpectralAnalyzer(adj)
            features = sa.feature_vector()

            desc = shape_to_topology(B)

            all_shapes.append({
                "W": 4,
                "V": V,
                "idx": global_idx,
                "local_idx": local_idx,
                "topology_id": topo_id,
                "shape_hash": sh,
                "shape_matrix": B.tolist(),
                "n_edges": int(B.sum()),
                "density": desc["density"],
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

    assert len(all_shapes) == 61, (
        f"Expected 61 four-penguin shapes, got {len(all_shapes)}"
    )
    return all_shapes


# ---------------------------------------------------------------------------
# Game instantiation
# ---------------------------------------------------------------------------

def _load_game(game_name: str) -> BaseGame:
    """Import and instantiate a game by registry name."""
    if game_name not in GAME_REGISTRY:
        raise ValueError(
            f"Unknown game '{game_name}'. Available: {list(GAME_REGISTRY)}"
        )
    module_path = GAME_REGISTRY[game_name]
    import importlib
    mod = importlib.import_module(module_path)

    # Convention: the module defines a class whose name is the CamelCase
    # version of the module name, or has a create_game() factory.
    if hasattr(mod, "create_game"):
        return mod.create_game()

    # Try to find a BaseGame subclass in the module.
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseGame)
            and attr is not BaseGame
        ):
            return attr()
    raise RuntimeError(f"Could not find a BaseGame subclass in {module_path}")


# ---------------------------------------------------------------------------
# Agent instantiation
# ---------------------------------------------------------------------------

def _make_agents_mock(
    shape_desc: Dict[str, Any],
    game: BaseGame,
    seed: int,
) -> Tuple[Dict[AgentID, BaseAgent], RoleAssignment, RolesTopo]:
    """Create RandomAgent-backed Worker/Validator/Thinker team for one side.

    The topology's bipartite structure is between Workers and Validators;
    the Thinker connects to all Validators.
    """
    W = shape_desc["W"]
    V = shape_desc["V"]
    B = np.array(shape_desc["shape_matrix"], dtype=np.int8)

    # Agent IDs for Team A (the team under test).
    worker_ids = [f"A_w{i}" for i in range(W)]
    validator_ids = [f"A_v{j}" for j in range(V)]
    thinker_id = "A_thinker"

    # Build role assignment
    ra = RoleAssignment.from_lists(
        workers=worker_ids,
        validators=validator_ids,
        thinkers=[thinker_id],
    )

    # Build topology graph (roles module Topology)
    edges: List[Tuple[AgentID, AgentID]] = []
    for wi in range(W):
        for vj in range(V):
            if B[wi, vj]:
                edges.append((worker_ids[wi], validator_ids[vj]))
    # Thinker connects to all validators
    for vj in range(V):
        edges.append((thinker_id, validator_ids[vj]))

    topo = RolesTopo(edges)

    # Create agents
    rng_seed = seed
    agents: Dict[AgentID, BaseAgent] = {}
    for wid in worker_ids:
        inner = RandomAgent(agent_id=wid, team_id="team_A", seed=rng_seed)
        agents[wid] = WorkerAgent(inner)
        rng_seed += 1
    for vid in validator_ids:
        inner = RandomAgent(agent_id=vid, team_id="team_A", seed=rng_seed)
        agents[vid] = ValidatorAgent(inner)
        rng_seed += 1
    inner = RandomAgent(agent_id=thinker_id, team_id="team_A", seed=rng_seed)
    agents[thinker_id] = ThinkerAgent(inner)

    return agents, ra, topo


def _make_agents_llm(
    shape_desc: Dict[str, Any],
    game: BaseGame,
    model: str,
    seed: int,
) -> Tuple[Dict[AgentID, BaseAgent], RoleAssignment, RolesTopo]:
    """Create LLM-backed agents for the topology under test."""
    from llm_team_gym.core.llm_agent import OpenAIAgent

    W = shape_desc["W"]
    V = shape_desc["V"]
    B = np.array(shape_desc["shape_matrix"], dtype=np.int8)

    worker_ids = [f"A_w{i}" for i in range(W)]
    validator_ids = [f"A_v{j}" for j in range(V)]
    thinker_id = "A_thinker"

    ra = RoleAssignment.from_lists(
        workers=worker_ids,
        validators=validator_ids,
        thinkers=[thinker_id],
    )

    edges: List[Tuple[AgentID, AgentID]] = []
    for wi in range(W):
        for vj in range(V):
            if B[wi, vj]:
                edges.append((worker_ids[wi], validator_ids[vj]))
    for vj in range(V):
        edges.append((thinker_id, validator_ids[vj]))
    topo = RolesTopo(edges)

    agents: Dict[AgentID, BaseAgent] = {}
    for wid in worker_ids:
        inner = OpenAIAgent(
            agent_id=wid, team_id="team_A", model=model, temperature=0.7,
        )
        agents[wid] = WorkerAgent(inner)
    for vid in validator_ids:
        inner = OpenAIAgent(
            agent_id=vid, team_id="team_A", model=model, temperature=0.3,
        )
        agents[vid] = ValidatorAgent(inner)
    inner = OpenAIAgent(
        agent_id=thinker_id, team_id="team_A", model=model, temperature=0.5,
    )
    agents[thinker_id] = ThinkerAgent(inner)

    return agents, ra, topo


# ---------------------------------------------------------------------------
# Single-match runner (simplified -- uses TeamOrchestrator per decision)
# ---------------------------------------------------------------------------

def _run_single_match(
    game: BaseGame,
    agents: Dict[AgentID, BaseAgent],
    ra: RoleAssignment,
    topo: RolesTopo,
    match_seed: int,
) -> Dict[str, Any]:
    """Run one match and return a result dict.

    For mock mode this uses the TeamOrchestrator to select actions.
    The game is reset and stepped until termination.
    """
    orch = TeamOrchestrator(
        role_assignment=ra,
        agents=agents,
        topology=topo,
    )

    obs = game.reset()
    game_rules = game.get_game_rules()

    # Notify agents of episode start
    for aid, agent in agents.items():
        if aid in obs:
            agent.on_episode_start(obs[aid], game_rules)

    done_all = False
    total_steps = 0
    total_llm_calls = 0
    t_start = time.time()

    while not done_all:
        actions_dict: Dict[AgentID, Action] = {}

        for aid in game.all_agents:
            legal = game.get_legal_moves(aid)
            if not legal:
                continue
            text_state = game.get_text_state(aid)

            # If this agent is in our orchestrated team, use the orchestrator
            if aid in agents:
                result = orch.run_round(
                    observation=obs.get(aid, {}),
                    text_state=text_state,
                    legal_moves=legal,
                    game_rules=game_rules,
                )
                actions_dict[aid] = result.final_action
                total_llm_calls += result.llm_calls
            else:
                # Opponent team: use a RandomAgent fallback
                rng = RandomAgent(
                    agent_id=aid, team_id="opponent", seed=match_seed + total_steps
                )
                actions_dict[aid] = rng.choose_action(
                    obs.get(aid, {}), text_state, legal, game_rules
                )

        if not actions_dict:
            break

        obs_new, rewards, dones, infos = game.step(actions_dict)
        obs = obs_new
        total_steps += 1

        done_all = dones.get("__all__", False)
        if total_steps > 500:  # safety cap
            break

    elapsed = time.time() - t_start

    # Collect final scores
    team_scores: Dict[str, float] = {}
    for team_id, agent_ids in game.teams.items():
        team_scores[team_id] = sum(
            rewards.get(aid, 0.0) for aid in agent_ids
        )

    # Determine winner
    winner = max(team_scores, key=team_scores.get) if team_scores else None

    return {
        "total_steps": total_steps,
        "total_llm_calls": total_llm_calls,
        "elapsed_s": elapsed,
        "team_scores": team_scores,
        "winner": winner,
        "done": done_all,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_exhaustive(
    game_name: str,
    n_matches: int,
    model: str,
    output_dir: str,
    seed: int = 42,
) -> None:
    """Run the full exhaustive sweep over all 61 topologies."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Enumerating all four-penguin shapes...")
    all_shapes = _enumerate_all_four_penguin_shapes()
    print(f"  Found {len(all_shapes)} shapes")

    # Save topology catalog
    catalog_path = os.path.join(output_dir, "topology_catalog.json")
    with open(catalog_path, "w") as f:
        json.dump(all_shapes, f, indent=2, default=str)
    print(f"  Topology catalog saved to {catalog_path}")

    # Load game
    print(f"\nLoading game: {game_name}")
    game = _load_game(game_name)
    print(f"  Game loaded: {game.__class__.__name__}")

    total_matches = len(all_shapes) * n_matches
    match_count = 0
    t_sweep_start = time.time()

    for shape_desc in all_shapes:
        topo_id = shape_desc["topology_id"]
        jsonl_path = os.path.join(output_dir, f"{topo_id}.jsonl")

        print(f"\n{'='*60}")
        print(
            f"Topology {shape_desc['idx']+1}/61: {topo_id}  "
            f"(W={shape_desc['W']}, V={shape_desc['V']}, "
            f"edges={shape_desc['n_edges']})"
        )
        print(f"  Shape matrix: {shape_desc['shape_matrix']}")

        with open(jsonl_path, "w") as fh:
            # Write topology header
            header = {
                "type": "topology_meta",
                "topology_id": topo_id,
                **{k: v for k, v in shape_desc.items() if k != "shape_matrix"},
                "shape_matrix": shape_desc["shape_matrix"],
                "game": game_name,
                "model": model,
                "n_matches": n_matches,
            }
            fh.write(json.dumps(header, default=str) + "\n")

            wins = 0
            scores: List[float] = []
            llm_calls_list: List[int] = []
            latencies: List[float] = []

            for mi in range(n_matches):
                match_seed = seed + shape_desc["idx"] * 10000 + mi

                # Instantiate agents for this topology
                if model == "mock":
                    agents, ra, topo = _make_agents_mock(
                        shape_desc, game, seed=match_seed
                    )
                else:
                    agents, ra, topo = _make_agents_llm(
                        shape_desc, game, model=model, seed=match_seed
                    )

                try:
                    result = _run_single_match(
                        game, agents, ra, topo, match_seed
                    )
                except Exception as e:
                    logger.error(
                        f"Match {mi} for {topo_id} failed: {e}"
                    )
                    result = {
                        "total_steps": 0,
                        "total_llm_calls": 0,
                        "elapsed_s": 0.0,
                        "team_scores": {},
                        "winner": None,
                        "done": False,
                        "error": str(e),
                    }

                # Record match result
                match_record = {
                    "type": "match_result",
                    "match_idx": mi,
                    "topology_id": topo_id,
                    "seed": match_seed,
                    **result,
                }
                fh.write(json.dumps(match_record, default=str) + "\n")

                # Accumulate stats
                team_a_score = result.get("team_scores", {}).get("team_A", 0.0)
                scores.append(team_a_score)
                llm_calls_list.append(result["total_llm_calls"])
                latencies.append(result["elapsed_s"])
                if result.get("winner") == "team_A":
                    wins += 1

                match_count += 1

            # Write topology summary
            summary = {
                "type": "topology_summary",
                "topology_id": topo_id,
                "n_matches": n_matches,
                "win_rate": wins / n_matches if n_matches > 0 else 0.0,
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "std_score": float(np.std(scores)) if scores else 0.0,
                "avg_llm_calls": float(np.mean(llm_calls_list)) if llm_calls_list else 0,
                "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
            }
            fh.write(json.dumps(summary, default=str) + "\n")

        print(
            f"  Results: win_rate={summary['win_rate']:.3f}, "
            f"avg_score={summary['avg_score']:.2f}, "
            f"avg_llm_calls={summary['avg_llm_calls']:.1f}"
        )
        print(f"  Progress: {match_count}/{total_matches} matches complete")

    elapsed_total = time.time() - t_sweep_start
    print(f"\n{'='*60}")
    print(f"Exhaustive sweep complete!")
    print(f"  Total matches: {match_count}")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exhaustive topology sweep for Paper 1 (Games).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--game", type=str, default="team_fish",
        help=f"Game to play. Available: {list(GAME_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--n_matches", type=int, default=50,
        help="Number of matches per topology.",
    )
    parser.add_argument(
        "--model", type=str, default="mock",
        help="Agent model: 'mock' for synthetic random, or an LLM model name.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/",
        help="Directory for output JSONL files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve output_dir relative to script location
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(
            os.path.join(_SCRIPT_DIR, args.output_dir)
        )

    run_exhaustive(
        game_name=args.game,
        n_matches=args.n_matches,
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
