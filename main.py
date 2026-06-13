"""
main.py — LLM-TeamGym demo: 2v2 Team Fish with live Pygame visualisation.

Run with:
    python main.py                    # Pygame window, random agents
    python main.py --no-render        # Headless, prints terminal log only
    python main.py --greedy           # Greedy agents instead of random
    python main.py --rows 8 --cols 8  # Larger grid
    python main.py --seed 42          # Reproducible game
    python main.py --delay 0.6        # Slower step speed (seconds)
    python main.py --matches 5        # Run 5 matches (tournament mode, headless)
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from llm_team_gym.core.base_agent import BaseAgent, GreedyAgent, RandomAgent
from llm_team_gym.envs.tournament import MatchRunner, TournamentRunner
from llm_team_gym.games.team_fish import TeamFishGame


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def build_agents(use_greedy: bool = False, seed_offset: int = 0) -> List[BaseAgent]:
    cls = GreedyAgent if use_greedy else RandomAgent

    if cls is RandomAgent:
        return [
            RandomAgent("A1", "team_A", seed=seed_offset + 0),
            RandomAgent("A2", "team_A", seed=seed_offset + 1),
            RandomAgent("B1", "team_B", seed=seed_offset + 2),
            RandomAgent("B2", "team_B", seed=seed_offset + 3),
        ]
    else:
        return [
            GreedyAgent("A1", "team_A"),
            GreedyAgent("A2", "team_A"),
            GreedyAgent("B1", "team_B"),
            GreedyAgent("B2", "team_B"),
        ]


# ---------------------------------------------------------------------------
# Pretty terminal summary
# ---------------------------------------------------------------------------

def print_match_summary(record) -> None:
    print("\n" + "=" * 60)
    print(f"  MATCH COMPLETE  —  {record.match_id}")
    print("=" * 60)
    for team, score in record.final_team_scores.items():
        marker = "  << WINNER" if team == record.winner else ""
        print(f"  {team:>10}   score = {score:.0f}{marker}")
    print(f"\n  Total steps : {record.total_steps}")
    print(f"  Duration    : {record.duration():.2f}s")
    print("=" * 60 + "\n")


def print_tournament_summary(stats: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  TOURNAMENT COMPLETE  —  {stats['n_matches']} matches")
    print("=" * 60)
    for team in sorted(stats["win_rates"], key=lambda t: stats["win_rates"][t], reverse=True):
        wr  = stats["win_rates"][team]
        avg = stats["avg_scores"][team]
        wins = stats["win_counts"].get(team, 0)
        print(f"  {team:>10}   wins={wins}   win%={wr*100:.1f}%   avg_score={avg:.1f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-TeamGym · 2v2 Team Fish Demo")
    parser.add_argument("--rows",      type=int,   default=6,    help="Grid rows")
    parser.add_argument("--cols",      type=int,   default=6,    help="Grid columns")
    parser.add_argument("--seed",      type=int,   default=None, help="RNG seed")
    parser.add_argument("--delay",     type=float, default=0.35, help="Seconds between steps when rendering")
    parser.add_argument("--greedy",    action="store_true",      help="Use GreedyAgents instead of Random")
    parser.add_argument("--no-render", action="store_true",      help="Disable Pygame window")
    parser.add_argument("--matches",   type=int,   default=1,    help="Number of matches (>1 = tournament, always headless)")
    parser.add_argument("--log-dir",   type=str,   default=None, help="Directory for JSONL transcripts")
    args = parser.parse_args()

    render = not args.no_render and args.matches == 1

    if args.matches > 1:
        # ---- Tournament mode ----
        print(f"\nStarting tournament: {args.matches} matches, "
              f"{'greedy' if args.greedy else 'random'} agents\n")

        def game_factory():
            return TeamFishGame(rows=args.rows, cols=args.cols, seed=args.seed)

        match_counter = [0]

        def agent_factory():
            match_counter[0] += 1
            return build_agents(use_greedy=args.greedy, seed_offset=match_counter[0] * 10)

        runner = TournamentRunner(
            game_factory=game_factory,
            agent_factory=agent_factory,
            n_matches=args.matches,
            runner_kwargs={
                "render": False,
                "verbose": True,
                "output_dir": args.log_dir,
            },
        )
        stats = runner.run()
        print_tournament_summary(stats)

    else:
        # ---- Single match mode ----
        print("\nStarting 2v2 Team Fish match…")
        print(f"  Grid : {args.rows}×{args.cols}")
        print(f"  Seed : {args.seed}")
        print(f"  Agents: {'Greedy' if args.greedy else 'Random'}")
        print(f"  Render: {render}\n")

        game = TeamFishGame(rows=args.rows, cols=args.cols, seed=args.seed)
        agents = build_agents(use_greedy=args.greedy)

        runner = MatchRunner(
            game=game,
            agents=agents,
            render=render,
            step_delay=args.delay if render else 0.0,
            output_dir=args.log_dir,
            verbose=True,
        )

        try:
            record = runner.run()
        except KeyboardInterrupt:
            print("\n[Interrupted by user]")
            game.close()
            sys.exit(0)

        print_match_summary(record)


if __name__ == "__main__":
    main()
