"""
Smoke-test all 7 new games: verifies reset, legal_moves, step, and game completion
using RandomAgent-equivalent logic — no Pygame window required.
"""

import random
import sys
import traceback

from llm_team_gym.core.base_agent import RandomAgent
from llm_team_gym.envs.tournament import MatchRunner
from llm_team_gym.games import (
    ConnectFourGame,
    ExtendedTicTacToeGame,
    MancalaGame,
    OthelloGame,
    GomokuGame,
    MicroChessGame,
    CaptureTheFlagGame,
)


def run_game(game, agents, label, max_steps=500):
    runner = MatchRunner(
        game=game,
        agents=agents,
        render=False,
        verbose=False,
        match_id=f"test_{label}",
    )
    try:
        record = runner.run()
        total  = record.total_steps
        winner = record.winner or "draw"
        scores = record.final_team_scores
        print(f"  [OK]  {label:<30}  steps={total:<4}  winner={winner}  scores={scores}")
        return True
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


def main():
    rng = random.Random(42)
    results = []

    # ------------------------------------------------------------------
    print("\n=== Connect Four ===")
    g = ConnectFourGame()
    agents = [RandomAgent("player_1", "player_1", seed=1),
              RandomAgent("player_2", "player_2", seed=2)]
    results.append(run_game(g, agents, "connect_four"))

    # ------------------------------------------------------------------
    print("\n=== Extended Tic-Tac-Toe ===")
    g = ExtendedTicTacToeGame()
    agents = [RandomAgent("player_X", "player_X", seed=3),
              RandomAgent("player_O", "player_O", seed=4)]
    results.append(run_game(g, agents, "extended_tic_tac_toe"))

    # ------------------------------------------------------------------
    print("\n=== Mancala ===")
    g = MancalaGame()
    agents = [RandomAgent("player_1", "player_1", seed=5),
              RandomAgent("player_2", "player_2", seed=6)]
    results.append(run_game(g, agents, "mancala"))

    # ------------------------------------------------------------------
    print("\n=== Othello ===")
    g = OthelloGame()
    agents = [RandomAgent("player_black", "player_black", seed=7),
              RandomAgent("player_white", "player_white", seed=8)]
    results.append(run_game(g, agents, "othello"))

    # ------------------------------------------------------------------
    print("\n=== Gomoku ===")
    g = GomokuGame()
    agents = [RandomAgent("player_black", "player_black", seed=9),
              RandomAgent("player_white", "player_white", seed=10)]
    # Gomoku can run very long with random agents; cap at 300 steps
    results.append(run_game(g, agents, "gomoku", max_steps=300))

    # ------------------------------------------------------------------
    print("\n=== Micro Chess ===")
    g = MicroChessGame()
    agents = [RandomAgent("player_white", "player_white", seed=11),
              RandomAgent("player_black", "player_black", seed=12)]
    results.append(run_game(g, agents, "micro_chess", max_steps=200))

    # ------------------------------------------------------------------
    print("\n=== Capture the Flag ===")
    g = CaptureTheFlagGame(max_steps=200)
    agents = [
        RandomAgent("A1", "team_A", seed=13),
        RandomAgent("A2", "team_A", seed=14),
        RandomAgent("B1", "team_B", seed=15),
        RandomAgent("B2", "team_B", seed=16),
    ]
    results.append(run_game(g, agents, "capture_the_flag", max_steps=200))

    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    passed = sum(results)
    print(f"  {passed}/{len(results)} games passed smoke tests.")
    print(f"{'='*55}\n")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
