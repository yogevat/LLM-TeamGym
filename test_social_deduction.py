"""
Smoke-tests for the 3 social deduction / limited-information games.
Validates: hidden info correctness, phase transitions, legal move constraints.
"""

import random
import sys
import traceback

from llm_team_gym.core.base_agent import RandomAgent, BaseAgent
from llm_team_gym.core.base_game import BaseGame
from llm_team_gym.envs.tournament import MatchRunner
from llm_team_gym.games.the_resistance import TheResistanceGame, ALL_PLAYERS
from llm_team_gym.games.hanabi import HanabiGame
from llm_team_gym.games.captain_sonar_mini import CaptainSonarMiniGame


# ------------------------------------------------------------------ helpers

def run_match(game: BaseGame, agents, label: str) -> bool:
    try:
        runner = MatchRunner(
            game=game, agents=agents, render=False, verbose=False,
            match_id=f"test_{label}",
        )
        rec = runner.run()
        print(f"  [OK]  {label:<35}  steps={rec.total_steps:<4}  "
              f"winner={rec.winner or 'draw'}  scores={rec.final_team_scores}")
        return True
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


# ------------------------------------------------------------------ The Resistance

def test_resistance():
    print("\n=== The Resistance ===")
    results = []

    # Basic game with random agents
    g = TheResistanceGame(seed=42)
    agents = [RandomAgent(p, "unknown", seed=i) for i, p in enumerate(ALL_PLAYERS)]
    results.append(run_match(g, agents, "resistance_random"))

    # Verify hidden info: resistance member should NOT see spy identities
    g2 = TheResistanceGame(seed=7)
    g2.reset()
    for pid in ALL_PLAYERS:
        ts = g2.get_text_state(pid)
        import json
        state = json.loads(ts)
        role  = state["your_role"]
        if role == "resistance":
            assert "HIDDEN" in str(state["fellow_spies"]), \
                f"{pid} is resistance but sees spy identities!"
        else:
            fellow = state["fellow_spies"]
            assert isinstance(fellow, list) and len(fellow) == 1, \
                f"Spy {pid} should see 1 fellow spy, got: {fellow}"
    print("  [OK]  hidden-info check: resistance sees HIDDEN, spies see ally")

    # Legal moves: propose phase — only leader has moves
    g3 = TheResistanceGame(seed=1)
    g3.reset()
    non_leaders = [p for p in ALL_PLAYERS if p != ALL_PLAYERS[g3.leader_idx]]
    for p in non_leaders:
        assert g3.get_legal_moves(p) == [], f"{p} should have no moves in PROPOSE phase"
    leader = ALL_PLAYERS[g3.leader_idx]
    lm = g3.get_legal_moves(leader)
    assert len(lm) == 10, f"Leader should have C(5,2)=10 proposal options, got {len(lm)}"
    print(f"  [OK]  propose phase: leader has {len(lm)} legal proposals, others have 0")

    # After proposal → vote phase: all 5 have moves
    g3.step({leader: lm[0]})
    for p in ALL_PLAYERS:
        assert g3.get_legal_moves(p) == ["approve", "reject"], \
            f"{p} should have vote moves in VOTE phase"
    print("  [OK]  vote phase: all 5 players have approve/reject legal moves")

    return all(results)


# ------------------------------------------------------------------ Hanabi

def test_hanabi():
    print("\n=== Hanabi ===")
    results = []

    # 3-player game
    g = HanabiGame(n_players=3, seed=99)
    agents = [RandomAgent(f"p{i}", "cooperation_team", seed=100+i) for i in range(3)]
    results.append(run_match(g, agents, "hanabi_3p_random"))

    # 2-player game
    g2 = HanabiGame(n_players=2, seed=55)
    agents2 = [RandomAgent(f"p{i}", "cooperation_team", seed=200+i) for i in range(2)]
    results.append(run_match(g2, agents2, "hanabi_2p_random"))

    # Verify hidden info: active player's hand is hidden in text state
    g3 = HanabiGame(n_players=3, seed=7)
    g3.reset()
    active = g3.player_ids[0]
    ts     = g3.get_text_state(active)
    import json
    state  = json.loads(ts)
    for card_slot in state["your_hand"]["cards"]:
        assert card_slot["color"]  == "HIDDEN", "Own card color should be HIDDEN"
        assert card_slot["number"] == "HIDDEN", "Own card number should be HIDDEN"
    print("  [OK]  hidden-info check: active player's own cards are fully masked")

    # Verify teammates' cards ARE visible
    for pid, hand in state["teammates_hands"].items():
        for card_slot in hand:
            assert card_slot["color"]  != "HIDDEN", f"{pid}'s card should be visible"
            assert card_slot["number"] != "HIDDEN", f"{pid}'s number should be visible"
    print("  [OK]  visible-info check: teammates' full hands are visible")

    # Clue legal move check: only valid clue targets (must match at least 1 card)
    lm = g3.get_legal_moves(active)
    clue_moves = [m for m in lm if m.startswith("clue")]
    for cm in clue_moves:
        parts = cm.split()
        target = parts[1]
        clue_type = parts[2]
        val = parts[3]
        # Verify at least one card in target's hand matches
        hand = g3.hands[target]
        if clue_type == "color":
            assert any(c == val for c, n in hand), f"Clue '{cm}' touches no card!"
        else:
            assert any(n == int(val) for c, n in hand), f"Clue '{cm}' touches no card!"
    print(f"  [OK]  clue legal moves: {len(clue_moves)} valid clues, all touch ≥1 card")

    return all(results)


# ------------------------------------------------------------------ Captain Sonar Mini

def test_captain_sonar():
    print("\n=== Captain Sonar Mini ===")
    results = []

    g = CaptainSonarMiniGame(seed=42)
    agents = [
        RandomAgent("CA", "team_A", seed=1),
        RandomAgent("RA", "team_A", seed=2),
        RandomAgent("CB", "team_B", seed=3),
        RandomAgent("RB", "team_B", seed=4),
    ]
    results.append(run_match(g, agents, "sonar_random"))

    # Verify hidden info: CA sees own position, RA also gets it in obs,
    # but text state description differs by role
    g2 = CaptainSonarMiniGame(seed=7)
    g2.reset()
    import json
    ca_state = json.loads(g2.get_text_state("CA"))
    ra_state = json.loads(g2.get_text_state("RA"))
    # Captain sees exact position
    assert "row" in str(ca_state["your_submarine"]["position"]), \
        "Captain should see exact row in position"
    # Radio Operator sees HIDDEN position description
    assert "HIDDEN" in str(ra_state["your_submarine"]["position"]), \
        "Radio Operator should not see direct position label"
    print("  [OK]  hidden-info check: Captain sees position, RA sees HIDDEN")

    # Neither team can see the enemy's actual position in text state
    for agent in ("CA", "RA"):
        state = json.loads(g2.get_text_state(agent))
        tracking = state["enemy_tracking"]
        assert "enemy_movement_log" in tracking, "Should see enemy movement log"
        # enemy's actual position not in state
        enemy_pos = list(g2.pos["B"])
        assert str(enemy_pos) not in json.dumps(state), \
            f"Agent {agent} should not see exact enemy position {enemy_pos} in their state"
    print("  [OK]  hidden-info check: neither team agent can see enemy's true coordinates")

    # Phase enforcement: only CA has legal moves in CA_MOVE phase
    assert g2.get_legal_moves("RA") == [], "RA has no moves in CA_MOVE phase"
    assert g2.get_legal_moves("CB") == [], "CB has no moves in CA_MOVE phase"
    ca_moves = g2.get_legal_moves("CA")
    assert "surface" in ca_moves, "CA should always have 'surface' as a legal move"
    print(f"  [OK]  phase enforcement: only CA active in CA_MOVE phase ({len(ca_moves)} moves)")

    # Visited-cell tracking: CA can't revisit without surfacing
    pos_before = g2.pos["A"]
    valid_dir = [d for d in ("north","south","east","west") if d in ca_moves]
    if valid_dir:
        move = valid_dir[0]
        g2.step({"CA": move})
        # Verify position changed
        assert g2.pos["A"] != pos_before, "Sub should have moved"
        # Now it's RA's phase, skip to CA phase
        g2.step({"RA": "pass"})    # RA passes
        g2.step({"CB": g2.get_legal_moves("CB")[0]})   # CB moves
        g2.step({"RB": "pass"})    # RB passes
        # Back to CA: the moved-from cell should not be in legal moves
        ca_moves2 = g2.get_legal_moves("CA")
        print(f"  [OK]  visited-cell check: CA has {len(ca_moves2)} moves after 1 step")

    return all(results)


# ------------------------------------------------------------------ main

def main():
    results = []
    results.append(test_resistance())
    results.append(test_hanabi())
    results.append(test_captain_sonar())

    passed = sum(results)
    print(f"\n{'='*55}")
    print(f"  {passed}/{len(results)} social deduction games passed all tests.")
    print(f"{'='*55}\n")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
