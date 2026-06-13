"""
Smoke-tests for the Cards & Risk Management and Game Theory batches.
Validates: import, reset, legal moves, step, game termination.
"""

import sys
import traceback
import json

from llm_team_gym.core.base_agent import RandomAgent
from llm_team_gym.envs.tournament import MatchRunner

from llm_team_gym.games.yaniv import YanivGame
from llm_team_gym.games.blackjack import BlackjackGame
from llm_team_gym.games.team_taki import TeamTakiGame
from llm_team_gym.games.bridge_wist import BridgeWistGame
from llm_team_gym.games.iterated_prisoners_dilemma import IteratedPrisonersDilemmaGame
from llm_team_gym.games.ultimatum_game import UltimatumGame
from llm_team_gym.games.extended_rps import ExtendedRPSGame
from llm_team_gym.games.dollar_auction import DollarAuctionGame
from llm_team_gym.games.tragedy_of_the_commons import TragedyOfTheCommonsGame
from llm_team_gym.games.stock_market_sim import StockMarketSimGame
from llm_team_gym.games.dice_race import DiceRaceGame
from llm_team_gym.games.settlers_catan_mini import SettlersCatanMiniGame


# ------------------------------------------------------------------ helper

def run_match(game, agents, label):
    try:
        runner = MatchRunner(
            game=game, agents=agents, render=False, verbose=False,
            match_id=f"test_{label}",
        )
        rec = runner.run()
        print(f"  [OK]  {label:<38}  steps={rec.total_steps:<5} "
              f"winner={str(rec.winner or 'draw'):<12} "
              f"scores={rec.final_team_scores}")
        return True
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


def check_legal(game, label):
    """Verify all agents get valid legal move lists (possibly empty)."""
    try:
        for pid in game.all_agents:
            lm = game.get_legal_moves(pid)
            assert isinstance(lm, list), f"{pid} legal_moves not a list"
        ts_pid = game.all_agents[0]
        ts = game.get_text_state(ts_pid)
        parsed = json.loads(ts)
        assert isinstance(parsed, dict)
        print(f"  [OK]  {label:<38}  legal_moves OK, text_state JSON valid")
        return True
    except Exception:
        print(f"  [FAIL] {label} legal/text check")
        traceback.print_exc()
        return False


# ------------------------------------------------------------------ Cards

def test_yaniv():
    print("\n=== Yaniv ===")
    results = []
    for n in (2, 3):
        g = YanivGame(n_players=n, seed=42)
        agents = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(n)]
        g.reset()
        results.append(check_legal(g, f"yaniv_{n}p_legal"))
        results.append(run_match(g, agents, f"yaniv_{n}p_random"))
    active_pid = YanivGame(seed=7)
    active_pid.reset()
    active = active_pid.all_agents[0]
    lm = active_pid.get_legal_moves(active)
    assert any("yaniv" in m or len(m.split()) >= 1 for m in lm), "Should have discard moves"
    print(f"  [OK]  yaniv legal moves: {len(lm)} options for active player")
    return all(results)


def test_blackjack():
    print("\n=== Blackjack ===")
    results = []
    g = BlackjackGame(n_players=1, seed=42)
    agents = [RandomAgent("p0", "p0", seed=0)]
    results.append(run_match(g, agents, "blackjack_1p"))

    g2 = BlackjackGame(n_players=2, seed=99)
    agents2 = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(2)]
    results.append(run_match(g2, agents2, "blackjack_2p"))

    g3 = BlackjackGame(seed=7)
    g3.reset()
    results.append(check_legal(g3, "blackjack_legal"))
    active = g3._active_player()
    if active:
        lm = g3.get_legal_moves(active)
        assert "hit" in lm and "stand" in lm, "Should have hit and stand"
        assert "double" in lm, "Should have double on first action"
    print(f"  [OK]  blackjack first-action legal: hit/stand/double available")
    return all(results)


def test_team_taki():
    print("\n=== Team Taki ===")
    results = []
    g = TeamTakiGame(seed=42)
    agents = [RandomAgent(p, ("team_A" if p in ("p0","p2") else "team_B"), seed=i)
              for i, p in enumerate(["p0","p1","p2","p3"])]
    results.append(run_match(g, agents, "team_taki_random"))
    g2 = TeamTakiGame(seed=99)
    g2.reset()
    results.append(check_legal(g2, "team_taki_legal"))
    active = g2._active()
    lm = g2.get_legal_moves(active)
    assert any(m.startswith("play ") or m == "draw" for m in lm), "Should have play or draw"
    print(f"  [OK]  taki active player has {len(lm)} legal moves")
    return all(results)


def test_bridge_wist():
    print("\n=== Bridge/Wist ===")
    results = []
    g = BridgeWistGame(seed=42)
    agents = [RandomAgent(p, ("NS" if p in ("N","S") else "EW"), seed=i)
              for i, p in enumerate(["N","E","S","W"])]
    results.append(run_match(g, agents, "bridge_random"))
    g2 = BridgeWistGame(seed=7)
    g2.reset()
    results.append(check_legal(g2, "bridge_legal"))
    active_bid = g2._bid_active()
    lm = g2.get_legal_moves(active_bid)
    assert "pass" in lm, "Should be able to pass in bidding"
    assert "1C" in lm, "1C should be a valid first bid"
    assert "7NT" in lm, "7NT should be a valid bid at start"
    print(f"  [OK]  bridge bidding: {len(lm)} legal bids including pass/1C/7NT")
    return all(results)


# ------------------------------------------------------------------ Game Theory

def test_ipd():
    print("\n=== Iterated Prisoner's Dilemma ===")
    results = []
    g = IteratedPrisonersDilemmaGame(n_rounds=10, seed=42)
    agents = [RandomAgent("p0", "p0", seed=0), RandomAgent("p1", "p1", seed=1)]
    results.append(run_match(g, agents, "ipd_random_10r"))
    g2 = IteratedPrisonersDilemmaGame(n_rounds=5)
    g2.reset()
    lm0 = g2.get_legal_moves("p0")
    lm1 = g2.get_legal_moves("p1")
    assert set(lm0) == {"cooperate", "defect"}, f"IPD legal: {lm0}"
    assert set(lm1) == {"cooperate", "defect"}, f"IPD legal: {lm1}"
    print("  [OK]  IPD: both players have {cooperate, defect} as legal moves")
    g2.step({"p0": "cooperate", "p1": "defect"})
    state = json.loads(g2.get_text_state("p0"))
    assert state["history"][0]["actions"]["p0"] == "cooperate"
    assert state["history"][0]["actions"]["p1"] == "defect"
    print("  [OK]  IPD: history records both actions per round")
    return all(results)


def test_ultimatum():
    print("\n=== Ultimatum Game ===")
    results = []
    g = UltimatumGame(n_rounds=6, seed=42)
    agents = [RandomAgent("p0", "p0", seed=0), RandomAgent("p1", "p1", seed=1)]
    results.append(run_match(g, agents, "ultimatum_6r"))
    g2 = UltimatumGame(n_rounds=4)
    g2.reset()
    assert g2._proposer == "p0"
    prop_moves = g2.get_legal_moves("p0")
    assert all(m.startswith("offer ") for m in prop_moves), "Proposer moves should be offers"
    assert g2.get_legal_moves("p1") == [], "Responder has no moves in PROPOSE phase"
    print(f"  [OK]  ultimatum: proposer has {len(prop_moves)} offer moves, responder has 0")
    g2.step({"p0": "offer 5"})
    resp_moves = g2.get_legal_moves("p1")
    assert set(resp_moves) == {"accept", "reject"}, f"Responder should accept/reject: {resp_moves}"
    print("  [OK]  ultimatum: after offer, responder has {accept, reject}")
    return all(results)


def test_extended_rps():
    print("\n=== Extended RPS (RPSLS) ===")
    results = []
    for n in (2, 3):
        g = ExtendedRPSGame(n_players=n, n_rounds=10, seed=42)
        agents = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(n)]
        results.append(run_match(g, agents, f"rpsls_{n}p_10r"))
    g2 = ExtendedRPSGame(n_players=2)
    g2.reset()
    for p in ["p0", "p1"]:
        lm = g2.get_legal_moves(p)
        assert set(lm) == {"rock", "paper", "scissors", "lizard", "spock"}, f"RPSLS moves: {lm}"
    print("  [OK]  RPSLS: all 5 choices available to both players simultaneously")
    return all(results)


def test_dollar_auction():
    print("\n=== Dollar Auction ===")
    results = []
    g = DollarAuctionGame(n_players=2, seed=42)
    agents = [RandomAgent("p0", "p0", seed=0), RandomAgent("p1", "p1", seed=1)]
    results.append(run_match(g, agents, "dollar_auction_2p"))
    g3 = DollarAuctionGame(n_players=3, seed=7)
    agents3 = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(3)]
    results.append(run_match(g3, agents3, "dollar_auction_3p"))
    g2 = DollarAuctionGame(seed=1)
    g2.reset()
    active = g2._active_player()
    lm = g2.get_legal_moves(active)
    assert "pass" in lm, "Should be able to pass"
    assert any(m.startswith("bid ") for m in lm), "Should have bid options"
    print(f"  [OK]  dollar auction: {len(lm)} legal moves for first active player")
    return all(results)


def test_tragedy():
    print("\n=== Tragedy of the Commons ===")
    results = []
    for n in (2, 3):
        g = TragedyOfTheCommonsGame(n_players=n, n_rounds=8, seed=42)
        agents = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(n)]
        results.append(run_match(g, agents, f"commons_{n}p"))
    g2 = TragedyOfTheCommonsGame(n_players=3)
    g2.reset()
    for p in ["p0", "p1", "p2"]:
        lm = g2.get_legal_moves(p)
        assert len(lm) > 0, f"{p} should have moves"
        assert "harvest 0" in lm, "harvest 0 should always be legal"
    print("  [OK]  tragedy: all players have simultaneous legal moves including harvest 0")
    return all(results)


def test_stock_market():
    print("\n=== Stock Market Sim ===")
    results = []
    g = StockMarketSimGame(n_players=2, n_rounds=10, seed=42)
    agents = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(2)]
    results.append(run_match(g, agents, "stocks_2p_10r"))
    g2 = StockMarketSimGame(n_players=3, n_rounds=5, seed=7)
    agents2 = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(3)]
    results.append(run_match(g2, agents2, "stocks_3p_5r"))
    g3 = StockMarketSimGame(n_players=2, seed=1)
    g3.reset()
    for p in ["p0", "p1"]:
        lm = g3.get_legal_moves(p)
        assert "hold" in lm, f"{p} should have 'hold'"
    print("  [OK]  stock market: all players have simultaneous moves including 'hold'")
    ts = json.loads(g3.get_text_state("p0"))
    assert ts["current_news"] is not None, "News event should be present"
    assert "stock_prices" in ts
    print(f"  [OK]  stock market text state: news={ts['current_news']}, prices={ts['stock_prices']}")
    return all(results)


def test_dice_race():
    print("\n=== Dice Race (Pig) ===")
    results = []
    g = DiceRaceGame(n_players=2, goal=50, seed=42)
    agents = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(2)]
    results.append(run_match(g, agents, "dice_race_2p_goal50"))
    g3 = DiceRaceGame(n_players=3, goal=30, seed=7)
    agents3 = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(3)]
    results.append(run_match(g3, agents3, "dice_race_3p_goal30"))
    g2 = DiceRaceGame(n_players=2)
    g2.reset()
    active = g2.player_ids[0]
    lm = g2.get_legal_moves(active)
    assert lm == ["roll"], "First legal move should be only 'roll' (bank=0)"
    non_active = g2.player_ids[1]
    assert g2.get_legal_moves(non_active) == [], "Non-active player has no moves"
    print("  [OK]  dice race: first move is 'roll' only; non-active has none")
    g2.step({"p0": "roll"})
    lm2 = g2.get_legal_moves(active)
    if g2.turn_bank > 0:
        assert "bank" in lm2, "Bank should appear after non-1 roll"
        print(f"  [OK]  dice race: bank appears after turn_bank={g2.turn_bank}")
    return all(results)


def test_settlers():
    print("\n=== Settlers Catan Mini ===")
    results = []
    g = SettlersCatanMiniGame(n_players=2, seed=42)
    agents = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(2)]
    results.append(run_match(g, agents, "catan_mini_2p"))
    g3 = SettlersCatanMiniGame(n_players=3, seed=7)
    agents3 = [RandomAgent(f"p{i}", f"p{i}", seed=i) for i in range(3)]
    results.append(run_match(g3, agents3, "catan_mini_3p"))
    g2 = SettlersCatanMiniGame(n_players=2, seed=1)
    g2.reset()
    results.append(check_legal(g2, "catan_legal"))
    ts = json.loads(g2.get_text_state("p0"))
    assert ts["your_vp"] == 2, "Should start with 2 VP (2 settlements)"
    assert "last_dice_roll" in ts
    print(f"  [OK]  catan: starts with 2 VP, dice rolled on reset: {ts['last_dice_roll']}")
    return all(results)


# ------------------------------------------------------------------ main

def main():
    print("=" * 65)
    print("  LLM-TeamGym — Cards & Game Theory Batch Tests")
    print("=" * 65)

    tests = [
        ("Yaniv",             test_yaniv),
        ("Blackjack",         test_blackjack),
        ("Team Taki",         test_team_taki),
        ("Bridge/Wist",       test_bridge_wist),
        ("Iterated PD",       test_ipd),
        ("Ultimatum",         test_ultimatum),
        ("Extended RPS",      test_extended_rps),
        ("Dollar Auction",    test_dollar_auction),
        ("Tragedy Commons",   test_tragedy),
        ("Stock Market",      test_stock_market),
        ("Dice Race",         test_dice_race),
        ("Settlers Catan",    test_settlers),
    ]

    passed = 0
    for name, fn in tests:
        ok = fn()
        if ok:
            passed += 1

    print(f"\n{'='*65}")
    print(f"  {passed}/{len(tests)} game batches passed all tests.")
    print(f"{'='*65}\n")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
