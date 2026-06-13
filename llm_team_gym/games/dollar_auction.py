"""
Dollar Auction — the Shubik auction paradox.

A prize worth PRIZE units is auctioned. Players bid in increments.
The twist: BOTH the highest bidder AND the second-highest bidder pay
their bids, but only the highest bidder receives the prize.

This creates a "sunk cost trap": players already committed to bidding
feel compelled to keep bidding to avoid losing their committed funds.

Actions:
  "bid N"  — place a bid of N (must be current_min_bid ≤ N ≤ budget)
  "pass"   — withdraw from the auction (stays out for the rest)

Turn-based: players take turns in order, excluding those who have passed.
Game ends when all but one player has passed (or only one remains).

Teams  : each player is their own team
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

PRIZE          = 100
START_BUDGET   = 150
BID_INCREMENT  = 5

BG_COLOR  = (10, 12, 20)
TEXT_CLR  = (238, 242, 255)
P_COLORS  = [(0, 212, 190), (255, 80, 120), (148, 92, 255), (255, 178, 50)]
PRIZE_CLR = (255, 215, 60)
PANEL_BG  = (18, 21, 34)
PANEL_BDR = (42, 48, 72)
TEXT_SEC  = (130, 140, 175)
DANGER_CLR= (255, 75, 80)


class DollarAuctionGame(BaseGame):
    """
    Dollar (Shubik) Auction. n_players bid for a prize. Both highest
    and second-highest bidder pay; only the highest gets the prize.
    """

    def __init__(self, n_players: int = 2, prize: int = PRIZE,
                 budget: int = START_BUDGET, increment: int = BID_INCREMENT,
                 seed=None):
        assert 2 <= n_players <= 4
        self.n_players  = n_players
        self.prize      = prize
        self.start_budget = budget
        self.increment  = increment
        self.player_ids = tuple(f"p{i}" for i in range(n_players))

        self.budgets:   Dict[AgentID, int] = {}
        self.current_bids: Dict[AgentID, int] = {}
        self._passed:   Set[AgentID] = set()
        self._turn_idx: int = 0
        self._min_next_bid: int = 0
        self._done:     bool = False
        self._winner:   Optional[str] = None
        self.history:   List[Dict] = []
        self._step:     int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self.budgets      = {p: self.start_budget for p in self.player_ids}
        self.current_bids = {p: 0 for p in self.player_ids}
        self._passed      = set()
        self._turn_idx    = 0
        self._min_next_bid = self.increment
        self._done        = False
        self._winner      = None
        self.history      = []
        self._step        = 0
        return self._obs()

    def _active_player(self) -> Optional[AgentID]:
        alive = [p for p in self.player_ids if p not in self._passed]
        if len(alive) <= 1:
            return None
        idx = self._turn_idx % len(self.player_ids)
        for _ in range(self.n_players):
            p = self.player_ids[idx % self.n_players]
            if p not in self._passed:
                return p
            idx += 1
        return None

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        active = self._active_player()
        if active is None or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip().lower()
        legal  = self.get_legal_moves(active)
        if action not in legal:
            infos[active] = {"error": f"Illegal action '{action}'"}
            return self._obs(), rewards, self._dones(), infos

        if action == "pass":
            self._passed.add(active)
            self.history.append({
                "step":   self._step,
                "player": active,
                "action": "pass",
                "bids":   dict(self.current_bids),
                "passed": list(self._passed),
            })
        else:
            n = int(action.split()[1])
            self.current_bids[active] = n
            self._min_next_bid = n + self.increment
            self.history.append({
                "step":         self._step,
                "player":       active,
                "action":       action,
                "bid_placed":   n,
                "min_next_bid": self._min_next_bid,
                "bids":         dict(self.current_bids),
            })

        self._step += 1
        alive = [p for p in self.player_ids if p not in self._passed]

        if len(alive) <= 1:
            self._end_auction(rewards)

        else:
            self._turn_idx += 1
            next_p = self._active_player()
            if next_p and self.budgets[next_p] < self._min_next_bid:
                self._passed.add(next_p)
                alive2 = [p for p in self.player_ids if p not in self._passed]
                if len(alive2) <= 1:
                    self._end_auction(rewards)
                else:
                    self._turn_idx += 1

        return self._obs(), rewards, self._dones(), infos

    def _end_auction(self, rewards: Dict[AgentID, float]) -> None:
        self._done = True
        active_bids = {p: self.current_bids[p] for p in self.player_ids}
        sorted_bidders = sorted(self.player_ids, key=lambda p: active_bids[p], reverse=True)
        winner = sorted_bidders[0]
        second = sorted_bidders[1] if len(sorted_bidders) > 1 else None

        winner_bid = active_bids[winner]
        second_bid = active_bids[second] if second else 0

        winner_net = self.prize - winner_bid
        self._winner = winner

        for p in self.player_ids:
            if p == winner:
                rewards[p] = float(winner_net)
                self.budgets[p] = self.start_budget - winner_bid + self.prize
            elif p == second:
                rewards[p] = float(-second_bid)
                self.budgets[p] = self.start_budget - second_bid
            else:
                rewards[p] = 0.0

    def get_text_state(self, agent_id: AgentID) -> str:
        active = self._active_player()
        sorted_bids = sorted(self.player_ids, key=lambda p: self.current_bids[p], reverse=True)
        state = {
            "agent_id":           agent_id,
            "is_your_turn":       active == agent_id,
            "active_player":      active,
            "step":               self._step,
            "your_budget":        self.budgets.get(agent_id, 0),
            "your_current_bid":   self.current_bids.get(agent_id, 0),
            "all_current_bids":   dict(self.current_bids),
            "highest_bid":        self.current_bids[sorted_bids[0]] if sorted_bids else 0,
            "second_highest_bid": self.current_bids[sorted_bids[1]] if len(sorted_bids) > 1 else 0,
            "min_next_bid":       self._min_next_bid,
            "prize_value":        self.prize,
            "players_passed":     list(self._passed),
            "players_still_in":   [p for p in self.player_ids if p not in self._passed],
            "history":            self.history[-10:],
            "legal_moves":        self.get_legal_moves(agent_id),
            "trap_warning": (
                "WARNING: The Dollar Auction trap. Both top-2 bidders pay their bids "
                "even if they don't win. Bidding beyond prize value guarantees net loss. "
                "Once committed, passing still costs you your current bid (if you're 2nd highest)."
            ),
            "game_over":   self._done,
            "winner":      self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or agent_id in self._passed:
            return []
        active = self._active_player()
        if active != agent_id:
            return []
        moves = ["pass"]
        max_bid = self.budgets[agent_id]
        bid = self._min_next_bid
        while bid <= max_bid:
            moves.append(f"bid {bid}")
            bid += self.increment
        return moves

    def get_game_rules(self) -> str:
        return f"""
=== DOLLAR AUCTION (Shubik Paradox) — Game Rules ===

PRIZE   : {self.prize} coins
BUDGET  : each player starts with {self.start_budget} coins
INCREMENT: minimum bid increment = {self.increment} coins
PLAYERS : {self.n_players}

THE TWIST
---------
Both the HIGHEST bidder AND the SECOND-HIGHEST bidder pay their bids.
Only the highest bidder receives the prize.

AUCTION FLOW
------------
Players take turns (in order, skipping those who passed).
On your turn you must either:
  "bid N"  — place bid of N (N ≥ current minimum bid, N ≤ your budget)
  "pass"   — withdraw from the auction (irreversible)

Minimum bid starts at {self.increment} and increases by {self.increment} after each new bid.

GAME ENDS when only one active bidder remains (all others passed or
are excluded by insufficient budget).

PAYOFFS
-------
  Winner  : gains PRIZE − own_bid     (can be negative!)
  2nd place: loses own_bid
  Others  : no change

THE TRAP
--------
Once you have bid, passing means losing your bid (if you end up 2nd highest).
This pressure causes rational actors to over-bid past the prize value.
Optimal strategy often involves credible commitment NOT to escalate.

ACTION FORMAT
-------------
  "bid N"   — bid N coins (N ∈ {{{self.increment}, {self.increment*2}, ...}} up to your budget)
  "pass"    — exit the auction (irrevocable)
""".strip()

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        try:
            import pygame
        except ImportError:
            return
        if not self._pygame_init:
            pygame.init()
            self._screen = pygame.display.set_mode((700, 420))
            pygame.display.set_caption("LLM-TeamGym · Dollar Auction")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        scr.blit(self._font.render(
            f"Prize: {self.prize}  Min next bid: {self._min_next_bid}  Winner: {self._winner or 'TBD'}",
            True, PRIZE_CLR), (10, 10))
        active = self._active_player()
        for i, pid in enumerate(self.player_ids):
            col = P_COLORS[i % len(P_COLORS)]
            x   = 10 + i * 170
            passed_txt = " [PASSED]" if pid in self._passed else ""
            active_txt = " ← ACTIVE" if pid == active else ""
            pygame.draw.rect(scr, col, (x, 50, 155, 90), border_radius=8)
            for j, txt in enumerate([
                f"{pid}{passed_txt}",
                f"Bid: {self.current_bids[pid]}",
                f"Budget: {self.budgets[pid]}",
                active_txt,
            ]):
                scr.blit(self._small.render(txt, True, (255,255,255)), (x+5, 58 + j*20))
        for ri, entry in enumerate(self.history[-16:]):
            y = 165 + ri * 16
            scr.blit(self._small.render(str(entry), True, TEXT_CLR), (10, y))
        pygame.display.flip(); self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {"bids": dict(self.current_bids), "passed": list(self._passed),
                "min_next_bid": self._min_next_bid, "step": self._step,
                "done": self._done, "winner": self._winner,
                "budgets": dict(self.budgets)}
        return {p: dict(snap) for p in self.player_ids}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
