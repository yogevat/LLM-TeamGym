"""
Stock Market Simulation — portfolio management under uncertainty.

Players manage a portfolio of cash and shares across 3 stocks (A, B, C).
Each round:
  1. A news event (public) is revealed, hinting at price changes.
  2. Players simultaneously choose one action.
  3. Stock prices update (news effect + collective buying pressure).

Portfolio value = cash + shares × current_price.
Win: highest portfolio value after N rounds.

Actions (one per round per player):
  "buy <stock>"    — buy 1 share of A/B/C at current price
  "sell <stock>"   — sell 1 share of A/B/C at current price
  "hold"           — do nothing this round
  "short <stock>"  — short-sell 1 share (profit if price falls)
  "cover <stock>"  — cover a short position

Teams : each player is their own team
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

STOCKS    = ("A", "B", "C")
START_PRICE   = 100.0
START_CASH    = 1000.0
MAX_SHARES    = 10
MAX_ROUNDS    = 20
PRICE_FLOOR   = 1.0
PRICE_CEIL    = 500.0

NEWS_EVENTS = [
    ("boom",           {"A":  0.12, "B":  0.08, "C":  0.05}),
    ("recession",      {"A": -0.10, "B": -0.08, "C": -0.06}),
    ("tech_surge",     {"A":  0.20, "B":  0.02, "C": -0.03}),
    ("energy_crisis",  {"A": -0.05, "B":  0.15, "C": -0.08}),
    ("commodity_boom", {"A": -0.03, "B": -0.02, "C":  0.22}),
    ("market_crash",   {"A": -0.18, "B": -0.14, "C": -0.10}),
    ("IPO_frenzy",     {"A":  0.08, "B":  0.18, "C":  0.06}),
    ("scandal_A",      {"A": -0.25, "B":  0.05, "C":  0.03}),
    ("scandal_B",      {"A":  0.03, "B": -0.22, "C":  0.04}),
    ("innovation_C",   {"A": -0.02, "B": -0.02, "C":  0.30}),
    ("rate_hike",      {"A": -0.07, "B": -0.06, "C": -0.05}),
    ("rate_cut",       {"A":  0.09, "B":  0.07, "C":  0.06}),
    ("stable_market",  {"A":  0.01, "B":  0.02, "C":  0.00}),
]

BG_COLOR  = (10, 12, 25)
TEXT_CLR  = (220, 220, 230)
UP_CLR    = (60, 200, 80)
DOWN_CLR  = (200, 60, 60)
P_COLORS  = [(70, 130, 220), (220, 70, 70), (60, 190, 80), (230, 200, 40)]
STCK_CLR  = {"A": (255, 215, 0), "B": (100, 180, 255), "C": (255, 130, 50)}


class StockMarketSimGame(BaseGame):
    """Multiplayer stock market portfolio simulation."""

    def __init__(self, n_players: int = 2, n_rounds: int = MAX_ROUNDS,
                 seed: Optional[int] = None):
        assert 2 <= n_players <= 4
        self.n_players   = n_players
        self.n_rounds    = n_rounds
        self.player_ids  = tuple(f"p{i}" for i in range(n_players))
        self._seed       = seed
        self._rng        = random.Random(seed)

        self.prices:     Dict[str, float] = {}
        self.cash:       Dict[AgentID, float] = {}
        self.shares:     Dict[AgentID, Dict[str, int]] = {}
        self.shorts:     Dict[AgentID, Dict[str, int]] = {}
        self._round:     int = 0
        self.history:    List[Dict] = []
        self._buf:       Dict[AgentID, str] = {}
        self._news:      Optional[Tuple[str, Dict[str, float]]] = None
        self._done:      bool = False
        self._winner:    Optional[str] = None

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[str, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng     = random.Random(self._seed)
        self.prices   = {s: START_PRICE for s in STOCKS}
        self.cash     = {p: START_CASH  for p in self.player_ids}
        self.shares   = {p: {s: 0 for s in STOCKS} for p in self.player_ids}
        self.shorts   = {p: {s: 0 for s in STOCKS} for p in self.player_ids}
        self._round   = 0
        self.history  = []
        self._buf     = {}
        self._news    = self._draw_news()
        self._done    = False
        self._winner  = None
        return self._obs()

    def _draw_news(self) -> Tuple[str, Dict[str, float]]:
        evt, effects = self._rng.choice(NEWS_EVENTS)
        noise = {s: self._rng.gauss(0, 0.03) for s in STOCKS}
        combined = {s: effects[s] + noise[s] for s in STOCKS}
        return evt, combined

    def _portfolio_value(self, pid: AgentID) -> float:
        val = self.cash[pid]
        for s in STOCKS:
            val += self.shares[pid][s] * self.prices[s]
            val -= self.shorts[pid][s] * self.prices[s]
        return val

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        before_vals = {p: self._portfolio_value(p) for p in self.player_ids}

        for pid, act in actions_dict.items():
            act = str(act).strip().lower()
            legal = self.get_legal_moves(pid)
            if act in legal:
                self._buf[pid] = act

        if not all(p in self._buf for p in self.player_ids):
            return self._obs(), rewards, self._dones(), infos

        round_actions = {p: self._buf.pop(p) for p in self.player_ids}
        buy_pressure  = {s: 0 for s in STOCKS}
        sell_pressure = {s: 0 for s in STOCKS}

        for pid, act in round_actions.items():
            parts = act.split()
            verb  = parts[0]
            stock = parts[1] if len(parts) > 1 else None
            price = self.prices[stock] if stock else 0
            if verb == "buy" and stock:
                self.cash[pid]    -= price
                self.shares[pid][stock] += 1
                buy_pressure[stock] += 1
            elif verb == "sell" and stock:
                self.cash[pid]    += price
                self.shares[pid][stock] -= 1
                sell_pressure[stock] += 1
            elif verb == "short" and stock:
                self.cash[pid]    += price
                self.shorts[pid][stock] += 1
                sell_pressure[stock] += 1
            elif verb == "cover" and stock:
                self.cash[pid]    -= price
                self.shorts[pid][stock] -= 1
                buy_pressure[stock] += 1

        news_name, effects = self._news
        old_prices = dict(self.prices)
        for s in STOCKS:
            market_effect = (buy_pressure[s] - sell_pressure[s]) * 0.02
            delta = effects[s] + market_effect
            new_p = self.prices[s] * (1 + delta)
            self.prices[s] = max(PRICE_FLOOR, min(PRICE_CEIL, new_p))

        after_vals = {p: self._portfolio_value(p) for p in self.player_ids}
        for p in self.player_ids:
            rewards[p] = after_vals[p] - before_vals[p]

        self.history.append({
            "round":    self._round,
            "news":     news_name,
            "effects":  {s: f"{effects[s]:+.2%}" for s in STOCKS},
            "prices":   {s: round(self.prices[s], 2) for s in STOCKS},
            "actions":  round_actions,
            "portfolios": {p: round(after_vals[p], 2) for p in self.player_ids},
        })
        self._round += 1
        self._news   = self._draw_news()

        if self._round >= self.n_rounds:
            self._done = True
            vals = {p: self._portfolio_value(p) for p in self.player_ids}
            best = max(vals.values())
            winners = [p for p in self.player_ids if vals[p] == best]
            self._winner = winners[0] if len(winners) == 1 else "draw"

        return self._obs(), rewards, self._dones(), infos

    def get_text_state(self, agent_id: AgentID) -> str:
        state = {
            "agent_id":         agent_id,
            "round":            self._round,
            "n_rounds":         self.n_rounds,
            "rounds_left":      self.n_rounds - self._round,
            "current_news":     self._news[0] if self._news else None,
            "news_hint":        {s: ("positive" if self._news[1][s] > 0.02 else
                                     "negative" if self._news[1][s] < -0.02 else "neutral")
                                 for s in STOCKS} if self._news else {},
            "stock_prices":     {s: round(self.prices[s], 2) for s in STOCKS},
            "your_cash":        round(self.cash.get(agent_id, 0), 2),
            "your_shares":      dict(self.shares.get(agent_id, {})),
            "your_short_positions": dict(self.shorts.get(agent_id, {})),
            "your_portfolio_value": round(self._portfolio_value(agent_id), 2),
            "all_portfolio_values": {p: round(self._portfolio_value(p), 2)
                                     for p in self.player_ids},
            "recent_history":   self.history[-5:],
            "legal_moves":      self.get_legal_moves(agent_id),
            "game_over":        self._done,
            "winner":           self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or agent_id not in self.player_ids:
            return []
        moves = ["hold"]
        p = agent_id
        for s in STOCKS:
            if self.cash[p] >= self.prices[s] and self.shares[p][s] < MAX_SHARES:
                moves.append(f"buy {s}")
            if self.shares[p][s] > 0:
                moves.append(f"sell {s}")
            if self.cash[p] >= self.prices[s] and self.shorts[p][s] < MAX_SHARES:
                moves.append(f"short {s}")
            if self.shorts[p][s] > 0:
                moves.append(f"cover {s}")
        return moves

    def get_game_rules(self) -> str:
        return f"""
=== STOCK MARKET SIMULATION — Game Rules ===

PLAYERS  : {self.n_players} simultaneous players
ROUNDS   : {self.n_rounds}
STOCKS   : A, B, C  (each starts at ${START_PRICE:.0f})
START CASH: ${START_CASH:.0f} per player

EACH ROUND
----------
1. A NEWS EVENT is revealed publicly (positive/neutral/negative for each stock).
   The hint direction helps but is noisy — prices also depend on player actions.

2. All players submit ONE action SIMULTANEOUSLY.

3. Stock prices update:
   • News effect (based on the event)
   • Market pressure: each buy adds +2% per buy, each sell −2%
   • Small random noise

ACTIONS
-------
  "buy <stock>"    — buy 1 share at current price (costs cash)
  "sell <stock>"   — sell 1 share at current price (gains cash)
  "short <stock>"  — borrow and immediately sell 1 share (gain cash now, owe future)
  "cover <stock>"  — buy back 1 shorted share (pay cash to close short)
  "hold"           — do nothing

CONSTRAINTS
-----------
• Max {MAX_SHARES} shares (long or short) per stock per player
• Cash must cover the purchase price (buy or cover)
• Prices capped: ${PRICE_FLOOR:.0f} – ${PRICE_CEIL:.0f}

PORTFOLIO VALUE
---------------
  Cash + (owned_shares × price) − (shorted_shares × current_price)

WIN CONDITION
-------------
After {self.n_rounds} rounds, highest total portfolio value wins.

ACTION FORMAT
-------------
  "buy A" | "sell B" | "short C" | "cover A" | "hold"
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
            self._screen = pygame.display.set_mode((800, 480))
            pygame.display.set_caption("LLM-TeamGym · Stock Market Sim")
            self._font  = pygame.font.SysFont("monospace", 18, bold=True)
            self._small = pygame.font.SysFont("monospace", 13)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        news = self._news[0] if self._news else "N/A"
        scr.blit(self._font.render(
            f"Round {self._round}/{self.n_rounds}  News: {news}  Winner: {self._winner or 'TBD'}",
            True, TEXT_CLR), (10, 8))
        for si, s in enumerate(STOCKS):
            px = self.prices[s]
            prev = self.history[-1]["prices"][s] if self.history else START_PRICE
            clr = UP_CLR if px >= prev else DOWN_CLR
            scr.blit(self._small.render(f"{s}: ${px:.1f}", True, STCK_CLR[s]), (10 + si * 150, 35))
        for i, pid in enumerate(self.player_ids):
            pval = self._portfolio_value(pid)
            x    = 10 + i * 195
            pygame.draw.rect(scr, P_COLORS[i % len(P_COLORS)], (x, 60, 180, 90), border_radius=8)
            for j, txt in enumerate([
                pid,
                f"Val: ${pval:.0f}",
                f"Cash: ${self.cash[pid]:.0f}",
                "Shares: " + " ".join(f"{s}:{self.shares[pid][s]}" for s in STOCKS),
            ]):
                scr.blit(self._small.render(txt, True, (255,255,255)), (x+5, 67+j*19))
        for ri, entry in enumerate(self.history[-17:]):
            y   = 165 + ri * 18
            row = (f"R{entry['round']:>2}: {entry['news']:<18} "
                   + "  ".join(f"{p}→{entry['actions'].get(p,'?')[:6]}" for p in self.player_ids))
            scr.blit(self._small.render(row, True, TEXT_CLR), (10, y))
        pygame.display.flip(); self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {"round": self._round, "prices": dict(self.prices),
                "portfolios": {p: self._portfolio_value(p) for p in self.player_ids},
                "news": self._news[0] if self._news else None,
                "done": self._done, "winner": self._winner}
        return {p: dict(snap) for p in self.player_ids}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
