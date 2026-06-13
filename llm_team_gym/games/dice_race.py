"""
Dice Race — The Pig Game (risk management, 1v1 or multiplayer).

On your turn, you roll a six-sided die repeatedly. You accumulate a
"turn bank." At any point you may "bank" to secure those points. But if
you roll a 1, you lose the entire turn bank and your turn ends.

First player to reach GOAL total points wins.

Action format : "roll"  or  "bank"
Turn order    : p0 → p1 → … (turn-based; only active player has moves)
Teams         : each player is their own team
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

GOAL = 100

BG_COLOR   = (10, 12, 20)
P_COLORS   = [(0, 212, 190), (255, 80, 120), (148, 92, 255), (255, 178, 50)]
FONT_COLOR = (238, 242, 255)
PIG_COLOR  = (148, 92, 255)
PANEL_BG   = (18, 21, 34)
PANEL_BDR  = (42, 48, 72)
TEXT_SEC   = (130, 140, 175)
WARNING_CLR= (255, 215, 60)


class DiceRaceGame(BaseGame):
    """Pig dice game — first to GOAL points wins."""

    def __init__(self, n_players: int = 2, goal: int = GOAL, seed: Optional[int] = None):
        self.n_players   = n_players
        self.goal        = goal
        self.player_ids  = tuple(f"p{i}" for i in range(n_players))
        self._seed       = seed
        self._rng        = random.Random(seed)

        self.scores:       Dict[AgentID, int] = {}
        self.turn_bank:    int = 0
        self._turn_idx:    int = 0
        self._last_roll:   Optional[int] = None
        self._done:        bool = False
        self._winner:      Optional[AgentID] = None
        self._step:        int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng      = random.Random(self._seed)
        self.scores    = {p: 0 for p in self.player_ids}
        self.turn_bank = 0
        self._turn_idx = 0
        self._last_roll = None
        self._done     = False
        self._winner   = None
        self._step     = 0
        return self._obs()

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        active  = self.player_ids[self._turn_idx % self.n_players]

        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip().lower()
        if action not in self.get_legal_moves(active):
            infos[active] = {"error": f"Illegal action '{action}'"}
            return self._obs(), rewards, self._dones(), infos

        if action == "bank":
            self.scores[active] += self.turn_bank
            infos[active] = {"banked": self.turn_bank, "total": self.scores[active]}
            self.turn_bank = 0
            self._last_roll = None
            if self.scores[active] >= self.goal:
                self._done   = True
                self._winner = active
                rewards[active] = 1.0
                for p in self.player_ids:
                    if p != active:
                        rewards[p] = -1.0
            else:
                self._turn_idx = (self._turn_idx + 1) % self.n_players

        elif action == "roll":
            roll = self._rng.randint(1, 6)
            self._last_roll = roll
            if roll == 1:
                self.turn_bank = 0
                infos[active] = {"rolled": 1, "pig": True, "lost_bank": True}
                self._turn_idx = (self._turn_idx + 1) % self.n_players
            else:
                self.turn_bank += roll
                infos[active] = {"rolled": roll, "turn_bank": self.turn_bank}

        self._step += 1
        return self._obs(), rewards, self._dones(), infos

    def get_text_state(self, agent_id: AgentID) -> str:
        active = self.player_ids[self._turn_idx % self.n_players]
        state = {
            "agent_id": agent_id,
            "is_your_turn": active == agent_id,
            "active_player": active,
            "step": self._step,
            "your_score": self.scores.get(agent_id, 0),
            "all_scores": dict(self.scores),
            "goal": self.goal,
            "points_needed": max(0, self.goal - self.scores.get(agent_id, 0)),
            "current_turn_bank": self.turn_bank if active == agent_id else 0,
            "last_roll": self._last_roll,
            "legal_moves": self.get_legal_moves(agent_id),
            "risk_note": (
                "Rolling 1 loses ALL turn bank points. "
                "Banking secures current turn bank to total score. "
                f"First to {self.goal} wins."
            ),
            "winner": self._winner,
            "game_over": self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        active = self.player_ids[self._turn_idx % self.n_players]
        if active != agent_id:
            return []
        moves = ["roll"]
        if self.turn_bank > 0:
            moves.append("bank")
        return moves

    def get_game_rules(self) -> str:
        return f"""
=== THE PIG DICE GAME — Game Rules ===

OBJECTIVE
---------
First player to accumulate {self.goal} or more total points wins.

TURN STRUCTURE
--------------
On your turn, you repeatedly ROLL a standard six-sided die:

  "roll" → Roll the die.
    • Rolled 2–6 : add the result to your TURN BANK (temporary).
    • Rolled 1   : lose all turn bank points ("pig!"), turn ends.

  "bank" → Secure your current turn bank to your TOTAL score. Turn ends.

RISK vs. REWARD
---------------
Rolling accumulates points quickly but risks losing the turn bank.
Banking is safe but may be sub-optimal if the bank is small.

Optimal strategy: consider your current total vs. the opponent's,
and the probability of rolling a 1 (1/6 ≈ 16.7%).

WIN CONDITION
-------------
First player to reach {self.goal} total points wins.

ACTION FORMAT
-------------
  "roll"  — roll the die (always legal)
  "bank"  — bank turn points (only legal if turn bank > 0)
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
            self._screen = pygame.display.set_mode((600, 320))
            pygame.display.set_caption("LLM-TeamGym · Dice Race (Pig)")
            self._font  = pygame.font.SysFont("monospace", 26, bold=True)
            self._small = pygame.font.SysFont("monospace", 16)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        active = self.player_ids[self._turn_idx % self.n_players]
        for i, pid in enumerate(self.player_ids):
            col = P_COLORS[i % len(P_COLORS)]
            x   = 30 + i * 200
            pygame.draw.rect(scr, col, (x, 40, 160, 180), border_radius=12)
            for j, txt in enumerate([
                pid, f"Score: {self.scores[pid]}/{self.goal}",
                f"Bank: {self.turn_bank if pid==active else '-'}",
                "← ACTIVE" if pid == active else "",
            ]):
                lbl = self._small.render(txt, True, (255,255,255))
                scr.blit(lbl, (x + 10, 52 + j * 28))
        info = (f"Last roll: {self._last_roll or '-'}   "
                f"Step: {self._step}   "
                f"Winner: {self._winner or 'ongoing'}")
        scr.blit(self._small.render(info, True, FONT_COLOR), (30, 250))
        pygame.display.flip(); self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    def _obs(self) -> Dict[AgentID, Observation]:
        active = self.player_ids[self._turn_idx % self.n_players]
        snap   = {"scores": dict(self.scores), "active": active,
                  "turn_bank": self.turn_bank, "step": self._step,
                  "done": self._done, "winner": self._winner}
        return {p: dict(snap) for p in self.player_ids}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
