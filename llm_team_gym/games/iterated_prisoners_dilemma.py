"""
Iterated Prisoner's Dilemma — classic game theory benchmark.

Two players simultaneously choose to "cooperate" or "defect" each round.
Payoffs per round (p0, p1):
  Both cooperate  → (3, 3)
  Both defect     → (1, 1)
  p0 C, p1 D      → (0, 5)
  p0 D, p1 C      → (5, 0)

Full round history is visible to both players (so strategies like Tit-for-Tat
can be evaluated). After N rounds, highest total payoff wins.

Action format : "cooperate"  or  "defect"
Timing        : SIMULTANEOUS — both players move each round
Teams         : each player is their own team
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("cooperate", "cooperate"): (3.0, 3.0),
    ("cooperate", "defect"):    (0.0, 5.0),
    ("defect",    "cooperate"): (5.0, 0.0),
    ("defect",    "defect"):    (1.0, 1.0),
}

PLAYERS = ("p0", "p1")

BG_COLOR  = (12, 12, 22)
GRID_CLR  = (50, 60, 80)
C_COLOR   = (60, 190, 80)
D_COLOR   = (200, 60, 60)
TEXT_CLR  = (220, 220, 230)


class IteratedPrisonersDilemmaGame(BaseGame):
    """
    Classic IPD — two players, N simultaneous rounds.

    Both players choose simultaneously each round; history is fully
    visible so agents can implement reactive strategies.
    """

    def __init__(self, n_rounds: int = 20, seed: Optional[int] = None):
        self.n_rounds  = n_rounds
        self._round    = 0
        self.scores:   Dict[AgentID, float] = {}
        self.history:  List[Dict] = []
        self._actions_buffer: Dict[AgentID, str] = {}
        self._done:    bool = False
        self._winner:  Optional[str] = None

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {p: [p] for p in PLAYERS}

    def reset(self) -> Dict[AgentID, Observation]:
        self._round   = 0
        self.scores   = {p: 0.0 for p in PLAYERS}
        self.history  = []
        self._actions_buffer = {}
        self._done    = False
        self._winner  = None
        return self._obs()

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in PLAYERS}
        infos   = {p: {}  for p in PLAYERS}

        if self._done:
            return self._obs(), rewards, self._dones(), infos

        for pid, act in actions_dict.items():
            act = str(act).strip().lower()
            if act in ("cooperate", "defect"):
                self._actions_buffer[pid] = act

        if not all(p in self._actions_buffer for p in PLAYERS):
            return self._obs(), rewards, self._dones(), infos

        a0 = self._actions_buffer.pop("p0")
        a1 = self._actions_buffer.pop("p1")
        r0, r1 = PAYOFF[(a0, a1)]
        self.scores["p0"] += r0
        self.scores["p1"] += r1
        self.history.append({
            "round": self._round,
            "actions": {"p0": a0, "p1": a1},
            "payoffs": {"p0": r0, "p1": r1},
            "cumulative": dict(self.scores),
        })
        rewards["p0"] = r0; rewards["p1"] = r1
        infos["p0"] = {"p1_played": a1}; infos["p1"] = {"p0_played": a0}
        self._round += 1
        if self._round >= self.n_rounds:
            self._done = True
            s0, s1 = self.scores["p0"], self.scores["p1"]
            if s0 > s1:   self._winner = "p0"
            elif s1 > s0: self._winner = "p1"
            else:         self._winner = None

        return self._obs(), rewards, self._dones(), infos

    def get_text_state(self, agent_id: AgentID) -> str:
        opp = "p1" if agent_id == "p0" else "p0"
        state = {
            "agent_id": agent_id,
            "round": self._round,
            "n_rounds": self.n_rounds,
            "rounds_remaining": self.n_rounds - self._round,
            "your_cumulative_score": self.scores.get(agent_id, 0.0),
            "opponent_cumulative_score": self.scores.get(opp, 0.0),
            "history": self.history,
            "payoff_matrix": {
                "you_C_opp_C": f"you={PAYOFF[('cooperate','cooperate')][0]}, opp={PAYOFF[('cooperate','cooperate')][1]}",
                "you_C_opp_D": f"you={PAYOFF[('cooperate','defect')][0]}, opp={PAYOFF[('cooperate','defect')][1]}",
                "you_D_opp_C": f"you={PAYOFF[('defect','cooperate')][0]}, opp={PAYOFF[('defect','cooperate')][1]}",
                "you_D_opp_D": f"you={PAYOFF[('defect','defect')][0]}, opp={PAYOFF[('defect','defect')][1]}",
            },
            "legal_moves": self.get_legal_moves(agent_id),
            "game_over": self._done,
            "winner": self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or agent_id not in PLAYERS:
            return []
        return ["cooperate", "defect"]

    def get_game_rules(self) -> str:
        return f"""
=== ITERATED PRISONER'S DILEMMA — Game Rules ===

PLAYERS   : p0 and p1
ROUNDS    : {self.n_rounds} simultaneous rounds

Each round, both players secretly and simultaneously choose:
  "cooperate" — work together
  "defect"    — betray the other

PAYOFF MATRIX (per round):
  Both cooperate  → each receives 3 points
  Both defect     → each receives 1 point
  One defects     → defector gets 5, cooperator gets 0

SIMULTANEOUS CHOICE
-------------------
Both players submit their action each round; moves are revealed
together at the end of the round. Full history is visible.

STRATEGY SPACE
--------------
Classic strategies: Tit-for-Tat, Always Defect, Always Cooperate,
Grim Trigger, Pavlov (Win-Stay/Lose-Shift), etc.

Optimal long-run cooperation requires trusting the other agent
while protecting against exploitation.

WIN CONDITION
-------------
After {self.n_rounds} rounds, the player with the highest CUMULATIVE score wins.
Draw if scores are equal.

ACTION FORMAT
-------------
  "cooperate" — cooperate this round
  "defect"    — defect this round
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
            self._screen = pygame.display.set_mode((720, 440))
            pygame.display.set_caption("LLM-TeamGym · Iterated Prisoner's Dilemma")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        # Header
        h = self._font.render(
            f"Round {self._round}/{self.n_rounds}  |  "
            f"p0={self.scores['p0']:.0f}  p1={self.scores['p1']:.0f}  |  "
            f"Winner: {self._winner or 'TBD'}",
            True, TEXT_CLR)
        scr.blit(h, (20, 15))
        # History table
        headers = ["Rnd", "p0", "p1", "p0 pts", "p1 pts"]
        col_x   = [20, 90, 160, 250, 340]
        for ci, hdr in enumerate(headers):
            lbl = self._small.render(hdr, True, (180, 180, 200))
            scr.blit(lbl, (col_x[ci], 55))
        max_rows = 18
        start_row = max(0, len(self.history) - max_rows)
        for ri, entry in enumerate(self.history[start_row:]):
            y = 80 + ri * 18
            a0, a1 = entry["actions"]["p0"], entry["actions"]["p1"]
            p0, p1 = entry["payoffs"]["p0"],  entry["payoffs"]["p1"]
            for ci, txt in enumerate([
                str(entry["round"]), a0[:1].upper(), a1[:1].upper(),
                f"{p0:.0f}", f"{p1:.0f}"
            ]):
                color = C_COLOR if (ci in (1,2) and txt == "C") else (
                        D_COLOR if (ci in (1,2) and txt == "D") else TEXT_CLR)
                scr.blit(self._small.render(txt, True, color), (col_x[ci], y))
        pygame.display.flip(); self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {"round": self._round, "scores": dict(self.scores),
                "history": list(self.history), "done": self._done,
                "winner": self._winner}
        return {p: dict(snap) for p in PLAYERS}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in PLAYERS}
        d["__all__"] = self._done
        return d
