"""
Ultimatum Game — fairness and negotiation benchmark.

In each round, the PROPOSER splits a pot of TOTAL coins between themselves
and the RESPONDER. The RESPONDER can accept or reject the offer:
  • Accept → both receive the offered split.
  • Reject → both receive nothing (0 coins).

Roles alternate every round. After K rounds, the player with the higher
total is the winner.

Actions:
  As Proposer : "offer N"  (N = 0 .. TOTAL coins for responder, rest to proposer)
  As Responder: "accept"   or   "reject"

Timing : turn-based within each round (Proposer → Responder)
Teams  : each player is their own team
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

TOTAL   = 10
PLAYERS = ("p0", "p1")

BG_COLOR = (12, 12, 25)
P_COLOR  = [(70, 130, 220), (220, 80, 60)]
TEXT_CLR = (220, 220, 230)


class UltimatumGame(BaseGame):
    """
    Iterated Ultimatum Game with alternating proposer/responder roles.

    Phase PROPOSE: active proposer submits "offer N".
    Phase RESPOND: responder submits "accept" or "reject".
    """

    def __init__(self, n_rounds: int = 10, total: int = TOTAL, seed=None):
        self.n_rounds   = n_rounds
        self.total      = total
        self._round     = 0
        self.scores:    Dict[AgentID, float] = {}
        self.history:   List[Dict] = []
        self._phase     = "PROPOSE"
        self._proposer  = "p0"
        self._responder = "p1"
        self._pending_offer: Optional[int] = None
        self._done:     bool = False
        self._winner:   Optional[str] = None

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {p: [p] for p in PLAYERS}

    def reset(self) -> Dict[AgentID, Observation]:
        self._round = 0
        self.scores = {p: 0.0 for p in PLAYERS}
        self.history = []
        self._phase  = "PROPOSE"
        self._proposer  = "p0"
        self._responder = "p1"
        self._pending_offer = None
        self._done   = False
        self._winner = None
        return self._obs()

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in PLAYERS}
        infos   = {p: {}  for p in PLAYERS}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        if self._phase == "PROPOSE":
            if self._proposer not in actions_dict:
                return self._obs(), rewards, self._dones(), infos
            action = str(actions_dict[self._proposer]).strip().lower()
            if not action.startswith("offer "):
                infos[self._proposer] = {"error": "Must use 'offer N'"}
                return self._obs(), rewards, self._dones(), infos
            try:
                n = int(action.split()[1])
            except (IndexError, ValueError):
                infos[self._proposer] = {"error": "Invalid offer format"}
                return self._obs(), rewards, self._dones(), infos
            if n not in range(self.total + 1):
                infos[self._proposer] = {"error": f"Offer must be 0–{self.total}"}
                return self._obs(), rewards, self._dones(), infos
            self._pending_offer = n
            self._phase = "RESPOND"

        elif self._phase == "RESPOND":
            if self._responder not in actions_dict:
                return self._obs(), rewards, self._dones(), infos
            action = str(actions_dict[self._responder]).strip().lower()
            if action not in ("accept", "reject"):
                infos[self._responder] = {"error": "Must 'accept' or 'reject'"}
                return self._obs(), rewards, self._dones(), infos
            offer = self._pending_offer
            if action == "accept":
                proposer_gain = float(self.total - offer)
                responder_gain = float(offer)
                self.scores[self._proposer]  += proposer_gain
                self.scores[self._responder] += responder_gain
                rewards[self._proposer]  = proposer_gain
                rewards[self._responder] = responder_gain
            else:
                proposer_gain = responder_gain = 0.0

            self.history.append({
                "round":      self._round,
                "proposer":   self._proposer,
                "responder":  self._responder,
                "offer_to_responder": offer,
                "proposer_keeps": self.total - offer,
                "response":   action,
                "proposer_gain":  proposer_gain,
                "responder_gain": responder_gain,
                "cumulative": dict(self.scores),
            })
            self._round += 1
            self._proposer, self._responder = self._responder, self._proposer
            self._pending_offer = None
            self._phase = "PROPOSE"
            if self._round >= self.n_rounds:
                self._done = True
                s0, s1 = self.scores["p0"], self.scores["p1"]
                self._winner = "p0" if s0 > s1 else ("p1" if s1 > s0 else "draw")

        return self._obs(), rewards, self._dones(), infos

    def get_text_state(self, agent_id: AgentID) -> str:
        opp = "p1" if agent_id == "p0" else "p0"
        state = {
            "agent_id":    agent_id,
            "round":       self._round,
            "n_rounds":    self.n_rounds,
            "rounds_left": self.n_rounds - self._round,
            "phase":       self._phase,
            "your_role":   ("proposer" if agent_id == self._proposer else "responder"),
            "you_are_proposer": agent_id == self._proposer,
            "pot_size":    self.total,
            "pending_offer_to_you": (
                self._pending_offer if (self._phase == "RESPOND" and agent_id == self._responder) else None
            ),
            "pending_offer_to_opponent": (
                self._pending_offer if (self._phase == "RESPOND" and agent_id == self._proposer) else None
            ),
            "your_cumulative_score":       self.scores.get(agent_id, 0.0),
            "opponent_cumulative_score":   self.scores.get(opp, 0.0),
            "history": self.history,
            "legal_moves": self.get_legal_moves(agent_id),
            "game_over":   self._done,
            "winner":      self._winner,
            "strategy_hint": (
                "As proposer: a purely rational responder accepts any offer ≥ 1. "
                "But LLM agents often reject 'unfair' offers even at personal cost. "
                "As responder: rejecting punishes unfair proposers but costs you too."
            ),
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or agent_id not in PLAYERS:
            return []
        if self._phase == "PROPOSE" and agent_id == self._proposer:
            return [f"offer {n}" for n in range(self.total + 1)]
        if self._phase == "RESPOND" and agent_id == self._responder:
            return ["accept", "reject"]
        return []

    def get_game_rules(self) -> str:
        return f"""
=== ULTIMATUM GAME — Game Rules ===

PLAYERS : p0 and p1
ROUNDS  : {self.n_rounds} (roles alternate each round)
POT SIZE: {self.total} coins

HOW EACH ROUND WORKS
--------------------
1. PROPOSER submits  "offer N"  (N = 0..{self.total}: coins offered to responder)
   Proposer keeps ({self.total} - N) coins.

2. RESPONDER submits  "accept"  or  "reject"
   • "accept" → proposer keeps ({self.total}-N), responder gets N.
   • "reject" → BOTH players receive 0 coins for this round.

ROLE ALTERNATION
----------------
Roles alternate every round:
  Round 0: p0 proposes, p1 responds.
  Round 1: p1 proposes, p0 responds.
  ...and so on.

WIN CONDITION
-------------
After {self.n_rounds} rounds, the player with the higher cumulative total wins.

STRATEGIC CONSIDERATIONS
------------------------
Game theory (rational self-interest) predicts: proposer offers 1 (minimum),
responder accepts any positive offer. However, human (and LLM) agents exhibit
strong fairness norms — offers below ~30% of the pot are often rejected.
This creates a strategic tension between self-interest and fairness perception.

ACTION FORMAT
-------------
  Proposer : "offer N"  where N is 0..{self.total}
  Responder: "accept"  or  "reject"
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
            self._screen = pygame.display.set_mode((760, 460))
            pygame.display.set_caption("LLM-TeamGym · Ultimatum Game")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        top = (f"Round {self._round}/{self.n_rounds}  Phase: {self._phase}  "
               f"p0={self.scores['p0']:.0f}  p1={self.scores['p1']:.0f}  "
               f"Winner: {self._winner or 'TBD'}")
        scr.blit(self._font.render(top, True, TEXT_CLR), (10, 10))
        if self._pending_offer is not None:
            off_txt = (f"Pending offer: {self._pending_offer} to {self._responder} "
                       f"(proposer keeps {self.total - self._pending_offer})")
            scr.blit(self._small.render(off_txt, True, (255, 215, 0)), (10, 45))
        for ri, entry in enumerate(self.history[-20:]):
            y   = 80 + ri * 18
            row = (f"R{entry['round']:>2}: {entry['proposer']} offered {entry['offer_to_responder']} "
                   f"→ {entry['response']}  (+{entry['proposer_gain']:.0f}/{entry['responder_gain']:.0f})")
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
        snap = {"round": self._round, "phase": self._phase, "scores": dict(self.scores),
                "proposer": self._proposer, "pending_offer": self._pending_offer,
                "history": list(self.history), "done": self._done, "winner": self._winner}
        return {p: dict(snap) for p in PLAYERS}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in PLAYERS}
        d["__all__"] = self._done
        return d
