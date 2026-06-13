"""
Extended Rock-Paper-Scissors — Rock-Paper-Scissors-Lizard-Spock (2–4 players).

Win matrix (row beats column):
  Rock     beats Scissors, Lizard
  Paper    beats Rock, Spock
  Scissors beats Paper, Lizard
  Lizard   beats Paper, Spock
  Spock    beats Rock, Scissors

2-player: standard competitive; 3-4 players: round-robin scoring.
Simultaneous reveal each round. N rounds total.

Action format : "rock" | "paper" | "scissors" | "lizard" | "spock"
Timing        : SIMULTANEOUS — all players move each round
Teams         : each player is their own team
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

CHOICES = ("rock", "paper", "scissors", "lizard", "spock")

BEATS: Dict[str, List[str]] = {
    "rock":     ["scissors", "lizard"],
    "paper":    ["rock",     "spock"],
    "scissors": ["paper",    "lizard"],
    "lizard":   ["paper",    "spock"],
    "spock":    ["rock",     "scissors"],
}

BEATS_REASON: Dict[Tuple[str, str], str] = {
    ("rock",     "scissors"): "Rock crushes Scissors",
    ("rock",     "lizard"):   "Rock crushes Lizard",
    ("paper",    "rock"):     "Paper covers Rock",
    ("paper",    "spock"):    "Paper disproves Spock",
    ("scissors", "paper"):    "Scissors cuts Paper",
    ("scissors", "lizard"):   "Scissors decapitates Lizard",
    ("lizard",   "paper"):    "Lizard eats Paper",
    ("lizard",   "spock"):    "Lizard poisons Spock",
    ("spock",    "rock"):     "Spock vaporizes Rock",
    ("spock",    "scissors"): "Spock smashes Scissors",
}

BG_COLOR  = (10, 12, 20)
C_COLORS  = [(0, 212, 190), (255, 80, 120), (148, 92, 255), (255, 178, 50)]
TEXT_CLR  = (238, 242, 255)
WIN_CLR   = (255, 215, 60)
PANEL_BG  = (18, 21, 34)
PANEL_BDR = (42, 48, 72)
TEXT_SEC  = (130, 140, 175)


class ExtendedRPSGame(BaseGame):
    """Rock-Paper-Scissors-Lizard-Spock — N simultaneous rounds."""

    def __init__(self, n_players: int = 2, n_rounds: int = 20, seed=None):
        assert 2 <= n_players <= 4, "2–4 players supported"
        self.n_players  = n_players
        self.n_rounds   = n_rounds
        self.player_ids = tuple(f"p{i}" for i in range(n_players))

        self._round      = 0
        self.scores:     Dict[AgentID, float] = {}
        self.history:    List[Dict] = []
        self._buf:       Dict[AgentID, str] = {}
        self._done:      bool = False
        self._winner:    Optional[str] = None

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self._round  = 0
        self.scores  = {p: 0.0 for p in self.player_ids}
        self.history = []
        self._buf    = {}
        self._done   = False
        self._winner = None
        return self._obs()

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        for pid, act in actions_dict.items():
            act = str(act).strip().lower()
            if act in CHOICES and pid in self.player_ids:
                self._buf[pid] = act

        if not all(p in self._buf for p in self.player_ids):
            return self._obs(), rewards, self._dones(), infos

        round_actions = {p: self._buf.pop(p) for p in self.player_ids}
        round_rewards, outcomes = self._resolve(round_actions)
        for p in self.player_ids:
            rewards[p] = round_rewards[p]
            self.scores[p] += round_rewards[p]

        self.history.append({
            "round": self._round,
            "actions": round_actions,
            "rewards": round_rewards,
            "outcomes": outcomes,
            "cumulative": dict(self.scores),
        })
        self._round += 1
        if self._round >= self.n_rounds:
            self._done = True
            best_score = max(self.scores.values())
            winners = [p for p in self.player_ids if self.scores[p] == best_score]
            self._winner = winners[0] if len(winners) == 1 else "draw"

        return self._obs(), rewards, self._dones(), infos

    def _resolve(self, actions: Dict[AgentID, str]) -> Tuple[Dict, List[str]]:
        rewards  = {p: 0.0 for p in self.player_ids}
        outcomes: List[str] = []
        pids = list(self.player_ids)
        for i, a in enumerate(pids):
            for j, b in enumerate(pids):
                if i >= j:
                    continue
                ca, cb = actions[a], actions[b]
                if ca == cb:
                    outcomes.append(f"{a}({ca}) ties {b}({cb})")
                elif cb in BEATS[ca]:
                    rewards[a]  += 1.0; rewards[b]  -= 1.0
                    reason = BEATS_REASON.get((ca, cb), "")
                    outcomes.append(f"{a}({ca}) beats {b}({cb}): {reason}")
                else:
                    rewards[b]  += 1.0; rewards[a]  -= 1.0
                    reason = BEATS_REASON.get((cb, ca), "")
                    outcomes.append(f"{b}({cb}) beats {a}({ca}): {reason}")
        return rewards, outcomes

    def get_text_state(self, agent_id: AgentID) -> str:
        state = {
            "agent_id":    agent_id,
            "round":       self._round,
            "n_rounds":    self.n_rounds,
            "rounds_left": self.n_rounds - self._round,
            "your_score":  self.scores.get(agent_id, 0.0),
            "all_scores":  dict(self.scores),
            "choices":     list(CHOICES),
            "beats_chart": {c: BEATS[c] for c in CHOICES},
            "history":     self.history[-10:],
            "legal_moves": self.get_legal_moves(agent_id),
            "game_over":   self._done,
            "winner":      self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or agent_id not in self.player_ids:
            return []
        return list(CHOICES)

    def get_game_rules(self) -> str:
        return f"""
=== EXTENDED RPS — Rock-Paper-Scissors-Lizard-Spock ===

PLAYERS : {self.n_players} simultaneous players
ROUNDS  : {self.n_rounds}

WIN MATRIX
----------
Rock     beats Scissors (crushes), Lizard (crushes)
Paper    beats Rock (covers),     Spock (disproves)
Scissors beats Paper (cuts),      Lizard (decapitates)
Lizard   beats Paper (eats),      Spock (poisons)
Spock    beats Rock (vaporizes),  Scissors (smashes)

SCORING (per round, pairwise):
  Win  → +1 point per opponent beaten
  Tie  → no points exchanged
  Loss → -1 point per winner

SIMULTANEOUS
------------
All players choose secretly and simultaneously. Moves are revealed
together. Full history of all rounds is visible.

WIN CONDITION
-------------
After {self.n_rounds} rounds, the player with the highest total score wins.

ACTION FORMAT
-------------
  "rock" | "paper" | "scissors" | "lizard" | "spock"
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
            pygame.display.set_caption("LLM-TeamGym · Extended RPS")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        header = f"Round {self._round}/{self.n_rounds}"
        scr.blit(self._font.render(header, True, TEXT_CLR), (20, 10))
        for i, pid in enumerate(self.player_ids):
            col = C_COLORS[i % len(C_COLORS)]
            x   = 20 + i * 190
            pygame.draw.rect(scr, col, (x, 45, 170, 55), border_radius=8)
            scr.blit(self._small.render(f"{pid}: {self.scores[pid]:.0f}", True, (255,255,255)), (x+8, 52))
            scr.blit(self._small.render(f"Win: {self._winner or 'TBD'}", True, (255,255,255)), (x+8, 73))
        max_rows = 18
        start = max(0, len(self.history) - max_rows)
        for ri, entry in enumerate(self.history[start:]):
            y   = 120 + ri * 18
            row = f"R{entry['round']:>2}: " + "  ".join(
                f"{p}={entry['actions'][p][:2]}" for p in self.player_ids)
            scr.blit(self._small.render(row, True, TEXT_CLR), (20, y))
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
                "history": self.history[-5:], "done": self._done, "winner": self._winner}
        return {p: dict(snap) for p in self.player_ids}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
