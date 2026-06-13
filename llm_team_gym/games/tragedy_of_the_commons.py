"""
Tragedy of the Commons — collective resource management.

A shared resource (e.g., a fishery) regenerates each round but can be
depleted by over-harvesting. Players choose how much to harvest each round.

  • Resource regenerates by REGEN_RATE (%) each round (capped at MAX_RESOURCE).
  • If resource drops below COLLAPSE_THRESHOLD, the fishery collapses and
    yields 0 forever.
  • Higher individual harvest → more personal score, but risks collapse.
  • Optimal group outcome: restrained harvesting to sustain the resource.

Actions   : "harvest N"  (N = integer 0 .. MAX_HARVEST, limited by remaining resource)
Timing    : SIMULTANEOUS — all players harvest at the same time
Teams     : each player is their own team (individual scores)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

START_RESOURCE      = 100.0
MAX_RESOURCE        = 100.0
REGEN_RATE          = 0.25        # 25 % of current stock (logistic-like)
COLLAPSE_THRESHOLD  = 10.0
MAX_HARVEST         = 20

BG_COLOR   = (10, 20, 15)
GOOD_CLR   = (60, 200, 80)
WARN_CLR   = (230, 180, 30)
DEAD_CLR   = (200, 40, 40)
TEXT_CLR   = (220, 230, 220)
P_COLORS   = [(70, 130, 220), (220, 70, 70), (60, 190, 80), (230, 200, 40)]


class TragedyOfTheCommonsGame(BaseGame):
    """
    Shared fishery: cooperate to sustain it, or over-harvest for short-term gain.
    """

    def __init__(self, n_players: int = 3, n_rounds: int = 15,
                 regen_rate: float = REGEN_RATE,
                 collapse_threshold: float = COLLAPSE_THRESHOLD,
                 seed=None):
        assert 2 <= n_players <= 4
        self.n_players   = n_players
        self.n_rounds    = n_rounds
        self.regen_rate  = regen_rate
        self.collapse_threshold = collapse_threshold
        self.player_ids  = tuple(f"p{i}" for i in range(n_players))

        self._round:   int = 0
        self.resource: float = START_RESOURCE
        self.scores:   Dict[AgentID, float] = {}
        self._collapsed: bool = False
        self.history:  List[Dict] = []
        self._buf:     Dict[AgentID, int] = {}
        self._done:    bool = False
        self._winner:  Optional[str] = None

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[str, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self._round    = 0
        self.resource  = START_RESOURCE
        self.scores    = {p: 0.0 for p in self.player_ids}
        self._collapsed = False
        self.history   = []
        self._buf      = {}
        self._done     = False
        self._winner   = None
        return self._obs()

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        for pid, act in actions_dict.items():
            act = str(act).strip().lower()
            if act.startswith("harvest "):
                try:
                    n = int(act.split()[1])
                    if 0 <= n <= MAX_HARVEST and pid in self.player_ids:
                        self._buf[pid] = n
                except (IndexError, ValueError):
                    pass

        if not all(p in self._buf for p in self.player_ids):
            return self._obs(), rewards, self._dones(), infos

        harvests = {p: self._buf.pop(p) for p in self.player_ids}

        if self._collapsed:
            actual = {p: 0 for p in self.player_ids}
        else:
            total_request = sum(harvests.values())
            if total_request > self.resource:
                scale = self.resource / total_request if total_request > 0 else 0
                actual = {p: harvests[p] * scale for p in self.player_ids}
                self.resource = 0.0
            else:
                actual = {p: float(harvests[p]) for p in self.player_ids}
                self.resource -= total_request

            if self.resource <= self.collapse_threshold:
                self._collapsed = True

            if not self._collapsed and self.resource > 0:
                regen = self.resource * self.regen_rate
                self.resource = min(MAX_RESOURCE, self.resource + regen)

        for p in self.player_ids:
            self.scores[p] += actual[p]
            rewards[p] = actual[p]

        self.history.append({
            "round":    self._round,
            "resource_before": round(self.resource + sum(actual.values()), 2),
            "harvests": harvests,
            "actual":   {p: round(actual[p], 2) for p in self.player_ids},
            "resource_after": round(self.resource, 2),
            "collapsed": self._collapsed,
            "cumulative": {p: round(self.scores[p], 2) for p in self.player_ids},
        })
        self._round += 1
        if self._round >= self.n_rounds:
            self._done = True
            best = max(self.scores.values())
            winners = [p for p in self.player_ids if self.scores[p] == best]
            self._winner = winners[0] if len(winners) == 1 else "draw"

        return self._obs(), rewards, self._dones(), infos

    def get_text_state(self, agent_id: AgentID) -> str:
        safe_harvest = max(0, int(
            (self.resource * self.regen_rate) / self.n_players
        ))
        state = {
            "agent_id":          agent_id,
            "round":             self._round,
            "n_rounds":          self.n_rounds,
            "rounds_left":       self.n_rounds - self._round,
            "shared_resource":   round(self.resource, 2),
            "max_resource":      MAX_RESOURCE,
            "collapse_threshold": self.collapse_threshold,
            "fishery_collapsed": self._collapsed,
            "regen_rate_pct":    f"{self.regen_rate*100:.0f}%",
            "your_score":        round(self.scores.get(agent_id, 0.0), 2),
            "all_scores":        {p: round(v, 2) for p, v in self.scores.items()},
            "sustainable_harvest_hint": (
                f"At current stock ({self.resource:.1f}), the resource regenerates "
                f"{self.resource * self.regen_rate:.1f} units/round. "
                f"If split equally among {self.n_players} players, "
                f"each could sustainably take ~{safe_harvest}."
            ),
            "history":           self.history[-5:],
            "legal_moves":       self.get_legal_moves(agent_id),
            "game_over":         self._done,
            "winner":            self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or agent_id not in self.player_ids:
            return []
        if self._collapsed:
            return ["harvest 0"]
        cap = min(MAX_HARVEST, int(self.resource))
        return [f"harvest {n}" for n in range(cap + 1)]

    def get_game_rules(self) -> str:
        return f"""
=== TRAGEDY OF THE COMMONS — Game Rules ===

PLAYERS   : {self.n_players} simultaneous players
ROUNDS    : {self.n_rounds}
START RES : {START_RESOURCE:.0f} units (max {MAX_RESOURCE:.0f})
REGEN     : {self.regen_rate*100:.0f}% of current stock per round
COLLAPSE  : if resource drops ≤ {self.collapse_threshold}, fishery collapses permanently

EACH ROUND (SIMULTANEOUS)
--------------------------
Every player submits  "harvest N"  (N = 0..{MAX_HARVEST}, capped by available resource).

Total harvested is deducted from the shared resource FIRST.
Then the remaining resource regenerates (×{self.regen_rate:.2f}).

If total requests exceed available stock, harvests are pro-rated proportionally.

COLLAPSE
--------
Once resource ≤ {self.collapse_threshold}, it collapses. Future harvests yield 0.
Group coordination is required to avoid this outcome.

WIN CONDITION
-------------
After {self.n_rounds} rounds, the player with the highest cumulative harvest total wins.

STRATEGIC TENSION
-----------------
• Each player is individually incentivized to harvest as much as possible.
• Collectively, over-harvesting destroys the resource for everyone.
• Optimal group strategy: each harvests ≤ (resource × regen_rate / n_players).
• Tragedy occurs when no one restrains themselves — even though all would
  be better off with coordination.

ACTION FORMAT
-------------
  "harvest N"  — harvest N units (N ≥ 0, N ≤ min({MAX_HARVEST}, remaining_resource))
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
            self._screen = pygame.display.set_mode((720, 420))
            pygame.display.set_caption("LLM-TeamGym · Tragedy of the Commons")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        res_pct = self.resource / MAX_RESOURCE
        res_clr = GOOD_CLR if res_pct > 0.3 else (WARN_CLR if res_pct > 0.1 else DEAD_CLR)
        collapsed = " [COLLAPSED]" if self._collapsed else ""
        scr.blit(self._font.render(
            f"Round {self._round}/{self.n_rounds}  Resource: {self.resource:.1f}/{MAX_RESOURCE}{collapsed}",
            True, res_clr), (10, 10))
        pygame.draw.rect(scr, (30, 40, 30), (10, 40, 700, 22), border_radius=4)
        pygame.draw.rect(scr, res_clr, (10, 40, int(700 * res_pct), 22), border_radius=4)
        for i, pid in enumerate(self.player_ids):
            col = P_COLORS[i % len(P_COLORS)]
            x   = 10 + i * 175
            pygame.draw.rect(scr, col, (x, 75, 160, 60), border_radius=8)
            scr.blit(self._small.render(f"{pid}", True, (255,255,255)), (x+6, 82))
            scr.blit(self._small.render(f"Total: {self.scores[pid]:.1f}", True, (255,255,255)), (x+6, 103))
        for ri, entry in enumerate(self.history[-16:]):
            y   = 150 + ri * 17
            row = (f"R{entry['round']:>2}: res={entry['resource_after']:.1f}  " +
                   "  ".join(f"{p}→{entry['harvests'][p]}" for p in self.player_ids))
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
        snap = {"round": self._round, "resource": self.resource,
                "scores": dict(self.scores), "collapsed": self._collapsed,
                "history": self.history[-5:], "done": self._done, "winner": self._winner}
        return {p: dict(snap) for p in self.player_ids}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
