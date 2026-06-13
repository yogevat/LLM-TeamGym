"""
The Resistance — 5-player social deduction game.

3 Resistance members vs 2 Spies. Spies know each other; Resistance members
do not know anyone's true role. Players debate and vote on mission teams over
5 rounds. Spies try to sabotage 3 missions; Resistance tries to pass 3.

Hidden information
------------------
  - Spies know who the other spy is (visible in their text state).
  - Resistance members only know their OWN role.
  - Mission vote tallies (sabotage count) are revealed, but WHO voted what is
    hidden — vote cards are shuffled before being revealed.

Phase cycle per round
---------------------
  PROPOSE  → leader selects a team of N players
  VOTE     → all 5 players simultaneously vote approve / reject
    if majority rejects → next leader tries (max 5 consecutive rejections)
    if 5 consecutive rejections → spies win immediately
  MISSION  → only the proposed team simultaneously votes pass / sabotage
    spies may choose to sabotage (invisible to others except count)
    1 sabotage (or 2 in round 4) → mission fails
    0 sabotages → mission passes

Win condition
-------------
  Resistance wins : 3 missions passed
  Spies win       : 3 missions failed  OR  5 consecutive rejected proposals

Agents    : "p0", "p1", "p2", "p3", "p4"
Teams     : {"resistance": [...], "spies": [...]}  (real roles, used for scoring)

Action format
-------------
  PROPOSE phase (leader only)  : "p0 p2"  or  "p1 p2 p4"
  VOTE phase    (all players)  : "approve"  or  "reject"
  MISSION phase (team members) : "pass"  or  "sabotage"
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

ALL_PLAYERS: Tuple[AgentID, ...] = ("p0", "p1", "p2", "p3", "p4")
# Mission team sizes for each round (5-player standard table)
MISSION_SIZES: Tuple[int, ...] = (2, 3, 2, 3, 3)
# Round index that requires 2 sabotages to fail (0-indexed)
DOUBLE_FAIL_ROUND = 3
MAX_FAILED_VOTES  = 5

PROPOSE = "PROPOSE"
VOTE    = "VOTE"
MISSION = "MISSION"

# Pygame
P_RAD   = 34
P_GAP   = 20
PAD     = 28
INFO_H  = 80
ROUND_H = 38
BG_COLOR      = (10, 12, 20)
RES_COLOR     = (0, 212, 190)
SPY_COLOR     = (255, 75, 80)
NEUTRAL_COLOR = (55, 60, 85)
APPROVE_COL   = (55, 225, 130)
REJECT_COL    = (255, 75, 80)
PASS_COL      = (55, 225, 130)
FAIL_COL      = (255, 75, 80)
FONT_COLOR    = (238, 242, 255)
GOLD_COLOR    = (255, 215, 60)
PANEL_BG      = (18, 21, 34)
PANEL_BDR     = (42, 48, 72)
TEXT_SEC      = (130, 140, 175)


class TheResistanceGame(BaseGame):
    """
    Full 5-player Resistance with vote tracking and mission sabotage.

    Roles are randomised each reset. The `teams` property exposes real
    alignment (used by MatchRunner for scoring) but `get_text_state`
    strictly guards spy identity from resistance members.
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng  = random.Random(seed)

        # Assigned in reset()
        self.roles:       Dict[AgentID, str] = {}   # "resistance" | "spy"
        self._teams_cache: Dict[TeamID, List[AgentID]] = {}

        self.phase:          str = PROPOSE
        self.leader_idx:     int = 0
        self.current_round:  int = 0
        self.missions_passed:  int = 0
        self.missions_failed:  int = 0
        self.failed_votes:   int = 0

        self.proposed_team:  List[AgentID] = []
        self.votes_buffer:   Dict[AgentID, str] = {}   # pending votes this phase
        self.history:        List[Dict[str, Any]] = []  # completed round records

        self._done:   bool = False
        self._winner: Optional[str] = None   # "resistance" or "spies"
        self._step:   int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return self._teams_cache  # set in reset()

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self._rng = random.Random(self._seed)

        # Randomly assign 2 spies
        spy_ids   = set(self._rng.sample(ALL_PLAYERS, 2))
        self.roles = {p: ("spy" if p in spy_ids else "resistance") for p in ALL_PLAYERS}
        self._teams_cache = {
            "resistance": [p for p in ALL_PLAYERS if self.roles[p] == "resistance"],
            "spies":      [p for p in ALL_PLAYERS if self.roles[p] == "spy"],
        }

        self.phase          = PROPOSE
        self.leader_idx     = 0
        self.current_round  = 0
        self.missions_passed = 0
        self.missions_failed = 0
        self.failed_votes   = 0
        self.proposed_team  = []
        self.votes_buffer   = {}
        self.history        = []
        self._done          = False
        self._winner        = None
        self._step          = 0

        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {p: 0.0 for p in ALL_PLAYERS}
        infos:   Dict[AgentID, Info]   = {p: {}  for p in ALL_PLAYERS}

        if self._done:
            return self._obs(), rewards, self._dones(), infos

        if self.phase == PROPOSE:
            self._handle_propose(actions_dict, infos)
        elif self.phase == VOTE:
            if self._collect_votes(actions_dict, infos):
                self._resolve_vote(infos)
        elif self.phase == MISSION:
            if self._collect_mission_votes(actions_dict, infos):
                self._resolve_mission(rewards, infos)

        self._step += 1
        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        role = self.roles.get(agent_id, "unknown")
        spy_ids = [p for p, r in self.roles.items() if r == "spy"]
        leader = ALL_PLAYERS[self.leader_idx % len(ALL_PLAYERS)]
        mission_size = MISSION_SIZES[self.current_round] if self.current_round < 5 else 0

        state: Dict[str, Any] = {
            "agent_id": agent_id,
            "your_role": role,
            "fellow_spies": (
                [p for p in spy_ids if p != agent_id]
                if role == "spy" else
                "[HIDDEN — you are Resistance and cannot see spy identities]"
            ),
            "phase": self.phase,
            "current_round": self.current_round + 1,
            "current_leader": leader,
            "is_leader": leader == agent_id,
            "mission_size_this_round": mission_size,
            "proposed_team": self.proposed_team,
            "in_proposed_team": agent_id in self.proposed_team,
            "missions_passed": self.missions_passed,
            "missions_failed": self.missions_failed,
            "consecutive_failed_votes": self.failed_votes,
            "votes_needed_to_win": {"resistance": 3 - self.missions_passed,
                                     "spies":      3 - self.missions_failed},
            "round_history": self.history,
            "legal_moves": self.get_legal_moves(agent_id),
            "game_over": self._done,
            "winner": self._winner,
            "strategic_context": self._strategic_hint(agent_id, role),
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        leader = ALL_PLAYERS[self.leader_idx % len(ALL_PLAYERS)]

        if self.phase == PROPOSE:
            if agent_id != leader:
                return []
            size = MISSION_SIZES[self.current_round]
            return [
                " ".join(combo)
                for combo in combinations(ALL_PLAYERS, size)
            ]

        if self.phase == VOTE:
            return ["approve", "reject"]

        if self.phase == MISSION:
            if agent_id not in self.proposed_team:
                return []
            role = self.roles.get(agent_id, "resistance")
            return ["pass", "sabotage"] if role == "spy" else ["pass"]

        return []

    def get_game_rules(self) -> str:
        return """
=== THE RESISTANCE — Game Rules ===

OVERVIEW
--------
5 players: 3 Resistance members and 2 Spies. Spies know each other's identity;
Resistance members know only their own role. Through 5 rounds of missions,
Resistance tries to pass 3 missions while Spies try to sabotage 3.

HIDDEN INFORMATION
------------------
  Spies     : know who the other spy is. Can choose to sabotage missions.
  Resistance: only know their own role. Must deduce spies from voting patterns.

  Mission votes (pass/sabotage) are SHUFFLED before being counted — you see
  HOW MANY sabotages occurred, but NOT who submitted them.

ROUND STRUCTURE
---------------
Each round has 3 phases:

  1. PROPOSE (leader only):
     The current leader selects a team of N players for the mission.
     Action: space-separated player IDs, e.g., "p0 p2"

  2. VOTE (all 5 players simultaneously):
     Each player votes to APPROVE or REJECT the proposed team.
     Action: "approve"  or  "reject"
     - Simple majority (≥3 votes) → approved, proceed to MISSION.
     - Majority rejects → FAILED VOTE. Next player becomes leader, propose again.
     - 5 consecutive failed votes → SPIES WIN immediately.

  3. MISSION (proposed team members only, simultaneously):
     Each team member plays a card:
       "pass"     → works for both Resistance and Spies
       "sabotage" → only available to Spies (Resistance CANNOT sabotage)
     Outcome:
       - Round 4 only: needs 2+ sabotages to fail. All others: 1+ sabotage fails.
       - Mission PASSES if 0 sabotages. FAILS otherwise.

WIN CONDITIONS
--------------
  Resistance wins : 3 missions passed.
  Spies win       : 3 missions failed  OR  5 consecutive rejected proposals.

MISSION TEAM SIZES (5 players)
-------------------------------
  Round 1: 2 players
  Round 2: 3 players
  Round 3: 2 players
  Round 4: 3 players (needs 2 sabotages to fail)
  Round 5: 3 players

STRATEGY NOTES
--------------
  Resistance: Observe voting patterns. Spies must sometimes approve to avoid
    suspicion, but may give themselves away by consistently voting together.
  Spies: Balance sabotage with appearing cooperative. Not every spy needs to
    sabotage every mission — one sabotage is enough (in rounds 1–3,5).

ACTION FORMAT
-------------
  PROPOSE : "p0 p2"  or  "p1 p2 p4"  (player IDs, space-separated)
  VOTE    : "approve"  or  "reject"
  MISSION : "pass"     or  "sabotage" (sabotage only legal for spies)
""".strip()

    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        try:
            import pygame
        except ImportError:
            return

        if not self._pygame_init:
            pygame.init()
            n_rounds = 5
            w = PAD * 2 + len(ALL_PLAYERS) * (P_RAD * 2 + P_GAP) - P_GAP
            w = max(w, 700)
            h = PAD * 3 + P_RAD * 2 + n_rounds * ROUND_H + INFO_H + 40
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · The Resistance")
            self._font  = pygame.font.SysFont("monospace", 18, bold=True)
            self._small = pygame.font.SysFont("monospace", 13)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr  = self._screen
        w    = scr.get_width()
        scr.fill(BG_COLOR)
        leader = ALL_PLAYERS[self.leader_idx % len(ALL_PLAYERS)]

        # Player row
        player_row_y = PAD + P_RAD
        n = len(ALL_PLAYERS)
        total_w = n * (P_RAD * 2 + P_GAP) - P_GAP
        start_x = (w - total_w) // 2 + P_RAD
        px_map: Dict[AgentID, int] = {}
        for i, pid in enumerate(ALL_PLAYERS):
            cx = start_x + i * (P_RAD * 2 + P_GAP)
            py = player_row_y
            px_map[pid] = cx
            role   = self.roles.get(pid, "?")
            color  = SPY_COLOR if role == "spy" else RES_COLOR
            border = GOLD_COLOR if pid == leader else (200, 200, 200)
            pygame.draw.circle(scr, color, (cx, py), P_RAD)
            pygame.draw.circle(scr, border, (cx, py), P_RAD, 3 if pid == leader else 1)
            # Crown for leader
            if pid == leader:
                lbl = self._small.render("👑", True, GOLD_COLOR)
                scr.blit(lbl, (cx - lbl.get_width() // 2, py - P_RAD - 16))
            # Player label
            lbl = self._small.render(pid, True, (255, 255, 255))
            scr.blit(lbl, (cx - lbl.get_width() // 2, py - lbl.get_height() // 2))
            # Role label below circle
            role_lbl = self._small.render(role[0].upper(), True, (200, 200, 200))
            scr.blit(role_lbl, (cx - role_lbl.get_width() // 2, py + P_RAD + 4))

        # Round history
        table_y = player_row_y + P_RAD + 40
        header_lbl = self._font.render("Rnd  Team Proposed      Votes  Result", True, FONT_COLOR)
        scr.blit(header_lbl, (PAD, table_y))
        pygame.draw.line(scr, (80, 80, 100), (PAD, table_y + 22), (w - PAD, table_y + 22))
        for i, rec in enumerate(self.history):
            row_y = table_y + 26 + i * ROUND_H
            team_str   = " ".join(rec.get("team", []))
            votes_str  = f"A:{rec.get('approve_count',0)} R:{rec.get('reject_count',0)}"
            result     = rec.get("result", "?")
            result_col = PASS_COL if result == "passed" else (
                          REJECT_COL if result == "vote_rejected" else FAIL_COL)
            fail_cnt   = rec.get("sabotage_count", "")
            line = f"  {rec.get('round',i)+1}    {team_str:<18} {votes_str:<12} {result}"
            if fail_cnt:
                line += f" (sab:{fail_cnt})"
            lbl = self._small.render(line, True, result_col)
            scr.blit(lbl, (PAD, row_y))

        # Current phase
        phase_y = table_y + 26 + 5 * ROUND_H + 10
        pygame.draw.rect(scr, PANEL_BG, (PAD, phase_y, w - PAD * 2, INFO_H), border_radius=8)
        pygame.draw.rect(scr, PANEL_BDR, (PAD, phase_y, w - PAD * 2, INFO_H), 1, border_radius=8)
        if self._done:
            msg   = f"GAME OVER — {self._winner.upper()} WIN!"
            color = RES_COLOR if self._winner == "resistance" else SPY_COLOR
        else:
            msg   = (f"Phase: {self.phase} | Round {self.current_round + 1} | "
                     f"Leader: {leader} | Proposed: {self.proposed_team} | "
                     f"Consecutive rejections: {self.failed_votes}")
            color = FONT_COLOR
        lbl = self._small.render(msg, True, color)
        scr.blit(lbl, (PAD + 8, phase_y + 12))
        score_str = f"Resistance: {self.missions_passed} passed  |  Spies: {self.missions_failed} failed"
        slbl = self._font.render(score_str, True, FONT_COLOR)
        scr.blit(slbl, (PAD + 8, phase_y + 40))

        pygame.display.flip()
        self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_propose(self, actions: Dict[AgentID, Action], infos: Dict[AgentID, Info]) -> None:
        leader = ALL_PLAYERS[self.leader_idx % len(ALL_PLAYERS)]
        if leader not in actions:
            return
        action = str(actions[leader]).strip()
        legal  = self.get_legal_moves(leader)
        if action not in legal:
            infos[leader] = {"error": f"Illegal proposal '{action}'"}
            return
        self.proposed_team = action.split()
        self.phase = VOTE
        self.votes_buffer = {}
        infos[leader] = {"proposed": self.proposed_team}

    def _collect_votes(self, actions: Dict[AgentID, Action], infos: Dict[AgentID, Info]) -> bool:
        """Accumulate votes. Returns True when all 5 have voted."""
        for pid in ALL_PLAYERS:
            if pid in actions and pid not in self.votes_buffer:
                v = str(actions[pid]).strip()
                if v in ("approve", "reject"):
                    self.votes_buffer[pid] = v
                    infos[pid] = {"voted": v}
        return len(self.votes_buffer) == len(ALL_PLAYERS)

    def _resolve_vote(self, infos: Dict[AgentID, Info]) -> None:
        approvals = sum(1 for v in self.votes_buffer.values() if v == "approve")
        rejections = len(ALL_PLAYERS) - approvals

        record: Dict[str, Any] = {
            "round":          self.current_round,
            "team":           list(self.proposed_team),
            "approve_count":  approvals,
            "reject_count":   rejections,
            "vote_breakdown": dict(self.votes_buffer),
        }

        if approvals > len(ALL_PLAYERS) // 2:
            # Approved → proceed to mission
            self.failed_votes = 0
            self.phase = MISSION
            self.votes_buffer = {}
            record["result"] = "approved"
            # We push the record only when the mission resolves
            # Stash it as pending
            self._pending_vote_record = record
        else:
            # Rejected
            self.failed_votes += 1
            record["result"] = "vote_rejected"
            self.history.append(record)

            if self.failed_votes >= MAX_FAILED_VOTES:
                self._done   = True
                self._winner = "spies"
            else:
                self.leader_idx += 1
                self.phase      = PROPOSE
                self.proposed_team = []
                self.votes_buffer  = {}

    def _collect_mission_votes(self, actions: Dict[AgentID, Action], infos: Dict[AgentID, Info]) -> bool:
        for pid in self.proposed_team:
            if pid in actions and pid not in self.votes_buffer:
                v = str(actions[pid]).strip()
                legal = self.get_legal_moves(pid)
                if v in legal:
                    self.votes_buffer[pid] = v
                    infos[pid] = {"mission_voted": "sabotage" if v == "sabotage" else "pass"}
        return len(self.votes_buffer) == len(self.proposed_team)

    def _resolve_mission(self, rewards: Dict[AgentID, Reward], infos: Dict[AgentID, Info]) -> None:
        sabotages = sum(1 for v in self.votes_buffer.values() if v == "sabotage")
        fails_needed = 2 if self.current_round == DOUBLE_FAIL_ROUND else 1
        mission_passed = sabotages < fails_needed

        record = getattr(self, "_pending_vote_record", {
            "round": self.current_round, "team": list(self.proposed_team)
        })
        record["sabotage_count"] = sabotages
        record["result"] = "passed" if mission_passed else "failed"
        self.history.append(record)

        if mission_passed:
            self.missions_passed += 1
        else:
            self.missions_failed += 1

        if self.missions_passed >= 3:
            self._done   = True
            self._winner = "resistance"
            for p in ALL_PLAYERS:
                rewards[p] = 1.0 if self.roles[p] == "resistance" else -1.0
        elif self.missions_failed >= 3:
            self._done   = True
            self._winner = "spies"
            for p in ALL_PLAYERS:
                rewards[p] = 1.0 if self.roles[p] == "spy" else -1.0
        else:
            self.current_round += 1
            self.leader_idx    += 1
            self.phase          = PROPOSE
            self.proposed_team  = []
            self.votes_buffer   = {}

    # ------------------------------------------------------------------
    def _strategic_hint(self, agent_id: AgentID, role: str) -> str:
        if role == "spy":
            spies = [p for p, r in self.roles.items() if r == "spy" and p != agent_id]
            return (f"You are a SPY. Your fellow spy is {spies}. "
                    "Blend in by sometimes voting approve; sabotage strategically.")
        return ("You are RESISTANCE. Observe who proposes and votes with whom. "
                "Spies will sometimes reveal themselves through suspicious patterns.")

    # ------------------------------------------------------------------
    def _obs(self) -> Dict[AgentID, Observation]:
        base = {
            "phase":           self.phase,
            "current_round":   self.current_round,
            "leader":          ALL_PLAYERS[self.leader_idx % len(ALL_PLAYERS)],
            "proposed_team":   list(self.proposed_team),
            "missions_passed": self.missions_passed,
            "missions_failed": self.missions_failed,
            "failed_votes":    self.failed_votes,
            "history":         self.history,
            "done":            self._done,
            "winner":          self._winner,
        }
        obs = {}
        for p in ALL_PLAYERS:
            o = dict(base)
            o["your_role"] = self.roles.get(p, "unknown")
            obs[p] = o
        return obs

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in ALL_PLAYERS}
        d["__all__"] = self._done
        return d
