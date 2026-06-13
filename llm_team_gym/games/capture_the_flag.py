"""
Capture the Flag — 2v2 team game on an open grid.

Four agents (A1, A2 vs B1, B2) navigate a rectangular grid.
Each team has a flag at their base. To score, grab the enemy flag and
return it to your own base. Get tagged by an enemy (occupy the same cell)
and you respawn at your base, dropping any flag you were carrying.

Grid layout (default 12 wide × 8 tall):
  Team A base zone : columns 0–1   (spawn at rows 3, 4)
  Team B base zone : columns 10–11 (spawn at rows 3, 4)
  Neutral zone     : columns 2–9
  Team A flag start: (3, 0)
  Team B flag start: (4, 11)

Turn order : A1 → B1 → A2 → B2 → A1 → …
Action format : "up" | "down" | "left" | "right" | "stay"
Win condition : first team to score CAPTURES_TO_WIN flag captures,
                OR the team with more captures after MAX_STEPS steps.

Teams : {"team_A": ["A1", "A2"], "team_B": ["B1", "B2"]}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

ROWS   = 8
COLS   = 12
CAPTURES_TO_WIN = 3
MAX_STEPS = 300

_DELTA: Dict[str, Tuple[int, int]] = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
    "stay":  ( 0,  0),
}

A_BASE_COLS: Set[int] = {0, 1}
B_BASE_COLS: Set[int] = {10, 11}
A_SPAWNS = [(3, 0), (4, 1)]
B_SPAWNS = [(3, 11), (4, 10)]
A_FLAG_START = (3, 0)
B_FLAG_START = (4, 11)

TURN_ORDER: Tuple[AgentID, ...] = ("A1", "B1", "A2", "B2")

# Pygame
CELL   = 60
PAD    = 16
INFO_H = 100
BG_COLOR      = (15,  15,  25)
NEUTRAL_COLOR = (35,  35,  50)
A_ZONE_COLOR  = (25,  55,  90)
B_ZONE_COLOR  = (90,  25,  25)
GRID_LINE     = (55,  55,  70)
A_COLOR       = (70, 130, 220)
B_COLOR       = (220, 70,  70)
FLAG_A_COLOR  = (150, 200, 255)
FLAG_B_COLOR  = (255, 160, 160)
FONT_COLOR    = (220, 220, 220)


class CaptureTheFlagGame(BaseGame):
    """
    2v2 Capture the Flag on a 12×8 open grid.

    State per agent: position, has_flag
    Team state: flag_at (current flag position), captures (score)
    """

    def __init__(
        self,
        rows: int = ROWS,
        cols: int = COLS,
        captures_to_win: int = CAPTURES_TO_WIN,
        max_steps: int = MAX_STEPS,
    ):
        self.rows = rows
        self.cols = cols
        self.captures_to_win = captures_to_win
        self.max_steps = max_steps

        # Agent positions
        self.positions:  Dict[AgentID, Tuple[int, int]] = {}
        self.has_flag:   Dict[AgentID, bool] = {}
        # Flag locations (None if an agent is carrying it)
        self.flag_pos:   Dict[str, Optional[Tuple[int, int]]] = {}
        # Team scores
        self.captures:   Dict[str, int] = {}
        self._turn_idx:  int = 0
        self._step:      int = 0
        self._done:      bool = False
        self._winner:    Optional[str] = None

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"team_A": ["A1", "A2"], "team_B": ["B1", "B2"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self.positions = {
            "A1": A_SPAWNS[0], "A2": A_SPAWNS[1],
            "B1": B_SPAWNS[0], "B2": B_SPAWNS[1],
        }
        self.has_flag  = {a: False for a in TURN_ORDER}
        self.flag_pos  = {"team_A": A_FLAG_START, "team_B": B_FLAG_START}
        self.captures  = {"team_A": 0, "team_B": 0}
        self._turn_idx = 0
        self._step     = 0
        self._done     = False
        self._winner   = None
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {a: 0.0 for a in TURN_ORDER}
        infos:   Dict[AgentID, Info]   = {a: {}  for a in TURN_ORDER}

        active = TURN_ORDER[self._turn_idx % len(TURN_ORDER)]
        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip().lower()
        legal  = self.get_legal_moves(active)
        if action not in legal:
            infos[active] = {"error": f"Illegal action '{action}'", "legal": legal}
            return self._obs(), rewards, self._dones(), infos

        team   = "team_A" if active in ("A1", "A2") else "team_B"
        enemy_team = "team_B" if team == "team_A" else "team_A"

        # Move
        dr, dc = _DELTA[action]
        old_r, old_c = self.positions[active]
        new_r = max(0, min(self.rows - 1, old_r + dr))
        new_c = max(0, min(self.cols - 1, old_c + dc))
        self.positions[active] = (new_r, new_c)

        event_log: List[str] = []

        # Pickup enemy flag
        if (not self.has_flag[active]
                and self.flag_pos[enemy_team] == (new_r, new_c)):
            self.has_flag[active] = True
            self.flag_pos[enemy_team] = None
            event_log.append("flag_pickup")

        # Score: return enemy flag to own base
        scored = False
        if self.has_flag[active] and self._in_base(new_r, new_c, team):
            self.has_flag[active] = False
            self.captures[team] += 1
            # Enemy flag respawns at their base
            if enemy_team == "team_A":
                self.flag_pos["team_A"] = A_FLAG_START
            else:
                self.flag_pos["team_B"] = B_FLAG_START
            rewards[active] += 2.0
            event_log.append("scored")
            scored = True

        # Tagging: check if we land on any enemy
        for enemy_id in TURN_ORDER:
            if self._same_team(active, enemy_id):
                continue
            if self.positions[enemy_id] == (new_r, new_c):
                # Tag the enemy
                enemy_team_e = "team_A" if enemy_id in ("A1", "A2") else "team_B"
                if self.has_flag[enemy_id]:
                    # Drop the flag where they are (current cell)
                    self.flag_pos[team] = (new_r, new_c)
                    self.has_flag[enemy_id] = False
                    event_log.append(f"recovered_flag_by_tagging_{enemy_id}")
                # Respawn enemy at their base
                spawn_idx = 0 if enemy_id in ("A1", "B1") else 1
                spawns = A_SPAWNS if enemy_team_e == "team_A" else B_SPAWNS
                self.positions[enemy_id] = spawns[spawn_idx]
                rewards[active] += 0.5
                event_log.append(f"tagged_{enemy_id}")

        self._turn_idx += 1
        self._step += 1
        infos[active] = {"events": event_log, "new_pos": (new_r, new_c)}

        # Win check
        if self.captures[team] >= self.captures_to_win:
            self._done = True
            self._winner = team
            for a in self.teams[team]:
                rewards[a] += 1.0
            for a in self.teams[enemy_team]:
                rewards[a] -= 1.0
        elif self._step >= self.max_steps:
            self._done = True
            if self.captures["team_A"] > self.captures["team_B"]:
                self._winner = "team_A"
            elif self.captures["team_B"] > self.captures["team_A"]:
                self._winner = "team_B"
            # else draw

        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        team   = "team_A" if agent_id in ("A1", "A2") else "team_B"
        enemy_team = "team_B" if team == "team_A" else "team_A"
        pos    = self.positions[agent_id]
        state = {
            "agent_id": agent_id,
            "team": team,
            "your_position": {"row": pos[0], "col": pos[1]},
            "carrying_flag": self.has_flag[agent_id],
            "is_your_turn": TURN_ORDER[self._turn_idx % len(TURN_ORDER)] == agent_id,
            "active_agent": TURN_ORDER[self._turn_idx % len(TURN_ORDER)],
            "step": self._step,
            "max_steps": self.max_steps,
            "captures": self.captures,
            "captures_to_win": self.captures_to_win,
            "your_flag_location": (
                "carried_by_enemy" if self.flag_pos[team] is None
                else list(self.flag_pos[team])
            ),
            "enemy_flag_location": (
                f"carried_by_{[a for a in TURN_ORDER if self.has_flag[a] and a not in self.teams[team]]}"
                if self.flag_pos[enemy_team] is None
                else list(self.flag_pos[enemy_team])
            ),
            "all_positions": {a: list(self.positions[a]) for a in TURN_ORDER},
            "teammates_carrying": {
                a: self.has_flag[a] for a in self.teams[team] if a != agent_id
            },
            "grid_info": {
                "rows": self.rows,
                "cols": self.cols,
                "your_base_cols": list(A_BASE_COLS if team == "team_A" else B_BASE_COLS),
                "enemy_base_cols": list(B_BASE_COLS if team == "team_A" else A_BASE_COLS),
            },
            "legal_moves": self.get_legal_moves(agent_id),
            "strategy_tip": (
                "Grab the enemy flag then return to YOUR base. "
                "Tag enemies (move onto their cell) to make them drop your flag."
            ),
            "winner": self._winner,
            "game_over": self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        active = TURN_ORDER[self._turn_idx % len(TURN_ORDER)]
        if active != agent_id:
            return []
        r, c = self.positions[agent_id]
        moves = []
        for action, (dr, dc) in _DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                moves.append(action)
        return moves

    def get_game_rules(self) -> str:
        return f"""
=== CAPTURE THE FLAG — Game Rules ===

SETUP
-----
Grid: {self.rows} rows × {self.cols} columns (open field, no obstacles).
Teams: Team A (agents A1, A2) vs Team B (agents B1, B2).

Base zones:
  Team A base : columns 0–1   (left side)
  Team B base : columns 10–11 (right side)
  Neutral zone: columns 2–9

Flags:
  Team A's flag starts at row 3, col 0 (Team A's base).
  Team B's flag starts at row 4, col 11 (Team B's base).

TURN ORDER
----------
A1 → B1 → A2 → B2 → A1 → …  (round-robin, one step per turn)

OBJECTIVE
---------
First team to capture {self.captures_to_win} enemy flags wins.
If max {self.max_steps} steps are reached, the team with more captures wins.

SCORING A CAPTURE (3 steps)
----------------------------
1. GRAB   : Move onto the cell where the ENEMY flag currently is.
            You are now "carrying" it (shown as carrying_flag = true).
2. RETURN : Move back into YOUR OWN BASE ZONE (your base columns) while carrying the flag.
            → +1 capture for your team; enemy flag resets to their base.
3. Score  : First to {self.captures_to_win} captures wins (+1.0 reward).

TAGGING
-------
Move onto the same cell as an ENEMY agent:
  → The enemy is tagged and immediately respawns at their base.
  → If the tagged enemy was carrying YOUR flag, your flag is dropped at that cell
    (you or a teammate must pick it up later and return it to your base OR it stays dropped).
  → Tagger earns +0.5 reward.

NOTES
-----
- You cannot be tagged in your own base zone. (Flags ARE droppable there if tagged outside.)
- Actually tagging is universal — plan accordingly: carrying the flag = high-risk state.
- The flag can be dropped mid-field if its carrier is tagged; any teammate can pick it up.

ACTION FORMAT
-------------
  "up"    — move 1 row up    (row index decreases)
  "down"  — move 1 row down  (row index increases)
  "left"  — move 1 col left
  "right" — move 1 col right
  "stay"  — do not move (valid, but wastes a turn)

Always choose from the provided legal_moves list.
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
            w = PAD * 2 + self.cols * CELL
            h = PAD * 2 + self.rows * CELL + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · Capture the Flag")
            self._font  = pygame.font.SysFont("monospace", 18, bold=True)
            self._small = pygame.font.SysFont("monospace", 13)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        scr.fill(BG_COLOR)

        # Draw grid cells
        for r in range(self.rows):
            for c in range(self.cols):
                x = PAD + c * CELL
                y = PAD + r * CELL
                if c in A_BASE_COLS:
                    bg = A_ZONE_COLOR
                elif c in B_BASE_COLS:
                    bg = B_ZONE_COLOR
                else:
                    bg = NEUTRAL_COLOR
                pygame.draw.rect(scr, bg, (x, y, CELL, CELL))
                pygame.draw.rect(scr, GRID_LINE, (x, y, CELL, CELL), 1)

        # Draw flags (when on ground)
        for team, fpos in self.flag_pos.items():
            if fpos is not None:
                fr, fc = fpos
                fx = PAD + fc * CELL + CELL // 2
                fy = PAD + fr * CELL + CELL // 2
                fcolor = FLAG_A_COLOR if team == "team_A" else FLAG_B_COLOR
                # Triangle flag
                points = [(fx, fy - 18), (fx + 14, fy - 10), (fx, fy - 2)]
                pygame.draw.polygon(scr, fcolor, points)
                pygame.draw.line(scr, fcolor, (fx, fy - 18), (fx, fy + 10), 2)

        # Draw agents
        for agent_id in TURN_ORDER:
            ar, ac = self.positions[agent_id]
            ax = PAD + ac * CELL + CELL // 2
            ay = PAD + ar * CELL + CELL // 2
            color = A_COLOR if agent_id in ("A1", "A2") else B_COLOR
            pygame.draw.circle(scr, color, (ax, ay), CELL // 2 - 6)
            pygame.draw.circle(scr, (255, 255, 255), (ax, ay), CELL // 2 - 6, 2)
            lbl = self._small.render(agent_id, True, (255, 255, 255))
            scr.blit(lbl, (ax - lbl.get_width() // 2, ay - lbl.get_height() // 2))
            if self.has_flag[agent_id]:
                # Flag indicator ring
                fcolor = FLAG_B_COLOR if agent_id in ("A1", "A2") else FLAG_A_COLOR
                pygame.draw.circle(scr, fcolor, (ax, ay), CELL // 2 - 4, 3)

        # Highlight active agent
        active = TURN_ORDER[self._turn_idx % len(TURN_ORDER)]
        ar, ac = self.positions[active]
        ax = PAD + ac * CELL + CELL // 2
        ay = PAD + ar * CELL + CELL // 2
        pygame.draw.circle(scr, (255, 255, 0), (ax, ay), CELL // 2 - 2, 2)

        # Info bar
        board_bottom = PAD * 2 + self.rows * CELL
        pygame.draw.rect(scr, (25, 25, 35), (0, board_bottom, scr.get_width(), INFO_H))
        if self._winner:
            msg   = f"Winner: {self._winner}! (A:{self.captures['team_A']} B:{self.captures['team_B']})"
            color = A_COLOR if self._winner == "team_A" else B_COLOR
        elif self._done:
            msg, color = "Draw! Max steps reached.", FONT_COLOR
        else:
            msg   = (f"Active: {active}   "
                     f"A: {self.captures['team_A']} caps   B: {self.captures['team_B']} caps   "
                     f"Step {self._step}/{self.max_steps}")
            color = A_COLOR if active in ("A1", "A2") else B_COLOR
        lbl = self._small.render(msg, True, color)
        scr.blit(lbl, (scr.get_width() // 2 - lbl.get_width() // 2, board_bottom + 22))
        legend = self._small.render(
            "Blue zone=Team A base   Red zone=Team B base   Yellow ring=active agent   Colored ring=carrying flag",
            True, (130, 130, 150),
        )
        scr.blit(legend, (scr.get_width() // 2 - legend.get_width() // 2, board_bottom + 50))

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
    def _in_base(self, r: int, c: int, team: str) -> bool:
        base_cols = A_BASE_COLS if team == "team_A" else B_BASE_COLS
        return c in base_cols

    def _same_team(self, a1: AgentID, a2: AgentID) -> bool:
        a_agents = {"A1", "A2"}
        return (a1 in a_agents) == (a2 in a_agents)

    def _obs(self) -> Dict[AgentID, Observation]:
        base = {
            "positions": {a: self.positions[a] for a in TURN_ORDER},
            "has_flag":  {a: self.has_flag[a]  for a in TURN_ORDER},
            "flag_pos":  {t: self.flag_pos[t]  for t in ("team_A", "team_B")},
            "captures":  dict(self.captures),
            "active":    TURN_ORDER[self._turn_idx % len(TURN_ORDER)],
            "step":      self._step,
            "winner":    self._winner,
            "done":      self._done,
        }
        return {a: dict(base) for a in TURN_ORDER}

    def _dones(self) -> Dict[AgentID, Done]:
        d = {a: self._done for a in TURN_ORDER}
        d["__all__"] = self._done
        return d
