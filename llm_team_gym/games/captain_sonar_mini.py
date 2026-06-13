"""
Captain Sonar Mini — 2v2 asymmetric grid coordination game.

Two teams each operate a submarine on a shared 8×8 grid. Each team has:

  Captain (CA / CB)        : knows own sub's exact position; chooses movement.
  Radio Operator (RA / RB) : cannot see own sub's position; tracks the ENEMY
                              movement log to deduce their location; fires torpedoes.

The teams alternate:  CA moves → RA acts → CB moves → RB acts → repeat.

Hidden information
------------------
  Enemy sub position is NEVER directly visible to either player.
  The ONLY information about the enemy comes from:
    1. Their publicly-announced movement directions  (e.g., "north", "east")
    2. Sonar ping results (sector hint)
    3. Torpedo hit/miss feedback

  Own sub's position IS known to the Captain and Radio Operator on the SAME team
  (shared via the "team_channel" field in both text states).

Grid & movement
---------------
  8 rows × 8 columns, 0-indexed.  Several fixed islands block movement.
  Subs move 1 step per turn. Cannot enter islands or move off-grid.
  Captains surface (reveal position) to reset their visited-cell restriction
  (optional rule enforced: subs may NOT revisit a cell without surfacing first).

Radio Operator actions
----------------------
  "fire <row> <col>"  — launch torpedo at exact grid cell
                         Hit if it lands on enemy sub → enemy loses 1 life.
  "sonar"             — reveals which quadrant (NW/NE/SW/SE) enemy sub is in.
  "pass"              — no action this turn.

Captain actions
---------------
  "north" / "south" / "east" / "west"  — move sub 1 step (direction announced publicly)
  "surface"  — reveal exact position; reset visited-cell set; next movement unrestricted.

Lives : each team starts with 4.
Win   : first team to reduce the enemy to 0 lives wins.
Max   : 200 steps (then highest-life team wins, or draw).

Agents : "CA", "RA", "CB", "RB"
Teams  : {"team_A": ["CA", "RA"], "team_B": ["CB", "RB"]}
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

ROWS, COLS = 8, 8
START_LIVES = 4
MAX_STEPS   = 200

# Fixed island cells (row, col) — subs cannot enter these
ISLANDS: FrozenSet[Tuple[int, int]] = frozenset([
    (1, 3), (2, 5), (3, 2), (4, 4), (5, 1), (5, 6), (7, 3)
])

DELTAS: Dict[str, Tuple[int, int]] = {
    "north": (-1,  0),
    "south": ( 1,  0),
    "west":  ( 0, -1),
    "east":  ( 0,  1),
}

# Phase order
PHASES = ("CA_MOVE", "RA_ACT", "CB_MOVE", "RB_ACT")

# Quadrant helper
def _quadrant(r: int, c: int) -> str:
    return ("NW" if c < COLS // 2 else "NE") if r < ROWS // 2 else \
           ("SW" if c < COLS // 2 else "SE")

# Pygame
CELL   = 70
PAD    = 20
INFO_H = 110
BG_COLOR    = (15,  15,  25)
OCEAN_COLOR = (20,  40,  90)
ISLAND_COLOR= (90,  70,  40)
GRID_LINE   = (30,  60, 110)
A_COLOR     = (70, 130, 220)
B_COLOR     = (220, 70,  70)
SONAR_COLOR = (200, 200,  80)
FONT_COLOR  = (220, 220, 220)
VISIT_A     = (40,  80, 140)
VISIT_B     = (140, 40,  40)


class CaptainSonarMiniGame(BaseGame):
    """
    Captain Sonar Mini — 4 players (CA, RA, CB, RB) on an 8×8 grid.

    State exposed per agent is carefully filtered to enforce hidden information.
    """

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng  = random.Random(seed)

        # Sub positions (secret from enemy)
        self.pos:     Dict[str, Tuple[int, int]] = {}   # "A" / "B" → (r, c)
        self.visited: Dict[str, Set[Tuple[int, int]]] = {}

        # Movement logs (public — announced each turn)
        self.move_log: Dict[str, List[str]] = {"A": [], "B": []}

        # Sonar results (private per team)
        self.sonar_results: Dict[str, List[str]] = {"A": [], "B": []}

        # Team channel: last 5 actions of each agent (visible to teammate only)
        self.team_log: Dict[AgentID, List[str]] = {a: [] for a in ("CA", "RA", "CB", "RB")}

        self.lives:    Dict[str, int] = {"A": START_LIVES, "B": START_LIVES}
        self._phase_idx: int = 0
        self._step:      int = 0
        self._done:      bool = False
        self._winner:    Optional[str] = None   # "team_A" | "team_B"

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"team_A": ["CA", "RA"], "team_B": ["CB", "RB"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self._rng = random.Random(self._seed)
        valid = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in ISLANDS]

        # Place subs at least 4 cells apart
        pos_a = self._rng.choice(valid)
        far   = [p for p in valid if abs(p[0]-pos_a[0]) + abs(p[1]-pos_a[1]) >= 4]
        pos_b = self._rng.choice(far if far else valid)

        self.pos      = {"A": pos_a, "B": pos_b}
        self.visited  = {"A": {pos_a}, "B": {pos_b}}
        self.move_log = {"A": [], "B": []}
        self.sonar_results = {"A": [], "B": []}
        self.team_log = {"CA": [], "RA": [], "CB": [], "RB": []}
        self.lives    = {"A": START_LIVES, "B": START_LIVES}
        self._phase_idx = 0
        self._step      = 0
        self._done      = False
        self._winner    = None
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {a: 0.0 for a in ("CA", "RA", "CB", "RB")}
        infos:   Dict[AgentID, Info]   = {a: {}  for a in ("CA", "RA", "CB", "RB")}

        if self._done:
            return self._obs(), rewards, self._dones(), infos

        phase   = PHASES[self._phase_idx % len(PHASES)]
        active  = phase.split("_")[0] + ("" if "_" not in phase else "")
        # Map phase to agent
        phase_agent = {"CA_MOVE": "CA", "RA_ACT": "RA", "CB_MOVE": "CB", "RB_ACT": "RB"}[phase]

        if phase_agent not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[phase_agent]).strip().lower()
        legal  = self.get_legal_moves(phase_agent)

        if action not in legal:
            infos[phase_agent] = {"error": f"Illegal action '{action}'", "legal": legal[:10]}
            return self._obs(), rewards, self._dones(), infos

        team = "A" if phase_agent in ("CA", "RA") else "B"
        enemy = "B" if team == "A" else "A"

        if phase in ("CA_MOVE", "CB_MOVE"):
            self._handle_captain(phase_agent, team, action, infos)
        else:
            self._handle_radio(phase_agent, team, enemy, action, infos, rewards)

        self._step      += 1
        self._phase_idx  = (self._phase_idx + 1) % len(PHASES)

        if self._step >= MAX_STEPS and not self._done:
            self._done = True
            la, lb = self.lives["A"], self.lives["B"]
            if la > lb:
                self._winner = "team_A"
                rewards["CA"] = rewards["RA"] = 1.0
                rewards["CB"] = rewards["RB"] = -1.0
            elif lb > la:
                self._winner = "team_B"
                rewards["CB"] = rewards["RB"] = 1.0
                rewards["CA"] = rewards["RA"] = -1.0

        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        team  = "A" if agent_id in ("CA", "RA") else "B"
        enemy = "B" if team == "A" else "A"
        phase = PHASES[self._phase_idx % len(PHASES)]
        is_captain = agent_id in ("CA", "CB")
        teammate   = {"CA": "RA", "RA": "CA", "CB": "RB", "RB": "CB"}[agent_id]

        state: Dict[str, Any] = {
            "agent_id": agent_id,
            "role": "Captain" if is_captain else "Radio Operator",
            "team": f"team_{team}",
            "phase": phase,
            "is_your_turn": PHASES[self._phase_idx % len(PHASES)].split("_")[0] + ("" if "_" not in phase else "") == agent_id.split("_")[0] if False else
                            PHASES[self._phase_idx % len(PHASES)] == (f"C{team}_MOVE" if is_captain else f"R{team}_ACT"),
            "step": self._step,
            "lives": {f"team_{team}": self.lives[team], f"team_{enemy}": self.lives[enemy]},
            "team_channel": {
                "description": "Recent actions from your teammate (shared within team only):",
                "teammate_log": self.team_log[teammate][-5:],
            },
        }

        if is_captain:
            state["your_submarine"] = {
                "position": {"row": self.pos[team][0], "col": self.pos[team][1]},
                "visited_cells": sorted(self.visited[team]),
                "note": (
                    "YOU know your exact position. Enemy does NOT. "
                    "Announce direction to Radio Operator via team_channel context."
                ),
            }
        else:
            state["your_submarine"] = {
                "position": "HIDDEN — Radio Operators do not track own sub position directly.",
                "own_movement_log": self.move_log[team],
                "note": (
                    "Deduce own sub's position from the movement log + starting area. "
                    "Your Captain will surface to confirm if needed."
                ),
            }

        # Enemy tracking info (available to both captain and radio op)
        state["enemy_tracking"] = {
            "enemy_movement_log": self.move_log[enemy],
            "movement_log_length": len(self.move_log[enemy]),
            "sonar_results_this_team": self.sonar_results[team],
            "grid_info": {
                "rows": ROWS, "cols": COLS,
                "islands": sorted(ISLANDS),
                "quadrants": {"NW": "rows 0-3 cols 0-3", "NE": "rows 0-3 cols 4-7",
                              "SW": "rows 4-7 cols 0-3", "SE": "rows 4-7 cols 4-7"},
            },
            "deduction_note": (
                "The enemy started somewhere on the grid. Eliminate impossible positions "
                "by applying their movement announcements against island locations and grid edges. "
                "Sonar narrows to one quadrant. Fire when confident."
            ),
        }

        state["legal_moves"] = self.get_legal_moves(agent_id)
        state["game_over"]   = self._done
        state["winner"]      = self._winner
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        phase = PHASES[self._phase_idx % len(PHASES)]
        expected_agent = {"CA_MOVE": "CA", "RA_ACT": "RA", "CB_MOVE": "CB", "RB_ACT": "RB"}[phase]
        if agent_id != expected_agent:
            return []

        team = "A" if agent_id in ("CA", "CB") else "B"
        is_captain = agent_id in ("CA", "CB")

        if is_captain:
            r, c  = self.pos[team]
            moves = ["surface"]
            for direction, (dr, dc) in DELTAS.items():
                nr, nc = r + dr, c + dc
                if (0 <= nr < ROWS and 0 <= nc < COLS
                        and (nr, nc) not in ISLANDS
                        and (nr, nc) not in self.visited[team]):
                    moves.append(direction)
            # If completely boxed in except surface: surface is the only option (already included)
            return moves

        else:
            # Radio Operator
            moves = ["pass", "sonar"]
            for r in range(ROWS):
                for c in range(COLS):
                    moves.append(f"fire {r} {c}")
            return moves

    def get_game_rules(self) -> str:
        return """
=== CAPTAIN SONAR MINI — Game Rules ===

OVERVIEW
--------
2v2 asymmetric submarine warfare. Team A (CA, RA) vs Team B (CB, RB).
Each team controls a submarine moving on an 8×8 ocean grid.
The teams alternate turns. Within each team, Captain and Radio Operator
act in sequence (Captain first, then Radio Operator).

ROLES
-----
  Captain (CA / CB)
    • Knows own submarine's EXACT position.
    • Chooses movement direction each turn.
    • Announces direction PUBLICLY (enemy radio op hears it).
    • Can surface to reveal position and reset visited-cell restriction.

  Radio Operator (RA / RB)
    • CANNOT directly see own submarine's position.
    • Hears own Captain's moves via team_channel.
    • Hears enemy's movement announcements.
    • Tracks possible enemy positions via the movement log + sonar.
    • Fires torpedoes and uses sonar.

HIDDEN INFORMATION
------------------
  Enemy sub position is NEVER directly shown. Deduce it by:
    1. Tracking the enemy movement log (every direction they announce).
    2. Using sonar to learn their quadrant.
    3. Applying island & grid-edge constraints to eliminate possibilities.

  Own sub position IS visible to the Captain. Radio Operator deduces
  it from the own movement log or asks Captain to surface.

TURN ORDER
----------
  CA moves → RA acts → CB moves → RB acts → repeat.

CAPTAIN ACTIONS (per turn, announce direction publicly)
--------------------------------------------------------
  "north"   — move sub 1 row up    (row index decreases)
  "south"   — move sub 1 row down  (row index increases)
  "west"    — move sub 1 column left
  "east"    — move sub 1 column right
  "surface" — reveal exact position to ALL; reset visited-cell set.
              No torpedo or sonar penalty, but gives away location!

  VISITED CELL RULE: Subs may NOT revisit a previously occupied cell
  without surfacing first. "surface" clears the restriction.

RADIO OPERATOR ACTIONS (per turn)
----------------------------------
  "fire <row> <col>" — launch torpedo. Hits if enemy sub is at exactly (row, col).
                        HIT → enemy loses 1 life. MISS → nothing.
  "sonar"            — ping: learn which quadrant (NW/NE/SW/SE) enemy is in.
                        Result stored in sonar_results (visible to team only).
  "pass"             — no action this turn.

GRID
----
  8 rows × 8 columns (row 0 = north edge, row 7 = south edge).
  Islands (impassable): (1,3) (2,5) (3,2) (4,4) (5,1) (5,6) (7,3)
  Quadrants: NW=rows 0-3,cols 0-3 | NE=rows 0-3,cols 4-7
             SW=rows 4-7,cols 0-3 | SE=rows 4-7,cols 4-7

LIVES & WIN
-----------
  Each team starts with 4 lives. Torpedo hit → -1 life.
  First team to reach 0 lives LOSES.
  If max steps (200) reached: team with more lives wins. Tie = draw.

TEAM COMMUNICATION
------------------
  team_channel shows the last 5 actions of your teammate.
  Use this to coordinate: Captain's surfacing or movement
  context helps Radio Operator decide whether to fire now.

ACTION FORMAT
-------------
  Captain  : "north"   "south"   "east"   "west"   "surface"
  Radio Op : "fire 3 5"   "sonar"   "pass"
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
            board_px = ROWS * CELL
            w = PAD * 3 + board_px * 2   # two grids side by side
            h = PAD * 2 + board_px + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · Captain Sonar Mini")
            self._font  = pygame.font.SysFont("monospace", 16, bold=True)
            self._small = pygame.font.SysFont("monospace", 12)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        w   = scr.get_width()
        scr.fill(BG_COLOR)
        board_px = ROWS * CELL

        def draw_grid(offset_x, team, enemy):
            color  = A_COLOR if team == "A" else B_COLOR
            v_col  = VISIT_A if team == "A" else VISIT_B
            e_color= B_COLOR if team == "A" else A_COLOR
            for r in range(ROWS):
                for c in range(COLS):
                    x = offset_x + c * CELL
                    y = PAD + r * CELL
                    if (r, c) in ISLANDS:
                        pygame.draw.rect(scr, ISLAND_COLOR, (x, y, CELL, CELL))
                    elif (r, c) in self.visited[team]:
                        pygame.draw.rect(scr, v_col, (x, y, CELL, CELL))
                    else:
                        pygame.draw.rect(scr, OCEAN_COLOR, (x, y, CELL, CELL))
                    pygame.draw.rect(scr, GRID_LINE, (x, y, CELL, CELL), 1)

            # Own sub
            sr, sc = self.pos[team]
            sx = offset_x + sc * CELL + CELL // 2
            sy = PAD + sr * CELL + CELL // 2
            pygame.draw.circle(scr, color, (sx, sy), CELL // 3)
            pygame.draw.circle(scr, (255, 255, 255), (sx, sy), CELL // 3, 2)
            lbl = self._small.render(f"T{team}", True, (255, 255, 255))
            scr.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))

            # Enemy sub (shown for god-view in render only)
            er, ec = self.pos[enemy]
            ex_px = offset_x + ec * CELL + CELL // 2
            ey_px = PAD + er * CELL + CELL // 2
            pygame.draw.circle(scr, e_color, (ex_px, ey_px), CELL // 4)
            pygame.draw.circle(scr, (255, 255, 255), (ex_px, ey_px), CELL // 4, 1)

            # Grid labels
            tl = self._font.render(f"Team {team} view", True, color)
            scr.blit(tl, (offset_x, PAD - 18))
            life_lbl = self._font.render(f"Lives: {self.lives[team]}", True, color)
            scr.blit(life_lbl, (offset_x + board_px - life_lbl.get_width() - 4, PAD - 18))

        draw_grid(PAD, "A", "B")
        draw_grid(PAD * 2 + board_px, "B", "A")

        # Info bar
        info_y = PAD + board_px + 10
        pygame.draw.rect(scr, (25, 25, 35), (0, info_y, w, INFO_H))
        phase = PHASES[self._phase_idx % len(PHASES)]
        if self._done:
            msg   = f"GAME OVER — {self._winner or 'Draw'}  (A:{self.lives['A']} B:{self.lives['B']})"
            color = A_COLOR if self._winner == "team_A" else (B_COLOR if self._winner == "team_B" else FONT_COLOR)
        else:
            msg   = (f"Phase: {phase}  |  Step: {self._step}/{MAX_STEPS}  |  "
                     f"Lives — A:{self.lives['A']}  B:{self.lives['B']}")
            color = A_COLOR if "A" in phase else B_COLOR
        lbl = self._font.render(msg, True, color)
        scr.blit(lbl, (PAD, info_y + 10))

        ml_a = "A moves: " + " ".join(self.move_log["A"][-10:])
        ml_b = "B moves: " + " ".join(self.move_log["B"][-10:])
        scr.blit(self._small.render(ml_a, True, A_COLOR), (PAD, info_y + 40))
        scr.blit(self._small.render(ml_b, True, B_COLOR), (PAD, info_y + 58))
        scr.blit(self._small.render("(Filled circle=own sub, hollow=enemy — god view only)", True, (100,100,120)), (PAD, info_y + 78))

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
    def _handle_captain(self, agent: AgentID, team: str, action: str, infos: Dict) -> None:
        self.team_log[agent].append(action)
        if action == "surface":
            r, c = self.pos[team]
            self.visited[team] = {(r, c)}
            self.move_log[team].append("SURFACE")
            infos[agent] = {"surfaced_at": (r, c)}
        else:
            dr, dc  = DELTAS[action]
            r, c    = self.pos[team]
            nr, nc  = r + dr, c + dc
            self.pos[team] = (nr, nc)
            self.visited[team].add((nr, nc))
            self.move_log[team].append(action[0].upper())  # "N"/"S"/"E"/"W"
            infos[agent] = {"moved": action, "new_pos": (nr, nc)}

    def _handle_radio(
        self, agent: AgentID, team: str, enemy: str,
        action: str, infos: Dict, rewards: Dict
    ) -> None:
        self.team_log[agent].append(action)
        if action == "pass":
            infos[agent] = {"action": "pass"}
            return

        if action == "sonar":
            er, ec = self.pos[enemy]
            quad = _quadrant(er, ec)
            result = f"step_{self._step}:enemy_in_{quad}"
            self.sonar_results[team].append(result)
            infos[agent] = {"sonar_result": result}
            return

        # fire row col
        parts = action.split()
        tr, tc = int(parts[1]), int(parts[2])
        er, ec = self.pos[enemy]
        if (tr, tc) == (er, ec):
            self.lives[enemy] -= 1
            infos[agent] = {"fired_at": (tr, tc), "hit": True, "enemy_lives_left": self.lives[enemy]}
            if self.lives[enemy] <= 0:
                self._done   = True
                self._winner = f"team_{team}"
                rewards["CA" if team=="A" else "CB"] = 1.0
                rewards["RA" if team=="A" else "RB"] = 1.0
                rewards["CB" if team=="A" else "CA"] = -1.0
                rewards["RB" if team=="A" else "RA"] = -1.0
        else:
            infos[agent] = {"fired_at": (tr, tc), "hit": False}

    def _obs(self) -> Dict[AgentID, Observation]:
        phase = PHASES[self._phase_idx % len(PHASES)]
        base  = {
            "phase":      phase,
            "step":       self._step,
            "move_log_A": list(self.move_log["A"]),
            "move_log_B": list(self.move_log["B"]),
            "lives":      dict(self.lives),
            "done":       self._done,
            "winner":     self._winner,
        }
        obs: Dict[AgentID, Observation] = {}
        for agent in ("CA", "RA", "CB", "RB"):
            team  = "A" if agent in ("CA", "RA") else "B"
            enemy = "B" if team == "A" else "A"
            o = dict(base)
            o["team_own_pos"]    = self.pos[team]    # both CA and RA see own pos
            o["sonar_results"]   = list(self.sonar_results[team])
            o["team_log"]        = {a: list(self.team_log[a]) for a in (
                ("CA", "RA") if team == "A" else ("CB", "RB")
            )}
            obs[agent] = o
        return obs

    def _dones(self) -> Dict[AgentID, Done]:
        d = {a: self._done for a in ("CA", "RA", "CB", "RB")}
        d["__all__"] = self._done
        return d
