"""
Connect Four — classic 6×7 gravity-drop game (1v1).

Players alternate dropping tokens into columns. A token falls to the
lowest empty row in the chosen column. First to align 4 in a row
(horizontal, vertical, or diagonal) wins.

Action format : column index as int or str  (0–6)
Turn order    : player_1 → player_2 → …
Teams         : {"player_1": ["player_1"], "player_2": ["player_2"]}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

ROWS, COLS = 6, 7
WIN_LEN = 4

# Pygame constants
CELL  = 82
GAP   = 6
PAD   = GAP
INFO_H = 100
BG_COLOR    = (10, 12, 20)
BOARD_COLOR = (18, 55, 140)
EMPTY_COLOR = (10, 14, 35)
P1_COLOR    = (0, 212, 190)
P2_COLOR    = (255, 80, 120)
FONT_COLOR  = (238, 242, 255)
PANEL_BG    = (18, 21, 34)
PANEL_BDR   = (42, 48, 72)
TEXT_SEC    = (130, 140, 175)
WARNING_CLR = (255, 215, 60)

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class ConnectFourGame(BaseGame):
    """
    Standard 6×7 Connect Four.

    grid[row][col]:  0 = empty, 1 = player_1, 2 = player_2
    Row 0 = top of the board; Row 5 = bottom (pieces fall downward).
    """

    def __init__(self, seed: Optional[int] = None):
        self.grid: List[List[int]] = []
        self._turn_player: AgentID = "player_1"
        self._done: bool = False
        self._winner: Optional[AgentID] = None
        self._step: int = 0
        self._last_col: Optional[int] = None
        self._last_row: Optional[int] = None

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"player_1": ["player_1"], "player_2": ["player_2"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self.grid = [[0] * COLS for _ in range(ROWS)]
        self._turn_player = "player_1"
        self._done = False
        self._winner = None
        self._step = 0
        self._last_col = self._last_row = None
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {"player_1": 0.0, "player_2": 0.0}
        infos:   Dict[AgentID, Info]   = {"player_1": {}, "player_2": {}}

        active = self._turn_player
        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = actions_dict[active]
        col = int(action)
        legal = self.get_legal_moves(active)

        if col not in legal and str(col) not in [str(m) for m in legal]:
            infos[active] = {"error": f"Illegal column {col}", "legal": legal}
            return self._obs(), rewards, self._dones(), infos

        # Drop piece
        row = self._drop_row(col)
        token = 1 if active == "player_1" else 2
        self.grid[row][col] = token
        self._last_col, self._last_row = col, row
        self._step += 1

        # Check win
        if self._check_win(row, col, token):
            self._done = True
            self._winner = active
            rewards[active] = 1.0
            opp = "player_2" if active == "player_1" else "player_1"
            rewards[opp] = -1.0
        elif all(self.grid[0][c] != 0 for c in range(COLS)):
            # Board full — draw
            self._done = True
        else:
            self._turn_player = "player_2" if active == "player_1" else "player_1"

        infos[active] = {"placed_at": (row, col)}
        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        token = 1 if agent_id == "player_1" else 2
        state = {
            "agent_id": agent_id,
            "your_token": token,
            "is_your_turn": self._turn_player == agent_id,
            "active_player": self._turn_player,
            "step": self._step,
            "grid": self.grid,
            "grid_legend": {"0": "empty", "1": "player_1 (red)", "2": "player_2 (yellow)"},
            "legal_columns": self.get_legal_moves(agent_id),
            "last_move": {"col": self._last_col, "row": self._last_row},
            "winner": self._winner,
            "game_over": self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or self._turn_player != agent_id:
            return []
        return [c for c in range(COLS) if self.grid[0][c] == 0]

    def get_game_rules(self) -> str:
        return """
=== CONNECT FOUR — Game Rules ===

OBJECTIVE
---------
Be the first player to connect 4 of your tokens in a straight line:
horizontally, vertically, or diagonally.

BOARD
-----
6 rows × 7 columns. Tokens fall under gravity to the lowest empty row
in the chosen column. Row 0 is the top, Row 5 is the bottom.

TOKENS
------
  player_1 = 1 (red)    plays first
  player_2 = 2 (yellow)

TURN STRUCTURE
--------------
Players alternate turns. On your turn choose a column (0–6).
Your token drops to the lowest empty cell in that column.
A full column (top row occupied) is illegal.

WIN CONDITION
-------------
4 consecutive tokens of your color in any direction.
If the board fills with no winner, the game is a draw.

ACTION FORMAT
-------------
  A single integer: the column index (0 to 6 inclusive).
  Example: 3  → drop token in the centre column.

Always choose from the provided legal_columns list.
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
            w = PAD + COLS * (CELL + GAP)
            h = PAD + (ROWS + 1) * (CELL + GAP) + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · Connect Four")
            self._font  = pygame.font.SysFont("monospace", 24, bold=True)
            self._small = pygame.font.SysFont("monospace", 16)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        scr.fill(BG_COLOR)

        board_top = PAD + CELL + GAP  # row 0 of board starts below column header
        board_w = COLS * (CELL + GAP) + GAP
        board_h = ROWS * (CELL + GAP) + GAP
        pygame.draw.rect(scr, BOARD_COLOR, (PAD, board_top - GAP, board_w, board_h + GAP), border_radius=10)

        for r in range(ROWS):
            for c in range(COLS):
                x = PAD + GAP + c * (CELL + GAP)
                y = board_top + r * (CELL + GAP)
                val = self.grid[r][c]
                color = P1_COLOR if val == 1 else P2_COLOR if val == 2 else EMPTY_COLOR
                pygame.draw.circle(scr, color, (x + CELL // 2, y + CELL // 2), CELL // 2 - 2)
                if r == self._last_row and c == self._last_col:
                    pygame.draw.circle(scr, (255, 255, 255), (x + CELL // 2, y + CELL // 2), CELL // 2 - 2, 3)

        # Column header numbers
        for c in range(COLS):
            x = PAD + GAP + c * (CELL + GAP) + CELL // 2
            lbl = self._small.render(str(c), True, FONT_COLOR)
            scr.blit(lbl, (x - lbl.get_width() // 2, PAD))

        # Info bar
        info_y = board_top + board_h + GAP
        pygame.draw.rect(scr, PANEL_BG, (0, info_y, scr.get_width(), INFO_H))
        if self._winner:
            msg = f"Winner: {self._winner}!"
            color = P1_COLOR if self._winner == "player_1" else P2_COLOR
        elif self._done:
            msg = "Draw!"
            color = FONT_COLOR
        else:
            msg = f"Turn: {self._turn_player}"
            color = P1_COLOR if self._turn_player == "player_1" else P2_COLOR
        lbl = self._font.render(msg, True, color)
        scr.blit(lbl, (scr.get_width() // 2 - lbl.get_width() // 2, info_y + 20))
        step_lbl = self._small.render(f"Step {self._step}", True, FONT_COLOR)
        scr.blit(step_lbl, (scr.get_width() // 2 - step_lbl.get_width() // 2, info_y + 55))

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
    # Helpers
    # ------------------------------------------------------------------

    def _drop_row(self, col: int) -> int:
        for r in range(ROWS - 1, -1, -1):
            if self.grid[r][col] == 0:
                return r
        raise ValueError(f"Column {col} is full.")

    def _check_win(self, row: int, col: int, token: int) -> bool:
        for dr, dc in _DIRS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == token:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= WIN_LEN:
                return True
        return False

    def _obs(self) -> Dict[AgentID, Observation]:
        snapshot = {
            "grid": [row[:] for row in self.grid],
            "active_player": self._turn_player,
            "step": self._step,
            "winner": self._winner,
            "done": self._done,
        }
        return {"player_1": dict(snapshot), "player_2": dict(snapshot)}

    def _dones(self) -> Dict[AgentID, Done]:
        return {"player_1": self._done, "player_2": self._done, "__all__": self._done}
