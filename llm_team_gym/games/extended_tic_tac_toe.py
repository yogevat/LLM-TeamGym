"""
Extended Tic-Tac-Toe — 5×5 board, need 4-in-a-row to win (1v1).

Classic Tic-Tac-Toe scaled up: the larger board prevents trivial first-mover
advantage and requires actual strategic positioning.

Action format : "row col"  (e.g., "2 3" for row 2, column 3, zero-indexed)
Turn order    : player_X → player_O → …
Teams         : {"player_X": ["player_X"], "player_O": ["player_O"]}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

SIZE    = 5
WIN_LEN = 4

# Pygame
CELL   = 110
GAP    = 6
PAD    = 20
INFO_H = 100

BG_COLOR   = (10, 12, 20)
GRID_COLOR = (42, 48, 72)
X_COLOR    = (0, 212, 190)
O_COLOR    = (255, 80, 120)
WIN_COLOR  = (255, 215, 60)
FONT_COLOR = (238, 242, 255)
EMPTY_COLOR= (18, 21, 34)
PANEL_BG   = (18, 21, 34)
PANEL_BDR  = (42, 48, 72)
TEXT_SEC   = (130, 140, 175)
WARNING_CLR= (255, 215, 60)

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class ExtendedTicTacToeGame(BaseGame):
    """
    5×5 Tic-Tac-Toe where 4 consecutive marks wins.

    grid[row][col]: 0 = empty, 1 = X (player_X), 2 = O (player_O)
    """

    def __init__(self):
        self.grid: List[List[int]] = []
        self._turn: AgentID = "player_X"
        self._done: bool = False
        self._winner: Optional[AgentID] = None
        self._winning_cells: List[Tuple[int, int]] = []
        self._step: int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"player_X": ["player_X"], "player_O": ["player_O"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self.grid = [[0] * SIZE for _ in range(SIZE)]
        self._turn = "player_X"
        self._done = False
        self._winner = None
        self._winning_cells = []
        self._step = 0
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {"player_X": 0.0, "player_O": 0.0}
        infos:   Dict[AgentID, Info]   = {"player_X": {}, "player_O": {}}

        active = self._turn
        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip()
        legal  = self.get_legal_moves(active)
        if action not in legal:
            infos[active] = {"error": f"Illegal move '{action}'", "legal": legal}
            return self._obs(), rewards, self._dones(), infos

        row, col = map(int, action.split())
        token = 1 if active == "player_X" else 2
        self.grid[row][col] = token
        self._step += 1

        win_cells = self._find_win(row, col, token)
        if win_cells:
            self._done = True
            self._winner = active
            self._winning_cells = win_cells
            rewards[active] = 1.0
            opp = self._opponent(active)
            rewards[opp] = -1.0
        elif all(self.grid[r][c] != 0 for r in range(SIZE) for c in range(SIZE)):
            self._done = True   # draw
        else:
            self._turn = self._opponent(active)

        infos[active] = {"placed_at": (row, col)}
        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        mark = "X" if agent_id == "player_X" else "O"
        state = {
            "agent_id": agent_id,
            "your_mark": mark,
            "is_your_turn": self._turn == agent_id,
            "active_player": self._turn,
            "step": self._step,
            "grid": self.grid,
            "grid_legend": {"0": "empty", "1": "X (player_X)", "2": "O (player_O)"},
            "legal_moves": self.get_legal_moves(agent_id),
            "winner": self._winner,
            "game_over": self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or self._turn != agent_id:
            return []
        return [
            f"{r} {c}"
            for r in range(SIZE)
            for c in range(SIZE)
            if self.grid[r][c] == 0
        ]

    def get_game_rules(self) -> str:
        return """
=== EXTENDED TIC-TAC-TOE (5×5, 4-in-a-row) — Game Rules ===

OBJECTIVE
---------
Be the first player to place 4 of your marks in an unbroken line.

BOARD
-----
5 rows × 5 columns (row 0 = top-left, row 4 = bottom-right).
Cells are identified by "row col" (0-indexed).

MARKS
-----
  player_X = 1   plays first (X)
  player_O = 2   plays second (O)

TURN STRUCTURE
--------------
Players alternate. On your turn, choose any empty cell.

WIN CONDITION
-------------
4 consecutive identical marks in a row, column, or diagonal.
The 5×5 board allows for 12 possible winning lines (vs 8 in 3×3).
If the board fills with no winner, the result is a draw.

ACTION FORMAT
-------------
  "row col"  — two space-separated integers, each 0–4.
  Example: "2 3" places your mark at row 2, column 3.

Always choose from the provided legal_moves list exactly as shown.
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
            w = PAD * 2 + SIZE * (CELL + GAP) - GAP
            h = PAD * 2 + SIZE * (CELL + GAP) - GAP + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · 5×5 Tic-Tac-Toe")
            self._font  = pygame.font.SysFont("monospace", 52, bold=True)
            self._small = pygame.font.SysFont("monospace", 18)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        scr.fill(BG_COLOR)

        win_set = set(self._winning_cells)

        for r in range(SIZE):
            for c in range(SIZE):
                x = PAD + c * (CELL + GAP)
                y = PAD + r * (CELL + GAP)
                is_win = (r, c) in win_set
                bg = WIN_COLOR if is_win else EMPTY_COLOR
                pygame.draw.rect(scr, bg, (x, y, CELL, CELL), border_radius=10)
                val = self.grid[r][c]
                if val == 1:
                    lbl = self._font.render("X", True, X_COLOR)
                    scr.blit(lbl, (x + CELL // 2 - lbl.get_width() // 2,
                                   y + CELL // 2 - lbl.get_height() // 2))
                elif val == 2:
                    lbl = self._font.render("O", True, O_COLOR)
                    scr.blit(lbl, (x + CELL // 2 - lbl.get_width() // 2,
                                   y + CELL // 2 - lbl.get_height() // 2))
                # Row/col hint
                coord = self._small.render(f"{r},{c}", True, GRID_COLOR)
                scr.blit(coord, (x + 4, y + 4))

        # Info bar
        board_bottom = PAD + SIZE * (CELL + GAP)
        pygame.draw.rect(scr, PANEL_BG, (0, board_bottom, scr.get_width(), INFO_H))
        if self._winner:
            msg   = f"Winner: {self._winner}!"
            color = X_COLOR if self._winner == "player_X" else O_COLOR
        elif self._done:
            msg, color = "Draw!", FONT_COLOR
        else:
            msg   = f"Turn: {self._turn}"
            color = X_COLOR if self._turn == "player_X" else O_COLOR
        lbl = self._small.render(msg, True, color)
        scr.blit(lbl, (scr.get_width() // 2 - lbl.get_width() // 2, board_bottom + 20))
        step_lbl = self._small.render(f"Step {self._step} | need 4-in-a-row", True, FONT_COLOR)
        scr.blit(step_lbl, (scr.get_width() // 2 - step_lbl.get_width() // 2, board_bottom + 50))

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
    def _find_win(self, row: int, col: int, token: int) -> List[Tuple[int, int]]:
        """Return winning cells if token at (row,col) completes a 4-in-a-row."""
        for dr, dc in _DIRS:
            cells = [(row, col)]
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < SIZE and 0 <= c < SIZE and self.grid[r][c] == token:
                    cells.append((r, c))
                    r += dr * sign
                    c += dc * sign
            if len(cells) >= WIN_LEN:
                return cells
        return []

    def _opponent(self, agent_id: AgentID) -> AgentID:
        return "player_O" if agent_id == "player_X" else "player_X"

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {
            "grid": [row[:] for row in self.grid],
            "active_player": self._turn,
            "step": self._step,
            "winner": self._winner,
            "done": self._done,
        }
        return {"player_X": dict(snap), "player_O": dict(snap)}

    def _dones(self) -> Dict[AgentID, Done]:
        return {"player_X": self._done, "player_O": self._done, "__all__": self._done}
