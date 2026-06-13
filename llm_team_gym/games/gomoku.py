"""
Gomoku (Five-in-a-Row) — 15×15 grid (1v1).

Players alternate placing stones. The first to align exactly 5 (or more)
in an unbroken row — horizontally, vertically, or diagonally — wins.

This implementation uses standard free Gomoku rules (no restriction on
overlines; 5 or more consecutive stones count as a win).

Action format : "row col"  (0-indexed, e.g., "7 7" for the centre)
Turn order    : player_black → player_white → …
Teams         : {"player_black": ["player_black"], "player_white": ["player_white"]}
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

SIZE    = 15
WIN_LEN = 5
BLACK, WHITE, EMPTY = 1, 2, 0
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Pygame
CELL   = 40
PAD    = 28
INFO_H = 90
BG_COLOR    = (10, 12, 20)
BOARD_COLOR = (188, 148, 72)
LINE_COLOR  = (115, 85, 38)
BLACK_COLOR = (15, 15, 15)
WHITE_COLOR = (245, 245, 245)
LAST_MARK   = (255, 80, 120)
HINT_COLOR  = (55, 225, 130)
FONT_COLOR  = (238, 242, 255)
STAR_COLOR  = (75, 52, 18)
PANEL_BG    = (18, 21, 34)
PANEL_BDR   = (42, 48, 72)
TEXT_SEC    = (130, 140, 175)
WARNING_CLR = (255, 215, 60)


class GomokuGame(BaseGame):
    """
    Free Gomoku on a 15×15 board.

    grid[row][col]: 0=empty, 1=black, 2=white
    """

    def __init__(self, size: int = SIZE, win_len: int = WIN_LEN):
        self.size    = size
        self.win_len = win_len
        self.grid: List[List[int]] = []
        self._turn: AgentID = "player_black"
        self._done: bool = False
        self._winner: Optional[AgentID] = None
        self._last: Optional[Tuple[int, int]] = None
        self._winning_cells: List[Tuple[int, int]] = []
        self._step: int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"player_black": ["player_black"], "player_white": ["player_white"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self.grid = [[EMPTY] * self.size for _ in range(self.size)]
        self._turn = "player_black"
        self._done = False
        self._winner = None
        self._last = None
        self._winning_cells = []
        self._step = 0
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {"player_black": 0.0, "player_white": 0.0}
        infos:   Dict[AgentID, Info]   = {"player_black": {}, "player_white": {}}

        active = self._turn
        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip()
        legal  = self.get_legal_moves(active)

        if action not in legal:
            infos[active] = {"error": f"Illegal move '{action}'", "legal_count": len(legal)}
            return self._obs(), rewards, self._dones(), infos

        row, col = map(int, action.split())
        token = BLACK if active == "player_black" else WHITE
        self.grid[row][col] = token
        self._last = (row, col)
        self._step += 1

        win_cells = self._find_win(row, col, token)
        if win_cells:
            self._done = True
            self._winner = active
            self._winning_cells = win_cells
            rewards[active] = 1.0
            rewards[self._opponent(active)] = -1.0
        elif all(self.grid[r][c] != EMPTY for r in range(self.size) for c in range(self.size)):
            self._done = True   # board full = draw
        else:
            self._turn = self._opponent(active)

        infos[active] = {"placed_at": (row, col)}
        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        token = BLACK if agent_id == "player_black" else WHITE
        empty_count = sum(self.grid[r][c] == EMPTY for r in range(self.size) for c in range(self.size))
        legal = self.get_legal_moves(agent_id)
        # Return compact grid + summary; full 15×15 grid included for completeness
        state = {
            "agent_id": agent_id,
            "your_token": token,
            "token_legend": {"0": "empty", "1": "black", "2": "white"},
            "is_your_turn": self._turn == agent_id,
            "active_player": self._turn,
            "step": self._step,
            "board_size": self.size,
            "win_length": self.win_len,
            "empty_cells_remaining": empty_count,
            "last_move": list(self._last) if self._last else None,
            "grid": self.grid,
            "legal_moves_count": len(legal),
            "legal_moves_sample": legal[:20],
            "note": (
                "All empty cells are legal. Choose a strategic position to build "
                f"{self.win_len} in a row. Only first 20 legal moves shown above."
            ),
            "winner": self._winner,
            "game_over": self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or self._turn != agent_id:
            return []
        return [
            f"{r} {c}"
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r][c] == EMPTY
        ]

    def get_game_rules(self) -> str:
        return f"""
=== GOMOKU (FIVE-IN-A-ROW) — Game Rules ===

OBJECTIVE
---------
Be the first to place {self.win_len} consecutive stones of your colour
in an unbroken line — horizontally, vertically, or diagonally.

BOARD
-----
{self.size}×{self.size} intersection grid (like a Go board).
Row 0 = top, Col 0 = left (0-indexed). Centre = ({self.size//2}, {self.size//2}).

STONES
------
  player_black = 1   plays first (traditional advantage)
  player_white = 2

TURN STRUCTURE
--------------
Players alternate placing one stone per turn on any empty intersection.
Once placed, stones cannot move or be removed.

WIN CONDITION
-------------
{self.win_len} or more consecutive stones of your colour in any straight line.
If the board fills completely with no winner, the game is a draw.

STRATEGY NOTES
--------------
- Control the centre; build threats in multiple directions simultaneously.
- Watch for opponent's open-ended 3-in-a-row threats.
- Double-three and double-four setups create unavoidable forks.

ACTION FORMAT
-------------
  "row col"  — two space-separated integers (0-indexed).
  Example: "7 7" places a stone at the board centre.

Legal moves: all empty cells. Choose strategically.
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
            board_px = PAD * 2 + (self.size - 1) * CELL
            w = board_px
            h = board_px + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · Gomoku")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 13)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        board_px = PAD * 2 + (self.size - 1) * CELL
        scr.fill(BOARD_COLOR)

        # Draw grid lines
        for i in range(self.size):
            x = PAD + i * CELL
            y = PAD + i * CELL
            pygame.draw.line(scr, LINE_COLOR, (PAD, y), (PAD + (self.size-1)*CELL, y), 1)
            pygame.draw.line(scr, LINE_COLOR, (x, PAD), (x, PAD + (self.size-1)*CELL), 1)

        # Star points (standard 15x15 positions: {3,7,11} × {3,7,11})
        star_pts = [3, 7, 11] if self.size == 15 else []
        for sr in star_pts:
            for sc in star_pts:
                sx = PAD + sc * CELL
                sy = PAD + sr * CELL
                pygame.draw.circle(scr, STAR_COLOR, (sx, sy), 4)

        win_set = set(self._winning_cells)

        # Draw stones
        for r in range(self.size):
            for c in range(self.size):
                val = self.grid[r][c]
                if val == EMPTY:
                    continue
                cx = PAD + c * CELL
                cy = PAD + r * CELL
                color = BLACK_COLOR if val == BLACK else WHITE_COLOR
                pygame.draw.circle(scr, color, (cx, cy), CELL // 2 - 2)
                if (r, c) in win_set:
                    pygame.draw.circle(scr, (220, 60, 60), (cx, cy), CELL // 2 - 2, 3)
                if (r, c) == self._last and self._last:
                    pygame.draw.circle(scr, LAST_MARK, (cx, cy), 5)

        # Info bar
        info_y = board_px
        pygame.draw.rect(scr, PANEL_BG, (0, info_y, scr.get_width(), INFO_H))
        if self._winner:
            msg   = f"Winner: {self._winner}! (5-in-a-row)"
            color = (180, 180, 180) if self._winner == "player_black" else WHITE_COLOR
        elif self._done:
            msg, color = "Draw! Board full.", FONT_COLOR
        else:
            msg   = f"Turn: {self._turn}   Step: {self._step}"
            color = (180, 180, 180) if self._turn == "player_black" else WHITE_COLOR
        lbl = self._small.render(msg, True, color)
        scr.blit(lbl, (scr.get_width() // 2 - lbl.get_width() // 2, info_y + 25))

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
        for dr, dc in _DIRS:
            cells = [(row, col)]
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while (0 <= r < self.size and 0 <= c < self.size
                       and self.grid[r][c] == token):
                    cells.append((r, c))
                    r += dr * sign; c += dc * sign
            if len(cells) >= self.win_len:
                return cells
        return []

    def _opponent(self, agent_id: AgentID) -> AgentID:
        return "player_white" if agent_id == "player_black" else "player_black"

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {
            "grid": [row[:] for row in self.grid],
            "active_player": self._turn,
            "step": self._step,
            "last_move": self._last,
            "winner": self._winner,
            "done": self._done,
        }
        return {"player_black": dict(snap), "player_white": dict(snap)}

    def _dones(self) -> Dict[AgentID, Done]:
        return {
            "player_black": self._done,
            "player_white": self._done,
            "__all__": self._done,
        }
