"""
Othello / Reversi — 8×8 grid, flipping game (1v1).

Placing a piece flanks one or more opponent pieces in any of 8 directions;
all flanked pieces flip to your colour. If you have no legal move you must
pass. If both players have no legal move, the game ends.

Action format : "row col"  (0-indexed, e.g., "3 4")
               or "pass"   (only when no placement is possible)
Turn order    : player_black → player_white → …  (black plays first)
Teams         : {"player_black": ["player_black"], "player_white": ["player_white"]}
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

SIZE = 8
BLACK, WHITE, EMPTY = 1, 2, 0
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Pygame
CELL   = 72
GAP    = 2
PAD    = 16
INFO_H = 90
BG_COLOR     = (15,  15,  25)
BOARD_BG     = (20,  110,  50)
LINE_COLOR   = (10,   80,  30)
BLACK_COLOR  = (20,   20,  20)
WHITE_COLOR  = (240, 240, 240)
HINT_COLOR   = (180, 220,  80)
FONT_COLOR   = (220, 220, 220)


class OthelloGame(BaseGame):
    """
    Standard Othello / Reversi on an 8×8 board.

    grid[row][col]: 0=empty, 1=black, 2=white
    """

    def __init__(self):
        self.grid: List[List[int]] = []
        self._turn: AgentID = "player_black"
        self._done: bool = False
        self._winner: Optional[AgentID] = None
        self._consecutive_passes: int = 0
        self._step: int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"player_black": ["player_black"], "player_white": ["player_white"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self.grid = [[EMPTY] * SIZE for _ in range(SIZE)]
        mid = SIZE // 2
        self.grid[mid - 1][mid - 1] = WHITE
        self.grid[mid - 1][mid]     = BLACK
        self.grid[mid][mid - 1]     = BLACK
        self.grid[mid][mid]         = WHITE
        self._turn = "player_black"
        self._done = False
        self._winner = None
        self._consecutive_passes = 0
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
            infos[active] = {"error": f"Illegal move '{action}'", "legal": legal}
            return self._obs(), rewards, self._dones(), infos

        token = BLACK if active == "player_black" else WHITE
        flipped = 0

        if action == "pass":
            self._consecutive_passes += 1
            infos[active] = {"passed": True}
        else:
            row, col = map(int, action.split())
            flipped = self._place(row, col, token)
            self._consecutive_passes = 0
            infos[active] = {"placed_at": (row, col), "flipped": flipped}

        self._step += 1
        self._turn = self._opponent(active)

        # Game ends when both players pass consecutively
        if self._consecutive_passes >= 2:
            self._finish(rewards)
            return self._obs(), rewards, self._dones(), infos

        # Skip turn if next player has no moves
        if not self._legal_placements(token=WHITE if active == "player_black" else BLACK):
            self._consecutive_passes += 1
            self._turn = active  # revert turn to current player (who still has moves)
            if self._consecutive_passes >= 2:
                self._finish(rewards)

        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        token = BLACK if agent_id == "player_black" else WHITE
        b_count = sum(self.grid[r][c] == BLACK for r in range(SIZE) for c in range(SIZE))
        w_count = sum(self.grid[r][c] == WHITE for r in range(SIZE) for c in range(SIZE))
        legal   = self.get_legal_moves(agent_id)
        state = {
            "agent_id": agent_id,
            "your_token": token,
            "token_legend": {"0": "empty", "1": "black", "2": "white"},
            "is_your_turn": self._turn == agent_id,
            "active_player": self._turn,
            "step": self._step,
            "grid": self.grid,
            "black_count": b_count,
            "white_count": w_count,
            "legal_moves": legal,
            "note": "Use 'pass' only if legal_moves == ['pass']",
            "winner": self._winner,
            "game_over": self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or self._turn != agent_id:
            return []
        token = BLACK if agent_id == "player_black" else WHITE
        placements = self._legal_placements(token)
        if not placements:
            return ["pass"]
        return [f"{r} {c}" for r, c in sorted(placements)]

    def get_game_rules(self) -> str:
        return """
=== OTHELLO / REVERSI — Game Rules ===

OBJECTIVE
---------
Have the most pieces of your colour when the board is full or both
players are forced to pass consecutively.

BOARD
-----
8×8 grid. Initial centre:
  (3,3)=White  (3,4)=Black
  (4,3)=Black  (4,4)=White
Row 0 = top, Col 0 = left (0-indexed).

TOKENS
------
  player_black = 1   plays first
  player_white = 2

PLACEMENT RULES
---------------
You must place your token on an empty cell such that at least one
straight line (horizontal, vertical, or diagonal) has your token at
both ends with only opponent tokens in between.
All opponent tokens on every such flanked line are immediately FLIPPED
to your colour.

PASSING
-------
If you have no legal placement, you MUST pass ("pass").
If BOTH players pass consecutively, the game ends immediately.

GAME END
--------
Board full OR both players pass → player with more tokens wins.
Equal counts = draw.

ACTION FORMAT
-------------
  "row col"   — two space-separated integers (0-indexed), e.g., "2 4"
  "pass"      — only valid when you have no legal placement.

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
            board_px = SIZE * (CELL + GAP) + GAP
            w = PAD * 2 + board_px
            h = PAD * 2 + board_px + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · Othello")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        scr.fill(BG_COLOR)

        board_px = SIZE * (CELL + GAP) + GAP
        pygame.draw.rect(scr, BOARD_BG, (PAD, PAD, board_px, board_px), border_radius=8)

        token = BLACK if self._turn == "player_black" else WHITE
        hints: Set[Tuple[int, int]] = set()
        if not self._done:
            hints = self._legal_placements(token)

        for r in range(SIZE):
            for c in range(SIZE):
                x = PAD + GAP + c * (CELL + GAP)
                y = PAD + GAP + r * (CELL + GAP)
                pygame.draw.rect(scr, LINE_COLOR, (x - GAP, y - GAP, CELL + GAP, CELL + GAP))
                pygame.draw.rect(scr, BOARD_BG, (x, y, CELL, CELL))
                val = self.grid[r][c]
                cx, cy = x + CELL // 2, y + CELL // 2
                if val == BLACK:
                    pygame.draw.circle(scr, BLACK_COLOR, (cx, cy), CELL // 2 - 4)
                elif val == WHITE:
                    pygame.draw.circle(scr, WHITE_COLOR, (cx, cy), CELL // 2 - 4)
                elif (r, c) in hints:
                    pygame.draw.circle(scr, HINT_COLOR, (cx, cy), 8)

        # Info bar
        board_bottom = PAD * 2 + board_px
        pygame.draw.rect(scr, (25, 25, 35), (0, board_bottom, scr.get_width(), INFO_H))
        b = sum(self.grid[r][c] == BLACK for r in range(SIZE) for c in range(SIZE))
        w_cnt = sum(self.grid[r][c] == WHITE for r in range(SIZE) for c in range(SIZE))
        if self._winner:
            msg   = f"Winner: {self._winner}! (B:{b} W:{w_cnt})"
            color = BLACK_COLOR if self._winner == "player_black" else WHITE_COLOR
        elif self._done:
            msg, color = f"Draw! (B:{b} W:{w_cnt})", FONT_COLOR
        else:
            msg   = f"Turn: {self._turn}  | B:{b}  W:{w_cnt}"
            color = (180, 180, 180) if self._turn == "player_black" else WHITE_COLOR
        lbl = self._small.render(msg, True, color)
        scr.blit(lbl, (scr.get_width() // 2 - lbl.get_width() // 2, board_bottom + 25))

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
    def _flanked(self, row: int, col: int, token: int) -> List[Tuple[int, int]]:
        """Return all cells that would be flipped by placing *token* at (row, col)."""
        opp = WHITE if token == BLACK else BLACK
        to_flip = []
        for dr, dc in _DIRS:
            line = []
            r, c = row + dr, col + dc
            while 0 <= r < SIZE and 0 <= c < SIZE and self.grid[r][c] == opp:
                line.append((r, c))
                r += dr; c += dc
            if line and 0 <= r < SIZE and 0 <= c < SIZE and self.grid[r][c] == token:
                to_flip.extend(line)
        return to_flip

    def _legal_placements(self, token: int) -> Set[Tuple[int, int]]:
        result = set()
        for r in range(SIZE):
            for c in range(SIZE):
                if self.grid[r][c] == EMPTY and self._flanked(r, c, token):
                    result.add((r, c))
        return result

    def _place(self, row: int, col: int, token: int) -> int:
        flip_cells = self._flanked(row, col, token)
        self.grid[row][col] = token
        for r, c in flip_cells:
            self.grid[r][c] = token
        return len(flip_cells)

    def _finish(self, rewards: Dict[AgentID, Reward]) -> None:
        self._done = True
        b = sum(self.grid[r][c] == BLACK for r in range(SIZE) for c in range(SIZE))
        w = sum(self.grid[r][c] == WHITE for r in range(SIZE) for c in range(SIZE))
        if b > w:
            self._winner = "player_black"
            rewards["player_black"] = 1.0; rewards["player_white"] = -1.0
        elif w > b:
            self._winner = "player_white"
            rewards["player_white"] = 1.0; rewards["player_black"] = -1.0

    def _opponent(self, agent_id: AgentID) -> AgentID:
        return "player_white" if agent_id == "player_black" else "player_black"

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {
            "grid": [row[:] for row in self.grid],
            "active_player": self._turn,
            "step": self._step,
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
