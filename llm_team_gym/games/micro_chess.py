"""
Micro Chess — Gardner Mini Chess on a 5×5 board (1v1).

This is the standard Gardner (1969) mini-chess variant:
  - 5 ranks (1–5) × 5 files (a–e)
  - All standard chess pieces with their standard moves
  - No castling, no en passant
  - Pawns auto-promote to Queen on reaching the final rank
  - Win: checkmate (opponent has no legal moves while in check)
  - Draw: stalemate (opponent has no legal moves but is NOT in check)

Board coordinates:
  row 0 = rank 1 (white's back rank)   col 0 = file a
  row 4 = rank 5 (black's back rank)   col 4 = file e

Initial layout (Gardner setup):
  Rank 5 (row 4): r n b q k   (black)
  Rank 4 (row 3): p p p p p   (black pawns)
  Rank 3 (row 2): . . . . .
  Rank 2 (row 1): P P P P P   (white pawns)
  Rank 1 (row 0): R N B Q K   (white)

Action format : "a1b2"  — from-square to-square in algebraic notation.
                file (a–e) + rank (1–5), concatenated twice.
Turn order    : player_white → player_black → …
Teams         : {"player_white": ["player_white"], "player_black": ["player_black"]}
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

# Piece types
KING   = "K"
QUEEN  = "Q"
ROOK   = "R"
BISHOP = "B"
KNIGHT = "N"
PAWN   = "P"
WHITE  = "white"
BLACK  = "black"

_SLIDING = {QUEEN, ROOK, BISHOP}

# Pygame
SQ     = 100
PAD    = 20
INFO_H = 110
LIGHT  = (240, 217, 181)
DARK   = (181, 136,  99)
HL_LAST= (100, 200, 100, 160)   # last-move highlight
HL_CHK = (220,  60,  60)        # check highlight
W_COLOR= (250, 250, 250)
B_COLOR= (30,   30,  30)
FONT_COLOR = (220, 220, 220)
BG_COLOR   = (15,  15,  25)

PIECE_UNICODE = {
    (KING,   WHITE): "♔", (QUEEN,  WHITE): "♕",
    (ROOK,   WHITE): "♖", (BISHOP, WHITE): "♗",
    (KNIGHT, WHITE): "♘", (PAWN,   WHITE): "♙",
    (KING,   BLACK): "♚", (QUEEN,  BLACK): "♛",
    (ROOK,   BLACK): "♜", (BISHOP, BLACK): "♝",
    (KNIGHT, BLACK): "♞", (PAWN,   BLACK): "♟",
}


class MicroChessGame(BaseGame):
    """
    Gardner Mini Chess (5×5).

    board[row][col] = None  |  (piece_type: str, color: str)
    """

    def __init__(self):
        self.board: List[List[Optional[Tuple[str, str]]]] = []
        self._turn: AgentID = "player_white"
        self._done: bool = False
        self._winner: Optional[AgentID] = None
        self._draw: bool = False
        self._step: int = 0
        self._last_from: Optional[Tuple[int, int]] = None
        self._last_to:   Optional[Tuple[int, int]] = None
        self._in_check:  bool = False

        self._pygame_init = False
        self._screen = self._font_pieces = self._font_ui = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"player_white": ["player_white"], "player_black": ["player_black"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        # Row 0 = rank 1 (white back rank), row 4 = rank 5 (black back rank)
        back = [ROOK, KNIGHT, BISHOP, QUEEN, KING]
        self.board = [[None] * 5 for _ in range(5)]
        for c, p in enumerate(back):
            self.board[0][c] = (p, WHITE)   # white back rank
            self.board[4][c] = (p, BLACK)   # black back rank
        for c in range(5):
            self.board[1][c] = (PAWN, WHITE)
            self.board[3][c] = (PAWN, BLACK)
        self._turn = "player_white"
        self._done = False
        self._winner = None
        self._draw = False
        self._step = 0
        self._last_from = self._last_to = None
        self._in_check = False
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {"player_white": 0.0, "player_black": 0.0}
        infos:   Dict[AgentID, Info]   = {"player_white": {}, "player_black": {}}

        active = self._turn
        color  = WHITE if active == "player_white" else BLACK
        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip().lower()
        legal  = self.get_legal_moves(active)

        if action not in legal:
            infos[active] = {"error": f"Illegal move '{action}'", "legal_count": len(legal)}
            return self._obs(), rewards, self._dones(), infos

        from_pos = self._alg_to_pos(action[:2])
        to_pos   = self._alg_to_pos(action[2:])
        captured_piece = self.board[to_pos[0]][to_pos[1]]

        self.board = self._apply_move(self.board, from_pos, to_pos)
        self._last_from, self._last_to = from_pos, to_pos
        self._step += 1

        opp_color = BLACK if color == WHITE else WHITE
        opp_agent = "player_black" if active == "player_white" else "player_white"

        # Check state for opponent
        self._in_check = self._is_in_check(self.board, opp_color)
        opp_legal = self._all_legal_for(opp_color)
        infos[active] = {
            "from": action[:2], "to": action[2:],
            "captured": captured_piece[0] if captured_piece else None,
            "check": self._in_check,
        }

        if not opp_legal:
            self._done = True
            if self._in_check:
                # Checkmate
                self._winner = active
                rewards[active] = 1.0
                rewards[opp_agent] = -1.0
            else:
                # Stalemate — draw
                self._draw = True
        else:
            self._turn = opp_agent

        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        color = WHITE if agent_id == "player_white" else BLACK
        grid_display = []
        for r in range(4, -1, -1):   # rank 5 at top
            row_str = []
            for c in range(5):
                p = self.board[r][c]
                if p is None:
                    row_str.append(".")
                else:
                    sym = p[0] if p[1] == WHITE else p[0].lower()
                    row_str.append(sym)
            grid_display.append(" ".join(row_str))
        state = {
            "agent_id": agent_id,
            "your_color": color,
            "is_your_turn": self._turn == agent_id,
            "active_player": self._turn,
            "step": self._step,
            "board_display": {
                "desc": "Ranks 5→1 top→bottom. Uppercase=white, lowercase=black. '.'=empty",
                "files": "  a b c d e",
                "rows": [f"rank{5-i} {row}" for i, row in enumerate(grid_display)],
            },
            "in_check": self._in_check and self._turn == agent_id,
            "legal_moves": self.get_legal_moves(agent_id),
            "last_move": {
                "from": self._pos_to_alg(*self._last_from) if self._last_from else None,
                "to":   self._pos_to_alg(*self._last_to)   if self._last_to   else None,
            },
            "winner": self._winner,
            "draw": self._draw,
            "game_over": self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or self._turn != agent_id:
            return []
        color = WHITE if agent_id == "player_white" else BLACK
        return self._all_legal_for(color)

    def get_game_rules(self) -> str:
        return """
=== MICRO CHESS (GARDNER 5×5 MINI-CHESS) — Game Rules ===

OBJECTIVE
---------
Checkmate your opponent's King: leave it in check with no legal escape.

BOARD
-----
5 files (a–e, left→right) × 5 ranks (1–5, bottom→top).
Uppercase pieces = White, lowercase = Black.

INITIAL SETUP (rank 5 = top)
  rank 5: r n b q k   (black back rank)
  rank 4: p p p p p   (black pawns)
  rank 3: . . . . .   (empty)
  rank 2: P P P P P   (white pawns)
  rank 1: R N B Q K   (white back rank)

PIECE MOVEMENTS (standard chess rules)
---------------------------------------
  K / k  King   : 1 step in any direction
  Q / q  Queen  : any number of steps in any direction (blocked by pieces)
  R / r  Rook   : any number of steps horizontally or vertically
  B / b  Bishop : any number of steps diagonally
  N / n  Knight : L-shape (±1,±2 or ±2,±1), jumps over pieces
  P / p  Pawn   : forward 1 step; 2 steps from starting rank; captures diagonally
                  White pawns move UP (toward rank 5); Black pawns move DOWN.

SPECIAL RULES
-------------
  Promotion : Pawn reaching the opponent's back rank promotes automatically to Queen.
  Check     : You may NOT make a move that leaves your own King in check.
  No castling, no en passant.

WIN / DRAW CONDITIONS
---------------------
  Checkmate : Your move leaves opponent in check with no legal moves → YOU WIN.
  Stalemate : Your move gives opponent no legal moves but they are NOT in check → DRAW.

ACTION FORMAT
-------------
  "from_square to_square" concatenated, e.g., "e2e4" or "b1c3".
  file ∈ {a,b,c,d,e}, rank ∈ {1,2,3,4,5}.

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
            w = PAD * 2 + 5 * SQ
            h = PAD * 2 + 5 * SQ + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · Micro Chess")
            # Try a font with Unicode chess symbols; fall back gracefully
            try:
                self._font_pieces = pygame.font.SysFont("dejavusans", 58, bold=True)
            except Exception:
                self._font_pieces = pygame.font.SysFont("monospace", 40, bold=True)
            self._font_ui = pygame.font.SysFont("monospace", 20, bold=True)
            self._small   = pygame.font.SysFont("monospace", 14)
            self._clock   = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        scr.fill(BG_COLOR)

        last_squares = {self._last_from, self._last_to} - {None}
        color_in_check = None
        if self._in_check:
            color_in_check = WHITE if self._turn == "player_white" else BLACK

        # Draw board (rank 5 at top → row 4 of board drawn first)
        for display_r in range(5):
            board_r = 4 - display_r   # board row 4 = rank 5 = display row 0
            for c in range(5):
                x = PAD + c * SQ
                y = PAD + display_r * SQ
                is_light = (board_r + c) % 2 == 0
                sq_color = LIGHT if is_light else DARK

                if (board_r, c) in last_squares:
                    sq_color = tuple(min(255, v + 40) for v in sq_color)

                pygame.draw.rect(scr, sq_color, (x, y, SQ, SQ))

                piece = self.board[board_r][c]
                if piece:
                    p_type, p_color = piece
                    # Highlight king in check
                    if p_type == KING and p_color == color_in_check:
                        pygame.draw.rect(scr, HL_CHK, (x, y, SQ, SQ))

                    sym = PIECE_UNICODE.get(piece, p_type)
                    fg  = W_COLOR if p_color == WHITE else B_COLOR
                    lbl = self._font_pieces.render(sym, True, fg)
                    # Draw shadow for visibility on same-colour squares
                    shadow_fg = B_COLOR if p_color == WHITE else W_COLOR
                    shd = self._font_pieces.render(sym, True, shadow_fg)
                    ox = x + SQ // 2 - lbl.get_width() // 2
                    oy = y + SQ // 2 - lbl.get_height() // 2
                    scr.blit(shd, (ox + 1, oy + 1))
                    scr.blit(lbl, (ox, oy))

                # File label (bottom of board)
                if display_r == 4:
                    f_lbl = self._small.render("abcde"[c], True, FONT_COLOR)
                    scr.blit(f_lbl, (x + SQ - f_lbl.get_width() - 3, y + SQ - f_lbl.get_height() - 2))
                # Rank label (left edge)
                if c == 0:
                    r_lbl = self._small.render(str(board_r + 1), True, FONT_COLOR)
                    scr.blit(r_lbl, (x + 3, y + 3))

        # Info bar
        info_y = PAD * 2 + 5 * SQ
        pygame.draw.rect(scr, (25, 25, 35), (0, info_y, scr.get_width(), INFO_H))
        if self._winner:
            msg   = f"Checkmate! Winner: {self._winner}"
            color = (240, 200, 50)
        elif self._draw:
            msg, color = "Stalemate — Draw!", FONT_COLOR
        else:
            chk_str = "  [CHECK!]" if self._in_check else ""
            msg   = f"Turn: {self._turn}{chk_str}"
            color = (240, 240, 240) if self._turn == "player_white" else (160, 160, 180)
        lbl = self._font_ui.render(msg, True, color)
        scr.blit(lbl, (scr.get_width() // 2 - lbl.get_width() // 2, info_y + 20))
        step_lbl = self._small.render(f"Step {self._step}  |  Micro Chess (5×5 Gardner)", True, FONT_COLOR)
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
    # Move generation & validation
    # ------------------------------------------------------------------

    def _pseudo_legal(
        self, board: List[List[Optional[Tuple[str, str]]]], r: int, c: int
    ) -> List[Tuple[int, int]]:
        """All pseudo-legal destinations for the piece at board[r][c] (ignores check)."""
        cell = board[r][c]
        if cell is None:
            return []
        piece, color = cell
        opp = BLACK if color == WHITE else WHITE
        moves: List[Tuple[int, int]] = []

        def in_bounds(rr: int, cc: int) -> bool:
            return 0 <= rr < 5 and 0 <= cc < 5

        def slide(deltas):
            for dr, dc in deltas:
                rr, cc = r + dr, c + dc
                while in_bounds(rr, cc):
                    p = board[rr][cc]
                    if p is None:
                        moves.append((rr, cc))
                    elif p[1] == opp:
                        moves.append((rr, cc)); break
                    else:
                        break
                    rr += dr; cc += dc

        def step_once(deltas):
            for dr, dc in deltas:
                rr, cc = r + dr, c + dc
                if in_bounds(rr, cc):
                    p = board[rr][cc]
                    if p is None or p[1] == opp:
                        moves.append((rr, cc))

        if piece == ROOK:
            slide([(0,1),(0,-1),(1,0),(-1,0)])
        elif piece == BISHOP:
            slide([(1,1),(1,-1),(-1,1),(-1,-1)])
        elif piece == QUEEN:
            slide([(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)])
        elif piece == KNIGHT:
            step_once([(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)])
        elif piece == KING:
            step_once([(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)])
        elif piece == PAWN:
            dr = 1 if color == WHITE else -1
            start_r = 1 if color == WHITE else 3
            # Forward
            rr = r + dr
            if in_bounds(rr, c) and board[rr][c] is None:
                moves.append((rr, c))
                # Two-square advance from starting rank
                rr2 = r + 2 * dr
                if r == start_r and in_bounds(rr2, c) and board[rr2][c] is None:
                    moves.append((rr2, c))
            # Diagonal captures
            for dc in (-1, 1):
                cc = c + dc
                rr = r + dr
                if in_bounds(rr, cc) and board[rr][cc] is not None and board[rr][cc][1] == opp:
                    moves.append((rr, cc))

        return moves

    def _apply_move(
        self,
        board: List[List[Optional[Tuple[str, str]]]],
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
    ) -> List[List[Optional[Tuple[str, str]]]]:
        """Return a new 5×5 board after applying the move (with promotion)."""
        new_board = [row[:] for row in board]
        piece = new_board[from_pos[0]][from_pos[1]]
        new_board[to_pos[0]][to_pos[1]] = piece
        new_board[from_pos[0]][from_pos[1]] = None
        # Pawn promotion
        if piece and piece[0] == PAWN:
            if piece[1] == WHITE and to_pos[0] == 4:
                new_board[to_pos[0]][to_pos[1]] = (QUEEN, WHITE)
            elif piece[1] == BLACK and to_pos[0] == 0:
                new_board[to_pos[0]][to_pos[1]] = (QUEEN, BLACK)
        return new_board

    def _king_pos(
        self, board: List[List[Optional[Tuple[str, str]]]], color: str
    ) -> Optional[Tuple[int, int]]:
        for r in range(5):
            for c in range(5):
                p = board[r][c]
                if p and p[0] == KING and p[1] == color:
                    return (r, c)
        return None

    def _is_in_check(
        self, board: List[List[Optional[Tuple[str, str]]]], color: str
    ) -> bool:
        king = self._king_pos(board, color)
        if king is None:
            return True
        opp = BLACK if color == WHITE else WHITE
        for r in range(5):
            for c in range(5):
                p = board[r][c]
                if p and p[1] == opp:
                    if king in self._pseudo_legal(board, r, c):
                        return True
        return False

    def _all_legal_for(self, color: str) -> List[str]:
        moves = []
        for r in range(5):
            for c in range(5):
                p = self.board[r][c]
                if p and p[1] == color:
                    for to_r, to_c in self._pseudo_legal(self.board, r, c):
                        new_board = self._apply_move(self.board, (r, c), (to_r, to_c))
                        if not self._is_in_check(new_board, color):
                            moves.append(
                                self._pos_to_alg(r, c) + self._pos_to_alg(to_r, to_c)
                            )
        return moves

    # ------------------------------------------------------------------
    # Algebraic notation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alg_to_pos(sq: str) -> Tuple[int, int]:
        col = ord(sq[0]) - ord("a")       # 'a'→0 … 'e'→4
        row = int(sq[1]) - 1               # '1'→0 … '5'→4
        return (row, col)

    @staticmethod
    def _pos_to_alg(row: int, col: int) -> str:
        return "abcde"[col] + str(row + 1)

    # ------------------------------------------------------------------
    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {
            "board": [[str(cell) if cell else None for cell in row] for row in self.board],
            "active_player": self._turn,
            "step": self._step,
            "in_check": self._in_check,
            "winner": self._winner,
            "draw": self._draw,
            "done": self._done,
        }
        return {"player_white": dict(snap), "player_black": dict(snap)}

    def _dones(self) -> Dict[AgentID, Done]:
        return {
            "player_white": self._done,
            "player_black": self._done,
            "__all__": self._done,
        }
