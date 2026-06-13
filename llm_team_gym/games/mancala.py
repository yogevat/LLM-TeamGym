"""
Mancala (Kalah variant) — classic 6-pit, 4-seed mathematical board game (1v1).

Board layout (14 positions in circular sowing order):
  Positions 0–5   : player_1's pits  (left → right from P1's view)
  Position  6     : player_1's store (mancala)
  Positions 7–12  : player_2's pits  (left → right from P2's view)
  Position  13    : player_2's store (mancala)

Circular sequence for sowing: 0→1→…→5→6→7→…→12→13→0→…
  player_1 skips position 13 (P2's store)
  player_2 skips position  6 (P1's store)

Opposite pit formula: opposite(i) = 12 − i  (valid for pits 0–5 and 7–12)

Action format : pit index 0–5 from the current player's own side.
Teams         : {"player_1": ["player_1"], "player_2": ["player_2"]}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

PITS_PER_SIDE = 6
SEEDS_PER_PIT = 4
P1_STORE = 6
P2_STORE = 13

# Pygame
PIT_R  = 38   # pit circle radius
PIT_D  = PIT_R * 2
GAP    = 18
STORE_W = 70
STORE_H = PIT_D * 2 + GAP
PAD    = 30
INFO_H = 90
BG_COLOR   = (10, 12, 20)
BOARD_COLOR = (100, 65, 30)
PIT_EMPTY   = (55, 40, 20)
PIT_ACTIVE  = (80, 58, 28)
P1_COLOR    = (0, 212, 190)
P2_COLOR    = (255, 80, 120)
SEED_COLOR  = (255, 215, 60)
STORE_COLOR = (35, 25, 12)
FONT_COLOR  = (238, 242, 255)
PANEL_BG    = (18, 21, 34)
PANEL_BDR   = (42, 48, 72)
TEXT_SEC    = (130, 140, 175)
WARNING_CLR = (255, 215, 60)


class MancalaGame(BaseGame):
    """
    Kalah Mancala: 6 pits per side, 4 seeds per pit.

    Extra-turn rule : landing in own store → play again.
    Capture rule    : landing in empty own pit while opposite pit has seeds
                      → capture both piles into own store.
    End condition   : one side is completely empty → other side's seeds
                      swept into that player's store; higher store wins.
    """

    def __init__(self, seeds_per_pit: int = SEEDS_PER_PIT):
        self.seeds_per_pit = seeds_per_pit
        self.board: List[int] = []
        self._turn: AgentID = "player_1"
        self._done: bool = False
        self._winner: Optional[AgentID] = None
        self._step: int = 0
        self._extra_turn: bool = False

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"player_1": ["player_1"], "player_2": ["player_2"]}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        # 14 positions: 0-5 P1 pits, 6 P1 store, 7-12 P2 pits, 13 P2 store
        self.board = [self.seeds_per_pit] * PITS_PER_SIDE + [0] + \
                     [self.seeds_per_pit] * PITS_PER_SIDE + [0]
        self._turn = "player_1"
        self._done = False
        self._winner = None
        self._step = 0
        self._extra_turn = False
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {"player_1": 0.0, "player_2": 0.0}
        infos:   Dict[AgentID, Info]   = {"player_1": {}, "player_2": {}}

        active = self._turn
        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = int(actions_dict[active])
        legal  = self.get_legal_moves(active)

        if action not in legal:
            infos[active] = {"error": f"Illegal pit {action}", "legal": legal}
            return self._obs(), rewards, self._dones(), infos

        internal_pit = action if active == "player_1" else 7 + action
        self._extra_turn, captured = self._sow(internal_pit, active)
        self._step += 1

        info: Dict[str, Any] = {"sowed_from": action, "extra_turn": self._extra_turn}
        if captured > 0:
            info["captured"] = captured
        infos[active] = info

        # Check if game ends
        if self._check_end():
            self._sweep_remaining()
            p1 = self.board[P1_STORE]
            p2 = self.board[P2_STORE]
            self._done = True
            if p1 > p2:
                self._winner = "player_1"
                rewards["player_1"] = 1.0; rewards["player_2"] = -1.0
            elif p2 > p1:
                self._winner = "player_2"
                rewards["player_2"] = 1.0; rewards["player_1"] = -1.0
            # else draw: rewards stay 0
        elif not self._extra_turn:
            self._turn = self._opponent(active)

        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        if agent_id == "player_1":
            own_pits   = self.board[0:6]
            opp_pits   = self.board[7:13]
            own_store  = self.board[P1_STORE]
            opp_store  = self.board[P2_STORE]
        else:
            own_pits   = self.board[7:13]
            opp_pits   = self.board[0:6]
            own_store  = self.board[P2_STORE]
            opp_store  = self.board[P1_STORE]

        state = {
            "agent_id": agent_id,
            "is_your_turn": self._turn == agent_id,
            "active_player": self._turn,
            "step": self._step,
            "your_pits": {"indices_0_to_5": own_pits, "desc": "Your 6 pits from left to right"},
            "opponent_pits": {"indices_0_to_5": opp_pits},
            "your_store": own_store,
            "opponent_store": opp_store,
            "legal_moves": self.get_legal_moves(agent_id),
            "extra_turn_active": self._extra_turn,
            "winner": self._winner,
            "game_over": self._done,
            "raw_board": self.board,
            "raw_board_legend": (
                "Indices 0-5: P1 pits | 6: P1 store | 7-12: P2 pits | 13: P2 store"
            ),
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or self._turn != agent_id:
            return []
        if agent_id == "player_1":
            return [i for i in range(PITS_PER_SIDE) if self.board[i] > 0]
        else:
            return [i for i in range(PITS_PER_SIDE) if self.board[7 + i] > 0]

    def get_game_rules(self) -> str:
        return f"""
=== MANCALA (KALAH) — Game Rules ===

OBJECTIVE
---------
Collect more seeds in your store (mancala) than your opponent.

BOARD LAYOUT
------------
Each player has 6 pits and 1 store (mancala).
Initial seeds per pit: {self.seeds_per_pit}

  P2 store | P2 pit 5 ... P2 pit 0 | P1 store
           | P1 pit 0 ... P1 pit 5 |

Internal indexing: P1 pits→0–5, P1 store→6, P2 pits→7–12, P2 store→13

SOWING
------
Pick a non-empty pit on your side (index 0–5 from your perspective).
Pick up ALL seeds and sow them counter-clockwise, one per pit:
  P1 sows: 0→1→2→3→4→5→store(6)→7→8→…→12→0→… (skips P2 store)
  P2 sows: 7→8→9→10→11→12→store(13)→0→1→…→5→7→… (skips P1 store)

SPECIAL RULES
-------------
Extra turn : If your last seed lands in YOUR store, you take another turn.
Capture    : If your last seed lands in an EMPTY pit on YOUR side AND the
             directly opposite pit has seeds, capture both piles into your store.
             (Opposite of pit i is pit (12 − i) in the internal numbering.)

GAME END
--------
When all pits on either side are empty, the other player's remaining seeds
go into their store. The player with more seeds in their store wins.

ACTION FORMAT
-------------
  A single integer 0–5: your pit index (0 = leftmost, 5 = rightmost from your view).
  Example: 2 → sow from your third pit.

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
            w = PAD * 2 + STORE_W * 2 + GAP * 2 + PITS_PER_SIDE * (PIT_D + GAP) - GAP
            h = PAD * 2 + STORE_H + INFO_H
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · Mancala")
            self._font  = pygame.font.SysFont("monospace", 22, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        w   = scr.get_width()
        scr.fill(BG_COLOR)

        # Board background
        board_rect = (PAD, PAD, w - PAD * 2, STORE_H + GAP * 2)
        pygame.draw.rect(scr, BOARD_COLOR, board_rect, border_radius=15)

        pit_area_x = PAD + STORE_W + GAP
        pit_area_w = PITS_PER_SIDE * (PIT_D + GAP) - GAP

        def draw_pit(cx, cy, seeds, active_side):
            color = PIT_ACTIVE if active_side else PIT_EMPTY
            pygame.draw.circle(scr, color, (cx, cy), PIT_R)
            lbl = self._font.render(str(seeds), True, SEED_COLOR)
            scr.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

        def draw_store(x, y, w2, h2, seeds, color):
            pygame.draw.rect(scr, STORE_COLOR, (x, y, w2, h2), border_radius=10)
            pygame.draw.rect(scr, color, (x, y, w2, h2), border_radius=10, width=3)
            lbl = self._font.render(str(seeds), True, SEED_COLOR)
            scr.blit(lbl, (x + w2 // 2 - lbl.get_width() // 2,
                           y + h2 // 2 - lbl.get_height() // 2))

        p1_active = self._turn == "player_1"
        p2_active = self._turn == "player_2"

        # P1 pits (bottom row): indices 0–5, left to right
        row1_cy = PAD + GAP + STORE_H - PIT_R  # bottom row centre y
        for i in range(PITS_PER_SIDE):
            cx = pit_area_x + i * (PIT_D + GAP) + PIT_R
            draw_pit(cx, row1_cy, self.board[i], p1_active)
            idx_lbl = self._small.render(str(i), True, (180, 160, 120))
            scr.blit(idx_lbl, (cx - idx_lbl.get_width() // 2, row1_cy + PIT_R + 4))

        # P2 pits (top row): indices 7–12, but displayed right-to-left
        # so that pit 12 is above pit 0 (opposite) and pit 7 is above pit 5
        row2_cy = PAD + GAP + PIT_R
        for i in range(PITS_PER_SIDE):
            # P2 pit shown at column i from left = internal pit 12 - i
            internal = 12 - i
            cx = pit_area_x + i * (PIT_D + GAP) + PIT_R
            p2_local = 12 - internal  # = i → local index displayed as (5 - i) from P2's view
            draw_pit(cx, row2_cy, self.board[internal], p2_active)
            idx_lbl = self._small.render(str(5 - i), True, (180, 160, 120))
            scr.blit(idx_lbl, (cx - idx_lbl.get_width() // 2, row2_cy - PIT_R - 14))

        # Stores
        store_y = PAD + GAP
        draw_store(PAD, store_y, STORE_W, STORE_H, self.board[P2_STORE], P2_COLOR)   # left = P2
        draw_store(w - PAD - STORE_W, store_y, STORE_W, STORE_H, self.board[P1_STORE], P1_COLOR)  # right = P1

        # Labels
        p2_lbl = self._small.render("P2", True, P2_COLOR)
        scr.blit(p2_lbl, (PAD + STORE_W // 2 - p2_lbl.get_width() // 2, PAD - 2))
        p1_lbl = self._small.render("P1", True, P1_COLOR)
        scr.blit(p1_lbl, (w - PAD - STORE_W + STORE_W // 2 - p1_lbl.get_width() // 2, PAD + STORE_H + GAP + 2))

        # Info bar
        info_y = PAD * 2 + STORE_H + GAP * 2
        pygame.draw.rect(scr, PANEL_BG, (0, info_y, w, INFO_H))
        if self._winner:
            msg   = f"Winner: {self._winner}!  (P1:{self.board[P1_STORE]} P2:{self.board[P2_STORE]})"
            color = P1_COLOR if self._winner == "player_1" else P2_COLOR
        elif self._done:
            msg   = f"Draw!  (P1:{self.board[P1_STORE]} P2:{self.board[P2_STORE]})"
            color = FONT_COLOR
        else:
            msg   = f"Turn: {self._turn}" + ("  [EXTRA TURN]" if self._extra_turn else "")
            color = P1_COLOR if self._turn == "player_1" else P2_COLOR
        lbl = self._small.render(msg, True, color)
        scr.blit(lbl, (w // 2 - lbl.get_width() // 2, info_y + 20))

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
    def _sow(self, start_pit: int, player: AgentID) -> tuple[bool, int]:
        """Sow seeds from start_pit. Returns (extra_turn, seeds_captured)."""
        seeds = self.board[start_pit]
        self.board[start_pit] = 0
        skip_store = P2_STORE if player == "player_1" else P1_STORE
        own_store  = P1_STORE  if player == "player_1" else P2_STORE
        own_pits   = set(range(6)) if player == "player_1" else set(range(7, 13))

        pos = (start_pit + 1) % 14
        last_pos = start_pit
        while seeds > 0:
            if pos == skip_store:
                pos = (pos + 1) % 14
                continue
            self.board[pos] += 1
            seeds -= 1
            last_pos = pos
            pos = (pos + 1) % 14

        # Extra turn?
        extra_turn = (last_pos == own_store)

        # Capture?
        captured = 0
        if (last_pos in own_pits
                and self.board[last_pos] == 1
                and not extra_turn):
            opp_pos = 12 - last_pos
            if self.board[opp_pos] > 0:
                captured = self.board[last_pos] + self.board[opp_pos]
                self.board[own_store] += captured
                self.board[last_pos] = 0
                self.board[opp_pos]  = 0

        return extra_turn, captured

    def _check_end(self) -> bool:
        return (all(self.board[i] == 0 for i in range(6)) or
                all(self.board[i] == 0 for i in range(7, 13)))

    def _sweep_remaining(self) -> None:
        self.board[P1_STORE] += sum(self.board[0:6])
        self.board[P2_STORE] += sum(self.board[7:13])
        for i in list(range(6)) + list(range(7, 13)):
            self.board[i] = 0

    def _opponent(self, agent_id: AgentID) -> AgentID:
        return "player_2" if agent_id == "player_1" else "player_1"

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {
            "board": self.board[:],
            "active_player": self._turn,
            "step": self._step,
            "winner": self._winner,
            "done": self._done,
            "extra_turn": self._extra_turn,
        }
        return {"player_1": dict(snap), "player_2": dict(snap)}

    def _dones(self) -> Dict[AgentID, Done]:
        return {"player_1": self._done, "player_2": self._done, "__all__": self._done}
