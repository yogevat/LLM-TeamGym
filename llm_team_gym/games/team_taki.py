"""
Team Taki (2v2) — Uno-style team card game.

4 players in 2 teams: (p0, p2) vs (p1, p3) — partners sit across.
First TEAM to have BOTH members empty their hands wins.

CARD TYPES (per color: Red, Blue, Yellow, Green)
  • Number cards : 1–9
  • STOP         : next player loses their turn
  • +2           : next player draws 2 cards and loses turn
  • TAKI         : play multiple same-color cards in one turn

WILD CARDS (black)
  • COLOR_CHANGE  : change active color
  • +4            : next player draws 4 and loses turn; play any color next

A card is PLAYABLE if it matches the current color OR current rank/type.
Wild cards are always playable.

Actions:
  "play <card>"             — play one matching card
  "taki <c1> <c2> ..."      — play a Taki + same-color sequence (all same color)
  "draw"                    — draw 1 card from deck (if no playable card)

Card format: <color><type>   e.g. R3, B7, GSTOP, Y+2, RTAKI, CC (Color Change), +4

Teams  : team_A (p0, p2)  vs  team_B (p1, p3)
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

COLORS  = ('R', 'B', 'Y', 'G')
NUMBERS = tuple(str(i) for i in range(1, 10))
SPECIALS = ('STOP', '+2', 'TAKI')
WILDS   = ('CC', '+4')

PLAYERS = ('p0', 'p1', 'p2', 'p3')
TEAM_A  = ('p0', 'p2')
TEAM_B  = ('p1', 'p3')

HAND_SIZE = 8

BG_COLOR  = (15, 15, 30)
TEXT_CLR  = (230, 230, 230)
CMAP      = {'R': (200, 40, 40), 'B': (40, 80, 200), 'Y': (200, 180, 30), 'G': (40, 170, 60)}
P_COLORS  = [(70, 130, 220), (220, 70, 70), (60, 190, 80), (230, 200, 40)]


def _build_deck() -> List[str]:
    deck: List[str] = []
    for c in COLORS:
        for n in NUMBERS:
            deck.extend([f"{c}{n}"] * 2)
        for sp in SPECIALS:
            deck.append(f"{c}{sp}")
    for w in WILDS:
        deck.extend([w] * 4)
    return deck


def _card_color(card: str) -> Optional[str]:
    if card in WILDS:
        return None
    return card[0] if card[0] in COLORS else None


def _card_type(card: str) -> str:
    if card in WILDS:
        return card
    return card[1:]


def _is_playable(card: str, top: str, active_color: str) -> bool:
    if card in WILDS:
        return True
    cc = _card_color(card)
    ct = _card_type(card)
    top_color = active_color
    top_type  = _card_type(top) if top not in WILDS else ""
    return cc == top_color or ct == top_type


class TeamTakiGame(BaseGame):
    """Team Taki — 4-player 2v2 Uno variant."""

    def __init__(self, seed: Optional[int] = None, max_steps: int = 800):
        self._seed      = seed
        self._max_steps = max_steps
        self._rng       = random.Random(seed)

        self.hands:      Dict[AgentID, List[str]] = {}
        self._deck:      List[str] = []
        self._discard:   List[str] = []
        self._color:     str = 'R'
        self._turn_idx:  int = 0
        self._direction: int = 1
        self._skip_next: bool = False
        self._draw_pend: int = 0
        self._drawn_this_turn: bool = False
        self._taki_active: bool = False
        self._taki_pid:    Optional[AgentID] = None
        self._taki_color:  Optional[str] = None
        self._done:      bool = False
        self._winner:    Optional[str] = None
        self._step:      int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[str, List[AgentID]]:
        return {"team_A": list(TEAM_A), "team_B": list(TEAM_B)}

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng       = random.Random(self._seed)
        self._deck      = _build_deck()
        self._rng.shuffle(self._deck)
        self.hands      = {p: [] for p in PLAYERS}
        for _ in range(HAND_SIZE):
            for p in PLAYERS:
                self.hands[p].append(self._draw_card())
        start = self._draw_card()
        while start in WILDS:
            self._deck.insert(0, start)
            self._rng.shuffle(self._deck)
            start = self._draw_card()
        self._discard   = [start]
        self._color     = _card_color(start) or 'R'
        self._turn_idx  = 0
        self._direction = 1
        self._skip_next = False
        self._draw_pend = 0
        self._drawn_this_turn = False
        self._taki_active = False
        self._taki_pid    = None
        self._taki_color  = None
        self._done      = False
        self._winner    = None
        self._step      = 0
        return self._obs()

    def _draw_card(self) -> str:
        if not self._deck:
            top = self._discard[-1]
            self._deck = self._discard[:-1]
            self._rng.shuffle(self._deck)
            self._discard = [top]
        return self._deck.pop()

    def _active(self) -> AgentID:
        return PLAYERS[self._turn_idx % 4]

    def _advance_turn(self, steps: int = 1) -> None:
        self._turn_idx = (self._turn_idx + self._direction * steps) % 4
        self._drawn_this_turn = False

    def _apply_skip(self) -> None:
        if self._skip_next:
            self._advance_turn()
            self._skip_next = False

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in PLAYERS}
        infos   = {p: {}  for p in PLAYERS}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        active = self._active()
        if active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip()
        legal  = self.get_legal_moves(active)
        if action not in legal:
            infos[active] = {"error": f"Illegal: '{action}'"}
            return self._obs(), rewards, self._dones(), infos

        top = self._discard[-1]
        self._step += 1

        if self._step >= self._max_steps and not self._done:
            self._done   = True
            self._winner = "draw"
            return self._obs(), rewards, self._dones(), infos

        if action == "draw":
            forced = max(self._draw_pend, 1)
            for _ in range(forced):
                self.hands[active].append(self._draw_card())
            self._draw_pend = 0
            self._advance_turn()
            self._apply_skip()

        elif action.startswith("play "):
            card = action[5:]
            self.hands[active].remove(card)
            self._discard.append(card)
            if card in WILDS:
                if card == '+4':
                    self._draw_pend  += 4
                    self._skip_next   = True
                    self._color       = self._rng.choice(COLORS)
                else:
                    self._color = self._rng.choice(COLORS)
            else:
                self._color = _card_color(card) or self._color
                ct = _card_type(card)
                if ct == 'STOP':
                    self._skip_next = True
                elif ct == '+2':
                    self._draw_pend += 2
                    self._skip_next  = True
                elif ct == 'TAKI':
                    self._taki_active = True
                    self._taki_pid    = active
                    self._taki_color  = _card_color(card)
                    return self._obs(), rewards, self._dones(), infos
            if not self.hands[active]:
                rewards, self._winner = self._check_team_win(active)
                self._done = True
                return self._obs(), rewards, self._dones(), infos
            self._advance_turn()
            self._apply_skip()

        elif action == "end_taki":
            self._taki_active = False
            self._taki_pid    = None
            self._taki_color  = None
            self._advance_turn()
            self._apply_skip()

        elif action.startswith("taki "):
            tokens = action.split()[1:]
            for card in tokens:
                self.hands[active].remove(card)
                self._discard.append(card)
                self._color = _card_color(card) or self._color
            self._taki_active = False
            self._taki_pid    = None
            self._taki_color  = None
            if not self.hands[active]:
                rewards, self._winner = self._check_team_win(active)
                self._done = True
                return self._obs(), rewards, self._dones(), infos
            self._advance_turn()
            self._apply_skip()

        return self._obs(), rewards, self._dones(), infos

    def _check_team_win(self, pid: AgentID) -> Tuple[Dict[AgentID, float], str]:
        rewards: Dict[AgentID, float] = {p: 0.0 for p in PLAYERS}
        if pid in TEAM_A:
            partner = TEAM_A[1] if pid == TEAM_A[0] else TEAM_A[0]
            if not self.hands[partner]:
                for p in TEAM_A: rewards[p]  =  1.0
                for p in TEAM_B: rewards[p]  = -1.0
                return rewards, "team_A"
        else:
            partner = TEAM_B[1] if pid == TEAM_B[0] else TEAM_B[0]
            if not self.hands[partner]:
                for p in TEAM_B: rewards[p]  =  1.0
                for p in TEAM_A: rewards[p]  = -1.0
                return rewards, "team_B"
        for p in PLAYERS:
            rewards[p] = 0.3 if p in (TEAM_A if pid in TEAM_A else TEAM_B) else -0.3
        return rewards, ("team_A" if pid in TEAM_A else "team_B") + "_partial"

    def get_text_state(self, agent_id: AgentID) -> str:
        active = self._active()
        top    = self._discard[-1] if self._discard else None
        team   = "team_A" if agent_id in TEAM_A else "team_B"
        partner = (TEAM_A[1] if agent_id == TEAM_A[0] else TEAM_A[0]) if agent_id in TEAM_A else \
                  (TEAM_B[1] if agent_id == TEAM_B[0] else TEAM_B[0])
        state = {
            "agent_id":       agent_id,
            "team":           team,
            "partner":        partner,
            "partner_hand_size": len(self.hands.get(partner, [])),
            "is_your_turn":   active == agent_id,
            "active_player":  active,
            "top_card":       top,
            "active_color":   self._color,
            "pending_draws":  self._draw_pend,
            "your_hand":      list(self.hands.get(agent_id, [])),
            "hand_sizes":     {p: len(self.hands[p]) for p in PLAYERS},
            "taki_in_progress": self._taki_active and self._taki_pid == agent_id,
            "taki_color":     self._taki_color,
            "legal_moves":    self.get_legal_moves(agent_id),
            "step":           self._step,
            "game_over":      self._done,
            "winner":         self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        active = self._active()
        if active != agent_id:
            return []

        hand  = self.hands[agent_id]
        top   = self._discard[-1] if self._discard else None

        if self._taki_active and self._taki_pid == agent_id:
            taki_clr = self._taki_color
            same_color = [c for c in hand if _card_color(c) == taki_clr and _card_type(c) != 'TAKI']
            moves = ["end_taki"]
            for subset_size in range(1, len(same_color) + 1):
                from itertools import combinations as comb
                for combo in comb(same_color, subset_size):
                    moves.append("taki " + " ".join(combo))
            return moves

        moves: List[str] = []
        playable = [c for c in hand if _is_playable(c, top or '', self._color)]

        if self._draw_pend > 0:
            stacks = [c for c in playable if _card_type(c) in ('+2', '+4')]
            if stacks:
                moves.extend(f"play {c}" for c in stacks)
            else:
                moves.append("draw")
        else:
            if playable:
                moves.extend(f"play {c}" for c in playable)
            if not self._drawn_this_turn:
                moves.append("draw")
        return moves

    def get_game_rules(self) -> str:
        return """
=== TEAM TAKI (2v2) — Game Rules ===

TEAMS    : team_A (p0, p2)  vs  team_B (p1, p3)
WIN      : BOTH team members empty their hands before the other team

DECK
----
Each of 4 colors (R,B,Y,G): numbers 1–9 (×2), STOP, +2, TAKI
Wild cards: CC (Color Change) ×4, +4 ×4

MATCHING RULE
-------------
A card is playable if it matches the TOP CARD by:
  • Same color, OR
  • Same type/number
Wild cards (CC, +4) are ALWAYS playable.

ACTIONS
-------
  "play <card>"         — play one matching card
  "taki <c1> <c2> ..."  — after playing a TAKI: play same-color cards in sequence
  "draw"               — draw a card (or forced draw if pending)

SPECIAL CARDS
-------------
  STOP       : next player loses their turn
  +2         : next player draws 2 and loses turn (stackable with another +2)
  TAKI       : keep playing same-color cards until you choose to stop or run out
  CC (Color Change): declare a new color (game picks randomly here)
  +4         : next draws 4 and loses turn; any color follows

CARD FORMAT
-----------
  R1 = Red 1        GSTOP = Green Stop
  B+2 = Blue +2     YTAKI = Yellow Taki
  CC = Color Change  +4 = Wild +4

ACTION FORMAT
-------------
  "play R5"               → play Red 5
  "play GSTOP"            → play Green Stop
  "taki R3 R7 RSTOP"      → after TAKI card: play these Red cards
  "draw"                  → draw a card
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
            self._screen = pygame.display.set_mode((900, 500))
            pygame.display.set_caption("LLM-TeamGym · Team Taki")
            self._font  = pygame.font.SysFont("monospace", 18, bold=True)
            self._small = pygame.font.SysFont("monospace", 13)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        active = self._active()
        top    = self._discard[-1] if self._discard else "?"
        color_r = CMAP.get(self._color, (150, 150, 150))
        scr.blit(self._font.render(
            f"Team Taki  Top:{top}  Color:{self._color}  Winner:{self._winner or 'ongoing'}  Draws:{self._draw_pend}",
            True, color_r), (10, 8))
        for i, pid in enumerate(PLAYERS):
            col  = P_COLORS[i]
            team = "A" if pid in TEAM_A else "B"
            x    = 10 + i * 220
            pygame.draw.rect(scr, col, (x, 40, 205, 120), border_radius=10)
            is_act = pid == active
            scr.blit(self._small.render(f"{pid} team_{team}{' ←' if is_act else ''}", True, (255,255,255)), (x+5, 46))
            hand = self.hands.get(pid, [])
            hand_str = " ".join(hand[:10])
            scr.blit(self._small.render(f"({len(hand)}) {hand_str[:22]}", True, (255,255,255)), (x+5, 66))
        pygame.display.flip(); self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    def _obs(self) -> Dict[AgentID, Observation]:
        snap = {
            "active":   self._active(),
            "top_card": self._discard[-1] if self._discard else None,
            "color":    self._color,
            "hand_sizes": {p: len(self.hands[p]) for p in PLAYERS},
            "done":     self._done,
            "winner":   self._winner,
        }
        result = {}
        for p in PLAYERS:
            result[p] = dict(snap)
            result[p]["your_hand"] = list(self.hands.get(p, []))
        return result

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in PLAYERS}
        d["__all__"] = self._done
        return d
