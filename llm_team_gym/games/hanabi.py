"""
Hanabi — cooperative card game (2–4 players) testing Theory of Mind.

Players cooperate to play cards in the correct order (1→5 for each of
5 colours). The twist: you CANNOT see your own cards but you CAN see
everyone else's. Teammates give limited "clues" (colour or number hints)
to help you deduce what you hold.

Hidden information
------------------
  - Each agent's `get_text_state` shows their own hand as opaque slot
    objects annotated ONLY with received clues — colour and number values
    are STRICTLY OMITTED.
  - All OTHER players' hands are fully visible (colour + number).
  - The complete clue history is shown to all players.

Resources
---------
  Clue tokens : 8 max. Giving a clue costs 1. Discarding gains 1.
  Fuse tokens : 3. Playing a wrong card costs 1. Zero fuses → score 0.

Actions (turn-based, one per active player)
-------------------------------------------
  "play N"              — play card at hand index N (0-indexed)
  "discard N"           — discard card at index N (gains 1 clue token)
  "clue <pid> color <C>"  — give colour clue (costs 1 clue token)
  "clue <pid> number <N>" — give number clue (costs 1 clue token)

Win / scoring
-------------
  Score = sum of firework heights (max 25). Logged as reward = score/25.
  Game ends when: deck exhausted + last round played, OR 3 fuses used,
  OR all fireworks reach 5 (perfect score).
  Players win (+1.0) on score ≥ 20; draw (+0.0) on 10–19; lose (-1.0) < 10.

Agents     : "p0", "p1" [, "p2", "p3"]  (configurable)
Teams      : {"cooperation_team": all_agents}   (fully cooperative)
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward,
    StepResult, TeamID,
)

COLORS = ("red", "blue", "green", "yellow", "white")
# Standard Hanabi card counts per colour
CARD_COUNTS: Dict[int, int] = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
HAND_SIZES: Dict[int, int]  = {2: 5, 3: 5, 4: 4}
MAX_CLUES  = 8
MAX_FUSES  = 3
WIN_SCORE  = 20
MID_SCORE  = 10

# Pygame
CARD_W  = 68
CARD_H  = 90
CARD_GAP = 8
PAD     = 20
INFO_H  = 80
BG_COLOR   = (15, 15, 25)
BACK_COLOR = (50, 50, 80)
FONT_COLOR = (220, 220, 220)
FUSE_COLOR = (200, 60, 60)
CLUE_COLOR = (60, 200, 120)
COLOUR_MAP: Dict[str, Tuple[int, int, int]] = {
    "red":    (220,  70,  70),
    "blue":   ( 60, 130, 220),
    "green":  ( 60, 190,  80),
    "yellow": (230, 200,  40),
    "white":  (230, 230, 230),
}


class HanabiGame(BaseGame):
    """
    Full Hanabi with strict hidden-information text states.

    hands[pid]       : list of (color, number) — NEVER shown to owner.
    hand_clues[pid]  : list of sets of clue strings, one per card position.
    clue_log         : ordered list of all clue events.
    discard_pile     : list of (color, number) discarded.
    fireworks        : dict color → int (height of played pile, 0 means nothing played).
    """

    def __init__(self, n_players: int = 3, seed: Optional[int] = None):
        if n_players not in HAND_SIZES:
            raise ValueError(f"n_players must be 2, 3, or 4; got {n_players}")
        self.n_players = n_players
        self.player_ids: Tuple[AgentID, ...] = tuple(f"p{i}" for i in range(n_players))
        self._seed = seed
        self._rng  = random.Random(seed)

        self.hands:       Dict[AgentID, List[Tuple[str, int]]] = {}
        self.hand_clues:  Dict[AgentID, List[Set[str]]] = {}
        self.fireworks:   Dict[str, int] = {}
        self.deck:        List[Tuple[str, int]] = []
        self.discard_pile: List[Tuple[str, int]] = []
        self.clue_log:    List[Dict[str, Any]] = []
        self.clue_tokens: int = MAX_CLUES
        self.fuse_tokens: int = MAX_FUSES
        self._turn_idx:   int = 0
        self._last_round: bool = False
        self._last_round_remaining: int = 0
        self._done:       bool = False
        self._score:      int = 0
        self._step:       int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    # ------------------------------------------------------------------
    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {"cooperation_team": list(self.player_ids)}

    # ------------------------------------------------------------------
    def reset(self) -> Dict[AgentID, Observation]:
        self._rng = random.Random(self._seed)

        # Build and shuffle deck
        self.deck = []
        for color in COLORS:
            for num, count in CARD_COUNTS.items():
                self.deck.extend([(color, num)] * count)
        self._rng.shuffle(self.deck)

        hand_size = HAND_SIZES[self.n_players]
        self.hands      = {}
        self.hand_clues = {}
        for pid in self.player_ids:
            self.hands[pid]      = [self.deck.pop() for _ in range(hand_size)]
            self.hand_clues[pid] = [set() for _ in range(hand_size)]

        self.fireworks    = {c: 0 for c in COLORS}
        self.discard_pile = []
        self.clue_log     = []
        self.clue_tokens  = MAX_CLUES
        self.fuse_tokens  = MAX_FUSES
        self._turn_idx    = 0
        self._last_round  = False
        self._last_round_remaining = 0
        self._done        = False
        self._score       = 0
        self._step        = 0
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards: Dict[AgentID, Reward] = {p: 0.0 for p in self.player_ids}
        infos:   Dict[AgentID, Info]   = {p: {}  for p in self.player_ids}

        active = self.player_ids[self._turn_idx % self.n_players]
        if self._done or active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action = str(actions_dict[active]).strip()
        legal  = self.get_legal_moves(active)

        if action not in legal:
            infos[active] = {"error": f"Illegal action '{action}'", "legal_count": len(legal)}
            return self._obs(), rewards, self._dones(), infos

        parts = action.split()
        event_info: Dict[str, Any] = {}

        if parts[0] == "play":
            idx = int(parts[1])
            card = self.hands[active][idx]
            expected = self.fireworks[card[0]] + 1
            if card[1] == expected:
                # Successful play
                self.fireworks[card[0]] += 1
                event_info = {"played": card, "success": True}
                if card[1] == 5 and self.clue_tokens < MAX_CLUES:
                    self.clue_tokens += 1   # bonus clue for completing a colour
            else:
                # Misplay → lose a fuse
                self.discard_pile.append(card)
                self.fuse_tokens -= 1
                event_info = {"played": card, "success": False, "fuses_left": self.fuse_tokens}
            self._remove_card(active, idx)

        elif parts[0] == "discard":
            idx  = int(parts[1])
            card = self.hands[active][idx]
            self.discard_pile.append(card)
            self.clue_tokens = min(MAX_CLUES, self.clue_tokens + 1)
            event_info = {"discarded": card, "clue_tokens": self.clue_tokens}
            self._remove_card(active, idx)

        elif parts[0] == "clue":
            # "clue p1 color red"  or  "clue p1 number 3"
            target      = parts[1]
            clue_type   = parts[2]   # "color" or "number"
            clue_val    = parts[3]   # e.g., "red" or "3"
            self.clue_tokens -= 1

            touched = []
            for i, (c, n) in enumerate(self.hands[target]):
                match = (clue_type == "color" and c == clue_val) or \
                        (clue_type == "number" and n == int(clue_val))
                if match:
                    clue_str = f"is_{clue_val}"
                    self.hand_clues[target][i].add(clue_str)
                    touched.append(i)

            log_entry = {
                "step": self._step, "from": active, "to": target,
                "type": clue_type, "value": clue_val, "touched_positions": touched,
            }
            self.clue_log.append(log_entry)
            event_info = {"clue": log_entry, "clue_tokens_left": self.clue_tokens}

        infos[active] = event_info
        self._step    += 1
        self._turn_idx = (self._turn_idx + 1) % self.n_players

        # Check last-round trigger (deck exhausted)
        if not self._last_round and not self.deck:
            self._last_round           = True
            self._last_round_remaining = self.n_players  # one more turn per player

        if self._last_round:
            self._last_round_remaining -= 1

        # Check terminal conditions
        self._score = sum(self.fireworks.values())
        perfect     = all(v == 5 for v in self.fireworks.values())

        if self.fuse_tokens <= 0 or perfect or (self._last_round and self._last_round_remaining <= 0):
            self._done = True
            final_score = 0 if self.fuse_tokens <= 0 else self._score
            reward_val  = (1.0 if final_score >= WIN_SCORE else
                           0.0 if final_score >= MID_SCORE else -1.0)
            for p in self.player_ids:
                rewards[p] = reward_val

        return self._obs(), rewards, self._dones(), infos

    # ------------------------------------------------------------------
    def get_text_state(self, agent_id: AgentID) -> str:
        hand_size = HAND_SIZES[self.n_players]

        # OWN hand: positions with clues only — NO colour/number
        own_hand = []
        for i in range(len(self.hands[agent_id])):
            clues = sorted(self.hand_clues[agent_id][i])
            own_hand.append({
                "position": i,
                "known_clues": clues,
                "color": "HIDDEN",
                "number": "HIDDEN",
            })

        # OTHERS' hands: fully visible
        others_hands: Dict[str, Any] = {}
        for pid in self.player_ids:
            if pid == agent_id:
                continue
            others_hands[pid] = [
                {"position": i, "color": c, "number": n, "clues": sorted(self.hand_clues[pid][i])}
                for i, (c, n) in enumerate(self.hands[pid])
            ]

        # What's still playable on fireworks
        playable = {c: self.fireworks[c] + 1 for c in COLORS if self.fireworks[c] < 5}
        active   = self.player_ids[self._turn_idx % self.n_players]

        state: Dict[str, Any] = {
            "agent_id": agent_id,
            "is_your_turn": active == agent_id,
            "active_player": active,
            "step": self._step,
            "your_hand": {
                "description": (
                    "Your own cards are HIDDEN. You cannot see your colour or number. "
                    "Use clues from teammates to deduce what you hold."
                ),
                "cards": own_hand,
            },
            "teammates_hands": others_hands,
            "fireworks": self.fireworks,
            "next_playable": playable,
            "clue_tokens": self.clue_tokens,
            "fuse_tokens": self.fuse_tokens,
            "deck_remaining": len(self.deck),
            "last_round_triggered": self._last_round,
            "last_round_turns_left": self._last_round_remaining if self._last_round else None,
            "discard_pile_summary": self._discard_summary(),
            "clue_history": self.clue_log[-15:],  # last 15 clues
            "legal_moves": self.get_legal_moves(agent_id),
            "current_score": self._score,
            "game_over": self._done,
            "strategy_note": (
                "Give clues touching the MOST critical cards. Prioritise playing "
                f"a card if you know it completes {list(playable.items())}. "
                "Discard your oldest unclued card when tokens are needed."
            ),
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        active = self.player_ids[self._turn_idx % self.n_players]
        if active != agent_id:
            return []

        hand_size = len(self.hands[agent_id])
        moves: List[Action] = []
        # Play / discard any card
        for i in range(hand_size):
            moves.append(f"play {i}")
            moves.append(f"discard {i}")

        # Clues (only if tokens available)
        if self.clue_tokens > 0:
            for pid in self.player_ids:
                if pid == agent_id:
                    continue
                seen_colors: Set[str] = set()
                seen_numbers: Set[int] = set()
                for c, n in self.hands[pid]:
                    seen_colors.add(c)
                    seen_numbers.add(n)
                for col in seen_colors:
                    moves.append(f"clue {pid} color {col}")
                for num in seen_numbers:
                    moves.append(f"clue {pid} number {num}")

        return moves

    def get_game_rules(self) -> str:
        return f"""
=== HANABI — Game Rules ({self.n_players} Players) ===

OVERVIEW
--------
Hanabi is a COOPERATIVE game. All players share a single score and win
or lose together. Communicate carefully — clue tokens are scarce.

HIDDEN INFORMATION (critical)
------------------------------
  YOU CANNOT SEE YOUR OWN HAND.
    → your_hand shows "HIDDEN" for colour and number.
    → Only the clues you have received are shown under "known_clues".

  YOU CAN SEE ALL OTHER PLAYERS' HANDS.
    → teammates_hands shows each card's colour and number.
    → Use this to give targeted clues.

  Clue history is PUBLIC — all players see every clue given.

THE DECK
--------
5 colours: red, blue, green, yellow, white.
Each colour has 10 cards: three 1s, two 2s, two 3s, two 4s, one 5.
Total deck: 50 cards.
Hand size: {HAND_SIZES[self.n_players]} cards per player.

RESOURCES
---------
  Clue tokens : {MAX_CLUES} max. Giving a clue costs 1. Discarding gains 1 (up to max).
                Completing a colour pile (playing the 5) refills 1 clue token.
  Fuse tokens : {MAX_FUSES}. Playing a card that does NOT fit the sequence costs 1 fuse.
                Zero fuses → game ends immediately with score 0.

FIREWORK PILES
--------------
One pile per colour, played in strict ascending order: 1, 2, 3, 4, 5.
"next_playable" shows exactly which number each colour needs next.

TURN STRUCTURE
--------------
On your turn, do exactly ONE of:

  1. PLAY a card  →  "play N"  (N = hand index 0–{HAND_SIZES[self.n_players]-1})
     - If the card extends the firework for its colour → SUCCESS.
     - Otherwise → lose 1 fuse token; card is discarded.

  2. DISCARD a card  →  "discard N"
     - Gain 1 clue token (max {MAX_CLUES}).
     - Useful when you need tokens but have no safe play.

  3. GIVE a CLUE  →  "clue <player_id> color <colour>"
                 or  "clue <player_id> number <N>"
     - Costs 1 clue token.
     - Must touch at least one card in the target's hand.
     - A colour clue points out ALL cards of that colour in the hand.
     - A number clue points out ALL cards of that number in the hand.
     - The touched positions are recorded in clue history for everyone to see.

GAME END
--------
  Perfect score (25): all fireworks complete.
  Deck exhaustion: when the last card is drawn, each player gets one more turn.
  Fuse loss: 3 misplays → game over, score = 0.

SCORING
-------
  Score = sum of highest card played per colour (0–25).
  Your reward:  ≥ {WIN_SCORE} → +1.0 (win)  |  {MID_SCORE}–{WIN_SCORE-1} → 0.0  |  < {MID_SCORE} → -1.0 (loss)

ACTION FORMAT
-------------
  "play 2"               — play card at index 2
  "discard 0"            — discard card at index 0
  "clue p1 color red"    — tell p1 which of their cards are red
  "clue p2 number 3"     — tell p2 which of their cards are 3s

Always choose from the legal_moves list. Clue moves list only valid targets.
""".strip()

    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        try:
            import pygame
        except ImportError:
            return

        n = self.n_players
        hand_size = HAND_SIZES[n]

        if not self._pygame_init:
            pygame.init()
            w = PAD * 2 + max(n, 5) * (CARD_W + CARD_GAP) * 2
            h = PAD * 3 + n * (CARD_H + CARD_GAP * 3) + INFO_H + CARD_H + 30
            self._screen = pygame.display.set_mode((w, min(h, 900)))
            pygame.display.set_caption("LLM-TeamGym · Hanabi")
            self._font  = pygame.font.SysFont("monospace", 18, bold=True)
            self._small = pygame.font.SysFont("monospace", 12)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return

        scr = self._screen
        w   = scr.get_width()
        scr.fill(BG_COLOR)

        def draw_card(x, y, color, number, clues=None, is_back=False):
            bg = BACK_COLOR if is_back else COLOUR_MAP.get(color, (150, 150, 150))
            pygame.draw.rect(scr, bg, (x, y, CARD_W, CARD_H), border_radius=6)
            pygame.draw.rect(scr, (200, 200, 200), (x, y, CARD_W, CARD_H), border_radius=6, width=2)
            if not is_back:
                lbl = self._font.render(str(number), True,
                                        (30, 30, 30) if color == "yellow" else (240, 240, 240))
                scr.blit(lbl, (x + CARD_W // 2 - lbl.get_width() // 2,
                               y + CARD_H // 2 - lbl.get_height() // 2))
            elif clues:
                for j, clue in enumerate(list(clues)[:3]):
                    cl = self._small.render(clue, True, CLUE_COLOR)
                    scr.blit(cl, (x + 3, y + 4 + j * 16))

        # Firework piles
        fw_y = PAD
        for i, color in enumerate(COLORS):
            fx = PAD + i * (CARD_W + CARD_GAP)
            val = self.fireworks[color]
            bg = COLOUR_MAP[color]
            pygame.draw.rect(scr, bg, (fx, fw_y, CARD_W, CARD_H), border_radius=6)
            lbl = self._font.render(str(val), True,
                                    (30, 30, 30) if color == "yellow" else (240, 240, 240))
            scr.blit(lbl, (fx + CARD_W // 2 - lbl.get_width() // 2,
                           fw_y + CARD_H // 2 - lbl.get_height() // 2))
            clbl = self._small.render(color[:3].upper(), True, (200, 200, 200))
            scr.blit(clbl, (fx + CARD_W // 2 - clbl.get_width() // 2, fw_y + CARD_H + 2))

        # Token counts
        tok_x = PAD + 5 * (CARD_W + CARD_GAP) + 20
        ct_lbl = self._font.render(f"Clues: {self.clue_tokens}/{MAX_CLUES}", True, CLUE_COLOR)
        scr.blit(ct_lbl, (tok_x, fw_y))
        ft_lbl = self._font.render(f"Fuses: {self.fuse_tokens}/{MAX_FUSES}", True, FUSE_COLOR)
        scr.blit(ft_lbl, (tok_x, fw_y + 30))
        dk_lbl = self._small.render(f"Deck: {len(self.deck)}", True, FONT_COLOR)
        scr.blit(dk_lbl, (tok_x, fw_y + 58))

        # Player hands
        hand_start_y = fw_y + CARD_H + 40
        active = self.player_ids[self._turn_idx % self.n_players]
        for pi, pid in enumerate(self.player_ids):
            row_y = hand_start_y + pi * (CARD_H + CARD_GAP * 4)
            # Label
            color = (200, 220, 100) if pid == active else FONT_COLOR
            plbl = self._font.render(f"{pid} {'← active' if pid==active else ''}", True, color)
            scr.blit(plbl, (PAD, row_y - 18))
            for ci, (card_color, card_num) in enumerate(self.hands[pid]):
                cx = PAD + ci * (CARD_W + CARD_GAP)
                clues = self.hand_clues[pid][ci]
                draw_card(cx, row_y, card_color, card_num, clues)

        # Info bar
        info_y = hand_start_y + n * (CARD_H + CARD_GAP * 4) + 10
        pygame.draw.rect(scr, (25, 25, 35), (0, info_y, w, INFO_H))
        score = sum(self.fireworks.values())
        if self._done:
            msg   = f"GAME OVER — Score: {score}/25"
            color = CLUE_COLOR if score >= WIN_SCORE else FUSE_COLOR
        else:
            msg   = f"Step {self._step}  |  Score so far: {score}/25  |  Active: {active}"
            color = FONT_COLOR
        lbl = self._small.render(msg, True, color)
        scr.blit(lbl, (PAD, info_y + 18))
        recent = self.clue_log[-3:] if self.clue_log else []
        for j, cl in enumerate(recent):
            txt = (f"  Step {cl['step']}: {cl['from']} told {cl['to']} "
                   f"{cl['type']} {cl['value']} → positions {cl['touched_positions']}")
            llbl = self._small.render(txt, True, CLUE_COLOR)
            scr.blit(llbl, (PAD, info_y + 38 + j * 14))

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
    def _remove_card(self, pid: AgentID, idx: int) -> None:
        """Remove card at idx, draw replacement from deck."""
        self.hands[pid].pop(idx)
        self.hand_clues[pid].pop(idx)
        if self.deck:
            self.hands[pid].append(self.deck.pop())
            self.hand_clues[pid].append(set())

    def _discard_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Dict[int, int]] = {}
        for c, n in self.discard_pile:
            if c not in summary:
                summary[c] = {}
            summary[c][n] = summary[c].get(n, 0) + 1
        return summary

    def _obs(self) -> Dict[AgentID, Observation]:
        base = {
            "fireworks":    dict(self.fireworks),
            "clue_tokens":  self.clue_tokens,
            "fuse_tokens":  self.fuse_tokens,
            "deck_remaining": len(self.deck),
            "score":        self._score,
            "done":         self._done,
            "active_player": self.player_ids[self._turn_idx % self.n_players],
        }
        obs: Dict[AgentID, Observation] = {}
        for pid in self.player_ids:
            o = dict(base)
            # Never include own cards in raw observation either
            o["hands_visible"] = {
                p: [(c, n) for c, n in self.hands[p]]
                for p in self.player_ids if p != pid
            }
            o["own_clues"] = [sorted(cl) for cl in self.hand_clues[pid]]
            obs[pid] = o
        return obs

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
