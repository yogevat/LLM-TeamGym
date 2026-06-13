"""
Blackjack — casino-style card game against an automated dealer.

Standard casino rules:
  • Number cards (2–10) = face value.  Face cards (J/Q/K) = 10.  Ace = 1 or 11.
  • Dealer hits on soft ≤ 16, stands on hard/soft 17+.
  • Blackjack (Ace + 10-value on first 2 cards) pays 1.5× the bet.
  • Bust (total > 21) = automatic loss.

Multi-player variant: multiple independent players each play against the
same dealer independently (not against each other).

Actions:
  "hit"    — draw one more card
  "stand"  — end your turn (dealer resolves)
  "double" — double your bet and draw exactly one more card (first action only)

Teams  : each player is their own team; "house" is internal
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

SUITS  = ('S', 'H', 'D', 'C')
RANKS  = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')

BG_COLOR   = (10, 60, 10)
TEXT_CLR   = (230, 230, 230)
CARD_BG    = (255, 255, 255)
CARD_RED   = (200, 20, 20)
CARD_BLK   = (20, 20, 20)
P_COLORS   = [(70, 130, 220), (220, 70, 70), (60, 190, 80), (230, 200, 40)]


def _build_deck(n_decks: int = 1) -> List[Tuple[str, str]]:
    deck = [(r, s) for _ in range(n_decks) for s in SUITS for r in RANKS]
    return deck


def _rank_value(rank: str) -> int:
    if rank in ('J', 'Q', 'K'):
        return 10
    if rank == 'A':
        return 11
    return int(rank)


def _hand_value(hand: List[Tuple[str, str]]) -> int:
    total = 0
    aces  = 0
    for rank, _ in hand:
        v = _rank_value(rank)
        if v == 11:
            aces += 1
        total += v
    while total > 21 and aces:
        total -= 10
        aces  -= 1
    return total


def _hand_str(hand: List[Tuple[str, str]], hide_hole: bool = False) -> List[str]:
    cards = [f"{r}{s}" for r, s in hand]
    if hide_hole and len(cards) >= 2:
        cards[1] = "??"
    return cards


def _is_blackjack(hand: List[Tuple[str, str]]) -> bool:
    return len(hand) == 2 and _hand_value(hand) == 21


class BlackjackGame(BaseGame):
    """Casino Blackjack: 1–3 independent players vs an automated dealer."""

    def __init__(self, n_players: int = 1, n_decks: int = 4,
                 seed: Optional[int] = None):
        assert 1 <= n_players <= 3
        self.n_players   = n_players
        self.n_decks     = n_decks
        self.player_ids  = tuple(f"p{i}" for i in range(n_players))
        self._seed       = seed
        self._rng        = random.Random(seed)

        self._deck:      List[Tuple[str, str]] = []
        self._player_hands:  Dict[AgentID, List[Tuple[str, str]]] = {}
        self._dealer_hand:   List[Tuple[str, str]] = []
        self._doubled:   Dict[AgentID, bool] = {}
        self._standing:  Dict[AgentID, bool] = {}
        self._busted:    Dict[AgentID, bool] = {}
        self._phase:     str = "PLAYER_TURNS"  # PLAYER_TURNS → DEALER → DONE
        self._turn_idx:  int = 0
        self._done:      bool = False
        self._outcomes:  Dict[AgentID, str] = {}
        self._rewards:   Dict[AgentID, float] = {}
        self._step:      int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[str, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng     = random.Random(self._seed)
        self._deck    = _build_deck(self.n_decks)
        self._rng.shuffle(self._deck)
        self._player_hands = {p: [] for p in self.player_ids}
        self._dealer_hand  = []
        self._doubled  = {p: False for p in self.player_ids}
        self._standing = {p: False for p in self.player_ids}
        self._busted   = {p: False for p in self.player_ids}
        self._outcomes = {}
        self._rewards  = {p: 0.0 for p in self.player_ids}
        self._phase    = "PLAYER_TURNS"
        self._turn_idx = 0
        self._done     = False
        self._step     = 0

        for _ in range(2):
            for p in self.player_ids:
                self._player_hands[p].append(self._draw())
            self._dealer_hand.append(self._draw())

        self._check_instant_blackjacks()
        return self._obs()

    def _draw(self) -> Tuple[str, str]:
        if not self._deck:
            self._deck = _build_deck(self.n_decks)
            self._rng.shuffle(self._deck)
        return self._deck.pop()

    def _check_instant_blackjacks(self) -> None:
        dealer_bj = _is_blackjack(self._dealer_hand)
        for p in self.player_ids:
            if _is_blackjack(self._player_hands[p]):
                if dealer_bj:
                    self._outcomes[p] = "push"
                    self._rewards[p]  = 0.0
                else:
                    self._outcomes[p] = "blackjack"
                    self._rewards[p]  = 1.5
                self._standing[p] = True
        if dealer_bj and not any(p in self._outcomes for p in self.player_ids):
            for p in self.player_ids:
                if p not in self._outcomes:
                    self._outcomes[p] = "dealer_blackjack"
                    self._rewards[p]  = -1.0
                    self._standing[p] = True
        self._try_finish_if_all_done()

    def _active_player(self) -> Optional[AgentID]:
        for i in range(self.n_players):
            p = self.player_ids[(self._turn_idx + i) % self.n_players]
            if not self._standing[p] and not self._busted[p]:
                return p
        return None

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        if self._phase == "PLAYER_TURNS":
            active = self._active_player()
            if active is None:
                self._dealer_phase()
                rewards = dict(self._rewards)
                return self._obs(), rewards, self._dones(), infos

            if active not in actions_dict:
                return self._obs(), rewards, self._dones(), infos

            action = str(actions_dict[active]).strip().lower()
            legal  = self.get_legal_moves(active)
            if action not in legal:
                infos[active] = {"error": f"Illegal '{action}'"}
                return self._obs(), rewards, self._dones(), infos

            hand = self._player_hands[active]
            if action == "hit" or action == "double":
                hand.append(self._draw())
                if action == "double":
                    self._doubled[active] = True
                    self._standing[active] = True
                if _hand_value(hand) > 21:
                    self._busted[active] = True
                    self._standing[active] = True
                    self._outcomes[active] = "bust"
                    self._rewards[active]  = -2.0 if self._doubled[active] else -1.0
            elif action == "stand":
                self._standing[active] = True

            self._turn_idx += 1
            active2 = self._active_player()
            if active2 is None:
                self._dealer_phase()

        self._step += 1
        if self._done:
            rewards = dict(self._rewards)
        return self._obs(), rewards, self._dones(), infos

    def _dealer_phase(self) -> None:
        self._phase = "DEALER"
        while _hand_value(self._dealer_hand) < 17:
            self._dealer_hand.append(self._draw())
        dealer_val  = _hand_value(self._dealer_hand)
        dealer_bust = dealer_val > 21
        for p in self.player_ids:
            if p in self._outcomes:
                continue
            player_val = _hand_value(self._player_hands[p])
            mult = 2.0 if self._doubled[p] else 1.0
            if dealer_bust or player_val > dealer_val:
                self._outcomes[p] = "win"
                self._rewards[p]  = mult
            elif player_val == dealer_val:
                self._outcomes[p] = "push"
                self._rewards[p]  = 0.0
            else:
                self._outcomes[p] = "lose"
                self._rewards[p]  = -mult
        self._done  = True
        self._phase = "DONE"

    def _try_finish_if_all_done(self) -> None:
        if all(self._standing[p] or self._busted[p] for p in self.player_ids):
            self._dealer_phase()

    def get_text_state(self, agent_id: AgentID) -> str:
        active = self._active_player() if self._phase == "PLAYER_TURNS" else None
        dealer_visible = _hand_str(self._dealer_hand, hide_hole=not self._done)
        dealer_shown   = _hand_value([self._dealer_hand[0]]) if not self._done else _hand_value(self._dealer_hand)
        state = {
            "agent_id":     agent_id,
            "phase":        self._phase,
            "is_your_turn": active == agent_id,
            "your_hand":    _hand_str(self._player_hands.get(agent_id, [])),
            "your_value":   _hand_value(self._player_hands.get(agent_id, [])),
            "your_doubled": self._doubled.get(agent_id, False),
            "your_standing":self._standing.get(agent_id, False),
            "your_busted":  self._busted.get(agent_id, False),
            "your_outcome": self._outcomes.get(agent_id),
            "your_reward":  self._rewards.get(agent_id, 0.0),
            "dealer_visible_cards": dealer_visible,
            "dealer_visible_value": dealer_shown,
            "dealer_has_hole_card": not self._done,
            "other_players": {
                p: {
                    "hand":     _hand_str(self._player_hands[p]),
                    "value":    _hand_value(self._player_hands[p]),
                    "standing": self._standing[p],
                    "busted":   self._busted[p],
                    "outcome":  self._outcomes.get(p),
                }
                for p in self.player_ids if p != agent_id
            },
            "legal_moves":  self.get_legal_moves(agent_id),
            "game_over":    self._done,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done or self._phase != "PLAYER_TURNS":
            return []
        active = self._active_player()
        if active != agent_id:
            return []
        if self._standing[agent_id] or self._busted[agent_id]:
            return []
        hand    = self._player_hands[agent_id]
        moves   = ["hit", "stand"]
        if len(hand) == 2:
            moves.append("double")
        return moves

    def get_game_rules(self) -> str:
        return f"""
=== BLACKJACK — Casino Rules ===

PLAYERS : {self.n_players} (each plays independently against the dealer)
DECKS   : {self.n_decks}

CARD VALUES
-----------
  Number cards (2–10) = face value
  Face cards (J, Q, K) = 10
  Ace = 11 (or 1 if 11 would bust)

OBJECTIVE
---------
Get a hand value as close to 21 as possible WITHOUT exceeding it,
while beating the dealer's total.

HOW TO PLAY
-----------
Each player starts with 2 cards. The dealer has 2 cards but one is
face-down (the "hole card").

On your turn:
  "hit"    — draw one more card.
  "stand"  — keep your current hand; end your turn.
  "double" — (first action only) commit to one more card; reward/penalty doubled.

BUST : Hand total > 21 → automatic loss (reward = −1, or −2 if doubled).

DEALER RULES (automatic)
-------------------------
After all players stand or bust, the dealer reveals the hole card and
must HIT until reaching 17+ (hard or soft). Dealer stands on 17+.

OUTCOMES
--------
  Blackjack (Ace + 10-value on first 2 cards) → reward +1.5 (unless dealer also has BJ → push)
  Win  (player > dealer, or dealer busts)     → reward +1 (or +2 if doubled)
  Push (tie)                                  → reward  0
  Lose (player < dealer)                      → reward −1 (or −2 if doubled)
  Bust                                        → reward −1 (or −2 if doubled)

ACTION FORMAT
-------------
  "hit"    — draw a card
  "stand"  — end your turn
  "double" — double-down (first action only)
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
            self._screen = pygame.display.set_mode((800, 500))
            pygame.display.set_caption("LLM-TeamGym · Blackjack")
            self._font  = pygame.font.SysFont("monospace", 22, bold=True)
            self._small = pygame.font.SysFont("monospace", 16)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)

        def draw_card(x: int, y: int, card: str) -> None:
            hidden = card == "??"
            pygame.draw.rect(scr, CARD_BG, (x, y, 52, 72), border_radius=6)
            if hidden:
                pygame.draw.rect(scr, (30, 80, 150), (x+3, y+3, 46, 66), border_radius=4)
            else:
                color = CARD_RED if card[-1] in ('H', 'D') else CARD_BLK
                lbl = self._small.render(card, True, color)
                scr.blit(lbl, (x+4, y+4))

        dealer_cards = _hand_str(self._dealer_hand, hide_hole=not self._done)
        scr.blit(self._font.render("DEALER", True, TEXT_CLR), (20, 10))
        for ci, c in enumerate(dealer_cards):
            draw_card(20 + ci * 60, 38, c)
        dval = _hand_value([self._dealer_hand[0]]) if not self._done else _hand_value(self._dealer_hand)
        scr.blit(self._small.render(f"Shown: {dval}", True, TEXT_CLR), (20, 120))

        for i, pid in enumerate(self.player_ids):
            y0  = 160 + i * 130
            col = P_COLORS[i % len(P_COLORS)]
            outcome = self._outcomes.get(pid, "")
            scr.blit(self._font.render(f"{pid}  {outcome}", True, col), (20, y0))
            hand = _hand_str(self._player_hands.get(pid, []))
            for ci, c in enumerate(hand):
                draw_card(20 + ci * 60, y0 + 28, c)
            pval = _hand_value(self._player_hands.get(pid, []))
            scr.blit(self._small.render(f"Value: {pval}  Reward: {self._rewards.get(pid,0):+.1f}",
                                        True, TEXT_CLR), (20, y0 + 106))
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
            "phase":        self._phase,
            "dealer_shown": _hand_str(self._dealer_hand, hide_hole=not self._done),
            "outcomes":     dict(self._outcomes),
            "done":         self._done,
        }
        result = {}
        for p in self.player_ids:
            result[p] = dict(snap)
            result[p]["your_hand"]  = _hand_str(self._player_hands.get(p, []))
            result[p]["your_value"] = _hand_value(self._player_hands.get(p, []))
        return result

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
