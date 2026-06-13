"""
Yaniv — Israeli bluff-and-risk card game.

Each turn: discard one or more valid cards from hand, then draw one card.
At the START of any turn, call "yaniv" if your hand sum ≤ 7.

VALID DISCARDS
  • Single card       — always valid
  • Set               — 2+ cards of the same rank (Jokers are wild)
  • Run               — 3+ consecutive ranks in the same suit (Jokers wild)

YANIV CALL
  • If nobody has a lower (or equal) sum: caller wins the round.
  • ASSAF: if any other player has ≤ caller's sum, caller gets +30 penalty.
  • Lowest-sum player(s) get 0 penalty; others add their hand sum.

JOKERS : value = 0, wild for sets and runs.
Card values : A=1, 2–10=face, J=11, Q=12, K=13, JO=0.

Action format (DISCARD phase) : space-separated card codes OR "yaniv"
    e.g.  "5H 5D"   "3S 4S 5S"   "KH"   "JO 7C"
Action format (DRAW phase)    : "deck"   or   "pile"

Teams : each player is their own team
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

SUITS  = ('S', 'H', 'D', 'C')
RANKS  = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
RANK_VAL: Dict[str, int] = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'JO': 0,
}
RANK_ORDER: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}

YANIV_THRESHOLD = 7
ASSAF_PENALTY   = 30

BG_COLOR   = (15, 15, 30)
TEXT_CLR   = (230, 230, 230)
CARD_BG    = (255, 255, 245)
CARD_RED   = (190, 20, 20)
CARD_BLK   = (20, 20, 20)
P_COLORS   = [(70, 130, 220), (220, 70, 70), (60, 190, 80), (230, 200, 40)]


def _build_deck() -> List[Tuple[str, str]]:
    deck = [(r, s) for s in SUITS for r in RANKS]
    deck += [('JO', '*'), ('JO', '*')]
    return deck


def _card_str(card: Tuple[str, str]) -> str:
    r, s = card
    return f"{r}{s}" if r != 'JO' else 'JO'


def _parse_card(token: str) -> Optional[Tuple[str, str]]:
    token = token.strip().upper()
    if token in ('JO', 'JOKER'):
        return ('JO', '*')
    if len(token) < 2:
        return None
    suit = token[-1]
    rank = token[:-1]
    if suit not in SUITS or rank not in RANKS:
        return None
    return (rank, suit)


def _hand_sum(hand: List[Tuple[str, str]]) -> int:
    return sum(RANK_VAL[r] for r, _ in hand)


def _is_valid_combo(cards: List[Tuple[str, str]]) -> bool:
    if not cards:
        return False
    non_jokers = [(r, s) for r, s in cards if r != 'JO']
    joker_count = len(cards) - len(non_jokers)

    if len(cards) == 1:
        return True

    # Set: same rank (≥2 cards), jokers are wild
    if non_jokers:
        ranks_set = {r for r, _ in non_jokers}
        if len(ranks_set) == 1:
            return True

    # Run: 3+ consecutive same suit (jokers fill gaps)
    if len(cards) >= 3 and non_jokers:
        suits_set = {s for _, s in non_jokers}
        if len(suits_set) == 1:
            suit = next(iter(suits_set))
            sorted_ranks = sorted(RANK_ORDER[r] for r, _ in non_jokers)
            span = sorted_ranks[-1] - sorted_ranks[0] + 1
            if span <= len(cards):
                needed = span - len(non_jokers)
                if needed <= joker_count:
                    return True

    return False


def _valid_discards(hand: List[Tuple[str, str]]) -> List[str]:
    results: List[str] = []
    n = len(hand)
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            cards = [hand[i] for i in combo]
            if _is_valid_combo(cards):
                results.append(" ".join(_card_str(c) for c in cards))
    return results


class YanivGame(BaseGame):
    """Yaniv card game — 2–4 players."""

    def __init__(self, n_players: int = 3, seed: Optional[int] = None,
                 max_steps: int = 500):
        assert 2 <= n_players <= 4
        self.n_players   = n_players
        self.max_steps   = max_steps
        self.player_ids  = tuple(f"p{i}" for i in range(n_players))
        self._seed       = seed
        self._rng        = random.Random(seed)

        self.hands:       Dict[AgentID, List[Tuple[str, str]]] = {}
        self.penalties:   Dict[AgentID, int] = {}
        self._deck:       List[Tuple[str, str]] = []
        self._discard_pile: List[Tuple[str, str]] = []
        self._turn_idx:   int = 0
        self._phase:      str = "DISCARD"
        self._top_before: Optional[Tuple[str, str]] = None
        self._done:       bool = False
        self._winner:     Optional[str] = None
        self._last_event: Optional[Dict] = None
        self._step:       int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[str, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng         = random.Random(self._seed)
        self._deck        = _build_deck()
        self._rng.shuffle(self._deck)
        self.hands        = {p: [] for p in self.player_ids}
        self.penalties    = {p: 0  for p in self.player_ids}
        self._discard_pile = []
        self._turn_idx    = 0
        self._phase       = "DISCARD"
        self._top_before  = None
        self._done        = False
        self._winner      = None
        self._last_event  = None
        self._step        = 0
        for _ in range(5):
            for p in self.player_ids:
                self.hands[p].append(self._deck.pop())
        self._discard_pile.append(self._deck.pop())
        return self._obs()

    def _active(self) -> AgentID:
        return self.player_ids[self._turn_idx % self.n_players]

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        active = self._active()
        if active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos

        action_raw = str(actions_dict[active]).strip()
        action     = action_raw.lower()
        legal      = self.get_legal_moves(active)
        legal_lc   = [m.lower() for m in legal]
        if action not in legal_lc:
            infos[active] = {"error": f"Illegal action '{action_raw}'"}
            return self._obs(), rewards, self._dones(), infos
        action_raw = legal[legal_lc.index(action)]

        self._step += 1

        if self._step >= self.max_steps and not self._done:
            rewards, infos = self._resolve_yaniv(self._active())
            return self._obs(), rewards, self._dones(), infos

        if self._phase == "DISCARD":
            if action == "yaniv":
                rewards, infos = self._resolve_yaniv(active)
                return self._obs(), rewards, self._dones(), infos

            tokens = action_raw.split()
            played = [_parse_card(t) for t in tokens]
            for card in played:
                self.hands[active].remove(card)
                self._discard_pile.append(card)
            self._top_before = self._discard_pile[-len(played) - 1] if len(self._discard_pile) > len(played) else None
            self._phase      = "DRAW"

        elif self._phase == "DRAW":
            if action == "deck":
                if self._deck:
                    self.hands[active].append(self._deck.pop())
                else:
                    mid = len(self._discard_pile) // 2
                    refill = self._discard_pile[:mid]
                    self._rng.shuffle(refill)
                    self._deck = refill
                    self._discard_pile = self._discard_pile[mid:]
                    self.hands[active].append(self._deck.pop())
            else:
                drawn = self._top_before
                if drawn and drawn in self._discard_pile:
                    self._discard_pile.remove(drawn)
                    self.hands[active].append(drawn)
                elif self._discard_pile:
                    self.hands[active].append(self._discard_pile.pop())

            self._phase     = "DISCARD"
            self._top_before = None
            self._turn_idx  += 1

        return self._obs(), rewards, self._dones(), infos

    def _resolve_yaniv(self, caller: AgentID) -> Tuple[Dict[AgentID, float], Dict[AgentID, Dict]]:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        caller_sum = _hand_sum(self.hands[caller])
        assafed = any(
            _hand_sum(self.hands[p]) <= caller_sum
            for p in self.player_ids if p != caller
        )
        if assafed:
            self.penalties[caller] += caller_sum + ASSAF_PENALTY
            rewards[caller] = -1.0
            infos[caller]   = {"assaf": True, "your_sum": caller_sum, "penalty": caller_sum + ASSAF_PENALTY}
        else:
            min_sum = min(_hand_sum(self.hands[p]) for p in self.player_ids)
            for p in self.player_ids:
                h_sum = _hand_sum(self.hands[p])
                if p == caller:
                    self.penalties[p] += 0
                else:
                    self.penalties[p] += h_sum
                if h_sum == min_sum:
                    rewards[p] = 1.0
                elif p == caller:
                    rewards[p] = 0.5
                else:
                    rewards[p] = -0.5
            infos[caller] = {"yaniv": True, "your_sum": caller_sum}

        sums_info = {p: _hand_sum(self.hands[p]) for p in self.player_ids}
        infos["__all__"] = {"hand_sums": sums_info, "penalties": dict(self.penalties)}
        self._done   = True
        self._winner = min(self.player_ids, key=lambda p: self.penalties[p])
        return rewards, infos

    def get_text_state(self, agent_id: AgentID) -> str:
        active = self._active()
        hand   = self.hands.get(agent_id, [])
        top    = _card_str(self._discard_pile[-1]) if self._discard_pile else None
        state  = {
            "agent_id":       agent_id,
            "is_your_turn":   active == agent_id,
            "active_player":  active,
            "phase":          self._phase,
            "your_hand":      [_card_str(c) for c in hand],
            "your_hand_sum":  _hand_sum(hand),
            "can_call_yaniv": _hand_sum(hand) <= YANIV_THRESHOLD,
            "yaniv_threshold":YANIV_THRESHOLD,
            "discard_pile_top": top,
            "pile_size":      len(self._discard_pile),
            "deck_size":      len(self._deck),
            "other_players":  {
                p: {"hand_size": len(self.hands[p])}
                for p in self.player_ids if p != agent_id
            },
            "penalties":      dict(self.penalties),
            "legal_moves":    self.get_legal_moves(agent_id),
            "step":           self._step,
            "game_over":      self._done,
            "winner":         self._winner,
            "card_values":    {c: RANK_VAL[c] for c in RANK_VAL},
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        active = self._active()
        if active != agent_id:
            return []
        if self._phase == "DRAW":
            moves = ["deck"]
            if self._top_before is not None:
                moves.append("pile")
            return moves
        hand = self.hands[agent_id]
        moves: List[str] = []
        if _hand_sum(hand) <= YANIV_THRESHOLD:
            moves.append("yaniv")
        moves.extend(_valid_discards(hand))
        return moves

    def get_game_rules(self) -> str:
        return f"""
=== YANIV — Israeli Card Game Rules ===

PLAYERS : {self.n_players} (each is their own team)
DECK    : 54 cards (52 standard + 2 Jokers)

CARD VALUES
-----------
  Ace=1, 2-10=face value, Jack=11, Queen=12, King=13, Joker=0

TURN STRUCTURE
--------------
DISCARD PHASE: Choose one of:
  • "yaniv"          — call Yaniv (only if your hand sum ≤ {YANIV_THRESHOLD})
  • "<card(s)>"      — discard 1 or more cards in a valid combo:
      - Single  : any 1 card
      - Set     : 2+ cards of the SAME RANK (Jokers are wild)
      - Run     : 3+ consecutive ranks in the SAME SUIT (Jokers fill gaps)

DRAW PHASE: After discarding, draw 1 card:
  "deck"  — top of the face-down deck
  "pile"  — the card that was on top of the discard pile before your discard

YANIV CALL
----------
When your hand sum ≤ {YANIV_THRESHOLD}, you may call "yaniv" at the start of your turn.
Everyone reveals their hands:
  • If the caller has the LOWEST sum: caller wins (0 penalty), others add their sum.
  • ASSAF: if any opponent has sum ≤ caller's sum, the caller gets +{ASSAF_PENALTY} penalty!

CARD FORMAT
-----------
  "AS" = Ace of Spades    "10H" = Ten of Hearts
  "KD" = King of Diamonds  "JO" = Joker
  Suits: S=Spades, H=Hearts, D=Diamonds, C=Clubs

EXAMPLES
--------
  "5H"          → discard 5 of Hearts (single)
  "5H 5D 5S"   → discard three 5s (set)
  "3S 4S 5S"   → discard run of 3-4-5 Spades
  "JO 4H 5H"   → Joker + 4H + 5H = run (Joker acts as 3H or 6H)
  "deck"        → draw from deck
  "yaniv"       → call Yaniv
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
            self._screen = pygame.display.set_mode((900, 520))
            pygame.display.set_caption("LLM-TeamGym · Yaniv")
            self._font  = pygame.font.SysFont("monospace", 20, bold=True)
            self._small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        active = self._active()
        top = _card_str(self._discard_pile[-1]) if self._discard_pile else "—"
        scr.blit(self._font.render(
            f"Yaniv  Phase:{self._phase}  Top:{top}  Deck:{len(self._deck)}  "
            f"Winner:{self._winner or 'ongoing'}",
            True, TEXT_CLR), (10, 8))
        for i, pid in enumerate(self.player_ids):
            col  = P_COLORS[i % len(P_COLORS)]
            hand = self.hands.get(pid, [])
            x    = 10 + i * 220
            pygame.draw.rect(scr, col, (x, 45, 205, 130), border_radius=10)
            txt_act = " ← ACTIVE" if pid == active else ""
            scr.blit(self._small.render(f"{pid}{txt_act}", True, (255,255,255)), (x+6, 50))
            scr.blit(self._small.render(f"Sum:{_hand_sum(hand)}  Penalty:{self.penalties.get(pid,0)}", True, (255,255,255)), (x+6, 70))
            hand_txt = " ".join(_card_str(c) for c in hand)
            scr.blit(self._small.render(hand_txt[:28], True, (255,255,255)), (x+6, 90))
            yaniv_ok = _hand_sum(hand) <= YANIV_THRESHOLD
            scr.blit(self._small.render("CAN YANIV!" if yaniv_ok else "", True, (255,215,0)), (x+6, 110))
        pygame.display.flip(); self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    def _obs(self) -> Dict[AgentID, Observation]:
        active = self._active()
        snap   = {
            "active": active, "phase": self._phase,
            "discard_top": _card_str(self._discard_pile[-1]) if self._discard_pile else None,
            "hand_sizes": {p: len(self.hands[p]) for p in self.player_ids},
            "penalties": dict(self.penalties),
            "done": self._done, "winner": self._winner,
        }
        result = {}
        for p in self.player_ids:
            result[p] = dict(snap)
            result[p]["your_hand"] = [_card_str(c) for c in self.hands[p]]
        return result

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
