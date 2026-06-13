"""
Bridge/Wist — trick-taking card game with bidding (simplified Contract Bridge).

4 players: N (North), E (East), S (South), W (West)
Teams    : NS (North-South) vs EW (East-West)

BIDDING PHASE
  Players bid in turn (N→E→S→W→...) to declare a contract.
  Bid format: "<level><suit>"  e.g. "1NT", "2H", "3S", "4C", "7NT"
  Level : 1–7  (number of tricks ABOVE 6; level 1 = 7 tricks needed)
  Suit  : C < D < H < S < NT  (higher = higher contract)
  Special: "pass" — skip this bid
  Bid must be higher than the previous highest bid.
  Bidding ends when 3 consecutive players pass.

PLAY PHASE
  The team that won the bidding must make their contract.
  Trump suit = bid suit (or no trump if NT).
  The DECLARER (who first named the trump suit for their team) leads.
  Follow-suit is MANDATORY if you have a card of the led suit.
  Trick winner leads next.
  After 13 tricks, count tricks per team.

WIN
  Declaring team: reward +1 if tricks ≥ (level+6), else −1.
  Defending team: reward +1 if declaring team fails, else −1.

Action format (BIDDING): "1NT" | "2H" | "3S" | "4C" | "pass"
Action format (PLAY)   : card string e.g. "AH", "5S", "10D", "2C"
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Set, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

SUITS   = ('C', 'D', 'H', 'S')
RANKS   = ('2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A')
SUIT_NT = 'NT'
PLAYERS_ORDER = ('N', 'E', 'S', 'W')
PARTNERS      = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
TEAM_OF       = {'N': 'NS', 'S': 'NS', 'E': 'EW', 'W': 'EW'}

RANK_IDX: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}
SUIT_IDX: Dict[str, int] = {s: i for i, s in enumerate(SUITS + (SUIT_NT,))}

BG_COLOR  = (10, 40, 10)
TEXT_CLR  = (220, 230, 220)
CARD_BG   = (255, 255, 245)
CARD_RED  = (200, 20, 20)
CARD_BLK  = (20, 20, 20)
P_COLORS: Dict[str, Tuple] = {
    'N': (70, 130, 220), 'E': (220, 70, 70),
    'S': (60, 190, 80),  'W': (230, 200, 40),
}


def _build_deck() -> List[Tuple[str, str]]:
    return [(r, s) for s in SUITS for r in RANKS]


def _card_str(card: Tuple[str, str]) -> str:
    return f"{card[0]}{card[1]}"


def _parse_card(token: str) -> Optional[Tuple[str, str]]:
    t = token.strip().upper()
    for s in SUITS:
        if t.endswith(s):
            r = t[:-len(s)]
            if r in RANKS:
                return (r, s)
    return None


def _bid_rank(bid: str) -> Tuple[int, int]:
    if bid == "pass":
        return (-1, -1)
    level = int(bid[0])
    suit  = bid[1:]
    return (level, SUIT_IDX.get(suit, -1))


def _tricks_needed(level: int) -> int:
    return level + 6


def _trick_winner(cards_played: List[Tuple[AgentID, Tuple[str, str]]],
                  led_suit: str, trump: Optional[str]) -> AgentID:
    led_cards   = [(pid, c) for pid, c in cards_played if c[1] == led_suit]
    trump_cards = [(pid, c) for pid, c in cards_played if trump and c[1] == trump and c[1] != led_suit]
    if trump_cards:
        return max(trump_cards, key=lambda x: RANK_IDX[x[1][0]])[0]
    return max(led_cards, key=lambda x: RANK_IDX[x[1][0]])[0]


class BridgeWistGame(BaseGame):
    """Simplified Contract Bridge: bidding + 13 trick-play rounds."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng  = random.Random(seed)

        self.hands:         Dict[AgentID, List[Tuple[str, str]]] = {}
        self._phase:        str = "BIDDING"
        self._bid_turn_idx: int = 0
        self._bids:         List[Tuple[AgentID, str]] = []
        self._contract:     Optional[Tuple[AgentID, str]] = None
        self._trump:        Optional[str] = None
        self._declarer:     Optional[AgentID] = None
        self._defending_team: Optional[str] = None
        self._play_turn_idx:  int = 0
        self._trick:        List[Tuple[AgentID, Tuple[str, str]]] = []
        self._led_suit:     Optional[str] = None
        self._tricks:       Dict[str, int] = {"NS": 0, "EW": 0}
        self._tricks_played:int = 0
        self._consec_passes:int = 0
        self._done:         bool = False
        self._winner:       Optional[str] = None
        self._step:         int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[str, List[AgentID]]:
        return {"NS": ["N", "S"], "EW": ["E", "W"]}

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng = random.Random(self._seed)
        deck      = _build_deck()
        self._rng.shuffle(deck)
        self.hands = {}
        for i, p in enumerate(PLAYERS_ORDER):
            self.hands[p] = deck[i*13:(i+1)*13]
        self._phase        = "BIDDING"
        self._bid_turn_idx = 0
        self._bids         = []
        self._contract     = None
        self._trump        = None
        self._declarer     = None
        self._defending_team = None
        self._play_turn_idx  = 0
        self._trick        = []
        self._led_suit     = None
        self._tricks       = {"NS": 0, "EW": 0}
        self._tricks_played= 0
        self._consec_passes= 0
        self._done         = False
        self._winner       = None
        self._step         = 0
        return self._obs()

    def _bid_active(self) -> AgentID:
        return PLAYERS_ORDER[self._bid_turn_idx % 4]

    def _play_active(self) -> AgentID:
        return PLAYERS_ORDER[self._play_turn_idx % 4]

    def _highest_bid(self) -> Optional[str]:
        real = [(p, b) for p, b in self._bids if b != "pass"]
        return real[-1][1] if real else None

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in PLAYERS_ORDER}
        infos   = {p: {}  for p in PLAYERS_ORDER}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        if self._phase == "BIDDING":
            active = self._bid_active()
            if active not in actions_dict:
                return self._obs(), rewards, self._dones(), infos
            bid    = str(actions_dict[active]).strip().upper()
            legal  = self.get_legal_moves(active)
            if bid.lower() not in [m.lower() for m in legal]:
                infos[active] = {"error": f"Illegal bid '{bid}'"}
                return self._obs(), rewards, self._dones(), infos
            bid_norm = bid.lower() if bid.lower() == "pass" else bid
            self._bids.append((active, bid_norm))
            if bid_norm == "pass":
                self._consec_passes += 1
            else:
                self._consec_passes  = 0
            self._bid_turn_idx += 1

            if self._consec_passes >= 3 and len(self._bids) >= 4:
                self._start_play()
            elif self._consec_passes >= 4:
                self._done   = True
                self._winner = "draw"

        elif self._phase == "PLAY":
            active = self._play_active()
            if active not in actions_dict:
                return self._obs(), rewards, self._dones(), infos
            card_str = str(actions_dict[active]).strip().upper()
            card = _parse_card(card_str)
            legal = self.get_legal_moves(active)
            if card_str.lower() not in [m.lower() for m in legal]:
                infos[active] = {"error": f"Illegal card '{card_str}'"}
                return self._obs(), rewards, self._dones(), infos

            if card is None:
                return self._obs(), rewards, self._dones(), infos
            self.hands[active].remove(card)
            self._trick.append((active, card))
            if not self._led_suit:
                self._led_suit = card[1]
            self._play_turn_idx += 1

            if len(self._trick) == 4:
                winner_pid = _trick_winner(self._trick, self._led_suit, self._trump)
                win_team   = TEAM_OF[winner_pid]
                self._tricks[win_team] += 1
                self._tricks_played    += 1
                self._trick    = []
                self._led_suit = None
                self._play_turn_idx = PLAYERS_ORDER.index(winner_pid)
                if self._tricks_played == 13:
                    self._resolve(rewards)

        self._step += 1
        return self._obs(), rewards, self._dones(), infos

    def _start_play(self) -> None:
        highest = self._highest_bid()
        if not highest:
            self._done = True; self._winner = "draw"; return
        level   = int(highest[0])
        suit_part = highest[1:]
        self._trump = None if suit_part == "NT" else suit_part
        dec_team_bids = [(p, b) for p, b in self._bids if b != "pass" and
                         any(b.endswith(suit_part) for _ in [0])]
        for p, b in self._bids:
            if b != "pass" and b.endswith(suit_part):
                self._declarer = p
                break
        if not self._declarer:
            real = [(p, b) for p, b in self._bids if b != "pass"]
            self._declarer = real[-1][0] if real else PLAYERS_ORDER[0]
        self._defending_team = "NS" if self._declarer in ("E", "W") else "EW"
        self._contract  = (self._declarer, highest)
        lead_player     = PLAYERS_ORDER[(PLAYERS_ORDER.index(self._declarer) + 1) % 4]
        self._play_turn_idx = PLAYERS_ORDER.index(lead_player)
        self._phase = "PLAY"

    def _resolve(self, rewards: Dict[AgentID, float]) -> None:
        if not self._contract:
            self._done = True; self._winner = "draw"; return
        declarer, bid  = self._contract
        dec_team       = TEAM_OF[declarer]
        def_team       = "EW" if dec_team == "NS" else "NS"
        level          = int(bid[0])
        tricks_needed  = _tricks_needed(level)
        dec_tricks     = self._tricks[dec_team]
        _teams = {"NS": ["N", "S"], "EW": ["E", "W"]}
        if dec_tricks >= tricks_needed:
            for p in _teams[dec_team]: rewards[p] =  1.0
            for p in _teams[def_team]: rewards[p] = -1.0
            self._winner = dec_team
        else:
            for p in _teams[def_team]: rewards[p] =  1.0
            for p in _teams[dec_team]: rewards[p] = -1.0
            self._winner = def_team
        self._done = True

    def get_text_state(self, agent_id: AgentID) -> str:
        active  = self._bid_active() if self._phase == "BIDDING" else self._play_active()
        partner = PARTNERS[agent_id]
        team    = TEAM_OF[agent_id]
        state   = {
            "agent_id":     agent_id,
            "team":         team,
            "partner":      partner,
            "is_your_turn": active == agent_id,
            "phase":        self._phase,
            "your_hand":    [_card_str(c) for c in sorted(self.hands.get(agent_id, []),
                                                            key=lambda c: (SUITS.index(c[1]), RANK_IDX[c[0]]))],
            "bids_so_far":  [(p, b) for p, b in self._bids],
            "highest_bid":  self._highest_bid(),
            "contract":     (self._declarer, self._contract[1]) if self._contract else None,
            "trump_suit":   self._trump,
            "declarer":     self._declarer,
            "current_trick":[f"{_card_str(c)} by {pid}" for pid, c in self._trick],
            "led_suit":     self._led_suit,
            "tricks_won":   dict(self._tricks),
            "tricks_played":self._tricks_played,
            "tricks_needed":_tricks_needed(int(self._contract[1][0])) if self._contract else None,
            "legal_moves":  self.get_legal_moves(agent_id),
            "step":         self._step,
            "game_over":    self._done,
            "winner":       self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        if self._phase == "BIDDING":
            active = self._bid_active()
            if active != agent_id:
                return []
            highest = self._highest_bid()
            moves   = ["pass"]
            for level in range(1, 8):
                for suit in ('C', 'D', 'H', 'S', 'NT'):
                    bid = f"{level}{suit}"
                    if not highest or _bid_rank(bid) > _bid_rank(highest):
                        moves.append(bid)
            return moves

        elif self._phase == "PLAY":
            active = self._play_active()
            if active != agent_id:
                return []
            hand = self.hands.get(agent_id, [])
            if self._led_suit:
                follow = [c for c in hand if c[1] == self._led_suit]
                playable = follow if follow else hand
            else:
                playable = hand
            return [_card_str(c) for c in playable]

        return []

    def get_game_rules(self) -> str:
        return """
=== BRIDGE/WIST — Simplified Contract Bridge Rules ===

PLAYERS : N (North), E (East), S (South), W (West)
TEAMS   : NS (North-South)  vs  EW (East-West)
CARDS   : Standard 52-card deck, 13 cards each

BIDDING PHASE
-------------
Players bid in clockwise order: N→E→S→W→N→...
A bid: "<level><suit>"  (e.g. "2H" = level 2, trump Hearts)
  Level: 1–7 (tricks needed = level + 6; so "1" means 7 tricks)
  Suits: C (lowest) < D < H < S < NT (No Trump, highest)
Each bid MUST be higher than the previous (by level, then by suit rank).
  "pass" = no bid this round
Bidding ends: 3 consecutive passes after at least one real bid.

CONTRACT
--------
The last non-pass bid is the contract.
The player who first named the trump suit for their side is the DECLARER.
The opponent to Declarer's left leads first.

PLAY PHASE (13 tricks)
-----------------------
On your turn, play a card from your hand.
  • If you have a card matching the LED SUIT: you MUST play it (follow suit).
  • If you have no card in the led suit: play any card.

TRUMP: if a trump card is played (and led suit is not trump), the highest
trump beats all non-trump cards.

Trick winner leads next. Track tricks per team.

WIN CONDITION
-------------
Declaring team needs (level + 6) tricks.
  • Makes contract → declaring team wins (+1 reward)
  • Fails contract → defending team wins (+1 reward)

CARD FORMAT
-----------
  "AH" = Ace of Hearts   "KS" = King of Spades
  "10C" = Ten of Clubs   "2D" = Two of Diamonds
  Suits: C=Clubs, D=Diamonds, H=Hearts, S=Spades

BID FORMAT
----------
  "1NT" = 1 No Trump (7 tricks, no trump)
  "3H"  = 3 Hearts (9 tricks, Hearts trump)
  "7S"  = Grand Slam (13 tricks, Spades trump)
  "pass"= no bid
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
            self._screen = pygame.display.set_mode((900, 560))
            pygame.display.set_caption("LLM-TeamGym · Bridge/Wist")
            self._font  = pygame.font.SysFont("monospace", 18, bold=True)
            self._small = pygame.font.SysFont("monospace", 13)
            self._clock = pygame.time.Clock()
            self._pygame_init = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return
        scr = self._screen
        scr.fill(BG_COLOR)
        active = self._bid_active() if self._phase == "BIDDING" else self._play_active()
        contract_txt = f"{self._contract[0]}:{self._contract[1]}" if self._contract else "TBD"
        scr.blit(self._font.render(
            f"Bridge  Phase:{self._phase}  Contract:{contract_txt}  "
            f"Trump:{self._trump or 'NT'}  Winner:{self._winner or 'ongoing'}",
            True, TEXT_CLR), (10, 8))
        scr.blit(self._small.render(
            f"Tricks NS:{self._tricks['NS']}  EW:{self._tricks['EW']}  "
            f"Played:{self._tricks_played}/13",
            True, TEXT_CLR), (10, 32))
        scr.blit(self._small.render(
            "Bids: " + "  ".join(f"{p}:{b}" for p, b in self._bids[-8:]),
            True, (180, 200, 180)), (10, 50))
        positions = {'N': (400, 80), 'E': (720, 280), 'S': (400, 460), 'W': (80, 280)}
        for pid, (x, y) in positions.items():
            col  = P_COLORS[pid]
            hand = [_card_str(c) for c in self.hands.get(pid, [])]
            act  = " ←" if pid == active else ""
            pygame.draw.rect(scr, col, (x-70, y-35, 140, 65), border_radius=8)
            scr.blit(self._small.render(f"{pid}{act}", True, (255,255,255)), (x-65, y-30))
            scr.blit(self._small.render(f"({len(hand)}) {' '.join(hand[:7])}", True, (255,255,255)), (x-65, y-10))
        trick_txt = "  ".join(f"{pid}:{_card_str(c)}" for pid, c in self._trick)
        scr.blit(self._small.render(f"Current trick: {trick_txt}", True, TEXT_CLR), (10, 530))
        pygame.display.flip(); self._clock.tick(30)

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame; pygame.quit()
            except Exception:
                pass
            self._pygame_init = False

    def _obs(self) -> Dict[AgentID, Observation]:
        active = self._bid_active() if self._phase == "BIDDING" else self._play_active()
        snap   = {
            "phase":        self._phase,
            "active":       active,
            "contract":     (self._declarer, self._contract[1]) if self._contract else None,
            "trump":        self._trump,
            "tricks":       dict(self._tricks),
            "led_suit":     self._led_suit,
            "done":         self._done,
            "winner":       self._winner,
        }
        result = {}
        for p in PLAYERS_ORDER:
            result[p] = dict(snap)
            result[p]["your_hand"] = [_card_str(c) for c in self.hands.get(p, [])]
        return result

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in PLAYERS_ORDER}
        d["__all__"] = self._done
        return d
