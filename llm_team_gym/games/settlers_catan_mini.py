"""
Settlers of Catan Mini — simplified resource management + trading benchmark.

2–4 players compete to reach VICTORY_POINTS (10) first.

RESOURCES : wood, brick, sheep, wheat, ore
BUILDINGS :
  Road        (1 wood + 1 brick)                → not scored; needed for settlement expansion
  Settlement  (1 wood + 1 brick + 1 sheep + 1 wheat) → +1 VP, adds production slot
  City        (2 wheat + 3 ore)                 → upgrade settlement → +1 VP (now 2)
  Dev Card    (1 ore + 1 wheat + 1 sheep)       → 1 VP (simplified to always be VP card)

TURN FLOW (one active player at a time):
  1. Dice auto-roll → all players with matching tile numbers receive resources.
  2. TRADE phase: active player may trade or skip:
       "bank_trade N <res> for <res>"  — bank 4:1 (or 3:1 with harbor)
       "offer <pid> N <res> for N <res>" — propose to another player
       "skip_trade" — end trading phase
  3. Responder (if trade offer pending): "accept" or "reject"
  4. BUILD phase: active player builds or ends turn:
       "build road" | "build settlement" | "build city" | "buy_dev"
       "end_turn"

Teams : each player is their own team
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action, AgentID, BaseGame, Done, Info, Observation, Reward, StepResult, TeamID,
)

RESOURCES   = ('wood', 'brick', 'sheep', 'wheat', 'ore')
VICTORY_PTS = 10
MAX_ROUNDS  = 60

COSTS = {
    "road":       {"wood": 1, "brick": 1},
    "settlement": {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1},
    "city":       {"wheat": 2, "ore": 3},
    "dev":        {"ore": 1, "wheat": 1, "sheep": 1},
}

TILE_VP  = {"road": 0, "settlement": 1, "city": 2, "dev": 1}

BOARD_PRODUCTION: Dict[str, List[Tuple[str, int]]] = {
    "p0": [("wood", 5), ("brick", 6), ("sheep", 9), ("wheat", 8)],
    "p1": [("brick", 4), ("wheat", 6), ("ore", 5), ("sheep", 11)],
    "p2": [("sheep", 8), ("wood", 10), ("wheat", 9), ("ore", 4)],
    "p3": [("ore", 11), ("brick", 8), ("sheep", 3), ("wood", 6)],
}

HARBORS: Dict[str, str] = {"p0": "any", "p2": "ore"}

BG_COLOR  = (10, 30, 15)
TEXT_CLR  = (220, 230, 220)
P_COLORS  = [(70, 130, 220), (220, 70, 70), (60, 190, 80), (230, 200, 40)]
RES_CLR   = {
    "wood": (50, 140, 50), "brick": (180, 80, 40), "sheep": (160, 200, 80),
    "wheat": (230, 190, 30), "ore": (130, 130, 150),
}


def _zero_res() -> Dict[str, int]:
    return {r: 0 for r in RESOURCES}


class SettlersCatanMiniGame(BaseGame):
    """Simplified Settlers of Catan for LLM strategic trading evaluation."""

    def __init__(self, n_players: int = 3, seed: Optional[int] = None):
        assert 2 <= n_players <= 4
        self.n_players   = n_players
        self.player_ids  = tuple(f"p{i}" for i in range(n_players))
        self._seed       = seed
        self._rng        = random.Random(seed)

        self.resources:  Dict[AgentID, Dict[str, int]] = {}
        self.vp:         Dict[AgentID, int] = {}
        self.buildings:  Dict[AgentID, Dict[str, int]] = {}
        self.roads:      Dict[AgentID, int] = {}
        self.prod_slots: Dict[AgentID, List[Tuple[str, int]]] = {}
        self._last_roll: Optional[int] = None
        self._turn_idx:  int = 0
        self._phase:     str = "TRADE"
        self._pending_offer: Optional[Dict] = None
        self._trade_responded: bool = False
        self._round:     int = 0
        self._done:      bool = False
        self._winner:    Optional[str] = None
        self._step:      int = 0

        self._pygame_init = False
        self._screen = self._font = self._small = self._clock = None

    @property
    def teams(self) -> Dict[str, List[AgentID]]:
        return {p: [p] for p in self.player_ids}

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng = random.Random(self._seed)
        self.resources = {p: _zero_res() for p in self.player_ids}
        for p in self.player_ids:
            self.resources[p]["wood"]  = 1
            self.resources[p]["brick"] = 1
            self.resources[p]["sheep"] = 1
            self.resources[p]["wheat"] = 1
            self.resources[p]["ore"]   = 1
        self.vp          = {p: 2 for p in self.player_ids}
        self.buildings   = {p: {"road": 0, "settlement": 2, "city": 0, "dev": 0} for p in self.player_ids}
        self.roads       = {p: 0 for p in self.player_ids}
        self.prod_slots  = {p: list(BOARD_PRODUCTION.get(p, [("wood", 6)])) for p in self.player_ids}
        self._last_roll  = None
        self._turn_idx   = 0
        self._phase      = "TRADE"
        self._pending_offer = None
        self._trade_responded = False
        self._round      = 0
        self._done       = False
        self._winner     = None
        self._step       = 0
        self._roll_and_produce()
        return self._obs()

    def _active(self) -> AgentID:
        return self.player_ids[self._turn_idx % self.n_players]

    def _roll_and_produce(self) -> None:
        d1 = self._rng.randint(1, 6)
        d2 = self._rng.randint(1, 6)
        roll = d1 + d2
        self._last_roll = roll
        if roll == 7:
            return
        for p in self.player_ids:
            for (res, num) in self.prod_slots.get(p, []):
                if num == roll:
                    mult = 2 if self.buildings[p]["city"] > 0 else 1
                    self.resources[p][res] += mult

    def _has_resources(self, pid: AgentID, cost: Dict[str, int]) -> bool:
        return all(self.resources[pid].get(r, 0) >= n for r, n in cost.items())

    def _spend(self, pid: AgentID, cost: Dict[str, int]) -> None:
        for r, n in cost.items():
            self.resources[pid][r] -= n

    def _harbor_rate(self, pid: AgentID) -> int:
        h = HARBORS.get(pid)
        if h == "any":
            return 3
        return 4

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        rewards = {p: 0.0 for p in self.player_ids}
        infos   = {p: {}  for p in self.player_ids}
        if self._done:
            return self._obs(), rewards, self._dones(), infos

        if self._phase == "TRADE_RESPONSE":
            offer     = self._pending_offer
            responder = offer["responder"] if offer else None
            if responder and responder in actions_dict:
                action = str(actions_dict[responder]).strip().lower()
                if action == "accept":
                    op = offer["offerer"]
                    rp = offer["responder"]
                    for res, amt in offer["give"].items():
                        self.resources[op][res] -= amt
                        self.resources[rp][res] += amt
                    for res, amt in offer["receive"].items():
                        self.resources[rp][res] -= amt
                        self.resources[op][res] += amt
                    infos[op] = {"trade": "accepted"}
                    infos[rp] = {"trade": "accepted"}
                else:
                    infos[offer["offerer"]] = {"trade": "rejected"}
                self._pending_offer = None
                self._phase = "TRADE"
            return self._obs(), rewards, self._dones(), infos

        active = self._active()
        if active not in actions_dict:
            return self._obs(), rewards, self._dones(), infos
        action = str(actions_dict[active]).strip().lower()
        legal  = self.get_legal_moves(active)
        if action not in legal:
            infos[active] = {"error": f"Illegal: '{action}'"}
            return self._obs(), rewards, self._dones(), infos

        self._step += 1

        if self._phase == "TRADE":
            if action == "skip_trade":
                self._phase = "BUILD"
            elif action.startswith("bank_trade "):
                parts = action.split()
                amt   = int(parts[1])
                give_res = parts[2]
                get_res  = parts[4]
                self.resources[active][give_res] -= amt
                self.resources[active][get_res]  += 1
                infos[active] = {"bank_trade": f"{amt}{give_res}→1{get_res}"}
            elif action.startswith("offer "):
                parts    = action.split()
                target   = parts[1]
                give_amt = int(parts[2])
                give_res = parts[3]
                rcv_amt  = int(parts[5])
                rcv_res  = parts[6]
                self._pending_offer = {
                    "offerer":  active,
                    "responder": target,
                    "give":     {give_res: give_amt},
                    "receive":  {rcv_res: rcv_amt},
                }
                self._phase = "TRADE_RESPONSE"

        elif self._phase == "BUILD":
            if action == "end_turn":
                self._turn_idx += 1
                self._round    += 1
                self._phase     = "TRADE"
                self._roll_and_produce()
                if self._round >= MAX_ROUNDS:
                    self._done = True
                    best = max(self.vp.values())
                    w    = [p for p in self.player_ids if self.vp[p] == best]
                    self._winner = w[0] if len(w) == 1 else "draw"
            else:
                build_type = action.replace("build ", "").replace("buy_", "")
                if build_type == "dev":
                    build_type = "dev"
                cost = COSTS.get("road" if build_type == "road" else
                                 "settlement" if build_type == "settlement" else
                                 "city" if build_type == "city" else "dev")
                if cost and self._has_resources(active, cost):
                    self._spend(active, cost)
                    self.buildings[active][build_type] += 1
                    vp_gain = TILE_VP.get(build_type, 0)
                    if build_type == "city":
                        if self.buildings[active]["settlement"] > 0:
                            self.buildings[active]["settlement"] -= 1
                            vp_gain = 1
                    self.vp[active] += vp_gain
                    if build_type == "settlement":
                        choices = [(r, n) for r in RESOURCES for n in range(2, 13)
                                   if n not in [x[1] for x in self.prod_slots[active]]]
                        if choices:
                            slot = self._rng.choice(choices)
                            self.prod_slots[active].append(slot)
                    infos[active] = {"built": build_type, "vp": self.vp[active]}
                    if self.vp[active] >= VICTORY_PTS:
                        rewards[active]  = 1.0
                        for p in self.player_ids:
                            if p != active:
                                rewards[p] = -1.0
                        self._done   = True
                        self._winner = active

        return self._obs(), rewards, self._dones(), infos

    def get_text_state(self, agent_id: AgentID) -> str:
        active  = self._active()
        pending = self._pending_offer
        state   = {
            "agent_id":     agent_id,
            "is_your_turn": (
                active == agent_id or
                (self._phase == "TRADE_RESPONSE" and pending and pending["responder"] == agent_id)
            ),
            "active_player":  active,
            "phase":          self._phase,
            "last_dice_roll": self._last_roll,
            "round":          self._round,
            "your_resources": dict(self.resources.get(agent_id, {})),
            "your_vp":        self.vp.get(agent_id, 0),
            "your_buildings": dict(self.buildings.get(agent_id, {})),
            "your_production":[(r, n) for r, n in self.prod_slots.get(agent_id, [])],
            "your_harbor":    HARBORS.get(agent_id, "4:1 bank"),
            "all_vp":         dict(self.vp),
            "all_resource_counts": {p: sum(self.resources[p].values()) for p in self.player_ids},
            "pending_trade":  pending if (pending and (pending["offerer"] == agent_id
                                          or pending["responder"] == agent_id)) else None,
            "build_costs":    COSTS,
            "legal_moves":    self.get_legal_moves(agent_id),
            "game_over":      self._done,
            "winner":         self._winner,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self._done:
            return []
        if self._phase == "TRADE_RESPONSE":
            offer = self._pending_offer
            if offer and offer["responder"] == agent_id:
                return ["accept", "reject"]
            return []
        active = self._active()
        if active != agent_id:
            return []

        moves: List[str] = []
        res   = self.resources[agent_id]
        rate  = self._harbor_rate(agent_id)

        if self._phase == "TRADE":
            moves.append("skip_trade")
            for give_res in RESOURCES:
                if res.get(give_res, 0) >= rate:
                    for get_res in RESOURCES:
                        if get_res != give_res:
                            moves.append(f"bank_trade {rate} {give_res} for {get_res}")
            for target in self.player_ids:
                if target == agent_id:
                    continue
                for give_res in RESOURCES:
                    for give_amt in range(1, res.get(give_res, 0) + 1):
                        for get_res in RESOURCES:
                            if get_res != give_res:
                                target_res = self.resources[target]
                                for get_amt in range(1, target_res.get(get_res, 0) + 1):
                                    moves.append(f"offer {target} {give_amt} {give_res} for {get_amt} {get_res}")
            return moves[:50]

        elif self._phase == "BUILD":
            moves.append("end_turn")
            for name, cost in COSTS.items():
                if self._has_resources(agent_id, cost):
                    if name == "city" and self.buildings[agent_id]["settlement"] < 1:
                        continue
                    label = "buy_dev" if name == "dev" else f"build {name}"
                    moves.append(label)
            return moves

        return []

    def get_game_rules(self) -> str:
        return f"""
=== SETTLERS OF CATAN MINI — Game Rules ===

PLAYERS  : {self.n_players}  (each is their own team)
WIN      : First to {VICTORY_PTS} Victory Points (VP)
MAX RNDS : {MAX_ROUNDS}

RESOURCES
---------
  wood, brick, sheep, wheat, ore

BUILDINGS & COSTS
-----------------
  Road        : 1 wood + 1 brick          → no VP (enables expansion)
  Settlement  : 1 wood + 1 brick + 1 sheep + 1 wheat → +1 VP, adds production
  City        : 2 wheat + 3 ore           → upgrades settlement → +1 VP (2 total)
  Dev Card    : 1 ore + 1 wheat + 1 sheep → +1 VP (simplified Victory Point card)

PRODUCTION
----------
Each player has settlement intersections adjacent to numbered tiles.
When dice roll matches a tile number, all players with adjacent settlements receive 1
of that resource (cities produce 2). Desert tiles produce nothing.
On a roll of 7: no production (robber ignored in this simplified version).

TURN PHASES
-----------
1. Dice roll (automatic): resources distributed to all players.
2. TRADE PHASE (active player):
     "skip_trade"                         — proceed to build
     "bank_trade N <res> for <res>"       — 4:1 trade (3:1 with harbor)
     "offer <pid> N <res> for N <res>"    — propose to a player (they accept/reject)
3. BUILD PHASE (active player):
     "build road"       — build a road
     "build settlement" — build new settlement
     "build city"       — upgrade settlement to city
     "buy_dev"          — buy development card
     "end_turn"         — end your turn

HARBORS
-------
  p0 : 3:1 harbor (any resource, 3 for 1)
  p2 : ore harbor (3 ore for 1 of anything)
  Others: 4:1 bank

ACTION FORMAT EXAMPLES
----------------------
  "skip_trade"
  "bank_trade 4 wood for ore"
  "offer p1 2 wheat for 1 ore"    (p1 will then accept/reject)
  "accept" or "reject"
  "build settlement"
  "build city"
  "buy_dev"
  "end_turn"
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
            self._screen = pygame.display.set_mode((900, 540))
            pygame.display.set_caption("LLM-TeamGym · Settlers Catan Mini")
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
        scr.blit(self._font.render(
            f"Catan Mini  Roll:{self._last_roll}  Phase:{self._phase}  "
            f"Round:{self._round}  Winner:{self._winner or 'TBD'}",
            True, TEXT_CLR), (10, 8))
        for i, pid in enumerate(self.player_ids):
            col = P_COLORS[i % len(P_COLORS)]
            x   = 10 + i * 220
            pygame.draw.rect(scr, col, (x, 40, 205, 160), border_radius=10)
            act = " ←" if pid == active else ""
            scr.blit(self._small.render(f"{pid} VP={self.vp.get(pid,0)}{act}", True, (255,255,255)), (x+5, 46))
            res = self.resources.get(pid, {})
            for j, r in enumerate(RESOURCES):
                scr.blit(self._small.render(f"{r[:2]}:{res.get(r,0)}", True, (255,255,255)), (x+5 + (j%3)*65, 68 + (j//3)*18))
            bld = self.buildings.get(pid, {})
            scr.blit(self._small.render(
                f"set:{bld.get('settlement',0)} cit:{bld.get('city',0)} rd:{bld.get('road',0)} dev:{bld.get('dev',0)}",
                True, (255,255,255)), (x+5, 116))
        pending = self._pending_offer
        if pending:
            offer_txt = (f"OFFER: {pending['offerer']} → {pending['responder']}: "
                         f"{pending['give']} for {pending['receive']}")
            scr.blit(self._small.render(offer_txt, True, (255, 215, 0)), (10, 215))
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
            "phase":    self._phase,
            "active":   self._active(),
            "roll":     self._last_roll,
            "round":    self._round,
            "vp":       dict(self.vp),
            "resource_counts": {p: sum(self.resources[p].values()) for p in self.player_ids},
            "done":     self._done,
            "winner":   self._winner,
        }
        result = {}
        for p in self.player_ids:
            result[p] = dict(snap)
            result[p]["your_resources"] = dict(self.resources.get(p, {}))
        return result

    def _dones(self) -> Dict[AgentID, Done]:
        d = {p: self._done for p in self.player_ids}
        d["__all__"] = self._done
        return d
