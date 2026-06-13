"""
2v2 Team Fish — a grid-based tile-collection game inspired by "Hey, That's My Fish!"

Rules summary
-------------
- 4 agents: A1, A2 (Team A) vs B1, B2 (Team B) on an N×M grid.
- Each cell holds a fish value (1–3) or is empty (destroyed/water).
- On your turn, move your token in one of four cardinal directions any number
  of steps (like a rook in chess), stopping before occupied cells or grid edges.
- The cell you LEAVE is destroyed (becomes water, value 0).
- You score the fish value of the cell you LAND on.
- Teammates share a combined team score.
- A player is eliminated when they cannot move (surrounded by water/edges).
- The game ends when ALL players are eliminated.
- The team with the higher combined score wins.

Turn order: A1 → B1 → A2 → B2 → A1 → …  (interleaved so teams alternate fairly)

Action format:  "<direction> <steps>"
  direction ∈ {up, down, left, right}
  steps     ∈ {1, 2, 3, …}  (any valid number of steps in that direction)

Example: "right 2"  moves the agent 2 cells to the right.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from llm_team_gym.core.base_game import (
    Action,
    AgentID,
    BaseGame,
    Done,
    Info,
    Observation,
    Reward,
    StepResult,
    TeamID,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = ("up", "down", "left", "right")
DELTA: Dict[str, Tuple[int, int]] = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}

TEAM_A_AGENTS = ("A1", "A2")
TEAM_B_AGENTS = ("B1", "B2")
TURN_ORDER: Tuple[AgentID, ...] = ("A1", "B1", "A2", "B2")

# Pygame visual constants — only imported when rendering
CELL_SIZE   = 90          # px per grid cell
MARGIN      = 4           # px gap between cells
INFO_HEIGHT = 120         # px for scoreboard area at the bottom
FPS         = 30

TEAM_COLORS: Dict[str, Tuple[int, int, int]] = {
    "team_A": (70, 130, 220),   # blue
    "team_B": (220, 80,  70),   # red
}
AGENT_TEXT_COLOR = (255, 255, 255)
BG_COLOR         = (30,  30,  35)
WATER_COLOR      = (20,  20,  100)
CELL_COLORS      = {
    0: WATER_COLOR,
    1: (80,  140, 80),
    2: (120, 180, 60),
    3: (180, 210, 40),
}
FONT_COLOR       = (230, 230, 230)
SCORE_BG         = (45,  45,  50)


# ---------------------------------------------------------------------------
# TeamFishGame
# ---------------------------------------------------------------------------

class TeamFishGame(BaseGame):
    """
    2v2 Team Fish grid-collection game.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (default 6×6).
    max_fish : int
        Maximum fish value per cell (1–max_fish, uniform random).
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        max_fish: int = 3,
        seed: Optional[int] = None,
    ):
        self.rows = rows
        self.cols = cols
        self.max_fish = max_fish
        self._seed = seed
        self._rng = random.Random(seed)

        # Pygame state — lazily initialised on first render() call
        self._pygame_init = False
        self._screen = None
        self._font_large = None
        self._font_small = None
        self._clock = None

        # Game state (populated by reset())
        self.grid: List[List[int]] = []
        self.positions: Dict[AgentID, Tuple[int, int]] = {}
        self.scores: Dict[AgentID, float] = {}
        self.eliminated: Dict[AgentID, bool] = {}
        self._turn_index: int = 0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # BaseGame interface
    # ------------------------------------------------------------------

    @property
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        return {
            "team_A": list(TEAM_A_AGENTS),
            "team_B": list(TEAM_B_AGENTS),
        }

    def reset(self) -> Dict[AgentID, Observation]:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._turn_index = 0

        # Build grid: random fish values 1..max_fish
        self.grid = [
            [self._rng.randint(1, self.max_fish) for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

        # Place agents at four corners (deterministic, ensures non-overlap)
        corners = [
            (0,              0             ),
            (0,              self.cols - 1 ),
            (self.rows - 1,  0             ),
            (self.rows - 1,  self.cols - 1 ),
        ]
        self._rng.shuffle(corners)
        self.positions = {
            "A1": corners[0],
            "A2": corners[1],
            "B1": corners[2],
            "B2": corners[3],
        }

        # Zero out starting cells (no fish earned for starting position)
        for pos in self.positions.values():
            self.grid[pos[0]][pos[1]] = 0

        self.scores = {a: 0.0 for a in TURN_ORDER}
        self.eliminated = {a: False for a in TURN_ORDER}

        return self._build_observations()

    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        """
        Turn-based: only the current active agent should supply an action.

        actions_dict must contain exactly the active agent's id. Other keys are
        ignored so that the runner's generic "all agents with legal moves" loop
        works without modification.
        """
        active = self._active_agent()
        rewards: Dict[AgentID, Reward] = {a: 0.0 for a in TURN_ORDER}
        infos: Dict[AgentID, Info] = {a: {} for a in TURN_ORDER}

        if active is not None and active in actions_dict:
            action = actions_dict[active]
            fish_earned, info = self._apply_action(active, action)
            rewards[active] = fish_earned
            infos[active] = info

        self._advance_turn()
        self._step_count += 1

        observations = self._build_observations()
        dones = self._compute_dones()
        return observations, rewards, dones, infos

    def get_text_state(self, agent_id: AgentID) -> str:
        pos = self.positions[agent_id]
        team_id = self.agent_to_team()[agent_id]
        teammates = self.teammates_of(agent_id)
        opponents = self.opponents_of(agent_id)

        team_score = sum(self.scores[a] for a in self.teams[team_id])
        opp_team_id = "team_B" if team_id == "team_A" else "team_A"
        opp_score = sum(self.scores[a] for a in self.teams[opp_team_id])

        state = {
            "agent_id": agent_id,
            "team": team_id,
            "position": {"row": pos[0], "col": pos[1]},
            "your_score": self.scores[agent_id],
            "team_score": team_score,
            "opponent_team_score": opp_score,
            "is_my_turn": self._active_agent() == agent_id,
            "active_agent": self._active_agent(),
            "step": self._step_count,
            "legal_moves": self.get_legal_moves(agent_id),
            "grid": self.grid,
            "agent_positions": {a: {"row": r, "col": c} for a, (r, c) in self.positions.items()},
            "eliminated": self.eliminated,
            "teammates": teammates,
            "opponents": opponents,
        }
        return json.dumps(state, indent=2)

    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        if self.eliminated.get(agent_id, True):
            return []
        if self._active_agent() != agent_id:
            return []

        pos = self.positions[agent_id]
        occupied = set(self.positions.values())
        moves = []

        for direction, (dr, dc) in DELTA.items():
            r, c = pos
            for steps in range(1, max(self.rows, self.cols)):
                r2, c2 = r + dr * steps, c + dc * steps
                if not (0 <= r2 < self.rows and 0 <= c2 < self.cols):
                    break
                if (r2, c2) in occupied:
                    break
                if self.grid[r2][c2] == 0:
                    # water cell — can't land here but CAN pass through? No.
                    # Destroyed cells block movement (they are "missing tiles").
                    break
                moves.append(f"{direction} {steps}")
        return moves

    def get_game_rules(self) -> str:
        return """
=== 2v2 TEAM FISH — Game Rules ===

OBJECTIVE
---------
Your team (2 agents) must collectively collect more fish than the opposing team.
The game ends when all agents are eliminated (unable to move).

TEAMS
-----
- Team A: agents A1, A2  (share combined score)
- Team B: agents B1, B2  (share combined score)
- Turn order: A1 → B1 → A2 → B2 → A1 → …

THE GRID
--------
- The board is a 2D grid of cells, each holding 1–3 fish.
- A cell becomes WATER (value 0) when an agent leaves it. Water cells block movement.
- Agents start at the four corners; starting cells are pre-destroyed (no fish).

MOVEMENT
--------
- On your turn you move like a chess rook: any number of steps in one
  cardinal direction (up/down/left/right).
- You CANNOT pass through or land on: another agent, a water cell, or outside the grid.
- The cell you were standing on is DESTROYED after you move.

SCORING
-------
- You earn the fish value of the cell you LAND on.
- Both teammates' fish are added together for the team score.

ELIMINATION
-----------
- If you have no legal moves on your turn, you are ELIMINATED (score freezes).
- Eliminated agents are removed from turn order.

GAME END
--------
- When ALL agents are eliminated, the team with the higher combined score WINS.
- In case of a tie, both teams draw.

ACTION FORMAT
-------------
  "<direction> <steps>"

  direction : up | down | left | right
  steps     : positive integer (how many cells to move)

  Examples:
    "right 2"   — move 2 cells to the right
    "up 1"      — move 1 cell upward
    "down 3"    — move 3 cells downward

  Always choose from the provided list of legal moves exactly as shown.
""".strip()

    # ------------------------------------------------------------------
    # Pygame rendering
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[Any]:
        if mode != "human":
            return None

        try:
            import pygame
        except ImportError:
            print("[TeamFish] pygame not installed — skipping render.")
            return None

        if not self._pygame_init:
            pygame.init()
            w = self.cols * (CELL_SIZE + MARGIN) + MARGIN
            h = self.rows * (CELL_SIZE + MARGIN) + MARGIN + INFO_HEIGHT
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("LLM-TeamGym · 2v2 Team Fish")
            self._font_large = pygame.font.SysFont("monospace", 22, bold=True)
            self._font_small = pygame.font.SysFont("monospace", 14)
            self._clock = pygame.time.Clock()
            self._pygame_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        screen = self._screen
        screen.fill(BG_COLOR)

        # Agent position lookup for rendering
        pos_to_agent: Dict[Tuple[int, int], AgentID] = {v: k for k, v in self.positions.items()}
        a2t = self.agent_to_team()

        # Draw grid cells
        for r in range(self.rows):
            for c in range(self.cols):
                x = MARGIN + c * (CELL_SIZE + MARGIN)
                y = MARGIN + r * (CELL_SIZE + MARGIN)
                fish_val = self.grid[r][c]
                color = CELL_COLORS.get(fish_val, WATER_COLOR)
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), border_radius=8)

                # Fish value label
                if fish_val > 0:
                    label = self._font_large.render(str(fish_val), True, (30, 30, 30))
                    screen.blit(label, (x + CELL_SIZE // 2 - label.get_width() // 2,
                                       y + CELL_SIZE // 2 - label.get_height() // 2))

                # Agent token
                if (r, c) in pos_to_agent:
                    agent_id = pos_to_agent[(r, c)]
                    team_id  = a2t[agent_id]
                    token_color = TEAM_COLORS[team_id]
                    cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                    pygame.draw.circle(screen, token_color, (cx, cy), CELL_SIZE // 3)
                    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), CELL_SIZE // 3, 2)
                    # Agent label inside token
                    lbl = self._font_small.render(agent_id, True, AGENT_TEXT_COLOR)
                    screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

        # Scoreboard panel
        board_h = self.rows * (CELL_SIZE + MARGIN) + MARGIN
        pygame.draw.rect(screen, SCORE_BG, (0, board_h, screen.get_width(), INFO_HEIGHT))

        def team_summary(team_id: str, x_offset: int) -> None:
            agents = self.teams[team_id]
            team_score = int(sum(self.scores[a] for a in agents))
            color = TEAM_COLORS[team_id]
            label = self._font_large.render(f"{team_id.upper()}  Score: {team_score}", True, color)
            screen.blit(label, (x_offset, board_h + 10))
            for i, a in enumerate(agents):
                elim = "  [OUT]" if self.eliminated.get(a) else ""
                detail = self._font_small.render(
                    f"  {a}: {int(self.scores[a])} pts  pos={self.positions[a]}{elim}",
                    True, FONT_COLOR,
                )
                screen.blit(detail, (x_offset, board_h + 40 + i * 20))

        mid = screen.get_width() // 2
        team_summary("team_A", 20)
        team_summary("team_B", mid + 20)

        # Step / active agent
        active = self._active_agent()
        step_lbl = self._font_small.render(
            f"Step: {self._step_count}   Active: {active or 'GAME OVER'}",
            True, FONT_COLOR,
        )
        screen.blit(step_lbl, (mid - step_lbl.get_width() // 2, board_h + INFO_HEIGHT - 24))

        pygame.display.flip()
        self._clock.tick(FPS)
        return None

    def close(self) -> None:
        if self._pygame_init:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._pygame_init = False
            self._screen = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_agent(self) -> Optional[AgentID]:
        """Return the agent whose turn it is, skipping eliminated agents."""
        for _ in range(len(TURN_ORDER)):
            candidate = TURN_ORDER[self._turn_index % len(TURN_ORDER)]
            if not self.eliminated.get(candidate, False):
                return candidate
            self._turn_index += 1
        return None  # all eliminated

    def _advance_turn(self) -> None:
        self._turn_index = (self._turn_index + 1) % len(TURN_ORDER)
        # Skip over eliminated agents
        for _ in range(len(TURN_ORDER)):
            candidate = TURN_ORDER[self._turn_index % len(TURN_ORDER)]
            if not self.eliminated.get(candidate, False):
                return
            self._turn_index = (self._turn_index + 1) % len(TURN_ORDER)

    def _apply_action(self, agent_id: AgentID, action: Action) -> Tuple[float, Info]:
        """Parse and execute the action. Returns (fish_earned, info_dict)."""
        legal = self.get_legal_moves(agent_id)
        if action not in legal:
            # Invalid action — agent stays, earns nothing, cell is NOT destroyed.
            return 0.0, {"error": f"Illegal action '{action}'", "legal_moves": legal}

        parts = str(action).split()
        direction = parts[0]
        steps = int(parts[1])
        dr, dc = DELTA[direction]

        old_r, old_c = self.positions[agent_id]
        new_r = old_r + dr * steps
        new_c = old_c + dc * steps

        # Destroy the cell we're leaving
        self.grid[old_r][old_c] = 0

        # Collect fish from landing cell
        fish_earned = float(self.grid[new_r][new_c])
        self.grid[new_r][new_c] = 0  # land cell becomes occupied (will be destroyed on next move)
        # Actually: the rule is the cell is destroyed when you LEAVE it,
        # so restore the fish visually until they move again.
        # We store fish=0 to mark "occupied", and restore from score.
        # Simpler: keep grid[new_r][new_c] = 0 meaning "occupied/destroyed" —
        # the score is already captured in self.scores.
        self.positions[agent_id] = (new_r, new_c)
        self.scores[agent_id] += fish_earned

        info = {
            "moved_from": (old_r, old_c),
            "moved_to":   (new_r, new_c),
            "fish_earned": fish_earned,
            "direction": direction,
            "steps": steps,
        }
        return fish_earned, info

    def _build_observations(self) -> Dict[AgentID, Observation]:
        obs = {}
        for agent_id in TURN_ORDER:
            obs[agent_id] = {
                "grid": [row[:] for row in self.grid],
                "positions": dict(self.positions),
                "scores": dict(self.scores),
                "eliminated": dict(self.eliminated),
                "active_agent": self._active_agent(),
                "step": self._step_count,
            }
        return obs

    def _compute_dones(self) -> Dict[AgentID, Done]:
        """Eliminate agents with no legal moves; signal game-over when all done."""
        # First pass: check who is newly stuck
        for agent_id in TURN_ORDER:
            if self.eliminated.get(agent_id):
                continue
            # Temporarily set as active to compute legal moves
            saved_turn = self._turn_index
            # find agent's turn index
            for idx, a in enumerate(TURN_ORDER):
                if a == agent_id:
                    self._turn_index = idx
                    break
            if not self.get_legal_moves(agent_id):
                self.eliminated[agent_id] = True
            self._turn_index = saved_turn

        all_done = all(self.eliminated.values())
        dones: Dict[AgentID, Done] = {a: self.eliminated.get(a, False) for a in TURN_ORDER}
        dones["__all__"] = all_done
        return dones
