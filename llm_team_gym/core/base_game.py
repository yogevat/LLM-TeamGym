"""
Abstract base class for all games in LLM-TeamGym.

Supports turn-based and simultaneous-turn paradigms, team configurations,
text-friendly state representations, and optional Pygame rendering.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple


# Type aliases
AgentID = str
TeamID = str
Action = Any
Observation = Any
Reward = float
Done = bool
Info = Dict[str, Any]

StepResult = Tuple[
    Dict[AgentID, Observation],
    Dict[AgentID, Reward],
    Dict[AgentID, Done],
    Dict[AgentID, Info],
]


class BaseGame(abc.ABC):
    """
    Abstract base class every game in LLM-TeamGym must implement.

    Paradigm support:
      - Turn-based: only the active agent's id appears in `actions_dict` each step.
      - Simultaneous: all (or a subset of) agents supply actions in the same step.
      - Team play: agents grouped via the `teams` property; rewards are shared or
        aggregated at the env level.

    Subclass contract
    -----------------
    At minimum you must implement:
        reset, step, get_text_state, get_legal_moves, get_game_rules, teams
    Optionally override:
        render, close
    """

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(self) -> Dict[AgentID, Observation]:
        """
        Reset game to its initial state.

        Returns
        -------
        observations : dict
            Initial observations keyed by agent_id.
        """

    @abc.abstractmethod
    def step(self, actions_dict: Dict[AgentID, Action]) -> StepResult:
        """
        Advance the game by one step.

        Handles both turn-based games (actions_dict has exactly one entry for the
        active agent) and simultaneous-turn games (actions_dict has one entry per
        active agent this round).

        Parameters
        ----------
        actions_dict : dict
            Mapping of agent_id -> action for every agent acting this step.

        Returns
        -------
        observations : dict[AgentID, Observation]
            Next observations for each agent.
        rewards : dict[AgentID, Reward]
            Per-agent rewards earned this step.
        dones : dict[AgentID, Done]
            True when an agent's episode has ended; '__all__' key signals game over.
        infos : dict[AgentID, Info]
            Arbitrary per-agent diagnostic information.
        """

    @abc.abstractmethod
    def get_text_state(self, agent_id: AgentID) -> str:
        """
        Return a structured, text-friendly representation of the game state
        from the perspective of *agent_id*.

        Implementation notes
        --------------------
        - Omit information the agent cannot see (hidden cards, fog of war, etc.).
        - Format as JSON or a clearly labelled human-readable string so an LLM
          can parse it from a system/user prompt without additional processing.
        - Include the agent's own position, score, legal moves, and any team info.
        """

    @abc.abstractmethod
    def get_legal_moves(self, agent_id: AgentID) -> List[Action]:
        """
        Return all currently legal actions for *agent_id*.

        Used to constrain LLM prompts so the model cannot hallucinate
        out-of-bounds or otherwise invalid moves.
        """

    @abc.abstractmethod
    def get_game_rules(self) -> str:
        """
        Return a complete, plain-English description of the game suitable for
        injection into an LLM system prompt.

        Must cover:
          - Objective / win condition
          - Action format and legal action space
          - Scoring system
          - Turn structure (turn-based vs. simultaneous)
          - Team mechanics (if any)
          - Edge cases / special rules
        """

    @property
    @abc.abstractmethod
    def teams(self) -> Dict[TeamID, List[AgentID]]:
        """
        Return the team configuration.

        Examples
        --------
        Solo:      {"team_A": ["agent_0"], "team_B": ["agent_1"]}
        2v2:       {"team_A": ["A1", "A2"], "team_B": ["B1", "B2"]}
        Free-for-all: {"agent_0": ["agent_0"], "agent_1": ["agent_1"], ...}
        """

    # ------------------------------------------------------------------
    # Optional interface
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> Optional[Any]:
        """
        Render the current game state.

        Parameters
        ----------
        mode : str
            'human'  – display to screen (Pygame window).
            'rgb_array' – return an HxWx3 numpy array.
            'ansi'   – return a string for terminal display.

        Default implementation is a no-op; override in games that support
        visual rendering.
        """

    def close(self) -> None:
        """Clean up resources (Pygame window, network sockets, etc.)."""

    # ------------------------------------------------------------------
    # Convenience helpers (not abstract – subclasses may use as-is)
    # ------------------------------------------------------------------

    def agent_to_team(self) -> Dict[AgentID, TeamID]:
        """Invert the teams mapping: agent_id -> team_id."""
        return {
            agent_id: team_id
            for team_id, agent_ids in self.teams.items()
            for agent_id in agent_ids
        }

    def teammates_of(self, agent_id: AgentID) -> List[AgentID]:
        """Return all teammates of *agent_id* (excluding itself)."""
        mapping = self.agent_to_team()
        team_id = mapping.get(agent_id)
        if team_id is None:
            return []
        return [a for a in self.teams[team_id] if a != agent_id]

    def opponents_of(self, agent_id: AgentID) -> List[AgentID]:
        """Return all agents not on the same team as *agent_id*."""
        mapping = self.agent_to_team()
        team_id = mapping.get(agent_id)
        return [a for a, t in mapping.items() if t != team_id]

    @property
    def all_agents(self) -> List[AgentID]:
        """Flat list of every agent id registered in `teams`."""
        return [a for agents in self.teams.values() for a in agents]
