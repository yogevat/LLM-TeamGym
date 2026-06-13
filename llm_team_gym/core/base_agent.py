"""
Abstract base class for all agents in LLM-TeamGym.

Supports both LLM-backed agents (OpenAI, Anthropic, local models) and
deterministic scripted agents (random, greedy, rule-based).
"""

from __future__ import annotations

import abc
import random
from typing import Any, Dict, List, Optional

from llm_team_gym.core.base_game import Action, AgentID, Observation


class BaseAgent(abc.ABC):
    """
    Abstract agent that can act in any BaseGame.

    Subclasses implement `choose_action`, which receives:
      - the raw observation object from the game
      - the text-state string (pre-formatted for prompt injection)
      - the list of legal moves

    and returns a single action from the legal moves list.
    """

    def __init__(self, agent_id: AgentID, team_id: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.team_id = team_id
        self.config: Dict[str, Any] = config or {}

        # Episode memory: subclasses may populate this for multi-turn context.
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def choose_action(
        self,
        observation: Observation,
        text_state: str,
        legal_moves: List[Action],
        game_rules: str,
    ) -> Action:
        """
        Select an action given the current game state.

        Parameters
        ----------
        observation : Observation
            Raw structured observation from BaseGame.step / BaseGame.reset.
        text_state : str
            Pre-formatted text representation from BaseGame.get_text_state —
            ready to be injected into an LLM prompt.
        legal_moves : list
            All legal actions for this agent this turn.
        game_rules : str
            Complete rule description from BaseGame.get_game_rules — should be
            provided as a system prompt to LLM agents.

        Returns
        -------
        action : Action
            Must be an element of *legal_moves*.
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional to override)
    # ------------------------------------------------------------------

    def on_episode_start(self, initial_obs: Observation, game_rules: str) -> None:
        """Called once at the start of each episode. Clears history by default."""
        self.history.clear()

    def on_episode_end(self, final_obs: Observation, total_reward: float) -> None:
        """Called once when the episode finishes. Use for logging or learning."""

    def on_step_end(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Called after every step. Default appends to history."""
        self.history.append(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.agent_id})"

    def __repr__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Concrete scripted agents (no LLM dependency)
# ---------------------------------------------------------------------------


class RandomAgent(BaseAgent):
    """Selects uniformly at random from the legal moves. Useful as a baseline."""

    def __init__(
        self,
        agent_id: AgentID,
        team_id: str,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id, team_id, config)
        self.rng = random.Random(seed)

    def choose_action(
        self,
        observation: Observation,
        text_state: str,
        legal_moves: List[Action],
        game_rules: str,
    ) -> Action:
        if not legal_moves:
            raise ValueError(f"Agent {self.agent_id} has no legal moves.")
        return self.rng.choice(legal_moves)


class GreedyAgent(BaseAgent):
    """
    Selects the action whose key string sorts first (deterministic tie-break).

    Intended as a simple non-random scripted agent for smoke-testing.
    Subclass and override `score_action` to implement game-specific heuristics.
    """

    def choose_action(
        self,
        observation: Observation,
        text_state: str,
        legal_moves: List[Action],
        game_rules: str,
    ) -> Action:
        if not legal_moves:
            raise ValueError(f"Agent {self.agent_id} has no legal moves.")
        scored = sorted(legal_moves, key=self.score_action, reverse=True)
        return scored[0]

    def score_action(self, action: Action) -> float:
        """
        Assign a numeric score to an action.

        Default implementation converts action to string and uses its hash,
        giving a reproducible but arbitrary ordering. Override for heuristics.
        """
        return float(hash(str(action)) % 10_000)


class HumanCLIAgent(BaseAgent):
    """
    Prompts a human player via stdin. Useful for manual testing / demos.
    """

    def choose_action(
        self,
        observation: Observation,
        text_state: str,
        legal_moves: List[Action],
        game_rules: str,
    ) -> Action:
        print("\n" + "=" * 60)
        print(f"Agent: {self.agent_id}  |  Team: {self.team_id}")
        print(text_state)
        print(f"\nLegal moves: {legal_moves}")
        while True:
            raw = input("Enter your action: ").strip()
            # Try exact match first, then positional index.
            if raw in [str(m) for m in legal_moves]:
                for m in legal_moves:
                    if str(m) == raw:
                        return m
            try:
                idx = int(raw)
                if 0 <= idx < len(legal_moves):
                    return legal_moves[idx]
            except ValueError:
                pass
            print(f"Invalid input '{raw}'. Choose from: {legal_moves}")
