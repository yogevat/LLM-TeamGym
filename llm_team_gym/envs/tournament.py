"""
Match orchestrator for LLM-TeamGym.

`MatchRunner` drives a single game episode: it loops over agents in turn order,
queries each agent for an action, steps the game, calls the logger, and
triggers Pygame rendering when enabled.

`TournamentRunner` wraps multiple MatchRunner calls across agent pools,
collects aggregate statistics, and ranks teams by Elo or win-rate.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_team_gym.core.base_agent import BaseAgent
from llm_team_gym.core.base_game import AgentID, BaseGame, TeamID
from llm_team_gym.envs.logger import MatchLogger, MatchRecord


class MatchRunner:
    """
    Orchestrates a single match between registered agents on a given game.

    Turn-based vs. simultaneous:
      The game determines who acts each step via `get_legal_moves`. This runner
      collects actions from every agent that has at least one legal move, then
      calls game.step(actions_dict). Games that want strict turn-based control
      should return an empty list from `get_legal_moves` for inactive agents.

    Parameters
    ----------
    game : BaseGame
        Instantiated, configured game object.
    agents : list[BaseAgent]
        All participating agents. Must cover every agent_id in game.teams.
    render : bool
        Whether to call game.render() after each step.
    render_fps : int
        Target frame rate when rendering (sleep between steps).
    step_delay : float
        Additional artificial delay per step (seconds). Useful for demos.
    output_dir : str | None
        Directory for JSONL transcript output. None = no file output.
    verbose : bool
        Stream step-by-step logs to console.
    match_id : str | None
        Explicit match ID. Auto-generated UUID4 if None.
    """

    def __init__(
        self,
        game: BaseGame,
        agents: List[BaseAgent],
        render: bool = False,
        render_fps: int = 4,
        step_delay: float = 0.0,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        match_id: Optional[str] = None,
        on_step_callback: Optional[Callable[[int, Dict], None]] = None,
    ):
        self.game = game
        self.agents: Dict[AgentID, BaseAgent] = {a.agent_id: a for a in agents}
        self.render_enabled = render
        self.render_fps = render_fps
        self.step_delay = step_delay
        self.output_dir = output_dir
        self.verbose = verbose
        self.match_id = match_id or f"match_{uuid.uuid4().hex[:8]}"
        self.on_step_callback = on_step_callback

        # Validate coverage
        missing = set(game.all_agents) - set(self.agents.keys())
        if missing:
            raise ValueError(f"No agent objects provided for: {missing}")

    def run(self) -> MatchRecord:
        """Execute one complete episode and return the MatchRecord."""
        game = self.game
        game_rules = game.get_game_rules()

        logger = MatchLogger(
            match_id=self.match_id,
            game_name=type(game).__name__,
            teams=game.teams,
            agent_types={aid: type(a).__name__ for aid, a in self.agents.items()},
            output_dir=self.output_dir,
            verbose=self.verbose,
        )

        # Reset
        observations = game.reset()
        for agent_id, agent in self.agents.items():
            agent.on_episode_start(observations.get(agent_id), game_rules)

        cumulative_rewards: Dict[AgentID, float] = {a: 0.0 for a in self.agents}
        step_num = 0
        game_over = False

        if self.render_enabled:
            game.render(mode="human")

        while not game_over:
            # Collect actions from all agents that have legal moves this turn.
            actions_dict: Dict[AgentID, Any] = {}
            acting_agents: List[AgentID] = []

            for agent_id, agent in self.agents.items():
                legal = game.get_legal_moves(agent_id)
                if not legal:
                    continue
                text_state = game.get_text_state(agent_id)
                obs = observations.get(agent_id)
                action = agent.choose_action(obs, text_state, legal, game_rules)
                actions_dict[agent_id] = action
                acting_agents.append(agent_id)

            if not actions_dict:
                # No agent can move — treat as game-over (shouldn't happen in
                # well-designed games, but guards against infinite loops).
                break

            # Step the game
            observations, rewards, dones, infos = game.step(actions_dict)

            # Update cumulative rewards and notify agents
            for agent_id, reward in rewards.items():
                cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + reward
                done = dones.get(agent_id, False)
                self.agents[agent_id].on_step_end(
                    observations.get(agent_id), actions_dict.get(agent_id), reward, done, infos.get(agent_id, {})
                )

            logger.log_step(
                step=step_num,
                acting_agents=acting_agents,
                actions=actions_dict,
                rewards=rewards,
                dones=dones,
                infos=infos,
            )

            if self.on_step_callback:
                self.on_step_callback(step_num, {
                    "actions": actions_dict,
                    "rewards": rewards,
                    "dones": dones,
                    "infos": infos,
                })

            if self.render_enabled:
                game.render(mode="human")
                if self.step_delay > 0:
                    time.sleep(self.step_delay)
                elif self.render_fps > 0:
                    time.sleep(1.0 / self.render_fps)

            step_num += 1
            game_over = dones.get("__all__", False) or all(dones.get(a, False) for a in game.all_agents)

        # Compute team-level scores
        a2t = game.agent_to_team()
        final_team_scores: Dict[TeamID, float] = {}
        for agent_id, score in cumulative_rewards.items():
            team_id = a2t.get(agent_id, agent_id)
            final_team_scores[team_id] = final_team_scores.get(team_id, 0.0) + score

        winner = max(final_team_scores, key=lambda t: final_team_scores[t]) if final_team_scores else None

        # Notify agents of episode end
        for agent_id, agent in self.agents.items():
            agent.on_episode_end(observations.get(agent_id), cumulative_rewards[agent_id])

        record = logger.finalize(final_team_scores=final_team_scores, winner=winner)

        if self.render_enabled:
            game.close()

        return record


class TournamentRunner:
    """
    Runs multiple matches and aggregates win rates, average scores, and rankings.

    Parameters
    ----------
    game_factory : callable
        Zero-arg callable that returns a fresh BaseGame instance each match.
    agent_factory : callable
        Zero-arg callable that returns a fresh list of BaseAgent instances.
    n_matches : int
        Number of matches to run.
    runner_kwargs : dict
        Keyword arguments forwarded to MatchRunner (render, step_delay, etc.)
    """

    def __init__(
        self,
        game_factory: Callable[[], BaseGame],
        agent_factory: Callable[[], List[BaseAgent]],
        n_matches: int = 10,
        runner_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.game_factory = game_factory
        self.agent_factory = agent_factory
        self.n_matches = n_matches
        self.runner_kwargs = runner_kwargs or {}

    def run(self) -> Dict[str, Any]:
        """Execute all matches and return aggregated stats."""
        records: List[MatchRecord] = []
        win_counts: Dict[TeamID, int] = {}
        score_sums: Dict[TeamID, float] = {}

        for i in range(self.n_matches):
            game = self.game_factory()
            agents = self.agent_factory()
            runner = MatchRunner(
                game=game,
                agents=agents,
                match_id=f"tournament_match_{i:04d}",
                **self.runner_kwargs,
            )
            record = runner.run()
            records.append(record)

            if record.winner:
                win_counts[record.winner] = win_counts.get(record.winner, 0) + 1
            for team, score in record.final_team_scores.items():
                score_sums[team] = score_sums.get(team, 0.0) + score

        teams = list(score_sums.keys())
        stats = {
            "n_matches": self.n_matches,
            "win_counts": win_counts,
            "win_rates": {t: win_counts.get(t, 0) / self.n_matches for t in teams},
            "avg_scores": {t: score_sums[t] / self.n_matches for t in teams},
            "records": records,
        }
        return stats
