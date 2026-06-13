"""
Structured match logger for LLM-TeamGym.

Captures per-step events and writes JSONL transcripts. Designed to be
consumed by downstream analysis tools or LLM evaluation pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from llm_team_gym.core.base_game import AgentID, TeamID


@dataclass
class StepRecord:
    step: int
    acting_agents: List[AgentID]
    actions: Dict[AgentID, Any]
    rewards: Dict[AgentID, float]
    dones: Dict[AgentID, bool]
    infos: Dict[AgentID, Any]
    cumulative_rewards: Dict[AgentID, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class MatchRecord:
    match_id: str
    game_name: str
    teams: Dict[TeamID, List[AgentID]]
    agent_types: Dict[AgentID, str]
    steps: List[StepRecord] = field(default_factory=list)
    final_team_scores: Dict[TeamID, float] = field(default_factory=dict)
    winner: Optional[TeamID] = None
    total_steps: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def duration(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class MatchLogger:
    """
    Records every step of a match and optionally writes a JSONL transcript.

    Usage
    -----
    logger = MatchLogger(match_id="m001", game_name="TeamFish", teams=..., output_dir="logs/")
    logger.log_step(step=0, acting_agents=..., actions=..., rewards=..., dones=..., infos=...)
    logger.finalize(final_team_scores=..., winner="team_A")
    """

    def __init__(
        self,
        match_id: str,
        game_name: str,
        teams: Dict[TeamID, List[AgentID]],
        agent_types: Dict[AgentID, str],
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.record = MatchRecord(
            match_id=match_id,
            game_name=game_name,
            teams=teams,
            agent_types=agent_types,
        )
        self._cumulative: Dict[AgentID, float] = {
            a: 0.0 for agents in teams.values() for a in agents
        }
        self.output_dir = output_dir
        self.verbose = verbose

        self._console = logging.getLogger(f"LLMTeamGym.{match_id}")
        if not self._console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self._console.addHandler(handler)
        self._console.setLevel(logging.DEBUG if verbose else logging.WARNING)

        self._console.info(
            f"Match {match_id} started | game={game_name} | "
            f"teams={json.dumps(teams, default=str)}"
        )

    def log_step(
        self,
        step: int,
        acting_agents: List[AgentID],
        actions: Dict[AgentID, Any],
        rewards: Dict[AgentID, float],
        dones: Dict[AgentID, bool],
        infos: Dict[AgentID, Any],
    ) -> None:
        for agent_id, r in rewards.items():
            self._cumulative[agent_id] = self._cumulative.get(agent_id, 0.0) + r

        rec = StepRecord(
            step=step,
            acting_agents=acting_agents,
            actions=actions,
            rewards=rewards,
            dones=dones,
            infos=infos,
            cumulative_rewards=dict(self._cumulative),
        )
        self.record.steps.append(rec)

        if self.verbose:
            action_str = ", ".join(f"{a}→{v}" for a, v in actions.items())
            reward_str = ", ".join(f"{a}:{r:+.1f}" for a, r in rewards.items())
            self._console.debug(f"  Step {step:03d} | {action_str} | rewards [{reward_str}]")

    def finalize(
        self,
        final_team_scores: Dict[TeamID, float],
        winner: Optional[TeamID],
    ) -> MatchRecord:
        self.record.final_team_scores = final_team_scores
        self.record.winner = winner
        self.record.total_steps = len(self.record.steps)
        self.record.end_time = time.time()

        self._console.info(
            f"Match {self.record.match_id} complete | "
            f"steps={self.record.total_steps} | "
            f"scores={json.dumps(final_team_scores)} | "
            f"winner={winner} | "
            f"duration={self.record.duration():.2f}s"
        )

        if self.output_dir:
            self._write_jsonl()

        return self.record

    def _write_jsonl(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{self.record.match_id}.jsonl")
        with open(path, "w") as f:
            # Header record
            header = {
                "type": "match_meta",
                "match_id": self.record.match_id,
                "game_name": self.record.game_name,
                "teams": self.record.teams,
                "agent_types": self.record.agent_types,
                "start_time": self.record.start_time,
            }
            f.write(json.dumps(header, default=str) + "\n")
            # Step records
            for step in self.record.steps:
                f.write(json.dumps(asdict(step), default=str) + "\n")
            # Footer record
            footer = {
                "type": "match_result",
                "final_team_scores": self.record.final_team_scores,
                "winner": self.record.winner,
                "total_steps": self.record.total_steps,
                "duration": self.record.duration(),
                "end_time": self.record.end_time,
            }
            f.write(json.dumps(footer, default=str) + "\n")
        self._console.info(f"Transcript saved → {path}")
