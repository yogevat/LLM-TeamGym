"""Core module — agents, games, roles, topology, communication, and metrics."""

from llm_team_gym.core.base_game import BaseGame
from llm_team_gym.core.base_agent import BaseAgent, RandomAgent, GreedyAgent, HumanCLIAgent
from llm_team_gym.core.roles import (
    Role,
    WorkerAgent,
    ValidatorAgent,
    ThinkerAgent,
    TeamOrchestrator,
)
from llm_team_gym.core.topology import TopologyGraph, TopologyShape
from llm_team_gym.core.communication import Message, MessageChannel, CommunicationProtocol
from llm_team_gym.core.metrics import MetricsCollector, ExperimentTracker

# LLM agents depend on optional packages (openai, anthropic, requests).
# Import them in a try/except so core works without those dependencies.
try:
    from llm_team_gym.core.llm_agent import (
        LLMAgent,
        OpenAIAgent,
        AnthropicAgent,
        OllamaAgent,
        MockLLMAgent,
    )
except ImportError:  # pragma: no cover
    LLMAgent = None  # type: ignore[assignment,misc]
    OpenAIAgent = None  # type: ignore[assignment,misc]
    AnthropicAgent = None  # type: ignore[assignment,misc]
    OllamaAgent = None  # type: ignore[assignment,misc]
    MockLLMAgent = None  # type: ignore[assignment,misc]

__all__ = [
    # base
    "BaseGame",
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "HumanCLIAgent",
    # roles
    "Role",
    "WorkerAgent",
    "ValidatorAgent",
    "ThinkerAgent",
    "TeamOrchestrator",
    # topology
    "TopologyGraph",
    "TopologyShape",
    # communication
    "Message",
    "MessageChannel",
    "CommunicationProtocol",
    # metrics
    "MetricsCollector",
    "ExperimentTracker",
    # llm agents (may be None if deps missing)
    "LLMAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "OllamaAgent",
    "MockLLMAgent",
]
