"""
LLM-backed agents for LLM-TeamGym.

Provides concrete agent implementations that connect to real LLM APIs
(OpenAI, Anthropic, Ollama) as well as a MockLLMAgent for testing.

All agents extend BaseAgent and implement `choose_action` by:
  1. Building a prompt from the game state and legal moves.
  2. Calling the LLM API.
  3. Parsing a valid action from the response.

Optional dependencies are import-guarded so the module loads even when
only a subset of provider libraries is installed.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from llm_team_gym.core.base_agent import BaseAgent
from llm_team_gym.core.base_game import Action, AgentID, Observation

# ---------------------------------------------------------------------------
# Optional provider imports
# ---------------------------------------------------------------------------

try:
    import openai as _openai_lib
except ImportError:
    _openai_lib = None

try:
    import anthropic as _anthropic_lib
except ImportError:
    _anthropic_lib = None

try:
    import requests as _requests_lib
except ImportError:
    _requests_lib = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost tables (USD per 1K tokens, approximate)
# ---------------------------------------------------------------------------

_OPENAI_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}

_ANTHROPIC_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "claude-sonnet-4-20250514": {"prompt": 0.003, "completion": 0.015},
    "claude-opus-4-20250514": {"prompt": 0.015, "completion": 0.075},
    "claude-haiku-4-20250414": {"prompt": 0.0008, "completion": 0.004},
    "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
}

# Maximum number of retries when the LLM returns an invalid action.
_MAX_RETRIES = 3


# ===================================================================
# LLMAgent — abstract base for all LLM-backed agents
# ===================================================================


class LLMAgent(BaseAgent, abc.ABC):
    """
    Abstract base class for agents backed by a language model.

    Subclasses must implement ``_call_llm`` which sends a list of message
    dicts to the provider and returns the assistant's text response.
    """

    def __init__(
        self,
        agent_id: AgentID,
        team_id: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id, team_id, config)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

        # Usage tracking
        self.total_tokens: int = 0
        self.total_calls: int = 0
        self.total_cost: float = 0.0
        self.call_latencies: List[float] = []

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        text_state: str,
        legal_moves: List[Action],
        game_rules: str,
    ) -> List[Dict[str, str]]:
        """Build the message list (system + user) for the LLM call."""
        moves_str = ", ".join(str(m) for m in legal_moves)
        system_msg = (
            f"You are a game-playing AI agent (id={self.agent_id}, team={self.team_id}).\n\n"
            f"GAME RULES:\n{game_rules}\n\n"
            "INSTRUCTIONS:\n"
            "- Analyze the current game state carefully.\n"
            "- Choose the best action from the legal moves listed below.\n"
            "- Respond with ONLY your chosen action on the first line, "
            "exactly as it appears in the legal moves list.\n"
            "- You may add a brief explanation on subsequent lines, but "
            "the first line MUST be the action and nothing else."
        )
        user_msg = (
            f"CURRENT STATE:\n{text_state}\n\n"
            f"LEGAL MOVES: [{moves_str}]\n\n"
            "Your action:"
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _parse_action(
        self,
        response_text: str,
        legal_moves: List[Action],
    ) -> Optional[Action]:
        """
        Extract a valid action from the LLM's response.

        Strategy:
          1. Exact string match of the first non-empty line against
             ``str(move)`` for each legal move.
          2. Case-insensitive match.
          3. Substring / containment search across the full response.
          4. Integer index into the legal-moves list.

        Returns ``None`` if no valid action can be extracted.
        """
        if not legal_moves:
            return None

        # Get the first non-empty line.
        first_line = ""
        for line in response_text.strip().splitlines():
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break

        move_strs = [str(m) for m in legal_moves]

        # 1. Exact match on first line.
        if first_line in move_strs:
            return legal_moves[move_strs.index(first_line)]

        # 2. Case-insensitive match on first line.
        first_lower = first_line.lower()
        for i, ms in enumerate(move_strs):
            if ms.lower() == first_lower:
                return legal_moves[i]

        # 3. Substring search — check if any legal move string appears in
        #    the first line (longest first to avoid partial matches).
        sorted_moves = sorted(
            enumerate(move_strs), key=lambda x: len(x[1]), reverse=True
        )
        for i, ms in sorted_moves:
            if ms and ms in first_line:
                return legal_moves[i]

        # 4. Try to interpret the first line as an integer index.
        try:
            idx = int(re.sub(r"[^\d\-]", "", first_line))
            if 0 <= idx < len(legal_moves):
                return legal_moves[idx]
        except (ValueError, IndexError):
            pass

        # 5. Full-text substring search (for multi-line responses).
        full_lower = response_text.lower()
        for i, ms in sorted_moves:
            if ms and ms.lower() in full_lower:
                return legal_moves[i]

        return None

    # ------------------------------------------------------------------
    # Abstract LLM call
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages to the LLM and return the assistant's text response.

        Implementations should update ``self.total_tokens`` and
        ``self.total_cost`` based on usage data from the API response.
        """

    # ------------------------------------------------------------------
    # choose_action  (implements BaseAgent interface)
    # ------------------------------------------------------------------

    def choose_action(
        self,
        observation: Observation,
        text_state: str,
        legal_moves: List[Action],
        game_rules: str,
    ) -> Action:
        if not legal_moves:
            raise ValueError(f"Agent {self.agent_id} has no legal moves.")

        messages = self._build_prompt(text_state, legal_moves, game_rules)

        last_response = ""
        for attempt in range(_MAX_RETRIES):
            t0 = time.monotonic()
            try:
                last_response = self._call_llm(messages)
            except Exception as exc:
                logger.warning(
                    "LLM call failed for %s (attempt %d/%d): %s",
                    self.name, attempt + 1, _MAX_RETRIES, exc,
                )
                if attempt == _MAX_RETRIES - 1:
                    # All retries exhausted — fall back to random.
                    logger.error(
                        "All LLM retries exhausted for %s; choosing random move.",
                        self.name,
                    )
                    return random.choice(legal_moves)
                continue
            finally:
                elapsed = time.monotonic() - t0
                self.call_latencies.append(elapsed)
                self.total_calls += 1

            action = self._parse_action(last_response, legal_moves)
            if action is not None:
                return action

            # Invalid action — add feedback and retry.
            logger.info(
                "Agent %s returned invalid action on attempt %d: %r",
                self.name, attempt + 1, last_response[:120],
            )
            messages.append({"role": "assistant", "content": last_response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That is not a valid action. "
                        f"Please choose EXACTLY one of: [{', '.join(str(m) for m in legal_moves)}]\n"
                        "Respond with ONLY the action on the first line."
                    ),
                }
            )

        # Exhausted retries with valid API responses but unparseable actions.
        logger.warning(
            "Agent %s could not parse a valid action after %d attempts; "
            "falling back to random. Last response: %r",
            self.name, _MAX_RETRIES, last_response[:200],
        )
        return random.choice(legal_moves)

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    def on_episode_start(self, initial_obs: Observation, game_rules: str) -> None:
        super().on_episode_start(initial_obs, game_rules)

    def on_episode_end(self, final_obs: Observation, total_reward: float) -> None:
        super().on_episode_end(final_obs, total_reward)
        logger.info(
            "%s episode ended | calls=%d tokens=%d cost=$%.4f avg_latency=%.2fs",
            self.name,
            self.total_calls,
            self.total_tokens,
            self.total_cost,
            (sum(self.call_latencies) / len(self.call_latencies))
            if self.call_latencies
            else 0.0,
        )

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.agent_id}, {self.model})"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return a summary dict of usage statistics."""
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_latency": (
                sum(self.call_latencies) / len(self.call_latencies)
                if self.call_latencies
                else 0.0
            ),
            "call_latencies": list(self.call_latencies),
        }


# ===================================================================
# OpenAIAgent
# ===================================================================


class OpenAIAgent(LLMAgent):
    """
    Agent backed by the OpenAI Chat Completions API.

    Requires the ``openai`` package (``pip install openai``).
    API key is read from the constructor argument, the ``config`` dict,
    or the ``OPENAI_API_KEY`` environment variable, in that order.
    """

    def __init__(
        self,
        agent_id: AgentID,
        team_id: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if _openai_lib is None:
            raise ImportError(
                "The 'openai' package is required for OpenAIAgent. "
                "Install it with: pip install openai"
            )

        resolved_key = (
            api_key
            or (config or {}).get("api_key")
            or os.environ.get("OPENAI_API_KEY")
        )
        super().__init__(
            agent_id, team_id, model, temperature, max_tokens, resolved_key, config
        )

        self._client = _openai_lib.OpenAI(api_key=resolved_key)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Track token usage.
        usage = response.usage
        if usage:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self._update_cost(usage.prompt_tokens, usage.completion_tokens)

        return response.choices[0].message.content or ""

    def _update_cost(self, prompt_toks: int, completion_toks: int) -> None:
        """Estimate cost from the per-model pricing table."""
        costs = _OPENAI_COST_PER_1K.get(self.model)
        if costs:
            self.total_cost += (
                (prompt_toks / 1000) * costs["prompt"]
                + (completion_toks / 1000) * costs["completion"]
            )


# ===================================================================
# AnthropicAgent
# ===================================================================


class AnthropicAgent(LLMAgent):
    """
    Agent backed by the Anthropic Messages API.

    Requires the ``anthropic`` package (``pip install anthropic``).
    API key is read from the constructor argument, the ``config`` dict,
    or the ``ANTHROPIC_API_KEY`` environment variable, in that order.
    """

    def __init__(
        self,
        agent_id: AgentID,
        team_id: str,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if _anthropic_lib is None:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicAgent. "
                "Install it with: pip install anthropic"
            )

        resolved_key = (
            api_key
            or (config or {}).get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        super().__init__(
            agent_id, team_id, model, temperature, max_tokens, resolved_key, config
        )

        self._client = _anthropic_lib.Anthropic(api_key=resolved_key)
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        # Anthropic separates the system prompt from the message list.
        system_text = ""
        api_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                api_messages.append(msg)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_text.strip(),
            messages=api_messages,
        )

        # Track token usage.
        usage = response.usage
        if usage:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.total_tokens += usage.input_tokens + usage.output_tokens
            self._update_cost(usage.input_tokens, usage.output_tokens)

        # Extract text from content blocks.
        text_parts: List[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    def _update_cost(self, input_toks: int, output_toks: int) -> None:
        """Estimate cost from the per-model pricing table."""
        costs = _ANTHROPIC_COST_PER_1K.get(self.model)
        if costs:
            self.total_cost += (
                (input_toks / 1000) * costs["prompt"]
                + (output_toks / 1000) * costs["completion"]
            )


# ===================================================================
# OllamaAgent
# ===================================================================


class OllamaAgent(LLMAgent):
    """
    Agent backed by a local Ollama server (http://localhost:11434).

    Works with any model served by Ollama (e.g. ``llama3.1``,
    ``gemma3``, ``mistral``).  No API key required.

    Requires the ``requests`` library (usually already installed).
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        agent_id: AgentID,
        team_id: str,
        model: str = "llama3.1",
        temperature: float = 0.7,
        max_tokens: int = 512,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if _requests_lib is None:
            raise ImportError(
                "The 'requests' package is required for OllamaAgent. "
                "Install it with: pip install requests"
            )

        super().__init__(
            agent_id, team_id, model, temperature, max_tokens, api_key=None, config=config
        )
        self.base_url = (
            base_url
            or (config or {}).get("base_url")
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        resp = _requests_lib.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns token counts in various fields depending on version.
        if "prompt_eval_count" in data:
            prompt_toks = data.get("prompt_eval_count", 0)
            completion_toks = data.get("eval_count", 0)
            self.total_tokens += prompt_toks + completion_toks

        content = data.get("message", {}).get("content", "")
        return content


# ===================================================================
# MockLLMAgent  (for testing without real APIs)
# ===================================================================


class MockLLMAgent(LLMAgent):
    """
    Mock agent that simulates LLM behaviour by returning a random
    legal move.  Useful for unit tests, integration tests, and
    benchmarking the game loop without incurring API costs.

    Configurable parameters (via ``config`` dict):
      - ``latency`` (float): simulated call latency in seconds (default 0.05).
      - ``seed`` (int): random seed for reproducibility.
      - ``fail_rate`` (float): probability [0, 1) of simulating an API
        error on each call (default 0.0).
    """

    def __init__(
        self,
        agent_id: AgentID,
        team_id: str,
        model: str = "mock-llm",
        temperature: float = 0.7,
        max_tokens: int = 512,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id, team_id, model, temperature, max_tokens, api_key=None, config=config
        )
        cfg = config or {}
        self._latency: float = cfg.get("latency", 0.05)
        self._seed: Optional[int] = cfg.get("seed")
        self._fail_rate: float = cfg.get("fail_rate", 0.0)
        self._rng = random.Random(self._seed)

        # Store the last prompt for test inspection.
        self.last_messages: List[Dict[str, str]] = []

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        self.last_messages = list(messages)

        # Simulate latency.
        if self._latency > 0:
            time.sleep(self._latency)

        # Simulate occasional API failures.
        if self._fail_rate > 0 and self._rng.random() < self._fail_rate:
            raise RuntimeError("MockLLMAgent: simulated API failure")

        # Extract legal moves from the user message to pick a valid one.
        legal_moves_str = ""
        for msg in reversed(messages):
            if msg["role"] == "user" and "LEGAL MOVES:" in msg["content"]:
                # Parse the moves list from the prompt.
                match = re.search(r"LEGAL MOVES:\s*\[(.+?)\]", msg["content"])
                if match:
                    legal_moves_str = match.group(1)
                break

        if legal_moves_str:
            # Split on commas, strip whitespace.
            moves = [m.strip() for m in legal_moves_str.split(",") if m.strip()]
            if moves:
                chosen = self._rng.choice(moves)
                # Simulate token usage.
                fake_prompt_tokens = sum(len(m["content"]) // 4 for m in messages)
                fake_completion_tokens = len(chosen) // 4 + 1
                self.total_tokens += fake_prompt_tokens + fake_completion_tokens
                return chosen

        # Fallback: return "0" (first legal move by index).
        self.total_tokens += 10
        return "0"

    @property
    def name(self) -> str:
        return f"MockLLMAgent({self.agent_id})"


# ===================================================================
# Factory helper
# ===================================================================


def create_llm_agent(
    provider: str,
    agent_id: AgentID,
    team_id: str,
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMAgent:
    """
    Convenience factory to instantiate an LLM agent by provider name.

    Parameters
    ----------
    provider : str
        One of ``"openai"``, ``"anthropic"``, ``"ollama"``, ``"mock"``.
    agent_id : AgentID
        Unique agent identifier.
    team_id : str
        Team the agent belongs to.
    model : str, optional
        Model name; defaults vary per provider.
    **kwargs
        Forwarded to the agent constructor (``temperature``, ``api_key``, etc.).

    Returns
    -------
    LLMAgent
    """
    provider_lower = provider.lower()
    if provider_lower == "openai":
        return OpenAIAgent(agent_id, team_id, model=model or "gpt-4o-mini", **kwargs)
    elif provider_lower == "anthropic":
        return AnthropicAgent(
            agent_id, team_id, model=model or "claude-sonnet-4-20250514", **kwargs
        )
    elif provider_lower == "ollama":
        return OllamaAgent(agent_id, team_id, model=model or "llama3.1", **kwargs)
    elif provider_lower == "mock":
        return MockLLMAgent(agent_id, team_id, model=model or "mock-llm", **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: openai, anthropic, ollama, mock."
        )
