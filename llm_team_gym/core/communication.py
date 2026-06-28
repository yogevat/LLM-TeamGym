"""
Message-passing protocol between agents in a topology-aware team.

Provides:
  - Message / RoundResult data classes
  - MessageType enum (PROPOSAL, VERDICT, QUERY, RESPONSE, READOUT)
  - MessageChannel for topology-aware routing
  - CommunicationProtocol base class with two concrete protocols:
      * SynchronousProtocol  – one pass: Workers -> Validators -> Thinker
      * IterativeProtocol    – multiple revision passes of W -> V -> T

All implementations use only the Python standard library.
"""

from __future__ import annotations

import abc
import enum
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_team_gym.core.base_game import Action


# ======================================================================
# Enums
# ======================================================================

class MessageType(enum.Enum):
    """Types of messages agents can exchange."""
    PROPOSAL = "PROPOSAL"
    VERDICT = "VERDICT"
    QUERY = "QUERY"
    RESPONSE = "RESPONSE"
    READOUT = "READOUT"


class AgentRole(enum.Enum):
    """Roles an agent can play within a team topology."""
    WORKER = "WORKER"
    VALIDATOR = "VALIDATOR"
    THINKER = "THINKER"


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class Message:
    """A single message exchanged between two agents."""
    sender_id: str
    receiver_id: str
    content: str
    msg_type: MessageType
    timestamp: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Message({self.msg_type.value}: "
            f"{self.sender_id}->{self.receiver_id}, "
            f"{self.content[:60]!r})"
        )


@dataclass
class RoundResult:
    """Outcome of one communication round."""
    messages: List[Message]
    final_action: Action
    n_llm_calls: int
    n_rounds: int
    wall_time: float


# ======================================================================
# Topology
# ======================================================================

@dataclass
class Topology:
    """
    Describes a team's internal communication topology.

    Parameters
    ----------
    roles : dict
        Mapping of agent_id -> AgentRole.
    edges : set of (str, str) tuples
        Directed communication edges. (A, B) means A can send to B.
        Undirected communication is represented by including both (A, B)
        and (B, A).
    """
    roles: Dict[str, AgentRole]
    edges: Set[Tuple[str, str]]

    # -- convenience queries -------------------------------------------------

    def neighbours(self, agent_id: str) -> List[str]:
        """Return all agent ids that *agent_id* can send messages to."""
        return [b for (a, b) in self.edges if a == agent_id]

    def agents_with_role(self, role: AgentRole) -> List[str]:
        """Return all agent ids that hold a given role."""
        return [aid for aid, r in self.roles.items() if r == role]

    @property
    def workers(self) -> List[str]:
        return self.agents_with_role(AgentRole.WORKER)

    @property
    def validators(self) -> List[str]:
        return self.agents_with_role(AgentRole.VALIDATOR)

    @property
    def thinkers(self) -> List[str]:
        return self.agents_with_role(AgentRole.THINKER)

    def can_send(self, sender_id: str, receiver_id: str) -> bool:
        """Check whether sender is allowed to message receiver."""
        return (sender_id, receiver_id) in self.edges

    # -- factory helpers -----------------------------------------------------

    @classmethod
    def fully_connected(cls, agent_ids: List[str], roles: Dict[str, AgentRole]) -> "Topology":
        """Every agent can message every other agent."""
        edges: Set[Tuple[str, str]] = set()
        for a in agent_ids:
            for b in agent_ids:
                if a != b:
                    edges.add((a, b))
        return cls(roles=dict(roles), edges=edges)

    @classmethod
    def star(cls, center_id: str, spoke_ids: List[str],
             roles: Dict[str, AgentRole]) -> "Topology":
        """
        Spokes can send to center and center can send to spokes,
        but spokes cannot directly message each other.
        """
        edges: Set[Tuple[str, str]] = set()
        for s in spoke_ids:
            edges.add((s, center_id))
            edges.add((center_id, s))
        return cls(roles=dict(roles), edges=edges)


# ======================================================================
# MessageChannel
# ======================================================================

class MessageChannel:
    """
    Manages message routing between agents according to a topology.

    Messages that violate the topology (sender not connected to receiver)
    are silently dropped and recorded in ``dropped_messages``.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology
        # Pending inbox per agent (messages not yet consumed by receive()).
        self._inbox: Dict[str, List[Message]] = defaultdict(list)
        # Complete log of all successfully delivered messages.
        self._log: List[Message] = []
        # Messages that violated the topology.
        self.dropped_messages: List[Message] = []

    # -- core API ------------------------------------------------------------

    def send(self, message: Message) -> None:
        """
        Route a single message.

        The message is delivered only if the topology allows the edge
        sender_id -> receiver_id.  Otherwise it is added to
        ``dropped_messages``.
        """
        if self._topology.can_send(message.sender_id, message.receiver_id):
            self._inbox[message.receiver_id].append(message)
            self._log.append(message)
        else:
            self.dropped_messages.append(message)

    def receive(self, agent_id: str) -> List[Message]:
        """
        Consume and return all pending messages for *agent_id*.

        After this call the agent's inbox is empty until new messages
        arrive.
        """
        pending = list(self._inbox.get(agent_id, []))
        self._inbox[agent_id] = []
        return pending

    def broadcast(
        self,
        sender_id: str,
        content: str,
        msg_type: MessageType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send a message from *sender_id* to every agent it is connected to
        in the topology.
        """
        ts = time.monotonic()
        meta = metadata or {}
        for target_id in self._topology.neighbours(sender_id):
            msg = Message(
                sender_id=sender_id,
                receiver_id=target_id,
                content=content,
                msg_type=msg_type,
                timestamp=ts,
                metadata=dict(meta),
            )
            self.send(msg)

    def get_history(self, agent_id: str) -> List[Message]:
        """
        Return all successfully delivered messages where *agent_id* was
        either the sender or the receiver, in chronological order.
        """
        return [
            m for m in self._log
            if m.sender_id == agent_id or m.receiver_id == agent_id
        ]

    def clear(self) -> None:
        """Wipe all inboxes, logs, and dropped-message records."""
        self._inbox.clear()
        self._log.clear()
        self.dropped_messages.clear()


# ======================================================================
# CommunicationProtocol (abstract + two concrete subclasses)
# ======================================================================

class CommunicationProtocol(abc.ABC):
    """
    Defines the communication flow for one decision round.

    Subclasses implement ``run_round`` which orchestrates messages
    between team agents and returns a ``RoundResult`` containing
    the chosen action plus diagnostics.
    """

    @abc.abstractmethod
    def run_round(
        self,
        agents: Dict[str, Any],       # agent_id -> BaseAgent instance
        topology: Topology,
        game_state: str,
        legal_moves: List[Action],
    ) -> RoundResult:
        """
        Execute one communication round and produce a team action.

        Parameters
        ----------
        agents : dict
            Mapping agent_id -> BaseAgent.  The agents must have roles
            assigned in the topology.
        topology : Topology
            The team communication topology.
        game_state : str
            The text state visible to the team.
        legal_moves : list
            Legal actions available this turn.

        Returns
        -------
        RoundResult
        """


# ---------------------------------------------------------------------------
# Helpers shared by protocol implementations
# ---------------------------------------------------------------------------

def _format_proposals_for_prompt(proposals: List[Message]) -> str:
    """Render proposal messages into a readable block for prompt injection."""
    if not proposals:
        return "(no proposals received)"
    lines: List[str] = []
    for i, msg in enumerate(proposals, 1):
        lines.append(f"Proposal {i} (from {msg.sender_id}): {msg.content}")
    return "\n".join(lines)


def _format_verdicts_for_prompt(verdicts: List[Message]) -> str:
    """Render verdict messages into a readable block for prompt injection."""
    if not verdicts:
        return "(no verdicts received)"
    lines: List[str] = []
    for i, msg in enumerate(verdicts, 1):
        lines.append(f"Verdict {i} (from {msg.sender_id}): {msg.content}")
    return "\n".join(lines)


def _pick_action_from_text(raw_text: str, legal_moves: List[Action]) -> Action:
    """
    Best-effort extraction of a legal action from free-form agent output.

    Strategy:
      1. Exact match (case-insensitive) against str(move).
      2. Substring containment — first legal move whose string form appears
         in the raw text wins (longest match first to avoid partial hits).
      3. Fallback: first legal move.
    """
    text_lower = raw_text.strip().lower()

    # Exact match
    for move in legal_moves:
        if text_lower == str(move).lower():
            return move

    # Substring match (longest string form first to prefer specific matches)
    sorted_moves = sorted(legal_moves, key=lambda m: len(str(m)), reverse=True)
    for move in sorted_moves:
        if str(move).lower() in text_lower:
            return move

    # Fallback
    return legal_moves[0] if legal_moves else None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SynchronousProtocol
# ---------------------------------------------------------------------------

class SynchronousProtocol(CommunicationProtocol):
    """
    Single-pass synchronous protocol.

    Flow
    ----
    1. **Workers** each independently propose an action (PROPOSAL).
    2. **Validators** each review all proposals and emit a VERDICT.
    3. **Thinker** reads all proposals + verdicts and emits the final
       READOUT which determines the team action.

    If there is no thinker, the first validator's preferred action is used.
    If there are no validators, the first worker's proposal is used.
    """

    def run_round(
        self,
        agents: Dict[str, Any],
        topology: Topology,
        game_state: str,
        legal_moves: List[Action],
    ) -> RoundResult:
        t0 = time.monotonic()
        channel = MessageChannel(topology)
        n_llm_calls = 0
        all_messages: List[Message] = []

        moves_str = ", ".join(str(m) for m in legal_moves)

        # --- Phase 1: Workers propose ---
        worker_ids = topology.workers
        proposals: List[Message] = []
        for wid in worker_ids:
            agent = agents[wid]
            prompt = (
                f"You are a WORKER agent. Analyze the game state and propose "
                f"the best action.\n\n"
                f"Game state:\n{game_state}\n\n"
                f"Legal moves: {moves_str}\n\n"
                f"Reply with your proposed action and a brief justification."
            )
            response = agent.choose_action(
                observation=game_state,
                text_state=prompt,
                legal_moves=legal_moves,
                game_rules="",
            )
            n_llm_calls += 1

            # Broadcast proposal to all connected agents
            ts = time.monotonic()
            for target in topology.neighbours(wid):
                msg = Message(
                    sender_id=wid,
                    receiver_id=target,
                    content=str(response),
                    msg_type=MessageType.PROPOSAL,
                    timestamp=ts,
                    metadata={"action": response},
                )
                channel.send(msg)
                proposals.append(msg)
                all_messages.append(msg)

        # --- Phase 2: Validators review ---
        validator_ids = topology.validators
        verdicts: List[Message] = []
        for vid in validator_ids:
            incoming = channel.receive(vid)
            proposals_text = _format_proposals_for_prompt(incoming)
            agent = agents[vid]
            prompt = (
                f"You are a VALIDATOR agent. Review these proposals and decide "
                f"which action is best.\n\n"
                f"Game state:\n{game_state}\n\n"
                f"Legal moves: {moves_str}\n\n"
                f"Proposals:\n{proposals_text}\n\n"
                f"Reply with the best action and your reasoning."
            )
            response = agent.choose_action(
                observation=game_state,
                text_state=prompt,
                legal_moves=legal_moves,
                game_rules="",
            )
            n_llm_calls += 1

            ts = time.monotonic()
            for target in topology.neighbours(vid):
                msg = Message(
                    sender_id=vid,
                    receiver_id=target,
                    content=str(response),
                    msg_type=MessageType.VERDICT,
                    timestamp=ts,
                    metadata={"action": response},
                )
                channel.send(msg)
                verdicts.append(msg)
                all_messages.append(msg)

        # --- Phase 3: Thinker readout ---
        thinker_ids = topology.thinkers
        final_action: Action

        if thinker_ids:
            tid = thinker_ids[0]  # single thinker per team
            incoming = channel.receive(tid)
            # Separate proposals and verdicts from the inbox
            inbox_proposals = [m for m in incoming if m.msg_type == MessageType.PROPOSAL]
            inbox_verdicts = [m for m in incoming if m.msg_type == MessageType.VERDICT]

            proposals_text = _format_proposals_for_prompt(inbox_proposals)
            verdicts_text = _format_verdicts_for_prompt(inbox_verdicts)

            agent = agents[tid]
            prompt = (
                f"You are the THINKER agent. Based on the proposals and "
                f"validator verdicts, choose the final team action.\n\n"
                f"Game state:\n{game_state}\n\n"
                f"Legal moves: {moves_str}\n\n"
                f"Proposals:\n{proposals_text}\n\n"
                f"Verdicts:\n{verdicts_text}\n\n"
                f"Reply with ONLY the chosen action."
            )
            response = agent.choose_action(
                observation=game_state,
                text_state=prompt,
                legal_moves=legal_moves,
                game_rules="",
            )
            n_llm_calls += 1

            final_action = _pick_action_from_text(str(response), legal_moves)

            ts = time.monotonic()
            readout_msg = Message(
                sender_id=tid,
                receiver_id="__team__",
                content=str(final_action),
                msg_type=MessageType.READOUT,
                timestamp=ts,
                metadata={"action": final_action},
            )
            all_messages.append(readout_msg)

        elif verdicts:
            # No thinker: use first validator's verdict
            final_action = _pick_action_from_text(
                verdicts[0].content, legal_moves
            )
        elif proposals:
            # No validators or thinker: use first worker's proposal
            final_action = _pick_action_from_text(
                proposals[0].content, legal_moves
            )
        else:
            # Absolute fallback
            final_action = legal_moves[0] if legal_moves else None  # type: ignore[assignment]

        wall_time = time.monotonic() - t0
        return RoundResult(
            messages=all_messages,
            final_action=final_action,
            n_llm_calls=n_llm_calls,
            n_rounds=1,
            wall_time=wall_time,
        )


# ---------------------------------------------------------------------------
# IterativeProtocol
# ---------------------------------------------------------------------------

class IterativeProtocol(CommunicationProtocol):
    """
    Multi-pass iterative protocol with revision.

    Flow (repeated up to ``max_rounds`` times)
    -------------------------------------------
    1. **Workers** propose (or revise their proposals if feedback exists).
    2. **Validators** review and may request revisions (VERDICT).
    3. **Thinker** decides whether to accept or request another round.

    The loop exits early when the thinker accepts or when ``max_rounds``
    is reached.

    Parameters
    ----------
    max_rounds : int
        Maximum number of W -> V -> T passes. Defaults to 3.
    """

    def __init__(self, max_rounds: int = 3) -> None:
        self.max_rounds = max(1, max_rounds)

    def run_round(
        self,
        agents: Dict[str, Any],
        topology: Topology,
        game_state: str,
        legal_moves: List[Action],
    ) -> RoundResult:
        t0 = time.monotonic()
        channel = MessageChannel(topology)
        n_llm_calls = 0
        all_messages: List[Message] = []
        n_rounds_done = 0

        moves_str = ", ".join(str(m) for m in legal_moves)
        final_action: Action = legal_moves[0] if legal_moves else None  # type: ignore[assignment]

        # Track per-worker feedback across rounds for revision prompts.
        worker_feedback: Dict[str, List[str]] = defaultdict(list)

        for round_idx in range(self.max_rounds):
            n_rounds_done += 1
            channel.clear()

            # --- Phase 1: Workers propose / revise ---
            worker_ids = topology.workers
            proposals: List[Message] = []
            for wid in worker_ids:
                agent = agents[wid]
                feedback_block = ""
                if worker_feedback[wid]:
                    feedback_block = (
                        "\n\nPrevious feedback you should address:\n"
                        + "\n".join(worker_feedback[wid])
                    )
                prompt = (
                    f"You are a WORKER agent (round {round_idx + 1})."
                    f" Analyze the game state and propose the best action.\n\n"
                    f"Game state:\n{game_state}\n\n"
                    f"Legal moves: {moves_str}"
                    f"{feedback_block}\n\n"
                    f"Reply with your proposed action and a brief justification."
                )
                response = agent.choose_action(
                    observation=game_state,
                    text_state=prompt,
                    legal_moves=legal_moves,
                    game_rules="",
                )
                n_llm_calls += 1

                ts = time.monotonic()
                for target in topology.neighbours(wid):
                    msg = Message(
                        sender_id=wid,
                        receiver_id=target,
                        content=str(response),
                        msg_type=MessageType.PROPOSAL,
                        timestamp=ts,
                        metadata={"action": response, "round": round_idx + 1},
                    )
                    channel.send(msg)
                    proposals.append(msg)
                    all_messages.append(msg)

            # --- Phase 2: Validators review ---
            validator_ids = topology.validators
            verdicts: List[Message] = []
            for vid in validator_ids:
                incoming = channel.receive(vid)
                proposals_text = _format_proposals_for_prompt(incoming)
                agent = agents[vid]
                prompt = (
                    f"You are a VALIDATOR agent (round {round_idx + 1})."
                    f" Review these proposals.\n\n"
                    f"Game state:\n{game_state}\n\n"
                    f"Legal moves: {moves_str}\n\n"
                    f"Proposals:\n{proposals_text}\n\n"
                    f"If a proposal is good, reply with ACCEPT and the action. "
                    f"If not, reply with REVISE and feedback for the workers."
                )
                response = agent.choose_action(
                    observation=game_state,
                    text_state=prompt,
                    legal_moves=legal_moves,
                    game_rules="",
                )
                n_llm_calls += 1

                ts = time.monotonic()
                for target in topology.neighbours(vid):
                    msg = Message(
                        sender_id=vid,
                        receiver_id=target,
                        content=str(response),
                        msg_type=MessageType.VERDICT,
                        timestamp=ts,
                        metadata={"action": response, "round": round_idx + 1},
                    )
                    channel.send(msg)
                    verdicts.append(msg)
                    all_messages.append(msg)

            # --- Phase 3: Thinker decision ---
            thinker_ids = topology.thinkers
            accepted = False

            if thinker_ids:
                tid = thinker_ids[0]
                incoming = channel.receive(tid)
                inbox_proposals = [m for m in incoming if m.msg_type == MessageType.PROPOSAL]
                inbox_verdicts = [m for m in incoming if m.msg_type == MessageType.VERDICT]

                proposals_text = _format_proposals_for_prompt(inbox_proposals)
                verdicts_text = _format_verdicts_for_prompt(inbox_verdicts)

                agent = agents[tid]
                prompt = (
                    f"You are the THINKER agent (round {round_idx + 1} of "
                    f"{self.max_rounds}).\n\n"
                    f"Game state:\n{game_state}\n\n"
                    f"Legal moves: {moves_str}\n\n"
                    f"Proposals:\n{proposals_text}\n\n"
                    f"Verdicts:\n{verdicts_text}\n\n"
                    f"Choose the final action if satisfied, or request "
                    f"another revision round. Reply with ACCEPT <action> "
                    f"or REVISE <feedback>."
                )
                response = agent.choose_action(
                    observation=game_state,
                    text_state=prompt,
                    legal_moves=legal_moves,
                    game_rules="",
                )
                n_llm_calls += 1
                response_str = str(response)

                # Parse thinker decision
                response_lower = response_str.strip().lower()
                if response_lower.startswith("revise") and round_idx < self.max_rounds - 1:
                    # Distribute feedback to workers for next iteration
                    feedback_text = response_str.strip()
                    for wid in worker_ids:
                        worker_feedback[wid].append(
                            f"Thinker (round {round_idx + 1}): {feedback_text}"
                        )
                    # Also forward any validator feedback to workers
                    for v_msg in verdicts:
                        v_lower = v_msg.content.strip().lower()
                        if v_lower.startswith("revise"):
                            for wid in worker_ids:
                                worker_feedback[wid].append(
                                    f"Validator {v_msg.sender_id} "
                                    f"(round {round_idx + 1}): {v_msg.content}"
                                )
                else:
                    # Accept (or last round)
                    accepted = True
                    final_action = _pick_action_from_text(response_str, legal_moves)
                    ts = time.monotonic()
                    readout_msg = Message(
                        sender_id=tid,
                        receiver_id="__team__",
                        content=str(final_action),
                        msg_type=MessageType.READOUT,
                        timestamp=ts,
                        metadata={"action": final_action, "round": round_idx + 1},
                    )
                    all_messages.append(readout_msg)
            else:
                # No thinker: auto-accept after first pass
                accepted = True
                if verdicts:
                    final_action = _pick_action_from_text(
                        verdicts[0].content, legal_moves
                    )
                elif proposals:
                    final_action = _pick_action_from_text(
                        proposals[0].content, legal_moves
                    )

            if accepted:
                break

        wall_time = time.monotonic() - t0
        return RoundResult(
            messages=all_messages,
            final_action=final_action,
            n_llm_calls=n_llm_calls,
            n_rounds=n_rounds_done,
            wall_time=wall_time,
        )
