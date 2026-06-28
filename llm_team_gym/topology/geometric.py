"""Geometric topology generation for LLM agent teams.

Implements the medium-scale geometric generation pipeline from the
topology-aware LLM agent teams methodology:

1. Role sampling with categorical distribution and coverage guarantee
2. Spatial placement in the unit square
3. k-nearest-neighbor proximity graph construction
4. Role-hierarchy edge orientation (W -> V -> T)
5. Readout invariant enforcement (all Workers reach a Validator,
   all Validators reach a Thinker)
6. Spectral fingerprint delegation to spectral.py
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Role hierarchy: Worker -> Validator -> Thinker
# An edge from role A to role B is allowed iff ROLE_ORDER[A] < ROLE_ORDER[B].
# ---------------------------------------------------------------------------
ROLE_ORDER: Dict[str, int] = {"W": 0, "V": 1, "T": 2}
ROLE_LABELS: List[str] = ["W", "V", "T"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AgentNode:
    """A single agent in the geometric topology.

    Attributes
    ----------
    agent_id : str
        Unique identifier (UUID4 hex by default).
    role : str
        One of ``'W'`` (Worker), ``'V'`` (Validator), ``'T'`` (Thinker).
    position : np.ndarray
        2-D coordinate in [0, 1]^2.  Assigned during placement.
    """

    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: str = "W"
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        if self.role not in ROLE_LABELS:
            raise ValueError(
                f"Invalid role '{self.role}'; expected one of {ROLE_LABELS}"
            )
        self.position = np.asarray(self.position, dtype=np.float64)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class GeometricGenerator:
    """Full geometric topology generation pipeline.

    Parameters
    ----------
    n_agents : int
        Total number of agents (must be >= 3 so each role can appear).
    p_thinker : float
        Probability weight for the Thinker role.
    p_worker : float
        Probability weight for the Worker role.
    p_validator : float
        Probability weight for the Validator role.
    k_neighbors : int
        Number of nearest neighbors for the k-NN graph.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_agents: int = 6,
        p_thinker: float = 0.1,
        p_worker: float = 0.5,
        p_validator: float = 0.4,
        k_neighbors: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        if n_agents < 3:
            raise ValueError("n_agents must be >= 3 to guarantee all roles")
        self.n_agents = n_agents
        # Normalise probabilities so they sum to 1.
        total = p_thinker + p_worker + p_validator
        self.p_thinker = p_thinker / total
        self.p_worker = p_worker / total
        self.p_validator = p_validator / total
        self.k_neighbors = k_neighbors
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1 -- role sampling
    # ------------------------------------------------------------------
    def sample_roles(self) -> List[AgentNode]:
        """Draw *n_agents* agents with roles from categorical(p_T, p_W, p_V).

        At least one agent of each role is guaranteed.  The method first
        assigns one agent per role, then samples the remaining agents
        from the categorical distribution.

        Returns
        -------
        agents : list[AgentNode]
        """
        n = self.n_agents

        # Guarantee one of each role.
        guaranteed_roles: List[str] = ["T", "W", "V"]
        remaining = n - len(guaranteed_roles)

        sampled_roles = list(
            self.rng.choice(
                ROLE_LABELS,
                size=remaining,
                p=[self.p_worker, self.p_validator, self.p_thinker],
            )
        )
        all_roles = guaranteed_roles + sampled_roles
        self.rng.shuffle(all_roles)

        agents = [AgentNode(role=r) for r in all_roles]
        return agents

    # ------------------------------------------------------------------
    # Step 2 -- spatial placement
    # ------------------------------------------------------------------
    def place_agents(self, agents: List[AgentNode]) -> None:
        """Assign each agent a random position in [0, 1]^2 (in-place)."""
        for agent in agents:
            agent.position = self.rng.uniform(0.0, 1.0, size=2)

    # ------------------------------------------------------------------
    # Step 3 -- k-nearest-neighbor graph
    # ------------------------------------------------------------------
    def build_knn_graph(self, agents: List[AgentNode]) -> np.ndarray:
        """Build a symmetric adjacency matrix from k-NN over agent positions.

        Parameters
        ----------
        agents : list[AgentNode]

        Returns
        -------
        adj : np.ndarray of shape (n, n)
            Binary symmetric adjacency matrix (undirected k-NN graph).
        """
        n = len(agents)
        positions = np.array([a.position for a in agents])
        tree = KDTree(positions)

        # k+1 because query includes the point itself.
        k = min(self.k_neighbors + 1, n)
        _, indices = tree.query(positions, k=k)

        adj = np.zeros((n, n), dtype=np.float64)
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0  # symmetrise
        return adj

    # ------------------------------------------------------------------
    # Step 4 -- role-based orientation
    # ------------------------------------------------------------------
    def orient_edges(
        self, adj: np.ndarray, agents: List[AgentNode]
    ) -> np.ndarray:
        """Orient edges by role hierarchy and prune same-role edges.

        Allowed direction: W -> V -> T (lower order to higher order).
        Same-role edges are dropped.

        Parameters
        ----------
        adj : np.ndarray
            Undirected adjacency matrix.
        agents : list[AgentNode]

        Returns
        -------
        directed_adj : np.ndarray
            Directed adjacency matrix where ``directed_adj[i][j] = 1``
            means an edge from agent *i* to agent *j*.
        """
        n = len(agents)
        directed = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    continue
                ri = ROLE_ORDER[agents[i].role]
                rj = ROLE_ORDER[agents[j].role]
                if ri < rj:
                    directed[i, j] = 1.0  # i -> j
                elif rj < ri:
                    directed[j, i] = 1.0  # j -> i
                # same role: edge dropped
        return directed

    # ------------------------------------------------------------------
    # Step 5 -- readout invariant enforcement
    # ------------------------------------------------------------------
    def enforce_readout_invariant(
        self, adj: np.ndarray, agents: List[AgentNode]
    ) -> np.ndarray:
        """Repair the directed graph so every required path exists.

        Invariants:
        - Every Worker must be able to reach at least one Validator.
        - Every Validator must be able to reach at least one Thinker.

        Repair strategy: for each agent that violates the invariant,
        add a directed edge to the nearest agent of the required
        target role.

        Parameters
        ----------
        adj : np.ndarray
            Directed adjacency matrix (modified in-place and returned).
        agents : list[AgentNode]

        Returns
        -------
        adj : np.ndarray
            Repaired directed adjacency matrix.
        """
        n = len(agents)
        positions = np.array([a.position for a in agents])

        # Build index sets for each role.
        role_indices: Dict[str, List[int]] = {"W": [], "V": [], "T": []}
        for idx, a in enumerate(agents):
            role_indices[a.role].append(idx)

        def _can_reach_role(src: int, target_role: str) -> bool:
            """BFS from *src* following directed edges; return True if
            any node with *target_role* is reachable."""
            visited = set()
            queue = deque([src])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                if agents[node].role == target_role and node != src:
                    return True
                for nbr in range(n):
                    if adj[node, nbr] and nbr not in visited:
                        queue.append(nbr)
            return False

        def _nearest_of_role(src_idx: int, role: str) -> int:
            """Return the index of the nearest agent with given *role*."""
            candidates = role_indices[role]
            dists = np.linalg.norm(
                positions[candidates] - positions[src_idx], axis=1
            )
            return candidates[int(np.argmin(dists))]

        # Workers must reach a Validator.
        for w in role_indices["W"]:
            if not _can_reach_role(w, "V"):
                nearest_v = _nearest_of_role(w, "V")
                adj[w, nearest_v] = 1.0

        # Validators must reach a Thinker.
        for v in role_indices["V"]:
            if not _can_reach_role(v, "T"):
                nearest_t = _nearest_of_role(v, "T")
                adj[v, nearest_t] = 1.0

        return adj

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def generate(self) -> Tuple[List[AgentNode], np.ndarray]:
        """Run the complete geometric topology pipeline.

        Returns
        -------
        agents : list[AgentNode]
        adj : np.ndarray
            Directed adjacency matrix after orientation and invariant
            enforcement.
        """
        agents = self.sample_roles()
        self.place_agents(agents)
        adj = self.build_knn_graph(agents)
        adj = self.orient_edges(adj, agents)
        adj = self.enforce_readout_invariant(adj, agents)
        return agents, adj

    def generate_batch(
        self, n_topologies: int
    ) -> List[Tuple[List[AgentNode], np.ndarray]]:
        """Generate *n_topologies* independent topologies.

        Parameters
        ----------
        n_topologies : int

        Returns
        -------
        topologies : list of (agents, adj) tuples
        """
        return [self.generate() for _ in range(n_topologies)]


# ---------------------------------------------------------------------------
# Topology descriptor
# ---------------------------------------------------------------------------
class TopologyDescriptor:
    """Compute summary statistics of a directed topology.

    Parameters
    ----------
    adj : np.ndarray
        Directed adjacency matrix of shape ``(n, n)``.
    agents : list[AgentNode]
        Agent list aligned with *adj* indices.
    """

    def __init__(
        self, adj: np.ndarray, agents: List[AgentNode]
    ) -> None:
        self.adj = adj
        self.agents = agents
        self.n = len(agents)

    @property
    def node_count(self) -> int:
        """Total number of nodes."""
        return self.n

    @property
    def edge_count(self) -> int:
        """Total number of directed edges."""
        return int(np.sum(self.adj > 0))

    @property
    def degree_distribution(self) -> np.ndarray:
        """Out-degree + in-degree for every node."""
        out_deg = np.sum(self.adj > 0, axis=1)
        in_deg = np.sum(self.adj > 0, axis=0)
        return (out_deg + in_deg).astype(np.float64)

    @property
    def diameter(self) -> int:
        """Diameter of the underlying undirected graph.

        Uses BFS from each node.  If the graph is disconnected the
        diameter is reported as ``-1``.
        """
        # Symmetrise adjacency for diameter computation.
        sym = ((self.adj + self.adj.T) > 0).astype(np.float64)
        max_dist = 0
        for src in range(self.n):
            dist = self._bfs_distances(sym, src)
            if -1 in dist.values():
                return -1
            max_dist = max(max_dist, max(dist.values()))
        return int(max_dist)

    @staticmethod
    def _bfs_distances(
        adj: np.ndarray, src: int
    ) -> Dict[int, int]:
        """Return shortest-path distances from *src* in an undirected graph."""
        n = adj.shape[0]
        dist: Dict[int, int] = {src: 0}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nbr in range(n):
                if adj[node, nbr] > 0 and nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        # Mark unreachable nodes.
        for i in range(n):
            if i not in dist:
                dist[i] = -1
        return dist

    def feature_vector(self) -> np.ndarray:
        """Concatenate all descriptor features into a flat vector.

        Layout::

            [node_count, edge_count, diameter,
             degree_dist_mean, degree_dist_std, degree_dist_min, degree_dist_max,
             role_count_W, role_count_V, role_count_T]

        Returns
        -------
        vec : np.ndarray of shape ``(10,)``
        """
        dd = self.degree_distribution
        role_counts = np.zeros(3)
        for a in self.agents:
            role_counts[ROLE_ORDER[a.role]] += 1

        return np.array(
            [
                float(self.node_count),
                float(self.edge_count),
                float(self.diameter),
                float(np.mean(dd)),
                float(np.std(dd)),
                float(np.min(dd)),
                float(np.max(dd)),
                role_counts[0],  # W
                role_counts[1],  # V
                role_counts[2],  # T
            ]
        )


# ---------------------------------------------------------------------------
# Visualization utility
# ---------------------------------------------------------------------------
ROLE_COLORS: Dict[str, str] = {"W": "#1f77b4", "V": "#2ca02c", "T": "#d62728"}


def visualize_topology(
    agents: List[AgentNode],
    adj: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Geometric Topology",
) -> None:
    """Scatter plot of agents coloured by role with directed edges.

    Parameters
    ----------
    agents : list[AgentNode]
    adj : np.ndarray
        Directed adjacency matrix.
    save_path : str | None
        If given, save figure to this path instead of showing.
    title : str
        Plot title.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualize_topology; "
            "install it with: pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=(7, 7))
    n = len(agents)

    # Draw edges as arrows.
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                xi, yi = agents[i].position
                xj, yj = agents[j].position
                ax.annotate(
                    "",
                    xy=(xj, yj),
                    xytext=(xi, yi),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray",
                        alpha=0.5,
                        lw=0.8,
                    ),
                )

    # Draw nodes.
    for role in ROLE_LABELS:
        members = [a for a in agents if a.role == role]
        if not members:
            continue
        xs = [a.position[0] for a in members]
        ys = [a.position[1] for a in members]
        ax.scatter(
            xs,
            ys,
            c=ROLE_COLORS[role],
            label=role,
            s=120,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(title="Role")
    ax.grid(True, alpha=0.2)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
