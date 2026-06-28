"""
Topology/graph module for LLM-TeamGym.

Defines the graph structure connecting agents in a role-structured team
(Worker / Validator / Thinker architecture).  Workers produce candidate
outputs, Validators score and critique them, and Thinkers provide high-level
strategic guidance.

The core data structure is a bipartite graph between Workers (W) and
Validators (V), represented as a binary matrix B of shape (W, V) where
B[w, v] = 1 indicates Worker w sends its output to Validator v for
evaluation.

Topology *shapes* are equivalence classes of topologies under independent
permutation of Worker indices and Validator indices.  Shapes are counted
efficiently via Burnside's lemma applied to the S_W x S_V group action.
"""

from __future__ import annotations

import itertools
from collections import Counter
from math import gcd, factorial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TopologyGraph",
    "TopologyShape",
    "enumerate_all_topologies",
    "count_shapes",
    "star_topology",
    "random_topology",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _partitions(n: int):
    """Yield every integer partition of *n* as a tuple in descending order.

    Example: ``_partitions(4)`` yields ``(4,), (3,1), (2,2), (2,1,1),
    (1,1,1,1)``.
    """
    if n == 0:
        yield ()
        return

    def _helper(remaining: int, max_part: int):
        if remaining == 0:
            yield ()
            return
        for part in range(min(remaining, max_part), 0, -1):
            for rest in _helper(remaining - part, part):
                yield (part,) + rest

    yield from _helper(n, n)


def _partition_count(n: int, partition: Tuple[int, ...]) -> int:
    """Number of permutations in S_n whose cycle type equals *partition*.

    Formula: ``n! / prod_k (k^{c_k} * c_k!)`` where ``c_k`` counts parts
    equal to ``k``.
    """
    counts = Counter(partition)
    denom = 1
    for k, c in counts.items():
        denom *= (k ** c) * factorial(c)
    return factorial(n) // denom


def _canonical_form(B: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    """Lexicographically smallest matrix obtainable by permuting rows and cols.

    Tries all W! * V! permutations (acceptable for small W, V).
    """
    W, V = B.shape
    if W == 0 or V == 0:
        return tuple(tuple(int(x) for x in row) for row in B)
    best: Optional[Tuple[Tuple[int, ...], ...]] = None
    for row_perm in itertools.permutations(range(W)):
        for col_perm in itertools.permutations(range(V)):
            permuted = tuple(
                tuple(int(B[row_perm[i], col_perm[j]]) for j in range(V))
                for i in range(W)
            )
            if best is None or permuted < best:
                best = permuted
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# TopologyGraph
# ---------------------------------------------------------------------------


class TopologyGraph:
    """Bipartite graph connecting Workers to Validators in a team.

    The graph is stored as :pyattr:`bipartite_matrix`, a binary
    :class:`numpy.ndarray` of shape ``(n_workers, n_validators)``.
    Entry ``[w, v] = 1`` means Worker *w* sends output to Validator *v*.

    Thinker nodes are *not* part of the bipartite matrix; they connect
    to every Validator in the full adjacency view returned by
    :meth:`to_adjacency_matrix`.

    Parameters
    ----------
    n_workers : int
        Number of Worker agents (W).
    n_validators : int
        Number of Validator agents (V).
    n_thinkers : int
        Number of Thinker agents (T), default 1.
    """

    def __init__(
        self,
        n_workers: int,
        n_validators: int,
        n_thinkers: int = 1,
    ) -> None:
        if n_workers < 0 or n_validators < 0 or n_thinkers < 0:
            raise ValueError("Agent counts must be non-negative.")
        self.n_workers = n_workers
        self.n_validators = n_validators
        self.n_thinkers = n_thinkers
        self.bipartite_matrix: np.ndarray = np.zeros(
            (n_workers, n_validators), dtype=int
        )

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def from_matrix(
        cls, B: np.ndarray, n_thinkers: int = 1
    ) -> "TopologyGraph":
        """Create a topology from an explicit bipartite matrix *B* of shape (W, V)."""
        B = np.asarray(B, dtype=int)
        if B.ndim != 2:
            raise ValueError("Bipartite matrix must be 2-dimensional.")
        g = cls(B.shape[0], B.shape[1], n_thinkers)
        g.bipartite_matrix = B.copy()
        return g

    @classmethod
    def from_star(
        cls,
        n_workers: int,
        n_validators: int,
        n_thinkers: int = 1,
    ) -> "TopologyGraph":
        """Complete bipartite topology -- every Worker connects to every Validator."""
        g = cls(n_workers, n_validators, n_thinkers)
        g.bipartite_matrix[:] = 1
        return g

    @classmethod
    def from_chain(
        cls,
        n_workers: int,
        n_validators: int,
        n_thinkers: int = 1,
    ) -> "TopologyGraph":
        """Chain (bipartite path) topology: ``W0-V0-W1-V1-...``

        Worker *i* connects to Validator *i* (forward edge, if ``i < V``)
        and to Validator *i - 1* (backward edge, if ``i >= 1`` and
        ``i - 1 < V``).  Workers with indices beyond the Validator range
        that would otherwise be disconnected fall back to connecting to the
        last Validator.
        """
        g = cls(n_workers, n_validators, n_thinkers)
        if n_validators == 0:
            return g
        for w in range(n_workers):
            connected = False
            if w < n_validators:
                g.bipartite_matrix[w, w] = 1
                connected = True
            if w >= 1 and (w - 1) < n_validators:
                g.bipartite_matrix[w, w - 1] = 1
                connected = True
            if not connected:
                g.bipartite_matrix[w, n_validators - 1] = 1
        return g

    @classmethod
    def from_ring(
        cls,
        n_workers: int,
        n_validators: int,
        n_thinkers: int = 1,
    ) -> "TopologyGraph":
        """Ring (cyclic bipartite) topology.

        Worker *i* connects to Validator ``i % V`` and Validator
        ``(i - 1) % V``, forming a wrap-around ring.
        """
        g = cls(n_workers, n_validators, n_thinkers)
        if n_validators == 0:
            return g
        for w in range(n_workers):
            g.bipartite_matrix[w, w % n_validators] = 1
            g.bipartite_matrix[w, (w - 1) % n_validators] = 1
        return g

    # -- Edge manipulation --------------------------------------------------

    def add_edge(self, worker_idx: int, validator_idx: int) -> None:
        """Add a Worker-to-Validator connection."""
        self.bipartite_matrix[worker_idx, validator_idx] = 1

    def remove_edge(self, worker_idx: int, validator_idx: int) -> None:
        """Remove a Worker-to-Validator connection."""
        self.bipartite_matrix[worker_idx, validator_idx] = 0

    # -- Queries ------------------------------------------------------------

    def get_worker_connections(self, w_idx: int) -> List[int]:
        """Return Validator indices connected to Worker *w_idx*."""
        return list(np.where(self.bipartite_matrix[w_idx] == 1)[0])

    def get_validator_connections(self, v_idx: int) -> List[int]:
        """Return Worker indices connected to Validator *v_idx*."""
        return list(np.where(self.bipartite_matrix[:, v_idx] == 1)[0])

    def is_valid_relaxed(self) -> bool:
        """True iff every Worker has at least one Validator."""
        if self.n_workers == 0:
            return True
        return bool(np.all(self.bipartite_matrix.sum(axis=1) >= 1))

    def is_valid_strict(self) -> bool:
        """True iff every Worker has >= 1 Validator AND every Validator has >= 1 Worker."""
        if not self.is_valid_relaxed():
            return False
        if self.n_validators == 0:
            return True
        return bool(np.all(self.bipartite_matrix.sum(axis=0) >= 1))

    @property
    def n_edges(self) -> int:
        """Total number of Worker-Validator edges."""
        return int(np.sum(self.bipartite_matrix))

    # -- Full adjacency views -----------------------------------------------

    def to_adjacency_matrix(self) -> np.ndarray:
        """Symmetric ``(W+V+T) x (W+V+T)`` adjacency matrix.

        Node ordering:

        * Workers: ``[0, W)``
        * Validators: ``[W, W+V)``
        * Thinkers: ``[W+V, W+V+T)``

        Includes Worker-Validator edges from :pyattr:`bipartite_matrix`
        (symmetric) and Thinker-Validator edges (all Thinkers connect to all
        Validators, representing the strategic-oversight and user-interface
        layer).
        """
        W, V, T = self.n_workers, self.n_validators, self.n_thinkers
        N = W + V + T
        A = np.zeros((N, N), dtype=int)
        # Worker <-> Validator
        A[:W, W:W + V] = self.bipartite_matrix
        A[W:W + V, :W] = self.bipartite_matrix.T
        # Thinker <-> Validator (all-to-all)
        if T > 0 and V > 0:
            A[W + V:, W:W + V] = 1
            A[W:W + V, W + V:] = 1
        return A

    def to_networkx(self):
        """Convert to a ``networkx.DiGraph`` (requires *networkx*).

        Nodes are labelled ``W0, W1, ..., V0, V1, ..., T0, T1, ...`` and
        carry a ``role`` attribute (``"worker"``, ``"validator"``, or
        ``"thinker"``).  Worker-Validator and Thinker-Validator connections
        are represented as bidirectional directed edges.
        """
        try:
            import networkx as nx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "networkx is required for to_networkx()"
            ) from exc

        G = nx.DiGraph()
        w_labels = [f"W{i}" for i in range(self.n_workers)]
        v_labels = [f"V{j}" for j in range(self.n_validators)]
        t_labels = [f"T{k}" for k in range(self.n_thinkers)]

        for lbl in w_labels:
            G.add_node(lbl, role="worker")
        for lbl in v_labels:
            G.add_node(lbl, role="validator")
        for lbl in t_labels:
            G.add_node(lbl, role="thinker")

        # Worker <-> Validator
        for w in range(self.n_workers):
            for v in range(self.n_validators):
                if self.bipartite_matrix[w, v]:
                    G.add_edge(w_labels[w], v_labels[v])
                    G.add_edge(v_labels[v], w_labels[w])

        # Thinker <-> Validator
        for t in range(self.n_thinkers):
            for v in range(self.n_validators):
                G.add_edge(t_labels[t], v_labels[v])
                G.add_edge(v_labels[v], t_labels[t])

        return G

    # -- Equality / hashing -------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopologyGraph):
            return NotImplemented
        return (
            self.n_workers == other.n_workers
            and self.n_validators == other.n_validators
            and self.n_thinkers == other.n_thinkers
            and np.array_equal(self.bipartite_matrix, other.bipartite_matrix)
        )

    def __hash__(self) -> int:
        return hash((
            self.n_workers,
            self.n_validators,
            self.n_thinkers,
            self.bipartite_matrix.tobytes(),
        ))

    def __repr__(self) -> str:
        return (
            f"TopologyGraph(W={self.n_workers}, V={self.n_validators}, "
            f"T={self.n_thinkers}, edges={self.n_edges})"
        )

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "n_workers": self.n_workers,
            "n_validators": self.n_validators,
            "n_thinkers": self.n_thinkers,
            "bipartite_matrix": self.bipartite_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TopologyGraph":
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        g = cls(d["n_workers"], d["n_validators"], d["n_thinkers"])
        g.bipartite_matrix = np.array(d["bipartite_matrix"], dtype=int)
        return g


# ---------------------------------------------------------------------------
# TopologyShape
# ---------------------------------------------------------------------------


class TopologyShape:
    """Equivalence class of topologies under role-preserving isomorphism.

    Two topologies share the same *shape* iff one can be obtained from the
    other by independently permuting Worker indices and Validator indices.
    The canonical form is the lexicographically smallest bipartite matrix
    in the equivalence class (rows and columns sorted).
    """

    def __init__(self, topology: TopologyGraph) -> None:
        self._n_workers = topology.n_workers
        self._n_validators = topology.n_validators
        self._canonical: Tuple[Tuple[int, ...], ...] = _canonical_form(
            topology.bipartite_matrix
        )

    def canonical_form(self) -> np.ndarray:
        """Return the canonical (normalized) bipartite matrix."""
        return np.array(self._canonical, dtype=int)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopologyShape):
            return NotImplemented
        return (
            self._n_workers == other._n_workers
            and self._n_validators == other._n_validators
            and self._canonical == other._canonical
        )

    def __hash__(self) -> int:
        return hash((self._n_workers, self._n_validators, self._canonical))

    def __repr__(self) -> str:
        rows = ", ".join(str(list(r)) for r in self._canonical)
        return (
            f"TopologyShape(W={self._n_workers}, V={self._n_validators}, "
            f"canonical=[{rows}])"
        )


# ---------------------------------------------------------------------------
# Enumeration and counting utilities
# ---------------------------------------------------------------------------


def enumerate_all_topologies(
    W: int, V: int, relaxed: bool = True
) -> List[TopologyGraph]:
    """Generate every valid W x V bipartite connection pattern.

    Returns *all* distinct binary matrices satisfying the validity
    constraint (including isomorphic duplicates).  To collapse into unique
    shapes, wrap each result in :class:`TopologyShape` and collect into a
    set.

    Parameters
    ----------
    W : int
        Number of Workers.
    V : int
        Number of Validators.
    relaxed : bool
        If True, require every Worker has >= 1 Validator.
        If False, additionally require every Validator has >= 1 Worker.

    Raises
    ------
    ValueError
        When ``W * V > 20`` (to prevent combinatorial explosion).
    """
    if W * V > 20:
        raise ValueError(
            f"W*V = {W * V} exceeds the brute-force limit of 20.  "
            f"Use count_shapes() for efficient counting."
        )
    # Edge cases
    if V == 0:
        if W == 0:
            return [TopologyGraph(0, 0)]
        return []  # workers exist but no validators -> never valid
    if W == 0:
        if relaxed:
            return [TopologyGraph(0, V)]
        return []  # strict requires each validator has >= 1 worker

    results: List[TopologyGraph] = []
    total_bits = W * V
    for bits in range(1 << total_bits):
        B = np.zeros((W, V), dtype=int)
        for idx in range(total_bits):
            if bits & (1 << idx):
                B[idx // V, idx % V] = 1
        g = TopologyGraph.from_matrix(B)
        valid = g.is_valid_relaxed() if relaxed else g.is_valid_strict()
        if valid:
            results.append(g)
    return results


def count_shapes(W: int, V: int, relaxed: bool = True) -> int:
    """Count distinct topology shapes via Burnside's lemma.

    The group ``S_W x S_V`` acts on ``W x V`` binary matrices by
    independently permuting rows (Workers) and columns (Validators).
    Two matrices in the same orbit define the same *shape*.

    For **relaxed** validity the constraint is that every row (Worker) has
    at least one ``1``.  For **strict** validity, every column (Validator)
    must additionally have at least one ``1``.

    The fixed-point count per conjugacy class ``(lambda_W, lambda_V)``
    decomposes as follows.  Let ``lambda_W = (a_1, ..., a_k)`` and
    ``lambda_V = (b_1, ..., b_l)`` be the cycle-type partitions.

    *Relaxed*::

        F = prod_{i=1}^{k} (2^{sum_j gcd(a_i, b_j)} - 1)

    *Strict* (inclusion--exclusion over row-cycles *and* col-cycles)::

        F = sum_{S subset row-cycles} sum_{T subset col-cycles}
            (-1)^{|S|+|T|}  2^{sum_{i not in S, j not in T} gcd(a_i, b_j)}

    Parameters
    ----------
    W, V : int
        Number of Workers and Validators.
    relaxed : bool
        Validity mode (default True).

    Returns
    -------
    int

    Examples
    --------
    >>> count_shapes(4, 1)
    1
    >>> count_shapes(4, 2)
    9
    >>> count_shapes(4, 3)
    51
    >>> sum(count_shapes(4, v) for v in range(1, 4))
    61
    """
    # Edge cases
    if W == 0:
        if relaxed:
            return 1
        return 1 if V == 0 else 0
    if V == 0:
        return 0  # workers exist, no validators -> no valid topology

    total = 0
    partitions_w = list(_partitions(W))
    partitions_v = list(_partitions(V))

    for p_w in partitions_w:
        n_w = _partition_count(W, p_w)
        for p_v in partitions_v:
            n_v = _partition_count(V, p_v)

            if relaxed:
                # Product formula over row-cycles
                fixed = 1
                for a in p_w:
                    n_orbits = sum(gcd(a, b) for b in p_v)
                    fixed *= (2 ** n_orbits) - 1
            else:
                # Inclusion-exclusion over row-cycles AND col-cycles
                k, ell = len(p_w), len(p_v)
                fixed = 0
                for s_mask in range(1 << k):
                    s_bits = bin(s_mask).count("1")
                    s_sign = (-1) ** s_bits
                    for t_mask in range(1 << ell):
                        t_bits = bin(t_mask).count("1")
                        sign = s_sign * ((-1) ** t_bits)
                        free = 0
                        for i in range(k):
                            if s_mask & (1 << i):
                                continue
                            for j in range(ell):
                                if t_mask & (1 << j):
                                    continue
                                free += gcd(p_w[i], p_v[j])
                        fixed += sign * (2 ** free)

            total += n_w * n_v * fixed

    return total // (factorial(W) * factorial(V))


def star_topology(W: int, V: int, T: int = 1) -> TopologyGraph:
    """Create a complete bipartite (star) topology."""
    return TopologyGraph.from_star(W, V, T)


def random_topology(
    W: int,
    V: int,
    edge_prob: float = 0.5,
    seed: Optional[int] = None,
) -> TopologyGraph:
    """Sample a random topology with relaxed validity guaranteed.

    Each potential edge is included independently with probability
    *edge_prob*.  Any Worker left with zero connections is assigned one
    random Validator to restore relaxed validity.

    Parameters
    ----------
    W, V : int
        Number of Workers and Validators.
    edge_prob : float
        Per-edge inclusion probability (before validity fix-up).
    seed : int or None
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    B = (rng.random((W, V)) < edge_prob).astype(int)
    # Guarantee relaxed validity
    for w in range(W):
        if not np.any(B[w]):
            B[w, rng.randint(V)] = 1
    return TopologyGraph.from_matrix(B)
