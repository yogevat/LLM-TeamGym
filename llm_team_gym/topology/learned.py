"""
Controller-guided topology composition for large-scale (100+ agents) teams.

Implements the learned topology search methodology from the thesis:

    1. An LLM controller is prompted with a specification (roles, constraints,
       target task) and proposes a candidate topology.
    2. The candidate is instantiated, run, and scored via TournamentRunner.
    3. Results are logged with the topology's structural/spectral fingerprint.
    4. Topologies are clustered; best-performing clusters are identified.
    5. The controller is conditioned to generate more topologies within those
       families (cluster-guided resampling).

Classes
-------
TopologySpec
    Declarative specification of desired team topology.
TopologyController
    Proposes topologies via LLM prompting or constrained random generation.
TopologySearchLoop
    Iterative search loop with cluster-guided resampling.
TopologyEvaluator
    Wraps TournamentRunner for topology evaluation.

Dependencies: standard library + numpy.  LLM calls go through the
llm_agent module (optional import).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from llm_team_gym.core.topology import TopologyGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional imports -- graceful degradation when LLM backend is absent
# ---------------------------------------------------------------------------

try:
    from llm_team_gym.core.llm_agent import LLMAgent
except ImportError:
    LLMAgent = None  # type: ignore[assignment,misc]


# ===================================================================
# TopologySpec
# ===================================================================


@dataclass
class TopologySpec:
    """Declarative specification of a desired team topology.

    Encodes the number of agents by role, connectivity constraints, and
    any domain-specific requirements that should be conveyed to an LLM
    controller via :meth:`to_prompt`.

    Parameters
    ----------
    n_agents : int
        Total number of agents (convenience; must equal
        ``n_workers + n_validators + n_thinkers``).
    n_workers : int
        Number of Worker agents.
    n_validators : int
        Number of Validator agents.
    n_thinkers : int
        Number of Thinker agents (default 1).
    max_degree : int or None
        Maximum number of edges per Worker node.  ``None`` means
        unconstrained (complete bipartite is allowed).
    sparsity_target : float
        Target edge density in (0, 1].  1.0 = complete bipartite;
        smaller values request sparser graphs.
    constraints : dict
        Arbitrary key-value pairs forwarded to the LLM prompt
        (e.g. ``{"symmetry": True, "hierarchy": "flat"}``).
    task_description : str
        Free-text description of the target task, included in the
        LLM prompt so the controller can tailor topology to the domain.
    """

    n_agents: int
    n_workers: int
    n_validators: int
    n_thinkers: int = 1
    max_degree: Optional[int] = None
    sparsity_target: float = 0.5
    constraints: Dict[str, Any] = field(default_factory=dict)
    task_description: str = ""

    def __post_init__(self) -> None:
        expected = self.n_workers + self.n_validators + self.n_thinkers
        if self.n_agents != expected:
            raise ValueError(
                f"n_agents ({self.n_agents}) != "
                f"n_workers + n_validators + n_thinkers ({expected})"
            )
        if not (0.0 < self.sparsity_target <= 1.0):
            raise ValueError(
                f"sparsity_target must be in (0, 1], got {self.sparsity_target}"
            )

    # ---- prompt generation -----------------------------------------------

    def to_prompt(self) -> str:
        """Render a natural-language specification for an LLM controller.

        The prompt instructs the model to produce a bipartite adjacency
        description (worker-to-validator edge list) that can be parsed
        by :meth:`TopologyController.parse_topology_response`.
        """
        lines: List[str] = [
            "Design a communication topology for a multi-agent team with "
            "the following specification:",
            "",
            f"  Workers (W):    {self.n_workers}",
            f"  Validators (V): {self.n_validators}",
            f"  Thinkers (T):   {self.n_thinkers}",
            f"  Total agents:   {self.n_agents}",
            "",
            f"  Target edge density (sparsity_target): {self.sparsity_target:.2f}",
        ]

        if self.max_degree is not None:
            lines.append(
                f"  Maximum edges per Worker: {self.max_degree}"
            )

        if self.task_description:
            lines.extend([
                "",
                f"  Task: {self.task_description}",
            ])

        if self.constraints:
            lines.append("")
            lines.append("  Additional constraints:")
            for key, value in self.constraints.items():
                lines.append(f"    - {key}: {value}")

        lines.extend([
            "",
            "Output format:",
            "  Provide the topology as an ADJACENCY LIST where each line is:",
            "    W<i> -> V<j1>, V<j2>, ...",
            "  Worker indices are 0-based (W0 through W{w}). "
            "Validator indices are 0-based (V0 through V{v}).".format(
                w=self.n_workers - 1, v=self.n_validators - 1,
            ),
            "  Thinkers connect to all Validators automatically; "
            "do NOT list Thinker edges.",
            "",
            "  Example for 3 Workers, 2 Validators:",
            "    W0 -> V0, V1",
            "    W1 -> V0",
            "    W2 -> V1",
            "",
            "Produce ONLY the adjacency list, one line per Worker.",
        ])
        return "\n".join(lines)


# ===================================================================
# Structural / spectral fingerprinting
# ===================================================================


def _compute_fingerprint(adj: np.ndarray) -> Dict[str, Any]:
    """Compute a structural and spectral fingerprint for a topology.

    The fingerprint captures properties useful for clustering topologies
    into families without relying on raw adjacency comparison:

    * **n_edges**: total worker-validator edges.
    * **density**: fraction of possible edges present.
    * **degree_mean / degree_std**: worker degree statistics.
    * **eigenvalues_top3**: top-3 singular values of the bipartite matrix
      (spectral signature invariant to permutation).
    * **sha256**: content hash of the flattened adjacency for dedup.

    Parameters
    ----------
    adj : np.ndarray
        Bipartite adjacency matrix of shape ``(W, V)``.

    Returns
    -------
    dict
        Fingerprint dictionary (all values JSON-serialisable).
    """
    W, V = adj.shape
    n_edges = int(np.sum(adj))
    max_edges = W * V if (W > 0 and V > 0) else 1
    density = n_edges / max_edges

    row_degrees = adj.sum(axis=1).astype(float)
    degree_mean = float(np.mean(row_degrees)) if W > 0 else 0.0
    degree_std = float(np.std(row_degrees)) if W > 0 else 0.0

    # Spectral: SVD of the bipartite matrix
    if W > 0 and V > 0 and n_edges > 0:
        try:
            sv = np.linalg.svd(adj.astype(float), compute_uv=False)
            top3 = sv[:3].tolist()
            # Pad to exactly 3 entries for consistency
            while len(top3) < 3:
                top3.append(0.0)
        except np.linalg.LinAlgError:
            top3 = [0.0, 0.0, 0.0]
    else:
        top3 = [0.0, 0.0, 0.0]

    content_hash = hashlib.sha256(adj.tobytes()).hexdigest()[:16]

    return {
        "n_edges": n_edges,
        "density": round(density, 6),
        "degree_mean": round(degree_mean, 4),
        "degree_std": round(degree_std, 4),
        "eigenvalues_top3": [round(v, 6) for v in top3],
        "sha256": content_hash,
    }


def _fingerprint_to_vector(fp: Dict[str, Any]) -> np.ndarray:
    """Flatten a fingerprint dict into a fixed-length numeric vector.

    Vector layout: [n_edges, density, degree_mean, degree_std, sv1, sv2, sv3].
    """
    return np.array([
        fp["n_edges"],
        fp["density"],
        fp["degree_mean"],
        fp["degree_std"],
        fp["eigenvalues_top3"][0],
        fp["eigenvalues_top3"][1],
        fp["eigenvalues_top3"][2],
    ], dtype=float)


# ===================================================================
# TopologyController
# ===================================================================


class TopologyController:
    """Proposes candidate topologies via LLM prompting or random generation.

    If an ``llm_agent`` is provided, the controller prompts it with the
    specification from :class:`TopologySpec` and parses the response into
    a bipartite adjacency matrix.  Without an LLM the controller falls
    back to constrained random generation.

    Parameters
    ----------
    llm_agent : LLMAgent or None
        An LLM agent instance (from ``llm_team_gym.core.llm_agent``)
        used to generate topology proposals.  ``None`` activates the
        random fallback.
    seed : int or None
        Random seed for the fallback generator.
    """

    def __init__(
        self,
        llm_agent: Any = None,
        seed: Optional[int] = None,
    ) -> None:
        self.llm_agent = llm_agent
        self._rng = np.random.RandomState(seed)

        # Cluster-guided conditioning context (set by the search loop).
        self._cluster_context: Optional[str] = None

    # ---- public API -------------------------------------------------------

    def propose(self, spec: TopologySpec) -> Dict[str, Any]:
        """Propose a candidate topology for *spec*.

        Returns a dict with keys:

        * ``"adjacency"``: ``np.ndarray`` bipartite matrix ``(W, V)``.
        * ``"topology"``: ``TopologyGraph`` instance.
        * ``"source"``: ``"llm"`` or ``"random"``.
        * ``"raw_response"``: the raw LLM text (empty for random).

        Parameters
        ----------
        spec : TopologySpec
            The topology specification to satisfy.

        Returns
        -------
        dict
        """
        if self.llm_agent is not None:
            return self._propose_via_llm(spec)
        return self._propose_random(spec)

    def set_cluster_context(self, context: str) -> None:
        """Condition subsequent LLM proposals on cluster-level guidance.

        The *context* string is prepended to the specification prompt so
        the controller biases generation toward a known-good family.
        """
        self._cluster_context = context

    def clear_cluster_context(self) -> None:
        """Remove cluster conditioning."""
        self._cluster_context = None

    # ---- LLM-based proposal -----------------------------------------------

    def _propose_via_llm(self, spec: TopologySpec) -> Dict[str, Any]:
        """Prompt the LLM and parse its response into a topology."""
        prompt = spec.to_prompt()
        if self._cluster_context:
            prompt = (
                "GUIDANCE FROM PREVIOUS SEARCH ITERATIONS:\n"
                f"{self._cluster_context}\n\n"
                "Taking the above guidance into account, "
                "propose a topology that follows the patterns of the "
                "best-performing cluster.\n\n"
                + prompt
            )

        # Build messages for the LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multi-agent topology designer. "
                    "Output ONLY the adjacency list in the requested format. "
                    "Do not include commentary or markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw_response = self.llm_agent._call_llm(messages)
        except Exception as exc:
            logger.warning(
                "LLM call failed in TopologyController: %s. "
                "Falling back to random generation.",
                exc,
            )
            result = self._propose_random(spec)
            result["source"] = "random_fallback"
            return result

        adj = self.parse_topology_response(raw_response, spec)
        topo = TopologyGraph.from_matrix(adj, n_thinkers=spec.n_thinkers)

        return {
            "adjacency": adj,
            "topology": topo,
            "source": "llm",
            "raw_response": raw_response,
        }

    # ---- random fallback ---------------------------------------------------

    def _propose_random(self, spec: TopologySpec) -> Dict[str, Any]:
        """Generate a topology randomly, respecting constraints."""
        W, V = spec.n_workers, spec.n_validators
        if W == 0 or V == 0:
            adj = np.zeros((W, V), dtype=int)
        else:
            # Sample edges with target sparsity
            adj = (self._rng.random((W, V)) < spec.sparsity_target).astype(int)

            # Enforce max_degree
            if spec.max_degree is not None:
                for w in range(W):
                    active = np.where(adj[w] == 1)[0]
                    if len(active) > spec.max_degree:
                        keep = self._rng.choice(
                            active, size=spec.max_degree, replace=False
                        )
                        adj[w] = 0
                        adj[w, keep] = 1

            # Guarantee relaxed validity: every Worker has >= 1 Validator
            for w in range(W):
                if not np.any(adj[w]):
                    adj[w, self._rng.randint(V)] = 1

        topo = TopologyGraph.from_matrix(adj, n_thinkers=spec.n_thinkers)
        return {
            "adjacency": adj,
            "topology": topo,
            "source": "random",
            "raw_response": "",
        }

    # ---- response parsing --------------------------------------------------

    @staticmethod
    def parse_topology_response(
        text: str, spec: TopologySpec
    ) -> np.ndarray:
        """Extract a bipartite adjacency matrix from LLM-generated text.

        Expected format (one line per Worker)::

            W0 -> V0, V1
            W1 -> V0
            W2 -> V1

        Also accepts variants with ``-->``, ``:``, or tab separation.
        Lines that cannot be parsed are silently skipped.  Workers not
        mentioned get a single random Validator to ensure validity.

        Parameters
        ----------
        text : str
            Raw LLM output.
        spec : TopologySpec
            The specification (provides W, V dimensions).

        Returns
        -------
        np.ndarray
            Binary matrix of shape ``(W, V)``.
        """
        W, V = spec.n_workers, spec.n_validators
        adj = np.zeros((W, V), dtype=int)
        rng = np.random.RandomState(42)  # deterministic fallback

        # Regex: capture worker index and the validator list
        #   W3 -> V0, V1, V2   |   W3 --> V0 V1 V2   |   W3: V0, V1
        line_pattern = re.compile(
            r"W(\d+)\s*(?:->|-->|:)\s*(.+)", re.IGNORECASE
        )
        validator_pattern = re.compile(r"V(\d+)", re.IGNORECASE)

        parsed_workers: set = set()

        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = line_pattern.match(line)
            if m is None:
                continue
            w_idx = int(m.group(1))
            if w_idx >= W:
                continue
            parsed_workers.add(w_idx)

            for vm in validator_pattern.finditer(m.group(2)):
                v_idx = int(vm.group(1))
                if v_idx < V:
                    adj[w_idx, v_idx] = 1

        # Enforce max_degree from spec
        if spec.max_degree is not None:
            for w in range(W):
                active = np.where(adj[w] == 1)[0]
                if len(active) > spec.max_degree:
                    keep = rng.choice(
                        active, size=spec.max_degree, replace=False
                    )
                    adj[w] = 0
                    adj[w, keep] = 1

        # Guarantee relaxed validity
        if V > 0:
            for w in range(W):
                if not np.any(adj[w]):
                    adj[w, rng.randint(V)] = 1

        return adj


# ===================================================================
# TopologyEvaluator
# ===================================================================


class TopologyEvaluator:
    """Evaluate a topology by running tournament matches.

    Wraps :class:`~llm_team_gym.envs.tournament.TournamentRunner` to
    score a given topology on success rate, average score, cost, and
    consistency.

    Parameters
    ----------
    game_factory : callable
        Zero-argument callable returning a fresh ``BaseGame`` instance.
    agent_factory : callable
        Callable ``(topology: TopologyGraph) -> List[BaseAgent]`` that
        builds agents wired according to the given topology.
    n_eval_matches : int
        Number of matches per evaluation (default 10).
    runner_kwargs : dict or None
        Extra keyword arguments forwarded to ``MatchRunner``.
    """

    def __init__(
        self,
        game_factory: Callable,
        agent_factory: Callable,
        n_eval_matches: int = 10,
        runner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.game_factory = game_factory
        self.agent_factory = agent_factory
        self.n_eval_matches = n_eval_matches
        self.runner_kwargs = runner_kwargs or {}

    def evaluate(self, topology: TopologyGraph) -> Dict[str, Any]:
        """Run *n_eval_matches* matches and aggregate results.

        Parameters
        ----------
        topology : TopologyGraph
            The topology to evaluate.

        Returns
        -------
        dict
            Keys: ``success_rate``, ``avg_score``, ``avg_cost``,
            ``score_std``, ``consistency``, ``n_matches``, ``win_counts``,
            ``scores``.
        """
        from llm_team_gym.envs.tournament import TournamentRunner

        def _agent_factory_wrapper():
            return self.agent_factory(topology)

        runner = TournamentRunner(
            game_factory=self.game_factory,
            agent_factory=_agent_factory_wrapper,
            n_matches=self.n_eval_matches,
            runner_kwargs=self.runner_kwargs,
        )

        stats = runner.run()

        # Aggregate scores across all matches
        all_scores: List[float] = []
        for record in stats["records"]:
            total = sum(record.final_team_scores.values())
            all_scores.append(total)

        avg_score = float(np.mean(all_scores)) if all_scores else 0.0
        score_std = float(np.std(all_scores)) if all_scores else 0.0

        # Success rate: fraction of matches with a declared winner
        n_with_winner = sum(
            1 for r in stats["records"] if r.winner is not None
        )
        success_rate = n_with_winner / max(len(stats["records"]), 1)

        # Consistency: inverse coefficient of variation (higher = more consistent)
        if avg_score != 0 and score_std > 0:
            consistency = 1.0 - min(score_std / abs(avg_score), 1.0)
        else:
            consistency = 1.0 if score_std == 0 else 0.0

        # Estimate cost from agent usage stats (sum of all agent costs)
        total_cost = 0.0
        for record in stats["records"]:
            # MatchRecord does not directly expose cost; use duration as proxy
            total_cost += record.duration()
        avg_cost = total_cost / max(len(stats["records"]), 1)

        return {
            "success_rate": round(success_rate, 4),
            "avg_score": round(avg_score, 4),
            "avg_cost": round(avg_cost, 6),
            "score_std": round(score_std, 4),
            "consistency": round(consistency, 4),
            "n_matches": len(stats["records"]),
            "win_counts": stats.get("win_counts", {}),
            "scores": all_scores,
        }


# ===================================================================
# Simple k-means clustering (no sklearn dependency)
# ===================================================================


def _kmeans(
    vectors: np.ndarray,
    k: int,
    max_iter: int = 50,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal k-means implementation on row vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Data matrix of shape ``(N, D)``.
    k : int
        Number of clusters (clamped to ``[1, N]``).
    max_iter : int
        Maximum iterations.
    seed : int
        Random seed for centroid initialisation.

    Returns
    -------
    labels : np.ndarray of shape (N,)
        Cluster assignment for each row.
    centroids : np.ndarray of shape (k, D)
        Final centroid positions.
    """
    rng = np.random.RandomState(seed)
    N, D = vectors.shape
    k = max(1, min(k, N))

    # Initialise centroids via random selection (k-means++)
    indices = rng.choice(N, size=k, replace=False)
    centroids = vectors[indices].copy()

    labels = np.zeros(N, dtype=int)

    for _ in range(max_iter):
        # Assignment step
        # Compute squared distances: (N, k)
        dists = np.sum(
            (vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update step
        for c in range(k):
            members = vectors[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    return labels, centroids


# ===================================================================
# TopologySearchLoop
# ===================================================================


@dataclass
class _ScoredTopology:
    """Internal record of one evaluated topology."""

    topology: TopologyGraph
    adjacency: np.ndarray
    score: float
    eval_result: Dict[str, Any]
    fingerprint: Dict[str, Any]
    source: str
    iteration: int
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "topology": self.topology.to_dict(),
            "score": self.score,
            "eval_result": self.eval_result,
            "fingerprint": self.fingerprint,
            "source": self.source,
            "iteration": self.iteration,
        }


class TopologySearchLoop:
    """Iterative topology search with cluster-guided resampling.

    Each iteration:

    1. The controller proposes ``n_candidates_per_iter`` topologies.
    2. Each candidate is evaluated by ``evaluator_fn`` (or a
       :class:`TopologyEvaluator`).
    3. Results are logged with structural/spectral fingerprints.
    4. After each iteration, topologies are clustered; the controller
       is conditioned to resample from the best-performing cluster.

    Parameters
    ----------
    spec : TopologySpec
        Topology specification.
    controller : TopologyController
        Generates candidate topologies.
    evaluator_fn : callable
        ``(TopologyGraph) -> dict`` scoring function.  Must return a dict
        with at least ``"avg_score"`` (float).  Can be a bound method of
        :class:`TopologyEvaluator`.
    n_iterations : int
        Number of search iterations (default 10).
    n_candidates_per_iter : int
        Candidates generated per iteration (default 5).
    n_clusters : int
        Number of clusters for the resampling step (default 3).
    score_key : str
        Key in the evaluator result dict used as the optimisation
        objective (default ``"avg_score"``).
    """

    def __init__(
        self,
        spec: TopologySpec,
        controller: TopologyController,
        evaluator_fn: Callable[[TopologyGraph], Dict[str, Any]],
        n_iterations: int = 10,
        n_candidates_per_iter: int = 5,
        n_clusters: int = 3,
        score_key: str = "avg_score",
    ) -> None:
        self.spec = spec
        self.controller = controller
        self.evaluator_fn = evaluator_fn
        self.n_iterations = n_iterations
        self.n_candidates_per_iter = n_candidates_per_iter
        self.n_clusters = n_clusters
        self.score_key = score_key

        # Full search log: list of scored topology records
        self.log: List[_ScoredTopology] = []
        self._iteration_count: int = 0

    # ---- single iteration -------------------------------------------------

    def run_iteration(self) -> List[_ScoredTopology]:
        """Generate, evaluate, and log one batch of candidates.

        Returns
        -------
        list of _ScoredTopology
            The scored candidates from this iteration.
        """
        self._iteration_count += 1
        iteration_results: List[_ScoredTopology] = []

        for _ in range(self.n_candidates_per_iter):
            proposal = self.controller.propose(self.spec)
            topology: TopologyGraph = proposal["topology"]
            adj: np.ndarray = proposal["adjacency"]

            # Evaluate
            try:
                eval_result = self.evaluator_fn(topology)
            except Exception as exc:
                logger.warning(
                    "Evaluation failed for a candidate in iteration %d: %s",
                    self._iteration_count, exc,
                )
                eval_result = {self.score_key: 0.0, "error": str(exc)}

            score = float(eval_result.get(self.score_key, 0.0))
            fingerprint = _compute_fingerprint(adj)

            record = _ScoredTopology(
                topology=topology,
                adjacency=adj,
                score=score,
                eval_result=eval_result,
                fingerprint=fingerprint,
                source=proposal.get("source", "unknown"),
                iteration=self._iteration_count,
                raw_response=proposal.get("raw_response", ""),
            )
            iteration_results.append(record)
            self.log.append(record)

        logger.info(
            "Iteration %d: evaluated %d candidates. "
            "Best score this round: %.4f",
            self._iteration_count,
            len(iteration_results),
            max(r.score for r in iteration_results) if iteration_results else 0.0,
        )

        return iteration_results

    # ---- clustering and resampling ----------------------------------------

    def cluster_and_resample(self) -> Optional[int]:
        """Cluster logged topologies and condition the controller.

        Identifies the best-performing cluster and sets cluster context
        on the controller so the next iteration's proposals are biased
        toward that family.

        Returns
        -------
        int or None
            Index of the best cluster, or ``None`` if clustering is not
            possible (too few data points).
        """
        if len(self.log) < self.n_clusters:
            logger.debug(
                "Not enough data points (%d) to form %d clusters; "
                "skipping cluster-guided resampling.",
                len(self.log), self.n_clusters,
            )
            return None

        # Build feature matrix from fingerprints
        vectors = np.array([
            _fingerprint_to_vector(r.fingerprint) for r in self.log
        ])

        # Normalise features to [0, 1] for balanced clustering
        col_min = vectors.min(axis=0)
        col_max = vectors.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0  # avoid division by zero
        vectors_norm = (vectors - col_min) / col_range

        k = min(self.n_clusters, len(self.log))
        labels, centroids = _kmeans(vectors_norm, k)

        # Find best cluster by average score
        cluster_scores: Dict[int, List[float]] = defaultdict(list)
        for i, record in enumerate(self.log):
            cluster_scores[int(labels[i])].append(record.score)

        best_cluster = max(
            cluster_scores,
            key=lambda c: float(np.mean(cluster_scores[c])),
        )
        best_avg = float(np.mean(cluster_scores[best_cluster]))

        # Build a description of the best cluster for conditioning
        best_members = [
            r for i, r in enumerate(self.log)
            if int(labels[i]) == best_cluster
        ]
        avg_fp = {
            "density": float(np.mean([m.fingerprint["density"] for m in best_members])),
            "degree_mean": float(np.mean([m.fingerprint["degree_mean"] for m in best_members])),
            "degree_std": float(np.mean([m.fingerprint["degree_std"] for m in best_members])),
            "n_edges": float(np.mean([m.fingerprint["n_edges"] for m in best_members])),
        }

        context = (
            f"The best-performing topology cluster (avg score {best_avg:.4f}) "
            f"has the following average structural properties:\n"
            f"  - Edge density: {avg_fp['density']:.4f}\n"
            f"  - Mean worker degree: {avg_fp['degree_mean']:.2f}\n"
            f"  - Degree std: {avg_fp['degree_std']:.2f}\n"
            f"  - Average edge count: {avg_fp['n_edges']:.1f}\n"
            f"Generate topologies with similar structural characteristics."
        )
        self.controller.set_cluster_context(context)

        logger.info(
            "Cluster analysis: best cluster %d (avg_score=%.4f, "
            "members=%d). Controller conditioned for next iteration.",
            best_cluster, best_avg, len(best_members),
        )

        return best_cluster

    # ---- full search loop -------------------------------------------------

    def run(self) -> Tuple[TopologyGraph, List[Dict[str, Any]]]:
        """Execute the full search loop.

        Returns
        -------
        best_topology : TopologyGraph
            The topology with the highest score across all iterations.
        full_log : list of dict
            Complete log of every evaluated topology, serialised.
        """
        logger.info(
            "Starting topology search: %d iterations x %d candidates = "
            "%d total evaluations.",
            self.n_iterations,
            self.n_candidates_per_iter,
            self.n_iterations * self.n_candidates_per_iter,
        )

        for iteration in range(self.n_iterations):
            self.run_iteration()

            # Cluster and condition after every iteration
            self.cluster_and_resample()

        # Identify best topology overall
        if not self.log:
            raise RuntimeError("Search loop produced no evaluated topologies.")

        best_record = max(self.log, key=lambda r: r.score)
        full_log = [r.to_dict() for r in self.log]

        logger.info(
            "Search complete. Best topology score: %.4f "
            "(source=%s, iteration=%d, edges=%d).",
            best_record.score,
            best_record.source,
            best_record.iteration,
            best_record.fingerprint["n_edges"],
        )

        return best_record.topology, full_log

    # ---- export ------------------------------------------------------------

    def export_log(self, path: str) -> None:
        """Save all iterations as a JSON file.

        Parameters
        ----------
        path : str
            Output file path (will be overwritten if it exists).
        """
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "spec": {
                "n_agents": self.spec.n_agents,
                "n_workers": self.spec.n_workers,
                "n_validators": self.spec.n_validators,
                "n_thinkers": self.spec.n_thinkers,
                "max_degree": self.spec.max_degree,
                "sparsity_target": self.spec.sparsity_target,
                "constraints": self.spec.constraints,
                "task_description": self.spec.task_description,
            },
            "search_params": {
                "n_iterations": self.n_iterations,
                "n_candidates_per_iter": self.n_candidates_per_iter,
                "n_clusters": self.n_clusters,
                "score_key": self.score_key,
            },
            "total_evaluated": len(self.log),
            "best_score": max(r.score for r in self.log) if self.log else None,
            "entries": [r.to_dict() for r in self.log],
        }

        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        logger.info("Search log exported to %s (%d entries).", path, len(self.log))
