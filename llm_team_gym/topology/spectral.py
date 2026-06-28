"""
Spectral characterization and clustering of agent communication topologies.

Implements the spectral-graph pipeline described in the thesis:
  1. Symmetrize directed adjacency matrices (treat each directed edge as
     undirected).
  2. Build the combinatorial Laplacian  L = D - A.
  3. Extract a feature vector per topology: algebraic connectivity (lambda_2),
     eigenvalue ratio (lambda_2 / lambda_max), degree statistics, diameter,
     and clustering coefficient.
  4. Cluster topologies in descriptor space using spectral clustering, rank
     them by task performance, and compute the Pareto front over multiple
     objectives.
"""

from __future__ import annotations

from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import linalg as la


# ---------------------------------------------------------------------------
# SpectralAnalyzer
# ---------------------------------------------------------------------------

class SpectralAnalyzer:
    """Spectral characterisation of a single agent-communication topology.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Square (n x n) adjacency matrix.  May be directed (asymmetric) or
        weighted; the analysis always operates on its symmetrised,
        unweighted form.
    """

    def __init__(self, adjacency_matrix: np.ndarray) -> None:
        A = np.asarray(adjacency_matrix, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("adjacency_matrix must be a square 2-D array")
        self._adj = A
        self._n = A.shape[0]
        # Lazily cached derived quantities
        self._sym: Optional[np.ndarray] = None
        self._lap: Optional[np.ndarray] = None
        self._eigvals: Optional[np.ndarray] = None

    # -- core transforms ---------------------------------------------------

    def symmetrize(self) -> np.ndarray:
        """Return the symmetrised (undirected, binary) adjacency matrix.

        An undirected edge (i, j) exists whenever *either* (i->j) or (j->i)
        is present in the original directed graph.  Self-loops are removed.
        """
        if self._sym is None:
            S = np.where((self._adj + self._adj.T) > 0, 1.0, 0.0)
            np.fill_diagonal(S, 0.0)
            self._sym = S
        return self._sym.copy()

    def laplacian(self) -> np.ndarray:
        """Compute the combinatorial Laplacian  L = D - A  of the
        symmetrised graph."""
        if self._lap is None:
            A = self.symmetrize()
            D = np.diag(A.sum(axis=1))
            self._lap = D - A
        return self._lap.copy()

    def eigenvalues(self) -> np.ndarray:
        """Sorted (ascending) eigenvalues of the Laplacian."""
        if self._eigvals is None:
            L = self.laplacian()
            vals = la.eigvalsh(L)  # real symmetric -> real eigenvalues
            vals = np.sort(np.real(vals))
            # Numerical noise: clamp tiny negatives to zero
            vals[vals < 0] = 0.0
            self._eigvals = vals
        return self._eigvals.copy()

    # -- scalar descriptors ------------------------------------------------

    def algebraic_connectivity(self) -> float:
        """Return lambda_2, the second-smallest Laplacian eigenvalue.

        For a connected graph this is strictly positive and characterises how
        well-connected the graph is (Fiedler value).
        """
        vals = self.eigenvalues()
        if len(vals) < 2:
            return 0.0
        return float(vals[1])

    def eigenvalue_ratio(self) -> float:
        """Return lambda_2 / lambda_max.

        A ratio close to 1 indicates a nearly uniform spectrum (e.g. the
        complete graph); close to 0 indicates a bottleneck topology.
        """
        vals = self.eigenvalues()
        if len(vals) < 2:
            return 0.0
        lam_max = vals[-1]
        if lam_max == 0.0:
            return 0.0
        return float(vals[1] / lam_max)

    def degree_distribution(self) -> np.ndarray:
        """Degree of each node in the symmetrised graph."""
        A = self.symmetrize()
        return A.sum(axis=1)

    def diameter(self) -> int:
        """Diameter of the symmetrised graph (longest shortest path).

        Uses BFS from every node.  Returns -1 for disconnected graphs
        (following the convention that diameter is undefined).
        """
        A = self.symmetrize()
        n = self._n
        if n == 0:
            return 0

        # Build adjacency list from the symmetrised matrix
        adj_list: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if A[i, j] > 0:
                    adj_list[i].append(j)

        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q: deque[int] = deque([src])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # Check connectivity
            if any(d == -1 for d in dist):
                return -1  # disconnected
            max_dist = max(max_dist, max(dist))
        return max_dist

    def clustering_coefficient(self) -> float:
        """Average local clustering coefficient of the symmetrised graph.

        For a node with degree < 2 the local coefficient is defined as 0.
        """
        A = self.symmetrize()
        n = self._n
        if n == 0:
            return 0.0

        coeffs: List[float] = []
        for i in range(n):
            neighbours = np.where(A[i] > 0)[0]
            k = len(neighbours)
            if k < 2:
                coeffs.append(0.0)
                continue
            # Count edges among neighbours
            sub = A[np.ix_(neighbours, neighbours)]
            triangles = sub.sum() / 2.0  # each edge counted twice
            possible = k * (k - 1) / 2.0
            coeffs.append(triangles / possible)
        return float(np.mean(coeffs))

    # -- composite descriptor ----------------------------------------------

    def feature_vector(self) -> np.ndarray:
        """Six-element descriptor for this topology:

        [lambda_2, lambda_2/lambda_max, avg_degree, max_degree,
         diameter, clustering_coeff]
        """
        deg = self.degree_distribution()
        diam = self.diameter()
        # If the graph is disconnected, use n as a sentinel for diameter
        diam_val = float(diam) if diam >= 0 else float(self._n)
        return np.array([
            self.algebraic_connectivity(),
            self.eigenvalue_ratio(),
            float(np.mean(deg)),
            float(np.max(deg)) if len(deg) > 0 else 0.0,
            diam_val,
            self.clustering_coefficient(),
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# TopologyClusterer
# ---------------------------------------------------------------------------

class TopologyClusterer:
    """Spectral clustering of topologies in descriptor space.

    Implements the pipeline manually:
      1. Build a Gaussian (RBF) similarity matrix from feature vectors.
      2. Compute the normalised Laplacian of the similarity graph.
      3. Embed into the first *k* eigenvectors.
      4. Run k-means in the embedded space.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (k).
    sigma : float or ``"auto"``
        Bandwidth of the RBF kernel.  ``"auto"`` uses the median pairwise
        distance heuristic.
    random_state : int or None
        Seed for reproducibility of k-means.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        sigma: Union[float, str] = "auto",
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.random_state = random_state
        self._labels: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._fitted_features: Optional[np.ndarray] = None

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
        """Squared Euclidean distance matrix (n x n)."""
        sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
        return sq_norms + sq_norms.T - 2.0 * X @ X.T

    def _rbf_similarity(self, X: np.ndarray) -> np.ndarray:
        """Gaussian similarity matrix with bandwidth ``self.sigma``."""
        D2 = self._pairwise_sq_distances(X)
        D2 = np.maximum(D2, 0.0)

        if self.sigma == "auto":
            # Median heuristic (off-diagonal distances)
            n = D2.shape[0]
            mask = ~np.eye(n, dtype=bool)
            dists = np.sqrt(D2[mask])
            sigma = float(np.median(dists)) if len(dists) > 0 else 1.0
            sigma = max(sigma, 1e-8)
        else:
            sigma = float(self.sigma)

        W = np.exp(-D2 / (2.0 * sigma ** 2))
        np.fill_diagonal(W, 0.0)
        return W

    @staticmethod
    def _kmeans(
        X: np.ndarray,
        k: int,
        max_iter: int = 300,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Minimal k-means implementation.

        Returns (labels, centroids).
        """
        if rng is None:
            rng = np.random.default_rng()
        n, d = X.shape
        # k-means++ initialisation
        centroids = np.empty((k, d), dtype=np.float64)
        idx = rng.integers(0, n)
        centroids[0] = X[idx]
        for c in range(1, k):
            dists = np.min(
                np.sum((X[:, None, :] - centroids[None, :c, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centroids[c] = X[idx]

        labels = np.zeros(n, dtype=np.intp)
        for _ in range(max_iter):
            # Assign
            d2 = np.sum(
                (X[:, None, :] - centroids[None, :, :]) ** 2, axis=2
            )
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    centroids[c] = members.mean(axis=0)
        return labels, centroids

    # -- public API --------------------------------------------------------

    def fit(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Cluster *feature_vectors* (n_topologies x n_features).

        Returns
        -------
        labels : np.ndarray of shape (n_topologies,)
        """
        X = np.asarray(feature_vectors, dtype=np.float64)
        n = X.shape[0]
        k = min(self.n_clusters, n)

        # Standardise features
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        Xs = (X - mu) / std

        rng = np.random.default_rng(self.random_state)

        if n <= k:
            labels = np.arange(n, dtype=np.intp)
        else:
            # Build similarity graph
            W = self._rbf_similarity(Xs)
            deg = W.sum(axis=1)
            deg[deg == 0] = 1.0
            D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
            # Normalised Laplacian:  L_sym = I - D^{-1/2} W D^{-1/2}
            L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

            # Eigen-decomposition (symmetric -> eigvalsh)
            vals, vecs = la.eigh(L_sym)
            # Take the first k eigenvectors (smallest eigenvalues)
            embedding = vecs[:, :k]
            # Row-normalise
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding = embedding / norms

            labels, centroids = self._kmeans(embedding, k, rng=rng)
            self._centroids = centroids

        self._labels = labels
        self._fitted_features = X
        return labels

    def silhouette_score(
        self,
        feature_vectors: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute the mean silhouette coefficient.

        Parameters
        ----------
        feature_vectors : (n, d) array
        labels : (n,) int array of cluster assignments

        Returns
        -------
        score : float in [-1, 1]
        """
        X = np.asarray(feature_vectors, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.intp)
        n = X.shape[0]
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1 or n <= 1:
            return 0.0

        # Pairwise Euclidean distances
        D = np.sqrt(np.maximum(self._pairwise_sq_distances(X), 0.0))

        sil = np.zeros(n, dtype=np.float64)
        for i in range(n):
            own = labels[i]
            own_mask = labels == own
            own_count = own_mask.sum()
            # a(i): mean intra-cluster distance
            if own_count > 1:
                a_i = D[i, own_mask].sum() / (own_count - 1)
            else:
                a_i = 0.0
            # b(i): smallest mean inter-cluster distance
            b_i = np.inf
            for lab in unique_labels:
                if lab == own:
                    continue
                other_mask = labels == lab
                other_count = other_mask.sum()
                if other_count == 0:
                    continue
                mean_d = D[i, other_mask].sum() / other_count
                b_i = min(b_i, mean_d)
            if b_i == np.inf:
                b_i = 0.0
            denom = max(a_i, b_i)
            sil[i] = (b_i - a_i) / denom if denom > 0 else 0.0
        return float(np.mean(sil))

    def predict(self, feature_vector: np.ndarray) -> int:
        """Assign a new topology to the nearest cluster centroid.

        Requires that :meth:`fit` has been called first.
        """
        if self._fitted_features is None or self._labels is None:
            raise RuntimeError("TopologyClusterer has not been fitted yet")
        fv = np.asarray(feature_vector, dtype=np.float64).ravel()

        if self._centroids is not None:
            # Use centroids in the embedded space -- approximate: use
            # label-mean in the original feature space instead for
            # interpretability.
            pass

        # Fallback (and primary method): nearest-centroid in feature space
        unique_labels = np.unique(self._labels)
        best_label = int(unique_labels[0])
        best_dist = np.inf
        for lab in unique_labels:
            mask = self._labels == lab
            centroid = self._fitted_features[mask].mean(axis=0)
            d = float(np.linalg.norm(fv - centroid))
            if d < best_dist:
                best_dist = d
                best_label = int(lab)
        return best_label


# ---------------------------------------------------------------------------
# TopologyRanker
# ---------------------------------------------------------------------------

class TopologyRanker:
    """Rank and select topologies by task performance, including
    multi-objective Pareto optimality."""

    @staticmethod
    def rank_by_performance(
        topologies: Sequence[Any],
        scores: Sequence[float],
    ) -> List[Tuple[Any, float]]:
        """Return *(topology, score)* pairs sorted descending by *scores*.

        Parameters
        ----------
        topologies : sequence
            Topology identifiers or objects (arbitrary).
        scores : sequence of float
            Scalar performance score for each topology (higher is better).

        Returns
        -------
        sorted_list : list of (topology, score)
        """
        paired = list(zip(topologies, scores))
        paired.sort(key=lambda t: t[1], reverse=True)
        return paired

    @staticmethod
    def pareto_front(
        topologies: Sequence[Any],
        metrics_matrix: np.ndarray,
    ) -> List[int]:
        """Identify the Pareto-optimal (non-dominated) topologies.

        All objectives are assumed to be *maximised*.

        Parameters
        ----------
        topologies : sequence of length n
            Topology identifiers (not used in the computation, but kept
            for API symmetry so the caller can index back).
        metrics_matrix : np.ndarray, shape (n, m)
            Each row is the m-dimensional objective vector for the
            corresponding topology.  Columns might be, e.g.,
            [success_rate, -cost, consistency].

        Returns
        -------
        indices : list of int
            Row indices of the Pareto-optimal topologies, in their
            original order.
        """
        M = np.asarray(metrics_matrix, dtype=np.float64)
        n = M.shape[0]
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # j dominates i  iff  all(M[j] >= M[i]) and any(M[j] > M[i])
                if np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
                    is_dominated[i] = True
                    break

        return [int(i) for i in range(n) if not is_dominated[i]]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compare_topologies(
    topo_list: Sequence[np.ndarray],
    metric_fn: Optional[Callable[[SpectralAnalyzer], float]] = None,
) -> Dict[str, List[Any]]:
    """Compare a list of adjacency matrices on spectral descriptors.

    Parameters
    ----------
    topo_list : sequence of np.ndarray
        Adjacency matrices to compare.
    metric_fn : callable, optional
        Extra scalar metric ``f(SpectralAnalyzer) -> float`` appended as
        the ``"custom_metric"`` column.  Ignored when *None*.

    Returns
    -------
    table : dict
        Column-oriented table (compatible with ``pandas.DataFrame(table)``).
        Keys: ``"index"``, ``"lambda_2"``, ``"eigen_ratio"``, ``"avg_degree"``,
        ``"max_degree"``, ``"diameter"``, ``"clustering_coeff"``, and
        optionally ``"custom_metric"``.
    """
    table: Dict[str, List[Any]] = {
        "index": [],
        "lambda_2": [],
        "eigen_ratio": [],
        "avg_degree": [],
        "max_degree": [],
        "diameter": [],
        "clustering_coeff": [],
    }
    if metric_fn is not None:
        table["custom_metric"] = []

    for i, adj in enumerate(topo_list):
        sa = SpectralAnalyzer(adj)
        fv = sa.feature_vector()
        table["index"].append(i)
        table["lambda_2"].append(fv[0])
        table["eigen_ratio"].append(fv[1])
        table["avg_degree"].append(fv[2])
        table["max_degree"].append(fv[3])
        table["diameter"].append(int(fv[4]))
        table["clustering_coeff"].append(fv[5])
        if metric_fn is not None:
            table["custom_metric"].append(metric_fn(sa))

    return table


def plot_spectral_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> Any:
    """Scatter-plot of topologies coloured by cluster label.

    Uses the first two principal components of *features* for the axes.
    If *save_path* is given the figure is saved and the ``matplotlib``
    figure object is returned (useful in headless environments).

    Parameters
    ----------
    features : (n, d) array
        Feature vectors (e.g. from :meth:`SpectralAnalyzer.feature_vector`).
    labels : (n,) int array
        Cluster assignments.
    save_path : str, optional
        File path (e.g. ``"clusters.png"``).  If *None* the plot is shown
        interactively when a display is available.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    try:
        import matplotlib
        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError:
        # matplotlib is optional; gracefully skip plotting
        return None

    X = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)

    # PCA -> 2D projection (manual, avoids sklearn dependency)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = Xc.T @ Xc / max(X.shape[0] - 1, 1)
    eigvals, eigvecs = la.eigh(cov)
    # Take the two largest eigenvectors
    idx = np.argsort(eigvals)[::-1][:2]
    proj = Xc @ eigvecs[:, idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1] if proj.shape[1] > 1 else np.zeros(mask.sum()),
            c=[cmap(i)],
            label=f"Cluster {lab}",
            edgecolors="k",
            linewidths=0.4,
            s=60,
            alpha=0.85,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Topology Spectral Clusters")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        try:
            plt.show()
        except Exception:
            pass  # headless environment

    return fig
