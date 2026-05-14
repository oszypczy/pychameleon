"""k-nearest-neighbor graph construction (CHAMELEON §4.2).

Builds a sparse k-NN graph where each vertex is a data point and edges connect
each point to its k most-similar neighbors. Edge weights are similarities
(1 / distance) — larger weight means more similar.

Uses scipy.spatial.cKDTree for O(n log n) neighbor lookup in low-dimensional
Euclidean space; the Moonpuck reference uses a naive O(n^2) scan which
dominates runtime on large inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pychameleon._types import AdjacencyList, EdgeWeights, FloatMatrix

# Cap for inverse-distance weight when two points coincide. Treating distance==0
# as "infinitely similar" with a finite, large weight keeps pymetis happy
# (it requires positive integer weights after quantization) without losing
# the qualitative ranking (duplicates remain the most-similar pair).
_DUPLICATE_WEIGHT = 1e12


def knn_graph(X: FloatMatrix, k: int) -> tuple[AdjacencyList, EdgeWeights]:
    """Build a symmetric k-NN graph from points.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input points. Must be dense, dtype float64 recommended.
    k : int
        Number of nearest neighbors per point (excluding self).

    Returns
    -------
    adjacency : list of ndarray
        ``adjacency[i]`` is the sorted array of neighbor indices of point ``i``.
    edge_weights : list of ndarray
        ``edge_weights[i][j]`` is the similarity between ``i`` and
        ``adjacency[i][j]``, computed as ``1 / euclidean_distance``.
    """
    n = X.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than n_samples={n}")

    tree = cKDTree(X)
    # Query k+1 neighbors so we can drop the trivial self-match at index 0.
    distances, indices = tree.query(X, k=k + 1)

    # Build directed neighbor sets first, then symmetrize.
    neighbors: list[set[int]] = [set() for _ in range(n)]
    weights_map: list[dict[int, float]] = [{} for _ in range(n)]

    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:], strict=True):
            j_int = int(j)
            if j_int == i:  # safeguard against degenerate query results
                continue
            w = _DUPLICATE_WEIGHT if dist == 0.0 else 1.0 / float(dist)
            # Symmetrize: write the edge in both directions, keeping the larger
            # of the two weights (they should agree, but rounding can differ).
            for src, dst in ((i, j_int), (j_int, i)):
                if dst not in neighbors[src]:
                    neighbors[src].add(dst)
                    weights_map[src][dst] = w
                else:
                    weights_map[src][dst] = max(weights_map[src][dst], w)

    adjacency: AdjacencyList = []
    edge_weights: EdgeWeights = []
    for i in range(n):
        sorted_neighbors = np.array(sorted(neighbors[i]), dtype=np.int64)
        sorted_weights = np.array(
            [weights_map[i][int(j)] for j in sorted_neighbors],
            dtype=np.float64,
        )
        adjacency.append(sorted_neighbors)
        edge_weights.append(sorted_weights)

    return adjacency, edge_weights
