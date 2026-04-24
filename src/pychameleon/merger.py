"""Agglomerative merging — CHAMELEON Phase II (§4.4).

Repeatedly merges the pair of sub-clusters with the highest
:func:`pychameleon.metrics.merge_score`, using a priority queue with lazy
invalidation (stale entries are dropped rather than eagerly removed, since
heapq has no decrease-key).

Stops when the cluster count reaches ``n_clusters`` or no adjacent pair has a
positive score.
"""
from __future__ import annotations

from pychameleon._types import AdjacencyList, EdgeWeights, Labels


def merge_to_k_clusters(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    initial_labels: Labels,
    n_clusters: int,
    alpha: float,
) -> Labels:
    """Agglomerate sub-clusters until ``n_clusters`` remain.

    Parameters
    ----------
    adjacency, edge_weights : k-NN graph from :mod:`pychameleon.graph`.
    initial_labels : sub-cluster assignments from :func:`pychameleon.partition.initial_subclusters`.
    n_clusters : target number of final clusters.
    alpha : exponent in the merge score (see :func:`pychameleon.metrics.merge_score`).

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Final cluster labels in [0, n_clusters).
    """
    raise NotImplementedError("merge_to_k_clusters will be implemented in module phase")
