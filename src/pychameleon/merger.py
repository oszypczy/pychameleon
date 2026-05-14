"""Agglomerative merging — CHAMELEON Phase II (§4.4).

Repeatedly merges the pair of sub-clusters with the highest
:func:`pychameleon.metrics.merge_score`, using a priority queue with lazy
invalidation (stale entries are dropped rather than eagerly removed, since
heapq has no decrease-key).

Stops when the cluster count reaches ``n_clusters`` or no adjacent pair has a
positive score.
"""
from __future__ import annotations

import heapq

import numpy as np

from pychameleon._types import AdjacencyList, EdgeWeights, Labels
from pychameleon.metrics import merge_score


def _adjacent_cluster_pairs(
    adjacency: AdjacencyList, labels: np.ndarray
) -> set[tuple[int, int]]:
    """Set of unordered (a, b) cluster pairs that share at least one k-NN edge."""
    pairs: set[tuple[int, int]] = set()
    n = len(adjacency)
    for v in range(n):
        cv = int(labels[v])
        for nbr in adjacency[v].tolist():
            cn = int(labels[nbr])
            if cv != cn:
                pairs.add((min(cv, cn), max(cv, cn)))
    return pairs


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
    labels = initial_labels.copy().astype(np.int64)
    active_clusters = set(np.unique(labels).tolist())

    if len(active_clusters) <= n_clusters:
        return _densify_labels(labels)

    # Version counter per cluster id; bumped on every merge that involves it.
    version = dict.fromkeys(active_clusters, 0)

    # Heap entries: (-score, version_a, version_b, cluster_a, cluster_b).
    # We push -score so heapq (min-heap) pops the largest score first.
    heap: list[tuple[float, int, int, int, int]] = []

    def push_pair(a: int, b: int) -> None:
        a_, b_ = (a, b) if a < b else (b, a)
        score = merge_score(adjacency, edge_weights, labels, a_, b_, alpha)
        if score > 0.0:
            heapq.heappush(heap, (-score, version[a_], version[b_], a_, b_))

    # Seed the heap with all adjacent cluster pairs.
    for a, b in _adjacent_cluster_pairs(adjacency, labels):
        push_pair(a, b)

    while len(active_clusters) > n_clusters and heap:
        neg_score, ver_a, ver_b, a, b = heapq.heappop(heap)

        # Lazy invalidation: skip stale entries whose endpoint versions advanced.
        if a not in active_clusters or b not in active_clusters:
            continue
        if ver_a != version[a] or ver_b != version[b]:
            continue
        if -neg_score <= 0.0:
            break

        # Merge b into a (keep the smaller id; minor implementation choice).
        labels[labels == b] = a
        active_clusters.discard(b)
        version[a] += 1
        # Recompute scores for all pairs that touched the merged cluster.
        # Find new neighbors of `a` after the merge.
        new_neighbors: set[int] = set()
        for v in np.where(labels == a)[0]:
            for nbr in adjacency[int(v)].tolist():
                cn = int(labels[nbr])
                if cn != a and cn in active_clusters:
                    new_neighbors.add(cn)
        for nbr_cluster in new_neighbors:
            push_pair(a, nbr_cluster)

    return _densify_labels(labels)


def _densify_labels(labels: np.ndarray) -> Labels:
    """Remap cluster ids to a contiguous range ``[0, k)`` preserving order."""
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique.tolist())}
    return np.array([remap[int(v)] for v in labels], dtype=np.int64)
