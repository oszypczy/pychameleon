"""Graph partitioning — CHAMELEON Phase I (§4.4).

Repeatedly bisects the largest current partition via pymetis until no partition
exceeds ``min_cluster_size``. Produces the initial fine-grain clustering that
Phase II will agglomerate.

This is the ``hMETIS`` step from the paper. We use ``pymetis.part_graph`` in
2-way mode with edge weights (similarities), mirroring the paper's use of
edge-cut minimization on the weighted k-NN graph.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pymetis

from pychameleon._types import AdjacencyList, EdgeWeights, Labels

# pymetis requires positive integer edge weights. We quantize 1/distance
# similarities by this factor; values larger than this would risk overflow
# of the int32 weight type METIS uses internally.
_WEIGHT_QUANTIZATION = 1_000_000
_MAX_INT_WEIGHT = 2_000_000_000  # well below INT32_MAX


def _connected_components(
    nodes: np.ndarray,
    adjacency: AdjacencyList,
) -> list[np.ndarray]:
    """Return the connected components of the sub-graph induced by ``nodes``.

    METIS bisection gives degenerate (effectively random) cuts on disconnected
    inputs, which in CHAMELEON would arbitrarily slice a real cluster in half
    instead of separating the components. We split components first so each
    bisection call sees a connected graph.
    """
    node_set = {int(g) for g in nodes}
    visited: set[int] = set()
    components: list[np.ndarray] = []

    for seed in nodes:
        seed_int = int(seed)
        if seed_int in visited:
            continue
        component: list[int] = []
        queue: deque[int] = deque([seed_int])
        visited.add(seed_int)
        while queue:
            v = queue.popleft()
            component.append(v)
            for nbr in adjacency[v].tolist():
                if nbr in node_set and nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        components.append(np.array(component, dtype=np.int64))

    return components


def _bisect(
    nodes: np.ndarray,
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Bisect a sub-graph induced by ``nodes`` using METIS.

    Returns ``(left, right)`` node-id arrays, or ``None`` if METIS cannot
    produce a non-trivial bisection (e.g., disconnected component too small).
    """
    n_sub = nodes.shape[0]
    if n_sub < 2:
        return None

    # Map global ids -> local 0..n_sub-1.
    global_to_local = {int(g): i for i, g in enumerate(nodes)}

    # Build CSR (xadj/adjncy/eweights) on the induced sub-graph.
    xadj: list[int] = [0]
    adjncy: list[int] = []
    eweights: list[int] = []

    for g in nodes:
        for nbr, w in zip(adjacency[int(g)], edge_weights[int(g)], strict=True):
            nbr_int = int(nbr)
            if nbr_int in global_to_local:  # keep only induced edges
                adjncy.append(global_to_local[nbr_int])
                quantized = max(1, min(_MAX_INT_WEIGHT, int(w * _WEIGHT_QUANTIZATION)))
                eweights.append(quantized)
        xadj.append(len(adjncy))

    if not adjncy:
        # No induced edges — split arbitrarily in half.
        mid = n_sub // 2
        return nodes[:mid], nodes[mid:]

    _, membership = pymetis.part_graph(
        2,
        xadj=xadj,
        adjncy=adjncy,
        eweights=eweights,
    )
    membership_arr = np.asarray(membership, dtype=np.int64)
    left = nodes[membership_arr == 0]
    right = nodes[membership_arr == 1]

    if left.size == 0 or right.size == 0:
        # METIS sometimes returns a degenerate partition; fall back to halving.
        mid = n_sub // 2
        return nodes[:mid], nodes[mid:]

    return left, right


def initial_subclusters(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    min_cluster_size: int,
) -> Labels:
    """Recursively bisect the k-NN graph until each sub-cluster ≤ ``min_cluster_size``.

    Parameters
    ----------
    adjacency : list of ndarray
        Neighbor lists (from :func:`pychameleon.graph.knn_graph`).
    edge_weights : list of ndarray
        Similarity weights aligned with ``adjacency``.
    min_cluster_size : int
        Stop bisecting a partition when its size drops below this threshold.
        The paper suggests 1-5% of n_samples; 2.5% is used in the original
        experiments.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Sub-cluster id for each point in [0, n_subclusters).
    """
    n = len(adjacency)
    all_nodes = np.arange(n, dtype=np.int64)

    # Seed the frontier with the connected components of the full graph —
    # bisecting a disconnected graph produces degenerate cuts that slice real
    # clusters instead of separating them.
    frontier: list[np.ndarray] = _connected_components(all_nodes, adjacency)
    finalized: list[np.ndarray] = []

    # Floor for any individual sub-cluster. Children of a bisection are not
    # allowed to fall below this — otherwise an unbalanced METIS cut could
    # spawn tiny fragments that the paper's "min ≈ 2.5% of n" guideline is
    # meant to prevent.
    half_floor = max(2, min_cluster_size // 2)

    while frontier:
        cluster = frontier.pop()
        # Stop bisecting once the cluster is at or below the floor.
        if cluster.size <= min_cluster_size:
            finalized.append(cluster)
            continue

        result = _bisect(cluster, adjacency, edge_weights)
        if result is None:
            finalized.append(cluster)
            continue
        left, right = result

        # If METIS produced a very unbalanced cut, treating the small child
        # as a leaf risks tiny clusters; refuse the cut entirely.
        if min(left.size, right.size) < half_floor:
            finalized.append(cluster)
            continue

        # Re-queue both children for further potential bisection. After a cut,
        # the children may themselves be disconnected (e.g., if METIS happened
        # to split an already-marginal bridge), so re-decompose each.
        # Components below the half-floor are too small to keep on their own —
        # merge them back into the largest sibling component of the same child
        # rather than spawning fragmentary clusters.
        for child in (left, right):
            components = _connected_components(child, adjacency)
            big = [c for c in components if c.size >= half_floor]
            small = [c for c in components if c.size < half_floor]
            if big and small:
                # Attach small fragments to the largest big component.
                big.sort(key=lambda c: c.size, reverse=True)
                merged = np.concatenate([big[0], *small])
                big[0] = merged
            elif not big and small:
                # All fragments — concatenate into one cluster, treat as leaf.
                big = [np.concatenate(small)]

            for component in big:
                if component.size <= min_cluster_size:
                    finalized.append(component)
                else:
                    frontier.append(component)

    labels = np.empty(n, dtype=np.int64)
    for cluster_id, members in enumerate(finalized):
        labels[members] = cluster_id

    return labels
