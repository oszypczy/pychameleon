"""Dynamic similarity metrics — CHAMELEON §4.3.

Three functions implement the paper's dynamic modeling:

- :func:`relative_interconnectivity` — eq. (1): how strongly two clusters are
  connected across their boundary, normalized by their internal connectivity.
- :func:`relative_closeness` — eq. (2): average edge weight across the boundary,
  normalized by internal edge weight averages.
- :func:`merge_score` — eq. (4): ``RI(Ci, Cj) * RC(Ci, Cj)**alpha``, the
  combined criterion that drives Phase II merging decisions.

All three require the ability to bisect a sub-cluster (to measure its internal
inter-connectivity via the min-cut bisector EC_{Ci}). That's delegated to
pymetis under the hood.
"""
from __future__ import annotations

import numpy as np
import pymetis

from pychameleon._types import AdjacencyList, EdgeWeights, Labels

_WEIGHT_QUANTIZATION = 1_000_000
_MAX_INT_WEIGHT = 2_000_000_000


# Per-fit cache for internal bisector statistics. Keyed by frozenset of node
# ids; this avoids re-bisecting the same sub-cluster every time it's involved
# in a merge-score computation. Cleared explicitly at the start of fit() so
# the cache doesn't leak across calls.
_internal_bisector_cache: dict[frozenset[int], tuple[float, float]] = {}


def reset_cache() -> None:
    """Clear the internal-bisector cache. Call at the start of each fit()."""
    _internal_bisector_cache.clear()


def _internal_bisector(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    cluster_nodes: np.ndarray,
) -> tuple[float, float]:
    """Bisect ``cluster_nodes`` and return (cut_weight_sum, mean_cut_weight).

    A cluster's "internal interconnectivity" |EC_C| is the summed weight of
    edges crossing a min-cut bisection of its induced sub-graph. The mean is
    used by RC's denominator.
    """
    key = frozenset(int(n) for n in cluster_nodes)
    cached = _internal_bisector_cache.get(key)
    if cached is not None:
        return cached

    n_sub = cluster_nodes.shape[0]
    if n_sub < 2:
        result = (0.0, 0.0)
        _internal_bisector_cache[key] = result
        return result

    global_to_local = {int(g): i for i, g in enumerate(cluster_nodes)}

    xadj: list[int] = [0]
    adjncy: list[int] = []
    eweights: list[int] = []
    raw_weights: list[float] = []  # parallel to adjncy, original float weights

    for g in cluster_nodes:
        for nbr, w in zip(adjacency[int(g)], edge_weights[int(g)], strict=True):
            nbr_int = int(nbr)
            if nbr_int in global_to_local:
                adjncy.append(global_to_local[nbr_int])
                quantized = max(1, min(_MAX_INT_WEIGHT, int(w * _WEIGHT_QUANTIZATION)))
                eweights.append(quantized)
                raw_weights.append(float(w))
        xadj.append(len(adjncy))

    if not adjncy:
        result = (0.0, 0.0)
        _internal_bisector_cache[key] = result
        return result

    _, membership = pymetis.part_graph(
        2,
        xadj=xadj,
        adjncy=adjncy,
        eweights=eweights,
    )
    membership_arr = np.asarray(membership, dtype=np.int64)

    # Sum weights of edges that cross the bisection. Each undirected edge
    # appears twice in the CSR (once per endpoint); dividing by 2 corrects.
    cut_total = 0.0
    cut_count = 0
    for i in range(n_sub):
        for k in range(xadj[i], xadj[i + 1]):
            j = adjncy[k]
            if membership_arr[i] != membership_arr[j]:
                cut_total += raw_weights[k]
                cut_count += 1
    cut_total /= 2.0
    cut_count //= 2
    mean_cut = cut_total / cut_count if cut_count > 0 else 0.0

    result = (cut_total, mean_cut)
    _internal_bisector_cache[key] = result
    return result


def _cross_edges(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    labels: Labels,
    cluster_i: int,
    cluster_j: int,
) -> np.ndarray:
    """Return weights of all undirected edges with one endpoint in i, the other in j."""
    crossing: list[float] = []
    for v in np.where(labels == cluster_i)[0]:
        nbrs = adjacency[int(v)]
        ws = edge_weights[int(v)]
        for nbr, w in zip(nbrs.tolist(), ws.tolist(), strict=True):
            if labels[nbr] == cluster_j:
                crossing.append(float(w))
    return np.asarray(crossing, dtype=np.float64)


def relative_interconnectivity(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    labels: Labels,
    cluster_i: int,
    cluster_j: int,
) -> float:
    """Compute RI(Ci, Cj) = |EC_{Ci,Cj}| / ((|EC_Ci| + |EC_Cj|) / 2).

    The numerator is the summed weight of edges crossing the Ci-Cj boundary;
    the denominator is the average of the internal inter-connectivities
    (bisector edge-cut sums) of the two clusters.
    """
    cross = _cross_edges(adjacency, edge_weights, labels, cluster_i, cluster_j)
    cross_sum = float(cross.sum())
    if cross_sum == 0.0:
        return 0.0

    nodes_i = np.where(labels == cluster_i)[0]
    nodes_j = np.where(labels == cluster_j)[0]
    ec_i, _ = _internal_bisector(adjacency, edge_weights, nodes_i)
    ec_j, _ = _internal_bisector(adjacency, edge_weights, nodes_j)

    denom = (ec_i + ec_j) / 2.0
    if denom == 0.0:
        return 0.0
    return cross_sum / denom


def relative_closeness(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    labels: Labels,
    cluster_i: int,
    cluster_j: int,
) -> float:
    """Compute RC(Ci, Cj) from eq. (2).

    Ratio of mean edge weight across the Ci-Cj boundary to the size-weighted
    mean of the two clusters' internal bisector edge weights.
    """
    cross = _cross_edges(adjacency, edge_weights, labels, cluster_i, cluster_j)
    if cross.size == 0:
        return 0.0
    mean_cross = float(cross.mean())

    nodes_i = np.where(labels == cluster_i)[0]
    nodes_j = np.where(labels == cluster_j)[0]
    _, mean_i = _internal_bisector(adjacency, edge_weights, nodes_i)
    _, mean_j = _internal_bisector(adjacency, edge_weights, nodes_j)

    size_i = nodes_i.shape[0]
    size_j = nodes_j.shape[0]
    denom = (size_i * mean_i + size_j * mean_j) / (size_i + size_j)
    if denom == 0.0:
        return 0.0
    return float(mean_cross / denom)


def merge_score(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    labels: Labels,
    cluster_i: int,
    cluster_j: int,
    alpha: float,
) -> float:
    """Compute RI(Ci, Cj) * RC(Ci, Cj)**alpha (eq. 4).

    ``alpha > 1`` biases toward closeness; ``alpha < 1`` toward
    interconnectivity. The original paper uses ``alpha = 2.0``.
    """
    ri = relative_interconnectivity(adjacency, edge_weights, labels, cluster_i, cluster_j)
    if ri == 0.0:
        return 0.0
    rc = relative_closeness(adjacency, edge_weights, labels, cluster_i, cluster_j)
    if rc == 0.0:
        return 0.0
    return float(ri * (rc**alpha))
