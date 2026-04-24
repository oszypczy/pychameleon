"""Graph partitioning — CHAMELEON Phase I (§4.4).

Repeatedly bisects the largest current partition via pymetis until no partition
exceeds ``min_cluster_size``. Produces the initial fine-grain clustering that
Phase II will agglomerate.

This is the ``hMETIS`` step from the paper. We use ``pymetis.part_graph`` in
2-way mode with edge weights (similarities), mirroring the paper's use of
edge-cut minimization on the weighted k-NN graph.
"""
from __future__ import annotations

from pychameleon._types import AdjacencyList, EdgeWeights, Labels


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
    raise NotImplementedError("initial_subclusters will be implemented in module phase")
