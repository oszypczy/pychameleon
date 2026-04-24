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

from pychameleon._types import AdjacencyList, EdgeWeights, Labels


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
