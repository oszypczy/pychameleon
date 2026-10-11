"""Shared type aliases for pychameleon.

Keeping these in one place makes signatures readable and refactorable.
"""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Dense 2-D array of samples (n_samples, n_features).
FloatMatrix: TypeAlias = NDArray[np.float64]

# 1-D integer array of cluster labels, shape (n_samples,).
Labels: TypeAlias = NDArray[np.int64]

# 1-D float array of edge weights.
Weights: TypeAlias = NDArray[np.float64]

# Adjacency list: for node i, `adjacency[i]` is the array of neighbor indices.
# Matches pymetis' native input format.
AdjacencyList: TypeAlias = list[NDArray[np.int64]]

# Edge weights aligned with AdjacencyList: `edge_weights[i][j]` is the weight
# of the edge from node `i` to its j-th neighbor `adjacency[i][j]`.
EdgeWeights: TypeAlias = list[NDArray[np.float64]]

# Memo of internal-bisector statistics (cut_weight_sum, mean_cut_weight),
# keyed by the node set of a sub-cluster. Owned by a single fit() call.
BisectorCache: TypeAlias = dict[frozenset[int], tuple[float, float]]
