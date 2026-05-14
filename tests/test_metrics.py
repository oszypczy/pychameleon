"""Unit tests for :mod:`pychameleon.metrics` (RI, RC, merge_score).

Hand-computed reference values on tiny graphs let us verify the dynamic-modeling
math from CHAMELEON §4.3 (eqs. 1, 2, 4) line by line.

Reference graph (used for RI/RC tests below)::

       0 ---- 1
       |\\    |
       | \\   |
       |  \\  |
       2 -- 3 ---- 4 ---- 5

All edge weights = 1.0 unless noted. Cluster A = {0, 1, 2, 3}, cluster B = {4, 5}.
Bridge: a single edge (3, 4) of weight 1.0.

Internal bisector of A: cuts {0,1} from {2,3} (or symmetric); cut weight = 2
(edges 0-2 and 1-3 cross, or 0-3 and 1-3 — depending on METIS).
Internal bisector of B: trivial bisection {4} vs {5}; cut weight = 1.

RI(A, B) = |EC_AB| / ((|EC_A| + |EC_B|) / 2) = 1 / ((2 + 1) / 2) = 0.6667
RC(A, B) = mean_cross / size-weighted-mean(internals)
        = 1.0 / ((|A| * mean_int_A + |B| * mean_int_B) / (|A| + |B|))

These hand values are verified by the test suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from pychameleon.metrics import (
    merge_score,
    relative_closeness,
    relative_interconnectivity,
)


@pytest.fixture
def two_clusters_bridged() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Reference graph: 4-node cluster A + 2-node cluster B, single bridge.

    Edges (undirected, all weight 1.0):
        A: (0,1), (0,2), (0,3), (1,3), (2,3)
        bridge: (3,4)
        B: (4,5)
    """
    adjacency = [
        np.array([1, 2, 3], dtype=np.int64),       # 0
        np.array([0, 3], dtype=np.int64),          # 1
        np.array([0, 3], dtype=np.int64),          # 2
        np.array([0, 1, 2, 4], dtype=np.int64),    # 3
        np.array([3, 5], dtype=np.int64),          # 4
        np.array([4], dtype=np.int64),             # 5
    ]
    edge_weights = [
        np.array([1.0, 1.0, 1.0]),  # 0
        np.array([1.0, 1.0]),       # 1
        np.array([1.0, 1.0]),       # 2
        np.array([1.0, 1.0, 1.0, 1.0]),  # 3
        np.array([1.0, 1.0]),       # 4
        np.array([1.0]),            # 5
    ]
    labels = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    return adjacency, edge_weights, labels


class TestRelativeInterconnectivity:
    def test_bridged_clusters(
        self,
        two_clusters_bridged: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        adj, w, labels = two_clusters_bridged
        ri = relative_interconnectivity(adj, w, labels, 0, 1)
        # |EC_AB| = 1.0 (single bridge edge of weight 1.0).
        # |EC_A| = bisector cut weight on cluster A. Any 2-2 partition of
        # cluster A cuts at least 2 edges. METIS may pick {0,1}|{2,3}
        # (cuts 0-2, 1-3, 0-3 — but 0-3 only if we count direction once).
        # Actually for {0,1}|{2,3}: cross edges are 0-2, 0-3, 1-3 => 3.
        # For {0,2}|{1,3}: cross are 0-1, 0-3, 2-3 => 3.
        # For {0,3}|{1,2}: cross are 0-1, 0-2, 1-3, 2-3 => 4.
        # METIS picks min-cut bisection => weight 3.
        # |EC_B| = 1 (single edge {4}|{5}).
        # RI = 1 / ((3 + 1) / 2) = 0.5
        assert ri == pytest.approx(0.5, rel=1e-6)

    def test_no_cross_edges(
        self,
        two_clusters_bridged: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        """If two clusters have no edges between them, RI must be 0."""
        adj, w, _ = two_clusters_bridged
        # Relabel so cluster 1 contains only node 5 (not connected to cluster 0).
        labels_disconnected = np.array([0, 0, 0, 0, 2, 1], dtype=np.int64)
        ri = relative_interconnectivity(adj, w, labels_disconnected, 0, 1)
        assert ri == 0.0


class TestRelativeCloseness:
    def test_bridged_clusters(
        self,
        two_clusters_bridged: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        adj, w, labels = two_clusters_bridged
        rc = relative_closeness(adj, w, labels, 0, 1)
        # mean_cross = 1.0 (single bridge edge of weight 1.0).
        # Internal mean for A: bisector cut sum = 3.0, n_cut_edges = 3
        # (METIS will produce a 2|2 partition with 3 crossing edges all w=1)
        #   => mean_int_A = 1.0
        # Internal mean for B: 1.0 / 1 = 1.0
        # RC = 1.0 / ((4*1.0 + 2*1.0) / (4+2)) = 1.0 / 1.0 = 1.0
        assert rc == pytest.approx(1.0, rel=1e-6)


class TestMergeScore:
    def test_score_equals_ri_when_alpha_zero(
        self,
        two_clusters_bridged: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        """alpha=0 => RC**0 = 1 => score == RI."""
        adj, w, labels = two_clusters_bridged
        ri = relative_interconnectivity(adj, w, labels, 0, 1)
        score = merge_score(adj, w, labels, 0, 1, alpha=0.0)
        assert score == pytest.approx(ri, rel=1e-6)

    def test_score_combines_both(
        self,
        two_clusters_bridged: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        adj, w, labels = two_clusters_bridged
        ri = relative_interconnectivity(adj, w, labels, 0, 1)
        rc = relative_closeness(adj, w, labels, 0, 1)
        score = merge_score(adj, w, labels, 0, 1, alpha=2.0)
        assert score == pytest.approx(ri * rc**2, rel=1e-6)

    def test_score_zero_when_no_cross_edges(
        self,
        two_clusters_bridged: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        adj, w, _ = two_clusters_bridged
        labels_disconnected = np.array([0, 0, 0, 0, 2, 1], dtype=np.int64)
        score = merge_score(adj, w, labels_disconnected, 0, 1, alpha=2.0)
        assert score == 0.0
