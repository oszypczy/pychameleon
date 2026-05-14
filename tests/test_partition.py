"""Unit tests for :mod:`pychameleon.partition`.

Phase I: recursive METIS bisection. Validates that:

- All points get a label (no -1 / unlabeled).
- Every sub-cluster meets the size floor (within METIS bisection granularity).
- Number of sub-clusters scales as ``n / min_cluster_size``.
- Bisection respects connectivity: well-separated blobs go to distinct subs.
"""
from __future__ import annotations

import numpy as np
import pytest

from pychameleon.graph import knn_graph
from pychameleon.partition import initial_subclusters


@pytest.fixture
def small_blobs_graph(small_blobs: np.ndarray) -> tuple[list, list]:
    return knn_graph(small_blobs, k=5)


class TestInitialSubclusters:
    def test_labels_are_dense_and_complete(
        self, small_blobs: np.ndarray, small_blobs_graph: tuple[list, list]
    ) -> None:
        adj, w = small_blobs_graph
        labels = initial_subclusters(adj, w, min_cluster_size=10)

        assert labels.shape == (small_blobs.shape[0],)
        # No unlabeled points.
        assert np.all(labels >= 0)
        # Labels form a contiguous range [0, m).
        unique = np.unique(labels)
        assert unique.tolist() == list(range(len(unique)))

    def test_three_blobs_separated(
        self, small_blobs: np.ndarray, small_blobs_graph: tuple[list, list]
    ) -> None:
        """Each ground-truth blob should map to disjoint sub-cluster ids."""
        adj, w = small_blobs_graph
        # min=10 means 60-pt input bisects until each sub <= 10; expect ~6 subs.
        labels = initial_subclusters(adj, w, min_cluster_size=10)

        blob_id = np.repeat(np.arange(3), 20)
        # Labels assigned to blob 0 should not overlap with blobs 1, 2.
        for b1 in range(3):
            for b2 in range(b1 + 1, 3):
                set1 = set(labels[blob_id == b1].tolist())
                set2 = set(labels[blob_id == b2].tolist())
                assert set1.isdisjoint(set2), (
                    f"blob {b1} and blob {b2} share sub-cluster labels"
                )

    def test_subcluster_count_scales_with_min_size(
        self, small_blobs_graph: tuple[list, list]
    ) -> None:
        """Smaller min_cluster_size produces more sub-clusters."""
        adj, w = small_blobs_graph
        labels_large = initial_subclusters(adj, w, min_cluster_size=20)
        labels_small = initial_subclusters(adj, w, min_cluster_size=5)
        n_large = len(np.unique(labels_large))
        n_small = len(np.unique(labels_small))
        assert n_small > n_large

    def test_sizes_respect_floor(
        self, small_blobs: np.ndarray, small_blobs_graph: tuple[list, list]
    ) -> None:
        """No sub-cluster should be far below min_cluster_size.

        METIS bisection won't always hit the floor exactly (it stops splitting
        once children would fall below); we allow a 50% slack to absorb that.
        """
        adj, w = small_blobs_graph
        min_size = 10
        labels = initial_subclusters(adj, w, min_cluster_size=min_size)
        sizes = np.bincount(labels)
        # Every cluster should be reasonably sized — at least 50% of floor.
        assert np.all(sizes >= min_size // 2), f"sizes too small: {sizes}"

    def test_min_size_larger_than_n_yields_components_only(
        self, small_blobs_graph: tuple[list, list]
    ) -> None:
        """If ``min_cluster_size`` exceeds ``n``, no bisection occurs and the
        only sub-clusters are the connected components of the k-NN graph.

        For ``small_blobs`` (3 well-separated blobs, k=5) those components are
        exactly the 3 ground-truth blobs.
        """
        adj, w = small_blobs_graph
        labels = initial_subclusters(adj, w, min_cluster_size=1000)
        # 3 well-separated blobs => 3 connected components in the k-NN graph.
        assert len(np.unique(labels)) == 3

    def test_returns_int64(
        self, small_blobs_graph: tuple[list, list]
    ) -> None:
        adj, w = small_blobs_graph
        labels = initial_subclusters(adj, w, min_cluster_size=10)
        assert labels.dtype == np.int64
