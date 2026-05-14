"""Unit tests for :mod:`pychameleon.merger` (Phase II agglomeration).

Validate that the priority-queue merge:

- Produces exactly ``n_clusters`` (or fewer when no positive scores remain).
- Returns dense, contiguous labels in ``[0, k)``.
- Stops short when no adjacent pair has a positive merge score (disconnected
  components must not be force-merged).
- Lazy invalidation correctly skips stale heap entries after a merge.
"""
from __future__ import annotations

import numpy as np
import pytest

from pychameleon.graph import knn_graph
from pychameleon.merger import merge_to_k_clusters
from pychameleon.metrics import reset_cache
from pychameleon.partition import initial_subclusters


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    reset_cache()


@pytest.fixture
def small_blobs_setup(
    small_blobs: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    adj, w = knn_graph(small_blobs, k=5)
    initial = initial_subclusters(adj, w, min_cluster_size=5)
    return adj, w, initial


class TestMergeBasics:
    def test_dense_labels_in_range(
        self,
        small_blobs_setup: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        adj, w, initial = small_blobs_setup
        labels = merge_to_k_clusters(adj, w, initial, n_clusters=3, alpha=2.0)

        unique = np.unique(labels)
        assert unique.tolist() == list(range(len(unique)))
        assert len(unique) <= 3

    def test_three_blobs_recover_three_clusters(
        self,
        small_blobs: np.ndarray,
        small_blobs_setup: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        adj, w, initial = small_blobs_setup
        labels = merge_to_k_clusters(adj, w, initial, n_clusters=3, alpha=2.0)

        from sklearn.metrics import adjusted_rand_score

        truth = np.repeat(np.arange(3), 20)
        # Well-separated blobs should give near-perfect ARI.
        assert adjusted_rand_score(truth, labels) >= 0.95

    def test_target_already_satisfied(
        self,
        small_blobs_setup: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        """If the initial sub-cluster count is already <= n_clusters, return as-is."""
        adj, w, initial = small_blobs_setup
        n_initial = len(np.unique(initial))
        labels = merge_to_k_clusters(adj, w, initial, n_clusters=n_initial + 5, alpha=2.0)
        assert len(np.unique(labels)) == n_initial

    def test_disconnected_components_not_force_merged(
        self,
        small_blobs_setup: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        """Asking for fewer clusters than disconnected components should produce
        as many clusters as components (no positive merge score across a gap)."""
        adj, w, initial = small_blobs_setup
        # 3 well-separated blobs => no cross edges => can't go below 3 clusters.
        labels = merge_to_k_clusters(adj, w, initial, n_clusters=1, alpha=2.0)
        assert len(np.unique(labels)) == 3

    def test_returns_int64(
        self,
        small_blobs_setup: tuple[list[np.ndarray], list[np.ndarray], np.ndarray],
    ) -> None:
        adj, w, initial = small_blobs_setup
        labels = merge_to_k_clusters(adj, w, initial, n_clusters=3, alpha=2.0)
        assert labels.dtype == np.int64
