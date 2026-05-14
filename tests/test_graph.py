"""Unit tests for :mod:`pychameleon.graph`.

Validate the k-NN graph construction with hand-computed reference values:

- 4-point unit square: each corner's two nearest neighbors are exactly known.
- 3-blob synthetic: edges should stay inside blobs (no cross-blob edges).
- Symmetry, no-self-loops, weight = 1/distance invariants.
"""
from __future__ import annotations

import numpy as np
import pytest

from pychameleon.graph import knn_graph


class TestKnnGraphBasics:
    def test_unit_square_k1(self) -> None:
        """4-point unit square, k=1: each corner's nearest is along an edge."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        adj, w = knn_graph(X, k=1)

        assert len(adj) == 4
        for i, neighbors in enumerate(adj):
            # symmetric => may have >=1 neighbor after symmetrization
            assert i not in neighbors.tolist(), f"node {i} has self-loop"
            assert len(neighbors) >= 1
        # Every edge weight should be 1/1.0 == 1.0 (unit-distance edges only).
        for weights in w:
            assert np.allclose(weights, 1.0)

    def test_unit_square_k2(self) -> None:
        """k=2: each corner connects to two unit-distance neighbors."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        adj, _ = knn_graph(X, k=2)

        for i, neighbors in enumerate(adj):
            assert i not in neighbors.tolist()
            # At k=2, the diagonal corner (distance sqrt(2)) is excluded;
            # symmetrization may add it back from another vertex's k-NN.
            assert len(neighbors) >= 2

    def test_symmetry(self, small_blobs: np.ndarray) -> None:
        """j in adj[i] <=> i in adj[j]."""
        adj, _ = knn_graph(small_blobs, k=5)
        for i, neighbors in enumerate(adj):
            for j in neighbors.tolist():
                assert i in adj[j].tolist(), f"asymmetry: {i}->{j} but not back"

    def test_no_self_loops(self, small_blobs: np.ndarray) -> None:
        adj, _ = knn_graph(small_blobs, k=5)
        for i, neighbors in enumerate(adj):
            assert i not in neighbors.tolist()

    def test_weight_is_inverse_distance(self) -> None:
        """w(i, j) ≈ 1 / euclidean_distance(i, j)."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        adj, w = knn_graph(X, k=3)

        for i, (neighbors, weights) in enumerate(zip(adj, w, strict=True)):
            for j, weight in zip(neighbors.tolist(), weights.tolist(), strict=True):
                expected = 1.0 / np.linalg.norm(X[i] - X[j])
                assert np.isclose(weight, expected, rtol=1e-9)

    def test_weights_aligned_with_adjacency(self, small_blobs: np.ndarray) -> None:
        """edge_weights[i] and adjacency[i] must have identical lengths."""
        adj, w = knn_graph(small_blobs, k=4)
        for neighbors, weights in zip(adj, w, strict=True):
            assert neighbors.shape == weights.shape


class TestKnnGraphSemantics:
    def test_three_blobs_edges_stay_local(self, small_blobs: np.ndarray) -> None:
        """For well-separated blobs, k-NN edges should not cross blob boundaries.

        small_blobs has 3 clusters of 20 points each, centered at (0,0), (5,5),
        (0,5). With k=5 (well below blob size 20), every neighbor should belong
        to the same blob.
        """
        adj, _ = knn_graph(small_blobs, k=5)
        # blob membership encoded by construction order: rows 0-19, 20-39, 40-59.
        blob_id = np.repeat(np.arange(3), 20)
        for i, neighbors in enumerate(adj):
            for j in neighbors.tolist():
                assert blob_id[i] == blob_id[j], (
                    f"edge {i}->{j} crosses blobs ({blob_id[i]} vs {blob_id[j]})"
                )

    def test_handles_duplicate_points(self) -> None:
        """Coincident points (distance 0) must not produce inf weights."""
        X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        _, w = knn_graph(X, k=2)
        for weights in w:
            assert np.all(np.isfinite(weights)), "weights must be finite"
            assert np.all(weights > 0)

    def test_returns_int64_indices(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        adj, _ = knn_graph(X, k=1)
        for neighbors in adj:
            assert neighbors.dtype == np.int64

    def test_returns_float64_weights(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        _, w = knn_graph(X, k=1)
        for weights in w:
            assert weights.dtype == np.float64


class TestKnnGraphValidation:
    def test_k_too_large(self) -> None:
        """k >= n is invalid (no point has n-1 distinct neighbors)."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        with pytest.raises((ValueError, IndexError)):
            knn_graph(X, k=3)
