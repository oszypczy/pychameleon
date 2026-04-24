"""Smoke tests for the :class:`Chameleon` estimator skeleton.

These tests verify that the sklearn-compatible API wiring is correct — the
actual clustering logic is covered by per-module tests once the internals are
implemented.
"""
from __future__ import annotations

import numpy as np
import pytest

from pychameleon import Chameleon


class TestChameleonAPI:
    def test_default_construction(self) -> None:
        c = Chameleon()
        assert c.n_clusters == 8
        assert c.k_nn == 10
        assert c.min_cluster_size == 0.025
        assert c.alpha == 2.0

    def test_get_params_roundtrip(self) -> None:
        c = Chameleon(n_clusters=5, k_nn=15, min_cluster_size=0.05, alpha=1.5)
        params = c.get_params()
        assert params == {
            "n_clusters": 5,
            "k_nn": 15,
            "min_cluster_size": 0.05,
            "alpha": 1.5,
        }

    def test_set_params(self) -> None:
        c = Chameleon().set_params(n_clusters=3, alpha=3.0)
        assert c.n_clusters == 3
        assert c.alpha == 3.0

    def test_clone_preserves_params(self) -> None:
        from sklearn.base import clone

        original = Chameleon(n_clusters=4, k_nn=20)
        cloned = clone(original)
        assert cloned.get_params() == original.get_params()
        assert cloned is not original

    def test_version_exposed(self) -> None:
        import pychameleon

        assert isinstance(pychameleon.__version__, str)
        assert pychameleon.__version__.count(".") == 2


class TestChameleonValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_clusters": 0}, "n_clusters"),
            ({"n_clusters": -1}, "n_clusters"),
            ({"k_nn": 1}, "k_nn"),
            ({"min_cluster_size": 0}, "min_cluster_size"),
            ({"min_cluster_size": 1.5}, "min_cluster_size"),
            ({"alpha": 0}, "alpha"),
            ({"alpha": -1.0}, "alpha"),
        ],
    )
    def test_invalid_params_rejected(
        self, small_blobs: np.ndarray, kwargs: dict, match: str
    ) -> None:
        c = Chameleon(**kwargs)
        with pytest.raises(ValueError, match=match):
            c.fit(small_blobs)


@pytest.mark.skip(reason="requires implementation of graph/partition/metrics/merger")
class TestChameleonEndToEnd:
    """End-to-end tests — enable once core modules are implemented."""

    def test_fit_predict_returns_labels(self, small_blobs: np.ndarray) -> None:
        labels = Chameleon(n_clusters=3).fit_predict(small_blobs)
        assert labels.shape == (60,)
        assert set(np.unique(labels).tolist()) <= {0, 1, 2}

    def test_aggregation_matches_reference(self, aggregation_xy: np.ndarray) -> None:
        from sklearn.metrics import adjusted_rand_score

        labels = Chameleon(n_clusters=7, k_nn=20, min_cluster_size=0.05).fit_predict(
            aggregation_xy
        )
        reference_labels = np.loadtxt(
            "benchmarks/reference_moonpuck/aggregation/labels.csv",
            delimiter=",",
            skiprows=1,
            usecols=(2,),
            dtype=np.int64,
        )
        assert adjusted_rand_score(reference_labels, labels) > 0.85
