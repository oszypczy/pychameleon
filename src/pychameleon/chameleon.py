"""The public :class:`Chameleon` estimator.

Wires together the four internal modules (``graph``, ``partition``,
``metrics``, ``merger``) behind a scikit-learn-compatible estimator.

The class intentionally follows the conventions of
:class:`sklearn.cluster.HDBSCAN` — parameter constraints, ``fit/fit_predict``
signatures, ``labels_`` attribute — so that ``check_estimator`` passes once
the internals are filled in (see tests/test_sklearn_api.py).
"""
from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import validate_data

from pychameleon import graph, merger, metrics, partition
from pychameleon._types import Labels


class Chameleon(ClusterMixin, BaseEstimator):  # type: ignore[misc]
    """CHAMELEON hierarchical clustering using dynamic modeling.

    Parameters
    ----------
    n_clusters : int, default=8
        Target number of clusters to return.
    k_nn : int, default=10
        Number of neighbors in the k-nearest-neighbor graph (§4.2). Larger
        ``k_nn`` produces a denser graph and can bridge wider gaps between
        clusters.
    min_cluster_size : int or float, default=0.025
        Minimum initial sub-cluster size in Phase I.

        - ``float`` in (0, 1): interpreted as a fraction of ``n_samples``
          (e.g. 0.025 -> 2.5% of n). This matches the paper's recommendation.
        - ``int`` >= 2: interpreted as an absolute count.
    alpha : float, default=2.0
        Exponent for relative closeness in the merge score
        ``score = RI(Ci, Cj) * RC(Ci, Cj) ** alpha`` (§4.4, eq. 4).
        ``alpha > 1`` favors closeness; ``alpha < 1`` favors
        interconnectivity.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label for each point in ``[0, n_clusters_)``.
    n_clusters_ : int
        Number of clusters actually produced. May be less than
        ``n_clusters`` if Phase II ran out of positive-score merges.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    See Also
    --------
    sklearn.cluster.HDBSCAN : density-based hierarchical clustering.
    sklearn.cluster.AgglomerativeClustering : static-model agglomerative.

    References
    ----------
    .. [1] Karypis, G., Han, E.-H., & Kumar, V. (1999). Chameleon:
       hierarchical clustering using dynamic modeling.
       Computer, 32(8), 68-75.

    Examples
    --------
    >>> import numpy as np
    >>> from pychameleon import Chameleon
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 2))
    >>> model = Chameleon(n_clusters=3).fit(X)  # doctest: +SKIP
    >>> model.labels_                            # doctest: +SKIP
    array([0, 1, 2, 0, ...])
    """

    # Fitted attributes (set during :meth:`fit`).
    # Declared at class level so static type checkers recognize them; sklearn
    # convention is that trailing-underscore attributes are NOT hyperparameters.
    labels_: Labels
    n_clusters_: int
    n_features_in_: int

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "k_nn": [Interval(Integral, 2, None, closed="left")],
        "min_cluster_size": [
            Interval(Real, 0.0, 1.0, closed="neither"),
            Interval(Integral, 2, None, closed="left"),
        ],
        "alpha": [Interval(Real, 0.0, None, closed="neither")],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        k_nn: int = 10,
        min_cluster_size: int | float = 0.025,
        alpha: float = 2.0,
    ) -> None:
        self.n_clusters = n_clusters
        self.k_nn = k_nn
        self.min_cluster_size = min_cluster_size
        self.alpha = alpha

    def fit(self, X: ArrayLike, y: None = None) -> Chameleon:
        """Cluster the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training points.
        y : Ignored
            Not used, present for sklearn pipeline compatibility.

        Returns
        -------
        self : Chameleon
            The fitted estimator.
        """
        self._validate_params()
        X_arr = validate_data(self, X, accept_sparse=False, dtype=np.float64)
        n_samples = X_arr.shape[0]

        min_size = self._resolve_min_cluster_size(n_samples)
        # Cap k_nn at n_samples - 1 so very small inputs still work (sklearn's
        # check_estimator hits this with n=10 against the default k_nn=10).
        effective_k = min(self.k_nn, max(1, n_samples - 1))

        # Reset the per-fit cache so internal-bisector results don't leak
        # across calls or estimator instances.
        metrics.reset_cache()

        # Phase 0: build sparse k-NN graph (§4.2)
        adjacency, edge_weights = graph.knn_graph(X_arr, effective_k)

        # Phase I: recursively bisect into sub-clusters (§4.4)
        initial_labels = partition.initial_subclusters(adjacency, edge_weights, min_size)

        # Phase II: agglomeratively merge with dynamic modeling (§4.4)
        self.labels_ = merger.merge_to_k_clusters(
            adjacency, edge_weights, initial_labels, self.n_clusters, self.alpha
        )
        self.n_clusters_ = int(np.unique(self.labels_).shape[0])
        return self

    def fit_predict(self, X: ArrayLike, y: None = None) -> Labels:
        """Convenience: ``self.fit(X).labels_``."""
        return self.fit(X).labels_

    def _resolve_min_cluster_size(self, n_samples: int) -> int:
        """Translate the ``min_cluster_size`` parameter into an absolute count."""
        if isinstance(self.min_cluster_size, float):
            return max(2, int(self.min_cluster_size * n_samples))
        return int(self.min_cluster_size)
