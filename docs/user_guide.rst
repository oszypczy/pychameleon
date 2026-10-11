User guide
==========

How CHAMELEON works
-------------------

CHAMELEON runs in three steps.

**0. k-nearest-neighbor graph.** Every sample becomes a vertex connected to
its ``k_nn`` nearest neighbors (the graph is symmetrized). The weight of an
edge is the similarity ``1 / distance``, so near points are strongly
connected and points in sparse regions are weakly connected. Neighbors are
found with a k-d tree in :math:`O(n \log n)`.

**I. Partitioning into sub-clusters.** The graph is split with METIS
min-cut bisection, always bisecting the largest remaining part, until every
part is no larger than ``min_cluster_size``. The result is many small,
tightly connected sub-clusters.

**II. Agglomerative merging with dynamic modeling.** Sub-clusters are
merged pairwise, best pair first, until ``n_clusters`` remain. Two measures
compare a candidate pair :math:`C_i, C_j` against the clusters' own internal
structure. Here :math:`EC_{C_i,C_j}` are the edges between the two clusters
and :math:`EC_{C_i}` are the edges cut when :math:`C_i` is itself bisected.

*Relative interconnectivity* — how strongly the clusters are linked,
compared with how strongly each one holds together:

.. math::

   RI(C_i, C_j) = \frac{|EC_{C_i,C_j}|}{\tfrac{1}{2}\left(|EC_{C_i}| + |EC_{C_j}|\right)}

*Relative closeness* — how close the clusters are (mean weight
:math:`\bar S` of connecting edges), compared with each cluster's internal
closeness:

.. math::

   RC(C_i, C_j) = \frac{\bar S_{EC_{C_i,C_j}}}
   {\frac{|C_i|}{|C_i|+|C_j|}\bar S_{EC_{C_i}} + \frac{|C_j|}{|C_i|+|C_j|}\bar S_{EC_{C_j}}}

The pair with the highest score

.. math::

   \text{score}(C_i, C_j) = RI(C_i, C_j) \cdot RC(C_i, C_j)^{\alpha}

is merged next. Because both measures are *relative*, the same rule adapts
to dense and sparse clusters alike — this is what lets CHAMELEON separate
clusters that a single global distance threshold cannot.

Merging stops early if no adjacent pair has a positive score; in that case
:attr:`~pychameleon.Chameleon.n_clusters_` is larger than the requested
``n_clusters``.

Choosing parameters
-------------------

``n_clusters`` (default ``8``)
   Target number of clusters. Unlike density-based methods, CHAMELEON needs
   it up front.

``k_nn`` (default ``10``)
   Neighbors per point in the graph. Larger values connect points across
   wider gaps and make the result smoother, but can bridge clusters that
   should stay separate. Values of 10–20 work well for 2-D data; increase it
   for noisy or higher-dimensional data. It is capped at ``n_samples - 1``.

``min_cluster_size`` (default ``0.025``)
   Size limit for Phase I sub-clusters. A float in ``(0, 1)`` is a fraction
   of ``n_samples`` (the paper's recommendation is 1–5%); an integer ≥ 2 is
   an absolute count. Smaller values give Phase II more, finer pieces to
   work with, at a higher computational cost.

``alpha`` (default ``2.0``)
   Weight of closeness relative to interconnectivity in the merge score.
   ``alpha > 1`` favors merging clusters that are close; ``alpha < 1``
   favors clusters that are strongly connected.

Practical notes
---------------

- Features should be on comparable scales — distances drive the whole
  algorithm. Use :class:`~sklearn.preprocessing.StandardScaler` in a
  :class:`~sklearn.pipeline.Pipeline` when they are not.
- Only Euclidean distance on dense input is supported.
- Phase II recomputes merge scores after each merge, so runtime grows
  quickly with the number of Phase I sub-clusters (``1 / min_cluster_size``).
  With the defaults, fitting 20 000 2-D points takes a few seconds.
- Like other clusterers without a ``predict`` method (e.g.
  :class:`~sklearn.cluster.AgglomerativeClustering`), the model labels the
  data it was fit on; use :meth:`~pychameleon.Chameleon.fit_predict`.

Reference
---------

Karypis, G., Han, E.-H., & Kumar, V. (1999). *Chameleon: hierarchical
clustering using dynamic modeling.* Computer, 32(8), 68–75.
`doi:10.1109/2.781637 <https://doi.org/10.1109/2.781637>`_

Karypis, G., & Kumar, V. (1998). *A fast and high quality multilevel scheme
for partitioning irregular graphs.* SIAM Journal on Scientific Computing,
20(1), 359–392. `doi:10.1137/S1064827595287997 <https://doi.org/10.1137/S1064827595287997>`_
