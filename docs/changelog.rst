Changelog
=========

0.1.0 (2026-10-11)
------------------

First public release.

- :class:`~pychameleon.Chameleon` estimator implementing the three phases of
  CHAMELEON: k-NN graph, METIS-based partitioning and agglomerative merging
  with relative interconnectivity and closeness.
- Passes scikit-learn's ``check_estimator``.
- Supports Python 3.11–3.13 on Linux and macOS.
