Changelog
=========

0.1.2 (2026-10-11)
------------------

Documentation and packaging only — no changes to the algorithm or API.

- Releases are archived on Zenodo with a citable DOI; ``CITATION.cff``
  added.
- Documentation notes that results are deterministic.
- Fixed the documentation badge in the README.

0.1.1 (2026-10-11)
------------------

Documentation and packaging only — no changes to the algorithm or API.

- Documentation published at https://chameleon-cluster.readthedocs.io
- PyPI page now links to the documentation and shows the updated README.

0.1.0 (2026-10-11)
------------------

First public release.

- :class:`~pychameleon.Chameleon` estimator implementing the three phases of
  CHAMELEON: k-NN graph, METIS-based partitioning and agglomerative merging
  with relative interconnectivity and closeness.
- Passes scikit-learn's ``check_estimator``.
- Supports Python 3.11–3.13 on Linux and macOS.
