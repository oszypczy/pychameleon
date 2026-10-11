chameleon-cluster
=================

**chameleon-cluster** is a Python implementation of CHAMELEON, the
hierarchical clustering algorithm of Karypis, Han & Kumar (1999), exposed
through a scikit-learn compatible estimator.

CHAMELEON finds clusters of arbitrary shape, density and size. Instead of a
fixed global merging rule, it decides whether two clusters belong together
by comparing how strongly and how closely they are connected *relative to
their own internal structure* — so-called dynamic modeling.

.. code-block:: python

   from pychameleon import Chameleon

   labels = Chameleon(n_clusters=2).fit_predict(X)

The estimator passes scikit-learn's :func:`~sklearn.utils.estimator_checks.check_estimator`,
so it works with :func:`~sklearn.base.clone`, pipelines and model selection
tools such as :class:`~sklearn.model_selection.GridSearchCV`.

.. plot::
   :include-source: False

   import matplotlib.pyplot as plt
   from sklearn.cluster import KMeans
   from sklearn.datasets import make_moons
   from pychameleon import Chameleon

   X, _ = make_moons(n_samples=600, noise=0.06, random_state=0)
   fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
   for ax, (name, model) in zip(axes, [
       ("KMeans", KMeans(n_clusters=2, n_init=10, random_state=0)),
       ("Chameleon", Chameleon(n_clusters=2)),
   ]):
       ax.scatter(X[:, 0], X[:, 1], c=model.fit_predict(X), s=6, cmap="coolwarm")
       ax.set_title(name)
       ax.set_xticks([]); ax.set_yticks([])

Citing
------

If you use chameleon-cluster in your work, please cite the original paper
and this software; citation metadata is in
`CITATION.cff <https://github.com/oszypczy/pychameleon/blob/main/CITATION.cff>`_
(GitHub shows a *Cite this repository* button for it).

.. toctree::
   :maxdepth: 2

   install
   user_guide
   examples
   api
   changelog
