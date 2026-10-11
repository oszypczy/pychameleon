Examples
========

Comparison with other clustering methods
----------------------------------------

Three toy datasets where cluster shape and density defeat centroid- and
variance-based methods. CHAMELEON runs with its default parameters; only
``n_clusters`` is set. The number in each corner is the adjusted Rand index
against the true labels (1.0 = perfect).

.. plot::

   import matplotlib.pyplot as plt
   from sklearn.cluster import AgglomerativeClustering, KMeans
   from sklearn.datasets import make_blobs, make_circles, make_moons
   from sklearn.metrics import adjusted_rand_score

   from pychameleon import Chameleon

   datasets = [
       ("Moons", make_moons(n_samples=600, noise=0.06, random_state=0), 2),
       ("Circles", make_circles(n_samples=600, noise=0.04, factor=0.45, random_state=0), 2),
       ("Varied density", make_blobs(
           n_samples=[300, 300, 150], centers=[[0, 0], [4, 0], [0, 4]],
           cluster_std=[0.4, 1.2, 0.4], random_state=0), 3),
   ]
   models = {
       "KMeans": lambda k: KMeans(n_clusters=k, n_init=10, random_state=0),
       "Agglomerative (Ward)": lambda k: AgglomerativeClustering(n_clusters=k),
       "Chameleon": lambda k: Chameleon(n_clusters=k),
   }

   fig, axes = plt.subplots(len(datasets), len(models), figsize=(9, 8.5),
                            layout="constrained")
   for row, (data_name, (X, y), k) in enumerate(datasets):
       for col, (model_name, make_model) in enumerate(models.items()):
           labels = make_model(k).fit_predict(X)
           ax = axes[row, col]
           ax.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap="tab10", vmax=9)
           ax.text(0.98, 0.03, f"{round(adjusted_rand_score(y, labels), 2) + 0.0:.2f}",
                   transform=ax.transAxes, ha="right", fontsize=10)
           ax.set_xticks([]); ax.set_yticks([])
           if row == 0:
               ax.set_title(model_name)
           if col == 0:
               ax.set_ylabel(data_name)

Using CHAMELEON in a pipeline
-----------------------------

``Chameleon`` is a regular scikit-learn estimator, so it composes with
preprocessing steps:

.. code-block:: python

   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from pychameleon import Chameleon

   pipe = make_pipeline(StandardScaler(), Chameleon(n_clusters=3, k_nn=15))
   labels = pipe.fit_predict(X)

Effect of ``alpha``
-------------------

``alpha`` trades off relative closeness against relative interconnectivity.
Inspecting a small grid is a quick way to see how sensitive a dataset is:

.. code-block:: python

   from sklearn.metrics import adjusted_rand_score

   for alpha in (0.5, 1.0, 2.0, 4.0):
       labels = Chameleon(n_clusters=3, alpha=alpha).fit_predict(X)
       print(alpha, adjusted_rand_score(y, labels))
