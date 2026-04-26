# 3. Struktura implementacji

> **Wymaganie #1 z PDF prowadzącego:** *"strukturze implementacji (funkcje, klasy)
> z odniesieniem do konkretnych kroków implementowanego algorytmu"*

## 3.1. Architektura — 5 modułów

![Architektura pakietu pychameleon](images/architecture.png)

| Moduł              | Sekcja paperu  | Główne funkcje / klasy                                                |
|--------------------|----------------|------------------------------------------------------------------------|
| `graph.py`         | §4.2           | `knn_graph(X, k) -> (adjacency, edge_weights)`                         |
| `partition.py`     | §4.4 Phase I   | `initial_subclusters(adj, weights, min_size) -> labels`                |
| `metrics.py`       | §4.3           | `relative_interconnectivity`, `relative_closeness`, `merge_score`       |
| `merger.py`        | §4.4 Phase II  | `merge_to_k_clusters(adj, weights, init_labels, k, α) -> labels`        |
| `chameleon.py`     | §4 całość      | `class Chameleon(ClusterMixin, BaseEstimator)`                          |

`_types.py` dostarcza wspólnych aliasów (`FloatMatrix`, `Labels`, `AdjacencyList`,
`EdgeWeights`). Publiczne API: jedna klasa `Chameleon` (`from pychameleon import Chameleon`).

## 3.2. Mapowanie kroków algorytmu na implementację

| Krok algorytmu (paper)                            | Implementacja                                       |
|---------------------------------------------------|----------------------------------------------------|
| §4.2 budowa grafu k-NN                             | `KDTree.query(X, k+1)`, symetryzacja adjacency      |
| §4.2 wagi krawędzi `1/d(xᵢ,xⱼ)`                    | `weight = 1.0 / dist`                                |
| §4.4 Phase I bisekcja największego sub-klastra     | `pymetis.part_graph(2, objtype='cut', ufactor=250)` |
| §4.4 Phase II inicjalne wyliczenie `score` par     | iteracja po parach sąsiadujących + push do heap     |
| §4.4 Phase II wybór pary o max `score`             | `heapq.heappop` z lazy invalidation (version cnt.)  |
| §4.4 Phase II aktualizacja po scaleniu             | push tylko sąsiadów scalonego super-klastra         |
| §4.3 obliczenie EC_Cᵢ (bisekcja wewnętrzna)        | `pymetis.part_graph(2, …)` z cache wyników          |

## 3.3. Klasa `Chameleon` — sklearn-compatible API

```python
class Chameleon(ClusterMixin, BaseEstimator):
    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "n_clusters":       [Interval(Integral, 1, None, closed="left")],
        "k_nn":             [Interval(Integral, 2, None, closed="left")],
        "min_cluster_size": [Interval(Real, 0.0, 1.0, closed="neither"),
                             Interval(Integral, 2, None, closed="left")],
        "alpha":            [Interval(Real, 0.0, None, closed="neither")],
    }

    def __init__(self, n_clusters=8, k_nn=10, min_cluster_size=0.025, alpha=2.0): ...

    def fit(self, X, y=None) -> "Chameleon":
        self._validate_params()
        X_arr = self._validate_data(X, accept_sparse=False, dtype=np.float64)
        adjacency, edge_weights = graph.knn_graph(X_arr, self.k_nn)
        initial_labels = partition.initial_subclusters(
            adjacency, edge_weights, self._resolve_min_cluster_size(len(X_arr))
        )
        self.labels_ = merger.merge_to_k_clusters(
            adjacency, edge_weights, initial_labels, self.n_clusters, self.alpha
        )
        return self
```

Walidacja parametrów jest **deklaratywna** (`_parameter_constraints` — sklearn
framework). Atrybut `labels_` zgodny z konwencją sklearn (trailing underscore).
`n_features_in_` ustawiane automatycznie przez `_validate_data`.

## 3.4. Decyzje projektowe

| Decyzja             | Wybór                | Uzasadnienie                                       |
|---------------------|----------------------|----------------------------------------------------|
| Build backend       | hatchling            | minimalny, nowoczesny `pyproject.toml`             |
| Layout              | src-layout           | zapobiega bugom import-order (PEP 660)             |
| Type checker        | `mypy --strict`      | standard sklearn-style libs                        |
| Linter              | `ruff`               | 10× szybszy od pylint+flake8+isort razem           |
| API wejścia         | `X` (n × d, float64) | konwencja sklearn; pipeline-compatible             |
| k-NN backend        | `scipy.spatial.KDTree`| `O(n log n)`, built-in dep sklearn                |
| Graph repr.         | `list[ndarray]` adj. | direct pymetis CSR; szybsze niż networkx           |

Wybór sklearn-compatible API od pierwszego commita motywowany długoterminowym
celem publikacji jako `pychameleon` na PyPI (lato 2026); precedens: HDBSCAN
przeniesiony z `scikit-learn-contrib` do `sklearn` core w 2023 r.
