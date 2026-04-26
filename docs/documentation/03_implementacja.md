# 3. Struktura implementacji

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

## 3.3. Decyzje projektowe

| Decyzja        | Wybór                    | Uzasadnienie                              |
|----------------|--------------------------|-------------------------------------------|
| Build backend  | hatchling                | minimalny, nowoczesny `pyproject.toml`    |
| Layout         | src-layout               | zapobiega bugom import-order (PEP 660)    |
| Type checker   | `mypy --strict`          | standard sklearn-style libs               |
| Linter         | `ruff`                   | 10× szybszy od pylint+flake8+isort razem  |
| API wejścia    | `X` (n × d, float64)     | konwencja sklearn; pipeline-compatible    |
| k-NN backend   | `scipy.spatial.KDTree`   | `O(n log n)`, built-in dep sklearn        |
| Graph repr.    | `list[ndarray]` adj.     | direct pymetis CSR; szybsze niż networkx  |

Wybór sklearn-compatible API od pierwszego commita motywowany długoterminowym
celem publikacji jako `pychameleon` na PyPI (lato 2026); precedens: HDBSCAN
przeniesiony z `scikit-learn-contrib` do `sklearn` core w 2023 r.
