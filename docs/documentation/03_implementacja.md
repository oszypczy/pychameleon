# 3. Struktura implementacji

> **Wymaganie #1 z PDF prowadzącego (str. 1):**
> *"strukturze implementacji (funkcje, klasy) z odniesieniem do konkretnych kroków
> implementowanego algorytmu"*

Niniejszy rozdział mapuje implementację jeden-do-jednego na sekcje paperu Karypis 1999.
Każda funkcja i klasa jest opisana z wyraźnym odniesieniem do kroku algorytmu, który
realizuje, oraz numeru sekcji w paperu źródłowym.

## 3.1. Architektura ogólna

Implementacja jest podzielona na **5 modułów** w katalogu `src/pychameleon/`:

| Moduł              | Sekcja paperu  | Faza algorytmu     | Główne funkcje / klasy                                   |
|--------------------|----------------|---------------------|----------------------------------------------------------|
| `graph.py`         | §4.2           | Faza 0              | `knn_graph(X, k)`                                        |
| `partition.py`     | §4.4 Phase I   | Faza I              | `initial_subclusters(adj, weights, min_size)`            |
| `metrics.py`       | §4.3           | (cross-cutting)     | `relative_interconnectivity`, `relative_closeness`, `merge_score` |
| `merger.py`        | §4.4 Phase II  | Faza II             | `merge_to_k_clusters(adj, weights, init_labels, k, α)`   |
| `chameleon.py`     | §4 całość      | spinający estymator | `class Chameleon(ClusterMixin, BaseEstimator)`           |

**Diagram architektury i przepływu danych:**

![Architektura pakietu pychameleon — 5 modułów mapowanych na sekcje paperu](../presentation/images/architecture.png)

`chameleon.py` jest publicznym API; pozostałe moduły są wywoływane sekwencyjnie
w metodzie `Chameleon.fit()`. `_types.py` (linie przerywane) dostarcza wspólnych
aliasów typów wszystkim pozostałym modułom.

**Diagram sekwencyjny wywołań w `Chameleon.fit(X)`:**

![Sekwencja wywołań między modułami w trakcie wywołania `Chameleon.fit(X)`](../presentation/images/algorithm_phases.png)

## 3.2. Moduł `graph.py` — §4.2 budowa grafu k-NN

**Plik:** `src/pychameleon/graph.py`
**Odpowiedzialność:** zbudowanie sparse k-najbliższych-sąsiadów grafu z punktów.

**Sygnatura funkcji (kopia 1:1 z `src/pychameleon/graph.py`):**

```python
def knn_graph(
    X: FloatMatrix,
    k: int,
) -> tuple[AdjacencyList, EdgeWeights]:
    """Build the symmetric k-NN graph used by CHAMELEON Phase I.

    Edges weights are 1 / distance (the paper uses inverse distance so that
    "closer" pairs get higher similarity).
    """
```

**Mapowanie krok paperu → linia kodu (planowane):**

| Krok paperu §4.2                          | Implementacja                                                              |
|-------------------------------------------|----------------------------------------------------------------------------|
| "build a *k*-nearest-neighbor graph"      | `tree = scipy.spatial.KDTree(X)`                                           |
| "for each point find *k* nearest neighbors" | `dists, idxs = tree.query(X, k=k+1)` (k+1 bo pierwszy to sam punkt)      |
| "set edge weight to similarity"           | `weight = 1.0 / dist` (inverse distance per paper)                         |
| "make graph symmetric"                    | dla każdego `(i, j)`: dodaj `j` do `adj[i]` ORAZ `i` do `adj[j]`           |

**Założenia projektowe:**

- `KDTree` zamiast naiwnego `O(n²)` — kluczowa optymalizacja vs Moonpuck (rozdz. 5),
- graf zwracany jako para list, kompatybilna z `pymetis.part_graph` (rozdz. 4).

## 3.3. Moduł `partition.py` — §4.4 Phase I rekurencyjna bisekcja

**Plik:** `src/pychameleon/partition.py`
**Odpowiedzialność:** podział grafu k-NN na `m` zwartych sub-klastrów.

**Sygnatura funkcji (z `src/pychameleon/partition.py`):**

```python
def initial_subclusters(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    min_cluster_size: int,
) -> Labels:
    """Phase I — partition the k-NN graph into ``m`` initial sub-clusters.

    Uses recursive 2-way min-cut bisection (METIS via pymetis) until every
    component has at most ``min_cluster_size`` vertices. ``m`` is determined
    by the data, not specified in advance.
    """
```

**Mapowanie krok paperu → implementacja:**

| Krok paperu §4.4 Phase I                                       | Implementacja                                                          |
|----------------------------------------------------------------|-----------------------------------------------------------------------|
| "while there exists a sub-cluster with > MinSize vertices"     | `while max(cluster_sizes) > min_cluster_size:`                        |
| "find the largest sub-cluster"                                 | `largest_idx = argmax(cluster_sizes)`                                 |
| "bisect it using min-cut partitioning"                         | `pymetis.part_graph(2, xadj=…, adjncy=…, eweights=…)`                 |
| "ufactor=250 (≤25% imbalance)"                                 | `pymetis_options.ufactor = 250`                                       |
| "return labels"                                                | `labels = np.array(…, dtype=np.int64)`                                |

**Wybór parametrów `pymetis`:**

- `objtype='cut'` (paper §4.4): minimalizujemy sumaryczną wagę krawędzi przeciętych,
- `ufactor=250` (paper §4.4): ograniczenie nierówności rozmiarów partycji do ±25%,
- `ncuts=1`: jedna próba (powtórzenia nie poprawiają jakości znacząco).

## 3.4. Moduł `metrics.py` — §4.3 miary podobieństwa

**Plik:** `src/pychameleon/metrics.py`
**Odpowiedzialność:** obliczanie RI, RC, merge_score między parami klastrów.

**Sygnatury funkcji (z `src/pychameleon/metrics.py`):**

```python
def relative_interconnectivity(
    adjacency, edge_weights, labels, cluster_i, cluster_j,
) -> float:
    """Compute RI(Ci, Cj) = |EC_{Ci,Cj}| / ((|EC_Ci| + |EC_Cj|) / 2)."""

def relative_closeness(
    adjacency, edge_weights, labels, cluster_i, cluster_j,
) -> float:
    """Compute RC(Ci, Cj) per eq. (2)."""

def merge_score(
    adjacency, edge_weights, labels, cluster_i, cluster_j, alpha,
) -> float:
    """Compute RI * RC**alpha (eq. 4)."""
```

**Mapowanie do równań paperu:**

| Funkcja                              | Równanie    | Sekcja  |
|--------------------------------------|-------------|---------|
| `relative_interconnectivity`         | eq. (1)     | §4.3    |
| `relative_closeness`                 | eq. (2)     | §4.3    |
| `merge_score`                        | eq. (4)     | §4.3    |

**Operacje pomocnicze (wewnętrzne, niewyeksportowane):**

- `_edge_cut_between(adj, weights, labels, ci, cj) → float`: sumaryczna waga
  krawędzi (i, j) takich, że `labels[i] == ci AND labels[j] == cj`.
- `_internal_interconnectivity(adj, weights, labels, ci) → float`: bisekcja
  pod-grafu klastra `ci` przez pymetis i suma wag przeciętych krawędzi (`|EC_Cᵢ|`).
- `_mean_edge_weight(weights, edges) → float`: średnia waga zbioru krawędzi.

**Kluczowa decyzja optymalizacyjna:** wartość `_internal_interconnectivity(ci)` jest
**cache'owana w słowniku** po pierwszym wyliczeniu i invalidowana dopiero po połączeniu
klastra `ci` z innym.

## 3.5. Moduł `merger.py` — §4.4 Phase II aglomeracja

**Plik:** `src/pychameleon/merger.py`
**Odpowiedzialność:** iteracyjne łączenie sub-klastrów o najwyższym `merge_score`.

**Sygnatura funkcji (z `src/pychameleon/merger.py`):**

```python
def merge_to_k_clusters(
    adjacency: AdjacencyList,
    edge_weights: EdgeWeights,
    initial_labels: Labels,
    target_k: int,
    alpha: float,
) -> Labels:
    """Phase II — agglomeratively merge sub-clusters until ``target_k`` remain.

    Uses a max-priority-queue keyed on merge_score = RI · RC**alpha (eq. 4).
    Lazy invalidation: stale heap entries are skipped, not removed eagerly.
    """
```

**Mapowanie krok paperu → implementacja:**

| Krok paperu §4.4 Phase II                                        | Implementacja                                              |
|------------------------------------------------------------------|-----------------------------------------------------------|
| "compute initial scores for all adjacent pairs"                  | `for (ci, cj) in adjacent_pairs(labels): heappush(...)`   |
| "select pair with highest score"                                 | `(-score, version, ci, cj) = heappop(queue)`              |
| "if score < threshold, stop"                                     | `if score <= 0: break`                                    |
| "merge pair, relabel"                                            | `labels[labels == ci] = cj` (kanonizacja na mniejszy id)  |
| "update scores for affected neighbors"                           | `for nb in neighbors_of_merged: heappush(...)`            |
| "loop until target k reached"                                    | `while n_active_clusters > target_k`                       |

**Lazy invalidation (kluczowy detal):**
Każdy klaster ma `version` (counter inkrementowany przy każdym scaleniu). Wpis w heap
przechowuje `(score, version_i, version_j, ci, cj)`. Po `heappop`, jeśli `version_i`
lub `version_j` nie odpowiada aktualnej wersji, wpis jest pomijany. Eliminuje to
konieczność O(n²) rebuild kolejki po każdym scaleniu.

## 3.6. Moduł `chameleon.py` — klasa estymatora

**Plik:** `src/pychameleon/chameleon.py`
**Odpowiedzialność:** publiczny interfejs zgodny ze scikit-learn.

**Klasa `Chameleon`:**

```python
class Chameleon(ClusterMixin, BaseEstimator):
    """CHAMELEON hierarchical clustering using dynamic modeling."""

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "n_clusters":       [Interval(Integral, 1, None, closed="left")],
        "k_nn":             [Interval(Integral, 2, None, closed="left")],
        "min_cluster_size": [
            Interval(Real, 0.0, 1.0, closed="neither"),
            Interval(Integral, 2, None, closed="left"),
        ],
        "alpha":            [Interval(Real, 0.0, None, closed="neither")],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        k_nn: int = 10,
        min_cluster_size: int | float = 0.025,
        alpha: float = 2.0,
    ) -> None: ...

    def fit(self, X: ArrayLike, y: None = None) -> Chameleon: ...
    def fit_predict(self, X: ArrayLike, y: None = None) -> Labels: ...
```

**Pipeline metody `fit`:**

```python
def fit(self, X, y=None):
    self._validate_params()                                       # sklearn API
    X_arr = self._validate_data(X, accept_sparse=False, dtype=np.float64)
    n_samples = X_arr.shape[0]
    min_size = self._resolve_min_cluster_size(n_samples)

    # Faza 0
    adjacency, edge_weights = graph.knn_graph(X_arr, self.k_nn)

    # Faza I
    initial_labels = partition.initial_subclusters(
        adjacency, edge_weights, min_size,
    )

    # Faza II
    self.labels_ = merger.merge_to_k_clusters(
        adjacency, edge_weights, initial_labels,
        target_k=self.n_clusters,
        alpha=self.alpha,
    )
    self.n_clusters_ = len(np.unique(self.labels_))
    return self
```

**Walidacja parametrów (sklearn `_parameter_constraints`):**
sklearn-ujący framework `_parameter_constraints` automatycznie waliduje typy i przedziały
przy każdym wywołaniu `fit()`. Typy są wykrywane przez `Interval(Integral|Real, …)`.
Walidacja jest **deklaratywna** — nie musimy pisać własnego kodu sprawdzającego.

**Zgodność z sklearn API:**

- dziedziczy `ClusterMixin` (zapewnia `fit_predict`) i `BaseEstimator` (zapewnia
  `get_params`, `set_params`, `clone`),
- `labels_` jako trailing-underscore atrybut (konwencja sklearn dla wyników fit),
- `n_features_in_` ustawiane automatycznie przez `_validate_data`.

Zgodność jest weryfikowana testem `tests/test_sklearn_api.py::test_check_estimator`
(odblokowanym po pełnej implementacji `fit`).

## 3.7. Moduł pomocniczy `_types.py`

**Plik:** `src/pychameleon/_types.py`
**Odpowiedzialność:** scentralizowane aliasy typów używane w całym pakiecie.

```python
FloatMatrix:   TypeAlias = NDArray[np.float64]            # shape (n, d)
Labels:        TypeAlias = NDArray[np.int64]               # shape (n,)
Weights:       TypeAlias = NDArray[np.float64]             # shape (n,) lub (m,)
AdjacencyList: TypeAlias = list[NDArray[np.int64]]         # adjacency[i] = sąsiedzi i
EdgeWeights:   TypeAlias = list[NDArray[np.float64]]       # weights[i][j] = waga
```

Zastosowanie aliasów daje nam:

- czytelność — `AdjacencyList` zamiast `list[NDArray[np.int64]]` w sygnaturach,
- spójność — jeden punkt zmiany jeśli zdecydujemy się na inny format reprezentacji
  (np. `scipy.sparse.csr_matrix` w przyszłości).

Szczegóły rozważań nad reprezentacją grafu znajdują się w rozdziale 4.

## 3.8. Punkt wejścia pakietu — `__init__.py`

**Plik:** `src/pychameleon/__init__.py`

```python
"""CHAMELEON hierarchical clustering."""
from pychameleon.chameleon import Chameleon

__all__ = ["Chameleon"]
__version__ = "0.1.0"
```

Publiczne API to **jedna klasa** `Chameleon`. Funkcje pomocnicze (`knn_graph`,
`initial_subclusters`, …) są dostępne przez `pychameleon.graph` itd., ale nie są
re-eksportowane z poziomu pakietu — celowo, by nie zaśmiecać przestrzeni nazw.

## 3.9. Decyzje projektowe

| Decyzja                     | Wybór                                | Uzasadnienie                                                         |
|-----------------------------|--------------------------------------|----------------------------------------------------------------------|
| Build backend Pythona       | `hatchling` + `pyproject.toml`       | minimalny, nowoczesny, brak `setup.py`                               |
| Layout pakietu              | `src-layout`                         | zapobiega bugom import-order w testach (PEP 660)                     |
| Type hints                  | strict, sprawdzane przez `mypy --strict` | zgodność z sklearn-style libs; pomoc w refactoringu                |
| Linter / formatter          | `ruff`                               | 10× szybszy od `pylint+flake8+isort`                                 |
| Testowanie                  | `pytest` + `pytest-cov`              | standard de facto                                                    |
| Walidacja parametrów        | `sklearn._parameter_constraints`     | deklaratywna, spójna z sklearn API                                   |
| Wersja Pythona              | 3.11+                                | nowoczesne typing (`X | Y`, `TypeAlias`); zgodne z venv referencyjnym |
| API wejścia                 | `X` (n × d, float64)                 | zgodne ze sklearn; matrix podobieństwa w przyszłej wersji            |
| Wersja algorytmu            | original CHAMELEON 1999              | dobrze udokumentowane; Chameleon2/Chameleon2++ jako roadmap          |

Wybór `sklearn`-compatible API od pierwszego commita jest świadomą decyzją
projektową motywowaną długoterminowym celem publikacji jako biblioteka
(referencyjny przykład: HDBSCAN, który trafił z `scikit-learn-contrib` do `sklearn` core
w 2023 roku).
