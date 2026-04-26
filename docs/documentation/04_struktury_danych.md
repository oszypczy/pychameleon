# 4. Struktury danych

> **Wymaganie #2 z PDF prowadzącego (str. 1):**
> *"strukturach danych wykorzystywanych do przechowywania danych źródłowych
> oraz struktur wykorzystywanych w implementacji algorytmu"*

Niniejszy rozdział opisuje wszystkie struktury danych używane w implementacji
CHAMELEON-a — zarówno do przechowywania **wejścia** (zbioru punktów), jak i
**wewnętrznych struktur algorytmu** (graf k-NN, kolejka priorytetowa, etykiety klastrów).

## 4.1. Dane źródłowe (wejście)

### 4.1.1. Format wejścia

Algorytm akceptuje wejście jako **dwuwymiarowy `numpy.ndarray`** o kształcie
`(n_samples, n_features)` typu `float64`:

| Atrybut          | Wartość                            |
|------------------|-------------------------------------|
| Klasa            | `numpy.ndarray`                     |
| Kształt          | `(n_samples, n_features)`          |
| Dtype            | `np.float64`                        |
| Ograniczenia     | bez `NaN`, bez `inf`                |
| Tryby            | gęsty (sparse niewspierane w v0.1) |

**Alias typu** w `src/pychameleon/_types.py`:

```python
from numpy.typing import NDArray
import numpy as np
from typing import TypeAlias

FloatMatrix: TypeAlias = NDArray[np.float64]
```

### 4.1.2. Walidacja wejścia

Walidacja jest delegowana do biblioteki sklearn za pomocą metody dziedziczonej
`_validate_data`:

```python
X_arr = self._validate_data(
    X,
    accept_sparse=False,
    dtype=np.float64,
)
```

Funkcja `_validate_data` automatycznie:

- konwertuje listę Pythona, `pandas.DataFrame` lub `np.float32` na `np.float64`,
- odrzuca wejście z `NaN`/`inf` (`force_all_finite=True` domyślnie),
- ustawia atrybut `self.n_features_in_`,
- ustawia `feature_names_in_` jeśli wejście to `DataFrame`.

To rozwiązanie eliminuje ryzyko niezgodności z konwencjami sklearn API
(istotne dla pipeline'ów typu `Pipeline([scaler, Chameleon()])`).

### 4.1.3. Wejścia testowe — loadery

Datasety używane w testach przechowywane są jako pliki CSV w `tests/data/`,
a fixture w `tests/conftest.py` ładuje je do `numpy.ndarray`:

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def aggregation_xy() -> np.ndarray:
    """Aggregation dataset (Gionis et al. 2007), 788 points, 7 ground-truth clusters."""
    return np.loadtxt(DATA_DIR / "Aggregation.csv", delimiter=" ", dtype=np.float64)

@pytest.fixture(scope="session")
def smileface_xy() -> np.ndarray:
    return np.loadtxt(DATA_DIR / "smileface.csv", delimiter=",", dtype=np.float64)

@pytest.fixture
def small_blobs() -> np.ndarray:
    """Deterministic 60-point synthetic dataset, 3 obvious clusters."""
    rng = np.random.default_rng(42)
    cluster_centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    return np.vstack([c + 0.3 * rng.standard_normal((20, 2)) for c in cluster_centers])
```

Charakterystyka zbiorów testowych: rozdz. 7.

## 4.2. Struktury wewnętrzne algorytmu

### 4.2.1. Graf k-najbliższych sąsiadów — adjacency list

Po fazie 0 dane są reprezentowane jako para list:

```python
AdjacencyList: TypeAlias = list[NDArray[np.int64]]
EdgeWeights:   TypeAlias = list[NDArray[np.float64]]
```

**Semantyka:**

- `adjacency[i]` — `np.ndarray` indeksów sąsiadów wierzchołka `i`,
- `edge_weights[i][j]` — waga krawędzi `(i, adjacency[i][j])`.

Ze względu na **symetrię** grafu k-NN, jeśli `j ∈ adjacency[i]`, to również
`i ∈ adjacency[j]` (ze tą samą wagą — `1/d(xᵢ, xⱼ)`).

**Przykład — graf trójkąta `n=3`, `k=2`:**

```python
adjacency = [
    np.array([1, 2]),  # wierzchołek 0 sąsiaduje z 1, 2
    np.array([0, 2]),
    np.array([0, 1]),
]
edge_weights = [
    np.array([0.5, 0.4]),
    np.array([0.5, 0.7]),
    np.array([0.4, 0.7]),
]
```

**Złożoność pamięciowa:** `O(n · k_nn)` — każdy wierzchołek ma stałą liczbę krawędzi.

### 4.2.2. Format CSR dla pymetis

`pymetis.part_graph(nparts, xadj=…, adjncy=…, eweights=…)` oczekuje formatu
**Compressed Sparse Row** (CSR):

| Tablica   | Typ                | Semantyka                                                          |
|-----------|--------------------|--------------------------------------------------------------------|
| `xadj`    | `list[int]`        | `xadj[i+1] - xadj[i]` = stopień wierzchołka `i`                    |
| `adjncy`  | `list[int]`        | spłaszczona konkatenacja list sąsiedztwa                           |
| `eweights`| `list[int]`        | wagi krawędzi (wymagane przez METIS jako liczby całkowite)         |

**Konwersja `AdjacencyList` → CSR** (na poziomie `partition.py` i `metrics.py`):

```python
def _to_csr(adjacency, edge_weights):
    xadj = [0]
    adjncy: list[int] = []
    eweights: list[int] = []
    for nbrs, weights in zip(adjacency, edge_weights):
        adjncy.extend(int(j) for j in nbrs)
        # METIS wymaga int wag - skalujemy float→int
        eweights.extend(int(w * SCALE) for w in weights)
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights
```

Skalowanie `SCALE = 10000` zachowuje 4 miejsca po przecinku.

### 4.2.3. Etykiety klastrów — `Labels`

```python
Labels: TypeAlias = NDArray[np.int64]
```

Wektor o kształcie `(n_samples,)` przechowuje **identyfikator klastra** dla każdego
punktu. Identyfikatory są dodatnie, zwarte w `[0, n_clusters_)`. Po fazie I:
`m` różnych etykiet (`m` — liczba sub-klastrów). Po fazie II: `n_clusters` różnych
etykiet (lub mniej, jeśli zabrakło merge'ów z dodatnim score).

### 4.2.4. Kolejka priorytetowa — `heapq`

Faza II używa **min-heap** Pythona z modułu `heapq`:

```python
import heapq

# Element heap'a:
# (-score, version_i, version_j, cluster_i, cluster_j)
queue: list[tuple[float, int, int, int, int]] = []

# Wstawianie:
heapq.heappush(queue, (-score, version_i, version_j, ci, cj))

# Pobieranie pary o najwyższym score:
neg_score, vi, vj, ci, cj = heapq.heappop(queue)
score = -neg_score  # max-heap przez negację
```

**Lazy invalidation:** zamiast usuwać przestarzałe wpisy z heap'a (co wymagałoby `O(n)`),
przechowujemy `version` każdego klastra. Po `heappop` sprawdzamy:

```python
if vi != versions[ci] or vj != versions[cj]:
    continue   # przestarzały wpis, pomiń
```

To daje **amortyzowany koszt `O(log m)`** na operację, vs `O(m²)` przy naiwnym rebuildzie.

### 4.2.5. Cache wewnętrznej interconnectivity

Wartość `|EC_Cᵢ|` (interconnectivity wewnętrzna klastra) wymaga bisekcji pod-grafu
przez pymetis. Jest to **drogie** (`O(|E_Cᵢ| + |V_Cᵢ| log |V_Cᵢ|)`) i niezmienne
dopóki klaster `Cᵢ` nie zostanie scalony.

Cache:

```python
internal_ic_cache: dict[int, float] = {}

def get_internal_interconnectivity(ci, …):
    if ci in internal_ic_cache:
        return internal_ic_cache[ci]
    value = _compute_via_pymetis(ci, …)
    internal_ic_cache[ci] = value
    return value

# Po scaleniu klastra:
del internal_ic_cache[ci]   # invalidate
del internal_ic_cache[cj]
```

To eliminuje powtarzane bisekcje tych samych klastrów — istotna optymalizacja vs
implementacja referencyjna Moonpuck (która recompute'uje przy każdym dostępie).

## 4.3. Decyzje projektowe — uzasadnienie wyborów

### 4.3.1. Dlaczego adjacency list, a nie `scipy.sparse.csr_matrix`?

| Cecha                          | Adjacency list (`list[ndarray]`) | `scipy.sparse.csr_matrix` |
|--------------------------------|----------------------------------|---------------------------|
| Bezpośrednia kompat. z pymetis | tak (potrzebna konwersja CSR int) | tak (potrzebna konwersja int) |
| Łatwość edycji (dodaj/usuń)    | łatwa (slicing list)             | trudna (immutable struct) |
| Pamięć dla `O(nk)` krawędzi    | `O(nk)`                          | `O(nk)`                   |
| Operacje wektorowe (numpy)     | tak na pojedynczym wierzchołku   | tak na całym grafie       |
| Czytelność dla developera      | wysoka                           | średnia                   |

Adjacency list jest preferowana ze względu na **łatwość iteracji wierzchołek-po-wierzchołku**,
co dominuje w Fazie II (przeglądamy sąsiadów scalanego klastra).

### 4.3.2. Dlaczego `heapq`, a nie `sortedcontainers.SortedList`?

| Operacja                         | `heapq`        | `SortedList`     |
|----------------------------------|----------------|------------------|
| Push                             | `O(log n)`     | `O(log n)`       |
| Pop min                          | `O(log n)`     | `O(log n)`       |
| Update klucza istniejącego elem. | brak (lazy inv.)| `O(log n)`     |
| Standard library?                | tak            | nie (zewnętrzna) |
| Narzut overhead                  | minimalny      | większy          |

Lazy invalidation eliminuje potrzebę aktualizacji elementów, więc `heapq` wystarcza.
Brak zewnętrznej zależności jest dodatkowym plusem.

### 4.3.3. Dlaczego `numpy.ndarray` dla etykiet, a nie `list[int]`?

| Cecha                            | `np.ndarray`   | `list[int]` |
|----------------------------------|----------------|-------------|
| Operacje wektorowe (`labels == c`) | tak (szybkie) | nie (pętla) |
| Maska boolean                    | natywna        | brak        |
| Pamięć dla `n=10^6`              | 8 MB           | ~28 MB      |
| Kompat. z sklearn `labels_`      | tak (konwencja)| nie         |

`numpy.ndarray` jest **konwencją sklearn** dla atrybutu `labels_` i jest znacząco
szybszy w operacjach typu "wybierz wszystkie punkty klastra c".

## 4.4. Podsumowanie — tabela wszystkich aliasów typów

```python
# src/pychameleon/_types.py
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

FloatMatrix:   TypeAlias = NDArray[np.float64]      # (n_samples, n_features)
Labels:        TypeAlias = NDArray[np.int64]        # (n_samples,)
Weights:       TypeAlias = NDArray[np.float64]      # tablica wag
AdjacencyList: TypeAlias = list[NDArray[np.int64]]  # adjacency[i] = sąsiedzi i
EdgeWeights:   TypeAlias = list[NDArray[np.float64]] # weights[i][j] = waga (i, adj[i][j])
```

Każdy moduł importuje aliasy z `_types.py` — pojedynczy punkt zmiany dla całego pakietu.
