# 4. Struktury danych

## 4.1. Dane źródłowe (wejście)

| Atrybut          | Wartość                                            |
|------------------|----------------------------------------------------|
| Klasa            | `numpy.ndarray`                                    |
| Kształt          | `(n_samples, n_features)`                          |
| Dtype            | `np.float64`                                       |
| Walidacja        | `sklearn.utils._validate_data(accept_sparse=False)` |
| Alias typu       | `FloatMatrix = NDArray[np.float64]`                 |

Walidacja delegowana do sklearn — automatyczna konwersja list/DataFrame, odrzucanie
NaN/inf, ustawianie `n_features_in_` i `feature_names_in_`. Zbiory testowe wczytywane
z plików CSV w `tests/data/` przez `np.loadtxt` (fixtures w `tests/conftest.py`).

## 4.2. Wewnętrzne struktury algorytmu

| Struktura                       | Reprezentacja                                | Po co tak                                  |
|---------------------------------|----------------------------------------------|--------------------------------------------|
| Graf k-NN — sąsiedzi            | `AdjacencyList = list[NDArray[int64]]`       | direct pymetis CSR compatibility           |
| Graf k-NN — wagi                | `EdgeWeights = list[NDArray[float64]]`       | równoległa indeksacja z adjacency          |
| Etykiety klastrów               | `Labels = NDArray[int64]` shape `(n,)`        | maski boolean, sklearn convention          |
| Kolejka priorytetowa (Faza II)  | `heapq` z `(-score, ver_i, ver_j, ci, cj)`    | `O(log m)` push/pop + lazy invalidation     |
| Wersje klastrów                 | `dict[int, int]`                              | invalidacja przestarzałych wpisów heap'a   |
| Cache `|EC_Cᵢ|`               | `dict[int, float]`                            | unikamy powtarzanych bisekcji pymetis      |

`adjacency[i]` to tablica indeksów sąsiadów wierzchołka `i`; `edge_weights[i][j]`
to waga krawędzi `(i, adjacency[i][j])`. Reprezentacja symetryczna: jeśli
`j ∈ adjacency[i]`, to również `i ∈ adjacency[j]` z tą samą wagą.

## 4.3. Format CSR dla pymetis

`pymetis.part_graph(nparts, xadj, adjncy, eweights)` wymaga formatu Compressed Sparse
Row z **całkowitymi** wagami. Konwersja `AdjacencyList → CSR` dokonywana w wywołaniach
`partition.py` i `metrics.py`; wagi `float` skalowane przez `SCALE = 10000` (4 miejsca po przecinku).

## 4.4. Uzasadnienie wyborów

**Adjacency list vs `scipy.sparse.csr_matrix`:** wybrana adjacency list ze względu
na **łatwiejszą iterację per-wierzchołek** (dominuje w Fazie II), a CSR i tak musimy
generować dla pymetis. **`heapq` vs `SortedList`:** `heapq` ze stdlib + lazy invalidation
daje amortyzowany `O(log m)` bez zewnętrznej zależności. **`numpy.ndarray` vs `list[int]`
dla labels:** wektorowe maski (`labels == c`) są O(n) zamiast O(n) w pętli, plus konwencja sklearn.
