# 7. Zbiory danych

W ramach projektu używane są **trzy zbiory referencyjne** (te same, które są
dostarczone z implementacją Moonpuck) oraz **syntetyczne fixture'y** generowane
deterministycznie w testach jednostkowych.

## 7.1. Zbiory referencyjne (vendored w `tests/data/`)

### 7.1.1. Aggregation

| Atrybut          | Wartość                                                     |
|------------------|-------------------------------------------------------------|
| Plik             | `tests/data/Aggregation.csv`                                |
| Liczba punktów   | 788                                                         |
| Wymiary          | 2 (x, y)                                                    |
| Format pliku     | CSV, separator: spacja                                      |
| Ground-truth     | TAK — 3. kolumna zawiera label klastra                      |
| Klastrów         | 7                                                           |
| Charakterystyka  | 7 klastrów o **różnych rozmiarach i kształtach**, niektóre "stykające się" |
| Pochodzenie      | Gionis A., Mannila H., Tsaparas P. *Clustering aggregation*. ACM TKDD 2007 |

**Charakterystyka algorytmiczna:** klasyczny benchmark dla algorytmów hierarchicznych —
zbiór zawiera wszystkie typowe trudności (różne gęstości, łączące się klastry,
"łańcuchy"). CHAMELEON (paper Karypis 1999, fig. 2c) pokonuje na tym zbiorze
single-linkage i CURE.

**Zalecane parametry** (z benchmarków Moonpuck):

```python
Chameleon(n_clusters=7, k_nn=20, min_cluster_size=40, alpha=2.0)
```

### 7.1.2. Smileface

| Atrybut          | Wartość                                                     |
|------------------|-------------------------------------------------------------|
| Plik             | `tests/data/smileface.csv`                                  |
| Liczba punktów   | 644                                                         |
| Wymiary          | 2 (x, y)                                                    |
| Format pliku     | CSV, separator: przecinek                                   |
| Ground-truth     | TAK — 3. kolumna                                            |
| Klastrów         | 4 (oczy + nos + uśmiech)                                    |
| Charakterystyka  | klastry o **niewypukłym kształcie** (uśmiech to łuk)        |
| Pochodzenie      | repozytorium Moonpuck/chameleon_cluster                     |

**Charakterystyka algorytmiczna:** test dla algorytmów wykrywających klastry
niewypukłe. k-means radykalnie zawodzi (centroidy nie reprezentują łuku
uśmiechu); CHAMELEON poprawnie wykrywa wszystkie 4 grupy.

**Zalecane parametry:**

```python
Chameleon(n_clusters=4, k_nn=10, min_cluster_size=20, alpha=2.0)
```

### 7.1.3. t4_8k

| Atrybut          | Wartość                                                     |
|------------------|-------------------------------------------------------------|
| Plik             | `tests/data/t4_8k.csv`                                      |
| Liczba punktów   | 8000                                                        |
| Wymiary          | 2 (x, y)                                                    |
| Format pliku     | CSV                                                          |
| Ground-truth     | NIE (tylko coordinates)                                     |
| Klastrów         | 6 (wzrokowo)                                                |
| Charakterystyka  | klastry **z dziurami i szumem tła**; jest to dataset DS3 z paperu Karypis 1999 |
| Pochodzenie      | Karypis 1999, fig. 4 (zbiór DS3); też Chameleon Moonpuck    |

**Charakterystyka algorytmiczna:** najtrudniejszy zbiór benchmarkowy dla algorytmów
hierarchicznych — zawiera klastry "z dziurkami" (np. spirala wewnętrzna), klastry
podłużne, oraz **rozproszone punkty szumu** w tle. CHAMELEON w paperu jest
porównywany na tym zbiorze z DBSCAN, CURE, ROCK i k-means — wszystkie pozostałe
zawodzą; CHAMELEON daje wzrokowo poprawne wyniki.

**Zalecane parametry** (z paperu i Moonpuck):

```python
Chameleon(n_clusters=6, k_nn=20, min_cluster_size=40, alpha=2.0)
```

**Uwaga o wydajności:** ze względu na rozmiar (8000 pkt) ten zbiór jest istotny
głównie dla testów **skalowalności**. Implementacja Moonpuck zajmuje 11 minut
(rozdz. 5); spodziewane przyspieszenie naszej implementacji to 50–100×
(do ~10 sekund).

## 7.2. Zbiory syntetyczne (fixtures)

### 7.2.1. `small_blobs` — fixture w `tests/conftest.py`

Deterministyczny zbiór 60 punktów w 3 wyraźnych klastrach. Używany w testach
jednostkowych jako "minimalny działający przykład".

```python
@pytest.fixture
def small_blobs() -> np.ndarray:
    """60 punktów (3 klastry × 20), Gaussian noise σ=0.3, deterministic."""
    rng = np.random.default_rng(42)
    cluster_centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    points = np.vstack([c + 0.3 * rng.standard_normal((20, 2)) for c in cluster_centers])
    return points
```

**Cechy:**

- 60 punktów w 3 klastrach po 20,
- centra `(0, 0)`, `(5, 5)`, `(0, 5)`,
- szum Gaussowski σ = 0.3,
- ziarno RNG = 42 (deterministyczne).

**Cel:** testy weryfikujące **podstawowe oczekiwane zachowanie** (czy algorytm
w ogóle działa) — używany w `test_aggregation_recovers_7_clusters` itp.

### 7.2.2. `make_blobs` z sklearn (testy skalowalności)

Dla testów skalowalności (rozdz. 6.8) używamy generatora z sklearn:

```python
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=n,
    centers=5,
    cluster_std=1.0,
    random_state=42,
)
```

To pozwala na łatwe generowanie zbiorów o `n ∈ [100, 50_000]`.

## 7.3. Podsumowanie

| Zbiór         | n     | dim | Klastrów | Ground truth | Cel testowania                              |
|---------------|-------|-----|----------|--------------|---------------------------------------------|
| Aggregation   | 788   | 2   | 7        | tak          | klastry różnych rozmiarów; benchmark hierarchical |
| smileface     | 644   | 2   | 4        | tak          | klastry niewypukłe (łuk uśmiechu)           |
| t4_8k         | 8000  | 2   | 6        | nie (visual) | klastry z szumem; benchmark skalowalności   |
| `small_blobs` | 60    | 2   | 3        | n/a (synth)  | testy jednostkowe — sanity check            |
| `make_blobs`  | var   | 2   | 5        | tak          | testy skalowalności n vs runtime            |
