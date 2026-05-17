# 5. Charakterystyka zbiorów danych

Eksperymenty z rozdziału 6 wykorzystują **siedem zbiorów rzeczywistych** (literatura) oraz **dwa generowane syntetycznie**. Wszystkie zbiory rzeczywiste są _vendored_ w repozytorium pod `tests/data/` — eksperymenty można reprodukować bez dostępu do sieci.

## 5.1. Zbiory rzeczywiste z literatury

| Zbiór           | n     | dim | Klastrów (GT) | Ground truth | Charakterystyka                                  | Pochodzenie           |
|-----------------|-------|-----|---------------|--------------|--------------------------------------------------|------------------------|
| **Aggregation** | 788   | 2   | 7             | tak (UEF)    | klastry różnych rozmiarów, łączące się           | Gionis et al. 2007    |
| **smileface**   | 644   | 2   | 4             | wizualnie    | klastry niewypukłe — łuk uśmiechu                | repo Moonpuck         |
| **t4_8k**       | 8000  | 2   | 6             | wizualnie    | klastry z dziurami i szumem tła (DS3 z paperu)   | Karypis 1999          |
| **DS1**         | 8000  | 2   | 6             | tak          | wydłużone klastry, rozdzielone                   | Karypis CLUTO (`t5.8k`) |
| **DS3**         | 8000  | 2   | 6             | tak          | klastry z ~10% szumu (label `-1`)                | Karypis CLUTO (`t4.8k`) |
| **DS4**         | 10000 | 2   | 9             | tak          | 9 klastrów, częściowo nakładających się          | Karypis CLUTO (`t7.10k`) |
| **DS5**         | 8000  | 2   | 8             | tak          | klastry różnych gęstości                         | Karypis CLUTO (`t8.8k`) |

**Aggregation** posiada ground truth pobrany z [UEF Clustering Datasets](https://cs.joensuu.fi/sipu/datasets/) i dopasowany 1:1 do punktów w `tests/data/Aggregation.csv`. Wykorzystywany jako zbiór benchmarkowy w sekcji 6.2 (porównanie z Moonpuck).

**Datasety DSx** pochodzą z oryginalnego repozytorium CLUTO Karypisa — zawierają etykiety klas, w tym `-1` oznaczające szum tła (ignorowany przy obliczaniu ARI). Pełna dyskusja w rozdziale 6.1 (DS3 jako granica algorytmu bez noise-handlingu).

**t4_8k vs DS3** — to ten sam point cloud (8000 punktów t4.8k z paperu Karypisa), ale w dwóch wersjach: `t4_8k` bez etykiet (do porównania z Moonpuck, sekcja 6.2), `DS3` z etykietami CLUTO (do oceny jakości vs GT, sekcja 6.1).

## 5.2. Zbiory syntetyczne

| Zbiór          | n             | dim   | Klastrów | Cel                                       |
|----------------|---------------|-------|----------|-------------------------------------------|
| `small_blobs`  | 60            | 2     | 3        | Deterministyczna fixture testów jednostkowych (`seed=42`)  |
| `make_blobs`   | var           | 2–50  | 5        | Testy skalowalności (sekcja 6.4)          |

`small_blobs` — 3 izotropowe gaussowskie blobs (`σ = 0.3`) wygenerowane w `tests/conftest.py`; służy jako sanity-check w testach jednostkowych.

`make_blobs` — generowane na bieżąco przez `sklearn.datasets.make_blobs(n_samples=n, centers=5, cluster_std=1.0, random_state=seed)`. Używane do badania skalowalności względem **liczby punktów** (`n ∈ {500, 1000, 2000, 5000, 10000, 20000, 50000}`, d = 2) oraz **wymiarowości** (`d ∈ {2, 5, 10, 20, 50}`, n = 2000). Każda konfiguracja powtarzana 3× z różnymi `random_state` dla redukcji wariancji.

## 5.3. Format wejściowy

Wszystkie zbiory są ładowane jako `np.ndarray` shape `(n, d)`, dtype `float64`. Loader rejestrowany w `pychameleon._datasets.ALL_DATASETS` udostępnia jednolite API:

```python
from pychameleon._datasets import load

ds = load("aggregation")
print(ds.X.shape, ds.y.shape if ds.y is not None else None)
# (788, 2) (788,)
```

Datasety bez ground truth (`smileface`, `t4_8k`) zwracają `ds.y = None`; metryki external (ARI, NMI) są w takich przypadkach raportowane jako `NaN`.
