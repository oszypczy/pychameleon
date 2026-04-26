# 7. Zbiory danych testowych

| Zbiór         | n     | dim | Klastrów | Ground truth | Charakterystyka                                |
|---------------|-------|-----|----------|--------------|------------------------------------------------|
| Aggregation   | 788   | 2   | 7        | tak          | klastry różnych rozmiarów, łączące się         |
| smileface     | 644   | 2   | 4        | tak          | klastry niewypukłe (łuk uśmiechu)              |
| t4_8k         | 8000  | 2   | 6        | wizualnie    | klastry z dziurami i szumem tła (DS3 z paperu) |
| `small_blobs` | 60    | 2   | 3        | n/a (synth)  | deterministyczna fixture, sanity check         |
| `make_blobs`  | var   | 2   | 5        | tak          | sklearn-generated, testy skalowalności         |

Zbiory referencyjne (Aggregation, smileface, t4_8k) są **vendored** w `tests/data/`.
Pochodzenie: Aggregation — Gionis et al., *ACM TKDD* 2007; smileface — repozytorium
Moonpuck; t4_8k — paper Karypis 1999 (zbiór DS3, fig. 4). `small_blobs` jest
deterministyczną fixture w `tests/conftest.py` (3 centra, σ=0.3, seed=42). Dla testów
skalowalności używamy `sklearn.datasets.make_blobs(n_samples=n, centers=5, random_state=42)`
z `n ∈ {100, 500, 1000, 5000, 10000, 50000}`.
