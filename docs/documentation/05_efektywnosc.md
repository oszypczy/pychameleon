# 5. Ocena efektywności

## 5.1. Złożoność teoretyczna (paper §4.5)

Niech `n` — liczba punktów, `m` — sub-klastrów po Fazie I (typowo `m ≈ n/20`),
`k` — parametr `k_nn`.

| Faza                              | Nasza impl.       | Moonpuck (referencyjna)      |
|-----------------------------------|-------------------|------------------------------|
| **Faza 0** k-NN graph             | `O(n log n)` KDTree | `O(n²)` naiwna macierz dist. |
| **Faza I** rekurencyjna bisekcja  | `O(n log² n)`     | `O(n log² n)`                |
| **Faza II** aglomeracja           | `O(m² log m)` lazy| `O(m³)` rebuild po scaleniu  |
| **Pamięć**                        | `O(nd + nk + m²)` | `O(n²)` (full distance mat.) |

**Sumarycznie:** dominuje `O(n log² n)`. Dla `n = 10⁴`: ok. `1.7 × 10⁵` operacji
vs `10⁸` przy `O(n²)` — różnica 3 rzędów wielkości.

## 5.2. Empiryczne benchmarki implementacji referencyjnej

Przygotowując projekt uruchomiłem implementację referencyjną Moonpuck
([link](https://github.com/Moonpuck/chameleon_cluster)) na 3 zbiorach. Wyniki w `benchmarks/reference_moonpuck/`:

| Zbiór       | n    | k | k_nn | m  | α   | Runtime (s) | Bottleneck                |
|-------------|------|---|------|-----|-----|-------------|---------------------------|
| aggregation | 788  | 7 | 20   | 40  | 2.0 | 9.95        | balanced                  |
| smileface   | 644  | 4 | 10   | 20  | 2.0 | 2.08        | balanced                  |
| t4_8k       | 8000 | 6 | 20   | 40  | 2.0 | **674.47**  | naiwny `O(n²)` k-NN       |

Środowisko: MacBook Pro Apple Silicon, Python 3.11, `pymetis>=2023.1`.

## 5.3. Predykcja przyspieszenia

Główne planowane optymalizacje:

1. **`scipy.spatial.KDTree`** zamiast naiwnego k-NN — speedup `O(n²) → O(n log n)`,
   asymptotycznie ≈ 47× dla `n = 8000`.
2. **Cache `_internal_interconnectivity`** — eliminacja powtarzanych bisekcji;
   Moonpuck recompute'uje przy każdym dostępie.
3. **Lazy invalidation kolejki** — zamiast `O(m²)` rebuildu po każdym scaleniu.
4. **numpy ufuncs** zamiast pętli Pythona w `metrics.py` — stała ~10–50×.

**Spodziewany całkowity speedup vs Moonpuck:** **50–100×**

## 5.4. Ograniczenia

- **KDTree** efektywne tylko dla niskich wymiarów (`d ≤ 10`); dla `d > 10` koszt
  rośnie do `O(n)`/zapytanie. Roadmap v0.3: zastąpienie przez Annoy/HNSW.
- **pymetis nie jest deterministyczny** (randomized refinement); `Chameleon().fit(X).labels_`
  może się minimalnie różnić między uruchomieniami. Mitigacja: w v0.2 parametr `random_state`.
- **Brak obsługi szumu** (zgodne z paperu Karypis 1999 — wszystkie punkty są przypisywane).
  Roadmap v0.2: parametr `min_samples_per_cluster` (analogia do HDBSCAN).
