# 6. Plan testów

Strategia walidacji jest 5-warstwowa, realizowana w `tests/` przy użyciu `pytest`,
`pytest-cov`, `numpy.testing`, `sklearn.utils.estimator_checks` oraz
`sklearn.metrics.adjusted_rand_score`.

| Warstwa             | Cel                                              | Liczba / próg                        | Status (26.04)              |
|---------------------|--------------------------------------------------|--------------------------------------|------------------------------|
| Jednostkowe         | Każdy moduł osobno; hand-computed wartości       | ~25 (5 modułów × ~5)                 | szkielet, treść w Etap 2     |
| Integracyjne (e2e)  | `fit()` na 3 datasetach                          | 7 testów; ARI > 0.85 vs ground truth | 12 testów skeleton pass      |
| sklearn-compat      | `check_estimator(Chameleon())`                   | ~30 wewnętrznych                     | po implementacji `fit()`     |
| Porównawcze         | Wyniki vs Moonpuck na tych samych parametrach    | 3 datasety × ARI > 0.9               | Etap 2                       |
| Parametryczne       | Sweep `k_nn`, `α`, `min_cluster_size`            | 3 sweep'y × ~5 wartości              | Etap 2                       |
| Skalowalności       | `n` vs runtime; slope log-log ≈ 1.1              | 6 punktów `n ∈ [100, 50000]`         | Etap 2                       |

**Pokrycie kodu (target):** `pytest --cov=pychameleon` ≥ **90% linii**.

## 6.1. Testy jednostkowe — kluczowe asercje

- **`test_graph.py`:** symetria grafu (`j ∈ adj[i] <=> i ∈ adj[j]`), brak self-loops,
  `weight ≈ 1/distance`, dla 3 odseparowanych blobów krawędzie tylko wewnątrz blobów.
- **`test_partition.py`:** wszystkie punkty otrzymują `label != -1`, każdy
  sub-klaster spełnia `|C| ≥ min_cluster_size`, liczba sub-klastrów `m ≈ n/min_size`.
- **`test_metrics.py`:** RI/RC/score liczone dla małych grafów (3–5 wierzchołków)
  porównywane z **ręcznie obliczonymi** wartościami referencyjnymi; `merge_score(α=0) == ri`.
- **`test_merger.py`:** zbieżność `len(unique(labels)) == target_k`; etykiety
  zwarte w `[0, k)`; lazy invalidation poprawnie pomija przestarzałe wpisy heap'a.

## 6.2. Tolerancja `ARI > 0.9` w testach porównawczych

Próg `0.9` (a nie `1.0`) ze względu na **niedeterministyczność `pymetis`** —
randomized refinement w METIS Kernighan-Lin daje minimalnie różne bisekcje między
uruchomieniami. Walidujemy zachowanie algorytmu, nie identyczność labels.
