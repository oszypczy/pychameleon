# 6. Plan testów

Strategia walidacji jest 5-warstwowa, realizowana w `tests/` przy użyciu `pytest`,
`pytest-cov`, `numpy.testing`, `sklearn.utils.estimator_checks` oraz
`sklearn.metrics.adjusted_rand_score`.

| Warstwa            | Cel                                                          | Liczba testów / próg                | Status na 26.04.2026                  |
|--------------------|--------------------------------------------------------------|--------------------------------------|---------------------------------------|
| Jednostkowe        | Każdy moduł osobno z fixture'ami i hand-computed wartościami | ~25 (5 modułów × ~5 testów)         | szkielet w `tests/`, treść w Etap 2  |
| Integracyjne (e2e) | `Chameleon().fit(X)` na 3 datasetach                         | 7 testów; ARI vs ground truth > 0.85| 12 testów na skeletonie pass         |
| sklearn-compat     | `check_estimator(Chameleon())` — pełna suite zgodności       | ~30 wewnętrznych testów             | szkielet, odblokowany po `fit()`      |
| Porównawcze        | Wyniki nasze vs Moonpuck na tych samych parametrach          | 3 datasety × `ARI > 0.9`            | Etap 2                                |
| Parametryczne      | Sweep `k_nn`, `α`, `min_cluster_size`                        | 3 sweep'y × ~5 wartości             | Etap 2                                |
| Skalowalności      | `n` vs runtime; weryfikacja slope log-log ≈ 1.1              | 6 punktów `n ∈ [100, 50000]`        | Etap 2                                |

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
