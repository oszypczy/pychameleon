# 6. Plan testów

Niniejszy rozdział opisuje strategię walidacji poprawności i jakości implementacji.
Plan testów jest pięciowarstwowy: testy **jednostkowe** (każdego modułu osobno),
**integracyjne** (end-to-end na zbiorach testowych), **porównawcze** (z implementacją
referencyjną Moonpuck), **parametryczne** (sweep parametrów), **skalowalności** (n vs runtime).

## 6.1. Narzędzia

- **pytest** ≥ 7.4 — test runner, fixtures, parametryzacja
- **pytest-cov** ≥ 4.1 — pokrycie kodu
- **numpy.testing** — `assert_allclose`, `assert_array_equal`
- **scikit-learn** — `sklearn.metrics.adjusted_rand_score` dla porównań,
  `sklearn.utils.estimator_checks.check_estimator` dla zgodności API
- **ruff** ≥ 0.1 — linting (część CI)
- **mypy** ≥ 1.7 (strict) — type checking (część CI)

Konfiguracja w `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--cov=pychameleon", "--cov-report=term-missing"]
```

## 6.2. Struktura katalogu `tests/`

```
tests/
├── conftest.py             # fixtures: aggregation_xy, smileface_xy, small_blobs
├── data/                   # vendored CSV datasets
│   ├── Aggregation.csv
│   ├── smileface.csv
│   └── t4_8k.csv
├── test_graph.py           # jednostkowe — knn_graph
├── test_partition.py       # jednostkowe — initial_subclusters
├── test_metrics.py         # jednostkowe — RI, RC, merge_score
├── test_merger.py          # jednostkowe — merge_to_k_clusters
├── test_chameleon.py       # integracyjne — pełen pipeline
└── test_sklearn_api.py     # zgodność check_estimator
```

## 6.3. Testy jednostkowe

### 6.3.1. `test_graph.py` — moduł `graph.py`

| Test                                       | Co weryfikuje                                                  |
|--------------------------------------------|----------------------------------------------------------------|
| `test_knn_graph_shape`                     | `len(adjacency) == n_samples`                                  |
| `test_knn_graph_symmetry`                  | `j ∈ adj[i] <=> i ∈ adj[j]` (symetryczny graf)                  |
| `test_knn_graph_weights_positive`          | wszystkie wagi > 0                                             |
| `test_knn_graph_weight_inverse_distance`   | `weight ≈ 1 / distance`                                        |
| `test_knn_graph_no_self_loops`             | `i ∉ adj[i]` dla wszystkich `i`                                |
| `test_knn_graph_k_neighbors_count`         | `len(adj[i]) >= k` (≥ z uwagi na symetryzację)                 |
| `test_knn_graph_two_clusters_disconnected` | dla dobrze odseparowanych klastrów: każdy wierzchołek ma sąsiadów tylko z własnego klastra |

### 6.3.2. `test_partition.py` — moduł `partition.py`

| Test                                            | Co weryfikuje                                                  |
|-------------------------------------------------|----------------------------------------------------------------|
| `test_initial_subclusters_assigns_all_points`   | każdy punkt ma label != -1                                    |
| `test_initial_subclusters_respects_min_size`    | każdy sub-klaster ma rozmiar ≥ `min_cluster_size` (mod 1)     |
| `test_initial_subclusters_count_reasonable`     | liczba sub-klastrów `m ≈ n / min_cluster_size`                |
| `test_initial_subclusters_disconnected_components`| dla 3 odseparowanych blobów: minimum 3 sub-klastry           |

### 6.3.3. `test_metrics.py` — moduł `metrics.py`

| Test                                          | Co weryfikuje                                                |
|-----------------------------------------------|--------------------------------------------------------------|
| `test_ri_hand_computed_triangle`              | dla grafu trójkątnego z ręcznie policzonym RI: `ri_func(...) ≈ expected` |
| `test_ri_disconnected_clusters_zero`          | RI dwóch całkowicie odseparowanych klastrów = 0              |
| `test_rc_hand_computed_triangle`              | dla grafu z ręcznie policzonym RC                            |
| `test_merge_score_combines_correctly`         | `merge_score(c, alpha) ≈ ri(c) * rc(c)**alpha` dla różnych α |
| `test_merge_score_alpha_zero_equals_ri`       | dla α=0: `merge_score == ri`                                 |
| `test_merge_score_symmetric`                  | `score(ci, cj) == score(cj, ci)`                             |

**Wartości referencyjne** są obliczone ręcznie na małych grafach (3–5 wierzchołków)
i zapisane jako stałe w testach.

### 6.3.4. `test_merger.py` — moduł `merger.py`

| Test                                          | Co weryfikuje                                                |
|-----------------------------------------------|--------------------------------------------------------------|
| `test_merge_to_k_clusters_target_count`       | po wykonaniu: `len(unique(labels)) == target_k`              |
| `test_merge_to_k_clusters_labels_consecutive` | etykiety zwarte w `[0, k)`                                   |
| `test_merge_to_k_clusters_alpha_2_default`    | dla α=2.0 wyniki zgodne z `merge_score = RI · RC²`           |
| `test_merge_to_k_clusters_no_positive_score`  | jeśli wszystkie pary mają score ≤ 0: stop, `n_clusters_ < target_k` |
| `test_merge_to_k_clusters_already_k`          | dla `m == target_k` na wejściu: brak operacji, identyczne labels |

## 6.4. Testy integracyjne (end-to-end)

### 6.4.1. `test_chameleon.py`

Testy przechodzące **już teraz** (na skeletonie, bez działającego `fit()`):

| Test                                              | Status |
|---------------------------------------------------|--------|
| `test_default_construction`                       | pass |
| `test_get_params_returns_init_args`               | pass |
| `test_set_params_updates_attributes`              | pass |
| `test_clone_preserves_params`                     | pass |
| `test_version_is_exported`                        | pass |
| `test_n_clusters_negative_raises`                 | pass |
| `test_k_nn_too_small_raises`                      | pass |
| `test_min_cluster_size_invalid_raises`            | pass |
| `test_alpha_negative_raises`                      | pass |
| (3 inne dla parametrów)                           | pass |

**Testy do odblokowania po implementacji `fit()`:**

| Test                                              | Co weryfikuje                                            |
|---------------------------------------------------|----------------------------------------------------------|
| `test_fit_returns_self`                           | `model.fit(X)` zwraca `self`                             |
| `test_fit_sets_labels_attribute`                  | po fit: `model.labels_.shape == (n,)`                    |
| `test_fit_sets_n_features_in`                     | `model.n_features_in_ == X.shape[1]`                     |
| `test_fit_predict_equivalent_to_fit_then_labels`  | `model.fit_predict(X) == model.fit(X).labels_`           |
| `test_aggregation_recovers_7_clusters`            | dla aggregation: `n_clusters_ == 7`, ARI vs ground truth > 0.85 |
| `test_smileface_recovers_4_clusters`              | dla smileface: `n_clusters_ == 4`                        |
| `test_small_blobs_recovers_3_clusters`            | dla `small_blobs`: `n_clusters_ == 3`, ARI > 0.95        |

## 6.5. Test zgodności sklearn API

### 6.5.1. `test_sklearn_api.py`

```python
from sklearn.utils.estimator_checks import check_estimator
from pychameleon import Chameleon

@pytest.mark.skip(reason="enable once Chameleon.fit is implemented")
def test_check_estimator() -> None:
    """Run sklearn's full estimator API compliance suite."""
    check_estimator(Chameleon())
```

`check_estimator` to suite ~30 testów weryfikujących:

- prawidłowe `__init__` / `get_params` / `set_params` / `clone`,
- `fit` zwraca `self`,
- atrybuty `labels_`, `n_features_in_` poprawnie ustawione po `fit`,
- powtórne `fit` z innymi danymi resetuje stan,
- estymator daje się serializować (zgodność z sklearn persistence layer),
- compatibility z `sklearn.pipeline.Pipeline`.

Ten test jest **kluczowy dla v0.2 (publikacja PyPI)** — passing `check_estimator`
to standard de facto dla bibliotek klasteryzacji.

## 6.6. Testy porównawcze z implementacją referencyjną

### 6.6.1. Cel

Zweryfikować, że nasza implementacja daje **wyniki zgodne** z referencyjną
implementacją Moonpuck — pomimo różnic optymalizacyjnych (KDTree, lazy invalidation,
cache) podstawowy algorytm musi dawać te same klastry.

### 6.6.2. Metryka zgodności

**Adjusted Rand Index (ARI):**

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(reference_labels, our_labels)
assert ari > 0.9, f"ARI too low: {ari}"
```

**Próg `ARI > 0.9`** (a nie `== 1.0`) ze względu na:

- niedeterministyczność `pymetis` — różne ziarna RNG → minimalnie różne bisekcje,
- różną kolejność rozstrzygnięć remisów (`merge_score`) między implementacjami.

### 6.6.3. Konfiguracja porównania

Nasza implementacja musi być uruchomiona z **dokładnie tymi samymi parametrami**
co Moonpuck, zapisanymi w `benchmarks/reference_moonpuck/<dataset>/meta.json`:

```json
{
  "parameters": {"k": 7, "knn": 20, "m": 40, "alpha": 2.0}
}
```

Skrypt `scripts/run_comparison.py` (do napisania w Etapie 2):

```bash
uv run python scripts/run_comparison.py --dataset aggregation
# → ARI: 0.94 (PASS)
```

## 6.7. Testy parametryczne (sweep)

Cel — zrozumieć czułość algorytmu na poszczególne parametry. Wyniki w prezentacji końcowej (Etap 2).

### 6.7.1. Sweep `k_nn`

Dla zbioru aggregation (`n=788`, ground-truth: 7 klastrów):

```python
@pytest.mark.parametrize("k_nn", [5, 10, 15, 20, 30, 50])
def test_aggregation_k_nn_sweep(aggregation_xy, k_nn):
    model = Chameleon(n_clusters=7, k_nn=k_nn).fit(aggregation_xy)
    # report ARI vs ground truth, runtime
```

### 6.7.2. Sweep `alpha`

```python
@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_aggregation_alpha_sweep(aggregation_xy, alpha):
    model = Chameleon(n_clusters=7, alpha=alpha).fit(aggregation_xy)
```

### 6.7.3. Sweep `min_cluster_size`

```python
@pytest.mark.parametrize("min_cluster_size", [0.01, 0.025, 0.05, 0.1])
def test_aggregation_min_size_sweep(aggregation_xy, min_cluster_size):
    model = Chameleon(n_clusters=7, min_cluster_size=min_cluster_size).fit(aggregation_xy)
```

## 6.8. Testy skalowalności

### 6.8.1. Cel

Zweryfikować empirycznie, że runtime naszej implementacji rośnie z `n` zgodnie
z teoretyczną złożonością `O(n log²n)`, a NIE `O(n²)` jak w Moonpuck.

### 6.8.2. Setup

Generujemy syntetyczne zbiory blob'ów o rosnącym `n`:

```python
import sklearn.datasets

n_values = [100, 500, 1000, 5000, 10_000, 50_000]
for n in n_values:
    X, _ = sklearn.datasets.make_blobs(n_samples=n, centers=5, random_state=42)
    t0 = time.perf_counter()
    Chameleon(n_clusters=5).fit(X)
    runtime = time.perf_counter() - t0
    print(f"n={n}: runtime={runtime:.2f}s")
```

### 6.8.3. Oczekiwany trend

Wykres `log(runtime) vs log(n)` powinien mieć **slope ≈ 1.1** (czyli `n log²n`),
a NIE `slope ≈ 2` (czyli `n²`). Dla porównania ten sam test uruchomimy też dla
Moonpuck.

## 6.9. Pokrycie kodu (coverage)

Cel: **≥ 90% pokrycie linii** dla modułów `pychameleon/*.py` (z wyłączeniem
docstringów i `__init__.py`).

```bash
uv run pytest --cov=pychameleon --cov-report=html
# → htmlcov/index.html
```

Brakujące pokrycie raportowane w PR (przed publikacją na PyPI).

## 6.10. CI (Continuous Integration)

W v0.1 (Etap 1+2 MED) — brak CI. W v0.2 (PyPI release): GitHub Actions na każdy
push/PR z matrycą `ubuntu/macos × Python 3.11/3.12/3.13`:

```yaml
# .github/workflows/ci.yml (v0.2)
- run: uv sync --all-extras
- run: uv run ruff check
- run: uv run mypy src/
- run: uv run pytest --cov=pychameleon
```

## 6.11. Podsumowanie planu testów

| Warstwa            | Testy                                   | Status w Etapie 1                    |
|--------------------|-----------------------------------------|--------------------------------------|
| Jednostkowe        | ~25 (5 modułów × ~5 testów)             | szkielet napisany, treść w Etapie 2 |
| Integracyjne       | ~7 testów `test_chameleon.py`           | 12 testów aktualnych pass; reszta po `fit()` |
| sklearn API        | `check_estimator` (~30 testów)          | szkielet, odblokowany po `fit()`     |
| Porównawcze        | 3 datasety × ARI > 0.9                  | Etap 2                               |
| Parametryczne      | 3 sweep'y × ~5 wartości                 | Etap 2                               |
| Skalowalności      | 6 punktów `n` × 2 implementacje         | Etap 2                               |
| Coverage           | ≥ 90% linii                             | mierzony w Etapie 2                  |
