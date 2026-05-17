# 4. Instrukcja użytkownika

## 4.1. Instalacja

Pakiet wymaga **Pythona ≥ 3.11**. Zalecany manager pakietów: `uv` (z drop-in `pip` jako fallback).

```bash
# wariant z uv (zalecany)
uv add pychameleon

# wariant z pip
pip install pychameleon
```

Zależności runtime są minimalne: `numpy`, `scipy`, `scikit-learn`, `pymetis`. Wszystkie zostaną zainstalowane automatycznie.

## 4.2. Minimalny przykład

```python
import numpy as np
from pychameleon import Chameleon

X = np.random.rand(500, 2)

model = Chameleon(n_clusters=5, k_nn=10, min_cluster_size=0.025, alpha=2.0)
labels = model.fit_predict(X)

print(f"Znaleziono {model.n_clusters_} klastrów; etykiety: {np.unique(labels)}")
```

API jest zgodne z konwencją scikit-learn — `Chameleon` można podać do `Pipeline`, `GridSearchCV`, `clone()`. `check_estimator(Chameleon())` przechodzi pełen zestaw sanity checks sklearn.

## 4.3. Parametry

| Parametr           | Default | Opis                                                        |
|--------------------|---------|-------------------------------------------------------------|
| `n_clusters`       | `8`     | Docelowa liczba klastrów                                    |
| `k_nn`             | `10`    | Liczba sąsiadów w grafie k-NN (Faza 0)                      |
| `min_cluster_size` | `0.025` | Minimalny rozmiar sub-klastra (frakcja `n` jeśli `< 1`, inaczej liczba bezwzględna) |
| `alpha`            | `2.0`   | Wykładnik RC w `merge_score = RI · RC^α`                     |

Atrybuty po `fit()`:

- `labels_` — `ndarray[int64]` shape `(n,)`, etykiety w `[0, n_clusters_)`
- `n_clusters_` — rzeczywista liczba znalezionych klastrów (≤ `n_clusters`)
- `n_features_in_` — liczba cech zaobserwowana na wejściu

## 4.4. Reprodukcja eksperymentów z dokumentacji

Repozytorium zawiera kompletny zestaw skryptów do reprodukcji wyników z rozdziału 6. Wszystkie wyniki są zapisywane jako CSV w `results/` (idempotent — re-runy pomijają obliczone wiersze):

```bash
# Klonowanie i setup
git clone https://github.com/oszypczy/pychameleon.git
cd pychameleon
uv sync --all-extras

# 1. Porównanie jakości na 7 datasetach (rozdz. 6.1, 6.2)
uv run python scripts/run_experiments.py compare

# 2. Sweep parametrów (rozdz. 6.3)
uv run python scripts/run_experiments.py sweep --param k_nn
uv run python scripts/run_experiments.py sweep --param alpha
uv run python scripts/run_experiments.py sweep --param min_cluster_size

# 3. Skalowalność wzgl. n i d (rozdz. 6.4)
uv run python scripts/run_experiments.py scalability

# 4. Grid-search HPO (rozdz. 6.3)
uv run python scripts/run_hpo.py --grid coarse

# 5. Eksport figur z wynikami do PNG (do dokumentacji)
uv run python scripts/export_figures.py
```

Każdy skrypt zapisuje wyniki do `results/*.csv` i (dla `compare`) etykiety predykcji per-punkt do `results/labels/<dataset>.csv`.

## 4.5. Notebooki analityczne

W katalogu `notebooks/` znajdują się 3 notebooki Jupytera prezentujące wyniki w formie interaktywnej:

| Notebook                                | Zawartość                                              |
|-----------------------------------------|--------------------------------------------------------|
| `01_jakosc_vs_baseline.ipynb`           | Jakość pychameleona na 4 datasetach z ground-truth     |
| `02_porownanie_z_moonpuck.ipynb`        | Porównanie z implementacją referencyjną (speedup, ARI) |
| `03_wrazliwosc_parametrow.ipynb`        | Wpływ k_nn, alpha, min_cluster_size na jakość          |

```bash
uv run jupyter lab notebooks/
```

Notebooki czytają wyłącznie z `results/*.csv` (read-only), więc można je odpalać niezależnie od skryptów eksperymentalnych.

## 4.6. Wymagania sprzętowe

Eksperymenty z rozdziału 6 uruchomiono na MacBook Pro z Apple Silicon (M1). Czasy referencyjne:

- `compare` (7 datasetów): ~20 s łącznie
- `sweep --param alpha` (4 datasety × 6 wartości): ~2 min
- `scalability` (n ∈ {500..50000} × 3 repeats): ~10 min
- `run_hpo.py --grid coarse` (4 datasety × 64 kombinacji): ~15 min

Pamięć: dla największego zbioru (n = 50000, d = 2) szczyt ~600 MB.

## 4.7. Uruchomienie testów

```bash
uv run pytest                  # wszystkie testy + coverage
uv run pytest tests/test_graph.py  # konkretny moduł
uv run ruff check .            # lint
uv run mypy src/               # type-check
```
