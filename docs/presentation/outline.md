# Outline prezentacji — Etap 1, deadline 08.05.2026

**Format:** .pptx, 13–15 slajdów, ~15–20 minut prezentacji + Q&A
**Cel:** przedstawienie projektu implementacji algorytmu CHAMELEON dr. Bembenikowi
podczas konsultacji (planowany wt 05.05.2026 16:00–17:45 MS Teams)

**Wymagania prowadzącego (PDF, str. 1–2):** prezentacja musi obejmować
**zakres projektu, wybrane technologie, plany testów**.

---

## Slajd 1 — Tytuł

```
CHAMELEON: hierarchical clustering using dynamic modeling
implementacja w Pythonie + projekt jako pychameleon

Oliwier Szypczyn
Metody Eksploracji Danych, sem. letni 2026
Politechnika Warszawska, EITI
Data: <data konsultacji>
```

## Slajd 2 — Zakres projektu > wymaganie #1 prezentacji

- **Algorytm:** CHAMELEON (Karypis, Han, Kumar 1999, IEEE Computer 32(8))
- **Cel akademicki:** zaliczenie MED — Etap 1 (08.05) + Etap 2 (02.06), max 45 pkt
- **Cel długoterminowy:** publikacja jako biblioteka **`pychameleon`** na PyPI (lato 2026)
- **Etap 1 deliverables:** dokumentacja projektu + ta prezentacja
- **Etap 2 deliverables:** pełna implementacja, eksperymenty, raport końcowy

## Slajd 3 — Algorytm: motywacja "dynamic modeling"

- Klasyczne hierarchiczne (single/complete/average linkage) — **statyczny model**
- CURE (1998), ROCK (1999) — też statyczne, oparte na reprezentantach
- CHAMELEON: **dynamiczny model** — kryterium łączenia uwzględnia **wewnętrzną
  charakterystykę** klastrów (gęstość, interconnectivity)
- Wynik: poprawna klasteryzacja klastrów o **różnych kształtach, rozmiarach, gęstościach**

*[diagram: porównanie wyników CHAMELEON vs k-means na zbiorze t4_8k]*

## Slajd 4 — Algorytm: dwie fazy

```
       X (punkty, n × d)
              │
              ▼
   Faza 0: graf k-NN  (§4.2)            ──► scipy.spatial.KDTree, O(n log n)
              │
              ▼
   Faza I: rekurencyjna bisekcja        ──► pymetis (METIS multilevel)
   (§4.4 Phase I)                          → m sub-klastrów
              │
              ▼
   Faza II: aglomeracja                 ──► priority queue (heapq)
   (§4.4 Phase II)                         + lazy invalidation
              │
              ▼
       labels_ (n,)
```

## Slajd 5 — Miary RI/RC/α

**Relative Interconnectivity (eq. 1):**
$$
RI(C_i, C_j) = \frac{|EC_{C_i,C_j}|}{(|EC_{C_i}| + |EC_{C_j}|) / 2}
$$

**Relative Closeness (eq. 2):**
$$
RC(C_i, C_j) = \frac{\bar S_{EC_{C_i,C_j}}}{\frac{|C_i|}{|C_i|+|C_j|}\bar S_{EC_{C_i}} + \frac{|C_j|}{|C_i|+|C_j|}\bar S_{EC_{C_j}}}
$$

**Merge score (eq. 4):**
$$
\text{score}(C_i, C_j) = RI \cdot RC^\alpha
$$

α=2.0 (paper) — preferuje closeness; konfigurowalne.

## Slajd 6 — Wybrane technologie: backend > wymaganie #2 prezentacji

| Komponent      | Technologia                       | Powód wyboru                                             |
|----------------|-----------------------------------|----------------------------------------------------------|
| Język          | Python 3.11+                       | nowoczesne typing, stabilne wheele dla pymetis           |
| k-NN           | `scipy.spatial.KDTree`             | O(n log n), built-in dependency sklearn                  |
| Bisekcja grafu | `pymetis` ≥ 2023.1                 | bundles METIS C-lib, brak system dep                     |
| sklearn API    | `BaseEstimator` + `ClusterMixin`   | konwencja → łatwa adopcja, ścieżka do sklearn-contrib    |
| Walidacja      | `_parameter_constraints`           | deklaratywna, spójna ze sklearn                          |
| Algebra        | `numpy` ≥ 1.24                     | ufuncs, wektoryzacja                                     |

## Slajd 7 — Wybrane technologie: tooling > wymaganie #2 prezentacji

| Narzędzie | Rola                              | Dlaczego                                            |
|-----------|-----------------------------------|----------------------------------------------------|
| `uv`       | menedżer pakietów + venv         | 10–100× szybszy od pip                            |
| `hatchling`| build backend                    | minimalny, nowoczesny, brak setup.py               |
| `ruff`     | linter + formatter               | 10× szybszy od pylint+flake8+isort                |
| `mypy`     | type checker (strict)            | wczesne wykrywanie bugów                           |
| `pytest`   | test runner + coverage           | standard de facto                                  |
| `pandas`+`seaborn` | eksperymenty (Etap 2)    | wykresy n vs runtime, sweep parametrów             |

## Slajd 8 — Architektura: 5 modułów

| Moduł                | Sekcja paperu | Odpowiedzialność                          |
|----------------------|---------------|-------------------------------------------|
| `graph.py`           | §4.2          | k-NN graph z KDTree                       |
| `partition.py`       | §4.4 Phase I  | rekurencyjna bisekcja przez pymetis       |
| `metrics.py`         | §4.3          | RI, RC, merge_score                       |
| `merger.py`          | §4.4 Phase II | aglomeracyjne łączenie z heapq            |
| `chameleon.py`       | §4 całość     | klasa estymatora sklearn                  |

*[diagram strzałek między modułami — data flow]*

## Slajd 9 — Struktury danych

| Struktura                | Reprezentacja                          | Po co tak                              |
|--------------------------|----------------------------------------|----------------------------------------|
| Dane wejściowe `X`       | `numpy.ndarray (n, d) float64`         | konwencja sklearn                      |
| Graf k-NN                | `list[ndarray int64]` adjacency        | kompat. z pymetis CSR                  |
| Wagi krawędzi            | `list[ndarray float64]`                | tożsame z adjacency                    |
| Etykiety klastrów        | `ndarray int64` shape (n,)             | maski boolean, sklearn convention      |
| Kolejka priorytetowa     | `heapq` z lazy invalidation            | O(log n) update bez rebuild            |
| Cache `\|EC_Cᵢ\|`        | `dict[int, float]`                     | redukcja powtarzanych bisekcji         |

## Slajd 10 — Efektywność: teoria + benchmarki

**Teoretyczna złożoność:**

$$
T = O(n \log n + n \log^2 n + m^2 \log m), \quad m \approx n/20
$$

**Empiryczne benchmarki** (referencyjna implementacja Moonpuck):

| Dataset        | n     | Runtime  | Bottleneck                                |
|----------------|-------|----------|-------------------------------------------|
| aggregation    | 788   | 9.95 s   | balanced                                  |
| smileface      | 644   | 2.08 s   | balanced                                  |
| t4_8k          | 8000  | 674 s    | naiwny O(n²) k-NN dominuje                |

**Spodziewane przyspieszenie naszej implementacji:** 50–100× dla n=8000 (z 11 min → ~10s)
dzięki KDTree + cache + lazy invalidation.

## Slajd 11 — Plan testów: jednostkowe > wymaganie #3 prezentacji

- **Każdy z 5 modułów** ma osobny plik testowy
- ~25 testów jednostkowych łącznie
- Wartości RI/RC liczone **ręcznie** na grafach 3–5 wierzchołków → fixture'y z oczekiwanymi wartościami
- `test_graph.py`: symetria grafu, no self-loops, weights = 1/dist
- `test_partition.py`: wszystkie punkty przypisane, balans rozmiarów
- `test_metrics.py`: hand-computed RI/RC/score
- `test_merger.py`: zbieżność do `target_k`, lazy invalidation correctness

## Slajd 12 — Plan testów: integracyjne i porównawcze > wymaganie #3 prezentacji

- **End-to-end** na 3 datasetach: aggregation → 7 klastrów, smileface → 4, t4_8k → 6
- **Porównawcze** vs Moonpuck: ARI > 0.9 (tolerancja na niedeterminizm pymetis)
- **sklearn `check_estimator`** — pełna suite zgodności (~30 testów)
- **Sweep parametrów:** k_nn ∈ {5, 10, 20, 50}; α ∈ {0.5, 1, 2, 3}; min_size ∈ {0.01, 0.025, 0.1}
- **Skalowalność:** n ∈ {100, …, 50000}; sprawdzamy slope wykresu log-log
- **Coverage:** target ≥ 90% linii

## Slajd 13 — Plan czasowy do 02.06

```
25.04 ────── DZIŚ
        │
        │  Etap 1: dokumentacja + prezentacja
        │
05.05 ────── KONSULTACJA
        │
08.05 ────── DEADLINE Etap 1 (15 pkt)
        │
        │  Etap 2: implementacja modułów
        │   tydz. 1 (09–15.05): graph + partition
        │   tydz. 2 (16–22.05): metrics + merger
        │   tydz. 3 (23–29.05): chameleon e2e + experiments
        │   tydz. 4 (30.05–01.06): finalizacja
        │
02.06 ────── DEADLINE Etap 2 (30 pkt)
```

## Slajd 14 — Stan obecny — co już mamy > boost

- > **Skeleton 5 modułów** z kompletnymi sygnaturami i docstringami
- > **12 testów jednostkowych** przechodzi (`get_params`, `set_params`, `clone`, walidacja parametrów, ...)
- > **`pyproject.toml`** z hatchling/pymetis/sklearn, `ruff` i `mypy --strict` clean
- > **3 referencyjne benchmarki** (Moonpuck, 788/644/8000 pkt) — zarchiwizowane w `benchmarks/reference_moonpuck/`
- > **`scripts/setup_reference.sh`** — pełna reprodukowalność implementacji odniesienia

*[zrzut ekranu: pytest -v output z 12 zielonymi testami; lub repository tree]*

## Slajd 15 — Pytania

```
Dziękuję za uwagę!

Pytania?

Repozytorium: github.com/oszypczy/pychameleon (private, w toku)
Kontakt: oliwier.szypczyn@tenvalleys.com
```

---

## Wizualizacje do przygotowania (osobno, do `docs/presentation/images/`)

1. **`architecture.png`** — diagram 5 modułów (mermaid → PNG export)
2. **`data_flow.png`** — diagram strzałek X → Phase 0 → Phase I → Phase II → labels_
3. **`comparison_kmeans_chameleon.png`** — porównanie wyników na t4_8k (jeśli czas)
4. **`runtime_bar.png`** — bar chart 3 datasetów (Moonpuck baseline)
5. **`reference_aggregation.png`**, **`reference_smileface.png`**, **`reference_t4_8k.png`** —
   bezpośredni reuse z `benchmarks/reference_moonpuck/*/plot.png`

## Dodatkowe notatki dla prelegenta

- **Czas:** 60–90 sekund na slajd merytoryczny, 120 sekund na slajdy 8/10 (architektura, efektywność)
- **Pytania spodziewane:**
  - "Dlaczego pymetis a nie własny min-cut?" → wielopoziomowy METIS to SOTA, użyty też w paperu Karypis 1999
  - "Skąd ARI > 0.9 jako próg?" → niedeterminizm pymetis; eksperyment empiryczny pokaże faktyczny rozrzut
  - "Czy planujemy obsługę szumu?" → nie w v0.1 (zgodność z paperu); rozważane v0.2
  - "Dlaczego nie sklearn `AgglomerativeClustering`?" → sklearn nie ma dynamic modeling, tylko statyczne linkage
  - "Jak skaluje się dla d > 10?" → KDTree słabnie; v0.3 z Annoy/HNSW
