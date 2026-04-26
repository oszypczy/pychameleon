# 5. Ocena efektywności proponowanego rozwiązania

> **Wymaganie #3 z PDF prowadzącego (str. 1):**
> *"ocenę efektywności proponowanego rozwiązania"*

Niniejszy rozdział łączy:

1. **analizę teoretyczną** złożoności obliczeniowej (sekcja §4.5 paperu Karypis 1999),
2. **empiryczne pomiary** referencyjnej implementacji Moonpuck na 3 zbiorach (788, 644, 8000 pkt),
3. **predykcję przyspieszeń** naszej implementacji oraz omówienie ograniczeń.

## 5.1. Złożoność teoretyczna

Niech `n` — liczba punktów, `m` — liczba sub-klastrów po Fazie I (typowo `m ≈ n / 20`),
`d` — wymiar punktów, `k_nn` — parametr grafu sąsiedztwa.

### 5.1.1. Faza 0: budowa grafu k-NN (§4.2)

| Implementacja                         | Złożoność          | Komentarz                                  |
|---------------------------------------|---------------------|---------------------------------------------|
| **Nasza** (`scipy.spatial.KDTree`)    | **O(n log n · d)**  | konstrukcja drzewa + n zapytań              |
| Naiwna (Moonpuck)                     | O(n²)               | pełna macierz par odległości                |

**Asymptotyczna przewaga:** dla `n = 10⁴` redukcja `n²/n log n ≈ 10000/13 ≈ 750×`.

### 5.1.2. Faza I: rekurencyjna bisekcja (§4.4 Phase I)

`pymetis` (wielopoziomowy METIS, Karypis & Kumar 1998) realizuje 2-way min-cut
w czasie:

$$
T_{\text{bisect}}(V, E) = O(|E| + |V| \log |V|)
$$

W fazie I wykonujemy `O(m)` bisekcji — drzewo bisekcji ma głębokość `log m` i każdy
poziom dotyczy łącznie `n` wierzchołków, więc:

$$
T_{\text{Phase I}} = O\bigl(n \log m\bigr) \cdot \log n
$$

Dla typowych `m ≈ n/20`: `T = O(n · log²n)`.

### 5.1.3. Faza II: aglomeracyjne łączenie (§4.4 Phase II)

| Operacja                                    | Złożoność      | Komentarz                                |
|---------------------------------------------|----------------|------------------------------------------|
| Inicjalne wpisy do heap'a (m² par)          | O(m² log m)    | dla każdej pary (Cᵢ, Cⱼ) sąsiadującej   |
| Pojedyncze pop z heap'a                     | O(log m)       | amortyzowany przy lazy invalidation     |
| Update sąsiadów po scaleniu                 | O(deg(Cᵢ) · log m) | deg ≈ k_nn dla typowych grafów        |
| Pełna pętla łączenia                         | **O(m² log m)**| dominujący termin                        |

### 5.1.4. Sumaryczna złożoność

$$
T_{\text{total}} = O\bigl(\,n \log n \cdot d \;+\; n \log^2 n \;+\; m^2 \log m\,\bigr)
$$

Dla typowych zbiorów (`m << n`, np. `m ≈ n/20`) dominuje **`O(n log²n)`** w Fazie I.

### 5.1.5. Złożoność pamięciowa

| Struktura                              | Pamięć       |
|----------------------------------------|--------------|
| `X` (dane źródłowe)                    | `O(nd)`      |
| Adjacency list + edge weights          | `O(n · k_nn)`|
| Etykiety `Labels`                      | `O(n)`       |
| Cache `internal_ic_cache`              | `O(m)`       |
| Heap kolejki priorytetowej             | `O(m²)` w worst-case (lazy invalidation) |

Sumarycznie: `O(nd + n · k_nn + m²)`. Dla `n = 10⁴`, `d = 2`, `k_nn = 20`, `m = 500`:
ok. **2.5 MB** — nieproblemowo.

## 5.2. Benchmarki referencyjne (Moonpuck)

W ramach przygotowania projektu uruchomiona została implementacja referencyjna
[Moonpuck/chameleon_cluster](https://github.com/Moonpuck/chameleon_cluster) (commit
`1c0a65ee6a79706e4d415dd7ca78da5d3c29906d`) z zastosowanymi patchami kompatybilności
(`metis` → `pymetis`, networkx 2.4+ API). Pełne wyniki są zapisane w
`benchmarks/reference_moonpuck/{aggregation,smileface,t4_8k}/`.

### 5.2.1. Środowisko testowe

| Komponent       | Specyfikacja                                |
|-----------------|---------------------------------------------|
| Maszyna         | MacBook Pro (Apple Silicon, macOS 25.4)     |
| Python          | 3.11.x                                      |
| METIS           | bundled w `pymetis>=2023.1`                 |
| networkx        | 3.x                                         |
| scipy           | 1.11+                                       |

### 5.2.2. Wyniki

| Dataset       | n     | k | k_nn | m  | α    | Runtime (s) | Klastrów znalezionych |
|---------------|-------|---|------|-----|------|-------------|-----------------------|
| aggregation   | 788   | 7 | 20   | 40  | 2.0  | 9.95        | 7                     |
| smileface     | 644   | 4 | 10   | 20  | 2.0  | 2.08        | 4                     |
| t4_8k         | 8000  | 6 | 20   | 40  | 2.0  | **674.47**  | 6                     |

**Obserwacja kluczowa:** runtime dla `t4_8k` (8000 pkt) wynosi 11 minut — i jest
zdominowany przez naiwny `O(n²)` k-NN. Dla porównania, predykowana złożoność `O(n log²n)`
naszej implementacji daje przybliżone przyspieszenie:

$$
\text{speedup} \approx \frac{n^2}{n \log^2 n} = \frac{n}{\log^2 n} \approx \frac{8000}{169} \approx 47\times
$$

Co oznacza spodziewany czas wykonania **kilkanaście sekund** zamiast 11 minut.

### 5.2.3. Wykresy referencyjne

Wykres clustering output dla każdego z 3 datasetów znajduje się w
`benchmarks/reference_moonpuck/<dataset>/plot.png`. W prezentacji (slajd 10)
i w eksperymentach końcowych Etapu 2 te wykresy będą porównywane bok-w-bok
z wynikami naszej implementacji.

## 5.3. Predykcja przyspieszeń vs Moonpuck

Zaplanowane optymalizacje w naszej implementacji vs Moonpuck:

| # | Optymalizacja                                            | Spodziewany efekt                              |
|---|----------------------------------------------------------|------------------------------------------------|
| 1 | `scipy.spatial.KDTree` zamiast naiwnego k-NN             | speedup `O(n²) → O(n log n)`, 47× dla n=8k     |
| 2 | Cache `_internal_interconnectivity`                      | redukcja powtarzanych bisekcji o `~m`          |
| 3 | Lazy invalidation zamiast O(m²) rebuild po każdym scaleniu | redukcja `m × O(m²) → O(m² log m)` całkowicie  |
| 4 | numpy ufuncs zamiast pętli Pythona w `metrics.py`        | constant factor ~10–50×                        |
| 5 | Adjacency list zamiast `networkx.Graph`                  | redukcja overhead'u — networkx ma wysoki narzut |

**Spodziewany całkowity speedup:** dla `n = 8000` ok. **50–100×** na korzyść
naszej implementacji (z 674s → ~7–15s).

## 5.4. Ograniczenia proponowanego rozwiązania

### 5.4.1. KDTree słabo skaluje w wysokich wymiarach

`scipy.spatial.KDTree` ma efektywne wyszukiwanie `O(log n)` na zapytanie tylko dla
**niskich wymiarów** (≤10D). Dla `d > 10` koszt rośnie do `O(n)` w worst-case
("curse of dimensionality"). Dla wysokowymiarowych danych planowane jest:

- **v0.3:** zastąpienie KDTree przez **Annoy** lub **HNSW** (approximate k-NN)
  — referencyjna implementacja w paperu Chameleon2++ (Bhattacharya et al. 2025).

### 5.4.2. pymetis nie jest deterministyczny

Multilevel METIS używa randomized refinement (Kernighan–Lin variant), więc różne
uruchomienia mogą dać minimalnie różne bisekcje. To wpływa na:

- **odtwarzalność testów** — testy porównawcze ARI vs Moonpuck używają tolerancji
  `ARI > 0.9` (rozdz. 6),
- **deterministyczność `Chameleon().fit(X).labels_`** — labels mogą się różnić
  między uruchomieniami nawet dla tego samego `X`. **Mitigacja:** w przyszłej
  wersji dodamy parametr `random_state: int | None`, który będzie przekazywany
  do `pymetis_options.seed`.

### 5.4.3. Pamięć `O(m²)` dla heap'a kolejki

W ekstremalnym przypadku każda para sub-klastrów może być wpisana do heap'a
z różnymi wersjami przed scaleniem. To daje teoretyczny worst-case `O(m³)` pamięci.
Praktycznie zwykle pozostaje `O(m²)`, ale dla `n = 10⁶` (`m ≈ 50000`) `m² = 2.5·10⁹` —
**za duże**. Mitigacja:

- **soft limit** w v0.2: jeśli heap przekroczy `MAX_HEAP_ENTRIES = 10⁷`, robimy
  pełen rebuild (jednokrotnie),
- alternatywnie ograniczamy parę kandydatów do **konektywnych** w grafie meta-klastrów,
  co redukuje do `O(m · avg_neighbors)`.

### 5.4.4. Brak obsługi szumu

CHAMELEON, w przeciwieństwie do DBSCAN/HDBSCAN, **przypisuje każdy punkt do klastra**.
W zbiorach z szumem (np. t4_8k zawiera "rozproszone" punkty tła), te punkty zostają
sztucznie przyłączone do najbliższego klastra. To celowy wybór projektowy paperu
Karypis 1999 — w v0.1 zachowujemy zgodność, ale w v0.2 rozważamy parametr
`min_samples_per_cluster: int` (klastry mniejsze niż próg → label `-1`, jak w sklearn).

## 5.5. Podsumowanie

| Aspekt                                | Ocena                                                          |
|---------------------------------------|----------------------------------------------------------------|
| Asymptotyczna złożoność czasowa       | `O(n log²n + m² log m)` — znacznie lepsza niż naiwne `O(n²)`   |
| Asymptotyczna złożoność pamięciowa    | `O(nd + n·k_nn + m²)` — typowo kilka MB                        |
| Spodziewane przyspieszenie vs Moonpuck | **50–100×** dla `n = 8000` (z 11 min → ~10s)                  |
| Skalowalność wymiarowa                | dobra dla `d ≤ 10`, ograniczona dla wyższych (motywacja v0.3)  |
| Determinizm                           | brak w v0.1 (`pymetis` randomized); planowane w v0.2           |
| Obsługa szumu                         | brak (zgodne z CHAMELEON 1999); rozważane w v0.2               |

**Wniosek:** zaplanowane rozwiązanie pozwala na obsługę zbiorów rzędu `10⁵–10⁶`
punktów w rozsądnym czasie (sekundy do minut), przy zachowaniu charakterystyki
algorytmu znanej z paperu Karypis 1999. Główne ograniczenie wymiarowości jest
identyfikowane i ma roadmap mitygacji.
