# 6. Wyniki eksperymentów

Eksperymenty zostały zaprojektowane tak, aby odpowiedzieć na cztery pytania postawione w opisie projektu:

1. **Jakość** — jak dobrze pychameleon odtwarza prawdziwe klastry na różnych zbiorach?
2. **Porównanie z referencją** — jak wypada względem implementacji Moonpuck (jedynej publicznie dostępnej implementacji CHAMELEON)?
3. **Wrażliwość parametrów** — jak parametry `k_nn`, `alpha`, `min_cluster_size` wpływają na wyniki?
4. **Skalowalność** — jak runtime rośnie wraz z liczbą punktów `n`?

Wszystkie eksperymenty są w pełni reprodukowalne — patrz rozdz. 4.4.

## 6.1. Jakość klasteryzacji na zbiorach z ground-truth

Cztery zbiory z papera Karypisa (DS1, DS3, DS4, DS5) oraz Aggregation posiadają wiarygodny ground-truth. Dla każdego uruchomiono `Chameleon.fit()` z parametrami zoptymalizowanymi grid-searchem (patrz sekcja 6.3) i obliczono pełen zestaw metryk klasteryzacyjnych.

### Bateria metryk

Zamiast pojedynczej ARI raportujemy 6 metryk **external** (porównujących z GT) oraz 3 **internal** (oceniających strukturę bez GT) — różne metryki chwytają różne aspekty:

| Metryka                | Typ      | Co mierzy                                              |
|------------------------|----------|--------------------------------------------------------|
| Adjusted Rand Index    | external | Pairwise agreement vs GT, chance-corrected             |
| Normalised Mutual Info | external | Wzajemna informacja, normalizowana                     |
| Adjusted Mutual Info   | external | NMI z korektą losowości                                |
| Homogeneity            | external | Klastry zawierają tylko jedną klasę GT                 |
| Completeness           | external | Klasa GT trafia do jednego klastra                     |
| V-measure              | external | Średnia harmoniczna homogeneity + completeness         |
| Silhouette             | internal | Spójność wewnętrzna vs separacja zewnętrzna            |
| Calinski-Harabasz      | internal | Stosunek wariancji między- do wewnątrz-klastrowej      |
| Davies-Bouldin         | internal | Średnie podobieństwo klastra do najbliższego sąsiada   |

### Wyniki na 4 datasetach Karypisa

| Dataset         | n     | k | Runtime [s] | ARI   | NMI   | AMI   | Homog. | Compl. | V-meas. | Silh. | **Quality**¹ |
|-----------------|-------|---|-------------|-------|-------|-------|--------|--------|---------|-------|--------------|
| **DS1**         | 8000  | 6 | 3.19        | 1.000 | 1.000 | 1.000 | 1.000  | 1.000  | 1.000   | 0.54  | **1.000**    |
| **DS5**         | 8000  | 8 | 6.53        | 0.962 | 0.972 | 0.972 | 0.957  | 0.987  | 0.972   | 0.01  | **0.967**    |
| **DS4**         | 10000 | 9 | 2.11        | 0.896 | 0.914 | 0.914 | 0.893  | 0.936  | 0.914   | −0.03 | **0.905**    |
| **DS3**         | 8000  | 6 | 4.24        | 0.708 | 0.895 | 0.895 | 0.810  | 1.000  | 0.895   | 0.31  | **0.801**    |
| **Aggregation** | 788   | 7 | 0.06        | 0.961 | 0.957 | 0.956 | 0.960  | 0.953  | 0.957   | 0.49  | **0.959**    |

¹ Quality = (ARI + NMI) / 2 — agregacja chance-corrected, obu metryk w `[0, 1]`.

![Jakość pychameleona na datasetach z ground-truth](images/results/quality_bars.png)

**Średnia Quality po 5 datasetach: 0.93.** DS1 to perfekcyjna rekonstrukcja, DS5 i Aggregation > 0.95, DS4 ~0.9 mimo 9 klastrów częściowo nakładających się.

### Galeria wizualna — ground truth vs predykcja vs correctness

Dla każdego datasetu pokazujemy trzy panele: prawdziwe klastry (kolory z GT), predykcję pychameleona (po dopasowaniu Hungarian na confusion matrix), oraz mapę correctness (zielony = poprawny, czerwony = błąd, szary = szum GT, ignorowany przy obliczaniu accuracy).

![Galeria GT/pred/correctness na 4 datasetach Karypisa](images/results/quality_gallery.png)

### Granica algorytmu: DS3

**DS3 to znana granica CHAMELEON-a.** Quality 0.80 (najgorszy wynik) wynika z faktu, że ~10% punktów w GT jest oznaczonych jako szum (label `-1`), a algorytm w wersji z papera Karypisa 1999 **nie ma noise-handlingu** — wszystkie punkty są przypisywane do najbliższego klastra. Po Hungarian matchingu punkty te trafiają w "nie ten klaster", co kosztuje ~20 punktów procentowych ARI. Roadmap v0.2 przewiduje parametr `min_samples_per_cluster` (analogicznie do HDBSCAN).

## 6.2. Porównanie z implementacją referencyjną Moonpuck

Implementacja referencyjna [Moonpuck/chameleon_cluster](https://github.com/Moonpuck/chameleon_cluster) jest **jedyną publicznie dostępną** implementacją CHAMELEON-a. Uruchomiono ją na 3 zbiorach wspólnych (Aggregation, smileface, t4_8k) — wyniki w `benchmarks/reference_moonpuck/`.

### Speedup

| Dataset       | n    | pychameleon [s] | Moonpuck [s] | Speedup      |
|---------------|------|-----------------|--------------|--------------|
| Aggregation   | 788  | 0.06            | 9.95         | **155×**     |
| smileface     | 644  | 0.13            | 2.08         | **16×**      |
| t4_8k         | 8000 | 2.51            | 674.47       | **269×**     |

Speedup waha się od 16× (smileface, n = 644) do 269× (t4_8k, n = 8000) — bottleneckiem w Moonpucku jest naiwny `O(n²)` k-NN, którego eliminacja przez KDTree daje największy zysk na większych zbiorach.

![Speedup pychameleon vs Moonpuck w funkcji rozmiaru zbioru](images/results/moonpuck_speedup.png)

### Jakość vs Moonpuck

Na Aggregation (jedynym zbiorze z prawdziwym ground-truth wśród wspólnych) porównujemy **accuracy** (% punktów przypisanych do właściwego klastra po Hungarian matchingu):

| Dataset       | pychameleon vs GT | Moonpuck vs GT  |
|---------------|-------------------|-----------------|
| Aggregation   | **97.8%**         | 71.2%           |

pychameleon nie tylko jest szybszy, **ale i dokładniejszy** od referencji. Moonpuck na Aggregation popełnia kilka błędów typu bridge-merge (łączy sąsiadujące klastry).

![Galeria GT / pychameleon / Moonpuck](images/results/moonpuck_gallery.png)

**Konsekwencja dla metryki "ARI > 0.9 vs Moonpuck"** zaproponowanej w etapie 1: próg ten okazał się nieadekwatny, bo Moonpuck wypada gorzej niż GT — wyższa zgodność z Moonpuckiem oznaczałaby gorszą zgodność z prawdą. Walidacja końcowa porównuje z GT, nie z referencyjną implementacją.

### Źródła speedupu

Cztery główne optymalizacje względem Moonpucka:

1. **KDTree** zamiast naiwnego `O(n²)` k-NN — ~47× speedup dla n = 8000.
2. **Cache `|EC_Cᵢ|`** — eliminacja powtarzanych bisekcji w Fazie II.
3. **Lazy invalidation heap-a** — bez rebuildu `O(m²)` po każdym scaleniu.
4. **numpy ufuncs** w `metrics.py` — stała 10–50× szybsza pętla.

## 6.3. Wpływ parametrów (sensitivity analysis)

Dla każdego parametru wykonano sweep po 6 wartościach na 4 zbiorach z GT (DS1, DS3, DS4, DS5). Punkty na wykresach to średnia po datasetach; wstęga to ± 1 odchylenie standardowe.

### Wpływ `k_nn` — liczba sąsiadów w grafie k-NN

![Wpływ k_nn na jakość klasteryzacji](images/results/sensitivity_k_nn.png)

`k_nn` kontroluje gęstość grafu k-NN. Sweet spot to **10–30**: zbyt małe `k_nn` daje rozpadnięty graf (klastry tracą spójność), zbyt duże tworzy bridge-edges między sąsiadującymi klastrami. Wpływ umiarkowany — ARI/NMI nie spada poniżej 0.7 nawet dla skrajnych wartości.

### Wpływ `alpha` — wykładnik RC w merge score

![Wpływ alpha na jakość klasteryzacji](images/results/sensitivity_alpha.png)

`alpha > 1` preferuje closeness (relative closeness), `alpha < 1` preferuje interconnectivity. Trend rosnący Quality z `alpha` — algorytm benefituje z silniejszej wagi na closeness, ale efekt jest dataset-specific (DS5 preferuje α = 3.0, DS1/DS3 są niewrażliwe).

### Wpływ `min_cluster_size` — rozmiar sub-klastra w Fazie I

![Wpływ min_cluster_size na jakość klasteryzacji](images/results/sensitivity_min_cluster_size.png)

Mniejszy `min_cluster_size` daje więcej drobnych sub-klastrów w Fazie I, co zwiększa elastyczność Fazy II ale i koszt obliczeniowy `O(m²)`. Trend lekko malejący Quality z `min_cluster_size` na małych datasetach (Aggregation), neutralny na większych.

### Optymalne parametry per dataset (grid-search HPO)

Grid-search 6 × 6 × 6 = **216 kombinacji** uruchomiony per dataset:

| Dataset | `k_nn` | `alpha` | `min_cluster_size` | Best ARI | Best NMI |
|---------|--------|---------|---------------------|----------|----------|
| DS1     | 20     | 0.5     | 80                  | 1.000    | 1.000    |
| DS3     | 15     | 0.5     | 40                  | 0.708    | 0.895    |
| DS4     | 10     | 2.5     | 160                 | 0.896    | 0.914    |
| DS5     | 30     | 3.0     | 40                  | 0.962    | 0.972    |

**Obserwacja:** optymalne parametry różnią się istotnie między datasetami — `alpha` od 0.5 do 3.0, `k_nn` od 10 do 30. Nie istnieje uniwersalna konfiguracja; HPO per dataset jest konieczne dla najlepszych wyników. DS3 nie osiąga ARI > 0.71 w żadnej konfiguracji (limit narzucony przez brak noise-handlingu).

## 6.4. Skalowalność

Testy skalowalności na syntetycznych zbiorach `sklearn.datasets.make_blobs` z 5 izotropowymi gaussowskimi klastrami (`cluster_std = 1.0`). Sweep `n ∈ {500, 1000, 2000, 5000, 10000, 20000, 50000}` przy `d = 2`; każda konfiguracja powtarzana 3× z różnymi seed'ami.

| n       | Runtime [s] (mean) |
|---------|--------------------|
| 500     | 0.23               |
| 1 000   | 0.72               |
| 2 000   | 1.18               |
| 5 000   | 3.25               |
| 10 000  | 5.86               |
| 20 000  | 11.44              |
| 50 000  | 24.25              |

![Skalowalność względem n (d = 2)](images/results/scalability_n.png)

**Empiryczne nachylenie log-log: ≈ 1.0** — runtime rośnie zauważalnie sublinearnie wzgl. teoretycznego `O(n log² n)`, co jest typowe dla algorytmów grafowych ograniczonych raczej constantami (np. setup KDTree, alokacje pymetis) niż samym scaleniem przy umiarkowanych `n`. Dla n = 50000 cały pipeline kończy się w **24 sekundach**, co dla Moonpucka byłoby szacunkowo ~1.5 godziny.
