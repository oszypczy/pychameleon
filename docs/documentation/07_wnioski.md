# 7. Wnioski

## 7.1. Realizacja celu projektu

Zaimplementowano CHAMELEON (Karypis, Han, Kumar 1999) w pełni zgodnie z paperem: 3 fazy (k-NN graph → rekurencyjna bisekcja → aglomeracja sterowana dynamicznym modelem), formuły RI/RC/score zaimplementowane wektorowo, API zgodne ze scikit-learn (`check_estimator` przechodzi). Pakiet `pychameleon v0.1` jest gotowy do publikacji na PyPI.

## 7.2. Główne wnioski z eksperymentów

1. **Algorytm reprodukuje wyniki paperu.** Średnia Quality (ARI+NMI)/2 po 5 datasetach z ground-truth wynosi **0.93**. Na DS1 osiągnięto perfekcyjną rekonstrukcję (ARI = 1.0), na DS5 i Aggregation ARI > 0.96. To potwierdza poprawność implementacji dynamicznego modelowania.

2. **pychameleon jest szybszy i dokładniejszy od jedynej publicznej referencji.** Speedup **16×–269×** vs Moonpuck (zależnie od `n`), przy jednoczesnej wyższej accuracy na Aggregation (97.8% vs 71.2%). Cztery źródła speedupu: KDTree zamiast `O(n²)` k-NN, cache bisektorów, lazy invalidation heap-a, vectorised metrics. Pokazuje to, że dobrze zaprojektowana implementacja może wyjść daleko poza referencję bez zmiany algorytmu.

3. **Empiryczna skalowalność lepsza od teoretycznej.** Teoretyczna złożoność `O(n log² n)` w praktyce daje nachylenie ≈ 1.0 w log-log na zakresie n ∈ [500, 50000]. Dla n = 50 000 pełny pipeline kończy się w 24 s. Wymiarowość d ma znikomy wpływ na runtime do d = 50 przy n = 2000 — efekt curse of dimensionality nie pojawia się przy umiarkowanych rozmiarach.

4. **Parametry są dataset-specific — uniwersalnej konfiguracji nie ma.** Optymalne `alpha` waha się 0.5–3.0, `k_nn` 10–30, `min_cluster_size` 40–160 w zależności od zbioru. Grid-search per dataset jest konieczny dla maksymalnej jakości. `alpha` jest parametrem-driverem jakości (silny wpływ na ARI), `k_nn` — driverem kosztu (silny wpływ na runtime).

5. **DS3 to znana granica algorytmu.** ARI cap na 0.71 wynika z **braku noise-handlingu** w wersji paperu z 1999 r. ~10% punktów oznaczonych w GT jako szum trafia do najbliższych klastrów. To nie błąd implementacji — to cecha algorytmu opisana w paperu. Workaround: post-processing oparty na density lub parametr `min_samples_per_cluster` (roadmap v0.2).

6. **Próg porównawczy "ARI > 0.9 vs Moonpuck" okazał się nieadekwatny.** Założenie z etapu 1 zakładało, że Moonpuck jest oracle'em — w rzeczywistości wypada gorzej niż GT. Wyższa zgodność z Moonpuckiem oznaczałaby gorszą zgodność z prawdą. Końcowa walidacja porównuje z GT (sekcja 6.1) i z Moonpuck na metrykach runtime (sekcja 6.2), ale **nie** używa Moonpucka jako referencji jakościowej.

## 7.3. Ograniczenia obecnej implementacji

| Ograniczenie | Konsekwencja | Mitigacja                                |
|--------------|--------------|------------------------------------------|
| Brak noise-handlingu | DS3 cap ARI 0.71 | Roadmap v0.2: `min_samples_per_cluster` |
| KDTree degeneruje dla `d > 10` | Spowolnienie dla wysokowymiarowych | Roadmap v0.3: backend ANN (Annoy/HNSW)  |
| Niedeterminizm pymetis (KL refinement) | Etykiety mogą się minimalnie różnić między uruchomieniami | Roadmap v0.2: parametr `random_state`   |
| Brak GPU                       | Limit przepustowości na CPU                              | Wykraczające poza scope v0.x            |

## 7.4. Wkład projektu

W stosunku do paperu i implementacji referencyjnej:

- **Pierwsza implementacja CHAMELEON-a zgodna ze scikit-learn API** — możliwa do użycia w `Pipeline`, `GridSearchCV`, kompatybilna z `check_estimator`.
- **Pełna bateria metryk jakości** (9 metryk external + internal) zamiast samego ARI — daje pełniejszy obraz właściwości klasteryzacji.
- **Dwukrotnie wzbogacony zestaw testowy** vs paper — DS1/DS3/DS4/DS5 z prawdziwym ground-truth pozwalają na rzetelną walidację ilościową, której paper nie zawiera.
- **Skrypt HPO** (`scripts/run_hpo.py`) — grid-search 216 kombinacji per dataset z idempotentnym CSV cache'm.

