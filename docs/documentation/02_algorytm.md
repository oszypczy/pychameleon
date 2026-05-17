# 2. Algorytm CHAMELEON

Algorytm wykonuje 3 etapy: (0) budowa grafu k-NN; (I) rekurencyjna bisekcja
grafu na `m` sub-klastrów; (II) aglomeracyjne łączenie sub-klastrów do
osiągnięcia `n_clusters`.

## 2.1. Faza 0 — graf k-NN (§4.2)

Buduje rzadki graf `G = (V, E, w)`: krawędź `(i, j)` istnieje wtedy, gdy `j`
jest jednym z *k* najbliższych sąsiadów `i` (lub odwrotnie). Waga krawędzi:
`w(i, j) = 1 / d(xᵢ, xⱼ)`. Realizacja przez `scipy.spatial.KDTree.query` w czasie
**`O(n log n)`** (vs naiwne `O(n²)` w implementacji referencyjnej).

## 2.2. Faza I — rekurencyjna bisekcja (§4.4 Phase I)

Dopóki istnieje sub-klaster o rozmiarze > `min_cluster_size`, dzielimy największy
2-way min-cutem. Realizacja przez `pymetis.part_graph(2, objtype='cut', ufactor=250)`
(METIS multilevel, Karypis & Kumar 1998). Wynik: `m ≈ n / 20` sub-klastrów.

## 2.3. Faza II — aglomeracja (§4.4 Phase II)

Kolejka priorytetowa (heapq) z parami sąsiadujących sub-klastrów, kluczowanymi
przez `merge_score`. Iteracyjnie pobieramy parę o najwyższym score, łączymy,
aktualizujemy sąsiedztwo. **Lazy invalidation** (version counter na każdym klastrze)
eliminuje potrzebę `O(m²)` rebuildu kolejki po każdym scaleniu.

## 2.4. Miary podobieństwa (§4.3)

Niech `EC_{Cᵢ,Cⱼ}` — krawędzie krzyżujące granicę między klastrami; `EC_Cᵢ` —
bisektor klastra `Cᵢ` (zbiór krawędzi przeciętych przy 2-way bisekcji `Cᵢ`).

**Relative Interconnectivity** (eq. 1):

$$
RI(C_i, C_j) = \frac{|EC_{C_i,C_j}|}{(|EC_{C_i}| + |EC_{C_j}|) / 2}
$$

**Relative Closeness** (eq. 2):

$$
RC(C_i, C_j) = \frac{\bar S_{EC_{C_i,C_j}}}{\frac{|C_i|}{|C_i|+|C_j|}\bar S_{EC_{C_i}} + \frac{|C_j|}{|C_i|+|C_j|}\bar S_{EC_{C_j}}}
$$

gdzie $\bar S_E$ — średnia waga krawędzi w zbiorze E.

**Merge score** (eq. 4):

$$
\text{score}(C_i, C_j) = RI(C_i, C_j) \cdot RC(C_i, C_j)^{\alpha}
$$

Wartość `α > 1` preferuje closeness; `α < 1` — interconnectivity. Domyślnie `α = 2.0`.

## 2.5. Parametry

| Parametr           | Domyślnie | Wpływ                                            |
|--------------------|-----------|--------------------------------------------------|
| `n_clusters`       | 8         | docelowa liczba klastrów                         |
| `k_nn`             | 10        | gęstość grafu k-NN                               |
| `min_cluster_size` | 0.025·n   | minimalny rozmiar sub-klastra w fazie I          |
| `alpha`            | 2.0       | eksponent RC w merge_score                       |
