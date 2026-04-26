# 1. Wprowadzenie

## 1.1. Cel projektu

Celem jest opracowanie projektu implementacji oraz pełnej implementacji
hierarchicznego algorytmu klasteryzacji **CHAMELEON** (Karypis, Han, Kumar 1999)
w nowoczesnej konwencji Pythona, ze zgodnością ze scikit-learn API.

## 1.2. Miejsce CHAMELEON-a wśród algorytmów klasteryzacji

CHAMELEON jest hierarchicznym, aglomeracyjnym algorytmem klasteryzacji
operującym na **grafie k-najbliższych sąsiadów**. W przeciwieństwie do wcześniejszych
algorytmów hierarchicznych (CURE 1998, ROCK 1999, klasyczne single/complete/average
linkage), które stosują **statyczny model klastra** (zbiór reprezentantów lub macierz
podobieństw), CHAMELEON wprowadza **dynamiczne modelowanie** — kryterium łączenia
dwóch klastrów uwzględnia ich **wewnętrzną charakterystykę** (gęstość, interconnectivity).

Dzięki temu algorytm poprawnie obsługuje klastry o **różnych kształtach, rozmiarach
i gęstościach** — sytuacje, w których zawodzą k-means, DBSCAN, CURE i ROCK (dowiedzione
empirycznie w paperu Karypis 1999, sekcja 5, na zbiorach DS1–DS5).

Kryterium decyzyjne fazy łączenia:

$$
\text{score}(C_i, C_j) = RI(C_i, C_j) \cdot RC(C_i, C_j)^{\alpha}
$$

gdzie *RI* (relative interconnectivity) i *RC* (relative closeness) są
**znormalizowanymi** miarami w stosunku do wewnętrznej struktury klastrów (rozdz. 2).
