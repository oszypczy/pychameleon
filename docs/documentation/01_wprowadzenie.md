# 1. Wprowadzenie

## 1.1. Cel projektu

Celem projektu jest opracowanie projektu implementacji oraz docelowo pełnej implementacji
hierarchicznego algorytmu klasteryzacji **CHAMELEON** według artykułu źródłowego:

> Karypis G., Han E., Kumar V., *CHAMELEON: A hierarchical clustering algorithm using
> dynamic modeling*, IEEE Computer: Special Issue on Data Analysis and Mining, vol. 32,
> no. 8, 1999, str. 68–75.

W ramach przedmiotu *Metody Eksploracji Danych* (MED) projekt podzielony jest na dwa etapy:

| Etap | Deadline    | Punkty | Deliverable                                                                       |
|------|-------------|--------|-----------------------------------------------------------------------------------|
| 1    | 08.05.2026  | 15     | Projekt implementacji (niniejszy dokument) + prezentacja założeń                  |
| 2    | 02.06.2026  | 30     | Pełna implementacja, eksperymenty, raport końcowy, prezentacja wyników            |

Niniejszy dokument odpowiada Etapowi 1.

## 1.2. Problem klasteryzacji hierarchicznej

Klasteryzacja jest podstawowym zadaniem analizy skupień w eksploracji danych —
nieinformacją uczeniem (unsupervised learning), w którym celem jest podział zbioru
obiektów `X = {x₁, …, xₙ}` na rozłączne grupy (klastry) tak, by obiekty w obrębie
jednej grupy były do siebie podobne, a obiekty z różnych grup — różne.

Algorytmy klasteryzacji **hierarchicznej** budują dendrogram zagnieżdżonych klastrów,
zamiast jednego płaskiego podziału. Wyróżnia się dwa wariant:

- **aglomeracyjny (bottom-up)** — start z `n` jednoelementowych klastrów; w kolejnych
  krokach łączy się dwa najbardziej podobne klastry w jeden,
- **podziałowy (top-down)** — start z jednego dużego klastra; w kolejnych krokach
  dzieli się go na mniejsze.

CHAMELEON należy do rodziny algorytmów aglomeracyjnych, ale różni się od klasycznych
podejść (single/complete/average linkage) sposobem **modelowania podobieństwa
międzyklastrowego**.

## 1.3. Miejsce CHAMELEON-a w taksonomii algorytmów klasteryzacji

| Algorytm                  | Typ           | Reprezentacja klastra              | Główna miara podobieństwa             | Ograniczenia                                   |
|---------------------------|---------------|-------------------------------------|----------------------------------------|------------------------------------------------|
| **k-means** (1967)        | partycjonujący | centroid                            | odległość euklidesowa                  | tylko klastry kuliste, ustalone *k*            |
| **DBSCAN** (1996)         | gęstościowy   | regiony gęstości                    | ε-sąsiedztwo + MinPts                  | jednolity próg gęstości                        |
| **CURE** (1998)           | hierarchiczny | wiele reprezentantów                | minimalna odległość między reprezent.  | statyczny model klastra                        |
| **ROCK** (1999)           | hierarchiczny | graf "linków"                       | współdzielone sąsiedztwo               | tylko dane kategorialne; statyczny model       |
| **CHAMELEON** (1999)      | hierarchiczny | sub-klaster z grafu k-NN            | RI × RCᵅ (dynamiczny model)            | wymaga ustawienia *k*, *m*, α                  |
| **OPTICS** (1999)         | gęstościowy   | wykres osiągalności                 | reachability distance                  | wymaga eksploracji wykresu                     |
| **HDBSCAN** (2013)        | gęstościowy   | drzewo gęstości                     | mutual reachability                    | brak hierarchii w sensie linkage               |

CHAMELEON pojawił się jako odpowiedź na ograniczenia algorytmów wcześniejszych,
w szczególności CURE i ROCK, które używają **statycznego modelu** klastra
(np. zbiór reprezentantów lub macierz podobieństwa kategorialnego). Statyczny model
nie odzwierciedla wewnętrznej struktury klastra ani charakterystyki połączenia
między klastrami — co prowadzi do błędnych decyzji o łączeniu w przypadku klastrów
o różnych kształtach, gęstościach i rozmiarach.

## 1.4. Motywacja: "dynamic modeling"

CHAMELEON wprowadza pojęcie **dynamicznego modelowania klastrów**: kryterium
łączenia dwóch klastrów `Cᵢ, Cⱼ` zależy nie tylko od ich wzajemnego podobieństwa
(jak w linkage), ale również od **wewnętrznej charakterystyki** każdego z nich.

Konkretnie, CHAMELEON łączy `Cᵢ` i `Cⱼ` wtedy, gdy:

1. ich wzajemne **interconnectivity** (sumaryczna waga krawędzi krzyżujących granicę
   między klastrami) jest porównywalne z wewnętrzną interconnectivity każdego z nich
   (tzw. *relative interconnectivity*, RI),
2. **closeness** (średnia waga krawędzi granicznych) jest porównywalna z wewnętrzną
   closeness każdego z nich (*relative closeness*, RC).

Dzięki temu CHAMELEON poprawnie obsługuje sytuacje, w których:

- klastry mają **różne gęstości** (statyczne miary łączyłyby gęste klastry
  nawet gdy są wyraźnie odseparowane od rzadkich, co tu się nie dzieje, bo RC
  normalizuje przez wewnętrzną gęstość),
- klastry mają **różne rozmiary** (RI normalizuje przez wewnętrzną interconnectivity,
  która rośnie z rozmiarem klastra),
- klastry mają **niewypukłe kształty** (single/average linkage często fragmentuje
  takie klastry; CHAMELEON utrzymuje je w całości dzięki k-NN grafowi).

Paper Karypis 1999, sekcja 5 ("Experimental Results"), demonstruje tę przewagę
na zbiorach syntetycznych (DS1–DS5, gdzie m.in. zbiór t4_8k używany w niniejszym
projekcie), na których k-means, CURE i DBSCAN dają wyraźnie gorsze wyniki.

## 1.5. Kontekst projektu i deliverables

Projekt jest realizowany jako pakiet **`pychameleon`** w nowoczesnej konwencji Python
(src-layout, type hints, sklearn API). Długoterminowy cel — niezależny od oceny MED —
to publikacja jako biblioteka dostępna przez `pip install pychameleon` (PyPI, lato 2026)
oraz potencjalny pull request do `scikit-learn-contrib`.

Wybór nowoczesnych konwencji (sklearn-compatible API od pierwszego commita, src-layout,
ruff/mypy w trybie strict, pytest z coverage) jest świadomą decyzją projektową,
szczegółowo omówioną w rozdziale 3.

## 1.6. Struktura niniejszego dokumentu

- **Rozdział 2** opisuje algorytm CHAMELEON: 2 fazy, definicje miar podobieństwa,
  pseudokod kluczowych operacji.
- **Rozdział 3** prezentuje **strukturę implementacji** — podział na 5 modułów
  z jednoznacznym mapowaniem na sekcje paperu Karypis 1999. Jest to najważniejszy
  rozdział z punktu widzenia wymagań prowadzącego.
- **Rozdział 4** opisuje **struktury danych** — zarówno format danych źródłowych
  (numpy.ndarray), jak i wewnętrzne struktury algorytmu (CSR adjacency, priority queue).
- **Rozdział 5** zawiera **ocenę efektywności**: złożoność teoretyczną z paragrafu
  §4.5 paperu oraz empiryczne benchmarki implementacji referencyjnej Moonpuck
  (3 datasety, 788–8000 punktów).
- **Rozdziały 6–7** dokumentują **plan testów** i **charakterystykę zbiorów testowych**.
- **Rozdział 8** zawiera bibliografię.
