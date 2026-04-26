# Dokumentacja projektu implementacji — CHAMELEON

**Autor:** Oliwier Szypczyn
**Przedmiot:** Metody Eksploracji Danych (MED), semestr letni 2025/2026
**Prowadzący:** dr inż. Robert Bembenik
**Wydział:** Politechnika Warszawska, Wydział Elektroniki i Technik Informacyjnych
**Temat:** *CHAMELEON: hierarchical clustering algorithm using dynamic modeling*
(Karypis G., Han E., Kumar V., IEEE Computer 32(8), 1999, str. 68–75)

---

## Status dokumentu

Niniejszy dokument stanowi **projekt implementacji** algorytmu CHAMELEON — wymagany
deliverable Etapu 1 projektu MED z deadline'em 08.05.2026. Zgodnie z wymaganiami
prowadzącego (`Projekt MED_2026l_rb.pdf`, str. 1–2), dokument zawiera:

- strukturę implementacji z odniesieniem do konkretnych kroków algorytmu (rozdz. 3),
- struktury danych dla danych źródłowych i algorytmu (rozdz. 4),
- ocenę efektywności proponowanego rozwiązania (rozdz. 5).

> "Projekt implementacji nie jest implementacją" — *Projekt MED_2026l_rb.pdf*

Pełna implementacja jest deliverablem Etapu 2 (deadline 02.06.2026).

---

## Spis treści

| Nr | Plik | Treść |
|----|------|-------|
| 1  | [`01_wprowadzenie.md`](01_wprowadzenie.md)         | Cel projektu, problem klasteryzacji hierarchicznej, miejsce CHAMELEON-a w taksonomii |
| 2  | [`02_algorytm.md`](02_algorytm.md)                 | Charakterystyka CHAMELEON: 2 fazy, k-NN graph, miary RI/RC/α |
| 3  | [`03_implementacja.md`](03_implementacja.md)       | **Struktura implementacji** — funkcje, klasy, mapping na kroki algorytmu |
| 4  | [`04_struktury_danych.md`](04_struktury_danych.md) | **Struktury danych** źródłowe i wewnętrzne algorytmu |
| 5  | [`05_efektywnosc.md`](05_efektywnosc.md)           | **Ocena efektywności** — złożoność teoretyczna i benchmarki referencyjne |
| 6  | [`06_plan_testow.md`](06_plan_testow.md)           | Plan testów: jednostkowe, integracyjne, porównawcze, parametryczne, skalowalności |
| 7  | [`07_zbiory_danych.md`](07_zbiory_danych.md)       | Charakterystyka zbiorów testowych (Aggregation, smileface, t4_8k, syntetyczne) |
| 8  | [`08_bibliografia.md`](08_bibliografia.md)         | Bibliografia |

---

## Kompilacja do PDF

```bash
bash scripts/build_docs.sh
# tworzy: docs/documentation/dokumentacja.pdf
```

## Repozytorium kodu

Kod źródłowy projektu znajduje się w `src/pychameleon/`. Implementacja będzie
publikowana docelowo jako pakiet `pychameleon` na PyPI (lato 2026).

- Repozytorium lokalne: `/Users/oszypczy/VSCode/studia/MED/`
- Stan na dzień oddania Etapu 1: skeleton 5 modułów ze ścisłą zgodnością z sekcjami
  paperu Karypis 1999 (§4.2, §4.3, §4.4), 12 testów jednostkowych przechodzi,
  konfiguracja `ruff` i `mypy --strict` bez błędów.
