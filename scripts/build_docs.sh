#!/usr/bin/env bash
# Build docs/documentation/dokumentacja.pdf from markdown chapters.
#
# Łączy 9 plików markdown (README + rozdziały 01-08) w jeden PDF używając pandoc.
# Wymaga: pandoc, LaTeX (xelatex). Opcjonalnie: eisvogel template (lepsze formatowanie).
#
# Usage: bash scripts/build_docs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_DIR="$REPO_ROOT/docs/documentation"
OUT_PDF="$DOC_DIR/dokumentacja.pdf"
IMG_DIR="$REPO_ROOT/docs/presentation/images"

# Sprawdz dostepnosc pandoc
if ! command -v pandoc &> /dev/null; then
    echo "BŁĄD: pandoc nie jest zainstalowany. Zainstaluj: brew install pandoc"
    exit 1
fi

# Sprawdz dostepnosc xelatex (lepsze niz pdflatex dla polskich znakow)
LATEX_ENGINE="xelatex"
if ! command -v xelatex &> /dev/null; then
    echo "Ostrzezenie: xelatex nie jest dostepny, fallback do pdflatex"
    LATEX_ENGINE="pdflatex"
fi

cd "$DOC_DIR"

# Lista plikow w kolejnosci (README jako preambula, potem rozdzialy 01..08)
FILES=(
    "README.md"
    "01_wprowadzenie.md"
    "02_algorytm.md"
    "03_implementacja.md"
    "04_struktury_danych.md"
    "05_efektywnosc.md"
    "06_plan_testow.md"
    "07_zbiory_danych.md"
    "08_bibliografia.md"
)

echo "Buduje: $OUT_PDF"
echo "Pliki wejsciowe: ${#FILES[@]} plikow"

# Pandoc command
# Uwaga: --highlight-style=monochrome (zamiast tango) eliminuje zaleznosc od pakietu LaTeX framed
pandoc "${FILES[@]}" \
    --output "$OUT_PDF" \
    --pdf-engine="$LATEX_ENGINE" \
    --variable=mainfont:"Helvetica Neue" \
    --variable=monofont:"Menlo" \
    --variable=geometry:margin=2.5cm \
    --variable=lang:pl-PL \
    --variable=fontsize:11pt \
    --variable=colorlinks:true \
    --variable=linkcolor:blue \
    --variable=urlcolor:blue \
    --variable=toccolor:black \
    --toc \
    --toc-depth=2 \
    --number-sections \
    --no-highlight \
    --resource-path=".:$IMG_DIR" \
    --metadata title="Projekt implementacji algorytmu CHAMELEON" \
    --metadata author="Oliwier Szypczyn" \
    --metadata date="$(date +%Y-%m-%d)" \
    --metadata subtitle="Metody Eksploracji Danych — Etap 1"

if [[ -f "$OUT_PDF" ]]; then
    PAGES=$(mdls -name kMDItemNumberOfPages "$OUT_PDF" 2>/dev/null | awk '{print $3}')
    SIZE=$(du -h "$OUT_PDF" | awk '{print $1}')
    echo ""
    echo "Sukces: $OUT_PDF"
    echo "Rozmiar: $SIZE, stron: ${PAGES:-?}"
else
    echo "BŁĄD: Nie udało się utworzyć $OUT_PDF"
    exit 1
fi
