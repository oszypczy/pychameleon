"""Fetch the Chameleon paper's synthetic 2D datasets (Karypis 1999, Figure 9).

The original paper used five synthetic 2D datasets DS1..DS5 with 6,000-10,000
points. The CLUTO toolkit (also by Karypis) bundles the same files; the
``deric/clustering-benchmark`` repository mirrors them as ARFF.

ARFF mapping (from CLUTO file headers):
- ``cluto-t4-8k``  → DS3 in the paper (8000 pts, 6 letter-shape clusters + noise)
- ``cluto-t5-8k``  → DS1 in the paper (8000 pts, 5 clusters + noise)
- ``cluto-t7-10k`` → DS4 in the paper (10000 pts, 8 clusters with overlaps)
- ``cluto-t8-8k``  → DS5 in the paper (8000 pts, 8 clusters of varying density)

Usage:
    python scripts/fetch_karypis_datasets.py
    python scripts/fetch_karypis_datasets.py --force
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO / "tests" / "data" / "karypis"
BASE_URL = (
    "https://raw.githubusercontent.com/deric/clustering-benchmark/"
    "master/src/main/resources/datasets/artificial"
)

# (filename without extension, paper name)
DATASETS = [
    ("cluto-t4-8k", "DS3"),
    ("cluto-t5-8k", "DS1"),
    ("cluto-t7-10k", "DS4"),
    ("cluto-t8-8k", "DS5"),
]


def parse_arff(arff_text: str) -> tuple[list[tuple[float, float]], list[str]]:
    """Extract (x, y) points and class labels from a CLUTO ARFF file."""
    points: list[tuple[float, float]] = []
    labels: list[str] = []
    in_data = False
    for line in arff_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.upper().startswith("@DATA"):
            in_data = True
            continue
        if not in_data or stripped.startswith("@"):
            continue
        parts = stripped.split(",")
        if len(parts) < 3:
            continue
        x = float(parts[0])
        y = float(parts[1])
        cls = parts[2].strip()
        points.append((x, y))
        labels.append(cls)
    return points, labels


def fetch_one(name: str, paper_name: str, force: bool) -> None:
    csv_path = TARGET_DIR / f"{name}.csv"
    if csv_path.exists() and not force:
        print(f"[skip] {name} — already present at {csv_path.relative_to(REPO)}")
        return

    url = f"{BASE_URL}/{name}.arff"
    print(f"[fetch] {name} from {url}")
    with urllib.request.urlopen(url, timeout=30) as response:
        arff_text = response.read().decode("utf-8")

    points, labels = parse_arff(arff_text)
    # Map "noise" -> -1 so downstream code can use np.int64 labels uniformly.
    int_labels = [-1 if lbl == "noise" else int(lbl) for lbl in labels]

    # Write a simple 3-column CSV: x, y, class. Header makes the format
    # unambiguous and pandas-friendly.
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("x,y,class\n")
        for (x, y), cls in zip(points, int_labels, strict=True):
            f.write(f"{x},{y},{cls}\n")

    n = len(points)
    n_noise = sum(1 for lbl in int_labels if lbl == -1)
    n_clusters = len({lbl for lbl in int_labels if lbl != -1})
    print(
        f"[done]  {name} ({paper_name}): {n} pts, {n_clusters} clusters, "
        f"{n_noise} noise -> {csv_path.relative_to(REPO)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if file exists"
    )
    args = parser.parse_args()

    for name, paper_name in DATASETS:
        fetch_one(name, paper_name, args.force)


if __name__ == "__main__":
    main()
