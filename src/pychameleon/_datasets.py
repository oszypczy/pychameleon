"""Internal benchmark dataset loaders.

Helpers for the experiment runner and notebooks. Not part of the public API —
users with their own data don't need any of this.

Datasets are vendored under ``tests/data/`` so the experiment suite is fully
reproducible without network access (after the initial fetch via
``scripts/fetch_karypis_datasets.py``).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pychameleon._types import FloatMatrix, Labels

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
KARYPIS_DIR = DATA_DIR / "karypis"


@dataclass(frozen=True)
class Dataset:
    """Lightweight container for a benchmark dataset."""

    name: str
    X: FloatMatrix
    y: Labels | None  # ground-truth labels, ``None`` if unavailable
    paper_name: str | None = None  # e.g. "DS1" for Karypis 1999 reference


def _load_csv_xy(path: Path, sep: str) -> FloatMatrix:
    return np.loadtxt(path, delimiter=sep, dtype=np.float64)


def _load_karypis(name: str, paper_name: str) -> Dataset:
    """Load a Karypis CLUTO dataset (3-col CSV: x, y, class; -1 = noise)."""
    path = KARYPIS_DIR / f"{name}.csv"
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    X = arr[:, :2].astype(np.float64)
    y = arr[:, 2].astype(np.int64)
    return Dataset(name=name, X=X, y=y, paper_name=paper_name)


def load_aggregation() -> Dataset:
    """788-point Aggregation (Gionis et al. 2007), 7 classes.

    GT pochodzi z UEF clustering datasets (cs.joensuu.fi/sipu/datasets/) i
    pokrywa się 1:1 z punktami w ``tests/data/Aggregation.csv``.
    """
    arr = np.loadtxt(
        KARYPIS_DIR / "aggregation-gt.csv",
        delimiter=",", skiprows=1, dtype=np.float64,
    )
    X = arr[:, :2].astype(np.float64)
    y = arr[:, 2].astype(np.int64)
    return Dataset(name="aggregation", X=X, y=y, paper_name="Aggregation")


def load_smileface() -> Dataset:
    """644-point smile face (4 features: eyebrows, eyes, mouth)."""
    X = _load_csv_xy(DATA_DIR / "smileface.csv", sep=",")
    return Dataset(name="smileface", X=X, y=None)


def load_t4_8k() -> Dataset:
    """8000-point t4.8k (paper benchmark, no GT in this copy)."""
    X = _load_csv_xy(DATA_DIR / "t4.8k.csv", sep=" ")
    return Dataset(name="t4_8k", X=X, y=None)


def load_karypis_ds1() -> Dataset:
    return _load_karypis("cluto-t5-8k", "DS1")


def load_karypis_ds3() -> Dataset:
    """t4.8k with ground-truth labels (CLUTO copy).

    Same shape as :func:`load_t4_8k` but includes class labels (and a ``-1``
    noise label for ~10% of points). Use this when you want quality metrics.
    """
    return _load_karypis("cluto-t4-8k", "DS3")


def load_karypis_ds4() -> Dataset:
    return _load_karypis("cluto-t7-10k", "DS4")


def load_karypis_ds5() -> Dataset:
    return _load_karypis("cluto-t8-8k", "DS5")


# Registry for experiment runners — keep keys stable, callers reference by name.
ALL_DATASETS: dict[str, Callable[[], Dataset]] = {
    "aggregation": load_aggregation,
    "smileface": load_smileface,
    "t4_8k": load_t4_8k,
    "ds1": load_karypis_ds1,
    "ds3": load_karypis_ds3,
    "ds4": load_karypis_ds4,
    "ds5": load_karypis_ds5,
}


def load(name: str) -> Dataset:
    """Load a benchmark dataset by registry key. See :data:`ALL_DATASETS`."""
    if name not in ALL_DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(ALL_DATASETS)}"
        )
    return ALL_DATASETS[name]()
