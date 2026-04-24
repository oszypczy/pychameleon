"""Shared pytest fixtures.

Small benchmark datasets (Aggregation, smileface, t4.8k) are vendored in
``tests/data/`` so the test suite is self-contained. Cross-validation against
the Moonpuck reference output uses ``benchmarks/reference_moonpuck/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def aggregation_xy() -> np.ndarray:
    """The 788-point Aggregation dataset, 7 ground-truth clusters."""
    return np.loadtxt(DATA_DIR / "Aggregation.csv", delimiter=" ", dtype=np.float64)


@pytest.fixture(scope="session")
def smileface_xy() -> np.ndarray:
    """The 644-point smileface dataset."""
    return np.loadtxt(DATA_DIR / "smileface.csv", delimiter=",", dtype=np.float64)


@pytest.fixture
def small_blobs() -> np.ndarray:
    """Deterministic 60-point 2D dataset with 3 obvious clusters."""
    rng = np.random.default_rng(42)
    cluster_centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    points = np.vstack([c + 0.3 * rng.standard_normal((20, 2)) for c in cluster_centers])
    return points
