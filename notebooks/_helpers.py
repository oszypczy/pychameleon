"""Notebook helpers — load precomputed results, ground-truth datasets, Moonpuck
reference. Notebooks stay read-only over ``results/`` and ``benchmarks/``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
LABELS = RESULTS / "labels"
MOONPUCK = REPO / "benchmarks" / "reference_moonpuck"

DATASETS_2D = ["aggregation", "smileface", "t4_8k", "ds1", "ds3", "ds4", "ds5"]
DATASETS_WITH_GT = ["ds1", "ds3", "ds4", "ds5"]
DATASETS_WITH_MOONPUCK = ["aggregation", "smileface", "t4_8k"]

PRETTY = {
    "aggregation": "Aggregation",
    "smileface": "Smile face",
    "t4_8k": "t4.8k",
    "ds1": "DS1 (t5.8k)",
    "ds3": "DS3 (t4.8k+GT)",
    "ds4": "DS4 (t7.10k)",
    "ds5": "DS5 (t8.8k)",
}


def load_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "comparison.csv")


def load_sweep(param: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"sweep_{param}.csv")


def load_scalability() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "scalability.csv")


def load_hpo_best() -> dict:
    with (RESULTS / "hpo_best.json").open() as f:
        return json.load(f)


def load_pychameleon_labels(name: str) -> pd.DataFrame:
    """Per-point (x0, x1, label) from a previous run_experiments.py compare run."""
    return pd.read_csv(LABELS / f"{name}.csv")


def load_moonpuck_labels(name: str) -> pd.DataFrame | None:
    """Per-point (x, y, cluster) from Moonpuck benchmark, or None."""
    path = MOONPUCK / name / "labels.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = ["x0", "x1", "label"]
    return df


def load_moonpuck_meta(name: str) -> dict | None:
    path = MOONPUCK / name / "meta.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_ground_truth(name: str) -> pd.DataFrame | None:
    """For Karypis DSx datasets — returns (x0, x1, label) with -1 = noise."""
    karypis_map = {
        "ds1": "cluto-t5-8k",
        "ds3": "cluto-t4-8k",
        "ds4": "cluto-t7-10k",
        "ds5": "cluto-t8-8k",
        "aggregation": "aggregation-gt",
        "t4_8k": "cluto-t4-8k",  # same point cloud as ds3
    }
    key = karypis_map.get(name)
    if key is None:
        return None
    path = REPO / "tests" / "data" / "karypis" / f"{key}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = ["x0", "x1", "label"]
    return df


def scatter(ax, df: pd.DataFrame, *, title: str, s: float = 4.0) -> None:
    """Coloured scatter with a consistent palette. Noise (-1) drawn light gray."""
    noise = df["label"] == -1
    if noise.any():
        ax.scatter(df.loc[noise, "x0"], df.loc[noise, "x1"],
                   c="lightgray", s=s, alpha=0.4, linewidths=0)
    real = df.loc[~noise]
    ax.scatter(real["x0"], real["x1"], c=real["label"],
               cmap="tab10", s=s, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def fmt(x: float, p: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "—"
    return f"{x:.{p}f}"


def align_labels_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Remap predicted labels onto ground-truth label space via Hungarian on
    the confusion matrix. Points with gt == -1 (noise) are ignored when
    building the matrix; the returned labels keep pychameleon's mapping for
    those points (they'll show up as "incorrect" if gt was noise)."""
    from scipy.optimize import linear_sum_assignment

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    mask = gt >= 0
    pred_real = pred[mask]
    gt_real = gt[mask]

    pred_ids = np.unique(pred_real)
    gt_ids = np.unique(gt_real)
    # Confusion: rows = pred cluster, cols = gt cluster.
    conf = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    for i, p in enumerate(pred_ids):
        for j, g in enumerate(gt_ids):
            conf[i, j] = int(np.sum((pred_real == p) & (gt_real == g)))
    # Hungarian maximises by minimising negative.
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {pred_ids[r]: gt_ids[c] for r, c in zip(row_ind, col_ind)}
    # Unmatched pred clusters keep their id offset above max gt to mark them
    # visually as "extra" — caller's correct/incorrect logic treats them as
    # mismatched anyway.
    extra = (gt_ids.max() if len(gt_ids) else 0) + 1
    remapped = np.array([mapping.get(p, extra) for p in pred])
    return remapped


def correctness_scatter(
    ax, xy: pd.DataFrame, gt: np.ndarray, pred_aligned: np.ndarray, *,
    title: str, s: float = 4.0,
) -> tuple[int, int, int]:
    """Plot points coloured by correct (green) vs incorrect (red) vs gt-noise
    (gray). Returns (n_correct, n_incorrect, n_noise)."""
    gt = np.asarray(gt)
    pred = np.asarray(pred_aligned)
    is_noise = gt == -1
    correct = (~is_noise) & (gt == pred)
    incorrect = (~is_noise) & (gt != pred)

    if is_noise.any():
        ax.scatter(xy.loc[is_noise, "x0"], xy.loc[is_noise, "x1"],
                   c="lightgray", s=s, alpha=0.4, linewidths=0)
    ax.scatter(xy.loc[correct, "x0"], xy.loc[correct, "x1"],
               c="#2CA02C", s=s, linewidths=0)
    ax.scatter(xy.loc[incorrect, "x0"], xy.loc[incorrect, "x1"],
               c="#D62728", s=s, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return int(correct.sum()), int(incorrect.sum()), int(is_noise.sum())
