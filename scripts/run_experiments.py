"""Headless compute runner for pychameleon experiments (Stage 2).

Subcommands:

- ``compare``      pychameleon vs ground-truth (and Moonpuck where available)
                   on all benchmark datasets.
- ``sweep``        parameter sensitivity (k_nn / alpha / min_cluster_size)
                   across datasets with ground truth.
- ``scalability``  runtime as a function of ``n`` (with d=2) and ``d``
                   (with n=2000) using ``sklearn.datasets.make_blobs``.

The runner is **idempotent**: it keeps a CSV per experiment and skips rows
already computed unless ``--force`` is passed. Notebooks in ``notebooks/``
read the CSV files; this script does **not** plot — that's the notebooks'
job.

Usage examples::

    python scripts/run_experiments.py compare
    python scripts/run_experiments.py sweep --param alpha
    python scripts/run_experiments.py scalability --kind n --force
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)

from pychameleon import Chameleon
from pychameleon._datasets import ALL_DATASETS, Dataset, load

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
LABELS_DIR = RESULTS_DIR / "labels"
MOONPUCK_DIR = REPO / "benchmarks" / "reference_moonpuck"


# ---------------------------------------------------------------------------
# Quality metric framework — addresses instructor's feedback on "uniwersalna
# metryka jakości grupowania". We compute the full set of sklearn metrics so
# notebooks can pivot freely without re-fitting.
# ---------------------------------------------------------------------------


def evaluate(
    X: np.ndarray, labels: np.ndarray, ground_truth: np.ndarray | None
) -> dict[str, float]:
    """Compute the full battery of clustering quality metrics.

    Returns a flat dict whose keys are stable across datasets so results stack
    cleanly in CSV / pandas. Metrics that don't apply (e.g. ARI without a
    ground truth, or silhouette with a single cluster) are stored as ``NaN``.
    """
    metrics: dict[str, float] = {}
    n_clusters = len(np.unique(labels))

    # Internal-only metrics: need >=2 clusters and >=2 points per cluster.
    if n_clusters >= 2 and n_clusters < X.shape[0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    else:
        metrics["silhouette"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
        metrics["davies_bouldin"] = float("nan")

    # External metrics: need ground truth. We mask out -1 (noise) labels
    # because CHAMELEON assigns every point to a cluster — comparing those
    # noise points would penalize unfairly.
    if ground_truth is not None:
        mask = ground_truth >= 0
        gt = ground_truth[mask]
        pred = labels[mask]
        if gt.size > 0:
            metrics["ari"] = float(adjusted_rand_score(gt, pred))
            metrics["nmi"] = float(normalized_mutual_info_score(gt, pred))
            metrics["ami"] = float(adjusted_mutual_info_score(gt, pred))
            h, c, v = homogeneity_completeness_v_measure(gt, pred)
            metrics["homogeneity"] = float(h)
            metrics["completeness"] = float(c)
            metrics["v_measure"] = float(v)
        else:
            for k in ("ari", "nmi", "ami", "homogeneity", "completeness", "v_measure"):
                metrics[k] = float("nan")
    else:
        for k in ("ari", "nmi", "ami", "homogeneity", "completeness", "v_measure"):
            metrics[k] = float("nan")

    metrics["n_clusters_found"] = float(n_clusters)
    return metrics


# ---------------------------------------------------------------------------
# CSV I/O — generic upsert helper. Each experiment uses one CSV; rows are
# keyed by a tuple of identifying columns so re-runs can skip what's done.
# ---------------------------------------------------------------------------


@dataclass
class CSVStore:
    path: Path
    key_columns: tuple[str, ...]

    def existing_keys(self) -> set[tuple[str, ...]]:
        if not self.path.exists():
            return set()
        with self.path.open() as f:
            reader = csv.DictReader(f)
            return {tuple(row[k] for k in self.key_columns) for row in reader}

    def append(self, row: dict[str, Any], header: list[str]) -> None:
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if new_file:
                writer.writeheader()
            # Coerce floats and bools to strings via csv (default behavior).
            writer.writerow({k: row.get(k, "") for k in header})

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
# Default parameters per dataset — drawn from Moonpuck benchmarks where
# available; otherwise reasonable defaults near the paper's recommendation.
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    # aggregation: parametry z HPO 2026-05-16 (ARI vs prawdziwy GT z UEF
    # Gionis 2007; results/aggregation_hpo_best.json). ARI=0.961.
    "aggregation": {"n_clusters": 7, "k_nn": 10, "min_cluster_size": 40, "alpha": 0.5},
    # t4_8k: parametry zgodne z Moonpuck benchmark
    # (z benchmarks/reference_moonpuck/t4_8k/meta.json) — zachowane dla
    # uczciwego porównania referencyjnego (ARI vs Moonpuck, sekcja 2 nb 02).
    "t4_8k": {"n_clusters": 6, "k_nn": 20, "min_cluster_size": 200, "alpha": 2.0},
    # smileface: parametry z HPO 2026-05-14 (ARI vs Moonpuck reference;
    # results/hpo_moonpuck.json). Δ ARI vs Moonpuck: +0.28.
    "smileface": {"n_clusters": 4, "k_nn": 50, "min_cluster_size": 160, "alpha": 2.5},
    # DS1/DS3/DS4/DS5: parametry z HPO 2026-05-14 (ARI vs ground truth;
    # results/hpo_best.json). Synchronizowane z notebooks/_helpers.py.
    "ds1": {"n_clusters": 6, "k_nn": 20, "min_cluster_size": 80, "alpha": 0.5},
    "ds3": {"n_clusters": 6, "k_nn": 15, "min_cluster_size": 40, "alpha": 0.5},
    "ds4": {"n_clusters": 9, "k_nn": 10, "min_cluster_size": 160, "alpha": 2.5},
    "ds5": {"n_clusters": 8, "k_nn": 30, "min_cluster_size": 40, "alpha": 3.0},
}


_MOONPUCK_MAP = {
    "aggregation": "aggregation",
    "smileface": "smileface",
    "t4_8k": "t4_8k",
}


def _moonpuck_labels(name: str) -> np.ndarray | None:
    """Return Moonpuck reference labels for a dataset, or ``None`` if absent."""
    key = _MOONPUCK_MAP.get(name)
    if key is None:
        return None
    path = MOONPUCK_DIR / key / "labels.csv"
    if not path.exists():
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2,), dtype=np.int64)


def _moonpuck_runtime(name: str) -> float:
    """Return Moonpuck runtime in seconds from meta.json, or NaN if absent."""
    import json
    key = _MOONPUCK_MAP.get(name)
    if key is None:
        return float("nan")
    path = MOONPUCK_DIR / key / "meta.json"
    if not path.exists():
        return float("nan")
    with path.open() as f:
        meta = json.load(f)
    val = meta.get("runtime_seconds")
    return float(val) if val is not None else float("nan")


def _save_labels(name: str, X: np.ndarray, labels: np.ndarray) -> None:
    """Persist (x_0, ..., x_{d-1}, label) per point for downstream notebooks."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = np.column_stack([X, labels.astype(np.int64)])
    header = ",".join([f"x{i}" for i in range(X.shape[1])] + ["label"])
    np.savetxt(
        LABELS_DIR / f"{name}.csv",
        out,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%.10g"] * X.shape[1] + ["%d"],
    )


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------


COMPARE_HEADER = [
    "dataset", "paper_name", "n", "n_clusters_target",
    "k_nn", "min_cluster_size", "alpha",
    "runtime_s", "runtime_moonpuck_s", "n_clusters_found",
    "ari", "nmi", "ami", "homogeneity", "completeness", "v_measure",
    "silhouette", "calinski_harabasz", "davies_bouldin",
    "ari_vs_moonpuck",
]


def cmd_compare(args: argparse.Namespace) -> None:
    store = CSVStore(RESULTS_DIR / "comparison.csv", key_columns=("dataset",))
    if args.force:
        store.clear()
    done = store.existing_keys()

    selected = args.only.split(",") if args.only else list(ALL_DATASETS)

    for name in selected:
        if (name,) in done:
            print(f"[skip] {name} (already in {store.path.name}; --force to rerun)")
            continue
        ds = load(name)
        params = DEFAULT_PARAMS[name]
        print(f"[run ] {name}: n={ds.X.shape[0]}, params={params}")

        t0 = time.perf_counter()
        labels = Chameleon(**params).fit_predict(ds.X)
        elapsed = time.perf_counter() - t0

        metrics = evaluate(ds.X, labels, ds.y)
        ref = _moonpuck_labels(name)
        ari_ref = (
            float(adjusted_rand_score(ref, labels)) if ref is not None else float("nan")
        )
        _save_labels(name, ds.X, labels)

        row = {
            "dataset": name,
            "paper_name": ds.paper_name or "",
            "n": ds.X.shape[0],
            "n_clusters_target": params["n_clusters"],
            "k_nn": params["k_nn"],
            "min_cluster_size": params["min_cluster_size"],
            "alpha": params["alpha"],
            "runtime_s": round(elapsed, 4),
            "runtime_moonpuck_s": _moonpuck_runtime(name),
            "ari_vs_moonpuck": ari_ref,
            **metrics,
        }
        store.append(row, COMPARE_HEADER)
        print(
            f"[done] {name}: {elapsed:.2f}s, "
            f"ARI={metrics['ari']:.3f}, "
            f"NMI={metrics['nmi']:.3f}, "
            f"Sil={metrics['silhouette']:.3f}"
        )


# ---------------------------------------------------------------------------
# Subcommand: sweep
# ---------------------------------------------------------------------------


SWEEP_GRID: dict[str, list[float]] = {
    "k_nn": [5, 10, 15, 20, 30, 50],
    "alpha": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "min_cluster_size": [0.01, 0.02, 0.025, 0.03, 0.04, 0.05],
}

SWEEP_HEADER = [
    "param", "value", "dataset", "paper_name", "n",
    "k_nn", "min_cluster_size", "alpha",
    "runtime_s", "n_clusters_found",
    "ari", "nmi", "ami", "homogeneity", "completeness", "v_measure",
    "silhouette", "calinski_harabasz", "davies_bouldin",
]


def cmd_sweep(args: argparse.Namespace) -> None:
    if args.param not in SWEEP_GRID:
        raise SystemExit(f"--param must be one of {list(SWEEP_GRID)}")

    store = CSVStore(
        RESULTS_DIR / f"sweep_{args.param}.csv",
        key_columns=("dataset", "value"),
    )
    if args.force:
        store.clear()
    done = store.existing_keys()

    # Sweep on datasets where ground truth is available — that's where ARI
    # tells us something. Internal metrics still computed for the rest.
    selected = args.only.split(",") if args.only else [
        d for d in ALL_DATASETS if load(d).y is not None
    ]

    grid = SWEEP_GRID[args.param]

    for name in selected:
        ds = load(name)
        base = dict(DEFAULT_PARAMS[name])
        for value in grid:
            key = (name, str(value))
            if key in done:
                continue
            params = dict(base)
            params[args.param] = value
            print(f"[run ] sweep[{args.param}={value}] on {name}")

            try:
                t0 = time.perf_counter()
                labels = Chameleon(**params).fit_predict(ds.X)
                elapsed = time.perf_counter() - t0
                metrics = evaluate(ds.X, labels, ds.y)
            except Exception as e:
                print(f"[skip] {name} {args.param}={value}: {e}")
                continue

            row = {
                "param": args.param,
                "value": value,
                "dataset": name,
                "paper_name": ds.paper_name or "",
                "n": ds.X.shape[0],
                "k_nn": params["k_nn"],
                "min_cluster_size": params["min_cluster_size"],
                "alpha": params["alpha"],
                "runtime_s": round(elapsed, 4),
                **metrics,
            }
            store.append(row, SWEEP_HEADER)
            print(
                f"[done] {name} {args.param}={value}: "
                f"ARI={metrics['ari']:.3f}, runtime={elapsed:.2f}s"
            )


# ---------------------------------------------------------------------------
# Subcommand: scalability
# ---------------------------------------------------------------------------


SCALABILITY_HEADER = [
    "kind", "n", "d", "n_clusters_target", "repeat", "k_nn",
    "runtime_s", "n_clusters_found",
    "silhouette", "calinski_harabasz", "davies_bouldin",
    "ari", "nmi",
]

SCALABILITY_GRID: dict[str, list[int]] = {
    "n": [500, 1000, 2000, 5000, 10000, 20000, 50000],
    "d": [2, 5, 10, 20, 50],
}


def _make_scalability_dataset(n: int, d: int, seed: int) -> Dataset:
    """Synthetic dataset for scaling experiments. 5 isotropic blobs."""
    X, y = make_blobs(
        n_samples=n, n_features=d, centers=5, cluster_std=1.0, random_state=seed
    )
    return Dataset(name=f"blobs_n{n}_d{d}_s{seed}", X=X.astype(np.float64), y=y.astype(np.int64))


def cmd_scalability(args: argparse.Namespace) -> None:
    store = CSVStore(
        RESULTS_DIR / "scalability.csv",
        key_columns=("kind", "n", "d", "repeat"),
    )
    if args.force:
        store.clear()
    done = store.existing_keys()

    repeats = args.repeats
    kinds = ["n", "d"] if args.kind == "all" else [args.kind]

    for kind in kinds:
        for value in SCALABILITY_GRID[kind]:
            for r in range(repeats):
                if kind == "n":
                    n, d = int(value), 2
                else:
                    n, d = 2000, int(value)
                key = (kind, str(n), str(d), str(r))
                if key in done:
                    continue
                ds = _make_scalability_dataset(n, d, seed=r)
                # k_nn stays modest to keep small-n cases valid.
                k_nn = min(20, max(5, n // 20))
                # Fairly aggressive min_cluster_size for synthetic blobs —
                # the goal here is timing, not optimal clustering.
                params: dict[str, Any] = {
                    "n_clusters": 5,
                    "k_nn": k_nn,
                    "min_cluster_size": max(20, n // 50),
                    "alpha": 2.0,
                }

                print(f"[run ] scalability[{kind}={value} rep={r}]: n={n}, d={d}")
                t0 = time.perf_counter()
                labels = Chameleon(**params).fit_predict(ds.X)
                elapsed = time.perf_counter() - t0

                metrics = evaluate(ds.X, labels, ds.y)
                row = {
                    "kind": kind,
                    "n": n,
                    "d": d,
                    "n_clusters_target": params["n_clusters"],
                    "repeat": r,
                    "k_nn": k_nn,
                    "runtime_s": round(elapsed, 4),
                    **{k: metrics.get(k, "") for k in (
                        "n_clusters_found", "silhouette", "calinski_harabasz",
                        "davies_bouldin", "ari", "nmi"
                    )},
                }
                store.append(row, SCALABILITY_HEADER)
                print(f"[done] {elapsed:.2f}s, ARI={metrics['ari']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_compare = subparsers.add_parser("compare", help="pychameleon on all datasets")
    p_compare.add_argument("--only", help="comma-separated dataset keys")
    p_compare.add_argument("--force", action="store_true", help="rerun even if done")
    p_compare.set_defaults(func=cmd_compare)

    p_sweep = subparsers.add_parser("sweep", help="parameter sensitivity")
    p_sweep.add_argument(
        "--param", required=True, choices=list(SWEEP_GRID), help="parameter to sweep"
    )
    p_sweep.add_argument("--only", help="comma-separated dataset keys")
    p_sweep.add_argument("--force", action="store_true", help="rerun even if done")
    p_sweep.set_defaults(func=cmd_sweep)

    p_scale = subparsers.add_parser("scalability", help="runtime vs n / vs d")
    p_scale.add_argument("--kind", choices=["n", "d", "all"], default="all")
    p_scale.add_argument("--repeats", type=int, default=3)
    p_scale.add_argument("--force", action="store_true", help="rerun even if done")
    p_scale.set_defaults(func=cmd_scalability)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
