"""Per-dataset hyperparameter optimization for CHAMELEON via grid search.

Maximizes ARI vs ground truth over the grid (k_nn, alpha, min_cluster_size).
``n_clusters`` is fixed to the number of classes in ground truth — we do not
search over it. Only datasets that have ground-truth labels are eligible.

Output:

- ``results/hpo_grid.csv`` — full per-run log (same schema as ``sweep_*.csv``,
  so notebooks can read it without changes). Idempotent: re-runs skip rows
  whose ``(dataset, k_nn, alpha, min_cluster_size)`` are already in the file
  unless ``--force`` is passed.
- ``results/hpo_best.json`` — top-1 parameters per dataset with the achieved
  metrics, written at the end.
- Stdout — a final summary table comparing best-found vs current
  ``DEFAULT_PARAMS`` in ``run_experiments.py``.

Usage::

    python scripts/run_hpo.py                          # all eligible datasets
    python scripts/run_hpo.py --only ds1,ds3
    python scripts/run_hpo.py --grid coarse            # smaller 4x4x4 grid
    python scripts/run_hpo.py --force                  # ignore cache
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from tqdm.auto import tqdm

from pychameleon import Chameleon
from pychameleon._datasets import ALL_DATASETS, load

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
MOONPUCK_DIR = REPO / "benchmarks" / "reference_moonpuck"

# Datasets without ground truth in the loader but with Moonpuck reference
# labels — HPO targets ARI vs Moonpuck for these. Map: dataset key → Moonpuck
# subdir under benchmarks/reference_moonpuck/.
MOONPUCK_REFERENCE: dict[str, str] = {
    "aggregation": "aggregation",
    "smileface": "smileface",
    # t4_8k intentionally omitted — it's the same point cloud as ds3, and ds3
    # already has real ground truth, so HPO on t4_8k vs Moonpuck would be
    # redundant.
}

# Current per-dataset defaults — mirrored from run_experiments.py so the
# end-of-run summary can show delta vs status quo. Keep in sync manually.
CURRENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "ds1": {"k_nn": 30, "min_cluster_size": 80, "alpha": 1.0},
    "ds3": {"k_nn": 10, "min_cluster_size": 80, "alpha": 2.5},
    "ds4": {"k_nn": 10, "min_cluster_size": 400, "alpha": 0.5},
    "ds5": {"k_nn": 20, "min_cluster_size": 160, "alpha": 2.0},
}

# Grids. ``min_cluster_size`` values are absolute (points), not fractions —
# Chameleon accepts both but absolute is what the existing DEFAULT_PARAMS use
# for these datasets, and keeps semantics stable across dataset sizes for
# direct comparison.
GRIDS: dict[str, dict[str, list[float]]] = {
    "fine": {
        "k_nn": [5, 10, 15, 20, 30, 50],
        "alpha": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "min_cluster_size": [40, 80, 160, 240, 320, 400],
    },
    "coarse": {
        "k_nn": [10, 20, 30, 50],
        "alpha": [0.5, 1.5, 2.0, 2.5],
        "min_cluster_size": [40, 80, 160, 320],
    },
}

HPO_HEADER = [
    "dataset", "paper_name", "n", "n_clusters_target", "ari_target",
    "k_nn", "min_cluster_size", "alpha",
    "runtime_s", "n_clusters_found",
    "ari", "nmi", "ami", "homogeneity", "completeness", "v_measure",
    "silhouette", "calinski_harabasz", "davies_bouldin",
]


# ---------------------------------------------------------------------------
# CSV upsert (copy of CSVStore from run_experiments.py — kept local so the
# script has no internal-module dependency).
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
            writer.writerow({k: row.get(k, "") for k in header})

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def read_all(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Metrics — same logic as run_experiments.evaluate, kept local.
# ---------------------------------------------------------------------------


def evaluate(X: np.ndarray, labels: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute clustering quality metrics. Ground truth required for ARI/NMI/etc.

    Mirrors run_experiments.evaluate but assumes ``gt is not None`` since HPO
    is only run on datasets with labels.
    """
    out: dict[str, float] = {}
    n_clusters = int(np.unique(labels).size)

    if 2 <= n_clusters < X.shape[0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["silhouette"] = float(silhouette_score(X, labels))
            out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    else:
        out["silhouette"] = float("nan")
        out["calinski_harabasz"] = float("nan")
        out["davies_bouldin"] = float("nan")

    mask = gt >= 0  # drop noise points (Karypis CLUTO uses -1)
    gt_m = gt[mask]
    pred_m = labels[mask]
    if gt_m.size > 0:
        out["ari"] = float(adjusted_rand_score(gt_m, pred_m))
        out["nmi"] = float(normalized_mutual_info_score(gt_m, pred_m))
        out["ami"] = float(adjusted_mutual_info_score(gt_m, pred_m))
        h, c, v = homogeneity_completeness_v_measure(gt_m, pred_m)
        out["homogeneity"] = float(h)
        out["completeness"] = float(c)
        out["v_measure"] = float(v)
    else:
        for k in ("ari", "nmi", "ami", "homogeneity", "completeness", "v_measure"):
            out[k] = float("nan")

    out["n_clusters_found"] = float(n_clusters)
    return out


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def load_moonpuck_labels(name: str) -> np.ndarray | None:
    """Load Moonpuck reference labels for a dataset, or None if unavailable."""
    sub = MOONPUCK_REFERENCE.get(name)
    if sub is None:
        return None
    path = MOONPUCK_DIR / sub / "labels.csv"
    if not path.exists():
        return None
    # CSV format: x, y, cluster — label is the 3rd column (index 2).
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2,), dtype=np.int64)


def resolve_ground_truth(name: str) -> tuple[np.ndarray | None, str]:
    """Return (labels, source) where source ∈ {'gt', 'moonpuck', 'none'}."""
    ds = load(name)
    if ds.y is not None:
        return ds.y, "gt"
    mp = load_moonpuck_labels(name)
    if mp is not None:
        return mp, "moonpuck"
    return None, "none"


def eligible_datasets() -> list[str]:
    """Datasets with either real ground truth or Moonpuck reference labels."""
    return [n for n in ALL_DATASETS if resolve_ground_truth(n)[0] is not None]


def run_hpo(
    datasets: list[str],
    grid: dict[str, list[float]],
    store: CSVStore,
    force: bool,
) -> dict[str, dict[str, Any]]:
    """Grid search per dataset. Returns ``{dataset: best_row}``."""
    done: set[tuple[str, ...]] = set() if force else store.existing_keys()

    # Seed best_per_dataset from any prior runs in the CSV so resuming preserves
    # the current best even if the run that produced it isn't repeated.
    best_per_dataset: dict[str, dict[str, Any]] = {}
    for prior in store.read_all():
        try:
            ari = float(prior.get("ari", "nan"))
        except ValueError:
            ari = float("nan")
        if np.isnan(ari):
            continue
        name = prior["dataset"]
        cur = best_per_dataset.get(name)
        if cur is None or ari > float(cur["ari"]):
            # Back-fill ari_target for rows written before this column existed.
            prior_filled = {**prior, "ari": ari}
            prior_filled.setdefault("ari_target", "gt")
            best_per_dataset[name] = prior_filled

    combos = list(
        itertools.product(grid["k_nn"], grid["alpha"], grid["min_cluster_size"])
    )

    outer = tqdm(datasets, desc="datasets", position=0, leave=True)
    for name in outer:
        ds = load(name)
        gt, source = resolve_ground_truth(name)
        assert gt is not None  # filtered in main()
        n_clusters = int(np.unique(gt[gt >= 0]).size)
        outer.set_postfix_str(
            f"{name} (n={ds.X.shape[0]}, k={n_clusters}, ari→{source})"
        )

        inner = tqdm(
            combos,
            desc=f"  {name}",
            position=1,
            leave=False,
            total=len(combos),
        )
        for k_nn, alpha, mcs in inner:
            key = (name, str(k_nn), str(alpha), str(mcs))
            if key in done:
                # Make sure best tracker reflects skipped row's value too —
                # already handled via the prior-rows seeding above.
                continue

            params = {
                "n_clusters": n_clusters,
                "k_nn": int(k_nn),
                "alpha": float(alpha),
                "min_cluster_size": int(mcs),
            }

            try:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = Chameleon(**params).fit_predict(ds.X)
                elapsed = time.perf_counter() - t0
                metrics = evaluate(ds.X, labels, gt)
            except Exception as e:
                inner.write(f"[fail] {name} k_nn={k_nn} α={alpha} mcs={mcs}: {e}")
                continue

            row = {
                "dataset": name,
                "paper_name": ds.paper_name or "",
                "n": ds.X.shape[0],
                "n_clusters_target": n_clusters,
                "ari_target": source,
                "k_nn": k_nn,
                "alpha": alpha,
                "min_cluster_size": mcs,
                "runtime_s": round(elapsed, 4),
                **metrics,
            }
            store.append(row, HPO_HEADER)

            cur_best = best_per_dataset.get(name)
            if cur_best is None or (
                not np.isnan(metrics["ari"])
                and metrics["ari"] > float(cur_best["ari"])
            ):
                best_per_dataset[name] = {**row, "ari": metrics["ari"]}
                inner.set_postfix(
                    best_ari=f"{metrics['ari']:.3f}",
                    cfg=f"k_nn={k_nn},α={alpha},mcs={mcs}",
                )

        inner.close()
        b = best_per_dataset.get(name)
        if b is not None:
            outer.write(
                f"  → {name}: best ARI = {float(b['ari']):.4f} "
                f"(k_nn={b['k_nn']}, α={b['alpha']}, mcs={b['min_cluster_size']})"
            )
        else:
            outer.write(f"  → {name}: no valid runs")

    outer.close()
    return best_per_dataset


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(best: dict[str, dict[str, Any]]) -> None:
    """Pretty-print top-1 per dataset and delta vs CURRENT_DEFAULTS."""
    print("\n" + "═" * 84)
    print("  HPO RESULTS — best ARI per dataset")
    print("═" * 84)
    header = (
        f"  {'dataset':<12} {'target':<10} {'best ARI':>9}  "
        f"{'k_nn':>5} {'α':>5} {'mcs':>5}  {'Δ ARI':>8}"
    )
    print(header)
    print("  " + "─" * 82)

    # Look up current ARI from comparison.csv for delta.
    comparison_path = RESULTS_DIR / "comparison.csv"
    current_ari: dict[str, float] = {}
    current_ari_mp: dict[str, float] = {}
    if comparison_path.exists():
        with comparison_path.open() as f:
            for row in csv.DictReader(f):
                try:
                    current_ari[row["dataset"]] = float(row["ari"])
                except (KeyError, ValueError):
                    pass
                try:
                    current_ari_mp[row["dataset"]] = float(row["ari_vs_moonpuck"])
                except (KeyError, ValueError):
                    pass

    for name, row in sorted(best.items()):
        ari = float(row["ari"])
        source = str(row.get("ari_target", "gt"))
        baseline = current_ari_mp if source == "moonpuck" else current_ari
        prev = baseline.get(name, float("nan"))
        delta = ari - prev if not np.isnan(prev) else float("nan")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "   n/a"
        print(
            f"  {name:<12} {source:<10} {ari:>9.4f}  "
            f"{int(row['k_nn']):>5} {float(row['alpha']):>5.2f} "
            f"{int(row['min_cluster_size']):>5}  {delta_str:>8}"
        )

    print("═" * 84)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", help="comma-separated dataset keys; default = all with ground truth"
    )
    parser.add_argument(
        "--grid", choices=list(GRIDS), default="fine",
        help="grid density (default: fine = 6×6×6 = 216 combos per dataset)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="ignore existing rows in hpo_grid.csv and rerun everything",
    )
    parser.add_argument(
        "--out", default="hpo_grid.csv",
        help="output CSV name under results/ (default: hpo_grid.csv). "
        "Use a separate file when running with a different ari_target than "
        "an existing CSV to avoid schema mismatch.",
    )
    parser.add_argument(
        "--best", default="hpo_best.json",
        help="output JSON name for best params (default: hpo_best.json)",
    )
    args = parser.parse_args()

    grid = GRIDS[args.grid]
    selected = args.only.split(",") if args.only else eligible_datasets()
    invalid = [n for n in selected if n not in ALL_DATASETS]
    if invalid:
        raise SystemExit(f"unknown dataset(s): {invalid}")

    # Filter to datasets where ARI is computable (real GT or Moonpuck ref).
    unusable = [n for n in selected if resolve_ground_truth(n)[0] is None]
    if unusable:
        print(f"[warn] skipping datasets with no reference labels: {unusable}")
        selected = [n for n in selected if n not in unusable]
    if not selected:
        raise SystemExit("no eligible datasets to optimize")

    sources = {n: resolve_ground_truth(n)[1] for n in selected}
    if any(s == "moonpuck" for s in sources.values()):
        mp_list = [n for n, s in sources.items() if s == "moonpuck"]
        print(f"[info] ARI computed vs Moonpuck reference for: {mp_list}")

    n_combos = (
        len(grid["k_nn"]) * len(grid["alpha"]) * len(grid["min_cluster_size"])
    )
    print(
        f"HPO · grid={args.grid} · "
        f"{len(selected)} dataset(s) × {n_combos} combos = {len(selected) * n_combos} runs"
    )
    print(f"datasets: {', '.join(selected)}")
    print()

    store = CSVStore(
        RESULTS_DIR / args.out,
        key_columns=("dataset", "k_nn", "alpha", "min_cluster_size"),
    )
    if args.force:
        store.clear()

    best = run_hpo(selected, grid, store, force=args.force)

    print(f"\n  best params written to: results/{args.best}")
    print(f"  full log:               results/{args.out}")

    # Persist best params for downstream use (e.g. updating DEFAULT_PARAMS).
    best_path = RESULTS_DIR / args.best
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w") as f:
        json.dump(
            {
                name: {
                    "n_clusters": int(row["n_clusters_target"]),
                    "ari_target": str(row.get("ari_target", "gt")),
                    "k_nn": int(row["k_nn"]),
                    "alpha": float(row["alpha"]),
                    "min_cluster_size": int(row["min_cluster_size"]),
                    "ari": float(row["ari"]),
                    "nmi": float(row.get("nmi", "nan")),
                    "runtime_s": float(row["runtime_s"]),
                }
                for name, row in best.items()
            },
            f,
            indent=2,
        )

    print_summary(best)


if __name__ == "__main__":
    main()
