"""Run Moonpuck CHAMELEON on all benchmark datasets and save results.

Produces for each dataset in benchmarks/reference_moonpuck/<name>/:
    meta.json    - parameters, timing, cluster stats
    labels.csv   - input points + assigned cluster
    plot.png     - 2D scatter colored by cluster

Usage:
    python scripts/run_reference_benchmarks.py              # skip existing
    python scripts/run_reference_benchmarks.py --force      # re-run all
    python scripts/run_reference_benchmarks.py --only t4_8k # single dataset
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
REFERENCE_SRC = REPO / "external" / "chameleon_cluster_reference"
OUTPUT_ROOT = REPO / "benchmarks" / "reference_moonpuck"

sys.path.insert(0, str(REFERENCE_SRC))
from chameleon import cluster  # noqa: E402

BENCHMARKS = {
    "aggregation": {
        "csv": "Aggregation.csv",
        "sep": " ",
        "params": {"k": 7, "knn": 20, "m": 40, "alpha": 2.0},
        "note": "7 ground-truth clusters; classic chain-bridged benchmark",
    },
    "smileface": {
        "csv": "smileface.csv",
        "sep": ",",
        "params": {"k": 4, "knn": 10, "m": 20, "alpha": 2.0},
        "note": "Face shape: eyebrows, eyes, mouth",
    },
    "t4_8k": {
        "csv": "t4.8k.csv",
        "sep": " ",
        "params": {"k": 6, "knn": 20, "m": 40, "alpha": 2.0},
        "note": "Original CHAMELEON paper benchmark; letter shapes + noise",
    },
}


def reference_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REFERENCE_SRC, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_one(name: str, spec: dict, force: bool) -> None:
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    if meta_path.exists() and not force:
        print(f"[skip] {name} (already exists; use --force to re-run)")
        return

    csv_path = REFERENCE_SRC / "datasets" / spec["csv"]
    df = pd.read_csv(csv_path, sep=spec["sep"], header=None)
    print(f"[run ] {name}: {df.shape[0]} points, params={spec['params']}")

    t0 = time.time()
    res = cluster(df.copy(), plot=False, **spec["params"])
    elapsed = time.time() - t0

    res.to_csv(out_dir / "labels.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(res[0], res[1], c=res["cluster"], cmap="tab10", s=10)
    ax.set_title(f"{name} - Moonpuck CHAMELEON {spec['params']} - {elapsed:.1f}s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_dir / "plot.png", dpi=100)
    plt.close(fig)

    cluster_sizes = {int(k): int(v) for k, v in res["cluster"].value_counts().sort_index().items()}
    meta = {
        "dataset": name,
        "source_csv": str(csv_path.relative_to(REPO)),
        "n_points": int(df.shape[0]),
        "n_dims": int(df.shape[1]),
        "parameters": spec["params"],
        "runtime_seconds": round(elapsed, 2),
        "n_clusters_found": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "note": spec["note"],
        "reference_impl": "Moonpuck/chameleon_cluster",
        "reference_commit": reference_commit(),
        "patches_applied": [
            "metis -> pymetis (with networkx-to-CSR adapter)",
            "networkx 2.4+ API: graph.node -> graph.nodes",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[done] {name}: {elapsed:.1f}s, {len(cluster_sizes)} clusters -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-run even if results exist")
    parser.add_argument("--only", help="run only this dataset (e.g. 'aggregation')")
    args = parser.parse_args()

    targets = {args.only: BENCHMARKS[args.only]} if args.only else BENCHMARKS
    for name, spec in targets.items():
        run_one(name, spec, args.force)


if __name__ == "__main__":
    main()
