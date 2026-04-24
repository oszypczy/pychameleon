# Benchmarks

Reference and (later) own-implementation runs of CHAMELEON.

## Layout

```
benchmarks/
├── reference_moonpuck/     # Moonpuck/chameleon_cluster output (patched)
│   ├── aggregation/
│   ├── smileface/
│   └── t4_8k/
└── our_impl/               # TODO: our implementation's output
```

Each experiment directory contains:

| File         | Content                                              |
|--------------|------------------------------------------------------|
| `meta.json`  | parameters, timing, cluster stats, commit hash       |
| `labels.csv` | input points + assigned cluster column               |
| `plot.png`   | 2D scatter colored by cluster                        |

## Reference: Moonpuck/chameleon_cluster

Source: <https://github.com/Moonpuck/chameleon_cluster>

Patches applied (see `external/chameleon_cluster_reference/`):

1. **`metis` → `pymetis`** — original wrapper requires system `libmetis`; `pymetis` bundles it. Added a CSR adapter (`_part_graph` in `graphtools.py`) that converts a weighted networkx graph to `xadj/adjncy/eweights` format.
2. **networkx 2.4+ API** — `graph.node[...]` → `graph.nodes[...]` (8 occurrences across `graphtools.py` and `chameleon.py`).

## Summary table

| Dataset       | Points | k | knn | m  | α   | Runtime | Clusters found |
|---------------|--------|---|-----|----|-----|---------|----------------|
| aggregation   |    788 | 7 |  20 | 40 | 2.0 |   9.95 s |              7 |
| smileface     |    644 | 4 |  10 | 20 | 2.0 |   2.08 s |              4 |
| t4_8k         |  8 000 | 6 |  20 | 40 | 2.0 | 674.47 s |              6 |

## Observations (on the reference implementation)

- **Scalability bottleneck**: naive O(n²) k-NN dominates runtime on larger inputs. Swap for KDTree (`scipy.spatial`) in our implementation.
- **No noise handling**: all points get assigned; scattered "noise" points in t4_8k join the nearest cluster instead of being excluded.
- **Runtime warnings** (`RuntimeWarning: Mean of empty slice`) during merging when bisected subgraphs have no internal edges — happens for very small sub-clusters. Needs explicit guard.
- **Parameter `m`** (initial sub-clusters) hard-coded at 40 regardless of `n`. Paper recommends `m ≈ n / (10..20)`.

## Reproducing

From the repo root:

```bash
cd external/chameleon_cluster_reference
.venv/bin/python ../../scripts/run_reference_benchmarks.py          # skip existing
.venv/bin/python ../../scripts/run_reference_benchmarks.py --force  # re-run all
.venv/bin/python ../../scripts/run_reference_benchmarks.py --only aggregation
```

Datasets live in `external/chameleon_cluster_reference/datasets/`; only Aggregation, smileface and t4.8k are shipped with Moonpuck. Extra benchmarks (t5.8k, t7.10k, t8.8k from the original paper) can be pulled from the R `seriation` package.
