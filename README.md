# pychameleon

Implementation of the CHAMELEON hierarchical clustering algorithm ([Karypis, Han, Kumar, 1999](https://glaros.dtc.umn.edu/gkhome/fetch/papers/chameleonCOMPUTER99.pdf)) with a scikit-learn compatible API.

CHAMELEON discovers clusters of arbitrary shapes, densities, and sizes by using a two-phase approach:

1. **Phase I**: partition a k-nearest-neighbor graph into many small sub-clusters via [METIS](https://www.cs.umn.edu/~metis).
2. **Phase II**: agglomeratively merge sub-clusters using **dynamic modeling** — decisions consider each cluster's internal inter-connectivity and closeness, not a fixed global threshold.

## Installation

```bash
uv add pychameleon        # with uv
# or
pip install pychameleon   # with pip
```

## Quick start

```python
import numpy as np
from pychameleon import Chameleon

X = np.random.rand(500, 2)

model = Chameleon(n_clusters=5, k_nn=10, min_cluster_size=0.025, alpha=2.0)
labels = model.fit_predict(X)

print(f"Found {model.n_clusters_} clusters, labels: {np.unique(labels)}")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | `8` | Target number of final clusters |
| `k_nn` | `10` | Number of neighbors in the k-NN graph |
| `min_cluster_size` | `0.025` | Minimum Phase-I sub-cluster size (fraction of `n` if `< 1`, else absolute count) |
| `alpha` | `2.0` | Exponent for relative closeness in the merge score: `RI · RC^alpha` |

## Development

```bash
git clone https://github.com/oszypczy/pychameleon.git
cd pychameleon
uv sync --all-extras

uv run pytest           # tests + coverage
uv run ruff check .     # lint
uv run mypy src/        # type check
```

## References

- Karypis, G., Han, E.-H., & Kumar, V. (1999). *Chameleon: hierarchical clustering using dynamic modeling*. Computer, 32(8), 68–75.
- Karypis, G., & Kumar, V. (1998). *A fast and high quality multilevel scheme for partitioning irregular graphs*. SIAM J. Sci. Comput., 20(1), 359–392.

## License

MIT
