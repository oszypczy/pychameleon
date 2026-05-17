# 8. Bibliografia

[1] **Karypis G., Han E.-H., Kumar V.** *CHAMELEON: A hierarchical clustering algorithm
    using dynamic modeling.* IEEE Computer 32(8), 1999, str. 68–75.
    [`docs/chameleon_karypis_1999.pdf`]

[2] **Karypis G., Kumar V.** *A fast and high quality multilevel scheme for partitioning
    irregular graphs.* SIAM J. Sci. Comput. 20(1), 1998, str. 359–392. [METIS;
    `docs/metis_karypis_1998.pdf`]

[3] **Bembenik R.** *Projekt MED 2026L: Wymagania i opis.* Politechnika Warszawska, 2026.
    [`docs/Projekt MED_2026l_rb.pdf`]

[4] **Moonpuck.** *chameleon_cluster — Python implementation of CHAMELEON.* Repozytorium
    GitHub, commit `1c0a65ee6a79706e4d415dd7ca78da5d3c29906d`.
    <https://github.com/Moonpuck/chameleon_cluster>

[5] **Gionis A., Mannila H., Tsaparas P.** *Clustering aggregation.* ACM TKDD 1(1), 2007.
    [pochodzenie zbioru Aggregation.csv]

[6] **Pedregosa F. et al.** *Scikit-learn: Machine learning in Python.* JMLR 12, 2011,
    str. 2825–2830. [`BaseEstimator` / `ClusterMixin` / `check_estimator`]

[7] **Virtanen P. et al.** *SciPy 1.0: Fundamental algorithms for scientific computing
    in Python.* Nature Methods 17, 2020, str. 261–272. [`scipy.spatial.KDTree`]

[8] **Klockner A.** *PyMETIS: Python wrapper for METIS.* PyPI, <https://pypi.org/project/PyMetis/>.

[9] **University of Eastern Finland.** *Clustering benchmark datasets.* Joensuu, 2008–2024.
    <https://cs.joensuu.fi/sipu/datasets/> [ground truth dla zbioru Aggregation]

[10] **Karypis G.** *CLUTO — Clustering Software.* Univ. of Minnesota, 1999.
    [zbiory DS1/DS3/DS4/DS5 — `t5.8k`, `t4.8k`, `t7.10k`, `t8.8k`]

## Kod źródłowy

Pełen kod źródłowy pakietu znajduje się w repozytorium:

<https://github.com/oszypczy/pychameleon>

Struktura:

```
src/pychameleon/      kod produkcyjny (5 modułów + types + datasets)
tests/                36 testów jednostkowych + e2e + sklearn-compat
scripts/              run_experiments.py, run_hpo.py, export_figures.py
notebooks/            3 notebooki analityczne
benchmarks/           reference_moonpuck — wyniki implementacji ref.
results/              CSV z wynikami eksperymentów (8 plików)
docs/                 niniejsza dokumentacja
```

Repozytorium zawiera kompletny zestaw skryptów do reprodukcji wszystkich wyników z rozdziału 6 — patrz rozdz. 4.4.
