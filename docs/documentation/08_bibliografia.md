# 8. Bibliografia

## Artykuły naukowe

[1] **Karypis G., Han E.-H., Kumar V.** *CHAMELEON: A hierarchical clustering
    algorithm using dynamic modeling.* IEEE Computer: Special Issue on Data Analysis
    and Mining, vol. 32, no. 8, 1999, str. 68–75.
    [główny artykuł źródłowy projektu — `docs/chameleon_karypis_1999.pdf`]

[2] **Karypis G., Kumar V.** *A fast and high quality multilevel scheme for
    partitioning irregular graphs.* SIAM Journal on Scientific Computing, vol. 20,
    no. 1, 1998, str. 359–392.
    [paper opisujący METIS — algorytm bisekcji wykorzystywany w Fazie I — `docs/metis_karypis_1998.pdf`]

[3] **Gionis A., Mannila H., Tsaparas P.** *Clustering aggregation.* ACM Transactions
    on Knowledge Discovery from Data (TKDD), vol. 1, no. 1, 2007.
    [pochodzenie zbioru Aggregation.csv]

[4] **Bhattacharya A., Bhasi A.** *Chameleon2: An Improved Graph-Based Clustering
    Algorithm.* ACM TKDD, 2019.
    [ulepszenie CHAMELEON — uwzględniane jako roadmap v0.3]

[5] **Karypis G., Han E.-H., Kumar V.** *CURE: An efficient clustering algorithm for
    large databases.* ACM SIGMOD, 1998. [poprzednik CHAMELEON-a — kontekst rozdz. 1]

[6] **Guha S., Rastogi R., Shim K.** *ROCK: A Robust Clustering Algorithm for
    Categorical Attributes.* IEEE ICDE, 1999. [poprzednik CHAMELEON-a — kontekst rozdz. 1]

[7] **Ester M., Kriegel H.-P., Sander J., Xu X.** *A density-based algorithm for
    discovering clusters in large spatial databases with noise.* KDD, 1996. [DBSCAN]

[8] **McInnes L., Healy J., Astels S.** *hdbscan: Hierarchical density-based clustering.*
    Journal of Open Source Software, vol. 2, no. 11, 2017, str. 205.
    [HDBSCAN — wzór projektowy dla pakietu pychameleon]

## Wymagania projektowe

[9] **Bembenik R.** *Projekt MED 2026L: Wymagania i opis.* Politechnika Warszawska,
    semestr letni 2026. [`docs/Projekt MED_2026l_rb.pdf`]

## Implementacje referencyjne

[10] **Moonpuck.** *chameleon_cluster — Python implementation of CHAMELEON algorithm.*
    Repozytorium GitHub, commit `1c0a65ee6a79706e4d415dd7ca78da5d3c29906d`.
    <https://github.com/Moonpuck/chameleon_cluster>
    [implementacja referencyjna używana do walidacji — patche w `scripts/setup_reference.sh`]

## Biblioteki i narzędzia

[11] **Harris C.R., Millman K.J., van der Walt S.J. et al.** *Array programming with
    NumPy.* Nature, vol. 585, 2020, str. 357–362. [biblioteka numpy]

[12] **Virtanen P., Gommers R., Oliphant T.E. et al.** *SciPy 1.0: fundamental algorithms
    for scientific computing in Python.* Nature Methods, vol. 17, 2020, str. 261–272.
    [`scipy.spatial.KDTree` — Faza 0]

[13] **Pedregosa F., Varoquaux G., Gramfort A. et al.** *Scikit-learn: Machine learning
    in Python.* JMLR, vol. 12, 2011, str. 2825–2830.
    [interfejs `BaseEstimator`/`ClusterMixin`/`check_estimator`]

[14] **Klockner A.** *PyMETIS: Python wrapper for METIS graph partitioning library.*
    Repozytorium PyPI: <https://pypi.org/project/PyMetis/>
    [wrapper METIS używany w Fazie I]

[15] **van Rossum G., Lehtosalo J., Langa Ł.** *PEP 484: Type Hints.* Python Enhancement
    Proposal, 2014.
    [system type hints używany w `_types.py`]

[16] **Astral.** *uv: An extremely fast Python package and project manager.*
    <https://github.com/astral-sh/uv>
    [menedżer pakietów używany w projekcie]

[17] **Astral.** *ruff: An extremely fast Python linter and code formatter.*
    <https://github.com/astral-sh/ruff>
    [linter w pre-commit i CI]

[18] **Lehtosalo J. et al.** *mypy: Optional static typing for Python.*
    <http://mypy-lang.org/>
    [type checker w trybie strict]
