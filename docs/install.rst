Installation
============

The package is published on PyPI as ``chameleon-cluster``; the import name is
``pychameleon``.

.. code-block:: bash

   pip install chameleon-cluster
   # or
   uv add chameleon-cluster

Requirements
------------

- Python 3.11, 3.12 or 3.13
- NumPy, SciPy, scikit-learn ≥ 1.6
- `pymetis <https://github.com/inducer/pymetis>`_ ≥ 2025.2 (Python bindings
  for the METIS graph partitioner)

Linux and macOS are supported out of the box: pymetis provides binary wheels
for both.

Windows
-------

pymetis publishes no Windows wheels on PyPI, so ``pip`` would try to compile
METIS from source and fail. Install pymetis from conda-forge first, then the
package itself:

.. code-block:: bash

   conda install -c conda-forge pymetis
   pip install chameleon-cluster

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/oszypczy/pychameleon.git
   cd pychameleon
   uv sync --all-extras

   uv run pytest           # tests + coverage
   uv run ruff check .     # lint
   uv run mypy src/        # type check
