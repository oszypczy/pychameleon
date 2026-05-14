"""scikit-learn compatibility tests.

Currently skipped because ``Chameleon.fit`` isn't implemented yet. Once the
internals work, remove the skip marker — ``check_estimator`` runs ~100 checks
verifying parameter handling, cloning, pickling, input validation, labels
shape, dtype, and sklearn conventions.

A passing :func:`sklearn.utils.estimator_checks.check_estimator` is the
prerequisite for any scikit-learn-contrib PR.
"""
from __future__ import annotations

from pychameleon import Chameleon


def test_check_estimator() -> None:
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(Chameleon())
