"""The package reproduces the original notebook's published results closely.

These run against the real Kaggle file and are skipped when ``data/train.csv``
is not present (for example when testing from the source distribution).

The notebook's linear regression received a sparse design matrix, so
scikit-learn fitted it with the iterative LSQR solver; its numbers are that
solver's result. See docs/DECISIONS.md (D1 and D2).

Because that design is ill-conditioned and rank-deficient, where LSQR stops
depends on the scikit-learn, scipy and BLAS build in use, so the regression
figures reproduce *exactly* only on Windows with Python 3.11-3.13. Elsewhere
they land within a few percent: the spread observed across the CI matrix is
MAE 20,230.73 to 21,139.99, about 4.3%. The reference values below stay the
notebook's published numbers; they are asserted with a relative tolerance
wide enough for that spread and still tight enough to catch a real
regression. The split itself is seeded and identical everywhere - swapping
only the solver moves MAE by about 250 on the same 292 test rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amesml import ClassificationMetrics, RegressionMetrics, TrainConfig, load_ames, train

DATA = Path(__file__).resolve().parents[1] / "data" / "train.csv"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="data/train.csv is not available")

#: Relative tolerance for the notebook's regression figures. The observed
#: cross-platform spread is about 4.3%; 6% leaves margin without letting a
#: real regression through (ridge, lasso and gradient boosting all differ by
#: far more than this from the notebook's linear baseline).
NOTEBOOK_RTOL = 0.06


@pytest.fixture(scope="module")
def ames() -> pd.DataFrame:
    return load_ames(DATA)


def test_dataset_shape(ames: pd.DataFrame) -> None:
    assert ames.shape == (1460, 81)


def test_linear_regression_matches_the_notebook(ames: pd.DataFrame) -> None:
    # Reference values are the notebook's published figures. They reproduce to
    # the cent only on Windows with Python 3.11-3.13; see this module's
    # docstring and docs/DECISIONS.md D1 for why the tolerance is relative.
    metrics = train(ames).metrics
    assert isinstance(metrics, RegressionMetrics)
    assert metrics.mae == pytest.approx(21_139.99, rel=NOTEBOOK_RTOL)
    assert metrics.rmse == pytest.approx(31_790.77, rel=NOTEBOOK_RTOL)
    assert metrics.r2 == pytest.approx(0.8682386, rel=NOTEBOOK_RTOL)
    assert metrics.n_samples == 292


def test_logistic_regression_matches_the_notebook(ames: pd.DataFrame) -> None:
    result = train(ames, TrainConfig(task="classification"))
    metrics = result.metrics
    assert isinstance(metrics, ClassificationMetrics)
    assert result.pipeline.threshold_ == 165_000.0
    assert metrics.confusion_matrix == ((152, 10), (6, 124))
    assert round(metrics.accuracy, 4) == 0.9452
    assert round(metrics.precision, 4) == 0.9254
    assert round(metrics.recall, 4) == 0.9538
    assert round(metrics.f1, 4) == 0.9394


def test_test_split_contains_categories_unseen_in_training(ames: pd.DataFrame) -> None:
    assert train(ames).unseen_categories == {"RoofMatl": ["Membran"], "Electrical": ["Mix"]}
