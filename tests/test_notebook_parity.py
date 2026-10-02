"""The package reproduces the original notebook's published results.

These run against the real Kaggle file and are skipped when ``data/train.csv``
is not present (for example when testing from the source distribution).

The notebook's linear regression received a sparse design matrix, so
scikit-learn fitted it with the iterative LSQR solver; its numbers are that
solver's result. See docs/DECISIONS.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amesml import ClassificationMetrics, RegressionMetrics, TrainConfig, load_ames, train

DATA = Path(__file__).resolve().parents[1] / "data" / "train.csv"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="data/train.csv is not available")


@pytest.fixture(scope="module")
def ames() -> pd.DataFrame:
    return load_ames(DATA)


def test_dataset_shape(ames: pd.DataFrame) -> None:
    assert ames.shape == (1460, 81)


def test_linear_regression_matches_the_notebook(ames: pd.DataFrame) -> None:
    metrics = train(ames).metrics
    assert isinstance(metrics, RegressionMetrics)
    assert metrics.mae == pytest.approx(21_139.99, abs=0.01)
    assert metrics.rmse == pytest.approx(31_790.77, abs=0.01)
    assert metrics.r2 == pytest.approx(0.8682386, abs=1e-7)
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
