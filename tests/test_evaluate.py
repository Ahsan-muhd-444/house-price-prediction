from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from amesml import evaluate_classifier, evaluate_regressor


def test_regression_metrics_by_hand() -> None:
    # Residuals -10, 10, -30, 40. MAE = 90 / 4; MSE = (100 + 100 + 900 + 1600) / 4 = 675;
    # R² = 1 - 2700 / 50000; MAPE = mean(0.10, 0.05, 0.10, 0.10).
    metrics = evaluate_regressor([100, 200, 300, 400], [110, 190, 330, 360])
    assert metrics.mae == pytest.approx(22.5)
    assert metrics.rmse == pytest.approx(math.sqrt(675))
    assert metrics.r2 == pytest.approx(0.946)
    assert metrics.mape == pytest.approx(0.0875)
    assert metrics.n_samples == 4


def test_regression_accepts_series_and_arrays() -> None:
    metrics = evaluate_regressor(pd.Series([1.0, 2.0]), np.array([1.0, 2.0]))
    assert (metrics.mae, metrics.rmse, metrics.r2) == (0.0, 0.0, 1.0)


def test_classification_metrics_by_hand() -> None:
    y_true = [1, 1, 1, 0, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0]  # TP 3, FN 1, FP 1, TN 3
    scores = [0.9, 0.4, 0.8, 0.2, 0.6, 0.1, 0.7, 0.3]  # 15 of 16 pairs ranked correctly
    metrics = evaluate_classifier(y_true, y_pred, scores)
    assert metrics.confusion_matrix == ((3, 1), (1, 3))
    assert metrics.accuracy == metrics.precision == metrics.recall == metrics.f1 == 0.75
    assert metrics.roc_auc == pytest.approx(15 / 16)


def test_roc_auc_needs_scores_and_both_classes() -> None:
    assert evaluate_classifier([1, 0], [1, 0]).roc_auc is None
    assert evaluate_classifier([1, 1], [1, 0], [0.9, 0.2]).roc_auc is None


def test_no_predicted_positives_gives_zero_precision_without_warning() -> None:
    metrics = evaluate_classifier([1, 0, 1], [0, 0, 0])
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: evaluate_regressor([], []), "empty"),
        (lambda: evaluate_regressor([1, 2], [1]), "differ in length"),
        (lambda: evaluate_classifier([0, 2], [0, 1]), "only the labels 0 and 1"),
        (lambda: evaluate_classifier([0, 1], [0, 1], [0.5]), "same length"),
    ],
)
def test_invalid_inputs(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_to_dict_is_json_serialisable() -> None:
    classification = evaluate_classifier([1, 0], [1, 0], [0.8, 0.1]).to_dict()
    assert classification["confusion_matrix"] == [[1, 0], [0, 1]]
    regression = evaluate_regressor([1.0, 2.0], [1.5, 2.5]).to_dict()
    json.dumps([classification, regression])
