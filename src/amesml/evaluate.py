"""Evaluation metrics for the regression and classification tasks.

Both functions take true and predicted values (array-likes or Series) rather
than a model, so they work for any predictions - including ones loaded from a
file - and are easy to check by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ArrayLike = Union[Sequence[float], NDArray[Any], "pd.Series[Any]"]


@dataclass(frozen=True)
class RegressionMetrics:
    """Error of predicted sale prices against the true prices.

    Attributes:
        mae: Mean absolute error, in dollars.
        rmse: Root mean squared error, in dollars. Penalises large misses.
        r2: Coefficient of determination; 1.0 is a perfect fit.
        mape: Mean absolute percentage error as a fraction (0.12 means 12%).
        n_samples: Number of predictions evaluated.
    """

    mae: float
    rmse: float
    r2: float
    mape: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a JSON-serialisable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Quality of above/below-threshold predictions.

    Class 1 means "sells at or above the threshold".

    Attributes:
        accuracy: Fraction of houses classified correctly.
        precision: Of houses predicted above the threshold, the fraction that were.
        recall: Of houses above the threshold, the fraction predicted so.
        f1: Harmonic mean of precision and recall.
        roc_auc: Area under the ROC curve, or ``None`` when no scores were given
            or the true labels contain only one class.
        confusion_matrix: ``[[TN, FP], [FN, TP]]``.
        n_samples: Number of predictions evaluated.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a JSON-serialisable dictionary."""
        data = asdict(self)
        data["confusion_matrix"] = [list(row) for row in self.confusion_matrix]
        return data


def evaluate_regressor(y_true: ArrayLike, y_pred: ArrayLike) -> RegressionMetrics:
    """Compute regression metrics.

    Args:
        y_true: Actual sale prices.
        y_pred: Predicted sale prices.

    Returns:
        MAE, RMSE, R² and MAPE.

    Raises:
        ValueError: The inputs are empty or of different lengths.
    """
    truth, predicted = _as_float_arrays(y_true, y_pred)
    return RegressionMetrics(
        mae=float(mean_absolute_error(truth, predicted)),
        # mean_squared_error(squared=False) was removed in scikit-learn 1.6 and
        # root_mean_squared_error only arrived in 1.4, so take the root here.
        rmse=float(np.sqrt(mean_squared_error(truth, predicted))),
        r2=float(r2_score(truth, predicted)),
        mape=float(mean_absolute_percentage_error(truth, predicted)),
        n_samples=int(truth.size),
    )


def evaluate_classifier(
    y_true: ArrayLike, y_pred: ArrayLike, y_score: ArrayLike | None = None
) -> ClassificationMetrics:
    """Compute binary classification metrics.

    Args:
        y_true: Actual labels (1 = at or above the threshold, 0 = below).
        y_pred: Predicted labels.
        y_score: Predicted probability of class 1, used for ROC-AUC.

    Returns:
        Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.

    Raises:
        ValueError: The inputs are empty, of different lengths, or contain
            labels other than 0 and 1.
    """
    truth, predicted = _as_label_arrays(y_true, y_pred)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    (tn, fp), (fn, tp) = matrix.tolist()

    roc_auc: float | None = None
    if y_score is not None and np.unique(truth).size == 2:
        scores = np.asarray(y_score, dtype=float).ravel()
        if scores.size != truth.size:
            raise ValueError("y_score must have the same length as y_true")
        roc_auc = float(roc_auc_score(truth, scores))

    return ClassificationMetrics(
        accuracy=float(accuracy_score(truth, predicted)),
        precision=float(precision_score(truth, predicted, zero_division=0)),
        recall=float(recall_score(truth, predicted, zero_division=0)),
        f1=float(f1_score(truth, predicted, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=((int(tn), int(fp)), (int(fn), int(tp))),
        n_samples=int(truth.size),
    )


def _as_float_arrays(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    truth = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    _check_lengths(truth, predicted)
    return truth, predicted


def _as_label_arrays(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    truth = np.asarray(y_true).ravel()
    predicted = np.asarray(y_pred).ravel()
    _check_lengths(truth, predicted)
    for name, values in (("y_true", truth), ("y_pred", predicted)):
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must contain only the labels 0 and 1")
    return truth.astype(np.int64), predicted.astype(np.int64)


def _check_lengths(truth: NDArray[Any], predicted: NDArray[Any]) -> None:
    if truth.size == 0:
        raise ValueError("Cannot evaluate an empty set of predictions")
    if truth.size != predicted.size:
        raise ValueError(f"y_true and y_pred differ in length ({truth.size} vs {predicted.size})")
