"""Figures for the README and documentation.

Needs matplotlib: ``pip install "amesml[plots]"``. Figures are drawn with
matplotlib's object-oriented API and never touch ``pyplot``, so these functions
keep no global state and work on headless servers without choosing a backend.

Every function writes a PNG and returns its path. :func:`generate_all` rebuilds
the full set from the raw data, training whatever models the figures need.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .config import TrainConfig
from .data import PathLike, train_test_split_df
from .evaluate import ArrayLike
from .models import predict, prepare_input, suppress_unknown_category_warnings, train

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from sklearn.pipeline import Pipeline

#: File names written by :func:`generate_all`. The prefix keeps them apart from
#: the notebook's original figures in ``images/``.
FIGURE_FILES: dict[str, str] = {
    "target_distribution": "amesml_target_distribution.png",
    "missing_values": "amesml_missing_values.png",
    "residuals": "amesml_residuals.png",
    "confusion_matrix": "amesml_confusion_matrix.png",
    "feature_importance": "amesml_feature_importance.png",
}

_ACCENT = "#2b6cb0"
_DPI = 120


def plot_target_distribution(
    frame: pd.DataFrame, path: PathLike, *, target: str = "SalePrice"
) -> Path:
    """Histogram of sale prices with the mean and median marked."""
    prices = frame[target].dropna()
    figure = _figure(9, 5)
    axes = figure.subplots()
    axes.hist(prices, bins=40, color=_ACCENT, alpha=0.85, edgecolor="white")
    mean, median = float(prices.mean()), float(prices.median())
    axes.axvline(mean, color="#c53030", linestyle="--", linewidth=2, label=f"Mean ${mean:,.0f}")
    axes.axvline(
        median, color="#2f855a", linestyle="-.", linewidth=2, label=f"Median ${median:,.0f}"
    )
    axes.set(title="Distribution of sale prices", xlabel="Sale price ($)", ylabel="Houses")
    axes.legend()
    return _save(figure, path)


def plot_missing_values(frame: pd.DataFrame, path: PathLike, *, top_n: int = 20) -> Path:
    """Horizontal bar chart of the columns with the most missing values."""
    share = frame.isna().mean().mul(100).sort_values(ascending=False)
    share = share[share > 0].head(top_n).iloc[::-1]
    figure = _figure(9, max(3.0, 0.32 * len(share) + 1.2))
    axes = figure.subplots()
    if share.empty:
        axes.text(0.5, 0.5, "No missing values", ha="center", va="center")
        axes.set_axis_off()
    else:
        axes.barh(share.index.astype(str), share.to_numpy(), color=_ACCENT)
        axes.set(xlabel="Missing (%)", xlim=(0, 100))
    axes.set_title("Missing values by column")
    return _save(figure, path)


def plot_residuals(y_true: ArrayLike, y_pred: ArrayLike, path: PathLike) -> Path:
    """Residuals against predicted price, next to the residual distribution."""
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = truth - predicted
    figure = _figure(12, 4.8)
    scatter, histogram = figure.subplots(1, 2)
    scatter.scatter(predicted, residuals, s=12, alpha=0.6, color=_ACCENT)
    scatter.axhline(0, color="#4a5568", linestyle="--", linewidth=1.5)
    scatter.set(
        title="Residuals vs predicted",
        xlabel="Predicted price ($)",
        ylabel="Actual - predicted ($)",
    )
    histogram.hist(residuals, bins=40, color=_ACCENT, alpha=0.85, edgecolor="white")
    histogram.axvline(0, color="#4a5568", linestyle="--", linewidth=1.5)
    histogram.set(title="Residual distribution", xlabel="Actual - predicted ($)", ylabel="Houses")
    return _save(figure, path)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    path: PathLike,
    *,
    labels: tuple[str, str] = ("Below threshold", "At/above threshold"),
) -> Path:
    """Annotated 2x2 confusion matrix, laid out as ``[[TN, FP], [FN, TP]]``."""
    counts = np.asarray(matrix, dtype=int)
    figure = _figure(6, 5)
    axes = figure.subplots()
    image = axes.imshow(counts, cmap="Blues")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    threshold = counts.max() / 2 if counts.size else 0
    for (row, column), value in np.ndenumerate(counts):
        colour = "white" if value > threshold else "#1a202c"
        axes.text(column, row, f"{value:d}", ha="center", va="center", fontsize=14, color=colour)
    axes.set_xticks([0, 1], [f"Predicted\n{labels[0]}", f"Predicted\n{labels[1]}"])
    axes.set_yticks([0, 1], [f"Actual\n{labels[0]}", f"Actual\n{labels[1]}"])
    axes.set_title("Price-class confusion matrix")
    return _save(figure, path)


def plot_feature_importance(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    path: PathLike,
    *,
    target: str = "SalePrice",
    top_n: int = 15,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Path:
    """Permutation importance of the raw input columns.

    Each column of ``frame`` is shuffled in turn and the drop in the model's
    score (R² for regression) is measured. Importances are reported per raw
    Ames column rather than per one-hot level, which keeps them readable, and
    the method works for every model family - histogram gradient boosting has
    no built-in ``feature_importances_``.
    """
    from sklearn.inspection import permutation_importance

    inputs = prepare_input(pipeline, frame)
    truth = frame[target].astype(float)
    if pipeline.task_ == "classification":
        truth = (truth >= pipeline.threshold_).astype(int)
    with suppress_unknown_category_warnings():
        result = permutation_importance(
            pipeline, inputs, truth, n_repeats=n_repeats, random_state=random_state
        )
    importance = pd.Series(result.importances_mean, index=inputs.columns).sort_values()
    importance = importance.tail(top_n)
    figure = _figure(9, max(3.0, 0.34 * len(importance) + 1.2))
    axes = figure.subplots()
    axes.barh(importance.index.astype(str), importance.to_numpy(), color=_ACCENT)
    axes.set(
        title="Permutation importance (held-out rows)", xlabel="Mean drop in score when shuffled"
    )
    return _save(figure, path)


def generate_all(
    frame: pd.DataFrame,
    out_dir: PathLike,
    *,
    random_state: int = 42,
    importance_repeats: int = 5,
) -> dict[str, Path]:
    """Rebuild every figure from the raw data.

    Trains the notebook's linear regression (residuals), its logistic
    regression (confusion matrix) and a gradient-boosting model (importance) on
    the standard 80/20 split.

    Args:
        frame: The full dataset, including ``SalePrice``.
        out_dir: Directory to write the PNG files into; created if needed.
        random_state: Seed for the split, models and permutations.
        importance_repeats: Shuffles per column for the importance plot.

    Returns:
        Figure name -> written file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "target_distribution": plot_target_distribution(
            frame, out / FIGURE_FILES["target_distribution"]
        ),
        "missing_values": plot_missing_values(frame, out / FIGURE_FILES["missing_values"]),
    }

    base = TrainConfig(random_state=random_state)
    regression = train(frame, base)
    _, test = train_test_split_df(frame, test_size=base.test_size, random_state=random_state)
    written["residuals"] = plot_residuals(
        test[base.target],
        predict(regression.pipeline, test).values,
        out / FIGURE_FILES["residuals"],
    )

    classification = train(frame, base.replace(task="classification"))
    matrix = classification.metrics.to_dict()["confusion_matrix"]
    written["confusion_matrix"] = plot_confusion_matrix(
        matrix, out / FIGURE_FILES["confusion_matrix"]
    )

    boosted = train(frame, base.replace(model="gbm"))
    written["feature_importance"] = plot_feature_importance(
        boosted.pipeline,
        test,
        out / FIGURE_FILES["feature_importance"],
        n_repeats=importance_repeats,
        random_state=random_state,
    )
    return written


def _figure(width: float, height: float) -> Figure:
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'Plotting needs matplotlib. Install it with: pip install "amesml[plots]"'
        ) from exc
    return Figure(figsize=(width, height), layout="constrained")


def _save(figure: Any, path: PathLike) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=_DPI)
    return target
