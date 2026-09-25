"""Training, saving, loading and prediction.

A trained model is an ordinary scikit-learn ``Pipeline`` with a few extra
fitted attributes that make it self-describing:

``task_``
    ``"regression"`` or ``"classification"``.
``input_features_``
    The raw columns the pipeline expects, in order.
``input_kinds_``
    ``"number"`` or ``"text"`` for each input column, used to coerce inputs.
``feature_defaults_``
    The most common training value of each input (the median for numeric
    measurements), used for any column an input leaves out.
``threshold_``
    Classification only: the price separating the two classes.
``training_data_sha256_``
    Fingerprint of the rows the pipeline was fitted on.
``amesml_config_``
    The :class:`~amesml.config.TrainConfig` used, as a dictionary.

On disk a model is a directory holding ``model.joblib`` and ``metadata.json``
(and ``metrics.json`` when it came from :func:`train`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.typing import NDArray
from pandas.api import types as ptypes
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ._version import __version__
from .config import TrainConfig
from .data import PathLike, dataset_fingerprint, train_test_split_df
from .errors import AmesSchemaError, ConfigError, ModelLoadError
from .evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classifier,
    evaluate_regressor,
)
from .features import ENGINEERED_FEATURES, SOURCE_COLUMNS, add_engineered_features
from .preprocess import build_preprocessor, infer_feature_types

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"
#: Bumped whenever the layout of ``metadata.json`` changes incompatibly.
METADATA_SCHEMA_VERSION = 1

Metrics = Union[RegressionMetrics, ClassificationMetrics]


def _sklearn_version() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


SKLEARN_VERSION = _sklearn_version()


# --------------------------------------------------------------------------- estimators


def make_estimator(
    task: str, model: str, *, alpha: float | None, random_state: int, max_iter: int
) -> Any:
    """Create the unfitted scikit-learn estimator for a task and model family.

    For classification, ``alpha`` is an inverse regularisation strength:
    ``C = 1 / alpha``, matching the regression models where a larger ``alpha``
    always means a simpler model.
    """
    if model in ("ridge", "lasso") and alpha is None:
        raise ConfigError(f"{model} needs a regularisation strength (alpha)")
    if task == "regression":
        if model == "linear":
            return LinearRegression()
        if model == "ridge":
            return Ridge(alpha=alpha)
        if model == "lasso":
            return Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
        if model == "gbm":
            return HistGradientBoostingRegressor(random_state=random_state)
    elif task == "classification":
        if model == "linear":
            return LogisticRegression(max_iter=max_iter, random_state=random_state)
        if model == "ridge":
            assert alpha is not None  # checked above; narrows the type
            return LogisticRegression(C=1.0 / alpha, max_iter=max_iter, random_state=random_state)
        if model == "lasso":
            assert alpha is not None
            return l1_logistic_regression(1.0 / alpha, random_state=random_state, max_iter=max_iter)
        if model == "gbm":
            return HistGradientBoostingClassifier(random_state=random_state)
    raise ConfigError(f"Unsupported task/model combination: {task}/{model}")


def l1_logistic_regression(c: float, *, random_state: int, max_iter: int) -> Any:
    """L1-penalised logistic regression across scikit-learn versions.

    scikit-learn 1.8 deprecated ``penalty=`` (removal in 1.10) in favour of
    ``l1_ratio``, but before 1.8 ``l1_ratio`` was ignored unless
    ``penalty="elasticnet"``. ``liblinear`` supports L1 everywhere and is far
    faster than ``saga`` on data of this size, with identical results.
    """
    if SKLEARN_VERSION >= (1, 8):
        return LogisticRegression(
            l1_ratio=1.0, C=c, solver="liblinear", max_iter=max_iter, random_state=random_state
        )
    return LogisticRegression(
        penalty="l1", C=c, solver="liblinear", max_iter=max_iter, random_state=random_state
    )


# --------------------------------------------------------------------------- feature plan


@dataclass(frozen=True)
class FeaturePlan:
    """Which columns a pipeline consumes and how each is preprocessed.

    Attributes:
        input_columns: Raw columns the pipeline reads, in data order.
        numeric: Numeric model features, including engineered ones.
        categorical: Categorical model features.
        codes: Members of ``categorical`` stored as numbers.
        zero_fill: Members of ``numeric`` whose gaps are filled with 0.
    """

    input_columns: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    codes: tuple[str, ...]
    zero_fill: tuple[str, ...]


def plan_features(frame: pd.DataFrame, config: TrainConfig) -> FeaturePlan:
    """Work out the feature columns for ``frame`` under ``config``.

    Raises:
        AmesSchemaError: A configured column is missing or has the wrong type.
    """
    reserved = {config.target}
    if config.id_column:
        reserved.add(config.id_column)
    if config.engineer_features:
        # Recomputed from raw columns inside the pipeline; never read as inputs.
        reserved.update(ENGINEERED_FEATURES)
    available = [c for c in frame.columns if c not in reserved]
    inferred = infer_feature_types(
        frame[available], exclude=(), categorical_codes=config.categorical_codes
    )

    forced_categorical = set(config.categorical_features or ())
    forced_numeric = set(config.numeric_features or ())
    if config.numeric_features is not None:
        numeric = list(config.numeric_features)
    else:
        numeric = [c for c in inferred.numeric if c not in forced_categorical]
    if config.categorical_features is not None:
        categorical = list(config.categorical_features)
    else:
        categorical = [c for c in inferred.categorical if c not in forced_numeric]

    missing = [c for c in numeric + categorical if c not in frame.columns]
    if missing:
        raise AmesSchemaError(f"Configured feature column(s) not in the data: {', '.join(missing)}")
    not_numeric = [c for c in numeric if not ptypes.is_numeric_dtype(frame[c])]
    if not_numeric:
        raise AmesSchemaError(
            f"Numeric feature(s) with non-numeric values: {', '.join(not_numeric)}"
        )
    if not numeric and not categorical:
        raise AmesSchemaError("No feature columns to train on")

    needed = set(numeric) | set(categorical)
    if config.engineer_features:
        absent = [c for c in SOURCE_COLUMNS if c not in frame.columns]
        if absent:
            raise AmesSchemaError(
                "engineer_features needs column(s) missing from the data: " + ", ".join(absent)
            )
        needed |= set(SOURCE_COLUMNS)
        numeric += [name for name in ENGINEERED_FEATURES if name not in numeric]

    codes = [c for c in categorical if ptypes.is_numeric_dtype(frame[c])]
    zero_fill = [c for c in config.zero_fill_features if c in numeric]
    return FeaturePlan(
        input_columns=tuple(c for c in frame.columns if c in needed),
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        codes=tuple(codes),
        zero_fill=tuple(zero_fill),
    )


def build_pipeline(plan: FeaturePlan, config: TrainConfig) -> Pipeline:
    """Assemble the unfitted pipeline for a feature plan."""
    # Trees split on thresholds and least squares is scale-invariant, so only
    # the penalised and logistic models need standardised inputs.
    scale = config.model != "gbm" and not (config.task == "regression" and config.model == "linear")
    steps: list[tuple[str, Any]] = []
    if config.engineer_features:
        steps.append(("features", FunctionTransformer(add_engineered_features)))
    steps.append(
        (
            "preprocess",
            build_preprocessor(
                plan.numeric,
                plan.categorical,
                scale,
                zero_fill_cols=plan.zero_fill,
                code_cols=plan.codes,
                drop_first=config.drop_first,
                # Histogram gradient boosting rejects sparse input. The linear
                # models keep scikit-learn's default (sparse here), which is
                # what makes the defaults reproduce the notebook: for sparse
                # input LinearRegression uses the iterative LSQR solver.
                dense=config.model == "gbm",
            ),
        )
    )
    steps.append(
        (
            "model",
            make_estimator(
                config.task,
                config.model,
                alpha=config.resolved_alpha,
                random_state=config.random_state,
                max_iter=config.max_iter,
            ),
        )
    )
    return Pipeline(steps)


# --------------------------------------------------------------------------- training


def resolve_threshold(prices: pd.Series, threshold: float | str) -> float:
    """Turn ``"median"`` into the median of ``prices``; pass numbers through."""
    if threshold == "median":
        return float(prices.median())
    return float(threshold)


def price_labels(prices: pd.Series, threshold: float) -> pd.Series:
    """Label each price 1 if it is at or above ``threshold``, otherwise 0."""
    return (prices >= threshold).astype(np.int64)


def train_regressor(frame: pd.DataFrame, config: TrainConfig | None = None) -> Pipeline:
    """Fit a sale-price regression pipeline on every row of ``frame``.

    Use :func:`~amesml.data.train_test_split_df` first to keep a test set, or
    :func:`train` to split, fit and evaluate in one call.

    Args:
        frame: Training rows, including the target column.
        config: Training options; ``config.task`` is ignored.

    Returns:
        The fitted pipeline.
    """
    return _fit(frame, _for_task(config, "regression"))


def train_classifier(frame: pd.DataFrame, config: TrainConfig | None = None) -> Pipeline:
    """Fit an above/below-threshold price classifier on every row of ``frame``.

    The threshold is resolved from ``frame`` itself (its median sale price by
    default) and stored on the pipeline as ``threshold_``.

    Args:
        frame: Training rows, including the target column.
        config: Training options; ``config.task`` is ignored.

    Returns:
        The fitted pipeline.
    """
    return _fit(frame, _for_task(config, "classification"))


def _for_task(config: TrainConfig | None, task: str) -> TrainConfig:
    config = config or TrainConfig()
    if config.task == task:
        return config
    # stratify is classification-only; drop it rather than fail validation.
    return config.replace(task=task, stratify=config.stratify and task == "classification")


def _fit(frame: pd.DataFrame, config: TrainConfig) -> Pipeline:
    prices = _target(frame, config.target)
    plan = plan_features(frame, config)
    inputs = frame[list(plan.input_columns)]
    pipeline = build_pipeline(plan, config)

    with suppress_unknown_category_warnings():
        if config.task == "classification":
            threshold = resolve_threshold(prices, config.threshold)
            labels = price_labels(prices, threshold)
            if labels.nunique() < 2:
                raise ConfigError(
                    f"A threshold of {threshold:,.0f} puts all {len(labels)} training rows "
                    "in one class; choose a threshold inside the price range"
                )
            pipeline.fit(inputs, labels)
            pipeline.threshold_ = threshold
        else:
            pipeline.fit(inputs, prices.astype(float))

    categorical = set(plan.categorical)
    pipeline.task_ = config.task
    pipeline.input_features_ = list(plan.input_columns)
    pipeline.input_kinds_ = {
        c: "number" if ptypes.is_numeric_dtype(inputs[c]) else "text" for c in plan.input_columns
    }
    pipeline.feature_defaults_ = _feature_defaults(inputs, categorical)
    pipeline.training_data_sha256_ = dataset_fingerprint(frame)
    pipeline.amesml_config_ = config.to_dict()
    pipeline.amesml_version_ = __version__
    logger.info(
        "Trained %s/%s on %d rows with %d numeric and %d categorical features",
        config.task,
        config.model,
        len(frame),
        len(plan.numeric),
        len(plan.categorical),
    )
    return pipeline


def _target(frame: pd.DataFrame, target: str) -> pd.Series:
    if target not in frame.columns:
        raise AmesSchemaError(f"Target column {target!r} not found in the data")
    prices = frame[target]
    if not ptypes.is_numeric_dtype(prices):
        raise AmesSchemaError(f"Target column {target!r} must be numeric")
    if prices.isna().any():
        raise AmesSchemaError(f"Target column {target!r} has missing values")
    return prices


def _feature_defaults(inputs: pd.DataFrame, categorical: set[str]) -> dict[str, Any]:
    """Typical value per input column: the median for measurements, else the mode."""
    defaults: dict[str, Any] = {}
    for column in inputs.columns:
        series = inputs[column]
        if column in categorical or not ptypes.is_numeric_dtype(series):
            defaults[column] = _mode(series)
        else:
            defaults[column] = _median(series)
    return defaults


def _median(series: pd.Series) -> float | None:
    value = series.median()
    return None if pd.isna(value) else float(value)


def _mode(series: pd.Series) -> Any:
    """Most frequent value, counting "missing" as a value of its own.

    For ``PoolQC`` the commonest value is *missing* (no pool). Ignoring missing
    values would make the default ``"Gd"`` and silently give every house with
    an unspecified pool a good one.
    """
    counts = series.value_counts(dropna=False)
    top = counts.max()
    tied = [value for value, count in counts.items() if count == top]
    # Ties are broken by value so the result never depends on row order.
    best = sorted(tied, key=lambda value: (_is_missing(value), str(value)))[0]
    return None if _is_missing(best) else _python_scalar(best)


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))


@dataclass
class TrainResult:
    """Outcome of :func:`train`: the fitted model and its held-out evaluation.

    Attributes:
        pipeline: The fitted pipeline.
        config: Configuration used.
        metrics: Scores on the held-out test rows.
        train_rows: Number of rows used for fitting.
        test_rows: Number of held-out rows scored.
        dataset_sha256: Fingerprint of the full dataset before splitting.
        unseen_categories: Test-set categories absent from the training rows.
    """

    pipeline: Pipeline
    config: TrainConfig
    metrics: Metrics
    train_rows: int
    test_rows: int
    dataset_sha256: str
    unseen_categories: dict[str, list[str]] = field(default_factory=dict)

    def save(self, directory: PathLike) -> Path:
        """Write ``model.joblib``, ``metadata.json`` and ``metrics.json``.

        Returns:
            The model directory.
        """
        model_file = save_model(
            self.pipeline,
            directory,
            metrics=self.metrics.to_dict(),
            extra={
                "dataset": {
                    "sha256": self.dataset_sha256,
                    "rows": self.train_rows + self.test_rows,
                    "train_rows": self.train_rows,
                    "test_rows": self.test_rows,
                }
            },
        )
        metrics_file = model_file.parent / METRICS_FILENAME
        metrics_file.write_text(_dumps(self.metrics.to_dict()), encoding="utf-8")
        return model_file.parent


def train(frame: pd.DataFrame, config: TrainConfig | None = None) -> TrainResult:
    """Split, fit and evaluate in one step.

    The rows are split with ``config.test_size`` and ``config.random_state``;
    the model is fitted on the training rows and scored on the rest.

    Args:
        frame: The full dataset, including the target column.
        config: Training options; defaults reproduce the notebook.

    Returns:
        The fitted pipeline together with its test-set metrics.
    """
    config = config or TrainConfig()
    train_frame, test_frame = split_for_training(frame, config)
    fit = train_classifier if config.task == "classification" else train_regressor
    pipeline = fit(train_frame, config)

    predictions = predict(pipeline, test_frame)
    truth = test_frame[config.target]
    metrics: Metrics
    if config.task == "classification":
        metrics = evaluate_classifier(
            price_labels(truth, pipeline.threshold_),
            predictions.values,
            predictions.probabilities,
        )
    else:
        metrics = evaluate_regressor(truth, predictions.values)
    if predictions.unseen_categories:
        logger.info(
            "The test rows contain categories never seen in training: %s",
            _describe_unseen(predictions.unseen_categories),
        )
    return TrainResult(
        pipeline=pipeline,
        config=config,
        metrics=metrics,
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        dataset_sha256=dataset_fingerprint(frame),
        unseen_categories=predictions.unseen_categories,
    )


def split_for_training(
    frame: pd.DataFrame, config: TrainConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the train/test split that :func:`train` uses for ``config``.

    Returns:
        ``(train, test)`` frames.
    """
    prices = _target(frame, config.target)
    stratify = None
    if config.stratify:
        # Labels for balancing the split only; the model's own threshold is
        # re-derived from the training rows so no test price leaks into it.
        stratify = price_labels(prices, resolve_threshold(prices, config.threshold))
    return train_test_split_df(
        frame, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )


# --------------------------------------------------------------------------- persistence


def save_model(
    pipeline: Pipeline,
    path: PathLike,
    *,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save a trained pipeline with a ``metadata.json`` sidecar.

    Args:
        pipeline: A pipeline returned by :func:`train_regressor`,
            :func:`train_classifier` or :func:`train`.
        path: A directory to create (``model.joblib`` is written inside it),
            or the path of a ``.joblib`` file whose directory receives
            ``metadata.json``. Keep one model per directory.
        metrics: Evaluation results to record.
        extra: Further top-level metadata entries.

    Returns:
        The path of the written model file.

    Raises:
        ValueError: ``pipeline`` was not produced by amesml.
    """
    if not hasattr(pipeline, "task_"):
        raise ValueError("save_model expects a pipeline returned by amesml's training functions")
    model_file, metadata_file = _bundle_paths(path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_file)

    metadata = build_metadata(pipeline, metrics=metrics)
    metadata["artifact"] = {"file": model_file.name, "sha256": _sha256_file(model_file)}
    if extra:
        metadata.update(extra)
    metadata_file.write_text(_dumps(metadata), encoding="utf-8")
    logger.info("Saved model to %s", model_file)
    return model_file


def build_metadata(
    pipeline: Pipeline, *, metrics: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Describe a trained pipeline as a JSON-serialisable dictionary."""
    config = dict(pipeline.amesml_config_)
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "amesml_version": getattr(pipeline, "amesml_version_", __version__),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": pipeline.task_,
        "model": config.get("model"),
        "estimator": type(pipeline.steps[-1][1]).__name__,
        "threshold": getattr(pipeline, "threshold_", None),
        "config": config,
        "metrics": dict(metrics) if metrics is not None else None,
        "features": {
            "inputs": list(pipeline.input_features_),
            "kinds": dict(pipeline.input_kinds_),
            "engineered": list(ENGINEERED_FEATURES) if config.get("engineer_features") else [],
        },
        "feature_defaults": dict(pipeline.feature_defaults_),
        "training_data_sha256": pipeline.training_data_sha256_,
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def load_model(path: PathLike) -> Pipeline:
    """Load a pipeline saved by :func:`save_model`.

    If ``metadata.json`` is present, the model file's SHA-256 is checked
    against it first. This catches corrupted or swapped files; it is not a
    security boundary, because unpickling runs code. Only load models you trust.

    Args:
        path: The model directory, or the ``.joblib`` file inside it.

    Raises:
        ModelLoadError: The file is missing, fails its checksum, or is not an
            amesml model.
    """
    model_file, metadata_file = _resolve_existing(path)
    if metadata_file.is_file():
        expected = _read_json(metadata_file).get("artifact", {}).get("sha256")
        if expected and _sha256_file(model_file) != expected:
            raise ModelLoadError(
                f"{model_file} does not match the checksum in {metadata_file}; "
                "the file was modified or replaced"
            )
    try:
        pipeline = joblib.load(model_file)
    except Exception as exc:  # unpickling can fail in many library-specific ways
        raise ModelLoadError(f"Could not load {model_file}: {exc}") from exc
    if not hasattr(pipeline, "task_"):
        raise ModelLoadError(f"{model_file} is not an amesml model")
    return pipeline


def load_metadata(path: PathLike) -> dict[str, Any]:
    """Read the ``metadata.json`` belonging to a saved model.

    Args:
        path: The model directory, or the ``.joblib`` file inside it.

    Raises:
        ModelLoadError: There is no metadata next to the model.
    """
    _, metadata_file = _resolve_existing(path)
    if not metadata_file.is_file():
        raise ModelLoadError(f"No {METADATA_FILENAME} found next to the model at {path}")
    return _read_json(metadata_file)


def _bundle_paths(path: PathLike) -> tuple[Path, Path]:
    target = Path(path)
    if target.suffix == ".joblib":
        return target, target.parent / METADATA_FILENAME
    return target / MODEL_FILENAME, target / METADATA_FILENAME


def _resolve_existing(path: PathLike) -> tuple[Path, Path]:
    target = Path(path)
    if target.is_dir():
        target = target / MODEL_FILENAME
    if not target.is_file():
        raise ModelLoadError(f"No model found at {path}")
    return target, target.parent / METADATA_FILENAME


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelLoadError(f"{path} must contain a JSON object")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dumps(data: Any) -> str:
    return json.dumps(_jsonable(data), indent=2, allow_nan=False) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    value = _python_scalar(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


# --------------------------------------------------------------------------- prediction


@dataclass(frozen=True)
class Predictions:
    """Predictions for a batch of houses.

    Attributes:
        task: ``"regression"`` or ``"classification"``.
        values: Predicted prices, or predicted labels (1 = at or above the threshold).
        probabilities: Classification only: probability of label 1.
        unseen_categories: Column -> category values the model never saw in
            training. Such values are encoded like the column's reference level.
    """

    task: str
    values: NDArray[Any]
    probabilities: NDArray[np.float64] | None = None
    unseen_categories: dict[str, list[str]] = field(default_factory=dict)

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        """Return the predictions as a DataFrame with descriptive column names."""
        if self.task == "classification":
            data: dict[str, Any] = {"predicted_label": self.values}
            if self.probabilities is not None:
                data["probability_above_threshold"] = self.probabilities
        else:
            data = {"predicted_price": self.values}
        frame: pd.DataFrame = pd.DataFrame(data, index=index)
        return frame


def records_to_frame(
    pipeline: Pipeline, records: Sequence[Mapping[str, Any]], *, strict: bool = True
) -> pd.DataFrame:
    """Turn JSON-style records into a model-ready frame.

    A field left out of a record takes the model's training default (see
    ``feature_defaults_``); a field sent as ``None`` is missing for that house
    and handled like a gap in the training data. For example, omit ``PoolQC``
    to get the typical house, or send ``"PoolQC": None`` to say there is no pool.

    Args:
        pipeline: A trained amesml pipeline.
        records: One mapping of column name to value per house.
        strict: Reject field names the model does not use, so a typo cannot
            silently fall back to a default.

    Raises:
        AmesSchemaError: ``strict`` is set and a record has an unknown field.
    """
    expected: list[str] = list(pipeline.input_features_)
    if strict:
        unknown = sorted({key for record in records for key in record} - set(expected))
        if unknown:
            raise AmesSchemaError(f"Unknown feature(s): {', '.join(unknown)}")
    defaults: dict[str, Any] = pipeline.feature_defaults_
    rows = [{**defaults, **dict(record)} for record in records]
    frame: pd.DataFrame = pd.DataFrame(rows, columns=expected)
    return frame


def prepare_input(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Align and type-check a frame of raw Ames columns for ``pipeline``.

    Columns the model does not use are dropped (so a full Kaggle file with
    ``Id`` and ``SalePrice`` works). Columns the model needs but the frame lacks
    are filled with training defaults.

    Raises:
        AmesSchemaError: A numeric column contains a value that is not a number.
    """
    expected: list[str] = list(pipeline.input_features_)
    kinds: dict[str, str] = pipeline.input_kinds_
    defaults: dict[str, Any] = pipeline.feature_defaults_
    absent = [c for c in expected if c not in frame.columns]
    if absent:
        logger.info("Filling %d absent column(s) with training defaults", len(absent))
    columns: dict[str, pd.Series] = {}
    for column in expected:
        if column in frame.columns:
            series = frame[column]
        else:
            series = pd.Series([defaults.get(column)] * len(frame), index=frame.index, dtype=object)
        columns[column] = _coerce(series, kinds[column], column)
    prepared: pd.DataFrame = pd.DataFrame(columns, index=frame.index)
    return prepared


def predict(pipeline: Pipeline, frame: pd.DataFrame) -> Predictions:
    """Predict for every row of ``frame``, which holds raw Ames columns.

    Args:
        pipeline: A trained amesml pipeline.
        frame: Houses to score; see :func:`prepare_input` for column handling.

    Returns:
        Prices for a regression model; labels and probabilities for a classifier.
    """
    inputs = prepare_input(pipeline, frame)
    unseen = unseen_categories(pipeline, inputs)
    if unseen:
        logger.warning("Categories not seen in training: %s", _describe_unseen(unseen))
    with suppress_unknown_category_warnings():
        if pipeline.task_ == "classification":
            labels = np.asarray(pipeline.predict(inputs)).astype(np.int64)
            probabilities = np.asarray(pipeline.predict_proba(inputs), dtype=float)[:, 1]
            return Predictions("classification", labels, probabilities, unseen)
        prices = np.asarray(pipeline.predict(inputs), dtype=float)
        return Predictions("regression", prices, None, unseen)


def unseen_categories(pipeline: Pipeline, inputs: pd.DataFrame) -> dict[str, list[str]]:
    """Find categorical values in ``inputs`` that the model never saw in training."""
    frame = inputs
    if "features" in pipeline.named_steps:
        frame = pipeline.named_steps["features"].transform(frame)
    found: dict[str, list[str]] = {}
    for name, branch, columns in pipeline.named_steps["preprocess"].transformers_:
        if name not in ("categorical", "categorical_codes"):
            continue
        columns = list(columns)
        imputed = branch[:-1].transform(frame[columns])  # every step before encoding
        categories = branch.named_steps["encode"].categories_
        for position, column in enumerate(columns):
            known = set(categories[position].tolist())
            unseen = {_display(v) for v in imputed[:, position] if v not in known}
            if unseen:
                found[column] = sorted(unseen)
    return found


@contextmanager
def suppress_unknown_category_warnings() -> Iterator[None]:
    """Silence scikit-learn's per-call warning about unknown categories.

    With ``drop_first`` the encoder warns on every call that meets a category
    absent from training, naming columns only by position. amesml reports those
    values itself, by column name, through :func:`unseen_categories`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
        yield


def _coerce(series: pd.Series, kind: str, column: str) -> pd.Series:
    # Element-wise on purpose: Series.replace and Series.where emit pandas 2.2's
    # "downcasting is deprecated" FutureWarning for some object columns.
    if kind == "number":
        if ptypes.is_numeric_dtype(series):
            return series.astype(float)
        values = [np.nan if _is_blank(v) else v for v in series.tolist()]
        numbers: pd.Series = pd.to_numeric(
            pd.Series(values, index=series.index, dtype=object), errors="coerce"
        )
        invalid = [v for v, n in zip(values, numbers.tolist()) if not _is_missing(v) and pd.isna(n)]
        if invalid:
            raise AmesSchemaError(f"Column {column!r} must be numeric; got {invalid[0]!r}")
        floats: pd.Series = numbers.astype(float)
        return floats
    text: pd.Series = pd.Series(
        [_as_text(v) for v in series.tolist()], index=series.index, dtype=object
    )
    return text


def _as_text(value: Any) -> Any:
    if _is_blank(value):
        return np.nan
    return value if isinstance(value, str) else str(value)


def _is_blank(value: Any) -> bool:
    return _is_missing(value) or (isinstance(value, str) and not value.strip())


def _display(value: Any) -> str:
    value = _python_scalar(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # category codes are stored as floats after coercion
    return str(value)


def _describe_unseen(unseen: Mapping[str, Sequence[str]]) -> str:
    return "; ".join(f"{column}: {', '.join(values)}" for column, values in unseen.items())
