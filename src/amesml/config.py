"""Training configuration.

A :class:`TrainConfig` records every choice that changes a trained model. It is
saved inside each model's ``metadata.json``, so any model can be retrained
exactly from its own metadata.

The defaults reproduce the original notebook: an unregularised linear model
(or a logistic regression for classification) on the raw Ames columns, with an
80/20 split seeded with 42.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import yaml

from .errors import ConfigError

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

TaskType = Literal["regression", "classification"]
ModelType = Literal["linear", "ridge", "lasso", "gbm"]
Threshold = Union[float, Literal["median"]]

TASKS: tuple[str, ...] = ("regression", "classification")
MODELS: tuple[str, ...] = ("linear", "ridge", "lasso", "gbm")

#: Regularisation strength used when ``alpha`` is left unset, per (task, model).
#: Each value was chosen by 5-fold cross-validation on the training split only;
#: see ``docs/DECISIONS.md``. For classification, ``C = 1 / alpha``.
DEFAULT_ALPHAS: dict[tuple[str, str], float] = {
    ("regression", "ridge"): 10.0,
    ("regression", "lasso"): 300.0,
    ("classification", "ridge"): 30.0,
    ("classification", "lasso"): 3.0,
}

_TUPLE_FIELDS = (
    "numeric_features",
    "categorical_features",
    "categorical_codes",
    "zero_fill_features",
)


@dataclass(frozen=True)
class TrainConfig:
    """Everything that determines how a model is trained.

    Attributes:
        task: ``"regression"`` predicts the sale price; ``"classification"``
            predicts whether a house sells at or above ``threshold``.
        model: Estimator family. ``linear`` is ordinary least squares (or
            logistic regression for classification), ``ridge`` and ``lasso``
            add L2 and L1 penalties, ``gbm`` is histogram gradient boosting.
        target: Name of the sale-price column.
        id_column: Identifier column that is never used as a feature.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for the train/test split and every seeded estimator.
        threshold: Price separating the two classes, or ``"median"`` to use the
            median sale price of the training split.
        stratify: Keep the class balance equal across the split. Only valid
            for classification.
        engineer_features: Add the derived features from
            :mod:`amesml.features` (total area, ages, bathrooms, porch area).
        numeric_features: Use exactly these numeric columns instead of
            inferring them from the data.
        categorical_features: Use exactly these categorical columns instead of
            inferring them from the data.
        categorical_codes: Columns stored as numbers that are really category
            codes, such as the dwelling-type code ``MSSubClass``.
        zero_fill_features: Numeric columns where a missing value means "none
            of this", so they are filled with 0 instead of the median.
        drop_first: Drop the first level of each one-hot encoded column, as the
            notebook did. This avoids the dummy-variable trap for the
            unregularised models.
        alpha: Regularisation strength for ``ridge`` and ``lasso``. ``None``
            selects the cross-validated default in :data:`DEFAULT_ALPHAS`.
        max_iter: Iteration limit for the iterative solvers.
    """

    task: TaskType = "regression"
    model: ModelType = "linear"
    target: str = "SalePrice"
    id_column: str | None = "Id"
    test_size: float = 0.2
    random_state: int = 42
    threshold: Threshold = "median"
    stratify: bool = False
    engineer_features: bool = False
    numeric_features: tuple[str, ...] | None = None
    categorical_features: tuple[str, ...] | None = None
    categorical_codes: tuple[str, ...] = ("MSSubClass", "MoSold")
    zero_fill_features: tuple[str, ...] = ("GarageYrBlt", "MasVnrArea")
    drop_first: bool = True
    alpha: float | None = None
    max_iter: int = 5000

    def __post_init__(self) -> None:
        # YAML and JSON produce lists; store tuples so the config stays hashable.
        for name in _TUPLE_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, _string_tuple(name, value))
        self._validate()

    @property
    def resolved_alpha(self) -> float | None:
        """The regularisation strength actually used, after applying defaults."""
        if self.model in ("linear", "gbm"):
            return None
        if self.alpha is not None:
            return self.alpha
        return DEFAULT_ALPHAS[(self.task, self.model)]

    def replace(self, **changes: Any) -> TrainConfig:
        """Return a copy with ``changes`` applied; the copy is validated too."""
        unknown = sorted(set(changes) - _field_names())
        if unknown:
            raise ConfigError(f"Unknown configuration option(s): {', '.join(unknown)}")
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON- and YAML-friendly dictionary of every option."""
        data = dataclasses.asdict(self)
        for name in _TUPLE_FIELDS:
            if data[name] is not None:
                data[name] = list(data[name])
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        """Build a config from a mapping, rejecting unknown keys."""
        if not isinstance(data, Mapping):
            raise ConfigError(f"Configuration must be a mapping, got {type(data).__name__}")
        unknown = sorted(set(data) - _field_names())
        if unknown:
            known = ", ".join(sorted(_field_names()))
            raise ConfigError(
                f"Unknown configuration option(s): {', '.join(unknown)}. Valid options: {known}"
            )
        return cls(**dict(data))

    @classmethod
    def from_file(cls, path: str | Path) -> TrainConfig:
        """Load a config from a ``.yaml``, ``.yml`` or ``.json`` file."""
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Configuration file not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                data = yaml.safe_load(text)
            elif suffix == ".json":
                data = json.loads(text)
            else:
                raise ConfigError(
                    f"Unsupported configuration format {suffix!r}; use .yaml, .yml or .json"
                )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Could not parse {path}: {exc}") from exc
        if data is None:
            data = {}
        return cls.from_dict(data)

    def _validate(self) -> None:
        if self.task not in TASKS:
            raise ConfigError(f"task must be one of {', '.join(TASKS)}; got {self.task!r}")
        if self.model not in MODELS:
            raise ConfigError(f"model must be one of {', '.join(MODELS)}; got {self.model!r}")
        if not isinstance(self.target, str) or not self.target:
            raise ConfigError("target must be a non-empty column name")
        if self.id_column is not None and not isinstance(self.id_column, str):
            raise ConfigError("id_column must be a column name or null")
        if not _is_real(self.test_size) or not 0 < self.test_size < 1:
            raise ConfigError(f"test_size must be between 0 and 1; got {self.test_size!r}")
        if not _is_int(self.random_state) or self.random_state < 0:
            raise ConfigError(
                f"random_state must be a non-negative integer; got {self.random_state!r}"
            )
        if self.threshold != "median" and (
            not _is_real(self.threshold) or not math.isfinite(self.threshold) or self.threshold <= 0
        ):
            raise ConfigError(
                f"threshold must be 'median' or a positive price; got {self.threshold!r}"
            )
        for name in ("stratify", "engineer_features", "drop_first"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be true or false")
        if self.stratify and self.task != "classification":
            raise ConfigError("stratify only applies to classification")
        if self.alpha is not None and (
            not _is_real(self.alpha) or not math.isfinite(self.alpha) or self.alpha <= 0
        ):
            raise ConfigError(f"alpha must be a positive number; got {self.alpha!r}")
        if not _is_int(self.max_iter) or self.max_iter <= 0:
            raise ConfigError(f"max_iter must be a positive integer; got {self.max_iter!r}")
        if self.numeric_features is not None and self.categorical_features is not None:
            overlap = sorted(set(self.numeric_features) & set(self.categorical_features))
            if overlap:
                raise ConfigError(
                    f"Column(s) listed as both numeric and categorical: {', '.join(overlap)}"
                )
        reserved = {self.target} | ({self.id_column} if self.id_column else set())
        for name in ("numeric_features", "categorical_features"):
            listed = getattr(self, name) or ()
            clash = sorted(reserved & set(listed))
            if clash:
                raise ConfigError(f"{name} may not include the target or id: {', '.join(clash)}")


def _field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(TrainConfig)}


def _string_tuple(name: str, value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a list of column names")
    if not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{name} must contain only non-empty column names")
    return tuple(value)


def _is_real(value: object) -> TypeGuard[float]:
    # bool is a subclass of int, but `test_size: true` is a mistake, not a number.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
