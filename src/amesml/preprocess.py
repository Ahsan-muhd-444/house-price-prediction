"""Column typing and the scikit-learn preprocessing transformer.

The transformer reproduces the notebook's choices, which follow from what a
missing value means in this dataset:

* Numeric gaps are filled with the training median - except columns listed in
  ``zero_fill_cols`` (``MasVnrArea``, ``GarageYrBlt``), where a gap means the
  feature is absent rather than unknown.
* Categorical gaps become the explicit level ``"Missing"``. ``NA`` in Ames means
  "no pool", "no basement" and so on; filling it with the most frequent level
  would claim that houses without a pool have the most common pool quality.
* Category codes stored as numbers (``MSSubClass``, ``MoSold``) get their own
  branch, filled with the most frequent code, because a column cannot mix the
  string ``"Missing"`` with integers and still be encoded.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

#: Value given to missing categorical entries before one-hot encoding.
MISSING_CATEGORY = "Missing"


@dataclass(frozen=True)
class FeatureTypes:
    """The model's input columns, split by how they are preprocessed."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """All feature columns, numeric first."""
        return self.numeric + self.categorical


def infer_feature_types(
    frame: pd.DataFrame,
    *,
    exclude: Iterable[str] = ("Id", "SalePrice"),
    categorical_codes: Iterable[str] = ("MSSubClass", "MoSold"),
) -> FeatureTypes:
    """Classify the columns of ``frame`` as numeric or categorical features.

    Columns with a numeric dtype are numeric features, except the category
    codes in ``categorical_codes``. Every other dtype - ``object``, pandas'
    ``str`` and ``category`` - is categorical. Checking "not numeric" rather
    than "object" matters: under pandas 3 text columns use the ``str`` dtype.

    Args:
        frame: Data whose columns to classify.
        exclude: Columns that are not features (the identifier and target).
        categorical_codes: Numeric columns that encode categories.

    Returns:
        The feature columns in their original order, with the category codes
        placed after the text columns.
    """
    excluded = set(exclude)
    codes = [c for c in categorical_codes if c in frame.columns and c not in excluded]
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column in excluded or column in codes:
            continue
        if ptypes.is_numeric_dtype(frame[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return FeatureTypes(numeric=tuple(numeric), categorical=tuple(categorical + codes))


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    scale: bool = True,
    *,
    zero_fill_cols: Sequence[str] = (),
    code_cols: Sequence[str] = (),
    drop_first: bool = False,
    dense: bool = False,
) -> Any:
    """Build the (unfitted) column transformer for the given feature columns.

    Args:
        numeric_cols: Numeric features; missing values become the median.
        categorical_cols: Categorical features; one-hot encoded.
        scale: Standardise numeric features. Needed by the regularised and
            logistic models; unnecessary for trees and plain least squares.
        zero_fill_cols: Members of ``numeric_cols`` whose missing values mean
            "none" and become 0. Applied in the order given.
        code_cols: Members of ``categorical_cols`` stored as numbers; missing
            values become the most frequent code.
        drop_first: Drop each column's first category after encoding. Unknown
            categories at prediction time are then encoded like that first
            category; scikit-learn reports them with a warning.
        dense: Always return a dense matrix. Required by the gradient-boosting
            models. Otherwise scikit-learn's default applies: the output is
            sparse when fewer than 30% of its entries are non-zero, which is
            the case for the Ames columns.

    Returns:
        An unfitted ``sklearn.compose.ColumnTransformer``.

    Raises:
        ValueError: No columns were given, or a ``zero_fill_cols`` /
            ``code_cols`` entry is not in the corresponding list.
    """
    numeric = list(numeric_cols)
    categorical = list(categorical_cols)
    if not numeric and not categorical:
        raise ValueError("build_preprocessor needs at least one feature column")
    _require_subset(zero_fill_cols, numeric, "zero_fill_cols", "numeric_cols")
    _require_subset(code_cols, categorical, "code_cols", "categorical_cols")

    zero_fill = [c for c in zero_fill_cols if c in numeric]
    median_fill = [c for c in numeric if c not in zero_fill]
    codes = [c for c in categorical if c in set(code_cols)]
    text = [c for c in categorical if c not in set(code_cols)]

    transformers: list[tuple[str, Any, list[str]]] = []
    if median_fill:
        transformers.append(
            ("numeric", _numeric_branch(SimpleImputer(strategy="median"), scale), median_fill)
        )
    if zero_fill:
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        transformers.append(("numeric_zero_fill", _numeric_branch(imputer, scale), zero_fill))
    if text:
        imputer = SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)
        branch = _categorical_branch(imputer, drop_first, dense)
        transformers.append(("categorical", branch, text))
    if codes:
        imputer = SimpleImputer(strategy="most_frequent")
        branch = _categorical_branch(imputer, drop_first, dense)
        transformers.append(("categorical_codes", branch, codes))
    sparse_threshold = 0.0 if dense else 0.3  # 0.3 is scikit-learn's default
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=sparse_threshold)


def _numeric_branch(imputer: Any, scale: bool) -> Any:
    steps: list[tuple[str, Any]] = [("impute", imputer)]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _categorical_branch(imputer: Any, drop_first: bool, dense: bool) -> Any:
    encoder = OneHotEncoder(
        handle_unknown="ignore", drop="first" if drop_first else None, sparse_output=not dense
    )
    return Pipeline(
        [
            ("normalise", FunctionTransformer(normalise_missing)),
            ("impute", imputer),
            ("encode", encoder),
        ]
    )


def normalise_missing(values: Any) -> Any:
    """Return ``values`` as an object array with every missing marker as NaN.

    ``SimpleImputer`` only recognises NaN. In an object column - what pandas 2
    builds from Python objects - ``None`` and ``pd.NA`` would otherwise survive
    imputation and become a category of their own.
    """
    # A copy: under pandas 3 copy-on-write, np.asarray returns a read-only view.
    array = np.array(values, dtype=object)
    array[pd.isna(array)] = np.nan
    return array


def _require_subset(items: Sequence[str], pool: Sequence[str], name: str, pool_name: str) -> None:
    stray = [c for c in items if c not in pool]
    if stray:
        raise ValueError(f"{name} must be a subset of {pool_name}; not found: {', '.join(stray)}")
