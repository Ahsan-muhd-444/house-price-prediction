"""Derived features for the Ames data.

Each function takes a frame with the raw Ames columns and returns one new
column as a Series, without modifying its input. :func:`add_engineered_features`
applies all of them at once and is what the training pipeline uses when
``TrainConfig.engineer_features`` is enabled, so a saved model recomputes these
columns from raw inputs at prediction time.

The original notebook did not engineer features; these are the standard Ames
derivations. They are opt-in so that the default models reproduce the notebook's
published results exactly. See ``docs/benchmarks.md`` for their effect.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .errors import AmesSchemaError


def total_square_footage(frame: pd.DataFrame) -> pd.Series:
    """Basement plus first- and second-floor area, in square feet.

    A missing basement area means there is no basement, so it counts as 0.
    """
    parts = frame[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].fillna(0)
    return _named(parts.sum(axis=1), "TotalSF")


def house_age(frame: pd.DataFrame) -> pd.Series:
    """Years between construction and sale, never negative."""
    return _named((frame["YrSold"] - frame["YearBuilt"]).clip(lower=0), "HouseAge")


def remodel_age(frame: pd.DataFrame) -> pd.Series:
    """Years between the last remodel (or construction) and sale.

    One record in the Kaggle data lists a remodel after the sale year; a
    negative age is meaningless, so it is clipped to 0.
    """
    return _named((frame["YrSold"] - frame["YearRemodAdd"]).clip(lower=0), "RemodAge")


def total_bathrooms(frame: pd.DataFrame) -> pd.Series:
    """Full bathrooms plus half of the half-baths, above ground and in the basement."""
    baths = frame[["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]].fillna(0)
    total = baths["FullBath"] + baths["BsmtFullBath"]
    total = total + 0.5 * (baths["HalfBath"] + baths["BsmtHalfBath"])
    return _named(total, "TotalBathrooms")


def total_porch_area(frame: pd.DataFrame) -> pd.Series:
    """Combined open, enclosed, three-season and screened porch area."""
    porches = frame[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].fillna(0)
    return _named(porches.sum(axis=1), "TotalPorchSF")


#: Engineered column name -> function computing it, in output order.
ENGINEERED_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "TotalSF": total_square_footage,
    "HouseAge": house_age,
    "RemodAge": remodel_age,
    "TotalBathrooms": total_bathrooms,
    "TotalPorchSF": total_porch_area,
}

# fmt: off
#: Raw columns the engineered features are computed from.
SOURCE_COLUMNS: tuple[str, ...] = (
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "YrSold", "YearBuilt", "YearRemodAdd",
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
    "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
)
# fmt: on


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with every engineered column appended.

    Existing columns with the same names are overwritten, so applying the
    function twice gives the same result as applying it once.

    Raises:
        AmesSchemaError: A raw column needed by one of the features is missing.
    """
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise AmesSchemaError("Cannot engineer features; missing column(s): " + ", ".join(missing))
    result: pd.DataFrame = frame.copy()
    for name, compute in ENGINEERED_FEATURES.items():
        result[name] = compute(frame)
    return result


def _named(values: pd.Series, name: str) -> pd.Series:
    # Series.rename is typed as returning Any (its overloads cover inplace=True).
    named: pd.Series = values.rename(name)
    return named
