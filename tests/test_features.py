from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amesml import AmesSchemaError, add_engineered_features
from amesml.features import (
    ENGINEERED_FEATURES,
    SOURCE_COLUMNS,
    house_age,
    remodel_age,
    total_bathrooms,
    total_porch_area,
    total_square_footage,
)


@pytest.fixture
def houses() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TotalBsmtSF": [800.0, np.nan],
            "1stFlrSF": [1000, 900],
            "2ndFlrSF": [500, 0],
            "YrSold": [2008, 2009],
            "YearBuilt": [1990, 2009],
            "YearRemodAdd": [2000, 2010],  # second house: remodel recorded after the sale
            "FullBath": [2, 1],
            "HalfBath": [1, 0],
            "BsmtFullBath": [1, np.nan],
            "BsmtHalfBath": [1, 0],
            "OpenPorchSF": [40, 0],
            "EnclosedPorch": [0, 10],
            "3SsnPorch": [0, 0],
            "ScreenPorch": [20, 0],
        }
    )


def test_total_square_footage_treats_missing_basement_as_none(houses: pd.DataFrame) -> None:
    result = total_square_footage(houses)
    assert result.tolist() == [2300.0, 900.0]
    assert result.name == "TotalSF"


def test_ages(houses: pd.DataFrame) -> None:
    assert house_age(houses).tolist() == [18, 0]
    # 2009 - 2010 would be -1; a negative age is clipped to 0.
    assert remodel_age(houses).tolist() == [8, 0]


def test_total_bathrooms_counts_half_baths_as_half(houses: pd.DataFrame) -> None:
    # 2 + 1 + 0.5 * (1 + 1) = 4.0; 1 + 0 + 0.5 * 0 = 1.0 (missing basement bath = 0)
    assert total_bathrooms(houses).tolist() == [4.0, 1.0]


def test_total_porch_area(houses: pd.DataFrame) -> None:
    assert total_porch_area(houses).tolist() == [60, 10]


def test_add_engineered_features_appends_without_mutating(houses: pd.DataFrame) -> None:
    original = houses.copy()
    result = add_engineered_features(houses)
    assert list(result.columns) == list(houses.columns) + list(ENGINEERED_FEATURES)
    pd.testing.assert_frame_equal(houses, original)
    pd.testing.assert_frame_equal(add_engineered_features(result), result)  # idempotent


def test_missing_source_columns_are_listed(houses: pd.DataFrame) -> None:
    with pytest.raises(AmesSchemaError, match="YrSold, YearBuilt"):
        add_engineered_features(houses.drop(columns=["YrSold", "YearBuilt"]))


def test_source_columns_cover_every_feature(houses: pd.DataFrame) -> None:
    assert set(houses.columns) == set(SOURCE_COLUMNS)
