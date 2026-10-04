from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from amesml import build_preprocessor, infer_feature_types
from amesml.preprocess import MISSING_CATEGORY


def test_infer_feature_types(ames_frame: pd.DataFrame) -> None:
    types = infer_feature_types(ames_frame)
    assert "Id" not in types.columns and "SalePrice" not in types.columns
    assert "GrLivArea" in types.numeric
    assert "Neighborhood" in types.categorical
    # Numeric category codes move to the end of the categorical list.
    assert types.categorical[-2:] == ("MSSubClass", "MoSold")
    assert "MSSubClass" not in types.numeric
    assert len(types.columns) == 79


@pytest.mark.parametrize("dtype", [object, "string", "category"])
def test_text_dtypes_are_all_categorical(ames_frame: pd.DataFrame, dtype: object) -> None:
    # pandas 3 reads text as the "str" dtype, not object; neither may be missed.
    frame = ames_frame[["GrLivArea", "Neighborhood"]].astype({"Neighborhood": dtype})
    types = infer_feature_types(frame)
    assert types.numeric == ("GrLivArea",)
    assert types.categorical == ("Neighborhood",)


def test_categorical_codes_absent_from_the_frame_are_ignored() -> None:
    frame = pd.DataFrame({"LotArea": [1, 2], "Street": ["a", "b"]})
    assert infer_feature_types(frame).categorical == ("Street",)


@pytest.fixture
def small() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": [100.0, 200.0, np.nan, 400.0],
            "veneer": [10.0, np.nan, 30.0, 40.0],
            "zone": ["A", "B", None, "A"],
            "code": [20, 60, 60, 20],
        }
    )


def test_imputation_and_encoding(small: pd.DataFrame) -> None:
    transformer = build_preprocessor(
        ["area", "veneer"],
        ["zone", "code"],
        scale=False,
        zero_fill_cols=["veneer"],
        code_cols=["code"],
        dense=True,
    )
    matrix = transformer.fit_transform(small)
    assert matrix.shape == (4, 2 + 3 + 2)  # 2 numeric; zone: A/B/Missing; code: 20/60
    assert matrix[2, 0] == 200.0  # median of 100, 200 and 400
    assert matrix[1, 1] == 0.0  # zero-filled
    zone_encoder = transformer.named_transformers_["categorical"].named_steps["encode"]
    assert MISSING_CATEGORY in zone_encoder.categories_[0]


def test_scaling(small: pd.DataFrame) -> None:
    matrix = build_preprocessor(["area"], [], scale=True, dense=True).fit_transform(small)
    assert np.isclose(matrix[:, 0].mean(), 0.0)
    assert np.isclose(matrix[:, 0].std(), 1.0)


def test_drop_first_removes_one_level_per_column(small: pd.DataFrame) -> None:
    full = build_preprocessor([], ["zone", "code"], code_cols=["code"], dense=True)
    dropped = build_preprocessor(
        [], ["zone", "code"], code_cols=["code"], drop_first=True, dense=True
    )
    assert full.fit_transform(small).shape[1] - dropped.fit_transform(small).shape[1] == 2


def test_output_is_sparse_by_default_for_one_hot_heavy_data() -> None:
    frame = pd.DataFrame({"id": [str(i) for i in range(50)]})
    assert sparse.issparse(build_preprocessor([], ["id"]).fit_transform(frame))
    assert not sparse.issparse(build_preprocessor([], ["id"], dense=True).fit_transform(frame))


def test_unknown_categories_are_encoded_as_zeros(small: pd.DataFrame) -> None:
    transformer = build_preprocessor([], ["zone"], dense=True).fit(small)
    assert transformer.transform(pd.DataFrame({"zone": ["Z"]})).sum() == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"numeric_cols": [], "categorical_cols": []}, "at least one feature"),
        ({"numeric_cols": ["a"], "categorical_cols": [], "zero_fill_cols": ["b"]}, "zero_fill"),
        ({"numeric_cols": [], "categorical_cols": ["a"], "code_cols": ["b"]}, "code_cols"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, list[str]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_preprocessor(**kwargs)
