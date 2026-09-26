"""Shared fixtures.

The suite runs on a synthetic Ames-shaped frame so it needs no download and
finishes in seconds. Tests against the real Kaggle file live in
``test_notebook_parity.py`` and are skipped when ``data/train.csv`` is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amesml import TrainConfig, train
from amesml.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = REPO_ROOT / "data" / "train.csv"
NEIGHBORHOODS = ("NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt")


def make_ames_frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a frame with every Ames column, realistic gaps and a learnable price."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"Id": np.arange(1, n_rows + 1)}

    for column in NUMERIC_COLUMNS:
        data[column] = rng.integers(0, 50, n_rows)
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.choice(["Ex", "Gd", "TA", "Fa"], n_rows).astype(object)

    quality = rng.integers(3, 11, n_rows)
    living = rng.integers(700, 3200, n_rows)
    built = rng.integers(1900, 2009, n_rows)
    sold = rng.integers(2006, 2011, n_rows)
    built = np.minimum(built, sold)
    first_floor = (living * rng.uniform(0.5, 1.0, n_rows)).astype(int)
    basement_finished = rng.integers(0, 900, n_rows)
    basement_unfinished = rng.integers(0, 600, n_rows)
    neighborhood = rng.choice(NEIGHBORHOODS, n_rows)

    data.update(
        {
            "MSSubClass": rng.choice([20, 30, 50, 60, 70, 90, 120, 160], n_rows),
            "MoSold": rng.integers(1, 13, n_rows),
            "YrSold": sold,
            "YearBuilt": built,
            "YearRemodAdd": np.minimum(built + rng.integers(0, 30, n_rows), sold),
            "OverallQual": quality,
            "OverallCond": rng.integers(3, 10, n_rows),
            "GrLivArea": living,
            "1stFlrSF": first_floor,
            "2ndFlrSF": living - first_floor,
            "LowQualFinSF": np.zeros(n_rows, dtype=int),
            "BsmtFinSF1": basement_finished,
            "BsmtFinSF2": np.zeros(n_rows, dtype=int),
            "BsmtUnfSF": basement_unfinished,
            "TotalBsmtSF": basement_finished + basement_unfinished,
            "FullBath": rng.integers(1, 4, n_rows),
            "HalfBath": rng.integers(0, 2, n_rows),
            "BsmtFullBath": rng.integers(0, 2, n_rows),
            "BsmtHalfBath": rng.integers(0, 2, n_rows),
            "GarageCars": rng.integers(0, 4, n_rows),
            "LotArea": rng.integers(3000, 20000, n_rows),
            "Neighborhood": neighborhood.astype(object),
            "MasVnrType": rng.choice(["None", "BrkFace", "Stone"], n_rows).astype(object),
        }
    )

    # Gaps the way the real data has them.
    lot_frontage = rng.integers(40, 120, n_rows).astype(float)
    lot_frontage[rng.random(n_rows) < 0.15] = np.nan
    data["LotFrontage"] = lot_frontage
    garage_year = built.astype(float)
    garage_year[rng.random(n_rows) < 0.06] = np.nan
    data["GarageYrBlt"] = garage_year
    veneer = rng.integers(0, 400, n_rows).astype(float)
    veneer[rng.random(n_rows) < 0.03] = np.nan
    data["MasVnrArea"] = veneer
    for column, share_missing in (("PoolQC", 0.97), ("Alley", 0.9), ("Fence", 0.6)):
        values = np.asarray(data[column], dtype=object)
        values[rng.random(n_rows) < share_missing] = np.nan
        data[column] = values

    premiums = {
        "NAmes": 0,
        "CollgCr": 15_000,
        "OldTown": -10_000,
        "Edwards": -5_000,
        "Somerst": 20_000,
        "NridgHt": 45_000,
    }
    premium = pd.Series(neighborhood).map(premiums)
    price = (
        25_000
        + 18_000 * quality
        + 55 * living
        + 300 * (built - 1900)
        + premium.to_numpy()
        + rng.normal(0, 12_000, n_rows)
    )
    data["SalePrice"] = np.maximum(price, 35_000).round().astype(int)
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def ames_frame() -> pd.DataFrame:
    """Synthetic Ames data. Treat as read-only; copy before modifying."""
    return make_ames_frame()


@pytest.fixture(scope="session")
def ames_csv(tmp_path_factory: pytest.TempPathFactory, ames_frame: pd.DataFrame) -> Path:
    """The synthetic data written the way Kaggle writes it, with ``NA`` for gaps."""
    path = tmp_path_factory.mktemp("data") / "train.csv"
    ames_frame.to_csv(path, index=False, na_rep="NA")
    return path


@pytest.fixture(scope="session")
def regression_dir(tmp_path_factory: pytest.TempPathFactory, ames_frame: pd.DataFrame) -> Path:
    """A saved linear regression model directory."""
    return train(ames_frame, TrainConfig()).save(tmp_path_factory.mktemp("models") / "reg")


@pytest.fixture(scope="session")
def classification_dir(tmp_path_factory: pytest.TempPathFactory, ames_frame: pd.DataFrame) -> Path:
    """A saved logistic regression classifier directory."""
    result = train(ames_frame, TrainConfig(task="classification"))
    return result.save(tmp_path_factory.mktemp("models") / "clf")
