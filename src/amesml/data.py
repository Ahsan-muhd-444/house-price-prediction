"""Loading, validating and splitting the Ames Housing dataset.

The dataset is the Kaggle "House Prices: Advanced Regression Techniques"
``train.csv``: 1,460 sales from Ames, Iowa, with 79 candidate features, an
``Id`` and the target ``SalePrice`` (De Cock, 2011).

In this file the string ``"NA"`` is a value, not an accident: it means the
house has no pool, no alley access, no basement, and so on. Elsewhere the
literal word ``"None"`` is a real category (``MasVnrType == "None"``), so pandas'
default list of missing-value markers - which includes ``"None"`` - would
silently destroy information. :func:`read_ames_csv` therefore treats only
``"NA"`` and empty fields as missing, exactly as the original notebook did.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import pandas as pd
import requests
from pandas.api import types as ptypes
from sklearn.model_selection import train_test_split

from .errors import AmesSchemaError, DatasetDownloadError

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike[str]]

#: Name of the target column.
TARGET = "SalePrice"
#: Name of the identifier column, never used as a feature.
ID_COLUMN = "Id"
#: Where :func:`load_ames` looks for the data when no path is given.
DEFAULT_DATA_PATH = Path("data") / "train.csv"

#: Immutable copy of the Kaggle training file, pinned to a commit of this
#: project's repository so the bytes can never change underneath a checksum.
DATASET_URL = (
    "https://raw.githubusercontent.com/Ahsan-muhd-444/house-price-prediction/"
    "cdfd5dadba47fbdbb348dbf2650e205c567260f1/data/train.csv"
)
#: SHA-256 of the file at :data:`DATASET_URL`.
DATASET_SHA256 = "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"

#: Only these markers mean "missing"; see the module docstring.
NA_VALUES: tuple[str, ...] = ("NA", "")

# fmt: off
#: Feature columns stored as numbers. ``MSSubClass`` and ``MoSold`` are numeric
#: codes for categories; :class:`~amesml.config.TrainConfig` treats them as such.
NUMERIC_COLUMNS: tuple[str, ...] = (
    "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt",
    "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold",
)
# fmt: on

# fmt: off
#: Feature columns stored as text.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
    "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
    "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition",
)
# fmt: on

#: All 79 candidate features, in no particular order.
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def read_ames_csv(path: PathLike) -> pd.DataFrame:
    """Read an Ames-format CSV with the dataset's missing-value conventions.

    No validation happens here, so this also reads prediction inputs that lack
    the target column. Use :func:`load_ames` for the training data.

    Args:
        path: CSV file to read.

    Returns:
        The parsed data.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=list(NA_VALUES))


def validate_ames(frame: pd.DataFrame, *, require_target: bool = True) -> None:
    """Check that ``frame`` has the Ames columns with usable types.

    Args:
        frame: Data to check.
        require_target: Also require a numeric, positive, complete ``SalePrice``.

    Raises:
        AmesSchemaError: Describing every problem found, so they can all be
            fixed in one pass.
    """
    if frame.empty:
        raise AmesSchemaError("The dataset is empty")

    duplicated = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
    if duplicated:
        raise AmesSchemaError(f"Duplicate column name(s): {', '.join(duplicated)}")

    problems: list[str] = []
    required = list(FEATURE_COLUMNS) + ([TARGET] if require_target else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        problems.append(f"missing {len(missing)} column(s): {_preview(missing)}")

    for column in NUMERIC_COLUMNS + ((TARGET,) if require_target else ()):
        if column in frame.columns and not ptypes.is_numeric_dtype(frame[column]):
            example = _first_non_numeric(frame[column])
            problems.append(f"column {column!r} must be numeric but contains {example!r}")

    if require_target and TARGET in frame.columns and ptypes.is_numeric_dtype(frame[TARGET]):
        target = frame[TARGET]
        if target.isna().any():
            problems.append(f"{TARGET} has {int(target.isna().sum())} missing value(s)")
        elif (target <= 0).any():
            problems.append(f"{TARGET} must be positive; found {int((target <= 0).sum())} row(s)")

    if problems:
        raise AmesSchemaError("Invalid Ames data: " + "; ".join(problems))


def load_ames(
    path: PathLike | None = None, *, download: bool = True, validate: bool = True
) -> pd.DataFrame:
    """Load the Ames Housing training data.

    Args:
        path: CSV file to read. When omitted, ``data/train.csv`` in the current
            directory is used, and downloaded first if it is not there.
        download: Allow the download when the default file is missing.
        validate: Check the columns and types with :func:`validate_ames`.

    Returns:
        One row per sale, including ``Id`` and ``SalePrice``.

    Raises:
        FileNotFoundError: ``path`` does not exist (an explicit path is never
            downloaded to).
        DatasetDownloadError: The default file had to be downloaded and could
            not be.
        AmesSchemaError: The data failed validation.
    """
    if path is None:
        csv_path = DEFAULT_DATA_PATH
        if not csv_path.exists():
            if not download:
                raise FileNotFoundError(f"{csv_path} not found and download=False")
            download_ames(csv_path)
    else:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = read_ames_csv(csv_path)
    logger.debug("Loaded %d rows and %d columns from %s", *frame.shape, csv_path)
    if validate:
        validate_ames(frame)
    return frame


def download_ames(
    destination: PathLike | None = None,
    *,
    url: str = DATASET_URL,
    sha256: str | None = DATASET_SHA256,
    timeout: float = 60.0,
    force: bool = False,
) -> Path:
    """Download the training CSV and verify its checksum.

    The file is written to a temporary name and renamed only after the checksum
    matches, so an interrupted or tampered download never replaces good data.

    Args:
        destination: Where to save the file; defaults to ``data/train.csv``.
        url: Source URL.
        sha256: Expected SHA-256 hex digest, or ``None`` to skip the check
            (only sensible for a custom ``url`` you trust).
        timeout: Network timeout in seconds.
        force: Download even if ``destination`` already exists.

    Returns:
        The path of the downloaded file.

    Raises:
        DatasetDownloadError: The request failed or the checksum did not match.
    """
    target = Path(destination) if destination is not None else DEFAULT_DATA_PATH
    if target.exists() and not force:
        logger.info("%s already exists; skipping download", target)
        return target

    logger.info("Downloading the Ames dataset from %s", url)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DatasetDownloadError(f"Could not download {url}: {exc}") from exc

    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    if sha256 is not None and digest != sha256.lower():
        raise DatasetDownloadError(
            f"Checksum mismatch for {url}: expected {sha256}, got {digest}. The file was not saved."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(content)
    os.replace(partial, target)
    logger.info("Saved %s (%d bytes, sha256 %s)", target, len(content), digest)
    return target


def train_test_split_df(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Sequence[Any] | pd.Series | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole rows into training and test frames.

    With the defaults this selects the same rows as the notebook's
    ``train_test_split(X, y, test_size=0.2, random_state=42)``.

    Args:
        frame: Data to split.
        test_size: Fraction of rows in the test frame.
        random_state: Seed for the shuffle.
        stratify: Labels to stratify on (use this for classification so both
            classes keep their proportions), or the name of a column in
            ``frame`` holding them.

    Returns:
        ``(train, test)``, each keeping the original row index.
    """
    labels = frame[stratify] if isinstance(stratify, str) else stratify
    train, test = train_test_split(
        frame, test_size=test_size, random_state=random_state, stratify=labels
    )
    return train, test


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Return a SHA-256 of the frame's contents, independent of platform.

    Used in model metadata to record exactly which data a model was trained on.
    The frame is serialised as CSV with ``\\n`` line endings so the same data
    hashes identically on every operating system.
    """
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _preview(items: Sequence[str], limit: int = 8) -> str:
    shown = ", ".join(items[:limit])
    return shown if len(items) <= limit else f"{shown}, ... (+{len(items) - limit} more)"


def _first_non_numeric(series: pd.Series) -> object:
    for value in series.dropna():
        try:
            float(value)
        except (TypeError, ValueError):
            return value
    return series.dtype
