from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amesml import TrainConfig, train
from amesml.data import train_test_split_df
from amesml.plots import (
    FIGURE_FILES,
    generate_all,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_missing_values,
    plot_residuals,
    plot_target_distribution,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    assert path.stat().st_size > 5_000  # an actual figure, not an empty canvas


def test_target_distribution(ames_frame: pd.DataFrame, tmp_path: Path) -> None:
    assert_png(plot_target_distribution(ames_frame, tmp_path / "a" / "target.png"))


def test_missing_values(ames_frame: pd.DataFrame, tmp_path: Path) -> None:
    assert_png(plot_missing_values(ames_frame, tmp_path / "missing.png"))


def test_missing_values_when_nothing_is_missing(tmp_path: Path) -> None:
    complete = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = plot_missing_values(complete, tmp_path / "none.png")
    assert path.read_bytes()[:8] == PNG_SIGNATURE


def test_residuals(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(200_000, 50_000, 100)
    assert_png(plot_residuals(truth, truth + rng.normal(0, 10_000, 100), tmp_path / "r.png"))


def test_confusion_matrix(tmp_path: Path) -> None:
    assert_png(plot_confusion_matrix([[152, 10], [6, 124]], tmp_path / "cm.png"))


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_feature_importance(ames_frame: pd.DataFrame, tmp_path: Path, task: str) -> None:
    # A handful of features keeps the permutations fast; generate_all below
    # exercises the full feature set.
    config = TrainConfig(
        task=task,
        model="gbm",
        numeric_features=("GrLivArea", "OverallQual", "YearBuilt", "LotArea", "GarageCars"),
        categorical_features=("Neighborhood",),
    )
    result = train(ames_frame, config)
    _, test = train_test_split_df(ames_frame)
    path = plot_feature_importance(result.pipeline, test, tmp_path / "fi.png", n_repeats=2)
    assert_png(path)


def test_generate_all(ames_frame: pd.DataFrame, tmp_path: Path) -> None:
    written = generate_all(ames_frame, tmp_path / "images", importance_repeats=2)
    assert set(written) == set(FIGURE_FILES)
    for name, path in written.items():
        assert path.name == FIGURE_FILES[name]
        assert_png(path)
