"""Training is deterministic: the same seed must give the same model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amesml import TrainConfig, predict, train

MODELS = ("linear", "ridge", "lasso", "gbm", "xgboost", "lightgbm")


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("model", MODELS)
def test_same_seed_same_predictions(ames_frame: pd.DataFrame, task: str, model: str) -> None:
    config = TrainConfig(task=task, model=model, random_state=7)
    first, second = train(ames_frame, config), train(ames_frame, config)
    np.testing.assert_array_equal(
        predict(first.pipeline, ames_frame).values, predict(second.pipeline, ames_frame).values
    )
    assert first.metrics == second.metrics
    assert first.dataset_sha256 == second.dataset_sha256


def test_different_seed_changes_the_split(ames_frame: pd.DataFrame) -> None:
    first = train(ames_frame, TrainConfig(random_state=1))
    second = train(ames_frame, TrainConfig(random_state=2))
    assert first.metrics != second.metrics
