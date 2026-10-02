from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from amesml import (
    ModelLoadError,
    TrainConfig,
    __version__,
    load_metadata,
    load_model,
    predict,
    save_model,
    train,
    train_regressor,
)


def test_train_result_save_writes_three_files(regression_dir: Path) -> None:
    assert sorted(p.name for p in regression_dir.iterdir()) == [
        "metadata.json",
        "metrics.json",
        "model.joblib",
    ]
    metrics = json.loads((regression_dir / "metrics.json").read_text("utf-8"))
    assert set(metrics) == {"mae", "rmse", "r2", "mape", "n_samples"}


def test_metadata_contents(regression_dir: Path, ames_frame: pd.DataFrame) -> None:
    metadata = load_metadata(regression_dir)
    assert metadata["schema_version"] == 1
    assert metadata["amesml_version"] == __version__
    assert (metadata["task"], metadata["model"]) == ("regression", "linear")
    assert metadata["estimator"] == "LinearRegression"
    assert metadata["threshold"] is None
    assert metadata["config"] == TrainConfig().to_dict()
    assert metadata["metrics"]["n_samples"] == 40
    assert len(metadata["features"]["inputs"]) == 79
    assert metadata["features"]["engineered"] == []
    assert metadata["feature_defaults"]["PoolQC"] is None
    assert metadata["dataset"]["rows"] == 200
    assert (metadata["dataset"]["train_rows"], metadata["dataset"]["test_rows"]) == (160, 40)
    assert len(metadata["dataset"]["sha256"]) == 64
    assert len(metadata["training_data_sha256"]) == 64
    assert len(metadata["artifact"]["sha256"]) == 64
    assert {"python", "scikit-learn", "pandas", "numpy", "joblib"} <= set(metadata["environment"])


def test_classifier_metadata_records_the_threshold(classification_dir: Path) -> None:
    metadata = load_metadata(classification_dir)
    assert metadata["task"] == "classification"
    assert metadata["threshold"] == load_model(classification_dir).threshold_
    assert metadata["metrics"]["confusion_matrix"][0][0] >= 0


def test_round_trip_gives_identical_predictions(
    regression_dir: Path, ames_frame: pd.DataFrame
) -> None:
    pipeline = train(ames_frame, TrainConfig()).pipeline
    loaded = load_model(regression_dir)
    np.testing.assert_array_equal(
        predict(pipeline, ames_frame).values, predict(loaded, ames_frame).values
    )
    assert load_model(regression_dir / "model.joblib").task_ == "regression"


def test_save_to_a_joblib_path(tmp_path: Path, ames_frame: pd.DataFrame) -> None:
    model_file = save_model(train_regressor(ames_frame), tmp_path / "custom.joblib")
    assert model_file == tmp_path / "custom.joblib"
    assert load_metadata(model_file)["artifact"]["file"] == "custom.joblib"
    assert load_model(model_file).task_ == "regression"


def test_metadata_is_strict_json(tmp_path: Path, ames_frame: pd.DataFrame) -> None:
    save_model(train_regressor(ames_frame), tmp_path, metrics={"bad": float("nan")})
    text = (tmp_path / "metadata.json").read_text("utf-8")
    assert "NaN" not in text
    assert json.loads(text)["metrics"] == {"bad": None}


def test_tampered_model_is_refused(tmp_path: Path, ames_frame: pd.DataFrame) -> None:
    save_model(train_regressor(ames_frame), tmp_path)
    other = train_regressor(ames_frame, TrainConfig(model="ridge"))
    joblib.dump(other, tmp_path / "model.joblib")
    with pytest.raises(ModelLoadError, match="does not match the checksum"):
        load_model(tmp_path)


def test_model_without_metadata_still_loads(tmp_path: Path, ames_frame: pd.DataFrame) -> None:
    save_model(train_regressor(ames_frame), tmp_path)
    (tmp_path / "metadata.json").unlink()
    assert load_model(tmp_path).task_ == "regression"
    with pytest.raises(ModelLoadError, match="No metadata.json"):
        load_metadata(tmp_path)


@pytest.mark.parametrize("content", ["{broken", "[1, 2]"])
def test_unreadable_metadata(tmp_path: Path, ames_frame: pd.DataFrame, content: str) -> None:
    save_model(train_regressor(ames_frame), tmp_path)
    (tmp_path / "metadata.json").write_text(content, encoding="utf-8")
    with pytest.raises(ModelLoadError):
        load_metadata(tmp_path)


def test_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="No model found"):
        load_model(tmp_path / "nothing")
    with pytest.raises(ModelLoadError, match="No model found"):
        load_model(tmp_path)  # an existing directory without model.joblib


def test_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "model.joblib").write_bytes(b"not a pickle")
    with pytest.raises(ModelLoadError, match="Could not load"):
        load_model(tmp_path)


def test_foreign_pipeline_is_refused(tmp_path: Path) -> None:
    plain = Pipeline([("model", LinearRegression())])
    with pytest.raises(ValueError, match="pipeline returned by amesml"):
        save_model(plain, tmp_path)
    joblib.dump(plain, tmp_path / "model.joblib")
    with pytest.raises(ModelLoadError, match="not an amesml model"):
        load_model(tmp_path)
