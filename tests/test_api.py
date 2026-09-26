from __future__ import annotations

import math
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from amesml import __version__, load_model, predict
from amesml.api import EXAMPLE_HOUSE, MODEL_PATH_ENV, app, create_app


def records(frame: pd.DataFrame, rows: int = 3) -> list[dict[str, Any]]:
    """Rows as JSON-ready dicts: NaN becomes None, the target and id are dropped."""
    subset = frame.drop(columns=["Id", "SalePrice"]).head(rows).astype(object)
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in subset.to_dict(orient="records")
    ]


@pytest.fixture(scope="module")
def client(regression_dir: Path, classification_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app([regression_dir, classification_dir])) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "tasks": ["classification", "regression"],
    }


def test_model_info(client: TestClient) -> None:
    info = client.get("/model/info").json()
    assert set(info) == {"regression", "classification"}
    assert info["classification"]["threshold"] > 0
    assert info["regression"]["config"]["model"] == "linear"


def test_predict_matches_the_python_api(
    client: TestClient, regression_dir: Path, ames_frame: pd.DataFrame
) -> None:
    response = client.post("/predict", json={"records": records(ames_frame)})
    assert response.status_code == 200, response.text
    body = response.json()
    expected = predict(load_model(regression_dir), ames_frame.head(3)).values
    assert body["predictions"] == pytest.approx(expected.round(2).tolist())
    assert body["model"]["task"] == "regression"
    assert body["warnings"] == []


def test_partial_records_use_training_defaults(client: TestClient) -> None:
    response = client.post("/predict", json={"records": [EXAMPLE_HOUSE, {"GrLivArea": 900}]})
    assert response.status_code == 200, response.text
    typical, small = response.json()["predictions"]
    assert all(math.isfinite(p) for p in (typical, small))
    assert typical > small  # 1,710 sq ft of good quality versus 900 sq ft of average


def test_classification(client: TestClient, ames_frame: pd.DataFrame) -> None:
    response = client.post("/predict/class", json={"records": records(ames_frame, 4)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["threshold"] > 0
    assert len(body["predictions"]) == 4
    for item in body["predictions"]:
        assert item["label"] in (0, 1)
        assert item["above_threshold"] is bool(item["label"])
        assert 0.0 <= item["probability"] <= 1.0


def test_unseen_categories_come_back_as_warnings(client: TestClient) -> None:
    response = client.post("/predict", json={"records": [{"Neighborhood": "Atlantis"}]})
    assert response.status_code == 200
    assert response.json()["warnings"] == [
        "Neighborhood: value(s) Atlantis were not seen in training and are treated as the "
        "column's reference category"
    ]


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"records": [{"GrLivAre": 1500}]}, "Unknown feature(s): GrLivAre"),
        ({"records": [{"LotArea": "huge"}]}, "'LotArea' must be numeric"),
        ({"records": []}, "at least 1 item"),
        ({"rows": []}, "records"),
    ],
)
def test_invalid_requests_are_422(
    client: TestClient, payload: dict[str, Any], fragment: str
) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert fragment in response.text


def test_too_many_records(client: TestClient) -> None:
    response = client.post("/predict", json={"records": [{}] * 1001})
    assert response.status_code == 422


def test_missing_task_is_503(regression_dir: Path) -> None:
    with TestClient(create_app([regression_dir])) as only_regression:
        response = only_regression.post("/predict/class", json={"records": [{}]})
    assert response.status_code == 503
    assert "No classification model is loaded" in response.json()["detail"]


def test_models_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, regression_dir: Path, classification_dir: Path
) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV, f" {regression_dir} , {classification_dir} ")
    with TestClient(create_app()) as from_env:
        assert from_env.get("/health").json()["tasks"] == ["classification", "regression"]


def test_startup_fails_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_PATH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="No model to serve"), TestClient(create_app()):
        pass


def test_startup_fails_with_two_models_for_one_task(regression_dir: Path) -> None:
    with (
        pytest.raises(RuntimeError, match="Two regression models"),
        TestClient(create_app([regression_dir, regression_dir])),
    ):
        pass


def test_model_without_metadata_is_still_served(regression_dir: Path, tmp_path: Path) -> None:
    shutil.copy(regression_dir / "model.joblib", tmp_path / "model.joblib")
    with TestClient(create_app([tmp_path])) as bare:
        response = bare.post("/predict", json={"records": [{}]})
    assert response.status_code == 200
    assert response.json()["model"] == {
        "task": "regression",
        "model": None,
        "estimator": None,
        "amesml_version": None,
        "created_at": None,
    }


def test_module_level_app_is_importable() -> None:
    assert app.title == "amesml"
    paths = set(app.openapi()["paths"])
    assert paths == {"/health", "/model/info", "/predict", "/predict/class"}
