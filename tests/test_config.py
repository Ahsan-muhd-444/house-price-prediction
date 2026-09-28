from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from amesml import ConfigError, TrainConfig
from amesml.config import DEFAULT_ALPHAS


def test_defaults_reproduce_the_notebook() -> None:
    config = TrainConfig()
    assert (config.task, config.model, config.target) == ("regression", "linear", "SalePrice")
    assert (config.test_size, config.random_state) == (0.2, 42)
    assert config.threshold == "median"
    assert config.drop_first is True
    assert config.engineer_features is False
    assert config.zero_fill_features == ("GarageYrBlt", "MasVnrArea")
    assert config.categorical_codes == ("MSSubClass", "MoSold")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (TrainConfig(model="linear"), None),
        (TrainConfig(model="gbm", alpha=5.0), None),
        (TrainConfig(model="ridge"), DEFAULT_ALPHAS[("regression", "ridge")]),
        (TrainConfig(model="lasso"), DEFAULT_ALPHAS[("regression", "lasso")]),
        (TrainConfig(task="classification", model="ridge"), 30.0),
        (TrainConfig(model="ridge", alpha=2.5), 2.5),
    ],
)
def test_resolved_alpha(config: TrainConfig, expected: float | None) -> None:
    assert config.resolved_alpha == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "forecasting"}, "task must be one of"),
        ({"model": "xgboost"}, "model must be one of"),
        ({"target": ""}, "target must be"),
        ({"id_column": 3}, "id_column"),
        ({"test_size": 0}, "test_size"),
        ({"test_size": 1}, "test_size"),
        ({"test_size": True}, "test_size"),
        ({"test_size": "0.2"}, "test_size"),
        ({"random_state": -1}, "random_state"),
        ({"random_state": 1.5}, "random_state"),
        ({"random_state": True}, "random_state"),
        ({"threshold": "mean"}, "threshold"),
        ({"threshold": 0}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"threshold": float("inf")}, "threshold"),
        ({"stratify": True}, "stratify only applies to classification"),
        ({"stratify": "yes"}, "stratify must be true or false"),
        ({"alpha": 0}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"max_iter": 0}, "max_iter"),
        ({"numeric_features": "GrLivArea"}, "must be a list"),
        ({"numeric_features": ["GrLivArea", ""]}, "non-empty column names"),
        (
            {"numeric_features": ["A"], "categorical_features": ["A"]},
            "both numeric and categorical",
        ),
        ({"numeric_features": ["SalePrice"]}, "may not include the target"),
        ({"categorical_features": ["Id"]}, "may not include the target or id"),
    ],
)
def test_invalid_options_are_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        TrainConfig(**changes)


def test_numeric_threshold_is_accepted() -> None:
    assert TrainConfig(task="classification", threshold=200_000).threshold == 200_000


def test_lists_become_tuples_so_the_config_is_hashable() -> None:
    config = TrainConfig(numeric_features=["GrLivArea", "OverallQual"])  # type: ignore[arg-type]
    assert config.numeric_features == ("GrLivArea", "OverallQual")
    assert hash(config) == hash(TrainConfig(numeric_features=("GrLivArea", "OverallQual")))


def test_dict_round_trip() -> None:
    config = TrainConfig(
        task="classification",
        model="lasso",
        stratify=True,
        threshold=180_000.0,
        categorical_features=("Neighborhood",),
        alpha=2.0,
    )
    data = config.to_dict()
    assert data["categorical_features"] == ["Neighborhood"]
    json.dumps(data)  # must be serialisable as-is
    assert TrainConfig.from_dict(data) == config


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ConfigError, match="Unknown configuration option.*learning_rate"):
        TrainConfig.from_dict({"model": "gbm", "learning_rate": 0.1})


def test_from_dict_rejects_non_mappings() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        TrainConfig.from_dict(["model", "gbm"])  # type: ignore[arg-type]


def test_replace_validates_and_rejects_unknown_options() -> None:
    config = TrainConfig()
    assert config.replace(model="ridge").model == "ridge"
    with pytest.raises(ConfigError, match="test_size"):
        config.replace(test_size=2)
    with pytest.raises(ConfigError, match="Unknown configuration option"):
        config.replace(n_estimators=10)


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_from_yaml(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"config{suffix}"
    path.write_text(
        "task: classification\nmodel: gbm\nthreshold: median\nstratify: true\n"
        "engineer_features: true\nzero_fill_features: [MasVnrArea]\n",
        encoding="utf-8",
    )
    config = TrainConfig.from_file(path)
    assert config.task == "classification"
    assert config.model == "gbm"
    assert config.stratify and config.engineer_features
    assert config.zero_fill_features == ("MasVnrArea",)


def test_from_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "ridge", "alpha": 4}), encoding="utf-8")
    config = TrainConfig.from_file(path)
    assert (config.model, config.alpha) == ("ridge", 4)


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert TrainConfig.from_file(path) == TrainConfig()


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("config.toml", "model = 'gbm'", "Unsupported configuration format"),
        ("bad.yaml", "model: [unclosed", "Could not parse"),
        ("bad.json", "{not json", "Could not parse"),
    ],
)
def test_from_file_errors(tmp_path: Path, name: str, content: str, message: str) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        TrainConfig.from_file(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        TrainConfig.from_file(tmp_path / "nope.yaml")
