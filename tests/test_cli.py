from __future__ import annotations

import json
import runpy
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from amesml import __version__, load_model, predict
from amesml import cli as cli_module
from amesml.cli import app

try:
    # Click < 8.2 mixes stderr into stdout unless told otherwise.
    runner = CliRunner(mix_stderr=False)
except TypeError:  # Click 8.2 removed the option and always keeps stderr separate.
    runner = CliRunner()


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def metrics_line(output: str, label: str) -> str:
    return next(line for line in output.splitlines() if label in line)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, ames_csv: Path) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("cli-models")
    result = invoke("train", "--data", str(ames_csv), "--out", str(out))
    assert result.exit_code == 0, result.output
    return out / "regression-linear", result.stdout


def test_version() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.strip() == f"amesml {__version__}"


def test_help_lists_every_command() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("data", "train", "evaluate", "predict", "plots", "serve"):
        assert command in result.stdout


def test_train_writes_a_model_and_reports_metrics(trained: tuple[Path, str]) -> None:
    model_dir, output = trained
    assert {p.name for p in model_dir.iterdir()} == {
        "model.joblib",
        "metadata.json",
        "metrics.json",
    }
    assert "MAE" in output and "R²" in output and "Saved model to" in output


def test_train_classifier_with_custom_name(ames_csv: Path, tmp_path: Path) -> None:
    result = invoke(
        "-q",
        "train",
        "--data",
        str(ames_csv),
        "--task",
        "classification",
        "--model",
        "gbm",
        "--out",
        str(tmp_path),
        "--name",
        "best",
    )
    assert result.exit_code == 0, result.output
    assert "Accuracy" in result.stdout and "ROC-AUC" in result.stdout
    metadata = json.loads((tmp_path / "best" / "metadata.json").read_text("utf-8"))
    assert (metadata["task"], metadata["model"]) == ("classification", "gbm")


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("model", ["xgboost", "lightgbm"])
def test_train_boosting_models(ames_csv: Path, tmp_path: Path, task: str, model: str) -> None:
    result = invoke(
        "-q",
        "train",
        "--data",
        str(ames_csv),
        "--task",
        task,
        "--model",
        model,
        "--out",
        str(tmp_path),
    )
    assert result.exit_code == 0, result.output
    model_dir = tmp_path / f"{task}-{model}"
    metadata = json.loads((model_dir / "metadata.json").read_text("utf-8"))
    assert (metadata["task"], metadata["model"]) == (task, model)
    houses = pd.read_csv(ames_csv)
    assert predict(load_model(model_dir), houses).values.shape == (len(houses),)


def test_train_boosting_model_without_the_extra(
    ames_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "xgboost", None)
    result = invoke("train", "--data", str(ames_csv), "--model", "xgboost", "--out", str(tmp_path))
    assert result.exit_code == 1
    assert 'pip install "amesml[boosting]"' in result.stderr
    assert "Traceback" not in result.output


def test_config_file_with_cli_overrides(ames_csv: Path, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "task: classification\nstratify: true\nmodel: ridge\nengineer_features: true\n", "utf-8"
    )
    result = invoke(
        "train",
        "--data",
        str(ames_csv),
        "--config",
        str(config),
        "--task",
        "regression",
        "--model",
        "lasso",
        "--out",
        str(tmp_path),
    )
    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / "regression-lasso" / "metadata.json").read_text("utf-8"))
    assert saved["config"]["engineer_features"] is True
    assert saved["config"]["stratify"] is False  # dropped with the switch to regression


def test_train_with_a_missing_data_file(tmp_path: Path) -> None:
    result = invoke("train", "--data", str(tmp_path / "missing.csv"))
    assert result.exit_code == 1
    assert "Dataset not found" in result.stderr
    assert "Traceback" not in result.output


def test_train_with_a_missing_config_file(tmp_path: Path) -> None:
    result = invoke("train", "--config", str(tmp_path / "missing.yaml"))
    assert result.exit_code == 1
    assert "Configuration file not found" in result.stderr


def test_evaluate_warns_when_scoring_training_rows(
    trained: tuple[Path, str], ames_csv: Path
) -> None:
    model_dir, _ = trained
    result = invoke("evaluate", "--model", str(model_dir), "--data", str(ames_csv))
    assert result.exit_code == 0, result.output
    assert "--split test" in result.stderr
    assert "all rows" in result.stdout


def test_evaluate_test_split_reproduces_training_metrics(
    trained: tuple[Path, str], ames_csv: Path
) -> None:
    model_dir, training_output = trained
    result = invoke(
        "evaluate", "--model", str(model_dir), "--data", str(ames_csv), "--split", "test"
    )
    assert result.exit_code == 0, result.output
    for label in ("MAE", "RMSE", "R²", "Rows"):
        assert metrics_line(result.stdout, label) == metrics_line(training_output, label)
    train_rows = invoke(
        "evaluate", "--model", str(model_dir), "--data", str(ames_csv), "--split", "train"
    )
    assert "160" in metrics_line(train_rows.stdout, "Rows")


def test_evaluate_classifier(ames_csv: Path, tmp_path: Path) -> None:
    invoke("train", "--data", str(ames_csv), "--task", "classification", "--out", str(tmp_path))
    result = invoke(
        "evaluate", "--model", str(tmp_path / "classification-linear"), "--data", str(ames_csv)
    )
    assert result.exit_code == 0
    assert "Confusion matrix" in result.stdout


def test_evaluate_needs_the_target(trained: tuple[Path, str], tmp_path: Path) -> None:
    model_dir, _ = trained
    unlabelled = tmp_path / "unlabelled.csv"
    unlabelled.write_text("Id,GrLivArea\n1,1500\n", "utf-8")
    result = invoke("evaluate", "--model", str(model_dir), "--data", str(unlabelled))
    assert result.exit_code == 1
    assert "no SalePrice column" in result.stderr


def test_evaluate_needs_a_config_in_the_metadata(
    trained: tuple[Path, str], ames_csv: Path, tmp_path: Path
) -> None:
    model_dir, _ = trained
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "model.joblib").write_bytes((model_dir / "model.joblib").read_bytes())
    metadata = json.loads((model_dir / "metadata.json").read_text("utf-8"))
    del metadata["config"]
    (copy / "metadata.json").write_text(json.dumps(metadata), "utf-8")
    result = invoke("evaluate", "--model", str(copy), "--data", str(ames_csv))
    assert result.exit_code == 1
    assert "no training config" in result.stderr


def test_predict_to_stdout(trained: tuple[Path, str], ames_csv: Path) -> None:
    model_dir, _ = trained
    result = invoke("predict", "--model", str(model_dir), "--input", str(ames_csv))
    assert result.exit_code == 0, result.output
    predictions = pd.read_csv(StringIO(result.stdout))
    assert list(predictions.columns) == ["Id", "predicted_price"]
    assert len(predictions) == 200


def test_predict_classes_to_file(ames_csv: Path, tmp_path: Path) -> None:
    invoke("train", "--data", str(ames_csv), "--task", "classification", "--out", str(tmp_path))
    out = tmp_path / "preds" / "classes.csv"
    result = invoke(
        "predict",
        "--model",
        str(tmp_path / "classification-linear"),
        "--input",
        str(ames_csv),
        "--out",
        str(out),
    )
    assert result.exit_code == 0, result.output
    predictions = pd.read_csv(out)
    assert list(predictions.columns) == ["Id", "predicted_label", "probability_above_threshold"]


def test_predict_with_a_missing_model(tmp_path: Path, ames_csv: Path) -> None:
    result = invoke("predict", "--model", str(tmp_path / "none"), "--input", str(ames_csv))
    assert result.exit_code == 1
    assert "No model found" in result.stderr


def test_plots_command_wiring(
    monkeypatch: pytest.MonkeyPatch, ames_csv: Path, tmp_path: Path
) -> None:
    # The figures themselves are tested in test_plots.py; this checks the command.
    from amesml import plots

    calls: list[tuple[int, Path]] = []

    def fake_generate_all(frame: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
        calls.append((len(frame), out_dir))
        return {"target_distribution": out_dir / "amesml_target_distribution.png"}

    monkeypatch.setattr(plots, "generate_all", fake_generate_all)
    # Rich wraps to the console width, and a deep tmp_path pushes the filename
    # over the edge mid-word on runners with long temp paths. The console is
    # built at import, so COLUMNS set now would not reach it; widen the console
    # object itself, for this test only (monkeypatch restores it).
    monkeypatch.setattr(cli_module.console, "width", 200)
    result = invoke("plots", "--data", str(ames_csv), "--out", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert calls == [(200, tmp_path)]
    assert "amesml_target_distribution.png" in result.stdout


def test_data_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_download(destination: Path, **kwargs: Any) -> Path:
        calls.append({"destination": destination, **kwargs})
        return destination

    monkeypatch.setattr(cli_module, "download_ames", fake_download)
    result = invoke("data", "download", "--out", str(tmp_path))
    assert result.exit_code == 0
    assert calls[0]["destination"] == tmp_path / "train.csv"
    assert calls[0]["sha256"] == cli_module.DATASET_SHA256

    custom = invoke("data", "download", "--out", str(tmp_path / "x.csv"), "--url", "https://e.x/y")
    assert custom.exit_code == 0
    assert calls[1]["destination"] == tmp_path / "x.csv"
    assert calls[1]["sha256"] is None
    assert "will not be verified" in custom.stderr


def test_serve_passes_models_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch, trained: tuple[Path, str]
) -> None:
    import uvicorn

    captured: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs))
    model_dir, _ = trained
    result = invoke("serve", "--model", str(model_dir), "--port", "9001")
    assert result.exit_code == 0, result.output
    assert captured["port"] == 9001 and captured["host"] == "127.0.0.1"
    assert captured["app"].title == "amesml"


def test_serve_without_the_api_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    result = invoke("serve")
    assert result.exit_code == 1
    assert 'pip install "amesml[api]"' in result.stderr


def test_verbose_logging(trained: tuple[Path, str], ames_csv: Path) -> None:
    model_dir, _ = trained
    result = invoke("-v", "predict", "--model", str(model_dir), "--input", str(ames_csv))
    assert result.exit_code == 0


def test_python_dash_m(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["amesml", "--version"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("amesml", run_name="__main__")
    assert raised.value.code == 0
    assert __version__ in capsys.readouterr().out
