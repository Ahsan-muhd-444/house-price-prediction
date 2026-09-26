"""The ``amesml`` command-line interface.

Typer reads these function signatures at runtime to build the CLI. Python 3.9
cannot evaluate ``X | None`` at runtime, so this module spells optional types
as ``Optional[X]`` and does not use ``from __future__ import annotations``.
"""

import contextlib
import logging
import sys
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ._version import __version__
from .config import TrainConfig
from .data import (
    DATASET_SHA256,
    DATASET_URL,
    dataset_fingerprint,
    download_ames,
    load_ames,
    read_ames_csv,
)
from .errors import AmesMLError
from .evaluate import evaluate_classifier, evaluate_regressor
from .models import (
    load_metadata,
    load_model,
    predict,
    price_labels,
    split_for_training,
)
from .models import train as train_model

app = typer.Typer(
    name="amesml",
    help="Reproducible Ames Housing price prediction: train, evaluate, predict and serve.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
data_app = typer.Typer(help="Manage the Ames Housing dataset.", no_args_is_help=True)
app.add_typer(data_app, name="data")

console = Console()
# Errors are printed unwrapped so they stay on one line in logs and CI output.
error_console = Console(stderr=True, soft_wrap=True, highlight=False)
logger = logging.getLogger("amesml")


class Task(str, Enum):
    """Prediction task."""

    regression = "regression"
    classification = "classification"


class ModelFamily(str, Enum):
    """Estimator family."""

    linear = "linear"
    ridge = "ridge"
    lasso = "lasso"
    gbm = "gbm"


class Split(str, Enum):
    """Which rows of a dataset to evaluate on."""

    all = "all"
    train = "train"
    test = "test"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"amesml {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logging.")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Only show warnings and errors.")
    ] = False,
) -> None:
    """Reproducible Ames Housing price prediction."""
    _configure_logging(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)


@data_app.command("download")
def data_download(
    out: Annotated[
        Path, typer.Option("--out", help="Directory (or .csv path) to save the dataset to.")
    ] = Path("data"),
    url: Annotated[Optional[str], typer.Option("--url", help="Download from another URL.")] = None,
    sha256: Annotated[
        Optional[str], typer.Option("--sha256", help="Expected SHA-256 of a custom --url.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Download the Ames training data and verify its checksum."""
    destination = out if out.suffix.lower() == ".csv" else out / "train.csv"
    if url is not None and sha256 is None:
        logger.warning("No --sha256 given for a custom --url; the download will not be verified")
    with _errors_as_exit():
        path = download_ames(
            destination,
            url=url or DATASET_URL,
            sha256=sha256 if url is not None else DATASET_SHA256,
            force=force,
        )
    console.print(f"Dataset ready at [bold]{escape(str(path))}[/bold]")


@app.command()
def train(
    data: Annotated[
        Optional[Path],
        typer.Option(
            "--data", help="Training CSV. Default: data/train.csv, downloaded if missing."
        ),
    ] = None,
    config: Annotated[
        Optional[Path], typer.Option("--config", help="YAML or JSON file of TrainConfig options.")
    ] = None,
    task: Annotated[Optional[Task], typer.Option("--task", help="Overrides the config.")] = None,
    model: Annotated[
        Optional[ModelFamily], typer.Option("--model", help="Overrides the config.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Directory to save models in.")] = Path(
        "models"
    ),
    name: Annotated[
        Optional[str], typer.Option("--name", help="Model directory name. Default: <task>-<model>.")
    ] = None,
) -> None:
    """Train a model, report held-out metrics and save it."""
    with _errors_as_exit():
        settings = TrainConfig.from_file(config) if config is not None else TrainConfig()
        overrides: dict[str, Any] = {}
        if task is not None:
            overrides["task"] = task.value
            if task is Task.regression:
                overrides["stratify"] = False
        if model is not None:
            overrides["model"] = model.value
        settings = settings.replace(**overrides)
        frame = load_ames(data)
        result = train_model(frame, settings)
        directory = result.save(out / (name or f"{settings.task}-{settings.model}"))
    console.print(_metrics_table(result.metrics.to_dict(), settings.task, "Held-out test set"))
    console.print(f"Saved model to [bold]{escape(str(directory))}[/bold]")


@app.command()
def evaluate(
    model: Annotated[Path, typer.Option("--model", help="Model directory or .joblib file.")],
    data: Annotated[Path, typer.Option("--data", help="CSV with a SalePrice column.")],
    split: Annotated[
        Split,
        typer.Option("--split", help="Score all rows, or recreate the model's train/test split."),
    ] = Split.all,
) -> None:
    """Score a saved model on labelled data."""
    with _errors_as_exit():
        pipeline = load_model(model)
        metadata = load_metadata(model)
        frame = read_ames_csv(data)
        if "config" not in metadata:
            raise AmesMLError(f"{model} has no training config in its metadata")
        settings = TrainConfig.from_dict(metadata["config"])
        if settings.target not in frame.columns:
            raise AmesMLError(f"{data} has no {settings.target} column to evaluate against")
        if split is not Split.all:
            train_rows, test_rows = split_for_training(frame, settings)
            frame = test_rows if split is Split.test else train_rows
        elif dataset_fingerprint(frame) == metadata.get("dataset", {}).get("sha256"):
            logger.warning(
                "This is the model's training dataset, so these scores include rows it was "
                "fitted on. Use --split test for an honest estimate."
            )
        predictions = predict(pipeline, frame)
        truth = frame[settings.target]
        if pipeline.task_ == "classification":
            metrics = evaluate_classifier(
                price_labels(truth, pipeline.threshold_),
                predictions.values,
                predictions.probabilities,
            ).to_dict()
        else:
            metrics = evaluate_regressor(truth, predictions.values).to_dict()
    console.print(
        _metrics_table(metrics, pipeline.task_, f"{escape(data.name)} ({split.value} rows)")
    )


@app.command("predict")
def predict_command(
    model: Annotated[Path, typer.Option("--model", help="Model directory or .joblib file.")],
    input_csv: Annotated[Path, typer.Option("--input", help="CSV of houses in the Ames format.")],
    out: Annotated[
        Optional[Path], typer.Option("--out", help="Write predictions here instead of stdout.")
    ] = None,
) -> None:
    """Predict prices (or price classes) for every house in a CSV."""
    with _errors_as_exit():
        pipeline = load_model(model)
        frame = read_ames_csv(input_csv)
        results = predict(pipeline, frame).to_frame(index=frame.index)
        if "Id" in frame.columns:
            results.insert(0, "Id", frame["Id"])
    if out is None:
        sys.stdout.write(results.to_csv(index=False, lineterminator="\n"))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False, lineterminator="\n")
    console.print(f"Wrote {len(results)} prediction(s) to [bold]{escape(str(out))}[/bold]")


@app.command()
def plots(
    data: Annotated[
        Optional[Path],
        typer.Option(
            "--data", help="Training CSV. Default: data/train.csv, downloaded if missing."
        ),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Directory for the PNG files.")] = Path(
        "images"
    ),
) -> None:
    """Regenerate the README figures from the data."""
    with _errors_as_exit():
        try:
            from .plots import generate_all
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise AmesMLError(str(exc)) from exc
        written = generate_all(load_ames(data), out)
    for name, path in written.items():
        console.print(f"{name:>20}  {path}", markup=False)


@app.command()
def serve(
    model: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--model",
            help="Model to serve; repeat to serve a regressor and a classifier together. "
            "Default: the comma-separated paths in $AMESML_MODEL_PATH.",
        ),
    ] = None,
    host: Annotated[str, typer.Option("--host", envvar="AMESML_HOST")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", envvar="AMESML_PORT")] = 8000,
) -> None:
    """Serve predictions over HTTP (needs the api extra)."""
    try:
        import uvicorn

        from .api import create_app
    except ImportError as exc:
        message = f'The API needs extra packages: pip install "amesml[api]" ({exc})'
        error_console.print(message, style="red", markup=False)
        raise typer.Exit(code=1) from exc
    uvicorn.run(create_app(model or None), host=host, port=port)


def _metrics_table(metrics: dict[str, Any], task: str, title: str) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    if task == "classification":
        for key, label in (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 score"),
        ):
            table.add_row(label, f"{metrics[key]:.2%}")
        auc = metrics.get("roc_auc")
        table.add_row("ROC-AUC", "n/a" if auc is None else f"{auc:.4f}")
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        table.add_row("Confusion matrix", f"TN={tn} FP={fp} FN={fn} TP={tp}")
    else:
        table.add_row("MAE", f"${metrics['mae']:,.0f}")
        table.add_row("RMSE", f"${metrics['rmse']:,.0f}")
        table.add_row("R²", f"{metrics['r2']:.4f}")
        table.add_row("MAPE", f"{metrics['mape']:.2%}")
    table.add_row("Rows", f"{metrics['n_samples']:,}")
    return table


@contextlib.contextmanager
def _errors_as_exit() -> Iterator[None]:
    """Report expected failures as one red line instead of a traceback."""
    try:
        yield
    except (AmesMLError, FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.debug("Command failed", exc_info=True)
        error_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


class _CliLogHandler(logging.Handler):
    """One plain line per record on the current ``sys.stderr``.

    Looking ``sys.stderr`` up at emit time (rather than binding it once) keeps
    the output following redirections, such as a test runner's capture.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            prefix = "" if record.levelno <= logging.INFO else f"{record.levelname.lower()}: "
            sys.stderr.write(prefix + message + "\n")
            sys.stderr.flush()  # as logging.StreamHandler does; buffered streams lose it otherwise
        except Exception:  # logging must never crash the program
            self.handleError(record)


def _configure_logging(level: int) -> None:
    # Replace rather than stack handlers, so repeated in-process invocations
    # (as in the test suite) do not print every message several times.
    for handler in list(logger.handlers):
        if isinstance(handler, _CliLogHandler):
            logger.removeHandler(handler)
    logger.addHandler(_CliLogHandler())
    logger.setLevel(level)


if __name__ == "__main__":  # pragma: no cover
    app()
