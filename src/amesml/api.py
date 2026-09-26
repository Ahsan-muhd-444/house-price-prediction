"""REST inference service.

Install with ``pip install "amesml[api]"`` and start it with ``amesml serve``,
or run ``uvicorn amesml.api:app`` with ``AMESML_MODEL_PATH`` set.

Models are loaded once, when the application starts. ``AMESML_MODEL_PATH`` (or
``create_app(model_paths=...)``) may list one regression and one classification
model, separated by commas, so a single process can serve both endpoints.

Pydantic and FastAPI evaluate annotations at runtime, and Python 3.9 cannot
evaluate ``X | None``; this module therefore spells unions with ``Optional``
and ``Union`` and does not use ``from __future__ import annotations``.
"""

import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from ._version import __version__
from .errors import AmesSchemaError, ModelLoadError
from .models import Predictions, load_metadata, load_model, predict, records_to_frame

logger = logging.getLogger(__name__)

#: Environment variable holding the comma-separated model paths to serve.
MODEL_PATH_ENV = "AMESML_MODEL_PATH"
#: Largest batch accepted by the prediction endpoints.
MAX_RECORDS = 1000

FeatureValue = Union[int, float, str, None]

EXAMPLE_HOUSE: dict[str, Any] = {
    "OverallQual": 7,
    "GrLivArea": 1710,
    "YearBuilt": 2003,
    "TotalBsmtSF": 856,
    "GarageCars": 2,
    "FullBath": 2,
    "Neighborhood": "CollgCr",
    "PoolQC": None,
}


class PredictRequest(BaseModel):
    """A batch of houses to score.

    Each record maps Ames column names to values. A field you leave out takes
    the model's typical training value; a field sent as ``null`` means the
    feature is absent - ``"PoolQC": null`` says the house has no pool.
    """

    model_config = ConfigDict(json_schema_extra={"examples": [{"records": [EXAMPLE_HOUSE]}]})

    records: list[dict[str, FeatureValue]] = Field(
        min_length=1,
        max_length=MAX_RECORDS,
        description="One object per house, keyed by Ames column name.",
    )


class ModelSummary(BaseModel):
    """Which model produced a response."""

    task: str
    model: Optional[str] = None
    estimator: Optional[str] = None
    amesml_version: Optional[str] = None
    created_at: Optional[str] = None


class PricePredictionResponse(BaseModel):
    """Predicted sale prices, in request order."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predictions": [208500.0],
                    "model": {"task": "regression", "model": "linear"},
                    "warnings": [],
                }
            ]
        }
    )

    predictions: list[float] = Field(description="Predicted sale price in dollars.")
    model: ModelSummary
    warnings: list[str] = Field(default_factory=list)


class ClassPrediction(BaseModel):
    """One house's price class."""

    label: int = Field(description="1 if predicted at or above the threshold, else 0.")
    above_threshold: bool
    probability: float = Field(description="Probability that the price is at or above it.")


class ClassPredictionResponse(BaseModel):
    """Predicted price classes, in request order."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predictions": [{"label": 1, "above_threshold": True, "probability": 0.93}],
                    "threshold": 165000.0,
                    "model": {"task": "classification", "model": "linear"},
                    "warnings": [],
                }
            ]
        }
    )

    predictions: list[ClassPrediction]
    threshold: float = Field(description="Sale price separating the two classes, in dollars.")
    model: ModelSummary
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service status."""

    status: Literal["ok"]
    version: str
    tasks: list[str] = Field(description="Tasks with a loaded model.")


@dataclass(frozen=True)
class LoadedModel:
    """A model ready to serve."""

    pipeline: Pipeline
    metadata: dict[str, Any]
    path: Path

    @property
    def summary(self) -> ModelSummary:
        return ModelSummary(
            task=self.pipeline.task_,
            model=self.metadata.get("model"),
            estimator=self.metadata.get("estimator"),
            amesml_version=self.metadata.get("amesml_version"),
            created_at=self.metadata.get("created_at"),
        )


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness check; also lists which tasks can be served."""
    return HealthResponse(status="ok", version=__version__, tasks=sorted(_models(request)))


@router.get("/model/info")
def model_info(request: Request) -> dict[str, dict[str, Any]]:
    """The ``metadata.json`` of every loaded model, keyed by task."""
    return {task: loaded.metadata for task, loaded in _models(request).items()}


@router.post("/predict", response_model=PricePredictionResponse)
def predict_price(body: PredictRequest, request: Request) -> PricePredictionResponse:
    """Predict the sale price of each house."""
    loaded = _model_for(request, "regression")
    predictions = _predict(loaded, body)
    return PricePredictionResponse(
        predictions=[round(float(value), 2) for value in predictions.values],
        model=loaded.summary,
        warnings=_warnings(predictions),
    )


@router.post("/predict/class", response_model=ClassPredictionResponse)
def predict_class(body: PredictRequest, request: Request) -> ClassPredictionResponse:
    """Predict whether each house sells at or above the price threshold."""
    loaded = _model_for(request, "classification")
    predictions = _predict(loaded, body)
    probabilities = predictions.probabilities
    assert probabilities is not None  # classifiers always produce probabilities
    return ClassPredictionResponse(
        predictions=[
            ClassPrediction(
                label=int(label), above_threshold=bool(label), probability=round(float(prob), 4)
            )
            for label, prob in zip(predictions.values, probabilities)
        ],
        threshold=float(loaded.pipeline.threshold_),
        model=loaded.summary,
        warnings=_warnings(predictions),
    )


def create_app(model_paths: Optional[Sequence[Union[str, Path]]] = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        model_paths: Models to serve - at most one per task. When omitted, the
            comma-separated paths in ``AMESML_MODEL_PATH`` are used.

    Returns:
        The application. Models are loaded when it starts, so a missing or
        invalid model stops startup with a clear error instead of failing on the
        first request.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        paths = [Path(p) for p in model_paths] if model_paths else _paths_from_environment()
        app.state.models = _load_models(paths)
        logger.info("Serving %s", ", ".join(f"{t}: {m.path}" for t, m in app.state.models.items()))
        yield

    app = FastAPI(
        title="amesml",
        version=__version__,
        summary="Ames Housing sale-price prediction.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


def _paths_from_environment() -> list[Path]:
    raw = os.environ.get(MODEL_PATH_ENV, "")
    paths = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    if not paths:
        raise RuntimeError(f"No model to serve: set {MODEL_PATH_ENV} or pass --model")
    return paths


def _load_models(paths: Sequence[Path]) -> dict[str, LoadedModel]:
    models: dict[str, LoadedModel] = {}
    for path in paths:
        pipeline = load_model(path)
        try:
            metadata = load_metadata(path)
        except ModelLoadError:
            metadata = {}
        task = str(pipeline.task_)
        if task in models:
            raise RuntimeError(
                f"Two {task} models given ({models[task].path} and {path}); serve one per task"
            )
        models[task] = LoadedModel(pipeline=pipeline, metadata=metadata, path=path)
    return models


def _models(request: Request) -> dict[str, LoadedModel]:
    models: dict[str, LoadedModel] = request.app.state.models
    return models


def _model_for(request: Request, task: str) -> LoadedModel:
    loaded = _models(request).get(task)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail=f"No {task} model is loaded. Start the service with one, for example: "
            f"amesml serve --model models/{task}-linear",
        )
    return loaded


def _predict(loaded: LoadedModel, body: PredictRequest) -> Predictions:
    try:
        frame = records_to_frame(loaded.pipeline, body.records, strict=True)
        return predict(loaded.pipeline, frame)
    except AmesSchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _warnings(predictions: Predictions) -> list[str]:
    return [
        f"{column}: value(s) {', '.join(values)} were not seen in training and are treated "
        "as the column's reference category"
        for column, values in predictions.unseen_categories.items()
    ]


#: Application for ``uvicorn amesml.api:app``; reads ``AMESML_MODEL_PATH`` at startup.
app = create_app()
