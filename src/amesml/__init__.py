"""amesml: reproducible Ames Housing price prediction as a package, CLI and API.

Quick start::

    from amesml import TrainConfig, evaluate_regressor, load_ames, train_regressor
    from amesml.data import train_test_split_df

    train_df, test_df = train_test_split_df(load_ames())
    model = train_regressor(train_df, TrainConfig(model="gbm"))
    print(evaluate_regressor(test_df["SalePrice"], model.predict(test_df)))
"""

from __future__ import annotations

import logging

from ._version import __version__
from .config import TrainConfig
from .data import load_ames, train_test_split_df
from .errors import AmesMLError, AmesSchemaError, ConfigError, DatasetDownloadError, ModelLoadError
from .evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classifier,
    evaluate_regressor,
)
from .features import add_engineered_features
from .models import (
    TrainResult,
    load_metadata,
    load_model,
    predict,
    save_model,
    train,
    train_classifier,
    train_regressor,
)
from .preprocess import build_preprocessor, infer_feature_types

# Library code only logs; applications decide where the messages go.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AmesMLError",
    "AmesSchemaError",
    "ClassificationMetrics",
    "ConfigError",
    "DatasetDownloadError",
    "ModelLoadError",
    "RegressionMetrics",
    "TrainConfig",
    "TrainResult",
    "__version__",
    "add_engineered_features",
    "build_preprocessor",
    "evaluate_classifier",
    "evaluate_regressor",
    "infer_feature_types",
    "load_ames",
    "load_metadata",
    "load_model",
    "predict",
    "save_model",
    "train",
    "train_classifier",
    "train_regressor",
    "train_test_split_df",
]
