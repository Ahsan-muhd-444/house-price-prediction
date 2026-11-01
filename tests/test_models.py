from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import sklearn
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from amesml import (
    AmesSchemaError,
    ClassificationMetrics,
    ConfigError,
    RegressionMetrics,
    TrainConfig,
    predict,
    train,
    train_classifier,
    train_regressor,
)
from amesml import models as models_module
from amesml.data import train_test_split_df
from amesml.features import ENGINEERED_FEATURES, SOURCE_COLUMNS
from amesml.models import (
    build_pipeline,
    l1_logistic_regression,
    make_estimator,
    plan_features,
    prepare_input,
    records_to_frame,
    resolve_threshold,
    split_for_training,
)

MODELS = ("linear", "ridge", "lasso", "gbm", "xgboost", "lightgbm")


@pytest.mark.parametrize(
    ("task", "model", "expected"),
    [
        ("regression", "linear", LinearRegression),
        ("regression", "ridge", Ridge),
        ("regression", "lasso", Lasso),
        ("regression", "gbm", HistGradientBoostingRegressor),
        ("regression", "xgboost", XGBRegressor),
        ("regression", "lightgbm", LGBMRegressor),
        ("classification", "linear", LogisticRegression),
        ("classification", "ridge", LogisticRegression),
        ("classification", "lasso", LogisticRegression),
        ("classification", "gbm", HistGradientBoostingClassifier),
        ("classification", "xgboost", XGBClassifier),
        ("classification", "lightgbm", LGBMClassifier),
    ],
)
def test_make_estimator_families(task: str, model: str, expected: type) -> None:
    estimator = make_estimator(task, model, alpha=4.0, random_state=1, max_iter=100)
    assert isinstance(estimator, expected)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("model", ["xgboost", "lightgbm"])
def test_boosting_models_are_seeded(task: str, model: str) -> None:
    estimator = make_estimator(task, model, alpha=None, random_state=7, max_iter=10)
    assert estimator.get_params()["random_state"] == 7


@pytest.mark.parametrize("model", ["xgboost", "lightgbm"])
def test_missing_boosting_library_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setitem(sys.modules, model, None)  # makes `import <model>` fail
    with pytest.raises(ConfigError, match=r'pip install "amesml\[boosting\]"'):
        make_estimator("regression", model, alpha=None, random_state=0, max_iter=10)


@pytest.mark.parametrize("model", ["gbm", "xgboost", "lightgbm"])
def test_tree_models_get_dense_input(ames_frame: pd.DataFrame, model: str) -> None:
    config = TrainConfig(model=model)
    plan = plan_features(ames_frame, config)
    preprocess = build_pipeline(plan, config).named_steps["preprocess"]
    assert not sparse.issparse(preprocess.fit_transform(ames_frame[list(plan.input_columns)]))


def test_classification_alpha_is_inverse_c() -> None:
    estimator = make_estimator("classification", "ridge", alpha=4.0, random_state=0, max_iter=10)
    assert estimator.C == 0.25


def test_make_estimator_errors() -> None:
    with pytest.raises(ConfigError, match="alpha"):
        make_estimator("regression", "ridge", alpha=None, random_state=0, max_iter=10)
    with pytest.raises(ConfigError, match="Unsupported"):
        make_estimator("ranking", "linear", alpha=None, random_state=0, max_iter=10)


@pytest.mark.parametrize("version", [(1, 7), (1, 8), (1, 10)])
def test_l1_logistic_regression_across_sklearn_versions(
    monkeypatch: pytest.MonkeyPatch, version: tuple[int, int]
) -> None:
    monkeypatch.setattr(models_module, "SKLEARN_VERSION", version)
    params = l1_logistic_regression(0.5, random_state=0, max_iter=10).get_params()
    assert params["C"] == 0.5 and params["solver"] == "liblinear"
    if version >= (1, 8):
        assert params["l1_ratio"] == 1.0
    else:
        assert params["penalty"] == "l1"


@pytest.mark.parametrize(
    ("raw", "expected"), [("1.9.1", (1, 9)), ("1.10.0rc1", (1, 10)), ("nightly", (0, 0))]
)
def test_sklearn_version_parsing(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: tuple[int, int]
) -> None:
    monkeypatch.setattr(sklearn, "__version__", raw)
    assert models_module._sklearn_version() == expected


def test_default_feature_plan(ames_frame: pd.DataFrame) -> None:
    plan = plan_features(ames_frame, TrainConfig())
    assert len(plan.input_columns) == 79
    assert "Id" not in plan.input_columns and "SalePrice" not in plan.input_columns
    assert plan.codes == ("MSSubClass", "MoSold")
    assert plan.zero_fill == ("GarageYrBlt", "MasVnrArea")


def test_feature_overrides(ames_frame: pd.DataFrame) -> None:
    config = TrainConfig(
        numeric_features=("GrLivArea", "OverallQual"), categorical_features=("Neighborhood",)
    )
    plan = plan_features(ames_frame, config)
    assert plan.input_columns == ("OverallQual", "GrLivArea", "Neighborhood")
    assert plan.zero_fill == ()


def test_overriding_one_role_keeps_the_other_inferred(ames_frame: pd.DataFrame) -> None:
    plan = plan_features(ames_frame, TrainConfig(categorical_features=("OverallQual",)))
    assert plan.categorical == ("OverallQual",)
    assert "OverallQual" not in plan.numeric
    assert plan.codes == ("OverallQual",)


def test_engineered_features_extend_the_plan(ames_frame: pd.DataFrame) -> None:
    config = TrainConfig(numeric_features=("GrLivArea",), engineer_features=True)
    plan = plan_features(ames_frame, config)
    assert plan.numeric == ("GrLivArea", *ENGINEERED_FEATURES)
    assert set(SOURCE_COLUMNS) <= set(plan.input_columns)


@pytest.mark.parametrize(
    ("frame_change", "config", "message"),
    [
        (lambda f: f, TrainConfig(numeric_features=("Nope",)), "not in the data"),
        (lambda f: f, TrainConfig(numeric_features=("Street",)), "non-numeric"),
        (
            lambda f: f.drop(columns=["YrSold"]),
            TrainConfig(numeric_features=("GrLivArea",), engineer_features=True),
            "engineer_features needs",
        ),
        (lambda f: f[["Id", "SalePrice"]], TrainConfig(), "No feature columns"),
    ],
)
def test_plan_errors(
    ames_frame: pd.DataFrame, frame_change: object, config: TrainConfig, message: str
) -> None:
    with pytest.raises(AmesSchemaError, match=message):
        plan_features(frame_change(ames_frame), config)


@pytest.mark.parametrize("model", MODELS)
def test_every_regressor_trains_and_predicts(ames_frame: pd.DataFrame, model: str) -> None:
    train_df, test_df = train_test_split_df(ames_frame)
    pipeline = train_regressor(train_df, TrainConfig(model=model))
    prices = predict(pipeline, test_df).values
    assert prices.shape == (len(test_df),)
    assert np.isfinite(prices).all()
    assert pipeline.task_ == "regression"


@pytest.mark.parametrize("model", MODELS)
def test_every_classifier_trains_and_predicts(ames_frame: pd.DataFrame, model: str) -> None:
    train_df, test_df = train_test_split_df(ames_frame)
    pipeline = train_classifier(train_df, TrainConfig(model=model))
    result = predict(pipeline, test_df)
    assert set(np.unique(result.values)) <= {0, 1}
    assert result.probabilities is not None
    assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()
    assert pipeline.threshold_ == train_df["SalePrice"].median()


def test_models_learn_the_signal(ames_frame: pd.DataFrame) -> None:
    regression = train(ames_frame, TrainConfig(model="ridge")).metrics
    classification = train(ames_frame, TrainConfig(task="classification")).metrics
    assert isinstance(regression, RegressionMetrics) and regression.r2 > 0.8
    # ROC-AUC rather than accuracy: on 40 held-out rows accuracy moves in steps of 2.5%.
    assert isinstance(classification, ClassificationMetrics)
    assert classification.roc_auc is not None and classification.roc_auc > 0.85


def test_engineered_features_are_recomputed_inside_the_model(ames_frame: pd.DataFrame) -> None:
    pipeline = train_regressor(ames_frame, TrainConfig(model="gbm", engineer_features=True))
    assert "features" in pipeline.named_steps
    assert not set(ENGINEERED_FEATURES) & set(pipeline.input_features_)
    assert np.isfinite(predict(pipeline, ames_frame.head()).values).all()


def test_fitted_attributes(ames_frame: pd.DataFrame) -> None:
    pipeline = train_regressor(ames_frame)
    assert pipeline.input_features_ == list(plan_features(ames_frame, TrainConfig()).input_columns)
    assert pipeline.input_kinds_["GrLivArea"] == "number"
    assert pipeline.input_kinds_["Neighborhood"] == "text"
    assert len(pipeline.training_data_sha256_) == 64
    assert pipeline.amesml_config_["model"] == "linear"


def test_defaults_count_missing_as_a_value(ames_frame: pd.DataFrame) -> None:
    defaults = train_regressor(ames_frame).feature_defaults_
    # Most houses have no pool, so the typical PoolQC is "missing", not the
    # commonest quality grade among the few houses that do have one.
    assert defaults["PoolQC"] is None
    assert defaults["GrLivArea"] == float(ames_frame["GrLivArea"].median())
    assert defaults["Neighborhood"] == ames_frame["Neighborhood"].mode().iloc[0]
    assert isinstance(defaults["MSSubClass"], int)  # a category code: the mode, not the median


def test_task_in_config_is_overridden_by_the_function(ames_frame: pd.DataFrame) -> None:
    config = TrainConfig(task="classification", stratify=True)
    assert train_regressor(ames_frame, config).task_ == "regression"
    assert train_classifier(ames_frame, TrainConfig()).task_ == "classification"


def test_thresholds(ames_frame: pd.DataFrame) -> None:
    prices = ames_frame["SalePrice"]
    assert resolve_threshold(prices, "median") == prices.median()
    assert resolve_threshold(prices, 200_000) == 200_000.0
    pipeline = train_classifier(ames_frame, TrainConfig(threshold=250_000))
    assert pipeline.threshold_ == 250_000.0
    with pytest.raises(ConfigError, match="one class"):
        train_classifier(ames_frame, TrainConfig(threshold=10_000_000))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.drop(columns=["SalePrice"]), "not found"),
        (lambda f: f.assign(SalePrice="cheap"), "must be numeric"),
        (lambda f: f.assign(SalePrice=np.nan), "missing values"),
    ],
)
def test_target_errors(ames_frame: pd.DataFrame, change: object, message: str) -> None:
    with pytest.raises(AmesSchemaError, match=message):
        train_regressor(change(ames_frame))


def test_train_reports_held_out_metrics(ames_frame: pd.DataFrame) -> None:
    result = train(ames_frame)
    assert (result.train_rows, result.test_rows) == (160, 40)
    assert result.metrics.n_samples == 40
    assert len(result.dataset_sha256) == 64


def test_stratified_split_preserves_the_class_balance(ames_frame: pd.DataFrame) -> None:
    config = TrainConfig(task="classification", stratify=True)
    _, test = split_for_training(ames_frame, config)
    full_share = (ames_frame["SalePrice"] >= ames_frame["SalePrice"].median()).mean()
    test_share = (test["SalePrice"] >= ames_frame["SalePrice"].median()).mean()
    assert abs(full_share - test_share) <= 1 / len(test)
    assert train(ames_frame, config).test_rows == len(test)


def test_records_to_frame_fills_absent_fields_but_keeps_explicit_nulls(
    ames_frame: pd.DataFrame,
) -> None:
    pipeline = train_regressor(ames_frame)
    frame = records_to_frame(pipeline, [{"GrLivArea": 1500}, {"GrLivArea": 900, "PoolQC": "Ex"}])
    assert frame["GrLivArea"].tolist() == [1500, 900]
    assert frame.loc[0, "Neighborhood"] == pipeline.feature_defaults_["Neighborhood"]
    assert frame.loc[1, "PoolQC"] == "Ex"
    explicit = records_to_frame(pipeline, [{"LotFrontage": None}])
    assert pd.isna(explicit.loc[0, "LotFrontage"])


def test_records_to_frame_rejects_unknown_fields(ames_frame: pd.DataFrame) -> None:
    pipeline = train_regressor(ames_frame)
    with pytest.raises(AmesSchemaError, match="Unknown feature.*GrLivAre"):
        records_to_frame(pipeline, [{"GrLivAre": 1500}])
    relaxed = records_to_frame(pipeline, [{"GrLivAre": 1500}], strict=False)
    assert "GrLivAre" not in relaxed.columns


def test_prepare_input_aligns_and_coerces(ames_frame: pd.DataFrame) -> None:
    pipeline = train_regressor(ames_frame)
    raw = pd.DataFrame(
        {"GrLivArea": ["1500", " "], "Neighborhood": [7, None], "Extra": [1, 2]}, dtype=object
    )
    prepared = prepare_input(pipeline, raw)
    assert list(prepared.columns) == pipeline.input_features_
    assert prepared["GrLivArea"].iloc[0] == 1500.0
    assert pd.isna(prepared["GrLivArea"].iloc[1])  # blank means missing
    assert prepared["Neighborhood"].tolist()[0] == "7"
    assert prepared["OverallQual"].iloc[0] == pipeline.feature_defaults_["OverallQual"]


def test_prepare_input_rejects_text_in_numeric_columns(ames_frame: pd.DataFrame) -> None:
    pipeline = train_regressor(ames_frame)
    with pytest.raises(AmesSchemaError, match="'LotArea' must be numeric; got 'large'"):
        prepare_input(pipeline, pd.DataFrame({"LotArea": ["large"]}))


def test_unseen_categories_are_reported_by_name(ames_frame: pd.DataFrame) -> None:
    pipeline = train_regressor(ames_frame)
    houses = ames_frame.head(2).assign(Neighborhood=["Atlantis", "NAmes"], MSSubClass=[999, 20])
    result = predict(pipeline, houses)  # no scikit-learn warning may escape
    assert result.unseen_categories == {"Neighborhood": ["Atlantis"], "MSSubClass": ["999"]}


def test_pipeline_predict_accepts_a_full_frame(ames_frame: pd.DataFrame) -> None:
    # The quick start calls pipeline.predict directly on rows that still carry
    # Id and SalePrice; the column transformer ignores the extra columns.
    pipeline = train_regressor(ames_frame)
    direct = pipeline.predict(ames_frame.head())
    np.testing.assert_allclose(direct, predict(pipeline, ames_frame.head()).values)


def test_predictions_to_frame(ames_frame: pd.DataFrame) -> None:
    regression = predict(train_regressor(ames_frame), ames_frame.head(3))
    assert list(regression.to_frame().columns) == ["predicted_price"]
    classification = predict(train_classifier(ames_frame), ames_frame.head(3))
    assert list(classification.to_frame().columns) == [
        "predicted_label",
        "probability_above_threshold",
    ]
