# Python API

Everything the CLI does is available from Python. The main entry points are
importable from the top-level package.

## Train and evaluate in one call

```python
from amesml import TrainConfig, load_ames, train

houses = load_ames()  # data/train.csv, downloaded if missing
result = train(houses, TrainConfig(model="gbm"))

result.metrics  # RegressionMetrics(mae=16826.0, ...)
result.metrics.to_dict()
result.save("models/regression-gbm")
```

`train` splits the rows with `test_size` and `random_state`, fits on the
training rows and scores the rest. The returned `TrainResult` also carries
`train_rows`, `test_rows`, `dataset_sha256` and any `unseen_categories` found
in the test rows.

## Control the split yourself

```python
from amesml import TrainConfig, evaluate_classifier, predict, train_classifier
from amesml.data import train_test_split_df

train_df, test_df = train_test_split_df(houses, test_size=0.2, random_state=42)
model = train_classifier(train_df, TrainConfig(model="lasso"))

predictions = predict(model, test_df)
labels = (test_df["SalePrice"] >= model.threshold_).astype(int)
evaluate_classifier(labels, predictions.values, predictions.probabilities)
```

`train_regressor` and `train_classifier` fit on every row they are given and
return a scikit-learn `Pipeline`. It works with scikit-learn tooling directly
(`model.predict(frame)`), and carries a few extra fitted attributes:

| Attribute | Meaning |
|---|---|
| `task_` | `"regression"` or `"classification"` |
| `input_features_` | raw columns the model reads |
| `feature_defaults_` | typical training value per column |
| `threshold_` | classification only: the price separating the classes |
| `training_data_sha256_` | fingerprint of the training rows |

## Predicting

`amesml.predict(model, frame)` is the safest way to score new data: it aligns
and type-checks the columns, fills missing ones, and reports categories the
model never saw:

```python
result = predict(model, new_houses)
result.values  # prices, or labels for a classifier
result.probabilities  # classifier only
result.unseen_categories  # e.g. {"Neighborhood": ["Atlantis"]}
result.to_frame()
```

## Save and load

```python
from amesml import load_metadata, load_model, save_model

save_model(model, "models/my-model")
model = load_model("models/my-model")  # verifies the file's checksum
metadata = load_metadata("models/my-model")
```

!!! warning
    Models are pickled. Loading one runs code, so only load models you trust.
    The checksum detects corrupted or swapped files; it is not a security
    boundary.

## Building blocks

```python
from amesml import add_engineered_features, build_preprocessor, infer_feature_types

types = infer_feature_types(houses)  # numeric vs categorical columns
transformer = build_preprocessor(types.numeric, types.categorical, scale=True)
houses_plus = add_engineered_features(houses)  # TotalSF, HouseAge, ...
```

See the [API reference](../reference/api.md) for every function.

## Errors

Every error amesml raises on purpose derives from `amesml.AmesMLError`:
`AmesSchemaError` (bad columns or values), `ConfigError`,
`DatasetDownloadError` and `ModelLoadError`. Each also derives from the
closest built-in (`ValueError` or `RuntimeError`).
