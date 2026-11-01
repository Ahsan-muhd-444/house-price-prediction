# Configuration file

Training options live in a `TrainConfig`, which can be written as YAML or JSON
and passed with `amesml train --config FILE` or `TrainConfig.from_file(FILE)`.
Unknown keys are rejected, so a typo cannot silently fall back to a default.

```yaml
--8<-- "examples/config.example.yaml"
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `task` | `regression` | `regression` predicts the price; `classification` predicts at/above `threshold` |
| `model` | `linear` | `linear`, `ridge`, `lasso`, `gbm`, `xgboost` or `lightgbm` (see below) |
| `target` | `SalePrice` | target column |
| `id_column` | `Id` | identifier, never used as a feature |
| `test_size` | `0.2` | share of rows held out by `train` |
| `random_state` | `42` | seed for the split and every seeded estimator |
| `threshold` | `median` | classification: a price, or the training rows' median |
| `stratify` | `false` | classification: keep the class balance in both splits |
| `engineer_features` | `false` | add total area, ages, bathrooms and porch area |
| `numeric_features` | inferred | use exactly these numeric columns |
| `categorical_features` | inferred | use exactly these categorical columns |
| `categorical_codes` | `[MSSubClass, MoSold]` | numeric columns that are category codes |
| `zero_fill_features` | `[GarageYrBlt, MasVnrArea]` | numeric columns where missing means "none" |
| `drop_first` | `true` | drop one level per one-hot encoded column |
| `alpha` | per model | regularisation strength for `ridge` and `lasso` |
| `max_iter` | `5000` | iteration limit for iterative solvers |

The defaults reproduce the original notebook closely - to the cent for the
classification, and to within about 5% for the regression figures, which
come from an iterative solver whose stopping point shifts with the
scikit-learn, scipy and BLAS build.

## Model families

| `model` | Regression | Classification | Default `alpha` |
|---|---|---|---|
| `linear` | ordinary least squares | logistic regression (`C = 1`) | - |
| `ridge` | ridge regression | L2-penalised logistic regression | 10 / 30 |
| `lasso` | lasso | L1-penalised logistic regression | 300 / 3 |
| `gbm` | histogram gradient boosting | histogram gradient boosting | - |
| `xgboost` | `XGBRegressor` | `XGBClassifier` | - |
| `lightgbm` | `LGBMRegressor` | `LGBMClassifier` | - |

For classification `C = 1 / alpha`, so in both tasks a larger `alpha` means a
simpler model. The defaults were chosen by 5-fold cross-validation on the
training split; see [Design decisions](../DECISIONS.md).

`xgboost` and `lightgbm` use those libraries with their default settings,
seeded from `random_state`. They are an optional extra, installed with
`pip install "amesml[boosting]"`; without it, training one of them fails
with an error naming the extra.

## Overriding feature types

Setting `numeric_features` or `categorical_features` replaces the inferred list
for that role only; the other role is still inferred from the remaining
columns. To train on a handful of columns, set both:

```yaml
numeric_features: [GrLivArea, OverallQual, YearBuilt, TotalBsmtSF, GarageCars]
categorical_features: [Neighborhood]
```

With `engineer_features: true` the engineered columns are added to the numeric
features, and the raw columns they need are read even if not listed.
