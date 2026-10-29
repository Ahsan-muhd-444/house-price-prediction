# Command line

Every command has `--help`. Global options go before the command:
`amesml -v train ...` shows debug logging, `amesml -q train ...` only warnings.

| Command | Does |
|---|---|
| `amesml data download` | Download `data/train.csv` and verify its checksum |
| `amesml train` | Train a model, print held-out metrics, save it |
| `amesml evaluate` | Score a saved model on labelled data |
| `amesml predict` | Predict for every row of a CSV |
| `amesml plots` | Regenerate the figures from the data |
| `amesml serve` | Run the REST API |

Errors are reported on one line with exit code 1; usage errors exit with 2.

## train

```bash
amesml train                                   # the notebook's linear regression
amesml train --model gbm                       # gradient boosting
amesml train --task classification             # the notebook's logistic regression
amesml train --config examples/config.example.yaml --name my-model
```

The model is saved to `models/<task>-<model>/` (change with `--out` and
`--name`) as three files:

- `model.joblib` - the fitted scikit-learn pipeline
- `metadata.json` - configuration, metrics, feature defaults, library versions,
  a fingerprint of the training data and a checksum of `model.joblib`
- `metrics.json` - the held-out metrics alone

`--task` and `--model` override the configuration file.

## evaluate

```bash
amesml evaluate --model models/regression-gbm --data data/train.csv --split test
```

`--split test` recreates the exact rows the model was held out on, so it
reproduces the metrics printed at training time. `--split train` scores the
training rows, and `--split all` (the default) every row; amesml warns when
those include rows the model was fitted on.

## predict

```bash
amesml predict --model models/regression-gbm --input houses.csv > predictions.csv
amesml predict --model models/classification-gbm --input houses.csv --out classes.csv
```

The input uses the Ames column names; `NA` and empty cells mean missing.
Columns the model does not use (such as `SalePrice`) are ignored, columns it
needs but the file lacks are filled with typical training values, and an `Id`
column is copied to the output. Regression writes `predicted_price`;
classification writes `predicted_label` (1 = at or above the threshold) and
`probability_above_threshold`.

## plots

```bash
pip install "amesml[plots]"
amesml plots --out images/
```

Writes the sale-price distribution, missing values per column, residuals of the
linear model, the classifier's confusion matrix and the gradient-boosting model's
permutation importance, as `images/amesml_*.png`.

## serve

```bash
amesml serve --model models/regression-gbm --model models/classification-gbm
```

Serves at most one model per task. Without `--model`, the comma-separated
paths in `AMESML_MODEL_PATH` are used. `--host` and `--port` also read
`AMESML_HOST` and `AMESML_PORT`. See [REST API](rest-api.md).

## data download

```bash
amesml data download                      # to data/train.csv
amesml data download --out elsewhere/     # to elsewhere/train.csv
amesml data download --url URL --sha256 HASH
```

The default source is pinned to a commit of this repository and checked
against a SHA-256 before anything is written.
