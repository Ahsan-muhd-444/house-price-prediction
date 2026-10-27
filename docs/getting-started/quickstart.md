# 5-minute quickstart

## 1. Install and get the data

```bash
pip install "amesml[api]"
amesml data download
```

## 2. Train a model

```bash
amesml train --model gbm
```

This trains on 80% of the 1,460 houses, scores the held-out 20% and prints the
results:

```text
     Held-out test set
┏━━━━━━━━┳━━━━━━━━━┓
┃ Metric ┃   Value ┃
┡━━━━━━━━╇━━━━━━━━━┩
│ MAE    │ $16,826 │
│ RMSE   │ $29,462 │
│ R²     │  0.8868 │
│ MAPE   │  10.02% │
│ Rows   │     292 │
└────────┴─────────┘
Saved model to models/regression-gbm
```

## 3. Predict

```bash
amesml predict --model models/regression-gbm --input data/train.csv --out predictions.csv
```

## 4. Serve it

```bash
amesml serve --model models/regression-gbm
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"OverallQual": 7, "GrLivArea": 1710, "Neighborhood": "CollgCr"}]}'
```

Interactive documentation for every endpoint is at
<http://127.0.0.1:8000/docs>.

## The same in Python

```python
from amesml import TrainConfig, evaluate_regressor, load_ames, predict, train_regressor
from amesml.data import train_test_split_df

train_df, test_df = train_test_split_df(load_ames())
model = train_regressor(train_df, TrainConfig(model="gbm"))
print(evaluate_regressor(test_df["SalePrice"], predict(model, test_df).values))
```

Next: the [user guide](../user-guide/cli.md).
