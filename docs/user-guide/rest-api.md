# REST API

```bash
pip install "amesml[api]"
amesml serve --model models/regression-gbm --model models/classification-gbm
```

The service loads its models once at startup and refuses to start if a model is
missing or invalid. Interactive documentation is served at `/docs`, and the
OpenAPI schema at `/openapi.json`.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | status, version and which tasks have a model |
| `GET` | `/model/info` | the `metadata.json` of each loaded model, by task |
| `POST` | `/predict` | predicted sale prices |
| `POST` | `/predict/class` | price class and probability for each house |

## Predicting

Send up to 1,000 houses as `records`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

```json
{
  "predictions": [180571.47, 145403.74],
  "model": {
    "task": "regression",
    "model": "gbm",
    "estimator": "HistGradientBoostingRegressor",
    "amesml_version": "0.1.0",
    "created_at": "2026-11-01T11:10:18+00:00"
  },
  "warnings": []
}
```

`/predict/class` takes the same body and returns:

```json
{
  "predictions": [
    {"label": 1, "above_threshold": true, "probability": 0.9431},
    {"label": 0, "above_threshold": false, "probability": 0.0003}
  ],
  "threshold": 165000.0,
  "model": {"task": "classification", "model": "gbm"},
  "warnings": []
}
```

## Partial records

Records use the Ames column names. You do not have to send all 79:

- **Leave a field out** and it takes the model's typical training value: the
  median for measurements, the most common value for everything else.
- **Send `null`** to say the feature is absent - for example `"PoolQC": null`
  for a house without a pool. This is how the dataset encodes absence, and the
  model handles it exactly as it did in training.

## Errors

| Status | When |
|---|---|
| `422` | an unknown field name (so typos fail loudly), text in a numeric field, an empty or oversized batch |
| `503` | no model is loaded for the requested task |

Categories the model never saw in training (an unknown `Neighborhood`, say) do
not fail the request. They are listed in `warnings` and encoded as the
column's reference category.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AMESML_MODEL_PATH` | - | comma-separated model paths, one per task |
| `AMESML_HOST` | `127.0.0.1` | interface to bind |
| `AMESML_PORT` | `8000` | port |

To run under your own ASGI server, use `amesml.api:app` (reads
`AMESML_MODEL_PATH`) or call `amesml.api.create_app([paths])`.
