# Original notebook models

The two pipelines the notebook saved with `joblib.dump`:

- `linear_regression_pipeline.joblib` - the sale-price regression
- `classification_pipeline.joblib` - the above/below-median classifier

They are kept for reference. Pickled scikit-learn objects only load reliably
with the scikit-learn version that created them, and these carry no metadata
recording which one that was. To get equivalent, documented models, train them
with the package instead:

```bash
amesml train --task regression --model linear
amesml train --task classification --model linear
```

Each trained model directory holds `model.joblib`, a `metadata.json` describing
how it was trained (versions, configuration, metrics, data fingerprint and a
checksum of the model file) and `metrics.json`.
