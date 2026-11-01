# amesml

**Reproducible Ames Housing price prediction as an installable package, a
command-line tool and a REST API.**

amesml turns a university house-price notebook into software you can install,
test, version and deploy. It predicts the sale price of a house in Ames, Iowa
from 79 property features, and separately whether it sells at or above the
median price.

```bash
pip install "amesml @ git+https://github.com/Ahsan-muhd-444/house-price-prediction@v0.1.1"
amesml train --model gbm
```

## Why amesml

- **Reproducible.** The default models reproduce the original notebook's
  published results closely - within about 5% - every model is seeded, and
  each saved model
  records the exact configuration, library versions and a fingerprint of the
  data it was trained on.
- **Installable.** A typed Python package with a CLI, tested on Linux, macOS
  and Windows across Python 3.9-3.13, with 99% test coverage.
- **Serveable.** A FastAPI service with validated requests and an
  OpenAPI page, packaged as a Docker image that starts with trained models.

## Results

On the notebook's held-out split (292 houses):

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear regression (the notebook) | $21,140 | $31,791 | 0.868 |
| Gradient boosting | $16,826 | $29,462 | 0.887 |

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic regression (the notebook) | 94.5% | 93.9% | 0.982 |
| Gradient boosting | 95.2% | 94.8% | 0.989 |

[All models, with cross-validation](benchmarks.md) - [Model card](model_card.md)

## Where next

- [Install](getting-started/installation.md) and work through the
  [5-minute quickstart](getting-started/quickstart.md).
- The [user guide](user-guide/cli.md) covers the CLI, the Python API, the
  configuration file, the REST API and Docker.
- [Architecture](architecture.md) and the [design decisions](DECISIONS.md)
  explain how it is built and why.
