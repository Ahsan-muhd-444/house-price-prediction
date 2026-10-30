# Docker

The image serves the REST API with a gradient-boosting regressor and
classifier.

```bash
docker compose up --build
# or
docker build -t amesml .
docker run --rm -p 8000:8000 amesml
```

Then open <http://localhost:8000/docs>.

## How the image is built

- **Two stages.** The builder installs `amesml[api]` into a virtual
  environment and prepares the models; the runtime stage copies only the
  environment and the two model directories onto a clean `python:3.11-slim`.
- **Models.** If `models/regression-gbm` and `models/classification-gbm` exist
  in the build context they are used as they are. Otherwise they are trained
  from `data/train.csv` during the build, so the image always starts with
  working models.
- **Small context.** `.dockerignore` is an allowlist: only `pyproject.toml`,
  `README.md`, `LICENSE`, `src/`, `data/` and `models/` are sent to the build.
- **Non-root.** The service runs as an unprivileged `amesml` user.
- **Health check.** Docker polls `GET /health`.

## Serving other models

Mount a model directory and point `AMESML_MODEL_PATH` at it:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models/regression-linear:/models/regression-linear:ro" \
  -e AMESML_MODEL_PATH=/models/regression-linear \
  amesml
```

!!! note
    A model trained outside the container may have been pickled with a
    different scikit-learn version. scikit-learn warns about this when loading;
    the safest course is to train inside the same image.
