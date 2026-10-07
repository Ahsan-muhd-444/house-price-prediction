# syntax=docker/dockerfile:1

# Multi-stage build: the builder installs the package and prepares the models;
# the runtime image holds only the virtual environment and the trained models.
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /src
COPY . .
RUN pip install ".[api]"
# Serve models from the build context if they are there; otherwise train them,
# so the image always starts with a working regressor and classifier.
RUN set -eu; \
    for task in regression classification; do \
      if [ ! -f "models/${task}-gbm/model.joblib" ]; then \
        amesml --quiet train --data data/train.csv --task "$task" --model gbm --out models; \
      fi; \
    done

FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="amesml" \
      org.opencontainers.image.description="Ames Housing sale-price prediction API" \
      org.opencontainers.image.source="https://github.com/Ahsan-muhd-444/house-price-prediction" \
      org.opencontainers.image.licenses="MIT"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AMESML_HOST=0.0.0.0 \
    AMESML_PORT=8000 \
    AMESML_MODEL_PATH=/app/models/regression-gbm,/app/models/classification-gbm
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin amesml
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=amesml:amesml /src/models/regression-gbm /app/models/regression-gbm
COPY --from=builder --chown=amesml:amesml /src/models/classification-gbm /app/models/classification-gbm
WORKDIR /app
USER amesml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["amesml", "serve"]
