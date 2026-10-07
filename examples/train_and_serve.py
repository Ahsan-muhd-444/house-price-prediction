"""Train a model with the Python API, save it, then serve and query it.

Run from the repository root:

    pip install -e ".[api]"
    python examples/train_and_serve.py

The script trains a gradient-boosting regressor, saves it under
``models/example-gbm``, starts the REST API on a free local port in a
background thread, sends ``examples/predict_request.json`` to it and prints
the predicted prices.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from amesml import TrainConfig, load_ames, train
from amesml.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    result = train(load_ames(ROOT / "data" / "train.csv"), TrainConfig(model="gbm"))
    print(f"Held-out MAE: ${result.metrics.to_dict()['mae']:,.0f}")
    model_dir = result.save(ROOT / "models" / "example-gbm")

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app([model_dir]), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.1)

    payload = (ROOT / "examples" / "predict_request.json").read_bytes()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.load(response)
    for price in body["predictions"]:
        print(f"Predicted price: ${price:,.0f}")

    server.should_exit = True
    thread.join()


if __name__ == "__main__":
    main()
