# Installation

amesml needs Python 3.9 or newer.

amesml is distributed through GitHub releases. Install a tagged version
straight from the repository:

```bash
pip install "amesml @ git+https://github.com/Ahsan-muhd-444/house-price-prediction@v0.1.1"            # the library and the amesml command
pip install "amesml[api] @ git+https://github.com/Ahsan-muhd-444/house-price-prediction@v0.1.1"       # plus the REST API (FastAPI, uvicorn)
pip install "amesml[plots] @ git+https://github.com/Ahsan-muhd-444/house-price-prediction@v0.1.1"     # plus figure generation (matplotlib)
```

Extras can be combined: `pip install "amesml[api,plots] @ git+https://github.com/Ahsan-muhd-444/house-price-prediction@v0.1.1"`.

## From a release wheel

Every [GitHub release](https://github.com/Ahsan-muhd-444/house-price-prediction/releases)
has the wheel and source distribution attached. Download the `.whl` and run:

```bash
pip install amesml-0.1.1-py3-none-any.whl
```

## From a clone

```bash
git clone https://github.com/Ahsan-muhd-444/house-price-prediction.git
cd house-price-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[api,plots]"
```

For development, install `".[dev]"` instead; see
[Contributing](../contributing.md).

## The dataset

The repository includes the Kaggle training file at `data/train.csv`. When you
install from GitHub or a release wheel, download it once:

```bash
amesml data download
```

This fetches a copy pinned to a fixed commit of the project repository and
verifies its SHA-256 before saving it. Commands that take `--data` also
download it automatically the first time if you leave the option out.

## Optional extras

| Extra | Adds | For |
|---|---|---|
| `api` | FastAPI, uvicorn, pydantic | `amesml serve` and `amesml.api` |
| `plots` | matplotlib | `amesml plots` and `amesml.plots` |
| `notebook` | Jupyter, matplotlib, seaborn | the original notebook |
| `test` | pytest and the above | running the test suite |
| `dev` | linters, type checker, build tools | contributing |
| `docs` | MkDocs Material, mkdocstrings | building this site |
