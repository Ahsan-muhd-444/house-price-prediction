# Contributing

Thank you for helping improve amesml. Bug reports, documentation fixes and new
features are all welcome. By taking part you agree to follow the
[Code of Conduct](https://github.com/Ahsan-muhd-444/house-price-prediction/blob/main/CODE_OF_CONDUCT.md).

## Development setup

The package supports Python 3.9+; the development tools need Python 3.11 or
newer.

```bash
git clone https://github.com/Ahsan-muhd-444/house-price-prediction.git
cd house-price-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install                       # pip install -e ".[dev,docs]" and pre-commit hooks
```

Without `make` (for example on Windows), run the commands from the
[Makefile](https://github.com/Ahsan-muhd-444/house-price-prediction/blob/main/Makefile)
directly.

## Make targets

| Target | Runs |
|---|---|
| `make lint` | `ruff check .` and `ruff format --check .` |
| `make format` | applies ruff's fixes and formatting |
| `make typecheck` | `mypy src` (strict) |
| `make test` | `pytest` with coverage; fails below 85% |
| `make build` | builds the sdist and wheel, then `twine check` |
| `make docs` | `mkdocs build --strict`; `make docs-serve` previews it |
| `make docker` | builds the API image |
| `make benchmark` | re-runs `scripts/benchmark.py` and rewrites `docs/benchmarks.md` |
| `make clean` | removes build output and caches |

## Tests and checks

Before opening a pull request, run:

```bash
make lint typecheck test docs
```

- Tests use a synthetic Ames-shaped dataset from `tests/conftest.py`, so they
  need no download. `tests/test_notebook_parity.py` also checks the published
  results against the real `data/train.csv`.
- Warnings are errors in the test suite. If a dependency starts warning, fix
  the cause; add a filter to `pyproject.toml` only for third-party noise, with
  a comment saying why.
- New behaviour needs tests. Coverage is above 99%; please keep it there.
- Library code logs through `logging.getLogger("amesml")` and never prints.

## Branches and pull requests

1. Open an issue first for anything larger than a small fix, so the approach
   can be agreed before you write code.
2. Branch from `main`: `feat/short-description`, `fix/...` or `docs/...`.
3. Keep each pull request to one change, with tests and documentation.
4. Fill in the pull request template. CI must pass before review.
5. The maintainer squash-merges; the pull request title becomes the commit
   message, so it must follow the convention below.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```text
feat: add an XGBoost model option
fix: keep "None" as a MasVnrType category when reading CSVs
docs: explain partial records in the REST API guide
test: cover the 503 path for a missing classifier
ci: cache pip downloads in the docs workflow
chore: bump ruff to 0.18
```

Use `feat!:` or a `BREAKING CHANGE:` footer for incompatible changes.

## Releasing (maintainers)

1. Update `CHANGELOG.md`: move items from *Unreleased* under a new version
   heading with the date.
2. Bump `__version__` in `src/amesml/_version.py` (semantic versioning).
3. Commit (`chore: release vX.Y.Z`), then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "amesml vX.Y.Z"
   git push origin main vX.Y.Z
   ```

4. The release workflow checks that the tag matches the version, builds the
   distributions, publishes them to PyPI with trusted publishing and creates a
   GitHub release with generated notes.
