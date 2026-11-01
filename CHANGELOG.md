# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `model: xgboost` and `model: lightgbm` for regression and classification,
  seeded from `random_state`. They need the new `boosting` extra
  (`pip install "amesml[boosting]"`); without it, training fails with an
  error naming the extra. `docs/benchmarks.md` now includes both.

### Fixed

- `test_notebook_parity.py` asserted the regression figures with an absolute
  tolerance of one cent, which only held on Windows with Python 3.11-3.13.
  The reference values are unchanged; they are now asserted with a 6%
  relative tolerance, covering the 4.3% spread observed across the CI
  matrix. See `docs/DECISIONS.md` D29.
- `test_cli.py::test_plots_command_wiring` failed wherever the temporary
  directory was long enough for Rich to wrap the filename mid-word. The
  test now widens the console for its own duration.

### Changed

- Documentation no longer claims the defaults reproduce the notebook
  "exactly" or "to the cent" without qualification.

## [0.1.0] - 2026-11-01

First packaged release of the house-price-prediction coursework as `amesml`.

### Added

- `amesml` Python package with a typed public API: `load_ames`,
  `build_preprocessor`, `train_regressor`, `train_classifier`, `train`,
  `evaluate_regressor`, `evaluate_classifier`, `predict`, `save_model` and
  `load_model`.
- Defaults that reproduce the original notebook's published results closely
  (linear regression MAE $21,139.99, R² 0.868; logistic regression accuracy
  94.5%, F1 93.9%). The classification figures reproduce exactly; the
  regression figures reproduce to the cent on Windows with Python 3.11-3.13
  and to within about 5% elsewhere, because they come from an iterative
  solver on an ill-conditioned matrix. See `docs/DECISIONS.md` D1.
- Ridge, lasso and histogram gradient-boosting models for both regression and
  above/below-median classification, with cross-validated default
  regularisation.
- Optional engineered features: total area, house and remodel age, total
  bathrooms and porch area.
- `TrainConfig`, loadable from YAML or JSON, with validation of every option.
- Dataset loading with schema validation and a checksummed, atomic download
  pinned to an immutable URL.
- Model bundles with `metadata.json`: configuration, metrics, feature defaults,
  training-data fingerprint, library versions and a checksum of the model file.
- `amesml` command-line interface: `data download`, `train`, `evaluate`
  (including `--split test` to reproduce held-out metrics), `predict`, `plots`
  and `serve`.
- FastAPI service with `/predict`, `/predict/class`, `/health` and
  `/model/info`, partial-record support and reporting of unseen categories.
- Multi-stage Docker image and Docker Compose file.
- Figures drawn with matplotlib: price distribution, missing values, residuals,
  confusion matrix and permutation importance.
- Test suite with 99% coverage, run on Linux, macOS and Windows for Python
  3.9-3.13.
- CI, release (PyPI trusted publishing), documentation, CodeQL, labeler and
  stale-issue workflows, plus Dependabot.
- MkDocs Material documentation: user guide, model card, benchmarks,
  architecture, design decisions and API reference.
- Contributing guide, Code of Conduct, security policy, governance, support
  page, citation metadata, and issue and pull request templates.

### Changed

- Repository reorganised: the report moved to `docs/`, the notebook's pickled
  models to `models/notebook/`, and the notebook is committed with its outputs
  stripped (the original is kept in `notebooks/archive/`).
- The author name and the licence's copyright holder corrected to Muhammad
  Ahsan.

[Unreleased]: https://github.com/Ahsan-muhd-444/house-price-prediction/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ahsan-muhd-444/house-price-prediction/releases/tag/v0.1.0
