# Issues to file

The roadmap as ready-to-paste GitHub issues. For each: copy the title, paste
the body, and apply the labels. Create the labels first if they do not exist
(`good first issue`, `enhancement`, `docs` and `ci`; `help wanted` is optional).

---

## 1. Add an XGBoost/LightGBM model option behind an optional extra

**Labels:** `good first issue`, `enhancement`

**Body:**

`amesml` offers `linear`, `ridge`, `lasso` and `gbm` (scikit-learn's histogram
gradient boosting). XGBoost and LightGBM are the libraries people most often
reach for on tabular data, and comparing them in `docs/benchmarks.md` would be
useful.

Add `model: xgboost` and `model: lightgbm` for both tasks. The libraries should
stay optional: install them with `pip install "amesml[boosting]"`, and raise a
clear error naming that extra if they are missing.

Where to start: `make_estimator` in `src/amesml/models.py`, `MODELS` in
`src/amesml/config.py`, and `build_pipeline` (both need dense input, like
`gbm`).

**Acceptance criteria:**

- [ ] `amesml train --model xgboost` and `--model lightgbm` work for regression and classification.
- [ ] A new `boosting` extra in `pyproject.toml`; a missing library gives an error that names the extra.
- [ ] Both are seeded from `random_state`; the reproducibility test covers them.
- [ ] `scripts/benchmark.py` includes them when installed, and `docs/benchmarks.md` is regenerated.
- [ ] The configuration and CLI docs list the new options.

---

## 2. Add `amesml explain` using SHAP

**Labels:** `good first issue`, `enhancement`

**Body:**

The permutation-importance figure shows which columns matter overall, but not
why a particular house got its price. Add `amesml explain --model PATH --input
CSV [--row N] [--out DIR]` that uses SHAP to explain predictions.

- Without `--row`: write a summary (beeswarm) plot for the input rows.
- With `--row N`: print the top contributing features for that house and write
  a waterfall plot.

SHAP should be an optional extra (`amesml[explain]`). Explanations should be
reported per raw Ames column, not per one-hot level, to match the importance
figure.

**Acceptance criteria:**

- [ ] `amesml explain` works for regression and classification models.
- [ ] Contributions are aggregated back to raw columns.
- [ ] The `explain` extra is optional; without it the command explains how to install it.
- [ ] Tests use the synthetic fixture and run in under 10 seconds.
- [ ] A user-guide section with an example plot.

---

## 3. Hyperparameter search with `amesml tune` (Optuna)

**Labels:** `enhancement`

**Body:**

Default regularisation strengths were chosen by hand with a small
cross-validated grid (see `docs/DECISIONS.md`, D8). A `tune` command would let
users search properly:

```bash
amesml tune --model gbm --trials 50 --out models/
```

Use Optuna with cross-validation on the training split only (the test split
must never inform tuning). Write the best configuration as a YAML file
loadable with `amesml train --config`, and record the study's results.

**Acceptance criteria:**

- [ ] `amesml tune` supports every model family with a sensible search space per model.
- [ ] Only training rows are used for the search; a test asserts the held-out rows are untouched.
- [ ] Produces `best_config.yaml` that `amesml train --config` accepts.
- [ ] Seeded and reproducible; Optuna is an optional extra.
- [ ] Documented in the user guide.

---

## 4. Cross-validation report with confidence intervals

**Labels:** `enhancement`

**Body:**

`amesml train` reports one held-out split. On 292 test houses, single-split
differences between models are often within noise (see the standard
deviations in `docs/benchmarks.md`). Add `amesml cv --model gbm --folds 5` that
reports each metric's mean, standard deviation and a 95% confidence interval,
and optionally saves the fold scores.

**Acceptance criteria:**

- [ ] `amesml cv` works for both tasks; classification uses stratified folds.
- [ ] Output includes mean, standard deviation and a 95% interval for every metric.
- [ ] `--json` writes the per-fold scores.
- [ ] `scripts/benchmark.py` reuses the same code path.
- [ ] Tests cover fold count, seeding and both tasks.

---

## 5. Dataset adapters: Kaggle `test.csv` and the full De Cock data

**Labels:** `enhancement`

**Body:**

`load_ames` supports the Kaggle `train.csv` only. Two related formats are
common:

- **Kaggle `test.csv`**: the same columns without `SalePrice`, used for
  submissions. `amesml predict` already handles it; add a documented loader
  and an example that writes a Kaggle submission file.
- **De Cock's original data** (2,930 sales, `AmesHousing.txt`): tab-separated,
  with different column names (`MS SubClass`, `Year Remod/Add`, ...) and extra
  `Order` and `PID` columns.

Add `load_ames(..., source="kaggle-train" | "kaggle-test" | "decock")` with a
column-name mapping for the De Cock file, and a checksummed download for it.

**Acceptance criteria:**

- [ ] De Cock columns are mapped to the Kaggle names; a test checks every mapped name is in the schema.
- [ ] Checksummed download for the De Cock file, pinned like the Kaggle one.
- [ ] Training on the full 2,930 rows works end to end.
- [ ] An example script producing a Kaggle submission CSV.

---

## 6. Streamlit demo app (`amesml demo`)

**Labels:** `good first issue`, `enhancement`

**Body:**

A small interactive demo would make the project easier to show: sliders and
dropdowns for the most important features (overall quality, living area, year
built, neighbourhood, garage, bathrooms), the predicted price, the
above/below-median probability, and the unseen-category warnings. Start it
with `amesml demo --model models/regression-gbm`.

Fields the user does not set should use the model's training defaults, exactly
as the REST API does (`records_to_frame` in `src/amesml/models.py`).

**Acceptance criteria:**

- [ ] `amesml demo` launches a Streamlit app; Streamlit is an optional extra.
- [ ] Prediction goes through `amesml.models.predict`, not a reimplementation.
- [ ] Works with the models produced by `amesml train` and the Docker image.
- [ ] A screenshot in the README and a short docs page.

---

## 7. Model registry: list and compare saved models

**Labels:** `enhancement`

**Body:**

After a few `amesml train` runs, `models/` holds several directories, each with
a `metadata.json`. Add:

- `amesml models list [--dir models/]`: a table of every saved model with task,
  model family, creation date, key metrics, and whether its training data
  matches the current `data/train.csv` fingerprint.
- `amesml models compare A B`: metrics side by side, plus configuration
  differences.

**Acceptance criteria:**

- [ ] Both commands read only `metadata.json`; they never unpickle a model.
- [ ] Models whose checksum no longer matches their metadata are flagged.
- [ ] `--json` output for scripting.
- [ ] Tests with a temporary directory of saved models.

---

## 8. Tutorial notebook: "From notebook to package"

**Labels:** `docs`

**Body:**

The repository shows the before (the coursework notebook) and the after (the
package), but not the path between them. Write a tutorial notebook that walks
through the conversion step by step:

1. reproducing the notebook's numbers with `amesml`;
2. why the linear regression depended on the sparse LSQR solver (D2);
3. moving preprocessing into a `Pipeline` so serving matches training;
4. configuration, persistence with metadata, and testing;
5. serving the model and calling it.

**Acceptance criteria:**

- [ ] `notebooks/from_notebook_to_package.ipynb`, committed without outputs.
- [ ] Runs top to bottom on a fresh install with `pip install "amesml[notebook]"`.
- [ ] Linked from `notebooks/README.md` and the documentation home page.

---

## 9. Publish the Docker image to GHCR on release

**Labels:** `ci`

**Body:**

The release workflow publishes to PyPI and creates a GitHub release, but users
still have to build the image themselves. On every `v*.*.*` tag, build the
image and push it to `ghcr.io/ahsan-muhd-444/amesml` tagged with the version
and `latest`.

**Acceptance criteria:**

- [ ] A job in `.github/workflows/release.yml` using `docker/build-push-action` with `packages: write`.
- [ ] Multi-architecture build (`linux/amd64`, `linux/arm64`).
- [ ] The image is smoke-tested over HTTP before it is pushed.
- [ ] OCI labels include the version and source revision.
- [ ] The README and Docker guide show `docker run ghcr.io/ahsan-muhd-444/amesml`.

---

## 10. Drift check: compare an incoming batch with the training data

**Labels:** `enhancement`

**Body:**

A model trained on 2006-2010 Ames sales can be asked about very different
houses. Add `amesml drift --model PATH --input CSV` that compares an incoming
batch with the training data and reports columns whose distribution has moved:

- numeric columns: population stability index or a two-sample KS test against
  stored training quantiles;
- categorical columns: share of unseen categories and changes in level
  frequencies.

Training-time statistics (quantiles, category frequencies) should be saved in
`metadata.json` so the check does not need the original data.

**Acceptance criteria:**

- [ ] `metadata.json` gains per-column training statistics (schema version bumped, old files still load).
- [ ] `amesml drift` prints a table of drifted columns with a configurable threshold, and exits non-zero on drift for use in CI.
- [ ] Optionally exposed as `POST /drift` in the API.
- [ ] Tests with synthetic shifted data.
