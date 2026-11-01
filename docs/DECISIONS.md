# Design decisions

Decisions made while turning the coursework notebook into the `amesml`
package, with the evidence behind each. Entries marked **Deviation** differ
from the original specification for the conversion; each says why.

## Reproducing the notebook

### D1. The defaults reproduce the notebook closely

The package's first promise is reproducibility, so `TrainConfig()` with no
arguments rebuilds the notebook's models: the same 80/20 split with seed 42,
the same column typing, imputation and encoding, `LinearRegression` for price
and `LogisticRegression(max_iter=5000)` for the class. Every improvement
(engineered features, regularised models, gradient boosting) is opt-in.

Result, verified by `tests/test_notebook_parity.py` on the real data: MAE
$21,139.99, RMSE $31,790.77, R² 0.8682386; accuracy 0.9452, precision 0.9254,
recall 0.9538, F1 0.9394, confusion matrix `[[152, 10], [6, 124]]` - identical
to the notebook's recorded outputs on Windows with Python 3.11-3.13.

The classification figures reproduce exactly on every platform. The
regression figures do not: they land within about 5% elsewhere, for the
reason given in D2 and quantified in D29.

### D2. The notebook's linear regression was solved by LSQR, and amesml keeps it so

The first version of the package produced dense design matrices and got
MAE $20,893 and R² 0.853 - not the notebook's numbers, although logistic
regression matched exactly. The cause: scikit-learn's `ColumnTransformer`
returns a **sparse** matrix when fewer than 30% of entries are non-zero, which
is true here (34 numeric columns beside about 250 one-hot columns). For sparse
input `LinearRegression` does not solve least squares exactly; it runs the
iterative LSQR solver, which stops at a tolerance on this ill-conditioned,
unscaled, rank-deficient design. The notebook's published figures are LSQR's
solution. (Logistic regression converges to the same optimum either way.)

amesml keeps scikit-learn's default sparse output for the linear models, so
the defaults reproduce the notebook, and gives a dense matrix only to gradient
boosting, which rejects sparse input. The exact least-squares solution is
slightly worse on R² (0.853), so nothing is lost.

### D3. Feature engineering is optional and off by default

**Deviation.** The specification asked for feature engineering "reproduced
from the notebook", but the notebook engineers no features. `features.py`
implements the standard Ames derivations (total square footage, house age,
remodel age, total bathrooms, porch area) as pure functions, enabled with
`engineer_features: true`. Turning them on by default would have broken D1.
Cross-validation shows they help gradient boosting (CV RMSE $29,586 to
$28,881) and make little difference to the linear models, whose features they
nearly duplicate. The remodel age is clipped at zero because one record lists
a remodel after its sale year.

### D4. Only `NA` and empty fields mean "missing"

In this dataset `NA` encodes absence ("no pool"), while the string `"None"` is
a real category (`MasVnrType`). pandas' default missing-value list includes
`"None"`, so `read_ames_csv` uses `keep_default_na=False` with `NA` and `""`
only - as the notebook did. Adding `""` changes nothing for the Kaggle file
(it has no empty fields) and makes user-supplied CSVs more forgiving.

## Preprocessing and models

### D5. Missing categories become an explicit `"Missing"` level

**Deviation.** The specification suggested most-frequent imputation for
categorical columns. In Ames a missing `PoolQC` means "no pool"; filling it
with the most frequent value among houses that do have a pool would invent a
pool. amesml fills text columns with the constant `"Missing"`, as the notebook
did. Category codes stored as numbers (`MSSubClass`, `MoSold`) get their own
branch with most-frequent imputation, because one column cannot be encoded
when it mixes the string `"Missing"` with integers.

`None` and `pd.NA` are converted to NaN before imputation: under pandas 2 an
object column keeps Python `None`, which `SimpleImputer` does not treat as
missing and which would otherwise become a category of its own. Found by the
Python 3.9/3.10 test runs.

### D6. `drop_first` stays on, and unseen categories are reported by name

The notebook one-hot encoded with `drop="first"`, avoiding the dummy-variable
trap for the unregularised models, and dropping it changes the results. With
`drop` set, scikit-learn warns on every call that meets a category absent from
training, naming columns only by position ("columns [14, 29]"). amesml
silences that warning and reports the values itself -
`{"RoofMatl": ["Membran"], "Electrical": ["Mix"]}` on the reference split -
in `Predictions.unseen_categories`, the CLI log and the API response.

### D7. What `ridge` and `lasso` mean for classification

`ridge` is L2-penalised logistic regression and `lasso` is L1-penalised
logistic regression - the standard "ridge/lasso logistic regression" - with
`C = 1 / alpha`, so in both tasks a larger `alpha` means a simpler model.
`RidgeClassifier` was rejected because it has no `predict_proba`, and the API
returns probabilities. L1 uses the `liblinear` solver: same scores as `saga`,
in 0.4 s instead of 14 s.

scikit-learn 1.8 deprecated `LogisticRegression(penalty=...)` (removal in
1.10) in favour of `l1_ratio`, but before 1.8 `l1_ratio` is ignored unless
`penalty="elasticnet"`. `l1_logistic_regression` chooses the right spelling
for the installed version; both branches are tested.

### D8. Default regularisation strengths come from cross-validation on the training split

`alpha` defaults are 10 (ridge) and 300 (lasso) for regression, 30 and 3 for
classification. Each was chosen by 5-fold cross-validation on the 1,168
training rows with the final preprocessing. The test rows were never used:
picking defaults by test score would leak the test set into the model. One
early experiment illustrates why this matters: a log-transformed target looked
better on the test split (R² 0.913) but was far worse under cross-validation
(RMSE $40,428 against $33,170), because of a few extreme houses.

### D9. Scale only where it matters

Numeric features are standardised for the penalised and logistic models,
which need it. Plain least squares is scale-invariant and the notebook did not
scale it (scaling would also change the LSQR path in D2); tree models split on
thresholds and ignore scale.

### D10. The classification threshold comes from the training rows

`threshold: median` resolves to the median sale price of the training rows
($165,000 on the reference split), as in the notebook, so no test price
influences the model. With `stratify: true` the split is balanced on the
full-data median, which only shapes the split; the model's threshold is still
re-derived from its training rows.

## Data and models on disk

### D11. The dataset download is pinned to a commit and checksummed

**Deviation.** The specification suggested the Kaggle mirror or the De Cock
original from `jse.amstat.org`. Kaggle requires a login, and the De Cock file
has 2,930 rows with different column names (a separate adapter, listed on the
roadmap). `download_ames` fetches this repository's own `data/train.csv` at a
fixed commit - an immutable URL - checks its SHA-256 before writing anything,
and writes through a temporary file so a failed or tampered download never
replaces good data. An explicit missing path is never downloaded to; only the
default location is.

### D12. `.gitattributes` keeps the dataset byte-identical on every platform

The original clone of this repository on Windows produced `train.csv` with
CRLF line endings (Git for Windows defaults to `core.autocrlf=true`), whose
hash no longer matched the pinned checksum. GitHub's Windows runners use the
same default. `data/** -text` stops Git from rewriting the data on any
platform.

### D13. A model is a directory with a checksum

`save_model` writes `model.joblib` beside `metadata.json` (configuration,
metrics, input features and defaults, the training data's fingerprint, library
versions and the model file's SHA-256), and `train` adds `metrics.json`.
`load_model` refuses a model file whose checksum no longer matches. This
catches corrupted and swapped files; it is not a security boundary, because
unpickling executes code. The documentation says to load only trusted models.

### D14. Training-data fingerprints are platform-independent

`dataset_fingerprint` hashes the frame serialised as CSV with `\n` line
endings, not the source file, so the same data has the same fingerprint on
every operating system and regardless of the file it came from. `amesml
evaluate` uses it to warn when you score a model on its own training data.

## Prediction and the API

### D15. Omitted fields take training defaults; `null` means absent

A request that leaves a field out gets the model's typical training value.
That value counts "missing" as a value of its own: for `PoolQC` the most
common value is missing (no pool). Ignoring missing values would make the
default `"Gd"` and give every house with an unspecified pool a good one. A
field sent as `null` means "absent" and is handled exactly as in training.

### D16. Unknown request fields are rejected

`{"GrLivAre": 1500}` returns `422 Unknown feature(s): GrLivAre` instead of
silently pricing a house of default size. Text in a numeric field is also a
`422`. One process can serve one model per task; `AMESML_MODEL_PATH` takes a
comma-separated list.

### D17. Models load at startup, and startup fails loudly

A missing or invalid model stops the service before it accepts requests, so a
misconfigured container fails its deployment instead of its first user.

## Packaging and compatibility

### D18. Python 3.9 stays supported, and is proven by tests, not by mypy

Python 3.9 reached end of life in October 2025, but the specification asked
for it, and supporting it costs little. mypy 2 can no longer target 3.9, so
`python_version = "3.10"` in the mypy configuration and 3.9 compatibility is
established by running the full suite on 3.9 (locally and in CI). The test
matrix is Linux, macOS and Windows × Python 3.9, 3.11, 3.12 and 3.13; the
whole suite was also run locally on 3.10. Python 3.10 and 3.13 are added to
the classifiers because they are tested.

The declared minimum versions (`pandas>=1.5`, `numpy>=1.23`,
`scikit-learn>=1.3`, `typer>=0.9`, `fastapi>=0.100`, and so on) were verified
by resolving the lowest allowed versions on Python 3.9: all 224 tests pass,
apart from two warnings raised inside those old libraries (Starlette 0.27's
test client leaves a stream open; joblib 1.3 cannot count CPU cores on
Windows 11, which removed `wmic`).

### D19. Runtime-annotated modules spell unions with `Optional`

Typer reads the CLI's signatures, and Pydantic the API's models, at runtime.
Python 3.9 cannot evaluate `X | None` at runtime, so `cli.py` and `api.py` use
`Optional[...]` and skip `from __future__ import annotations`. Every other
module uses the modern syntax. A comment in each file explains why.

### D20. matplotlib is an optional `plots` extra

**Deviation.** The core dependency list has no plotting library, and the API
image should not carry one, so plotting is the `plots` extra rather than a
core dependency. `plots.py` uses matplotlib's object-oriented `Figure` API and
never touches `pyplot`, so it keeps no global state and needs no display.

### D21. A `test` extra, and pinned linters

**Deviation.** The specification listed test tools under `dev`. A smaller
`test` extra installs on every supported Python and is what the CI matrix
uses. `dev` adds the tooling, which needs Python 3.11+ (pre-commit and
pandas-stubs); environment markers keep `pip install -e ".[dev]"` working on
older versions. `ruff` and `mypy` are pinned exactly so CI cannot start
failing without a code change; Dependabot proposes upgrades.

### D22. Tests treat every warning as an error

`filterwarnings = ["error"]` makes deprecations fail the build while they are
still cheap to fix. Two third-party warnings are ignored, each with a comment:
Starlette's notice when its test client falls back from `httpx2` (which needs
Python 3.11+) to `httpx`, and pyparsing deprecations triggered inside the old
matplotlib that Python 3.9 receives. The `test` extra installs `httpx2` where
it can.

### D23. Licence metadata uses the PEP 639 expression only

**Deviation.** The specification listed a "License :: OSI Approved :: MIT"
classifier. With `license = "MIT"` (a PEP 639 licence expression), PEP 639
forbids licence classifiers alongside it, and PyPI enforces this at upload -
a rejection that `twine check` does not catch, so it would surface only
during a release. The expression alone conveys the licence.

### D24. The README uses absolute image URLs

The README is also the PyPI project description, where relative image paths
break. Images point at `raw.githubusercontent.com` so they render on both.

## Infrastructure

### D25. The Docker image serves gradient boosting and trains it if needed

The benchmark shows gradient boosting is the strongest model for both tasks,
so the image serves `regression-gbm` and `classification-gbm`. If the build
context contains trained models they are copied; otherwise the builder stage
trains them from `data/train.csv`, so `docker compose up` works from a fresh
clone. `.dockerignore` is an allowlist (only the package, data and models are
sent), the runtime stage holds just the virtual environment and the models,
and the service runs as a non-root user with a health check.

### D26. Release and docs workflows guard themselves

The release workflow refuses a tag that does not match the package version,
publishes with PyPI trusted publishing (no stored token), then creates the
GitHub release with generated notes using the `gh` CLI rather than a
third-party action. The docs workflow also builds on pull requests, so a
broken page is caught before merge; it deploys only from `main`. CI smoke-tests
the built wheel and the running Docker container over HTTP.

### D27. Notebooks, the report and the old models are preserved

Nothing was deleted. The notebook keeps its place with outputs stripped (so
diffs stay readable), and the untouched original with every output is in
`notebooks/archive/`. The PDF report moved to `docs/`. The notebook's pickled
models moved to `models/notebook/` with a note: they record no library version
and are superseded by `amesml train`. `requirements.txt` remains as a pointer
to `.[notebook]`, so old instructions still work. Ruff excludes `notebooks/`.

### D28. Documentation stays on MkDocs Material for now

The specification asked for MkDocs Material. Its maintainers have announced
end of life on 5 May 2027, with Zensical as the successor and a compatibility
path for existing `mkdocs.yml` files. Material is fully supported until then,
so the site uses it as specified; migrating before May 2027 is a small,
mechanical follow-up.

## Bugs found by testing across environments

Running the suite on five Python versions and the minimum dependencies found
real defects before any user did:

- **Lost log lines.** The CLI's log handler did not flush, so under Click 8.1
  (Python 3.9) warnings from commands that then succeeded were silently lost.
- **`None` as a category** under pandas 2 (D5).
- **Read-only arrays.** Under pandas 3 copy-on-write, `np.asarray` on a frame
  returns a read-only view; the missing-value normaliser now copies.
- **Mangled install hint.** Rich parsed `[api]` in `pip install "amesml[api]"`
  as markup and printed `pip install "amesml"`. All interpolated CLI output is
  now escaped.
- **Click 8.2 removed `CliRunner(mix_stderr=...)`** while Python 3.9 can only
  install Click 8.1, which needs it; the tests handle both.

### D29. The notebook-parity test uses a relative tolerance

**Deviation.** D1 originally claimed the regression figures reproduce
exactly. They do not, outside one environment.

`tests/test_notebook_parity.py` asserted MAE, RMSE and R2 with an absolute
tolerance of one cent. That passed on Windows with Python 3.11-3.13 and
failed on every Linux and macOS job, and on Windows with Python 3.9.
Observed MAE across the CI matrix:

| Environment | MAE |
| --- | ---: |
| Windows, Python 3.11-3.13 | $21,139.99 (the notebook's figure) |
| macOS, Python 3.11-3.13 | $20,850.57 |
| Ubuntu 3.11, Windows 3.9 | $20,230.73 |

A spread of about 4.3%.

**The cause is the solver, not the split.** The split is seeded with 42 and
is identical everywhere - the same 292 test rows. Holding that split fixed
and changing only how the linear model is fitted moves MAE from $21,139.99
(sparse, LSQR) to $20,893.06 (dense, exact least squares): about $247 on
identical data. As D2 explains, the design matrix is sparse, ill-conditioned
and rank-deficient, so `LinearRegression` runs LSQR, which stops on a
tolerance rather than converging. Where it stops depends on the
scikit-learn, scipy and BLAS build in use.

**Decision.** Keep $21,139.99 and the other published figures as the
reference values, and assert them with `pytest.approx(..., rel=0.06)`.

**Rejected alternatives.**

- *Pin the dependencies to the versions that produce $21,139.99.* `amesml` is
  a library. Freezing downstream users to one resolution of scikit-learn,
  scipy and numpy so that one test passes is a worse trade than a looser
  assertion.
- *Mark the test `xfail` off Windows.* That hides the check across most of
  the matrix, which is where a regression would actually surface.
- *Drop the parity test.* It is the package's central correctness claim.

6% leaves margin over the observed 4.3% while staying far tighter than the
gap to any other model: ridge ($19,006), lasso ($18,550) and gradient
boosting ($16,826) all sit 10-20% from the linear baseline, so a modelling
or preprocessing regression still fails the test.

The classification figures are asserted exactly, unchanged: logistic
regression converges to the same optimum on sparse and dense input (D2), and
reproduced to the cent on all sixteen CI jobs.
