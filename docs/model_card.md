# Model card

Following the structure of Mitchell et al., *Model Cards for Model Reporting*
(FAT* 2019). It covers the models amesml trains with its defaults; the numbers
come from `scripts/benchmark.py` and the held-out split described below.

## Model details

- **Developer:** Muhammad Ahsan, Software Engineering, COMSATS University Lahore.
- **Version:** amesml 0.1.1 (November 2026).
- **Type:** supervised tabular models built with scikit-learn.
    - *Regression* predicts the sale price in US dollars.
    - *Classification* predicts whether a house sells at or above a threshold
      price - by default the median sale price of the training rows
      ($165,000 on the reference split).
- **Families:** ordinary least squares / logistic regression (the original
  notebook's models), ridge and lasso (L2- and L1-penalised), and histogram
  gradient boosting.
- **Inputs:** up to 79 property attributes from the Ames dataset (size,
  quality and condition ratings, age, location, amenities, sale type).
- **Licence:** MIT.
- **Contact:** [GitHub issues](https://github.com/Ahsan-muhd-444/house-price-prediction/issues).

## Intended use

- **Primary use:** teaching, learning and demonstrating a reproducible,
  production-style machine-learning workflow: packaging, testing, serving and
  documenting a model end to end.
- **Primary users:** students, instructors and engineers studying ML
  engineering practice.
- **Out of scope:** pricing, buying, selling, lending, insuring, taxing or
  appraising real property. The models describe one Iowa city between 2006
  and 2010; they do not know today's market, other regions, or anything not in
  the 79 columns. Do not use them for decisions that affect people.

## Factors

The data contains no information about buyers, sellers or occupants. Relevant
factors are properties of the houses: price band, age, neighbourhood, size and
quality. Neighbourhood is a feature, and because neighbourhood composition can
correlate with protected characteristics, location-driven predictions can
encode historic patterns of segregation and lending (see *Ethical
considerations*).

## Metrics

- **Regression:** mean absolute error (dollars; easy to interpret), root mean
  squared error (penalises large misses), R² and mean absolute percentage
  error (comparable across price levels).
- **Classification:** accuracy, precision, recall, F1 and ROC-AUC. The classes
  are balanced by construction (the threshold is a median), so accuracy is
  meaningful; ROC-AUC measures ranking independent of the threshold.
- **Uncertainty:** a single 292-house test split is noisy, so models are also
  compared with 5-fold cross-validation (mean ± standard deviation).

## Training and evaluation data

- **Dataset:** the Kaggle *House Prices: Advanced Regression Techniques*
  training file - 1,460 residential sales in Ames, Iowa, 2006-2010, with 81
  columns (an `Id`, 79 candidate features and `SalePrice`). It is a subset of
  the 2,930-sale dataset compiled by De Cock (2011) from the Ames City
  Assessor's Office.
- **Preprocessing:** `NA` means "absent" (no pool, no basement) and becomes an
  explicit `Missing` category; numeric gaps take the training median, except
  veneer area and garage year where a gap means "none" and becomes 0. Category
  codes (`MSSubClass`, `MoSold`) are one-hot encoded. All statistics are
  learned from the training rows only.
- **Split:** 80% training (1,168 houses), 20% held out (292), shuffled with
  seed 42 - the notebook's split.

## Quantitative analyses

Held-out test split, default configuration of each model. The linear
regression row reproduces the notebook to the cent only on Windows with
Python 3.11-3.13; on other platforms and Python versions it lands within
about 5%, because that model is fitted by an iterative solver on an
ill-conditioned sparse matrix (see DECISIONS D1 and D2). The classification
figures reproduce exactly everywhere.

| Model | MAE | RMSE | R² | 5-fold CV RMSE |
|---|---:|---:|---:|---:|
| Linear (notebook) | $21,140 | $31,791 | 0.868 | $34,316 ± 10,474 |
| Ridge | $19,006 | $30,751 | 0.877 | $33,279 ± 11,554 |
| Lasso | $18,550 | $31,581 | 0.870 | $33,667 ± 12,177 |
| Gradient boosting | $16,826 | $29,462 | 0.887 | $29,586 ± 6,396 |

| Model | Accuracy | F1 | ROC-AUC | 5-fold CV ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic (notebook) | 94.5% | 93.9% | 0.982 | 0.973 ± 0.007 |
| Gradient boosting | 95.2% | 94.8% | 0.989 | 0.977 ± 0.005 |

### Error by subgroup (regression, held-out split)

| Price band | Houses | Linear MAE | Linear MAPE | Boosting MAE | Boosting MAPE |
|---|---:|---:|---:|---:|---:|
| under $130k | 83 | $16,703 | 19.4% | $12,089 | 15.0% |
| $130k-165k | 81 | $18,284 | 12.6% | $10,479 | 7.2% |
| $165k-215k | 60 | $16,846 | 9.0% | $11,867 | 6.3% |
| over $215k | 68 | $33,747 | 10.0% | $34,544 | 10.6% |

| House age at sale | Houses | Linear MAPE | Boosting MAPE |
|---|---:|---:|---:|
| 0-10 years | 83 | 9.2% | 7.8% |
| 11-40 years | 91 | 11.9% | 8.6% |
| over 40 years | 118 | 17.0% | 12.6% |

**Relative error is highest for the cheapest and oldest houses** - roughly
twice the error rate of mid-market houses. **Absolute error is highest for
the most expensive houses**, and the largest single misses are luxury homes
that both models under-predict by around $200,000 (a $755,000 sale predicted
at about $504,000 by gradient boosting).

## Ethical considerations

- **Uneven accuracy falls on cheaper homes.** If these models were used to
  value property, owners of lower-priced and older homes would see the least
  accurate estimates relative to their home's value - the same groups for
  whom an error matters most. This is the main reason the intended use
  excludes real decisions.
- **Location as a proxy.** Neighbourhood strongly drives predictions. In
  housing, location-based valuation has historically carried discriminatory
  lending and appraisal patterns; a model trained on past sales reproduces
  past prices, including any bias in them.
- **No personal data.** The dataset describes properties and sales, not
  people.

## Caveats and recommendations

- The data is from 2006-2010, one city, and includes the start of the 2008
  housing crash. Prices since then are not represented.
- Four sales have more than 4,000 sq ft of living area. Two are priced like
  ordinary houses ($160,000 and $184,750) and are true outliers; De Cock
  recommends removing houses over 4,000 sq ft for teaching use. The models keep
  them, as the notebook did.
- A handful of categories occur only in the test split (for example the
  `Membran` roof material); unseen categories are reported, not silently
  guessed.
- Prefer the cross-validated scores when comparing models; the single split's
  differences are often within noise.
- Retrain and re-evaluate before any use outside this dataset.

## How to reproduce

```bash
pip install "amesml @ git+https://github.com/Ahsan-muhd-444/house-price-prediction@v0.1.1"
amesml train                                  # notebook linear regression
amesml train --task classification            # notebook logistic regression
amesml train --model gbm
python scripts/benchmark.py                   # every number on this page
```

Each saved model's `metadata.json` records the configuration, library versions
and a SHA-256 fingerprint of its training data.

## References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an
  End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
  <https://doi.org/10.1080/10691898.2011.11889627>
- Mitchell, M., et al. (2019). Model Cards for Model Reporting. *Proceedings of
  the Conference on Fairness, Accountability, and Transparency*, 220-229.
  <https://doi.org/10.1145/3287560.3287596>
