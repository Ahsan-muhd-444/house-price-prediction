# Notebooks

`house_price_prediction.ipynb` is the original analysis: data inspection,
missing-value and outlier analysis, the train/test split, the preprocessing
pipeline, a linear regression for the sale price and a logistic regression for
"above or below the median price". Its outputs are stripped to keep diffs
readable; `archive/house_price_prediction.ipynb` is the same notebook with
every output and figure as it was originally run.

## The package supersedes the modelling code

Everything from the train/test split onwards now lives in the `amesml` package,
and reproduces the notebook's published numbers exactly - to the cent for the
regression. In a notebook you can replace the modelling cells with:

```python
from amesml import TrainConfig, load_ames, train

houses = load_ames("../data/train.csv")
regression = train(houses)  # the notebook's linear regression
classification = train(houses, TrainConfig(task="classification"))

print(regression.metrics)       # MAE 21,139.99, RMSE 31,790.77, R² 0.8682
print(classification.metrics)   # accuracy 0.9452, F1 0.9394
```

## Running the notebook

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/house_price_prediction.ipynb
```

The notebook was last run with Python 3.13 and pandas 3. A pre-commit hook
(`nbstripout`) strips outputs on commit; the archived copy is excluded from it.
