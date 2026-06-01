# py-health-learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AthAsh007/py-health-learning/blob/main/notebooks/colab_quickstart.ipynb)

A small, **dependency-light teaching package** for learning healthcare
machine-learning workflows in Python. It runs out of the box in **Google
Colab** — no datasets to download, no GPU required.

Everything is built on `numpy`, `pandas`, `scikit-learn` and `matplotlib`
(all pre-installed in Colab).

---

## What's inside

```
py-health-learning/
├── src/pyhealth_learning/      # the importable package
│   ├── data.py                 # bundled + synthetic patient datasets
│   ├── preprocessing.py        # cleaning, encoding, split + scale
│   ├── models.py               # train classifiers / regressors
│   ├── evaluation.py           # metrics + ROC / confusion / importance plots
│   └── utils.py                # seeding, pretty printing
├── examples/                   # runnable end-to-end scripts (01–04)
├── notebooks/
│   └── colab_quickstart.ipynb  # ← open this in Colab
├── tests/                      # pytest smoke tests
└── data/                       # (no PHI — see data/README.md)
```

---

## Run it in Google Colab (recommended)

**Click the badge at the top of this README**, or:

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **File → Open notebook → GitHub** tab.
3. Paste `AthAsh007/py-health-learning` and open
   `notebooks/colab_quickstart.ipynb`.
4. **Runtime → Run all.**

The first cell clones this repo into Colab and `pip install -e .`'s it, so
Colab can **import every `.py` module** and run the example scripts directly:

```python
!git clone https://github.com/AthAsh007/py-health-learning.git
%cd py-health-learning
!pip install -q -e .

from pyhealth_learning import data, models, preprocessing, evaluation
```

> **Why this works:** Colab gives you a Linux VM. Cloning the repo onto it and
> installing the package puts `pyhealth_learning` on Python's import path, so
> `import pyhealth_learning` resolves to *these source files*. Push a change to
> GitHub, then `!git pull` in Colab to get it.

---

## Run it locally

```bash
git clone https://github.com/AthAsh007/py-health-learning.git
cd py-health-learning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# run any example
python examples/02_train_diabetes_classifier.py
```

---

## The examples

| Script | What it teaches |
| --- | --- |
| `examples/01_explore_data.py` | Load the datasets, inspect shape / class balance / missingness |
| `examples/02_train_diabetes_classifier.py` | Full train → evaluate loop (logistic vs. random forest) |
| `examples/03_evaluate_and_plot.py` | Save ROC curve, confusion matrix and feature-importance plots |
| `examples/04_clean_synthetic_pipeline.py` | Clean messy synthetic data, then cross-validate a readmission model |
| `examples/05_regression_diabetes_progression.py` | Regression path: predict continuous disease progression (MAE + R²) |
| `examples/06_vitals_feature_engineering.py` | Aggregate longitudinal vitals (mean / last / trend) → classify deterioration |
| `examples/07_labs_length_of_stay_regression.py` | Predict hospital length-of-stay from a synthetic lab panel |
| `examples/08_export_datasets_to_csv.py` | Write the synthetic datasets to `data/samples/*.csv` |
| `examples/09_data_leakage_demo.py` | Leaky vs leak-free preprocessing, and the AUC gap it causes |
| `examples/10_cross_validated_pipeline.py` | Compare models with leak-free `Pipeline` cross-validation |
| `examples/11_threshold_tuning.py` | Tune the decision threshold for a high-recall screening use-case |

---

## The datasets

All loaders live in `pyhealth_learning.data`. Real datasets come bundled with
scikit-learn (no download); synthetic ones are generated reproducibly and
contain **no real patient data**.

| Loader | Kind | Target | Shape |
| --- | --- | --- | --- |
| `load_breast_cancer_classification()` | real (sklearn) | `malignant` (binary) | wide tabular |
| `load_diabetes_classification()` | real (sklearn) | `diabetes` (binary) | wide tabular |
| `load_diabetes_regression()` | real (sklearn) | `progression` (continuous) | wide tabular |
| `make_synthetic_patients()` | synthetic | `readmitted` (binary) | messy tabular (missingness + outliers) |
| `make_synthetic_vitals()` | synthetic | `deteriorated` (binary) | **longitudinal** (row per patient-day) |
| `make_synthetic_labs()` | synthetic | `length_of_stay` (continuous) | lab panel |

Small CSV previews of the three synthetic datasets are committed under
[`data/samples/`](data/samples). Regenerate full-size copies any time with:

```bash
python data/generate_samples.py        # or: python examples/08_export_datasets_to_csv.py
```

---

## Quick API tour

```python
from pyhealth_learning import data, preprocessing, models, evaluation

df = data.load_diabetes_classification()
X_train, X_test, y_train, y_test, meta = preprocessing.split_and_scale(
    df, target="diabetes"
)
model = models.train_classifier(X_train, y_train, kind="forest")
print(evaluation.evaluate_classifier(model, X_test, y_test)["summary"])
```

`split_and_scale` fits its imputer and scaler on the **training fold only**, so
the test split never influences its own preprocessing.

### Avoiding data leakage

For cross-validation, wrap preprocessing in a `Pipeline` so it's refit on every
fold's training data — feed it the *raw* (un-imputed, un-scaled) features:

```python
pipe = models.make_classifier_pipeline(kind="forest")   # impute -> scale -> model
scores = models.cross_validate(pipe, X_raw, y, cv=5, scoring="roc_auc")
```

See `examples/09_data_leakage_demo.py` for a side-by-side of the leaky vs
leak-free way (and why it matters).

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

---

## A note on data & privacy

This repo ships **no real patient data**. Real datasets are pulled from
scikit-learn's bundled examples; "messy" practice data is generated
synthetically. Never commit protected health information (PHI). See
[`data/README.md`](data/README.md).
