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
