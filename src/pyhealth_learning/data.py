"""
Datasets for the learning examples.

Two kinds of data are provided:

1. **Real bundled datasets** that ship with scikit-learn (no download needed),
   wrapped as tidy :class:`pandas.DataFrame` objects.
2. **Synthetic patient records** generated reproducibly so you can practice
   cleaning, encoding and missing-value handling without touching protected
   health information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

__all__ = [
    "load_breast_cancer_classification",
    "load_diabetes_regression",
    "load_diabetes_classification",
    "make_synthetic_patients",
]


def load_breast_cancer_classification() -> pd.DataFrame:
    """Return the breast-cancer diagnostic dataset as a DataFrame.

    The binary target column ``malignant`` is 1 for malignant, 0 for benign.
    """
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # sklearn encodes target as 0=malignant, 1=benign; flip for readability.
    df["malignant"] = (bunch.target == 0).astype(int)
    df = df.drop(columns=["target"])
    return df


def load_diabetes_regression() -> pd.DataFrame:
    """Return the diabetes progression dataset (regression target)."""
    bunch = load_diabetes(as_frame=True, scaled=False)
    df = bunch.frame.copy()
    df = df.rename(columns={"target": "progression"})
    return df


def load_diabetes_classification(threshold: float = 140.0) -> pd.DataFrame:
    """Turn the diabetes regression target into a binary label.

    Parameters
    ----------
    threshold:
        Disease-progression score above which a patient is labelled positive.
        The default (140) roughly splits the cohort in half.
    """
    df = load_diabetes_regression()
    df["diabetes"] = (df.pop("progression") > threshold).astype(int)
    return df


def make_synthetic_patients(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a messy, realistic-looking synthetic patient table.

    Includes categorical fields, a few injected missing values and an outlier
    or two so the preprocessing examples have something to clean. The
    ``readmitted`` target is a noisy function of the features.

    Parameters
    ----------
    n:
        Number of patient rows to generate.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, size=n)
    sex = rng.choice(["M", "F"], size=n)
    bmi = np.round(rng.normal(27, 5, size=n), 1).clip(15, 55)
    systolic_bp = rng.normal(128, 16, size=n).round().clip(80, 220)
    glucose = rng.normal(110, 30, size=n).round().clip(60, 300)
    smoker = rng.choice(["never", "former", "current"], size=n, p=[0.5, 0.3, 0.2])
    n_prior_visits = rng.poisson(2, size=n)

    # Latent risk -> probability of 30-day readmission.
    risk = (
        0.03 * (age - 50)
        + 0.04 * (bmi - 27)
        + 0.02 * (systolic_bp - 128)
        + 0.015 * (glucose - 110)
        + 0.25 * (smoker == "current")
        + 0.20 * n_prior_visits
    )
    prob = 1 / (1 + np.exp(-(risk - 1.0)))
    readmitted = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"P{1000 + i}" for i in range(n)],
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "glucose": glucose,
            "smoker": smoker,
            "n_prior_visits": n_prior_visits,
            "readmitted": readmitted,
        }
    )

    # Inject some missingness so cleaning examples are meaningful.
    for col, frac in [("bmi", 0.05), ("glucose", 0.08), ("smoker", 0.03)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    # A couple of obvious data-entry outliers.
    if n > 10:
        df.loc[df.index[3], "systolic_bp"] = 400  # impossible BP
        df.loc[df.index[7], "bmi"] = 1.0           # impossible BMI

    return df
