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
    "make_synthetic_vitals",
    "make_synthetic_labs",
    "save_sample_datasets",
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


def make_synthetic_vitals(
    n_patients: int = 120, n_days: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Generate a *longitudinal* (long-format) vitals time series.

    Each patient contributes one row per day, so the table is suited to
    group-by / time-series feature engineering (see example 06). A latent
    per-patient risk drives a gentle daily drift in the vitals and the binary
    ``deteriorated`` outcome, which is constant within a patient.

    Columns
    -------
    patient_id, day, heart_rate, systolic_bp, resp_rate, temperature_c, spo2,
    deteriorated

    Parameters
    ----------
    n_patients:
        Number of distinct patients.
    n_days:
        Number of consecutive days recorded per patient.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # One latent risk score per patient; higher -> worse, drifting vitals.
    risk = rng.normal(0.0, 1.0, size=n_patients)
    deteriorated = (risk + rng.normal(0, 0.5, size=n_patients) > 0.8).astype(int)

    rows = []
    for p in range(n_patients):
        pid = f"V{2000 + p}"
        drift = risk[p]
        for day in range(n_days):
            # Vitals worsen slightly each day for higher-risk patients.
            t = day / max(n_days - 1, 1)
            heart_rate = rng.normal(78 + 12 * drift * t, 6)
            systolic_bp = rng.normal(124 - 8 * drift * t, 10)
            resp_rate = rng.normal(16 + 4 * drift * t, 2)
            temperature_c = rng.normal(36.8 + 0.6 * drift * t, 0.3)
            spo2 = rng.normal(97 - 3 * drift * t, 1.2)
            rows.append(
                {
                    "patient_id": pid,
                    "day": day,
                    "heart_rate": round(float(heart_rate), 1),
                    "systolic_bp": round(float(systolic_bp), 1),
                    "resp_rate": round(float(resp_rate), 1),
                    "temperature_c": round(float(temperature_c), 2),
                    "spo2": round(float(np.clip(spo2, 70, 100)), 1),
                    "deteriorated": int(deteriorated[p]),
                }
            )

    df = pd.DataFrame(rows)

    # A little realistic missingness in the oxygen-saturation readings.
    mask = rng.random(len(df)) < 0.04
    df.loc[mask, "spo2"] = np.nan

    return df


def make_synthetic_labs(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic lab panel with a continuous regression target.

    The ``length_of_stay`` target (hospital days) is a noisy function of age and
    the lab values, so this dataset is meant for the regression examples
    (:func:`pyhealth_learning.models.train_regressor`).

    Columns
    -------
    patient_id, age, sex, creatinine, bun, wbc, hemoglobin, sodium, potassium,
    crp, length_of_stay

    Parameters
    ----------
    n:
        Number of patient rows to generate.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 92, size=n)
    sex = rng.choice(["M", "F"], size=n)
    creatinine = rng.normal(1.0, 0.35, size=n).clip(0.3, 6.0).round(2)
    bun = rng.normal(16, 7, size=n).clip(4, 90).round(1)
    wbc = rng.normal(7.5, 2.8, size=n).clip(1.5, 30).round(1)
    hemoglobin = rng.normal(13.5, 1.8, size=n).clip(6, 19).round(1)
    sodium = rng.normal(139, 3.5, size=n).clip(120, 155).round(1)
    potassium = rng.normal(4.1, 0.5, size=n).clip(2.5, 7.0).round(2)
    crp = rng.gamma(shape=2.0, scale=6.0, size=n).clip(0, 300).round(1)

    # Length of stay grows with age, renal markers, inflammation and low Hb.
    base = (
        2.0
        + 0.04 * (age - 50)
        + 1.8 * (creatinine - 1.0)
        + 0.05 * (bun - 16)
        + 0.02 * (crp)
        + 0.30 * (13.5 - hemoglobin)
        + 0.10 * (wbc - 7.5)
    )
    length_of_stay = (base + rng.normal(0, 1.2, size=n)).clip(1, 60).round(1)

    return pd.DataFrame(
        {
            "patient_id": [f"L{3000 + i}" for i in range(n)],
            "age": age,
            "sex": sex,
            "creatinine": creatinine,
            "bun": bun,
            "wbc": wbc,
            "hemoglobin": hemoglobin,
            "sodium": sodium,
            "potassium": potassium,
            "crp": crp,
            "length_of_stay": length_of_stay,
        }
    )


def save_sample_datasets(dest: str = "data/samples", seed: int = 42) -> list[str]:
    """Write CSV samples of every synthetic dataset into ``dest``.

    Returns the list of files written. Real bundled datasets
    (breast-cancer, diabetes) are intentionally not exported — they load
    instantly from scikit-learn and shouldn't be committed.

    Parameters
    ----------
    dest:
        Output directory (created if missing).
    seed:
        Seed forwarded to each generator so the files are reproducible.
    """
    import os

    os.makedirs(dest, exist_ok=True)
    written = []
    tables = {
        "synthetic_patients.csv": make_synthetic_patients(n=500, seed=seed),
        "synthetic_vitals.csv": make_synthetic_vitals(seed=seed),
        "synthetic_labs.csv": make_synthetic_labs(seed=seed),
    }
    for name, frame in tables.items():
        path = os.path.join(dest, name)
        frame.to_csv(path, index=False)
        written.append(path)
    return written
