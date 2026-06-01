"""
Preprocessing helpers: cleaning, encoding, splitting and scaling.

These wrap common scikit-learn / pandas idioms so the example scripts stay
readable while still showing the real steps you'd take on clinical tabular
data.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__all__ = [
    "clean_patient_table",
    "encode_categoricals",
    "split_and_scale",
]


def clean_patient_table(
    df: pd.DataFrame,
    bmi_range: tuple[float, float] = (10.0, 70.0),
    bp_range: tuple[float, float] = (60.0, 250.0),
    impute: bool = True,
) -> pd.DataFrame:
    """Clean a patient table: clip impossible values, fill categoricals, and
    (optionally) median-impute numeric gaps.

    The steps are deliberately split by *what information they use*:

    - **Row-local** fixes use only the value in front of them, so they're safe
      to run before any train/test split:
        * physiologically impossible BMI / blood-pressure values are set to NaN;
        * missing categoricals get a constant ``"unknown"`` bucket.
    - **Statistical** fixes summarise a whole column and can therefore leak test
      information into training. Numeric **median imputation** is one of these.

    Parameters
    ----------
    impute:
        If ``True`` (default) median-impute the numeric columns here — convenient
        when you just want one clean table to explore. For a modelling workflow
        set ``impute=False`` and let :func:`split_and_scale` (or a
        :func:`pyhealth_learning.models.make_classifier_pipeline`) fit the median
        on the *training* fold only. See ``examples/09_data_leakage_demo.py``.
    """
    df = df.copy()

    # --- row-local fixes (leak nothing; safe before a split) ---
    if "bmi" in df:
        df.loc[~df["bmi"].between(*bmi_range), "bmi"] = np.nan
    if "systolic_bp" in df:
        df.loc[~df["systolic_bp"].between(*bp_range), "systolic_bp"] = np.nan

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        if col == "patient_id":
            continue
        df[col] = df[col].fillna("unknown")

    # --- statistical fix (uses the column median => fit on train only) ---
    if impute:
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_categoricals(
    df: pd.DataFrame, drop: Iterable[str] = ("patient_id",)
) -> pd.DataFrame:
    """One-hot encode object columns, dropping ID-like columns first."""
    df = df.drop(columns=[c for c in drop if c in df.columns])
    cat_cols = df.select_dtypes(include="object").columns
    return pd.get_dummies(df, columns=list(cat_cols), drop_first=True)


def split_and_scale(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.25,
    seed: int = 42,
    scale: bool = True,
    stratify: bool = True,
    impute: bool = True,
):
    """Encode, split into train/test, then impute and standardize the features.

    Both statistical transforms are **fit on the training fold only** and merely
    *applied* to the test fold, so no test information leaks into training.

    Parameters
    ----------
    stratify:
        Keep the class balance equal across train/test. Leave this on for
        classification; set it to ``False`` for regression (a continuous
        target can't be stratified).
    impute:
        Median-impute missing values (fit on train). Harmless no-op when the
        data has no gaps. Pair this with ``clean_patient_table(..., impute=False)``
        for a leak-free workflow.

    Returns
    -------
    X_train, X_test, y_train, y_test, meta
        ``meta`` carries the fitted ``"imputer"`` and ``"scaler"`` (either may be
        ``None``) plus the ``"features"`` name list, so you can apply the exact
        same transforms to new data later.
    """
    encoded = encode_categoricals(df)
    y = encoded[target].values
    X = encoded.drop(columns=[target])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if stratify else None,
    )

    imputer = None
    if impute:
        imputer = SimpleImputer(strategy="median").fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        {"imputer": imputer, "scaler": scaler, "features": feature_names},
    )
