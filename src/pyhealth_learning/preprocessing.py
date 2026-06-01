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
) -> pd.DataFrame:
    """Clip impossible values and fill missing numeric/categorical fields.

    - Physiologically impossible BMI / blood-pressure values are set to NaN.
    - Numeric columns are median-imputed; categoricals get a ``"unknown"`` bucket.
    """
    df = df.copy()

    if "bmi" in df:
        df.loc[~df["bmi"].between(*bmi_range), "bmi"] = np.nan
    if "systolic_bp" in df:
        df.loc[~df["systolic_bp"].between(*bp_range), "systolic_bp"] = np.nan

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        if col == "patient_id":
            continue
        df[col] = df[col].fillna("unknown")

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
):
    """Encode, split into train/test and standardize the features.

    Parameters
    ----------
    stratify:
        Keep the class balance equal across train/test. Leave this on for
        classification; set it to ``False`` for regression (a continuous
        target can't be stratified).

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
        ``scaler`` is the fitted :class:`StandardScaler` (or ``None`` if
        ``scale=False``) so you can reuse it on new data later.
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

    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Stash feature names on the scaler-less path too, for downstream plots.
    return X_train, X_test, y_train, y_test, {"scaler": scaler, "features": feature_names}
