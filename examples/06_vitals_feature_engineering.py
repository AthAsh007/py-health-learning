"""
Example 06 — Time-series feature engineering on longitudinal vitals.

The vitals table has one row per patient per day. Models want one row per
patient, so we aggregate each patient's daily readings into summary features
(mean, last value, and trend), then predict who deteriorates.

    python examples/06_vitals_feature_engineering.py
"""

import numpy as np

from pyhealth_learning import data, preprocessing, models, evaluation, utils

VITALS = ["heart_rate", "systolic_bp", "resp_rate", "temperature_c", "spo2"]


def featurize(vitals):
    """Collapse the long per-day table into one feature row per patient."""
    # Fill the small amount of missing spo2 with each patient's own median.
    vitals = vitals.copy()
    vitals["spo2"] = vitals.groupby("patient_id")["spo2"].transform(
        lambda s: s.fillna(s.median())
    )

    def per_patient(group):
        group = group.sort_values("day")
        feats = {"deteriorated": int(group["deteriorated"].iloc[0])}
        for col in VITALS:
            feats[f"{col}_mean"] = group[col].mean()
            feats[f"{col}_last"] = group[col].iloc[-1]
            # Simple linear trend (slope) over the recorded days.
            feats[f"{col}_slope"] = np.polyfit(group["day"], group[col], 1)[0]
        return feats

    rows = [per_patient(g) for _, g in vitals.groupby("patient_id")]
    import pandas as pd

    return pd.DataFrame(rows)


def main() -> None:
    utils.set_seed(42)

    utils.section("Raw longitudinal vitals")
    vitals = data.make_synthetic_vitals(n_patients=200, n_days=10)
    n_patients = vitals["patient_id"].nunique()
    print(f"rows={len(vitals)}  patients={n_patients}  days/patient={len(vitals)//n_patients}")
    print(vitals.head())

    utils.section("Aggregate to one row per patient")
    features = featurize(vitals)
    print(f"feature matrix: {features.shape}")
    print("deterioration rate:", round(features["deteriorated"].mean(), 3))

    utils.section("Train deterioration classifier")
    X_train, X_test, y_train, y_test, _ = preprocessing.split_and_scale(
        features, target="deteriorated"
    )
    model = models.train_classifier(X_train, y_train, kind="forest")
    print(evaluation.evaluate_classifier(model, X_test, y_test)["summary"])


if __name__ == "__main__":
    main()
