"""Smoke tests that exercise the full learning pipeline."""

import numpy as np

from pyhealth_learning import data, preprocessing, models, evaluation


def test_synthetic_patients_shape_and_target():
    df = data.make_synthetic_patients(n=200, seed=0)
    assert len(df) == 200
    assert df["readmitted"].isin([0, 1]).all()


def test_synthetic_vitals_is_longitudinal():
    df = data.make_synthetic_vitals(n_patients=20, n_days=7, seed=0)
    assert len(df) == 20 * 7
    # Each patient has exactly n_days rows and a single, constant outcome.
    per_patient = df.groupby("patient_id")
    assert (per_patient.size() == 7).all()
    assert (per_patient["deteriorated"].nunique() == 1).all()
    assert df["deteriorated"].isin([0, 1]).all()


def test_synthetic_labs_regression_target():
    df = data.make_synthetic_labs(n=150, seed=0)
    assert len(df) == 150
    assert "length_of_stay" in df.columns
    # Continuous target within the clipped range.
    assert df["length_of_stay"].between(1, 60).all()


def test_regressor_trains_on_labs():
    df = data.make_synthetic_labs(n=400, seed=0)
    X_train, X_test, y_train, y_test, _ = preprocessing.split_and_scale(
        df, target="length_of_stay", stratify=False
    )
    model = models.train_regressor(X_train, y_train)
    report = evaluation.evaluate_regressor(model, X_test, y_test)
    # Length of stay is a (noisy) linear function of the labs, so R² beats 0.
    assert report["r2"] > 0.3


def test_clean_removes_impossible_values():
    raw = data.make_synthetic_patients(n=200, seed=0)
    clean = preprocessing.clean_patient_table(raw)
    assert clean.isna().sum().sum() == 0
    assert clean["systolic_bp"].max() <= 250
    assert clean["bmi"].min() >= 10


def test_clean_without_impute_keeps_numeric_gaps():
    raw = data.make_synthetic_patients(n=300, seed=0)
    clean = preprocessing.clean_patient_table(raw, impute=False)
    # Row-local fixes still happen: impossible values clipped, categoricals filled.
    assert clean["systolic_bp"].max(skipna=True) <= 250
    assert clean.select_dtypes(include="object").isna().sum().sum() == 0
    # But numeric gaps are deliberately left for a train-fit imputer.
    assert clean[["bmi", "glucose"]].isna().sum().sum() > 0


def test_split_and_scale_imputes_on_train_only():
    raw = data.make_synthetic_patients(n=400, seed=0)
    clean = preprocessing.clean_patient_table(raw, impute=False)
    X_train, X_test, y_train, y_test, meta = preprocessing.split_and_scale(
        clean, target="readmitted"
    )
    # No NaNs survive, and the fitted imputer is returned for reuse on new data.
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert meta["imputer"] is not None


def test_pipeline_cross_validates_with_missing_data():
    raw = data.make_synthetic_patients(n=500, seed=0)
    clean = preprocessing.clean_patient_table(raw, impute=False)
    encoded = preprocessing.encode_categoricals(clean)
    X = encoded.drop(columns=["readmitted"]).values
    y = encoded["readmitted"].values
    assert np.isnan(X).any()  # the pipeline must handle these per fold
    pipe = models.make_classifier_pipeline(kind="logistic")
    scores = models.cross_validate(pipe, X, y, cv=3, scoring="roc_auc")
    assert len(scores) == 3
    assert scores.mean() > 0.6


def test_classifier_trains_and_scores():
    df = data.load_diabetes_classification()
    X_train, X_test, y_train, y_test, _ = preprocessing.split_and_scale(
        df, target="diabetes"
    )
    model = models.train_classifier(X_train, y_train, kind="logistic")
    report = evaluation.evaluate_classifier(model, X_test, y_test)
    # A trained model should beat random on this separable target.
    assert report["roc_auc"] > 0.6
    assert 0.0 <= report["accuracy"] <= 1.0


def test_cross_validation_runs():
    df = data.load_breast_cancer_classification()
    X_train, X_test, y_train, y_test, _ = preprocessing.split_and_scale(
        df, target="malignant"
    )
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    model = models.train_classifier(X_train, y_train, kind="forest")
    scores = models.cross_validate(model, X, y, cv=3, scoring="roc_auc")
    assert len(scores) == 3
    assert scores.mean() > 0.8
