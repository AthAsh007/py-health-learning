"""Smoke tests that exercise the full learning pipeline."""

import numpy as np

from pyhealth_learning import data, preprocessing, models, evaluation


def test_synthetic_patients_shape_and_target():
    df = data.make_synthetic_patients(n=200, seed=0)
    assert len(df) == 200
    assert df["readmitted"].isin([0, 1]).all()


def test_clean_removes_impossible_values():
    raw = data.make_synthetic_patients(n=200, seed=0)
    clean = preprocessing.clean_patient_table(raw)
    assert clean.isna().sum().sum() == 0
    assert clean["systolic_bp"].max() <= 250
    assert clean["bmi"].min() >= 10


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
