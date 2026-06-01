"""
Example 04 — Clean a messy synthetic patient table, then predict readmission.

Shows the full tabular workflow: cleaning impossible values + missingness,
one-hot encoding, training and cross-validation.

    python examples/04_clean_synthetic_pipeline.py
"""

from pyhealth_learning import data, preprocessing, models, evaluation, utils


def main() -> None:
    utils.set_seed(42)

    utils.section("Raw synthetic data")
    raw = data.make_synthetic_patients(n=800)
    print("missing before cleaning:\n", raw.isna().sum())
    print("max systolic_bp (note the outlier):", raw["systolic_bp"].max())

    utils.section("After cleaning")
    clean = preprocessing.clean_patient_table(raw)
    print("missing after cleaning:\n", clean.isna().sum())
    print("max systolic_bp:", clean["systolic_bp"].max())

    utils.section("Train readmission model")
    X_train, X_test, y_train, y_test, meta = preprocessing.split_and_scale(
        clean, target="readmitted"
    )
    model = models.train_classifier(X_train, y_train, kind="forest")
    print(evaluation.evaluate_classifier(model, X_test, y_test)["summary"])

    utils.section("5-fold cross-validated ROC AUC")
    import numpy as np

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    scores = models.cross_validate(model, X_all, y_all, cv=5, scoring="roc_auc")
    print(f"AUC: {scores.mean():.3f} +/- {scores.std():.3f}")


if __name__ == "__main__":
    main()
