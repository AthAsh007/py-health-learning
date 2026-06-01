"""
Example 10 — Compare models with leak-free cross-validation.

Builds full scikit-learn Pipelines (impute -> scale -> model) and cross-
validates them on the raw feature matrix, so every fold refits its own
preprocessing. This is the honest way to compare candidate models before
committing to one.

    python examples/10_cross_validated_pipeline.py
"""

from pyhealth_learning import data, preprocessing, models, utils


def cv_summary(name, pipe, X, y):
    scores = models.cross_validate(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"  {name:<22} AUC = {scores.mean():.3f} +/- {scores.std():.3f}")


def main() -> None:
    utils.set_seed(42)

    # Raw (un-imputed) features so the Pipeline does all preprocessing per fold.
    raw = data.make_synthetic_patients(n=800)
    clean = preprocessing.clean_patient_table(raw, impute=False)
    encoded = preprocessing.encode_categoricals(clean)
    X = encoded.drop(columns=["readmitted"]).values
    y = encoded["readmitted"].values

    utils.section("Readmission — logistic vs random forest (5-fold ROC AUC)")
    cv_summary("logistic", models.make_classifier_pipeline("logistic"), X, y)
    cv_summary("random forest", models.make_classifier_pipeline("forest"), X, y)

    # The same recipe transfers to any tabular dataset in the package.
    utils.section("Breast cancer — logistic vs random forest")
    bc = data.load_breast_cancer_classification()
    enc = preprocessing.encode_categoricals(bc)
    Xb = enc.drop(columns=["malignant"]).values
    yb = enc["malignant"].values
    cv_summary("logistic", models.make_classifier_pipeline("logistic"), Xb, yb)
    cv_summary("random forest", models.make_classifier_pipeline("forest"), Xb, yb)


if __name__ == "__main__":
    main()
