"""
Example 09 — Data leakage: why preprocessing must be fit on the training fold.

Imputation and scaling learn from the data (a column median, a column mean/std).
If you fit them on the *whole* dataset before cross-validating, each held-out
fold has already influenced its own preprocessing — that's **data leakage**, and
it makes your score optimistic. Doing the same steps *inside* a Pipeline refits
them per fold and removes the leak.

    python examples/09_data_leakage_demo.py
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from pyhealth_learning import data, preprocessing, models, utils

# Same classifier config on both sides, so the ONLY difference is where the
# imputer/scaler are fit (whole dataset vs each training fold).
CLASSIFIER = LogisticRegression(max_iter=1000, random_state=42)


def main() -> None:
    utils.set_seed(42)

    # Cleaned but NOT imputed: keep the numeric gaps so imputation has a job.
    raw = data.make_synthetic_patients(n=800)
    clean = preprocessing.clean_patient_table(raw, impute=False)
    encoded = preprocessing.encode_categoricals(clean)
    X = encoded.drop(columns=["readmitted"]).values
    y = encoded["readmitted"].values
    print(f"feature matrix: {X.shape}  (missing cells: {int(np.isnan(X).sum())})")

    utils.section("WRONG — fit imputer + scaler on ALL data, then cross-validate")
    # The held-out fold's rows contributed to the median and the mean/std.
    X_leaky = SimpleImputer(strategy="median").fit_transform(X)
    X_leaky = StandardScaler().fit_transform(X_leaky)
    leaky = cross_val_score(CLASSIFIER, X_leaky, y, cv=5, scoring="roc_auc")
    print(f"AUC: {leaky.mean():.4f} +/- {leaky.std():.4f}")

    utils.section("RIGHT — preprocessing inside a Pipeline, refit each fold")
    pipe = models.make_classifier_pipeline(kind="logistic")
    clean_cv = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"AUC: {clean_cv.mean():.4f} +/- {clean_cv.std():.4f}")

    utils.section("Takeaway")
    gap = leaky.mean() - clean_cv.mean()
    print(f"Leaky score is {gap:+.4f} vs leak-free.")
    print(
        "The gap is small for plain impute+scale, but grows fast once you add\n"
        "feature selection, target encoding or oversampling on the full data.\n"
        "Rule of thumb: anything that LEARNS from the data belongs in the\n"
        "Pipeline, so cross-validation refits it on each fold's training rows."
    )


if __name__ == "__main__":
    main()
