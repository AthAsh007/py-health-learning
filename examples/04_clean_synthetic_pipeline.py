"""
Example 04 — Clean a messy synthetic patient table, then predict readmission.

Shows the full tabular workflow: cleaning impossible values + missingness,
one-hot encoding, training and cross-validation.

    python examples/04_clean_synthetic_pipeline.py
"""

import pandas as pd

from pyhealth_learning import data, preprocessing, models, evaluation, utils


def show_before_after(raw, clean, rows, cols):
    """Print the SAME rows before vs after cleaning, value by value.

    Each cell reads ``old -> new`` so you can see exactly what changed: a blank
    (``nan``) becoming a real number means it was *filled*, not deleted; ``400``
    becoming a normal reading means an impossible value was corrected.
    """
    table = pd.DataFrame({"patient_id": raw.loc[rows, "patient_id"].values})
    for col in cols:
        before = raw.loc[rows, col].values
        after = clean.loc[rows, col].values
        table[col] = [f"{b}  ->  {a}" for b, a in zip(before, after)]
    print(table.to_string(index=False))


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

    utils.section("What actually changed (same rows, before -> after)")
    cols = ["bmi", "systolic_bp", "glucose", "smoker"]
    # The generator injects an impossible BP at row 3 and an impossible BMI at
    # row 7; add the first couple of rows that had a missing bmi so you can see
    # a blank get filled too.
    outliers = [3, 7]
    missing = [r for r in raw.index[raw["bmi"].isna()].tolist() if r not in outliers]
    show_before_after(raw, clean, outliers + missing[:2], cols)
    print(
        "\nNote: 'nan -> <number>' means a blank was FILLED with the column "
        "median (not deleted); '400.0 -> <number>' means an impossible reading "
        "was corrected."
    )

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
