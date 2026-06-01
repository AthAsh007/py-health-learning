"""
Example 05 — Regression: predict diabetes disease progression.

The earlier examples all classify. This one keeps the *continuous* target and
shows the regression path: train_regressor -> evaluate_regressor (MAE + R²).

    python examples/05_regression_diabetes_progression.py
"""

from pyhealth_learning import data, preprocessing, models, evaluation, utils


def main() -> None:
    utils.set_seed(42)

    utils.section("Load diabetes regression dataset")
    df = data.load_diabetes_regression()
    print(f"shape: {df.shape}")
    print(df[["age", "bmi", "bp", "progression"]].head())

    # Split + scale on the continuous target (stratify off for regression).
    X_train, X_test, y_train, y_test, meta = preprocessing.split_and_scale(
        df, target="progression", stratify=False
    )
    print(f"\ntrain={X_train.shape}  test={X_test.shape}")

    utils.section("Train linear regression baseline")
    model = models.train_regressor(X_train, y_train)
    report = evaluation.evaluate_regressor(model, X_test, y_test)
    print(report["summary"])

    utils.section("Largest-magnitude coefficients")
    coefs = sorted(
        zip(meta["features"], model.coef_), key=lambda kv: abs(kv[1]), reverse=True
    )
    for name, value in coefs[:5]:
        print(f"  {name:>10}: {value:+.2f}")


if __name__ == "__main__":
    main()
