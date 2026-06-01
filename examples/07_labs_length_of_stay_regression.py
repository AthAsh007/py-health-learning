"""
Example 07 — Predict hospital length-of-stay from a lab panel (regression).

Uses the synthetic lab dataset, whose continuous ``length_of_stay`` target is a
noisy function of age and lab values. Mirrors example 05 but on tabular data
with a categorical column (sex) that gets one-hot encoded.

    python examples/07_labs_length_of_stay_regression.py
"""

from pyhealth_learning import data, preprocessing, models, evaluation, utils


def main() -> None:
    utils.set_seed(42)

    utils.section("Synthetic lab panel")
    df = data.make_synthetic_labs(n=600)
    print(f"shape: {df.shape}")
    print(df.head())
    print(
        "\nlength_of_stay (days):",
        f"mean={df['length_of_stay'].mean():.1f}",
        f"max={df['length_of_stay'].max():.1f}",
    )

    utils.section("Train length-of-stay regressor")
    X_train, X_test, y_train, y_test, meta = preprocessing.split_and_scale(
        df, target="length_of_stay", stratify=False
    )
    model = models.train_regressor(X_train, y_train)
    print(evaluation.evaluate_regressor(model, X_test, y_test)["summary"])

    utils.section("Most influential labs")
    coefs = sorted(
        zip(meta["features"], model.coef_), key=lambda kv: abs(kv[1]), reverse=True
    )
    for name, value in coefs[:5]:
        print(f"  {name:>12}: {value:+.2f} days")


if __name__ == "__main__":
    main()
