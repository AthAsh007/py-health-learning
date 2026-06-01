"""
Example 02 — Train a diabetes-risk classifier end to end.

    python examples/02_train_diabetes_classifier.py
"""

from pyhealth_learning import data, preprocessing, models, evaluation, utils


def main() -> None:
    utils.set_seed(42)

    utils.section("Load + split")
    df = data.load_diabetes_classification()
    X_train, X_test, y_train, y_test, meta = preprocessing.split_and_scale(
        df, target="diabetes"
    )
    print(f"train={X_train.shape}  test={X_test.shape}")

    utils.section("Train logistic regression baseline")
    logreg = models.train_classifier(X_train, y_train, kind="logistic")
    print(evaluation.evaluate_classifier(logreg, X_test, y_test)["summary"])

    utils.section("Train random forest")
    forest = models.train_classifier(X_train, y_train, kind="forest")
    rf_report = evaluation.evaluate_classifier(forest, X_test, y_test)
    print(rf_report["summary"])
    print("\nclassification report:\n", rf_report["report"])


if __name__ == "__main__":
    main()
