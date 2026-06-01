"""
Example 03 — Evaluate a model and save diagnostic plots.

    python examples/03_evaluate_and_plot.py

Produces roc_curve.png, confusion_matrix.png and feature_importance.png in the
current directory.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so it works without a display
import matplotlib.pyplot as plt

from pyhealth_learning import data, preprocessing, models, evaluation, utils


def main() -> None:
    utils.set_seed(42)

    df = data.load_breast_cancer_classification()
    X_train, X_test, y_train, y_test, meta = preprocessing.split_and_scale(
        df, target="malignant"
    )
    model = models.train_classifier(X_train, y_train, kind="forest")
    report = evaluation.evaluate_classifier(model, X_test, y_test)
    print(report["summary"])

    evaluation.plot_roc_curve(y_test, report["y_proba"])
    plt.savefig("roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    evaluation.plot_confusion_matrix(
        report["confusion_matrix"], labels=("benign", "malignant")
    )
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    evaluation.plot_feature_importance(model, meta["features"], top_n=10)
    plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("Saved: roc_curve.png, confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()
