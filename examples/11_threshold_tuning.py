"""
Example 11 — Tune the decision threshold for a screening use-case.

A classifier outputs a probability; turning it into a yes/no needs a threshold.
The default 0.5 is rarely what you want clinically — for screening you'd rather
catch almost every positive (high recall/sensitivity) and accept more false
alarms. This picks the lowest threshold that hits a target recall and shows the
precision/recall trade-off it buys.

    python examples/11_threshold_tuning.py
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

from pyhealth_learning import data, preprocessing, models, evaluation, utils

TARGET_RECALL = 0.95


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return pred, precision, recall, (tn, fp, fn, tp)


def main() -> None:
    utils.set_seed(42)

    # Screening framing: "malignant" is the positive we must not miss.
    df = data.load_breast_cancer_classification()
    X_train, X_test, y_train, y_test, _ = preprocessing.split_and_scale(
        df, target="malignant"
    )
    model = models.train_classifier(X_train, y_train, kind="forest")
    report = evaluation.evaluate_classifier(model, X_test, y_test)
    proba = report["y_proba"]
    print(report["summary"])

    utils.section("Default threshold = 0.50")
    _, prec, rec, (tn, fp, fn, tp) = metrics_at(y_test, proba, 0.50)
    print(f"  precision={prec:.3f}  recall={rec:.3f}  (missed positives: {fn})")

    utils.section(f"Tightest threshold that still keeps recall >= {TARGET_RECALL}")
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    # recall decreases as the threshold rises, so the indices meeting the target
    # form a prefix; the LAST of them is the highest threshold (best precision)
    # that still satisfies the recall floor. (One fewer threshold than points.)
    ok = np.where(recall[:-1] >= TARGET_RECALL)[0]
    chosen = thresholds[ok[-1]] if len(ok) else 0.0
    _, prec, rec, (tn, fp, fn, tp) = metrics_at(y_test, proba, chosen)
    print(f"  threshold={chosen:.3f}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}  (missed positives: {fn})")
    print(f"  confusion: tn={tn} fp={fp} fn={fn} tp={tp}")

    utils.section("Trade-off")
    print(
        "Lowering the threshold catches more positives (recall up, fewer missed)\n"
        "at the cost of more false alarms (precision down). Pick the point that\n"
        "matches the clinical cost of a miss vs a false positive — not 0.5 by default."
    )


if __name__ == "__main__":
    main()
