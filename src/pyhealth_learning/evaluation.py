"""
Evaluation and plotting helpers.

Functions return plain dictionaries so they're easy to print, log or assert on
in tests, and optionally produce matplotlib figures for notebooks.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "evaluate_classifier",
    "evaluate_regressor",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "plot_feature_importance",
]


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Compute the standard binary-classification metrics."""
    y_pred = model.predict(X_test)
    proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, y_pred)

    summary = f"accuracy={acc:.3f}  roc_auc={auc:.3f}"
    return {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "report": classification_report(y_test, y_pred, digits=3),
        "y_proba": proba,
        "summary": summary,
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    """Compute regression metrics (MAE and R²)."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mae": mae, "r2": r2, "summary": f"mae={mae:.2f}  r2={r2:.3f}"}


def plot_roc_curve(y_test, y_proba, ax=None):
    """Plot a ROC curve. Returns the matplotlib axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return ax


def plot_confusion_matrix(cm, labels=("negative", "positive"), ax=None):
    """Plot a confusion matrix heatmap. Returns the matplotlib axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return ax


def plot_feature_importance(model, feature_names, top_n: int = 10, ax=None):
    """Bar plot of feature importances or absolute coefficients."""
    import matplotlib.pyplot as plt

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model exposes neither feature_importances_ nor coef_.")

    order = np.argsort(importance)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    values = importance[order]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(names)} features")
    return ax
