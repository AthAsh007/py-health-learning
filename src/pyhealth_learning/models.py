"""
Model training helpers.

Thin wrappers around scikit-learn estimators with sensible defaults for small
clinical tabular datasets. Swap in any estimator you like — these just make the
example scripts short.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score

__all__ = ["train_classifier", "train_regressor", "cross_validate"]


def train_classifier(X_train, y_train, kind: str = "logistic", seed: int = 42):
    """Train a binary classifier.

    Parameters
    ----------
    kind:
        ``"logistic"`` for an interpretable linear baseline, or
        ``"forest"`` for a random forest that captures non-linearities.
    """
    if kind == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif kind == "forest":
        model = RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classifier kind: {kind!r}")

    model.fit(X_train, y_train)
    return model


def train_regressor(X_train, y_train):
    """Train a simple linear regression baseline."""
    return LinearRegression().fit(X_train, y_train)


def cross_validate(model, X, y, cv: int = 5, scoring: str = "roc_auc"):
    """Return cross-validated scores for a model on the full dataset."""
    return cross_val_score(model, X, y, cv=cv, scoring=scoring)
