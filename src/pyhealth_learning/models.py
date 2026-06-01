"""
Model training helpers.

Thin wrappers around scikit-learn estimators with sensible defaults for small
clinical tabular datasets. Swap in any estimator you like — these just make the
example scripts short.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "train_classifier",
    "train_regressor",
    "cross_validate",
    "make_classifier_pipeline",
    "make_regressor_pipeline",
]


def _make_classifier(kind: str = "logistic", seed: int = 42):
    """Construct (but don't fit) a classifier of the requested kind."""
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if kind == "forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    raise ValueError(f"Unknown classifier kind: {kind!r}")


def train_classifier(X_train, y_train, kind: str = "logistic", seed: int = 42):
    """Train a binary classifier on already-preprocessed features.

    Parameters
    ----------
    kind:
        ``"logistic"`` for an interpretable linear baseline, or
        ``"forest"`` for a random forest that captures non-linearities.
    """
    return _make_classifier(kind, seed).fit(X_train, y_train)


def train_regressor(X_train, y_train):
    """Train a simple linear regression baseline."""
    return LinearRegression().fit(X_train, y_train)


def make_classifier_pipeline(kind: str = "logistic", seed: int = 42) -> Pipeline:
    """A leak-safe estimator: median-impute -> standardize -> classify.

    Bundling preprocessing *inside* the estimator means
    :func:`cross_validate` (or any scikit-learn CV) refits the imputer and
    scaler on each fold's training data only — no test information leaks in.
    Feed it the raw (un-imputed, un-scaled) feature matrix.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", _make_classifier(kind, seed)),
        ]
    )


def make_regressor_pipeline() -> Pipeline:
    """Leak-safe regression estimator: median-impute -> standardize -> linear fit."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def cross_validate(model, X, y, cv: int = 5, scoring: str = "roc_auc"):
    """Return cross-validated scores for an estimator.

    For a leak-free score pass a pipeline from :func:`make_classifier_pipeline`
    together with the *raw* features, so preprocessing is refit inside each fold.
    """
    return cross_val_score(model, X, y, cv=cv, scoring=scoring)
