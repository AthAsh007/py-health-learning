"""
pyhealth_learning
=================

A small, dependency-light teaching package for learning healthcare
machine-learning workflows in Python. Everything here is built on top of
``numpy``, ``pandas``, ``scikit-learn`` and ``matplotlib`` so it runs out of
the box in Google Colab with no extra installs.

Typical workflow::

    from pyhealth_learning import data, preprocessing, models, evaluation

    df = data.load_diabetes_classification()
    X_train, X_test, y_train, y_test, _ = preprocessing.split_and_scale(
        df, target="diabetes"
    )
    model = models.train_classifier(X_train, y_train)
    report = evaluation.evaluate_classifier(model, X_test, y_test)
    print(report["summary"])
"""

from . import data, preprocessing, models, evaluation, utils

__all__ = ["data", "preprocessing", "models", "evaluation", "utils"]
__version__ = "0.1.0"
