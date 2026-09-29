"""Binary ranking and probability-quality metrics for the anomaly target."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

METRIC_CONTRACT: dict[str, Any] = {
    "target_type": "binary",
    "positive_label": 1,
    "primary_metric": "roc_auc",
    "directions": {
        "roc_auc": "maximize",
        "average_precision": "maximize",
        "log_loss": "minimize",
        "brier_score": "minimize",
    },
}


def positive_class_scores(
    estimator: Any,
    features: Any,
    positive_label: int = 1,
) -> np.ndarray:
    """Extract validated positive-class probabilities using ``classes_`` identity."""

    classes = np.asarray(getattr(estimator, "classes_", None))
    if (
        classes.ndim != 1
        or classes.size != 2
        or np.count_nonzero(classes == 0) != 1
        or np.count_nonzero(classes == 1) != 1
    ):
        raise ValueError("estimator classes_ must contain both binary classes 0 and 1")
    if positive_label not in (0, 1):
        raise ValueError("positive_label must be one of the binary classes 0 and 1")

    positive_indices = np.flatnonzero(classes == positive_label)
    if positive_indices.size != 1:
        raise ValueError("estimator classes_ does not contain the requested positive label")

    try:
        expected_rows = len(features)
    except TypeError as error:
        raise TypeError("input features must have a row dimension") from error

    predict_proba = getattr(estimator, "predict_proba", None)
    if not callable(predict_proba):
        raise TypeError("estimator must provide predict_proba")
    try:
        probabilities = np.asarray(predict_proba(features), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("predict_proba output must be a numeric probability matrix") from error

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("predict_proba output shape must be (n_rows, 2)")
    if probabilities.shape[0] != expected_rows:
        raise ValueError("predict_proba row count must match input feature row count")
    _validate_probability_matrix(probabilities)
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("predict_proba rows must sum to 1")

    return probabilities[:, positive_indices[0]].copy()


def evaluate_binary_metrics(
    y_true: Any,
    positive_probabilities: Any,
) -> dict[str, Any]:
    """Validate binary labels/probabilities and report ranking and proper scores."""

    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    try:
        probabilities = np.asarray(positive_probabilities, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("positive_probabilities must contain numeric values") from error
    if probabilities.ndim != 1:
        raise ValueError("positive_probabilities must be one-dimensional")
    if labels.size != probabilities.size:
        raise ValueError("y_true and probabilities must have the same length")
    if labels.size == 0:
        raise ValueError("y_true and probabilities must not be empty")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("y_true must contain only binary labels 0 and 1")

    _validate_probability_vector(probabilities)
    positive_support = int(np.count_nonzero(labels == 1))
    negative_support = int(np.count_nonzero(labels == 0))
    if positive_support == 0 or negative_support == 0:
        raise ValueError("y_true must contain both labels 0 and 1")

    prevalence = positive_support / labels.size
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "positive_support": positive_support,
        "negative_support": negative_support,
        "prevalence": float(prevalence),
        "probability_checks": {
            "finite": True,
            "within_unit_interval": True,
            "minimum": float(probabilities.min()),
            "maximum": float(probabilities.max()),
        },
    }


def _validate_probability_vector(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be within [0, 1]")


def _validate_probability_matrix(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("predict_proba values must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("predict_proba values must be within [0, 1]")
