from __future__ import annotations

import importlib
import importlib.util
from math import log
from types import ModuleType

import numpy as np
import pytest


def _metrics_module() -> ModuleType:
    package_spec = importlib.util.find_spec("project.models")
    assert package_spec is not None, "project.models is required for metric APIs"

    module_spec = importlib.util.find_spec("project.models.metrics")
    assert module_spec is not None, "project.models.metrics is required for metric APIs"
    return importlib.import_module("project.models.metrics")


class ReversedClassOrderEstimator:
    """Deterministic predict_proba estimator with positive class first."""

    classes_ = np.array([1, 0])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.array([[0.8, 0.2], [0.3, 0.7]])[: len(features)]

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise AssertionError("Metrics must use probabilities, not hard labels")


def test_metric_contract_marks_roc_auc_primary_and_declares_metric_directions():
    metrics_module = _metrics_module()
    assert metrics_module.METRIC_CONTRACT["primary_metric"] == "roc_auc"
    assert metrics_module.METRIC_CONTRACT["positive_label"] == 1
    assert metrics_module.METRIC_CONTRACT["directions"] == {
        "roc_auc": "maximize",
        "average_precision": "maximize",
        "log_loss": "minimize",
        "brier_score": "minimize",
    }


def test_evaluate_binary_metrics_reports_distinct_ranking_and_probability_guardrails():
    metrics_module = _metrics_module()
    report = metrics_module.evaluate_binary_metrics(
        y_true=[0, 0, 1, 1],
        positive_probabilities=[0.1, 0.4, 0.35, 0.8],
    )

    assert report["roc_auc"] == pytest.approx(0.75)
    assert report["average_precision"] == pytest.approx(5 / 6)
    assert report["log_loss"] == pytest.approx(
        -(log(0.9) + log(0.6) + log(0.35) + log(0.8)) / 4
    )
    assert report["brier_score"] == pytest.approx(0.158125)
    assert report["positive_support"] == 2
    assert report["negative_support"] == 2
    assert report["prevalence"] == pytest.approx(0.5)
    assert report["probability_checks"] == {
        "finite": True,
        "within_unit_interval": True,
        "minimum": 0.1,
        "maximum": 0.8,
    }


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([-0.1, 0.2, 0.8, 0.9], r"within \[0, 1\]"),
        ([0.1, 0.2, 1.1, 0.9], r"within \[0, 1\]"),
        ([0.1, np.nan, 0.8, 0.9], "finite"),
        ([0.1, np.inf, 0.8, 0.9], "finite"),
        ([0.1, -np.inf, 0.8, 0.9], "finite"),
    ],
)
def test_evaluate_binary_metrics_rejects_invalid_probabilities(
    probabilities: list[float], message: str
):
    metrics_module = _metrics_module()
    with pytest.raises(ValueError, match=message):
        metrics_module.evaluate_binary_metrics([0, 0, 1, 1], probabilities)


def test_evaluate_binary_metrics_requires_both_target_classes():
    metrics_module = _metrics_module()
    with pytest.raises(ValueError, match="both labels 0 and 1"):
        metrics_module.evaluate_binary_metrics([0, 0], [0.1, 0.2])


def test_evaluate_binary_metrics_rejects_unsupported_target_labels():
    metrics_module = _metrics_module()
    with pytest.raises(ValueError, match="binary labels 0 and 1"):
        metrics_module.evaluate_binary_metrics([0, 2], [0.1, 0.8])


@pytest.mark.parametrize(
    ("y_true", "probabilities", "message"),
    [
        ([0, 1], [0.2], "same length"),
        ([[0, 1], [1, 0]], [0.2, 0.8], "one-dimensional"),
        ([0, 1], [[0.2], [0.8]], "one-dimensional"),
    ],
)
def test_evaluate_binary_metrics_rejects_malformed_inputs(
    y_true: list[int] | list[list[int]],
    probabilities: list[float] | list[list[float]],
    message: str,
):
    metrics_module = _metrics_module()
    with pytest.raises(ValueError, match=message):
        metrics_module.evaluate_binary_metrics(y_true, probabilities)


def test_positive_class_scores_selects_class_one_by_identity_not_column_position():
    metrics_module = _metrics_module()
    estimator = ReversedClassOrderEstimator()

    scores = metrics_module.positive_class_scores(estimator, np.zeros((2, 1)))

    assert np.array_equal(scores, [0.8, 0.3])


@pytest.mark.parametrize(
    ("classes", "probabilities", "message"),
    [
        ([0, 2], [[0.2, 0.8], [0.6, 0.4]], r"classes_.*0 and 1"),
        ([1, 0], [[0.2], [0.8]], "shape"),
        ([1, 0], [[0.7, 0.7], [0.3, 0.7]], "sum to 1"),
        ([1, 0], [[1.2, -0.2], [0.4, 0.6]], r"within \[0, 1\]"),
        ([1, 0], [[np.nan, np.nan], [0.4, 0.6]], "finite"),
    ],
)
def test_positive_class_scores_rejects_invalid_estimator_class_support_or_probabilities(
    classes: list[int], probabilities: list[list[float]], message: str
):
    metrics_module = _metrics_module()
    class Estimator:
        classes_ = np.asarray(classes)

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.asarray(probabilities)

    with pytest.raises(ValueError, match=message):
        metrics_module.positive_class_scores(Estimator(), np.zeros((2, 1)))


def test_positive_class_scores_rejects_prediction_rows_that_do_not_match_input():
    metrics_module = _metrics_module()
    class Estimator:
        classes_ = np.array([0, 1])

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.array([[0.2, 0.8]])

    with pytest.raises(ValueError, match="row count.*input"):
        metrics_module.positive_class_scores(Estimator(), np.zeros((2, 1)))
