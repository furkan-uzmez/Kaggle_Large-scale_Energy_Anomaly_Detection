"""Explicit candidate ranking and probability score boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype
from sklearn.metrics import average_precision_score, roc_auc_score

from project.experiments.candidate_contracts import CandidateScore, ScoreKind
from project.models.metrics import evaluate_binary_metrics


def _binary_classes(estimator: Any) -> np.ndarray:
    classes = np.asarray(getattr(estimator, 'classes_', None))
    if (classes.shape != (2,) or classes.dtype.kind not in 'iu'
            or set(classes.tolist()) != {0, 1}):
        raise ValueError('estimator classes_ must identify exactly integer binary classes 0 and 1')
    return classes


def _validate_features(estimator: Any, X: Any) -> None:
    if getattr(X, 'ndim', None) != 2 or len(X) == 0:
        raise ValueError('predictors must have a nonempty two-dimensional row shape')
    expected = getattr(estimator, 'feature_names_in_', None)
    if expected is not None and (not isinstance(X, pd.DataFrame)
                                 or tuple(X.columns) != tuple(expected)):
        raise ValueError('predictor schema must match exact fitted column order')
    width = getattr(estimator, 'n_features_in_', None)
    if width is not None and X.shape[1] != width:
        raise ValueError('predictor schema width must match fitted features')


def _numeric_output(output: Any, X: Any) -> np.ndarray:
    if (isinstance(output, (pd.Series, pd.DataFrame))
            and (not isinstance(X, pd.DataFrame) or not output.index.equals(X.index))):
        raise ValueError('score row index alignment must exactly match predictors')
    values = np.asarray(output)
    if values.dtype.kind not in 'iuf':
        raise TypeError('score output must be real numeric, not boolean, complex or text')
    if not np.isfinite(values).all():
        raise ValueError('score output must contain only finite numeric values')
    return values


def extract_candidate_score(estimator: Any, X: Any, score_kind: ScoreKind) -> CandidateScore:
    """Extract unmodified scores oriented toward integer positive label 1.

    Binary sklearn margins point toward ``classes_[1]``; reversed class order
    therefore reverses their sign. Probability matrices follow ``classes_``
    order; a pandas output must also carry that exact class-column order and
    the input row index. No ranking scores are calibrated or clipped here.
    """
    if score_kind not in ('positive_probability', 'decision_margin'):
        raise ValueError('unsupported explicit score kind')
    classes = _binary_classes(estimator)
    _validate_features(estimator, X)
    method_name = 'predict_proba' if score_kind == 'positive_probability' else 'decision_function'
    method = getattr(estimator, method_name, None)
    if not callable(method):
        raise TypeError(f'estimator must provide {method_name}')
    output = method(X)
    values = _numeric_output(output, X)
    if score_kind == 'positive_probability':
        if values.shape != (len(X), 2):
            raise ValueError('probability output shape must match input row count and two classes')
        if isinstance(output, pd.DataFrame):
            columns = np.asarray(output.columns)
            if columns.dtype.kind not in 'iu' or not np.array_equal(columns, classes):
                raise ValueError('probability column identities must match estimator class order')
        if ((values < 0) | (values > 1)).any():
            raise ValueError('probability output must be within [0, 1]')
        if not np.allclose(values.sum(axis=1), 1., rtol=0., atol=1e-6):
            raise ValueError('probability rows must sum to 1')
        scores = values[:, int(np.flatnonzero(classes == 1)[0])]
    else:
        if values.shape != (len(X),):
            raise ValueError('binary decision margin must be one-dimensional with exact row count')
        # Convert before negation: unsigned numeric margins must not wrap around.
        scores = values.astype(np.float64) * (1. if classes[1] == 1 else -1.)
    return CandidateScore(score_kind, scores)


def _target(y: Any, length: int) -> np.ndarray:
    dtype = getattr(y, 'dtype', np.asarray(y).dtype)
    values = np.asarray(y)
    if values.ndim != 1 or len(values) == 0 or len(values) != length:
        raise ValueError('target must be nonempty one-dimensional and match score length')
    if not is_integer_dtype(dtype) or is_bool_dtype(dtype) or pd.isna(values).any():
        raise ValueError('target must contain nonnull integer binary labels')
    if set(values.tolist()) != {0, 1}:
        raise ValueError('target must contain both integer binary classes 0 and 1')
    return values.astype(np.int64)


def evaluate_candidate_metrics(y: Any, score: CandidateScore) -> dict[str, Any]:
    """Evaluate ranking and score-kind-appropriate probability guardrails."""
    if not isinstance(score, CandidateScore):
        raise TypeError('typed CandidateScore is required')
    # Revalidate even frozen bindings: an externally corrupted object is not trust.
    validated = CandidateScore(score.kind, score.values, score.positive_label)
    labels = _target(y, len(validated.values))
    if validated.kind == 'positive_probability':
        # Preserve the stable metric module's float64 sklearn endpoint precision.
        return {**evaluate_binary_metrics(labels, validated.values), 'guardrail_status': 'computed'}
    positive_support = int(np.count_nonzero(labels == 1))
    return {
        'roc_auc': float(roc_auc_score(labels, validated.values)),
        'average_precision': float(average_precision_score(labels, validated.values)),
        'log_loss': None,
        'brier_score': None,
        'guardrail_status': 'requires_calibrated_probability',
        'positive_support': positive_support,
        'negative_support': len(labels) - positive_support,
        'prevalence': positive_support / len(labels),
    }
