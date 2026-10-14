"""Synthetic score math, identity, alignment and fitted-estimator contracts."""

from math import log

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from project.experiments.candidate_contracts import CandidateScore
from project.models.candidate_scores import (
    evaluate_candidate_metrics,
    extract_candidate_score,
)

PROBABILITY = 'positive_probability'
MARGIN = 'decision_margin'


class LiteralEstimator:
    """External estimator boundary supplying literal outputs, never scoring logic."""

    def __init__(self, output, classes=(0, 1)):
        self.output = output
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return self.output

    def decision_function(self, X):
        return self.output


def features():
    return pd.DataFrame({'reading': [-2., 2.]}, index=pd.Index([7, 3], name='row'))


def test_literal_margin_ranking_has_null_probability_guardrails():
    source = np.array([-2., 2.])
    score = extract_candidate_score(LiteralEstimator(source), features(), MARGIN)
    report = evaluate_candidate_metrics(np.array([0, 1]), score)
    np.testing.assert_array_equal(score.values, [-2., 2.])
    assert score.kind == MARGIN and score.positive_label == 1
    assert report['roc_auc'] == report['average_precision'] == 1.
    assert report['log_loss'] is None and report['brier_score'] is None
    assert report['guardrail_status'] == 'requires_calibrated_probability'
    assert not score.values.flags.writeable
    source[:] = 999.
    np.testing.assert_array_equal(score.values, [-2., 2.])


def test_reversed_class_probability_columns_and_margin_orientation():
    matrix = np.array([[.2, .8], [.9, .1]])
    score = extract_candidate_score(LiteralEstimator(matrix, (1, 0)), features(), PROBABILITY)
    np.testing.assert_array_equal(score.values, [.2, .9])
    reversed_margin = extract_candidate_score(
        LiteralEstimator(np.array([2., -2.]), (1, 0)), features(), MARGIN)
    np.testing.assert_array_equal(reversed_margin.values, [-2., 2.])


def test_dataframe_output_preserves_index_and_class_column_identity():
    frame = features()
    output = pd.DataFrame([[.2, .8], [.9, .1]], index=frame.index, columns=[1, 0])
    original = output.copy(deep=True)
    score = extract_candidate_score(LiteralEstimator(output, (1, 0)), frame, PROBABILITY)
    np.testing.assert_array_equal(score.values, [.2, .9])
    pd.testing.assert_frame_equal(output, original)
    with pytest.raises(ValueError, match='index|alignment'):
        extract_candidate_score(LiteralEstimator(output.iloc[::-1], (1, 0)), frame, PROBABILITY)
    with pytest.raises(ValueError, match='column|class'):
        extract_candidate_score(LiteralEstimator(output.rename(columns={1: 0, 0: 1}), (1, 0)),
                                frame, PROBABILITY)


@pytest.mark.parametrize('classes', [(0,), (0, 2), (0, 0), (False, True), (0., 1.), ('0', '1')])
def test_invalid_class_identity_is_rejected(classes):
    with pytest.raises(ValueError, match='class'):
        extract_candidate_score(LiteralEstimator(np.array([[.8, .2], [.1, .9]]), classes),
                                features(), PROBABILITY)


@pytest.mark.parametrize(('kind', 'output'), [
    (PROBABILITY, np.array([[.8, .2]])),
    (PROBABILITY, np.array([.2, .9])),
    (PROBABILITY, np.array([[.8, np.nan], [.1, .9]])),
    (PROBABILITY, np.array([[1.1, -.1], [.1, .9]])),
    (PROBABILITY, np.array([[.8, .8], [.1, .9]])),
    (PROBABILITY, np.array([[True, False], [False, True]])),
    (PROBABILITY, np.array([[.8, .2], [.1, .9]], dtype=complex)),
    (PROBABILITY, np.array([['.8', '.2'], ['.1', '.9']])),
    (MARGIN, np.array([np.inf, 2.])),
    (MARGIN, np.array([np.nan, 2.])),
    (MARGIN, np.array([True, False])),
    (MARGIN, np.array([-2., 2.], dtype=complex)),
    (MARGIN, np.array([-2.])),
    (MARGIN, np.array([[-2.], [2.]])),
])
def test_invalid_output_is_rejected_without_coercion_or_clipping(kind, output):
    with pytest.raises((ValueError, TypeError)):
        extract_candidate_score(LiteralEstimator(output), features(), kind)


def test_invalid_kind_and_missing_api_are_rejected():
    estimator = LiteralEstimator(np.array([-2., 2.]))
    with pytest.raises(ValueError, match='kind'):
        extract_candidate_score(estimator, features(), 'auto')
    with pytest.raises((ValueError, TypeError)):
        extract_candidate_score(object(), features(), PROBABILITY)


@pytest.mark.parametrize('kind', [PROBABILITY, MARGIN])
def test_real_fitted_pipelines_extract_scores_and_enforce_fitted_order(kind):
    frame = pd.DataFrame({'reading': [-2., -1., 1., 2.], 'hour': [0., 1., 0., 1.]})
    labels = np.array([0, 0, 1, 1])
    original = frame.copy(deep=True)
    model = LogisticRegression(random_state=42) if kind == PROBABILITY else SVC(kernel='linear')
    estimator = make_pipeline(StandardScaler(), model).fit(frame, labels)
    score = extract_candidate_score(estimator, frame, kind)
    expected = estimator.predict_proba(frame)[:, 1] if kind == PROBABILITY else estimator.decision_function(frame)
    np.testing.assert_array_equal(score.values, expected)
    assert score.values.dtype == np.dtype('float64')
    assert evaluate_candidate_metrics(labels, score)['roc_auc'] == 1.
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match='schema|order'):
        extract_candidate_score(estimator, frame[['hour', 'reading']], kind)


def test_probability_metrics_match_hand_math_and_do_not_mutate_inputs():
    labels = np.array([0, 0, 1, 1])
    score = CandidateScore(PROBABILITY, np.array([.1, .4, .35, .8]))
    report = evaluate_candidate_metrics(labels, score)
    assert report['roc_auc'] == pytest.approx(.75)
    assert report['average_precision'] == pytest.approx(5 / 6)
    assert report['log_loss'] == pytest.approx(-(log(.9) + log(.6) + log(.35) + log(.8)) / 4)
    assert report['brier_score'] == pytest.approx(.158125)
    assert report['positive_support'] == report['negative_support'] == 2
    assert report['prevalence'] == .5
    assert report['guardrail_status'] == 'computed'
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    np.testing.assert_array_equal(score.values, [.1, .4, .35, .8])


def test_probability_endpoints_use_stable_float64_logloss_precision():
    report = evaluate_candidate_metrics(np.array([0, 1]), CandidateScore(PROBABILITY, np.array([1., 0.])))
    assert report['brier_score'] == 1.
    assert report['log_loss'] == pytest.approx(-log(np.finfo(np.float64).eps))


@pytest.mark.parametrize('labels', [np.array([0, 0]), np.array([1, 1]), np.array([0, 2]),
                                   np.array([False, True]), np.array([0., 1.]),
                                   np.array([0]), np.array([[0, 1]]), np.array([0, np.nan])])
def test_metrics_require_aligned_two_class_integer_target(labels):
    with pytest.raises((ValueError, TypeError)):
        evaluate_candidate_metrics(labels, CandidateScore(MARGIN, np.array([-2., 2.])))


def test_nullable_integer_labels_are_supported_but_missing_labels_are_not():
    labels = pd.Series([0, 1], dtype='Int64')
    score = CandidateScore(MARGIN, np.array([-2., 2.]))
    assert evaluate_candidate_metrics(labels, score)['roc_auc'] == 1.
    labels.iloc[0] = pd.NA
    with pytest.raises(ValueError):
        evaluate_candidate_metrics(labels, score)


def test_metrics_require_typed_score_and_revalidate_tampered_binding():
    with pytest.raises(TypeError):
        evaluate_candidate_metrics(np.array([0, 1]), np.array([-2., 2.]))
    score = CandidateScore(MARGIN, np.array([-2., 2.]))
    object.__setattr__(score, 'values', np.array([np.nan, 2.]))
    with pytest.raises(ValueError):
        evaluate_candidate_metrics(np.array([0, 1]), score)
