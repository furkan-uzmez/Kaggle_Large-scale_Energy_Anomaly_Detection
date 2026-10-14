"""Synthetic contracts for the fold-trained meter distribution candidate stage.

Hand-computed expectations are literals, never a reimplementation of the
transformer used as its own oracle. Fitting consumes training meter values only:
no labels, groups, history, or held-out rows may alter fitted state.
"""

import importlib
import importlib.util
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from candidate_fixtures import CORE_COLUMNS, core_frame
from project.features.registry import (
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)
from project.features.transformers import make_imputation_preprocessor

METHODS = ("bin", "quantile")
OUTPUT_NAMES = {"bin": "meter_reading_bin", "quantile": "meter_reading_quantile"}


def api():
    name = "project.features.distributions"
    assert importlib.util.find_spec(name), "fold-trained distribution API is missing"
    return importlib.import_module(name)


def make(method, **parameters):
    return api().MeterDistributionTransformer(method, **parameters)


def frame_with(meter, index=None):
    """Return a core-schema frame whose meter readings are the supplied values."""
    values = np.asarray(list(meter), dtype=float)
    template = core_frame()
    repeats = max(1, -(-len(values) // len(template)))
    output = pd.concat([template] * repeats, ignore_index=True).iloc[: len(values)].copy()
    output["meter_reading"] = values
    if index is not None:
        output.index = pd.Index(index)
    return output


def fitted_state(transformer):
    """Normalize every fitted attribute so snapshots compare by value."""
    state = {}
    for name, value in sorted(vars(transformer).items()):
        if not name.endswith("_") or name.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            state[name] = value.tolist()
        elif isinstance(value, FeatureRegistry):
            state[name] = value.names
        else:
            state[name] = value
    return state


def feature_of(transformer, meter):
    """Return the appended distribution feature for the supplied meter values."""
    output = transformer.transform(frame_with(meter))
    return output[OUTPUT_NAMES[transformer.method]].to_numpy(dtype=float)


# --- Hand-computed mathematics -------------------------------------------------


def test_two_bin_fit_has_one_interior_edge_at_the_training_median():
    transformer = make("bin", n_bins=2).fit(frame_with([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(transformer.bin_interior_edges_, [1.5])
    assert transformer.n_bin_edges_ == 1
    np.testing.assert_allclose(
        feature_of(transformer, [-1.0, 1.5, 9.0, np.nan]),
        [0.0, 1.0, 1.0, np.nan],
    )


def test_empirical_midrank_cdf_uses_half_the_tied_mass_and_saturates_tails():
    transformer = make("quantile", n_quantiles=8).fit(frame_with([0.0, 0.0, 2.0, 2.0]))
    np.testing.assert_allclose(
        feature_of(transformer, [0.0, 1.0, 2.0]), [0.25, 0.5, 0.75]
    )
    np.testing.assert_allclose(
        feature_of(transformer, [-100.0, 100.0, np.nan]), [0.25, 0.75, np.nan]
    )
    np.testing.assert_allclose(
        [transformer.quantile_cdf_[0], transformer.quantile_cdf_[-1]], [0.25, 0.75]
    )


def test_bin_duplicate_interior_edges_collapse_and_record_the_actual_count():
    transformer = make("bin", n_bins=4).fit(frame_with([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(transformer.bin_interior_edges_, [0.0, 0.25])
    assert transformer.n_bin_edges_ == 2
    np.testing.assert_allclose(
        feature_of(transformer, [-5.0, 0.0, 0.1, 0.25, 5.0]), [0.0, 1.0, 1.0, 2.0, 2.0]
    )


def test_quantile_knots_deduplicate_below_the_requested_maximum():
    transformer = make("quantile", n_quantiles=256).fit(frame_with([0.0, 0.0, 2.0, 2.0]))
    knots = transformer.quantile_knots_
    assert transformer.n_quantile_knots_ == len(knots) < 256
    assert np.all(np.diff(knots) > 0)
    assert len(np.unique(knots)) == len(knots)
    np.testing.assert_allclose(feature_of(transformer, [1.0]), [0.5])


@pytest.mark.parametrize("n_bins", [2, 8, 64])
def test_bin_output_is_monotone_bounded_and_integral(n_bins):
    transformer = make("bin", n_bins=n_bins).fit(frame_with(np.arange(64, dtype=float)))
    probe = np.linspace(-50.0, 150.0, 97)
    values = feature_of(transformer, probe)
    assert np.all(np.diff(values) >= 0)
    assert values.min() == 0.0
    assert values.max() == transformer.n_bin_edges_ <= n_bins - 1
    np.testing.assert_allclose(values, np.floor(values))


@pytest.mark.parametrize("n_quantiles", [2, 64, 256])
def test_quantile_output_is_monotone_and_inside_the_fitted_cdf_range(n_quantiles):
    transformer = make("quantile", n_quantiles=n_quantiles).fit(
        frame_with(np.arange(64, dtype=float))
    )
    probe = np.linspace(-50.0, 150.0, 97)
    values = feature_of(transformer, probe)
    assert np.all(np.diff(values) >= 0)
    assert values.min() == transformer.quantile_cdf_[0] >= 0.0
    assert values.max() == transformer.quantile_cdf_[-1] <= 1.0


def test_two_point_training_support_is_the_smallest_accepted_fit():
    binned = make("bin", n_bins=2).fit(frame_with([4.0, 6.0]))
    np.testing.assert_allclose(binned.bin_interior_edges_, [5.0])
    quantiled = make("quantile", n_quantiles=2).fit(frame_with([4.0, 6.0]))
    np.testing.assert_allclose(feature_of(quantiled, [4.0, 5.0, 6.0]), [0.25, 0.5, 0.75])


# --- Insufficient training support --------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("meter", [[7.0], [7.0, np.nan], []])
def test_fewer_than_two_finite_training_values_fails(method, meter):
    with pytest.raises(ValueError, match="two finite"):
        make(method).fit(frame_with(meter))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("meter", [[5.0, 5.0], [5.0, 5.0, 5.0, np.nan], [0.0] * 16])
def test_constant_training_values_have_no_distinct_variation(method, meter):
    with pytest.raises(ValueError, match="distinct"):
        make(method).fit(frame_with(meter))


@pytest.mark.parametrize("method", METHODS)
def test_all_null_training_meter_fails_for_object_and_nullable_storage(method):
    frame = frame_with([np.nan] * 16)
    for column in (pd.array([pd.NA] * 16, dtype="Float64"), np.full(16, np.nan)):
        with pytest.raises(ValueError, match="two finite"):
            make(method).fit(frame.assign(meter_reading=column))


# --- Parameter bounds ----------------------------------------------------------


@pytest.mark.parametrize("method", ["Bin", "quantiles", "", "histogram", None, 1, b"bin"])
def test_only_the_two_declared_methods_are_supported(method):
    with pytest.raises(ValueError, match="method"):
        make(method).fit(frame_with([0.0, 1.0]))
    with pytest.raises(ValueError, match="method"):
        api().build_distribution_registry(method)


@pytest.mark.parametrize("n_bins", [1, 0, -2, 65, 1000, True, False, 8.0, "8", None])
def test_n_bins_requires_an_integer_inside_the_versioned_bounds(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        make("bin", n_bins=n_bins).fit(frame_with([0.0, 1.0]))
    with pytest.raises(ValueError, match="n_bins"):
        make("quantile", n_bins=n_bins).fit(frame_with([0.0, 1.0]))


@pytest.mark.parametrize("n_quantiles", [1, 0, -2, 257, 1000, True, False, 64.0, "64", None])
def test_n_quantiles_requires_an_integer_inside_the_versioned_bounds(n_quantiles):
    with pytest.raises(ValueError, match="n_quantiles"):
        make("quantile", n_quantiles=n_quantiles).fit(frame_with([0.0, 1.0]))
    with pytest.raises(ValueError, match="n_quantiles"):
        make("bin", n_quantiles=n_quantiles).fit(frame_with([0.0, 1.0]))


# --- Supervision and held-out isolation ---------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_fitted_state_is_independent_of_any_supplied_label_vector(method):
    frame = frame_with([0.0, 1.0, 2.0, 3.0, 10.0])
    unsupervised = make(method).fit(frame)
    for labels in (np.zeros(5), np.ones(5), np.array([0, 1, 0, 1, 1]), None):
        assert fitted_state(make(method).fit(frame, labels)) == fitted_state(unsupervised)
    with pytest.raises(TypeError):
        make(method).fit(frame, groups=core_frame().attrs["groups"])


@pytest.mark.parametrize("method", METHODS)
def test_large_held_out_batches_cannot_change_fitted_state_or_results(method):
    transformer = make(method).fit(frame_with([0.0, 1.0, 2.0, 3.0]))
    before = fitted_state(transformer)
    first = feature_of(transformer, [0.5, 2.5])
    extreme = np.concatenate([np.linspace(-1e6, 1e6, 500), np.full(12, np.nan)])
    transformer.transform(frame_with(extreme))
    transformer.transform(frame_with(extreme[::-1]))
    assert fitted_state(transformer) == before
    np.testing.assert_allclose(feature_of(transformer, [0.5, 2.5]), first)


# --- Schema, dtype and frame contracts ----------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_raw_predictors_are_preserved_without_category_expansion(method):
    frame = frame_with([0.0, 1.0, 2.0, 3.0], index=[7, 2, 8, 1])
    original = frame.copy(deep=True)
    output = make(method).fit_transform(frame)
    assert output.shape == (4, len(CORE_COLUMNS) + 1)
    assert output.columns.tolist() == [*CORE_COLUMNS, OUTPUT_NAMES[method]]
    assert output.index.equals(frame.index)
    pd.testing.assert_frame_equal(output[list(CORE_COLUMNS)], original)
    assert output[OUTPUT_NAMES[method]].dtype == np.dtype("float64")
    pd.testing.assert_frame_equal(frame, original)
    output.iloc[0, 0] = 999.0
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("method", METHODS)
def test_subset_pools_keep_their_ordered_schema_and_require_meter_reading(method):
    frame = frame_with([0.0, 1.0, 2.0, 3.0])[["hour", "meter_reading", "primary_use"]]
    transformer = make(method).fit(frame)
    output = transformer.transform(frame)
    assert output.columns.tolist() == ["hour", "meter_reading", "primary_use", OUTPUT_NAMES[method]]
    assert transformer.n_features_in_ == 3
    with pytest.raises(ValueError, match="meter_reading"):
        make(method).fit(frame.drop(columns="meter_reading"))
    with pytest.raises(ValueError, match="schema"):
        transformer.transform(frame[["meter_reading", "hour", "primary_use"]])
    with pytest.raises(ValueError, match="schema"):
        transformer.transform(frame.drop(columns="hour"))


@pytest.mark.parametrize("method", METHODS)
def test_empty_transform_returns_the_typed_fitted_schema(method):
    frame = frame_with([0.0, 1.0, 2.0, 3.0])
    transformer = make(method).fit(frame)
    empty = transformer.transform(frame.iloc[:0])
    assert len(empty) == 0
    assert empty.columns.tolist() == [*CORE_COLUMNS, OUTPUT_NAMES[method]]
    assert empty[OUTPUT_NAMES[method]].dtype == np.dtype("float64")
    with pytest.raises(ValueError, match="two finite"):
        make(method).fit(frame.iloc[:0])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "column", ["anomaly", "target", "building_id", "row_id", "gte_probe", "unknown_source"]
)
def test_forbidden_or_unknown_predictor_columns_are_rejected(method, column):
    frame = frame_with([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="schema|predictor|column"):
        make(method).fit(frame.assign(**{column: 1.0}))


@pytest.mark.parametrize("method", METHODS)
def test_meter_reading_dtype_and_finiteness_are_enforced(method):
    frame = frame_with([0.0, 1.0, 2.0, 3.0])
    transformer = make(method).fit(frame)
    for value in ("1", True, 1j, pd.Timestamp("2024-01-01")):
        with pytest.raises(TypeError, match="meter_reading"):
            make(method).fit(frame.assign(meter_reading=value))
        with pytest.raises(TypeError, match="meter_reading"):
            transformer.transform(frame.assign(meter_reading=value))
    for value in (np.inf, -np.inf):
        with pytest.raises(ValueError, match="nonfinite"):
            make(method).fit(frame.assign(meter_reading=value))
        with pytest.raises(ValueError, match="nonfinite"):
            transformer.transform(frame.assign(meter_reading=value))
    with pytest.raises(TypeError, match="DataFrame"):
        make(method).fit(frame.to_numpy())
    duplicated = pd.DataFrame(
        frame[["meter_reading", "hour", "hour"]].to_numpy(),
        columns=["meter_reading", "hour", "hour"],
    )
    with pytest.raises(ValueError, match="unique"):
        make(method).fit(duplicated)


@pytest.mark.parametrize("method", METHODS)
def test_nullable_numeric_training_storage_matches_float_storage(method):
    values = [0.0, 1.0, 2.0, np.nan]
    frame = frame_with(values)
    nullable = frame.assign(meter_reading=pd.array([0.0, 1.0, 2.0, pd.NA], dtype="Float64"))
    expected = make(method).fit(frame)
    assert fitted_state(make(method).fit(nullable)) == fitted_state(expected)
    np.testing.assert_allclose(
        make(method).fit_transform(nullable)[OUTPUT_NAMES[method]].to_numpy(dtype=float),
        expected.transform(frame)[OUTPUT_NAMES[method]].to_numpy(dtype=float),
        equal_nan=True,
    )


# --- sklearn estimator contracts ----------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_fitted_checks_cloning_and_parameter_change_detection(method):
    frame = frame_with([0.0, 1.0, 2.0, 3.0])
    transformer = make(method, n_bins=4, n_quantiles=16)
    for operation in (lambda: transformer.transform(frame), transformer.get_feature_names_out):
        with pytest.raises(NotFittedError):
            operation()
    transformer.fit(frame)
    assert transformer.get_params() == {"method": method, "n_bins": 4, "n_quantiles": 16}
    duplicate = clone(transformer)
    assert not hasattr(duplicate, "output_features_")
    assert fitted_state(duplicate.fit(frame)) == fitted_state(transformer)
    assert transformer.get_feature_names_out().tolist() == [*CORE_COLUMNS, OUTPUT_NAMES[method]]
    with pytest.raises(ValueError, match="input_features"):
        transformer.get_feature_names_out(["meter_reading"])
    for change in ({"method": "quantile" if method == "bin" else "bin"},
                   {"n_bins": 8}, {"n_quantiles": 32}):
        changed = clone(transformer).fit(frame)
        changed.set_params(**change)
        with pytest.raises(ValueError, match="parameter"):
            changed.transform(frame)
        with pytest.raises(ValueError, match="parameter"):
            changed.get_feature_names_out()


@pytest.mark.parametrize("method", METHODS)
def test_transformer_composes_with_the_fold_fitted_imputation_preprocessor(method):
    frame = frame_with([0.0, 1.0, np.nan, 3.0])
    pipeline = Pipeline([("distribution", make(method)), ("preprocess", make_imputation_preprocessor())])
    output = pipeline.fit_transform(frame)
    assert output.shape[0] == 4
    assert len(pipeline.get_feature_names_out()) == output.shape[1]
    assert pipeline.transform(frame.iloc[[0]]).shape[0] == 1


# --- Registry metadata --------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_registry_declares_outer_train_target_independent_meter_lineage(method):
    registry = api().build_distribution_registry(method)
    metadata = registry.get(OUTPUT_NAMES[method])
    assert metadata.source_columns == ("meter_reading",)
    assert metadata.transform_fit_boundary is TransformFitBoundary.OUTER_TRAIN
    assert metadata.target_dependence is TargetDependence.INDEPENDENT
    assert metadata.provenance_status is ProvenanceStatus.VERIFIED
    assert "missing" in metadata.fallback_behavior.casefold()
    assert registry.names == (*build_lead_baseline_registry().names, OUTPUT_NAMES[method])
    assert set(OUTPUT_NAMES.values()) & set(registry.names) == {OUTPUT_NAMES[method]}


@pytest.mark.parametrize("method", METHODS)
def test_unsafe_baseline_source_metadata_is_never_promoted(method, monkeypatch):
    baseline = build_lead_baseline_registry()
    unsafe = FeatureRegistry(
        replace(baseline.get(name), source_columns=("anomaly",)) if name == "meter_reading"
        else baseline.get(name)
        for name in baseline.names
    )
    monkeypatch.setattr(api(), "build_lead_baseline_registry", lambda: unsafe)
    with pytest.raises(ValueError, match="source|lineage"):
        api().build_distribution_registry(method)
    with pytest.raises(ValueError, match="source|lineage"):
        make(method).fit(frame_with([0.0, 1.0, 2.0, 3.0]))


def test_declared_output_names_and_bounds_match_the_reviewed_design():
    module = api()
    assert module.OUTPUT_NAMES == OUTPUT_NAMES
    assert module.N_BINS_BOUNDS == (2, 64)
    assert module.N_QUANTILES_BOUNDS == (2, 256)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("index", [[1, 1, 2, 3], [1, None, 2, 3]])
def test_invalid_row_identity_is_rejected_at_fit_and_transform(method, index):
    valid = frame_with([0, 1, 2, 3])
    invalid = frame_with([0, 1, 2, 3], index=index)
    transformer = make(method).fit(valid)
    with pytest.raises(ValueError, match="index"):
        make(method).fit(invalid)
    with pytest.raises(ValueError, match="index"):
        transformer.transform(invalid)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("column,value", [
    ("square_feet", np.inf), ("square_feet", True), ("floor_count", 1j),
    ("year_built", "2000"), ("hour", 24), ("weekday", -1),
    ("month", 1.5), ("primary_use", 42),
    ("primary_use", pd.Categorical(["Office"] * 4, categories=["Office", 42])),
])
def test_every_raw_predictor_domain_is_validated(method, column, value):
    valid = frame_with([0, 1, 2, 3])
    invalid = valid.assign(**{column: value})
    transformer = make(method).fit(valid)
    for operation in (lambda: make(method).fit(invalid), lambda: transformer.transform(invalid)):
        with pytest.raises((TypeError, ValueError), match=column):
            operation()


@pytest.mark.parametrize("method", METHODS)
def test_extreme_finite_support_has_finite_linear_edges_and_outputs(method):
    transformer = make(method, n_bins=4, n_quantiles=2).fit(frame_with([-1e308, 1e308]))
    values = feature_of(transformer, [-1.5e308, -1e308, -5e307, 0, 5e307, 1e308, 1.5e308])
    assert np.isfinite(values).all()
    if method == "bin":
        np.testing.assert_allclose(transformer.bin_interior_edges_, [-5e307, 0, 5e307])
        np.testing.assert_allclose(values, [0, 0, 1, 2, 3, 3, 3])
    else:
        np.testing.assert_allclose(transformer.quantile_knots_, [-1e308, 1e308])
        np.testing.assert_allclose(values, [0.25, 0.25, 0.375, 0.5, 0.625, 0.75, 0.75])


@pytest.mark.parametrize("method", METHODS)
def test_minimum_subnormal_support_preserves_endpoint_and_midpoint_semantics(method):
    smallest = float.fromhex("0x0.0000000000001p-1022")
    transformer = make(method, n_bins=2, n_quantiles=2).fit(frame_with([-smallest, smallest]))
    values = feature_of(transformer, [-smallest, 0.0, smallest])
    assert np.isfinite(values).all()
    if method == "quantile":
        np.testing.assert_array_equal(transformer.quantile_knots_, [-smallest, smallest])
        np.testing.assert_array_equal(values, [0.25, 0.5, 0.75])
    else:
        np.testing.assert_array_equal(transformer.bin_interior_edges_, [0.0])
        np.testing.assert_array_equal(values, [0.0, 1.0, 1.0])


@pytest.mark.parametrize("sign", [-1, 1])
def test_large_same_sign_intervals_preserve_linear_cdf_and_bin_semantics(sign):
    support = sorted([sign * 1e308, sign * 1.5e308])
    probes = [support[0], sign * 1.25e308, support[1]]
    quantile = make("quantile", n_quantiles=2).fit(frame_with(support))
    np.testing.assert_allclose(feature_of(quantile, probes), [0.25, 0.5, 0.75])
    binned = make("bin", n_bins=2).fit(frame_with(support))
    np.testing.assert_allclose(binned.bin_interior_edges_, [sign * 1.25e308])
    np.testing.assert_array_equal(feature_of(binned, probes), [0, 1, 1])
