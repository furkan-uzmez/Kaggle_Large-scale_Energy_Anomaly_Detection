"""Synthetic contracts for deterministic, provenance-safe candidate features."""

import importlib
import importlib.util
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from project.features.availability import validate_feature_availability
from project.features.registry import (
    AvailabilityStatus,
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)
from project.features.transformers import make_imputation_preprocessor

RECIPES = (
    "feat_missingness_indicators",
    "feat_safe_calendar_numeric_transforms",
    "feat_categorical_interactions",
)


def api():
    name = "project.features.candidates"
    assert importlib.util.find_spec(name), "deterministic candidate feature API is missing"
    return importlib.import_module(name)


@pytest.fixture
def frame():
    return pd.DataFrame({
        "meter_reading": [-3.0, -0.0, 0.0, np.nan],
        "primary_use": ["office", 'office|["6"]', "retail", None],
        "square_feet": [100.0, 200.0, np.nan, 0.0],
        "year_built": [2000.0, 2024.0, np.nan, 1990.0],
        "floor_count": [1.0, 2.0, 0.0, np.nan],
        "hour": [0.0, 6.0, 12.0, np.nan],
        "weekday": [0.0, 6.0, 2.0, np.nan],
        "month": [1.0, 12.0, 5.0, np.nan],
        "year": [2024.0, 2024.0, 2024.0, np.nan],
    }, index=pd.Index([7, 2, 7, 1], name="source_row"))


def make(recipe):
    return api().make_candidate_transformer(recipe)


def test_recipe_ids_resolve_to_actual_matrix():
    campaign = yaml.safe_load((Path(__file__).resolve().parents[2] /
        "configs/experiments/feature_campaign.yaml").read_text())
    assert set(RECIPES) <= {row["id"] for row in campaign["experiment_matrix"]}
    with pytest.raises(ValueError, match="recipe"):
        make("feat_frequency_encoding")


@pytest.mark.parametrize("recipe", RECIPES)
def test_clone_schema_index_no_mutation_and_rowwise_determinism(frame, recipe):
    original = frame.copy(deep=True)
    transformer = make(recipe)
    copied = clone(transformer)
    assert copied is not transformer
    assert copied.get_params() == transformer.get_params()
    output = transformer.fit_transform(frame, y=np.array([0, 1, 0, 1]))
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(output[frame.columns], original)
    assert output.index.equals(frame.index)
    assert output.columns.tolist() == transformer.get_feature_names_out().tolist()
    assert len(set(output.columns)) == len(output.columns)
    assert transformer.n_features_in_ == len(frame.columns)
    assert transformer.feature_names_in_.tolist() == frame.columns.tolist()
    pd.testing.assert_frame_equal(output, copied.fit_transform(frame, y=np.array([1, 0, 1, 0])))
    pd.testing.assert_frame_equal(output, transformer.transform(frame))
    pd.testing.assert_frame_equal(output.iloc[[1]], transformer.transform(frame.iloc[[1]]))
    output.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(frame, original)
    empty = transformer.transform(frame.iloc[:0])
    assert empty.index.equals(frame.iloc[:0].index)
    assert empty.columns.tolist() == transformer.get_feature_names_out().tolist()
    empty_fitted = make(recipe).fit_transform(frame.iloc[:0])
    assert empty_fitted.columns.tolist() == empty.columns.tolist()


def test_flags_include_validation_only_missingness_before_imputation(frame):
    transformer = make(RECIPES[0]).fit(frame.fillna({"meter_reading": 8.0}))
    result = transformer.transform(frame)
    for name in frame:
        np.testing.assert_array_equal(result[f"{name}_missing"], frame[name].isna().astype(np.uint8))
    assert result["meter_reading"].isna().iloc[-1]
    assert result["meter_reading_missing"].iloc[-1] == 1
    result = make(RECIPES[0]).fit_transform(pd.DataFrame({"meter_reading": pd.Series([pd.NA], dtype="Float64")}))
    assert result.meter_reading_missing.tolist() == [1]


def test_safe_numeric_values_signed_zero_hour_period_and_age(frame):
    result = make(RECIPES[1]).fit_transform(frame)
    log = result.meter_reading_signed_log1p.to_numpy()
    assert log[0] == pytest.approx(-np.log(4))
    assert log[1] == 0 and np.signbit(log[1])
    assert log[2] == 0 and not np.signbit(log[2])
    assert np.isnan(log[3])
    np.testing.assert_allclose(result.hour_sin.iloc[:3], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(result.hour_cos.iloc[:3], [1, 0, -1], atol=1e-15)
    assert result.hour_sin.isna().iloc[-1] and result.hour_cos.isna().iloc[-1]
    assert result.building_age.iloc[:2].tolist() == [24, 0]
    assert result.building_age.iloc[2:].isna().all()
    maximum = frame.iloc[[0]].copy()
    maximum["meter_reading"] = np.finfo(float).max
    assert np.isfinite(make(RECIPES[1]).fit_transform(maximum).meter_reading_signed_log1p).all()


def test_interactions_are_lossless_pairs_and_propagate_either_missing_component(frame):
    import json

    result = make(RECIPES[2]).fit_transform(frame)
    for calendar in ("hour", "weekday", "month"):
        name = f"primary_use_x_{calendar}"
        assert json.loads(result[name].iloc[1]) == [frame.primary_use.iloc[1], int(frame[calendar].iloc[1])]
        assert result[name].isna().iloc[3]
    missing = frame.iloc[[0, 1]].copy()
    missing.iloc[0, missing.columns.get_loc("primary_use")] = None
    missing.iloc[1, missing.columns.get_loc("hour")] = np.nan
    out = make(RECIPES[2]).fit_transform(missing)
    assert out.primary_use_x_hour.isna().all()
    unseen = frame.iloc[[0]].copy()
    unseen["primary_use"] = "unseen\"|[category]"
    assert json.loads(make(RECIPES[2]).fit(frame).transform(unseen).primary_use_x_hour.iloc[0]) == [unseen.primary_use.iloc[0], 0]


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("column", ["anomaly", "target", "building_id", "row_id", "gte_probe", "unknown", "hour_sin"])
def test_unknown_and_forbidden_columns_rejected_not_dropped(frame, recipe, column):
    frame[column] = 1
    with pytest.raises(ValueError, match="column|predictor|schema"):
        make(recipe).fit_transform(frame)


@pytest.mark.parametrize("recipe", RECIPES)
def test_fitted_schema_strict_and_unfitted_use_rejected(frame, recipe):
    transformer = make(recipe)
    with pytest.raises(NotFittedError):
        transformer.transform(frame)
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()
    transformer.fit(frame)
    for bad in (frame.iloc[:, ::-1], frame.drop(columns="meter_reading"), frame.assign(extra=1)):
        with pytest.raises(ValueError, match="schema|column"):
            transformer.transform(bad)
    with pytest.raises(ValueError, match="input_features"):
        transformer.get_feature_names_out(["invented"])
    with pytest.raises(ValueError, match="unique|duplicate"):
        make(recipe).fit(pd.concat([frame, frame[["hour"]]], axis=1))
    with pytest.raises(TypeError, match="DataFrame"):
        make(recipe).fit(frame.to_numpy())
    with pytest.raises(ValueError, match="column|feature"):
        make(recipe).fit(pd.DataFrame(index=frame.index))


@pytest.mark.parametrize("recipe,column", [(RECIPES[1], "hour"), (RECIPES[1], "year_built"), (RECIPES[2], "primary_use"), (RECIPES[2], "weekday")])
def test_recipe_required_inputs_fail_closed(frame, recipe, column):
    with pytest.raises(ValueError, match="required"):
        make(recipe).fit(frame.drop(columns=column))


@pytest.mark.parametrize("column,value", [
    ("meter_reading", np.inf), ("square_feet", -np.inf), ("hour", 24),
    ("hour", -1), ("hour", 1.5), ("weekday", 7), ("month", 0),
    ("month", 13), ("year", 0), ("year", 10000), ("year_built", 2025),
    ("year_built", 1990.5), ("floor_count", True), ("meter_reading", "3"),
    ("primary_use", 7), ("meter_reading", 1 + 2j),
])
def test_invalid_domains_and_types_rejected_in_fit_and_transform(frame, column, value):
    transformer = make(RECIPES[1]).fit(frame)
    bad = frame.copy()
    bad[column] = value
    for operation in (make(RECIPES[1]).fit, transformer.transform):
        with pytest.raises((ValueError, TypeError), match=column):
            operation(bad)


@pytest.mark.parametrize("recipe", RECIPES)
def test_metadata_lineage_fixed_boundary_and_baseline_unmodified(frame, recipe):
    baseline = build_lead_baseline_registry()
    before = tuple(baseline.get(name) for name in baseline.names)
    registry = api().build_candidate_registry(recipe)
    transformer = make(recipe).fit(frame)
    output = transformer.get_feature_names_out()
    assert set(output) <= set(registry.names)
    for name in output:
        meta = registry.get(name)
        validate_feature_availability(meta, datetime(2026, 1, 1, tzinfo=UTC))
        if name not in baseline.names:
            assert meta.transform_fit_boundary is TransformFitBoundary.FIXED
            assert set(meta.source_columns) <= set(baseline.names)
            assert meta.provenance_status is ProvenanceStatus.VERIFIED
            assert meta.target_dependence is TargetDependence.INDEPENDENT
            assert meta.prediction_time_availability is AvailabilityStatus.AVAILABLE
    assert tuple(build_lead_baseline_registry().get(name) for name in baseline.names) == before


@pytest.mark.parametrize("field,value", [
    ("provenance_status", ProvenanceStatus.PENDING),
    ("target_dependence", TargetDependence.TARGET_DERIVED),
    ("prediction_time_availability", AvailabilityStatus.UNAVAILABLE),
    ("source_columns", ("anomaly",)),
])
def test_derived_metadata_never_upgrades_unsafe_source(monkeypatch, field, value):
    baseline = build_lead_baseline_registry()
    unsafe = FeatureRegistry(replace(baseline.get(n), **{field: value}) if n == "meter_reading" else baseline.get(n) for n in baseline.names)
    monkeypatch.setattr(api(), "build_lead_baseline_registry", lambda: unsafe)
    with pytest.raises(ValueError, match="provenance|target|available|source|lineage"):
        api().build_candidate_registry(RECIPES[1])


@pytest.mark.parametrize("recipe", RECIPES)
def test_pipeline_composition_clone_names_and_no_validation_statistics(frame, recipe):
    pipeline = Pipeline([("candidate", make(recipe)), ("preprocess", make_imputation_preprocessor())])
    output = pipeline.fit_transform(frame)
    assert output.shape[0] == len(frame)
    validation = frame.iloc[[1]].copy()
    validation["meter_reading"] = np.nan
    validation["primary_use"] = "validation_only"
    first = pipeline.transform(validation)
    second = pipeline.transform(validation)
    if hasattr(first, "toarray"):
        first, second = first.toarray(), second.toarray()
    np.testing.assert_array_equal(first, second)
    assert len(pipeline.get_feature_names_out()) == output.shape[1]
    assert not hasattr(clone(pipeline).named_steps["candidate"], "feature_names_in_")


@pytest.mark.parametrize("recipe", RECIPES)
def test_transform_is_independent_of_training_values(frame, recipe):
    different_training = frame.copy()
    replacements = {
        "meter_reading": 999.0, "primary_use": "different_training_only",
        "square_feet": 400.0, "floor_count": 100.0, "hour": 23.0,
        "weekday": 4.0, "month": 3.0, "year": 2030.0, "year_built": 1980.0,
    }
    for name, value in replacements.items():
        different_training[name] = value
    first = make(recipe).fit(frame).transform(frame)
    second = make(recipe).fit(different_training).transform(frame)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("recipe,columns,derived", [
    (RECIPES[0], ["primary_use", "meter_reading"], ["primary_use_missing", "meter_reading_missing"]),
    (RECIPES[1], ["meter_reading", "hour", "year", "year_built"],
     ["meter_reading_signed_log1p", "hour_sin", "hour_cos", "building_age"]),
    (RECIPES[2], ["primary_use", "hour", "weekday", "month"],
     ["primary_use_x_hour", "primary_use_x_weekday", "primary_use_x_month"]),
])
def test_valid_baseline_subset_has_explicit_output_schema(frame, recipe, columns, derived):
    subset = frame[columns].copy()
    transformer = make(recipe)
    result = transformer.fit_transform(subset)
    assert result.columns.tolist() == columns + derived
    assert transformer.get_feature_names_out().tolist() == columns + derived
    pd.testing.assert_frame_equal(result[columns], subset)
    assert result.index.equals(subset.index)
