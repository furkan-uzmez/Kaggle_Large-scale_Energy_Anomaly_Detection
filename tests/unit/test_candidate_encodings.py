"""Synthetic contracts for the bounded encoding campaign stage."""

import importlib
import importlib.util
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from project.features.registry import (
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)
from project.features.transformers import make_imputation_preprocessor

RECIPES = ("feat_missingness_bitmask", "feat_frequency_encoding")
COLUMNS = (
    "meter_reading", "square_feet", "year_built", "floor_count", "hour",
    "weekday", "month", "year", "primary_use",
)


def api():
    name = "project.features.encodings"
    assert importlib.util.find_spec(name), "bounded candidate encoding API is missing"
    return importlib.import_module(name)


def make(recipe):
    return api().make_encoding_transformer(recipe)


@pytest.fixture
def frame():
    return pd.DataFrame({
        "meter_reading": [1., 2., np.nan, 4.], "square_feet": [100.] * 4,
        "year_built": [2000.] * 4, "floor_count": [1.] * 4,
        "hour": [0., 1., 2., 3.], "weekday": [0.] * 4,
        "month": [1.] * 4, "year": [2024.] * 4,
        "primary_use": ["office", "office", "retail", None],
    }, index=pd.Index([7, 2, 7, 1], name="source_row"))


@pytest.mark.parametrize("recipe", RECIPES)
def test_contract_clone_labels_index_names_empty_and_no_mutation(frame, recipe):
    original = frame.copy(deep=True)
    transformer = make(recipe)
    for operation in (lambda: transformer.transform(frame), transformer.get_feature_names_out):
        with pytest.raises(NotFittedError):
            operation()
    output = transformer.fit_transform(frame, np.zeros(len(frame)))
    pd.testing.assert_frame_equal(output, clone(transformer).fit_transform(frame, np.ones(len(frame))))
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(output[frame.columns], original)
    assert output.index.equals(frame.index)
    assert output.columns.tolist() == transformer.get_feature_names_out(frame.columns).tolist()
    assert output.shape[1] == 10
    assert transformer.n_features_in_ == 9
    assert transformer.transform(frame.iloc[:0]).columns.tolist() == output.columns.tolist()
    assert not hasattr(clone(transformer), "feature_names_in_")
    with pytest.raises(ValueError, match="row"):
        make(recipe).fit(frame.iloc[:0])
    with pytest.raises(ValueError, match="input_features"):
        transformer.get_feature_names_out(["unknown"])
    output.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(frame, original)


def test_all_512_masks_are_collision_free_canonical_and_bounded(frame):
    patterns = np.arange(512, dtype=np.uint16)
    exhaustive = pd.concat([frame.iloc[[0]]] * 512, ignore_index=True)
    for bit, column in enumerate(COLUMNS):
        exhaustive.loc[(patterns & (1 << bit)) != 0, column] = None if column == "primary_use" else np.nan
    first = make(RECIPES[0]).fit_transform(exhaustive)
    reversed_frame = exhaustive.iloc[:, ::-1]
    second = make(RECIPES[0]).fit_transform(reversed_frame)
    np.testing.assert_array_equal(first.missingness_bitmask, patterns)
    np.testing.assert_array_equal(second.missingness_bitmask, patterns)
    assert first.missingness_bitmask.dtype == np.dtype("uint16")
    assert second.columns.tolist() == list(reversed_frame.columns) + ["missingness_bitmask"]
    assert api().BITMASK_COLUMNS == COLUMNS


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_frequency_train_denominator_missing_unknown_and_no_adaptation(frame, dtype):
    frame["primary_use"] = pd.array(["office", "office", "retail", pd.NA], dtype=dtype)
    transformer = make(RECIPES[1]).fit(frame)
    validation = frame.copy()
    validation["primary_use"] = pd.array(["office", "unseen", pd.NA, np.nan], dtype=dtype)
    before = transformer.transform(validation)
    np.testing.assert_array_equal(before.primary_use_frequency, [.5, 0., 0., 0.])
    transformer.transform(pd.concat([validation] * 20))
    pd.testing.assert_frame_equal(before, transformer.transform(validation))
    assert transformer.transform(frame).primary_use_frequency.tolist() == [.5, .5, .25, 0.]
    assert before.primary_use_frequency.dtype == np.dtype("float64")
    pd.testing.assert_frame_equal(before.iloc[[0]], transformer.transform(validation.iloc[[0]]))


def test_nullable_numeric_and_all_null_category(frame):
    for name in COLUMNS[:-1]:
        frame[name] = pd.array([pd.NA] * 4, dtype="Float64")
    frame["primary_use"] = pd.array([pd.NA] * 4, dtype="string")
    assert make(RECIPES[0]).fit_transform(frame).missingness_bitmask.tolist() == [511] * 4
    assert make(RECIPES[1]).fit_transform(frame).primary_use_frequency.tolist() == [0.] * 4


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("column", ["anomaly", "target", "building_id", "row_id", "gte_probe", "unknown"])
def test_forbidden_and_overwidth_schema_rejected(frame, recipe, column):
    with pytest.raises(ValueError, match="schema|predictor|column"):
        make(recipe).fit(frame.assign(**{column: 1}))


@pytest.mark.parametrize("recipe", RECIPES)
def test_required_schema_dtype_and_parameter_mismatch(frame, recipe):
    transformer = make(recipe).fit(frame)
    for bad in (frame.drop(columns="hour"), frame.iloc[:, ::-1], pd.concat([frame, frame[["hour"]]], axis=1)):
        with pytest.raises(ValueError, match="schema|column|unique"):
            transformer.transform(bad)
    with pytest.raises(ValueError, match="schema|column"):
        make(recipe).fit(frame.drop(columns="hour"))
    with pytest.raises(TypeError, match="DataFrame"):
        make(recipe).fit(frame.to_numpy())
    for column, value in (("meter_reading", "1"), ("floor_count", True), ("hour", 1j), ("primary_use", 1)):
        with pytest.raises(TypeError, match=column):
            transformer.transform(frame.assign(**{column: value}))
    with pytest.raises(ValueError, match="nonfinite"):
        transformer.transform(frame.assign(meter_reading=np.inf))
    transformer.set_params(recipe_id=RECIPES[1] if recipe == RECIPES[0] else RECIPES[0])
    with pytest.raises(ValueError, match="parameter"):
        transformer.transform(frame)
    with pytest.raises(ValueError, match="recipe"):
        make("feat_conditional_cross_fitted_target_encoding")


@pytest.mark.parametrize("recipe,boundary,name,sources", [
    (RECIPES[0], TransformFitBoundary.FIXED, "missingness_bitmask", COLUMNS),
    (RECIPES[1], TransformFitBoundary.OUTER_TRAIN, "primary_use_frequency", ("primary_use",)),
])
def test_metadata_and_synthetic_pipeline(frame, recipe, boundary, name, sources):
    registry = api().build_encoding_registry(recipe)
    meta = registry.get(name)
    assert meta.source_columns == sources
    assert meta.transform_fit_boundary is boundary
    assert meta.target_dependence is TargetDependence.INDEPENDENT
    assert meta.provenance_status is ProvenanceStatus.VERIFIED
    assert "0" in meta.fallback_behavior
    pipeline = Pipeline([("encoding", make(recipe)), ("preprocess", make_imputation_preprocessor())])
    output = pipeline.fit_transform(frame)
    assert output.shape[0] == 4
    assert len(pipeline.get_feature_names_out()) == output.shape[1]
    assert pipeline.transform(frame.iloc[[0]]).shape[0] == 1


@pytest.mark.parametrize("recipe", RECIPES)
def test_unsafe_source_metadata_never_promoted(frame, monkeypatch, recipe):
    baseline = build_lead_baseline_registry()
    unsafe = FeatureRegistry(replace(baseline.get(n), source_columns=("anomaly",))
                             if n == "primary_use" else baseline.get(n) for n in baseline.names)
    monkeypatch.setattr(api(), "build_lead_baseline_registry", lambda: unsafe)
    with pytest.raises(ValueError, match="source|lineage"):
        make(recipe).fit(frame)


@pytest.mark.parametrize("recipe", RECIPES)
def test_missing_datetime_is_not_a_categorical_dtype(frame, recipe):
    with pytest.raises(TypeError, match="primary_use"):
        make(recipe).fit(frame.assign(primary_use=pd.NaT))


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("categories", [[1, 2], [pd.Timestamp("2024-01-01")]])
def test_null_categorical_rejects_nonstring_vocabulary(frame, recipe, categories):
    bad = frame.assign(primary_use=pd.Categorical([pd.NA] * len(frame), categories=categories))
    transformer = make(recipe).fit(frame)
    for operation in (make(recipe).fit, transformer.transform):
        with pytest.raises(TypeError, match="primary_use"):
            operation(bad)


def test_frequency_cardinality_cap_and_unused_vocabulary(frame):
    oversized = pd.concat([frame.iloc[[0]]] * 4097, ignore_index=True)
    oversized["primary_use"] = [f"category_{i}" for i in range(4097)]
    with pytest.raises(ValueError, match="cardinality"):
        make(RECIPES[1]).fit(oversized)
    frame["primary_use"] = pd.Categorical(frame.primary_use, categories=["office", "retail", "unused"])
    transformer = make(RECIPES[1]).fit(frame)
    assert set(transformer.frequencies_) == {"office", "retail"}


def test_recipe_ids_and_canonical_order_match_frozen_configuration():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    baseline = yaml.safe_load((root / "configs/experiments/baselines.yaml").read_text())
    campaign = yaml.safe_load((root / "configs/experiments/feature_campaign.yaml").read_text())
    assert COLUMNS == tuple(baseline["features"]["numeric"] + baseline["features"]["categorical"])
    assert set(RECIPES) <= {row["id"] for row in campaign["experiment_matrix"]}
