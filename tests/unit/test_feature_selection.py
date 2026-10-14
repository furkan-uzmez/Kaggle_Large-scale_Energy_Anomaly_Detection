"""Synthetic fold-local dimensionality/selection contracts; no campaign runs."""

import importlib
import importlib.util
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC

from project.features.candidates import (
    build_candidate_registry,
    make_candidate_transformer,
)
from project.features.registry import (
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)

PCA_RECIPE = "dim_pca_vs_passthrough"
SELECTOR_RECIPE = "dim_fold_local_selector_vs_passthrough"
BLOCK_RECIPE = "dim_semantic_block_drop"
POOL = ("meter_reading", "square_feet", "primary_use")


def api():
    name = "project.features.selection"
    assert importlib.util.find_spec(name), "fold-local selection API is missing"
    return importlib.import_module(name)


@pytest.fixture
def data():
    frame = pd.DataFrame({
        "meter_reading": np.tile([-3., -3., 3., 3.], 20),
        "square_feet": np.tile([-100., 100., -100., 100.], 20),
        "primary_use": ["office"] * 80,
    }, index=pd.Index(np.arange(80) * 3, name="source_row"))
    return frame, pd.Series((frame.meter_reading > 0).astype(int), index=frame.index)


def arms(recipe, pool=POOL, registry=None, **options):
    return api().make_selection_arms(recipe, pool,
        registry=registry if registry is not None else build_lead_baseline_registry(), **options)


@pytest.mark.parametrize("recipe,key", [(PCA_RECIPE, "pca"), (SELECTOR_RECIPE, "selector"), (BLOCK_RECIPE, "drop_building_metadata")])
def test_matrix_factories_are_fresh_cloneable_fair_and_fitted_names_are_auditable(data, recipe, key):
    frame, target = data
    campaign = yaml.safe_load((Path(__file__).resolve().parents[2] /
        "configs/experiments/feature_campaign.yaml").read_text())
    assert recipe in {r["id"] for r in campaign["experiment_matrix"]}
    options = {"drop_block": "building_metadata"} if recipe == BLOCK_RECIPE else {}
    pair = arms(recipe, **options)
    assert set(pair) == {"passthrough", key}
    assert pair[key] is not pair["passthrough"]
    assert pair[key].named_steps["model"].get_params() == pair["passthrough"].named_steps["model"].get_params()
    original = frame.copy(deep=True)
    for model in pair.values():
        cloned = clone(model)
        assert not hasattr(cloned.named_steps["preprocessor"], "feature_names_in_")
        model.fit(frame, target)
        output = model[:-1].transform(frame)
        assert output.index.equals(frame.index)
        assert output.columns.tolist() == model[:-1].get_feature_names_out().tolist()
        assert len(model.predict(frame)) == len(frame)
        audit = api().selection_audit(model)
        assert audit["selected_names"] == output.columns.tolist()
        assert audit["model_params"] == model.named_steps["model"].get_params()
        assert audit["fit_boundary"] == "outer_train"
        for name in output:
            metadata = audit["registry"].get(name)
            assert metadata.transform_fit_boundary is TransformFitBoundary.OUTER_TRAIN
            assert metadata.provenance_status is ProvenanceStatus.VERIFIED
            assert metadata.target_dependence is TargetDependence.INDEPENDENT
    pd.testing.assert_frame_equal(frame, original)


def test_pca_numeric_only_and_train_statistics_heldout_labels_invariant(data):
    frame, target = data
    train = frame.copy()
    train.iloc[0, 0] = np.nan
    first, second = arms(PCA_RECIPE)["pca"], arms(PCA_RECIPE)["pca"]
    first.fit(train, target)
    second.fit(train, 1 - target)
    pre = first.named_steps["preprocessor"]
    numeric = pre.column_transformer_.named_transformers_["numeric"]
    other = second.named_steps["preprocessor"].column_transformer_.named_transformers_["numeric"]
    assert numeric.named_steps["pca"].components_.shape[1] == 2
    assert 1 <= numeric.named_steps["pca"].n_components_ <= min(len(train), 2)
    np.testing.assert_array_equal(numeric.named_steps["pca"].components_, other.named_steps["pca"].components_)
    medians = train[list(POOL[:2])].median().to_numpy()
    np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_, medians)
    means = train[list(POOL[:2])].fillna(train[list(POOL[:2])].median()).mean().to_numpy()
    np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, means)
    before = numeric.named_steps["scaler"].mean_.copy()
    heldout = frame.iloc[:4].copy()
    heldout["meter_reading"] = 10000.
    heldout["primary_use"] = "heldout_only"
    out = pre.transform(heldout)
    assert out.filter(like="categorical__").eq(0).all().all()
    assert "heldout_only" not in " ".join(pre.get_feature_names_out())
    np.testing.assert_array_equal(numeric.named_steps["scaler"].mean_, before)
    np.testing.assert_array_equal(first.decision_function(heldout), first.decision_function(heldout))
    assert not hasattr(first, "predict_proba")
    audit = api().selection_audit(first)
    assert audit["probability_evaluation_status"] == "blocked_pending_fold_local_calibration"
    assert audit["final_promotion_status"] == "blocked_pending_fold_local_calibration"
    assert audit["preprocessor_params"]["pca_components"] == .95


def test_control_and_pca_share_identical_imputation_scaling_categories(data):
    frame, target = data
    pair = arms(PCA_RECIPE)
    for model in pair.values():
        model.fit(frame, target)
    pre = [m.named_steps["preprocessor"].column_transformer_ for m in pair.values()]
    for step in ("imputer", "scaler"):
        assert pre[0].named_transformers_["numeric"].named_steps[step].get_params() == pre[1].named_transformers_["numeric"].named_steps[step].get_params()
    for transform in pre:
        assert "pca" not in transform.named_transformers_["categorical"].named_steps
    assert pre[0].named_transformers_["categorical"].get_params().keys() == pre[1].named_transformers_["categorical"].get_params().keys()


def test_selector_changes_with_training_signal_not_heldout_data(data):
    frame, target = data
    meter = arms(SELECTOR_RECIPE)["selector"].fit(frame, target)
    area_target = (frame.square_feet > 0).astype(int)
    area = arms(SELECTOR_RECIPE)["selector"].fit(frame, area_target)
    meter_support = api().get_selection_support(meter)
    area_support = api().get_selection_support(area)
    assert meter_support["numeric__meter_reading"] and not meter_support["numeric__square_feet"]
    assert area_support["numeric__square_feet"] and not area_support["numeric__meter_reading"]
    heldout = frame.iloc[:4].copy()
    heldout["primary_use"] = "unknown"
    meter.predict(heldout)
    pd.testing.assert_series_equal(meter_support, api().get_selection_support(meter))
    selector = meter.named_steps["selector"]
    assert selector.threshold == 1e-5
    assert selector.estimator_.l1_ratio == 1 and selector.estimator_.solver == "liblinear"
    assert selector.estimator_.C == 1 and selector.estimator_.random_state == 42
    assert meter.named_steps["model"].l1_ratio == 0


def test_mask_only_stability_has_known_frequency_and_jaccard():
    supports = [pd.Series([True, False], index=["a", "b"]),
                pd.Series([True, True], index=["a", "b"])]
    result = api().selection_stability(supports)
    assert result["selection_frequency"].to_dict() == {"a": 1., "b": .5}
    np.testing.assert_array_equal(result["jaccard"], [[1., .5], [.5, 1.]])
    differing_names = [supports[0], pd.Series([True], index=["c"])]
    assert api().selection_stability(differing_names)["selection_frequency"].to_dict() == {"a": .5, "b": 0., "c": .5}
    for invalid in ([], [pd.Series([1], index=["a"])], [pd.Series([False], index=["a"])], [pd.Series([True, True], index=["a", "a"])]):
        with pytest.raises(ValueError, match="mask|support|unique|empty|boolean"):
            api().selection_stability(invalid)


@pytest.mark.parametrize("recipe", [PCA_RECIPE, SELECTOR_RECIPE, BLOCK_RECIPE])
def test_strict_pool_schema_empty_and_unfitted_errors(data, recipe):
    frame, target = data
    options = {"drop_block": "building_metadata"} if recipe == BLOCK_RECIPE else {}
    model = arms(recipe, **options)["passthrough"]
    with pytest.raises(NotFittedError):
        api().selection_audit(model)
    for invalid in (frame.iloc[:, ::-1], frame.assign(unknown=1), frame.drop(columns="meter_reading"), pd.concat([frame, frame[["primary_use"]]], axis=1)):
        with pytest.raises(ValueError, match="schema|column|pool"):
            clone(model).fit(invalid, target)
    with pytest.raises(ValueError, match="row|empty"):
        clone(model).fit(frame.iloc[:0], target.iloc[:0])
    model.fit(frame, target)
    empty = model[:-1].transform(frame.iloc[:0])
    assert empty.shape == (0, len(model[:-1].get_feature_names_out()))
    with pytest.raises(ValueError, match="schema|column"):
        model.predict(frame.iloc[:, ::-1])
    with pytest.raises(TypeError, match="DataFrame"):
        clone(model).fit(frame.to_numpy(), target)


@pytest.mark.parametrize("recipe", [PCA_RECIPE, SELECTOR_RECIPE, BLOCK_RECIPE])
def test_targets_nonfinite_types_and_misaligned_index_fail_closed(data, recipe):
    frame, target = data
    options = {"drop_block": "building_metadata"} if recipe == BLOCK_RECIPE else {}
    model = arms(recipe, **options)["passthrough"]
    for invalid_target in (np.zeros(len(target), dtype=int), target.astype(bool), target.iloc[::-1], np.full(len(target), 2)):
        with pytest.raises(ValueError, match="target|class|index|binary"):
            clone(model).fit(frame, invalid_target)
    for name, value in (("meter_reading", np.inf), ("meter_reading", 1j), ("square_feet", True), ("primary_use", 7)):
        with pytest.raises((ValueError, TypeError), match=name):
            clone(model).fit(frame.assign(**{name: value}), target)


@pytest.mark.parametrize("name", ["anomaly", "building_id", "gte_probe", "unknown"])
def test_forbidden_or_unregistered_pool_denied_even_with_metadata(name):
    base = build_lead_baseline_registry()
    forged = replace(base.get("meter_reading"), name=name, source_columns=("meter_reading",))
    registry = FeatureRegistry([base.get(n) for n in base.names] + [forged])
    with pytest.raises(ValueError, match="pool|predictor|registered|forbidden|lineage"):
        arms(PCA_RECIPE, (name,), registry)


@pytest.mark.parametrize("change", [
    {"provenance_status": ProvenanceStatus.PENDING},
    {"target_dependence": TargetDependence.TARGET_DERIVED},
    {"source_columns": ("anomaly",)},
    {"source_columns": ("hidden_source",)},
    {"available_at": datetime.now(UTC) + timedelta(days=1)},
])
def test_unsafe_provenance_and_hidden_lineage_denied(change):
    base = build_lead_baseline_registry()
    registry = FeatureRegistry(replace(base.get(n), **change) if n == "meter_reading" else base.get(n) for n in base.names)
    with pytest.raises(ValueError, match="provenance|target|lineage|source|availability"):
        arms(PCA_RECIPE, registry=registry)
    with pytest.raises(TypeError, match="FeatureRegistry"):
        arms(PCA_RECIPE, registry={"meter_reading": {"verified": True}})


def test_semantic_blocks_remove_declared_engineered_descendants(data):
    frame, target = data
    expanded = frame.assign(year=2024., year_built=2000., hour=np.arange(80) % 24,
                            weekday=0., month=1., floor_count=1.)
    transformer = make_candidate_transformer("feat_safe_calendar_numeric_transforms")
    engineered = transformer.fit_transform(expanded)
    registry = build_candidate_registry("feat_safe_calendar_numeric_transforms")
    pair = arms(BLOCK_RECIPE, tuple(engineered.columns), registry, drop_block="calendar")
    dropped = pair["drop_calendar"].fit(engineered, target)
    kept = dropped.named_steps["preprocessor"].retained_features_
    assert "hour_sin" not in kept and "hour_cos" not in kept and "building_age" not in kept
    assert "meter_reading_signed_log1p" in kept and "primary_use" in kept
    assert "year" not in kept and "hour" not in kept
    assert pair["passthrough"].named_steps["model"].get_params() == dropped.named_steps["model"].get_params()
    with pytest.raises(ValueError, match="block"):
        arms(BLOCK_RECIPE, drop_block="unknown")
    with pytest.raises(ValueError, match="block|empty"):
        arms(BLOCK_RECIPE, ("meter_reading",), drop_block="measurement")
    with pytest.raises(ValueError, match="block|absent"):
        arms(BLOCK_RECIPE, ("meter_reading",), drop_block="calendar")


@pytest.mark.parametrize("components", [True, 0, -1, 1.5, 0., 1., float("nan"), 100])
def test_invalid_pca_components_denied_without_fallback(data, components):
    frame, target = data
    with pytest.raises(ValueError, match="component"):
        arms(PCA_RECIPE, pca_components=components)["pca"].fit(frame, target)


def test_pca_degenerate_rows_and_selector_empty_fail_individual_constant_allowed(data):
    frame, target = data
    with pytest.raises(ValueError, match="row|sample"):
        arms(PCA_RECIPE)["pca"].named_steps["preprocessor"].fit(frame.iloc[:1])
    constant = frame.assign(meter_reading=1., square_feet=np.nan)
    with pytest.raises(ValueError, match="variance"):
        arms(PCA_RECIPE)["pca"].fit(constant, target)
    with pytest.raises(ValueError, match="empty|select"):
        arms(SELECTOR_RECIPE)["selector"].fit(constant, target)
    individual = frame.assign(square_feet=np.nan)
    model = arms(PCA_RECIPE)["pca"].fit(individual, target)
    assert np.isfinite(model.decision_function(individual)).all()
    with pytest.raises(ValueError, match="numeric"):
        arms(PCA_RECIPE, ("primary_use",))


def test_parent_pool_is_explicit_dynamic_and_registry_unchanged(data):
    frame, target = data
    candidate = make_candidate_transformer("feat_missingness_indicators")
    engineered = candidate.fit_transform(frame)
    registry = build_candidate_registry("feat_missingness_indicators")
    model = arms(SELECTOR_RECIPE, tuple(engineered.columns), registry)["passthrough"].fit(engineered, target)
    assert model.named_steps["preprocessor"].feature_names_in_.tolist() == engineered.columns.tolist()
    assert "numeric__meter_reading_missing" in model[:-1].get_feature_names_out()
    assert len(build_lead_baseline_registry().names) == 9
    with pytest.raises(ValueError, match="recipe"):
        arms("not_a_matrix_recipe")
    with pytest.raises(ValueError, match="pool"):
        arms(PCA_RECIPE, ())


def test_categorical_feature_names_cannot_collide_across_declared_inputs():
    raw = pd.DataFrame({
        "primary_use": ['x_hour_["office",0]', "office"],
        "hour": [0., 0.], "weekday": [0., 0.], "month": [1., 1.],
    }, index=[9, 2])
    producer = make_candidate_transformer("feat_categorical_interactions")
    pool = ("primary_use", "primary_use_x_hour")
    frame = producer.fit_transform(raw).loc[:, pool]
    model = arms(SELECTOR_RECIPE, pool, producer.registry_)["passthrough"]
    output = model[:-1].fit_transform(frame, np.array([0, 1]))
    assert output.columns.is_unique
    assert output.shape == (2, 4)
    assert output.index.equals(frame.index)
    assert model[:-1].get_feature_names_out().tolist() == output.columns.tolist()


@pytest.mark.parametrize("changes", [
    {"recipe_id": "unknown_recipe"},
    {"recipe_id": SELECTOR_RECIPE, "use_pca": True},
    {"recipe_id": PCA_RECIPE, "drop_block": "building_metadata"},
])
def test_direct_preprocessor_rejects_invalid_recipe_combinations(data, changes):
    frame, target = data
    preprocessor = arms(PCA_RECIPE)["passthrough"].named_steps["preprocessor"]
    preprocessor.set_params(**changes)
    with pytest.raises(ValueError, match="recipe|PCA|block"):
        preprocessor.fit(frame, target)


def test_fitted_names_and_support_fail_closed_after_parameter_change(data):
    frame, target = data
    model = arms(PCA_RECIPE)["passthrough"].fit(frame, target)
    model.named_steps["preprocessor"].set_params(use_pca=True)
    for operation in (model[:-1].get_feature_names_out,
                      lambda: api().get_selection_support(model),
                      lambda: api().selection_audit(model)):
        with pytest.raises(ValueError, match="parameter|policy|fit"):
            operation()


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_all_missing_categories_nullable_numeric_and_fresh_unknowns(data, dtype):
    frame, target = data
    frame = frame.copy()
    frame["meter_reading"] = pd.array(frame.meter_reading, dtype="Float64")
    frame["primary_use"] = pd.array([pd.NA] * len(frame), dtype=dtype)
    original, labels = frame.copy(deep=True), target.copy(deep=True)
    model = arms(PCA_RECIPE)["pca"].fit(frame, target)
    heldout = frame.iloc[:4].copy()
    heldout["primary_use"] = pd.array(["new"] * 4, dtype=dtype)
    output = model[:-1].transform(heldout)
    assert output.filter(like="categorical__").eq(0).all().all()
    assert np.isfinite(model.decision_function(heldout)).all()
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(target, labels)


def test_pca_sample_to_feature_ratio_is_not_an_invalid_component_request(data):
    frame, target = data
    expanded = frame.iloc[[0, 2]].copy().assign(floor_count=[1., 2.], year_built=[1990., 2000.])
    model = arms(PCA_RECIPE, tuple(expanded.columns))["pca"].fit(expanded, target.iloc[[0, 2]])
    pca = model.named_steps["preprocessor"].column_transformer_.named_transformers_["numeric"].named_steps["pca"]
    assert pca.n_components_ <= 2
    assert pca.components_.shape[1] == 4


@pytest.mark.parametrize("dtype", ["object", "category"])
def test_categorical_numeric_domain_is_rejected_even_when_values_are_missing(data, dtype):
    frame, target = data
    invalid = frame.copy()
    invalid["primary_use"] = (pd.Categorical([None] * len(frame), categories=[7])
                              if dtype == "category" else np.nan)
    with pytest.raises(TypeError, match="primary_use"):
        arms(PCA_RECIPE)["pca"].fit(invalid, target)


@pytest.mark.parametrize("arm", ["passthrough", "pca"])
def test_multicategory_svc_sparse_fit_and_unseen_inference_preserve_fitted_state(arm, monkeypatch):
    def forbid_densification(*args, **kwargs):
        pytest.fail("combined sparse design matrix must not be densified")

    for matrix_type in (sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array):
        monkeypatch.setattr(matrix_type, "toarray", forbid_densification)
    frame = pd.DataFrame({
        "meter_reading": [-3., -2., 2., 3.],
        "square_feet": [100., 200., 100., 200.],
        "primary_use": ["a", "b", "a", "b"],
    }, index=pd.Index([12, 7, 12, 3], name="source_row"))
    target = pd.Series([0, 0, 1, 1], index=frame.index)
    original, original_target = frame.copy(deep=True), target.copy(deep=True)
    model = arms(PCA_RECIPE, tuple(frame.columns))[arm]
    model.fit(frame, target)

    pre = model.named_steps["preprocessor"]
    training_output = pre.transform(frame)
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in training_output.dtypes)
    assert all(dtype.fill_value == 0 for dtype in training_output.dtypes)
    assert training_output.index.equals(frame.index)
    assert training_output.columns.tolist() == model[:-1].get_feature_names_out().tolist()
    numeric = pre.column_transformer_.named_transformers_["numeric"]
    encoder = pre.column_transformer_.named_transformers_["categorical"].named_steps["one_hot"]
    medians = numeric.named_steps["imputer"].statistics_.copy()
    means = numeric.named_steps["scaler"].mean_.copy()
    scales = numeric.named_steps["scaler"].scale_.copy()
    categories = [values.copy() for values in encoder.categories_]
    components = numeric.named_steps["pca"].components_.copy() if arm == "pca" else None
    names = pre.get_feature_names_out().copy()

    heldout = pd.DataFrame({
        "meter_reading": [-1000., 0., 1000.],
        "square_feet": [300., 150., 10.],
        "primary_use": ["unseen", "a", "b"],
    }, index=[40, 41, 42])
    heldout_original = heldout.copy(deep=True)
    encoded = pre.transform(heldout)
    assert encoded.filter(like="categorical__").iloc[0].eq(0).all()
    assert len(model.predict(heldout)) == 3
    scores = model.decision_function(heldout)
    assert scores.shape == (3,) and np.isfinite(scores).all()
    np.testing.assert_array_equal(model.decision_function(heldout), scores)

    # Compare against the library's supported CSR path, independently of our boundary adapter.
    reference = SVC(**model.named_steps["model"].get_params()).fit(
        training_output.sparse.to_coo().tocsr(), target)
    np.testing.assert_allclose(reference.decision_function(encoded.sparse.to_coo().tocsr()), scores)
    assert model.named_steps["model"]._sparse is True
    np.testing.assert_array_equal(model.named_steps["model"].feature_names_in_, names)
    assert model.named_steps["model"]._effective_probability is False
    assert not hasattr(model, "predict_proba")
    np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_, medians)
    np.testing.assert_array_equal(numeric.named_steps["scaler"].mean_, means)
    np.testing.assert_array_equal(numeric.named_steps["scaler"].scale_, scales)
    for actual, expected in zip(encoder.categories_, categories, strict=True):
        np.testing.assert_array_equal(actual, expected)
    if components is not None:
        np.testing.assert_array_equal(numeric.named_steps["pca"].components_, components)
    np.testing.assert_array_equal(pre.get_feature_names_out(), names)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(target, original_target)
    pd.testing.assert_frame_equal(heldout, heldout_original)
    assert api().selection_audit(model)["probability_evaluation_status"] == "blocked_pending_fold_local_calibration"
    assert not hasattr(clone(model).named_steps["model"], "support_vectors_")
