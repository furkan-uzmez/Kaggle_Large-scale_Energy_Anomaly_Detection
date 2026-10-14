"""Synthetic contracts for the honest cross-fitted target-encoding candidate stage.

Every numeric expectation is a hand-computed literal, so the encoder is never
used as its own oracle. Whole-building exclusion is checked against explicitly
expected held-out building identities, and label perturbations prove that an
inner fold's encodings depend only on that fold's own training groups.

The generated 16-row fixture gives eight buildings of two rows each with one
positive and one negative label per building, so every whole-building subset
has prior exactly 0.5. With ``n_splits=2`` the deterministic inner split holds
out buildings 00-03 (rows 0-7) first and buildings 04-07 (rows 8-15) second.
"""

import importlib
import importlib.util
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from candidate_fixtures import CORE_COLUMNS, binary_target, core_frame
from project.experiments.candidate_contracts import BlockedExperimentError
from project.experiments.inner_cv import build_inner_folds, inner_membership_digest
from project.features.availability import (
    select_predictor_columns,
    validate_feature_availability,
)
from project.features.registry import (
    AvailabilityStatus,
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)
from project.features.transformers import make_imputation_preprocessor

OUTPUT_NAME = "primary_use_target_mean"
SOURCE_COLUMN = "primary_use"

# Explicit per-row categories for the 16-row fixture. Labels are index % 2.
USES = (
    "office", "warehouse", None, "retail", "retail", "retail", "office", "school",
    "retail", "office", "school", "office", "retail", "school", "office", "retail",
)

# Hand-computed inner out-of-fold encodings for smoothing=2, n_splits=2.
#
# Fold 0 trains on rows 8-15 (prior 0.5): office sum_y=2 count=3 -> (2+2*.5)/(3+2)=0.6;
# retail sum_y=1 count=3 -> (1+1)/5=0.4; school sum_y=1 count=2 -> (1+1)/4=0.5;
# "warehouse" is unseen and row 2 is missing, so both fall back to the prior 0.5.
# Fold 1 trains on rows 0-7 (prior 0.5): office sum_y=0 count=2 -> (0+1)/4=0.25;
# retail sum_y=2 count=3 -> (2+1)/5=0.6; school sum_y=1 count=1 -> (1+1)/3=2/3.
CROSS_FITTED = (
    0.6, 0.5, 0.5, 0.4, 0.4, 0.4, 0.6, 0.5,
    0.6, 0.25, 2 / 3, 0.25, 0.6, 2 / 3, 0.25, 0.6,
)

# Hand-computed full supplied-training map for smoothing=2 over all 16 rows
# (prior 0.5): office sum_y=2 count=5 -> (2+1)/7=3/7; warehouse sum_y=1 count=1
# -> (1+1)/3=2/3; retail sum_y=3 count=6 -> (3+1)/8=0.5; school sum_y=2 count=3
# -> (2+1)/5=0.6; the missing row falls back to the prior 0.5.
FULL_MAP = (
    3 / 7, 2 / 3, 0.5, 0.5, 0.5, 0.5, 3 / 7, 0.6,
    0.5, 3 / 7, 0.6, 3 / 7, 0.5, 0.6, 3 / 7, 0.5,
)

# Deterministic held-out building identities of the inner folds for this fixture.
EXCLUDED_TWO = (
    ("building_00", "building_01", "building_02", "building_03"),
    ("building_04", "building_05", "building_06", "building_07"),
)
EXCLUDED_THREE = (
    ("building_01", "building_03", "building_07"),
    ("building_02", "building_05", "building_06"),
    ("building_00", "building_04"),
)
FOLD_ZERO_ROWS = tuple(range(8))
FOLD_ONE_ROWS = tuple(range(8, 16))


def api():
    name = "project.features.target_encoding"
    assert importlib.util.find_spec(name), "cross-fitted target-encoding API is missing"
    return importlib.import_module(name)


def make(**parameters):
    return api().GroupCrossFittedTargetEncoder(**parameters)


def fixture(uses=USES):
    """Return the generated core frame with explicit categories plus its metadata."""
    frame = core_frame()
    frame[SOURCE_COLUMN] = pd.Series(list(uses), index=frame.index, dtype=object)
    return frame, binary_target(frame), frame.attrs["groups"]


def encoded(output):
    return output[OUTPUT_NAME].to_numpy(dtype=float)


def fitted_state(encoder):
    """Normalize every public fitted attribute so snapshots compare by value."""
    state = {}
    for name, value in sorted(vars(encoder).items()):
        if not name.endswith("_") or name.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            state[name] = value.tolist()
        elif isinstance(value, FeatureRegistry):
            state[name] = value.names
        elif isinstance(value, dict):
            state[name] = sorted(value.items())
        else:
            state[name] = value
    return state


# --- Hand-computed cross-fitted mathematics ------------------------------------


def test_plan_hand_computed_category_prior_and_fallbacks():
    """sum_y=2, count=3, prior=.5, smoothing=2 -> .6; missing and unseen -> .5."""
    frame, target, groups = fixture()
    output = make(smoothing=2.0, n_splits=2).fit_transform(frame, target, groups=groups)
    values = encoded(output)
    assert values[0] == pytest.approx(0.6)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(0.5)
    np.testing.assert_allclose(values, CROSS_FITTED)


def test_default_smoothing_shrinks_every_category_towards_the_inner_prior():
    frame, target, groups = fixture()
    values = encoded(make(smoothing=1000.0, n_splits=2).fit_transform(frame, target, groups=groups))
    np.testing.assert_allclose(values, np.full(16, 0.5), atol=5e-3)
    assert not np.allclose(values, np.full(16, 0.5), atol=1e-12)


def test_smoothing_zero_limit_is_rejected_so_counts_cannot_dominate_alone():
    frame, target, groups = fixture()
    with pytest.raises(ValueError, match="smoothing"):
        make(smoothing=0.0).fit_transform(frame, target, groups=groups)


# --- Whole-building exclusion and label isolation ------------------------------


@pytest.mark.parametrize(
    ("n_splits", "expected"), [(2, EXCLUDED_TWO), (3, EXCLUDED_THREE)]
)
def test_recorded_inner_membership_matches_expected_excluded_buildings(n_splits, expected):
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=n_splits)
    encoder.fit_transform(frame, target, groups=groups)
    assert encoder.inner_excluded_groups_ == expected
    assert all(
        not set(a) & set(b)
        for index, a in enumerate(expected)
        for b in expected[index + 1:]
    )
    assert encoder.inner_membership_digest_ == inner_membership_digest(
        build_inner_folds(target, groups, n_splits=n_splits), groups
    )


def test_perturbing_a_held_out_buildings_labels_cannot_change_that_folds_encodings():
    """Fold 0 holds out building_00; its encodings depend only on fold 0 training."""
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2)
    base = encoded(encoder.fit_transform(frame, target, groups=groups))
    assert encoder.inner_excluded_groups_[0] == EXCLUDED_TWO[0]
    assert groups.iloc[0] == groups.iloc[1] == "building_00"

    perturbed_target = target.copy()
    perturbed_target.iloc[0], perturbed_target.iloc[1] = 1, 0
    perturbed_encoder = make(smoothing=2.0, n_splits=2)
    perturbed = encoded(
        perturbed_encoder.fit_transform(frame, perturbed_target, groups=groups)
    )

    assert perturbed_encoder.inner_excluded_groups_ == EXCLUDED_TWO
    np.testing.assert_allclose(
        perturbed[list(FOLD_ZERO_ROWS)], base[list(FOLD_ZERO_ROWS)]
    )
    assert not np.allclose(perturbed[list(FOLD_ONE_ROWS)], base[list(FOLD_ONE_ROWS)])


def test_own_row_label_never_enters_its_own_encoding():
    """Row 1's unique "warehouse" category is held out with its whole building."""
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2)
    cross_fitted = encoded(encoder.fit_transform(frame, target, groups=groups))
    assert [index for index, use in enumerate(USES) if use == "warehouse"] == [1]
    assert cross_fitted[1] == pytest.approx(0.5)
    assert encoder.full_training_map_["warehouse"] == pytest.approx(2 / 3)


def test_full_map_inference_differs_from_the_cross_fitted_training_output():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2)
    cross_fitted = encoded(encoder.fit_transform(frame, target, groups=groups))
    inference = encoded(encoder.transform(frame))
    np.testing.assert_allclose(cross_fitted, CROSS_FITTED)
    np.testing.assert_allclose(inference, FULL_MAP)
    assert not np.allclose(cross_fitted, inference)
    assert encoder.training_prior_ == pytest.approx(0.5)
    assert encoder.cross_fitted_ is True


def test_plain_fit_stores_only_the_full_map_and_records_a_non_cross_fitted_path():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2).fit(frame, target, groups=groups)
    assert encoder.cross_fitted_ is False
    assert not hasattr(encoder, "inner_excluded_groups_")
    assert not hasattr(encoder, "inner_membership_digest_")
    np.testing.assert_allclose(encoded(encoder.transform(frame)), FULL_MAP)
    assert sorted(encoder.full_training_map_) == ["office", "retail", "school", "warehouse"]


def test_altering_held_out_labels_and_categories_cannot_change_fitted_state():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2)
    encoder.fit_transform(frame, target, groups=groups)
    before = fitted_state(encoder)
    first = encoded(encoder.transform(frame))

    unseen = frame.copy(deep=True)
    unseen[SOURCE_COLUMN] = pd.Series(["datacenter"] * 16, index=frame.index, dtype=object)
    held_out = encoded(encoder.transform(unseen))

    assert fitted_state(encoder) == before
    np.testing.assert_allclose(held_out, np.full(16, 0.5))
    np.testing.assert_allclose(encoded(encoder.transform(frame)), first)


# --- Fitting support and class/index failure modes -----------------------------


def test_one_class_training_support_is_an_explicit_blocker():
    frame, _target, groups = fixture()
    for constant in (0, 1):
        ones = pd.Series(np.full(16, constant, dtype=np.int64), index=frame.index)
        with pytest.raises(BlockedExperimentError) as error:
            make(n_splits=2).fit_transform(frame, ones, groups=groups)
        assert error.value.code == "one_class_inner_partition"
        with pytest.raises(BlockedExperimentError):
            make(n_splits=2).fit(frame, ones, groups=groups)


def test_insufficient_group_support_blocks_instead_of_degrading_isolation():
    frame, target, _groups = fixture()
    single = pd.Series(["building_00"] * 16, index=frame.index, name="building_id")
    with pytest.raises(BlockedExperimentError) as error:
        make(n_splits=2).fit_transform(frame, target, groups=single)
    assert error.value.code == "insufficient_inner_groups"


@pytest.mark.parametrize("n_splits", [2, 3, 4, 5])
def test_every_training_row_is_encoded_exactly_once_for_each_fold_count(n_splits):
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=n_splits)
    values = encoded(encoder.fit_transform(frame, target, groups=groups))
    assert np.isfinite(values).all()
    assert len(encoder.inner_excluded_groups_) == n_splits
    flattened = [group for fold in encoder.inner_excluded_groups_ for group in fold]
    assert sorted(flattened) == sorted(set(groups.tolist()))


def test_misaligned_or_duplicated_identities_are_rejected():
    frame, target, groups = fixture()
    shifted = target.copy()
    shifted.index = pd.Index(np.arange(100, 116), name="row_index")
    with pytest.raises(ValueError, match="index"):
        make(n_splits=2).fit_transform(frame, shifted, groups=groups)
    with pytest.raises(ValueError, match="index"):
        make(n_splits=2).fit_transform(frame, target, groups=shifted.astype(str))
    with pytest.raises(ValueError, match="index"):
        make(n_splits=2).fit_transform(frame, target.iloc[:8], groups=groups)
    duplicated = frame.copy(deep=True)
    duplicated.index = pd.Index([0] * 16, name="row_index")
    with pytest.raises(ValueError, match="index"):
        make(n_splits=2).fit_transform(
            duplicated, target.set_axis(duplicated.index), groups=groups.set_axis(duplicated.index)
        )


@pytest.mark.parametrize(
    "labels",
    [
        pd.Series(np.tile([0.0, 1.0], 8)),
        pd.Series(np.tile([False, True], 8)),
        pd.Series(np.tile([0, 2], 8)),
        pd.Series([None, 1] * 8, dtype="object"),
    ],
)
def test_training_labels_must_be_nonboolean_binary_integers(labels):
    frame, _target, groups = fixture()
    labels = labels.set_axis(frame.index)
    with pytest.raises((ValueError, TypeError)):
        make(n_splits=2).fit_transform(frame, labels, groups=groups)


def test_group_metadata_must_be_a_nonnull_consistently_typed_series():
    frame, target, groups = fixture()
    for invalid in (
        groups.to_numpy(),
        groups.mask(groups == "building_00"),
        pd.Series([*["building_00"] * 15, 7], index=frame.index),
    ):
        with pytest.raises((ValueError, TypeError)):
            make(n_splits=2).fit_transform(frame, target, groups=invalid)
    with pytest.raises(TypeError):
        make(n_splits=2).fit_transform(frame, target)


# --- Parameter bounds ----------------------------------------------------------


@pytest.mark.parametrize(
    "smoothing", [0.0, -1.0, -0.5, 1000.1, 2000.0, np.nan, np.inf, True, False, "10", None, 1j]
)
def test_smoothing_must_be_finite_positive_and_bounded(smoothing):
    frame, target, groups = fixture()
    with pytest.raises((ValueError, TypeError), match="smoothing"):
        make(smoothing=smoothing, n_splits=2).fit_transform(frame, target, groups=groups)


@pytest.mark.parametrize("n_splits", [1, 0, -2, 6, 100, True, False, 3.0, "3", None])
def test_n_splits_must_be_an_integer_inside_the_inner_cv_bounds(n_splits):
    frame, target, groups = fixture()
    with pytest.raises((ValueError, TypeError), match="n_splits"):
        make(n_splits=n_splits).fit_transform(frame, target, groups=groups)


@pytest.mark.parametrize("seed", [-1, 2**32, True, False, 42.0, "42", None])
def test_seed_must_be_an_integer_inside_the_declared_range(seed):
    frame, target, groups = fixture()
    with pytest.raises((ValueError, TypeError), match="seed"):
        make(n_splits=2, seed=seed).fit_transform(frame, target, groups=groups)


@pytest.mark.parametrize("max_categories", [0, -1, 4097, 10000, True, False, 8.0, "8", None])
def test_max_categories_must_be_an_integer_inside_the_versioned_cap(max_categories):
    frame, target, groups = fixture()
    with pytest.raises((ValueError, TypeError), match="max_categories"):
        make(n_splits=2, max_categories=max_categories).fit_transform(
            frame, target, groups=groups
        )


def test_observed_category_cardinality_is_capped_before_any_map_is_learned():
    frame, target, groups = fixture()
    with pytest.raises(ValueError, match="cardinality|max_categories"):
        make(n_splits=2, max_categories=3).fit_transform(frame, target, groups=groups)
    with pytest.raises(ValueError, match="cardinality|max_categories"):
        make(n_splits=2, max_categories=3).fit(frame, target, groups=groups)
    accepted = make(smoothing=2.0, n_splits=2, max_categories=4)
    np.testing.assert_allclose(
        encoded(accepted.fit_transform(frame, target, groups=groups)), CROSS_FITTED
    )


def test_identical_seeds_and_parameters_reproduce_identical_encodings():
    frame, target, groups = fixture()
    first = make(smoothing=2.0, n_splits=3, seed=7)
    second = make(smoothing=2.0, n_splits=3, seed=7)
    np.testing.assert_allclose(
        encoded(first.fit_transform(frame, target, groups=groups)),
        encoded(second.fit_transform(frame, target, groups=groups)),
    )
    assert first.inner_membership_digest_ == second.inner_membership_digest_
    other = make(smoothing=2.0, n_splits=3, seed=11)
    other.fit_transform(frame, target, groups=groups)
    assert isinstance(other.inner_membership_digest_, str)


# --- Schema, dtype and frame contracts ----------------------------------------


def test_raw_predictors_are_preserved_and_no_input_is_mutated():
    frame, target, groups = fixture()
    original, labels, metadata = frame.copy(deep=True), target.copy(), groups.copy()
    output = make(smoothing=2.0, n_splits=2).fit_transform(frame, target, groups=groups)
    assert output.shape == (16, len(CORE_COLUMNS) + 1)
    assert output.columns.tolist() == [*CORE_COLUMNS, OUTPUT_NAME]
    assert output.index.equals(frame.index)
    assert output[OUTPUT_NAME].dtype == np.dtype("float64")
    pd.testing.assert_frame_equal(output[list(CORE_COLUMNS)], original)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(target, labels)
    pd.testing.assert_series_equal(groups, metadata)
    output.iloc[0, 0] = 999.0
    output.loc[:, OUTPUT_NAME] = -1.0
    pd.testing.assert_frame_equal(frame, original)


def test_subset_pools_keep_their_ordered_schema_and_require_primary_use():
    frame, target, groups = fixture()
    subset = frame[["hour", SOURCE_COLUMN, "meter_reading"]]
    encoder = make(smoothing=2.0, n_splits=2).fit(subset, target, groups=groups)
    output = encoder.transform(subset)
    assert output.columns.tolist() == ["hour", SOURCE_COLUMN, "meter_reading", OUTPUT_NAME]
    assert encoder.n_features_in_ == 3
    with pytest.raises(ValueError, match=SOURCE_COLUMN):
        make(n_splits=2).fit(subset.drop(columns=SOURCE_COLUMN), target, groups=groups)
    with pytest.raises(ValueError, match="schema"):
        encoder.transform(subset[[SOURCE_COLUMN, "hour", "meter_reading"]])
    with pytest.raises(ValueError, match="schema"):
        encoder.transform(subset.drop(columns="hour"))


def test_empty_transform_returns_the_typed_fitted_schema():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2).fit(frame, target, groups=groups)
    empty = encoder.transform(frame.iloc[:0])
    assert len(empty) == 0
    assert empty.columns.tolist() == [*CORE_COLUMNS, OUTPUT_NAME]
    assert empty[OUTPUT_NAME].dtype == np.dtype("float64")
    with pytest.raises((ValueError, BlockedExperimentError)):
        make(n_splits=2).fit(frame.iloc[:0], target.iloc[:0], groups=groups.iloc[:0])


@pytest.mark.parametrize(
    "column", ["anomaly", "target", "building_id", "row_id", "gte_probe", "unknown_source"]
)
def test_forbidden_or_unknown_predictor_columns_are_rejected(column):
    frame, target, groups = fixture()
    with pytest.raises(ValueError, match="schema|predictor|column"):
        make(n_splits=2).fit(frame.assign(**{column: 1.0}), target, groups=groups)


def test_primary_use_must_hold_strings_or_missing_values():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2).fit(frame, target, groups=groups)
    for value in (1.0, True, pd.Timestamp("2024-01-01")):
        broken = frame.assign(**{SOURCE_COLUMN: value})
        with pytest.raises(TypeError, match=SOURCE_COLUMN):
            make(n_splits=2).fit(broken, target, groups=groups)
        with pytest.raises(TypeError, match=SOURCE_COLUMN):
            encoder.transform(broken)
    with pytest.raises(TypeError, match="DataFrame"):
        make(n_splits=2).fit(frame.to_numpy(), target, groups=groups)
    duplicated = pd.DataFrame(
        frame[[SOURCE_COLUMN, "hour", "hour"]].to_numpy(),
        columns=[SOURCE_COLUMN, "hour", "hour"],
    )
    with pytest.raises(ValueError, match="unique"):
        make(n_splits=2).fit(duplicated, target, groups=groups)


def test_nullable_and_categorical_category_storage_match_object_storage():
    frame, target, groups = fixture()
    expected = make(smoothing=2.0, n_splits=2)
    baseline = encoded(expected.fit_transform(frame, target, groups=groups))
    for storage in ("string", pd.CategoricalDtype(sorted({u for u in USES if u}))):
        variant = frame.assign(**{SOURCE_COLUMN: frame[SOURCE_COLUMN].astype(storage)})
        encoder = make(smoothing=2.0, n_splits=2)
        np.testing.assert_allclose(
            encoded(encoder.fit_transform(variant, target, groups=groups)), baseline
        )
        assert sorted(encoder.full_training_map_) == sorted(expected.full_training_map_)


# --- sklearn estimator contracts ----------------------------------------------


def test_fitted_checks_cloning_and_parameter_change_detection():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2, seed=7, max_categories=64)
    for operation in (lambda: encoder.transform(frame), encoder.get_feature_names_out):
        with pytest.raises(NotFittedError):
            operation()
    encoder.fit_transform(frame, target, groups=groups)
    assert encoder.get_params() == {
        "smoothing": 2.0, "n_splits": 2, "seed": 7, "max_categories": 64,
    }
    duplicate = clone(encoder)
    assert not hasattr(duplicate, "output_features_")
    duplicate.fit_transform(frame, target, groups=groups)
    assert fitted_state(duplicate) == fitted_state(encoder)
    assert encoder.get_feature_names_out().tolist() == [*CORE_COLUMNS, OUTPUT_NAME]
    with pytest.raises(ValueError, match="input_features"):
        encoder.get_feature_names_out([SOURCE_COLUMN])
    for change in ({"smoothing": 10.0}, {"n_splits": 3}, {"seed": 42}, {"max_categories": 32}):
        changed = clone(encoder)
        changed.fit_transform(frame, target, groups=groups)
        changed.set_params(**change)
        with pytest.raises(ValueError, match="parameter"):
            changed.transform(frame)
        with pytest.raises(ValueError, match="parameter"):
            changed.get_feature_names_out()


def test_transform_accepts_predictors_only_and_never_refits():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2)
    encoder.fit_transform(frame, target, groups=groups)
    before = fitted_state(encoder)
    with pytest.raises(TypeError):
        encoder.transform(frame, target)
    with pytest.raises(TypeError):
        encoder.transform(frame, groups=groups)
    with pytest.raises(TypeError):
        encoder.transform(frame, y=target)
    encoder.transform(frame)
    assert fitted_state(encoder) == before


def test_pipeline_fit_routes_groups_to_the_cross_fitted_fit_transform():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2)
    pipeline = Pipeline([("features", encoder), ("preprocess", make_imputation_preprocessor())])
    matrix = pipeline.fit_transform(frame, target, features__groups=groups)
    assert matrix.shape[0] == 16
    assert len(pipeline.get_feature_names_out()) == matrix.shape[1]
    assert pipeline.named_steps["features"].cross_fitted_ is True
    np.testing.assert_allclose(
        encoded(pipeline.named_steps["features"].transform(frame)), FULL_MAP
    )
    assert pipeline.transform(frame.iloc[[0]]).shape[0] == 1
    with pytest.raises(TypeError):
        Pipeline(
            [("features", make(smoothing=2.0, n_splits=2)),
             ("preprocess", make_imputation_preprocessor())]
        ).fit(frame, target)


# --- Registry metadata and honest leakage declaration --------------------------


def test_registry_declares_target_derived_outer_train_primary_use_lineage():
    registry = api().build_target_encoding_registry()
    metadata = registry.get(OUTPUT_NAME)
    assert metadata.source_columns == (SOURCE_COLUMN,)
    assert metadata.target_dependence is TargetDependence.TARGET_DERIVED
    assert metadata.target_dependence is not TargetDependence.INDEPENDENT
    assert metadata.transform_fit_boundary is TransformFitBoundary.OUTER_TRAIN
    assert metadata.provenance_status is ProvenanceStatus.VERIFIED
    assert metadata.prediction_time_availability is AvailabilityStatus.AVAILABLE
    assert "prior" in metadata.fallback_behavior.casefold()
    assert registry.names == (*build_lead_baseline_registry().names, OUTPUT_NAME)


def test_generic_availability_guards_still_refuse_the_target_derived_feature():
    registry = api().build_target_encoding_registry()
    prediction_time = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="target-dependent"):
        validate_feature_availability(registry.get(OUTPUT_NAME), prediction_time)
    with pytest.raises(ValueError, match="target-dependent"):
        select_predictor_columns([SOURCE_COLUMN, OUTPUT_NAME], registry, prediction_time)


def test_unsafe_baseline_source_metadata_is_never_promoted(monkeypatch):
    frame, target, groups = fixture()
    baseline = build_lead_baseline_registry()
    unsafe = FeatureRegistry(
        replace(baseline.get(name), source_columns=("anomaly",)) if name == SOURCE_COLUMN
        else baseline.get(name)
        for name in baseline.names
    )
    monkeypatch.setattr(api(), "build_lead_baseline_registry", lambda: unsafe)
    with pytest.raises(ValueError, match="source|lineage"):
        api().build_target_encoding_registry()
    with pytest.raises(ValueError, match="source|lineage"):
        make(n_splits=2).fit(frame, target, groups=groups)


def test_declared_names_and_bounds_match_the_reviewed_design():
    module = api()
    assert module.OUTPUT_NAME == OUTPUT_NAME
    assert module.SOURCE_COLUMN == SOURCE_COLUMN
    assert module.SMOOTHING_BOUNDS == (0.0, 1000.0)
    assert module.MAX_CATEGORIES_BOUNDS == (1, 4096)


def test_plain_fit_accepts_full_map_without_inner_fold_group_support():
    frame, target, groups = fixture()
    frame, target, groups = frame.iloc[:4], target.iloc[:4], groups.iloc[:4]
    encoder = make(smoothing=2.0).fit(frame, target, groups=groups)
    assert encoder.cross_fitted_ is False
    np.testing.assert_allclose(encoded(encoder.transform(frame)), [1 / 3, 2 / 3, .5, 2 / 3])
    with pytest.raises(BlockedExperimentError, match="groups"):
        make(smoothing=2.0).fit_transform(frame, target, groups=groups)


def test_successful_plain_refit_clears_previous_cross_fit_evidence():
    frame, target, groups = fixture()
    encoder = make(smoothing=2.0, n_splits=2)
    encoder.fit_transform(frame, target, groups=groups)
    encoder.fit(frame, target, groups=groups)
    assert encoder.cross_fitted_ is False
    assert not hasattr(encoder, "inner_excluded_groups_")
    assert not hasattr(encoder, "inner_membership_digest_")
    np.testing.assert_allclose(encoded(encoder.transform(frame)), FULL_MAP)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_failed_refit_cannot_retain_stale_audited_fit(method):
    frame, target, groups = fixture()
    encoder = make(n_splits=2)
    encoder.fit_transform(frame, target, groups=groups)
    with pytest.raises(ValueError):
        getattr(encoder, method)(frame, target.iloc[:2], groups=groups)
    with pytest.raises(NotFittedError):
        encoder.transform(frame)
    assert not hasattr(encoder, "inner_membership_digest_")
    assert not hasattr(encoder, "cross_fitted_")


@pytest.mark.parametrize("index", [[0] * 16, [*range(15), None]])
def test_transform_rejects_duplicate_and_null_row_identity(index):
    frame, target, groups = fixture()
    encoder = make(n_splits=2).fit(frame, target, groups=groups)
    with pytest.raises(ValueError, match="index"):
        encoder.transform(frame.set_axis(index))


@pytest.mark.parametrize("column", [name for name in CORE_COLUMNS if name != SOURCE_COLUMN])
@pytest.mark.parametrize("bad", [np.inf, -np.inf, True, 1j, "1"])
def test_every_supplied_baseline_numeric_predictor_is_validated(column, bad):
    frame, target, groups = fixture()
    encoder = make(n_splits=2).fit(frame, target, groups=groups)
    broken = frame.assign(**{column: bad})
    with pytest.raises((TypeError, ValueError), match=column):
        encoder.transform(broken)
    for method in ("fit", "fit_transform"):
        with pytest.raises((TypeError, ValueError), match=column):
            getattr(make(n_splits=2), method)(broken, target, groups=groups)


def test_nullable_numeric_heldout_batches_preserve_state_and_input():
    frame, target, groups = fixture()
    frame = frame.assign(square_feet=frame.square_feet.astype("Float64"))
    encoder = make(n_splits=2).fit(frame, target, groups=groups)
    before = fitted_state(encoder)
    heldout = frame.iloc[:2].copy()
    heldout["square_feet"] = pd.Series([pd.NA, 1e300], index=heldout.index, dtype="Float64")
    original = heldout.copy(deep=True)
    together = encoder.transform(heldout)
    alone = encoder.transform(heldout.iloc[:1])
    pd.testing.assert_frame_equal(together.iloc[:1], alone)
    pd.testing.assert_frame_equal(heldout, original)
    assert fitted_state(encoder) == before


def test_unique_nonnull_compound_row_identity_survives_cross_fit_and_transform():
    frame, target, groups = fixture()
    index = pd.MultiIndex.from_tuples([(7, position) for position in range(16)])
    frame, target, groups = frame.set_axis(index), target.set_axis(index), groups.set_axis(index)
    encoder = make(smoothing=2.0, n_splits=2)
    output = encoder.fit_transform(frame, target, groups=groups)
    assert output.index.equals(index)
    np.testing.assert_allclose(encoded(output), CROSS_FITTED)
    np.testing.assert_allclose(encoded(encoder.transform(frame)), FULL_MAP)


@pytest.mark.parametrize("method", ["fit", "fit_transform", "transform"])
def test_repeated_identity_level_names_support_encoder_fitting_and_transform(method):
    index = pd.MultiIndex.from_tuples(
        [("a", 0), ("a", 1), ("b", 0), ("b", 1)], names=["identity", "identity"]
    )
    frame = pd.DataFrame({SOURCE_COLUMN: ["a", "a", "b", None]}, index=index)
    target = pd.Series([0, 1, 0, 1], index=index, dtype="int64")
    groups = pd.Series([0, 0, 1, 1], index=index)
    encoder = make(smoothing=2.0, n_splits=2)
    if method == "transform":
        distinct = index.set_names(["entity", "row"])
        encoder.fit(frame.set_axis(distinct), target.set_axis(distinct), groups=groups.set_axis(distinct))
    else:
        getattr(encoder, method)(frame, target, groups=groups)
    output = encoder.transform(frame)
    assert output.index.equals(index)
    assert output.index.names == ["identity", "identity"]
    np.testing.assert_allclose(encoded(output), [.5, .5, 1 / 3, .5])
    assert encoder.transform(frame.iloc[:0]).index.names == ["identity", "identity"]
    for tuples in (
        [("a", 0), ("a", 0), ("b", 0), ("b", 1)],
        [("a", 0), ("a", None), ("b", 0), ("b", 1)],
    ):
        invalid = frame.set_axis(pd.MultiIndex.from_tuples(tuples, names=index.names))
        with pytest.raises(ValueError, match="index"):
            encoder.transform(invalid)
        if method != "transform":
            with pytest.raises(ValueError, match="index"):
                getattr(make(n_splits=2), method)(
                    invalid, target.set_axis(invalid.index), groups=groups.set_axis(invalid.index)
                )
