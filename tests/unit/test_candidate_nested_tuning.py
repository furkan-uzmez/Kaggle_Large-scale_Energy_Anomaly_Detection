"""Task 9 bounded HGB nested tuning and SQLite study evidence tests."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from candidate_fixtures import (
    CORE_COLUMNS,
    binary_target,
    core_frame,
    synthetic_partitions,
)
from project.experiments import nested_tuning
from project.experiments.candidate_contracts import (
    BlockedExperimentError,
    FitBudget,
    FitCounter,
    StudySpec,
    TrainingPartition,
    training_view,
)
from project.experiments.inner_cv import build_inner_folds, inner_membership_digest
from project.experiments.nested_tuning import (
    CATEGORICAL_FEATURES,
    CORE_FEATURE_POOL,
    NUMERIC_FEATURES,
    TuningResult,
    _validate_constructed_arm,
    build_hgb_tuning_arm,
    build_study_manifest_payload,
    compute_study_manifest_hash,
    required_model_fits,
    resolve_campaign_root,
    tune_outer_training,
)
from project.models.candidate_factories import (
    _CandidateArm,
    _pipeline,
    _PredictorSchema,
)


@pytest.fixture
def tiny_training_partition() -> TrainingPartition:
    """Fixture providing an authoritative 16-row training partition with 4 groups."""
    partitions = synthetic_partitions()
    return training_view(partitions[0])


@pytest.fixture
def small_study_spec() -> StudySpec:
    """Minimal 2-trial, 2-inner-fold StudySpec within allowed HGB space."""
    return StudySpec(
        n_trials=2,
        inner_folds=2,
        seed=42,
        parameter_lists=(
            ("learning_rate", (0.03, 0.1)),
            ("max_leaf_nodes", (15, 31)),
            ("max_iter", (100,)),
        ),
    )


def test_hpo_fit_reservation_is_explicit():
    """Formula: T * K + 2 (+ K if score_inner_default)."""
    assert required_model_fits(n_trials=2, inner_folds=2, score_inner_default=False) == 6
    assert required_model_fits(n_trials=2, inner_folds=2, score_inner_default=True) == 8
    assert required_model_fits(n_trials=3, inner_folds=3, score_inner_default=False) == 11
    assert required_model_fits(n_trials=3, inner_folds=3, score_inner_default=True) == 14

    with pytest.raises(ValueError, match="n_trials"):
        required_model_fits(n_trials=True, inner_folds=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_trials"):
        required_model_fits(n_trials=0, inner_folds=2)
    with pytest.raises(ValueError, match="inner_folds"):
        required_model_fits(n_trials=2, inner_folds=1)
    with pytest.raises(ValueError, match="inner_folds"):
        required_model_fits(n_trials=2, inner_folds=6)
    with pytest.raises(TypeError, match="score_inner_default"):
        required_model_fits(n_trials=2, inner_folds=2, score_inner_default="true")  # type: ignore[arg-type]


def test_campaign_shared_root_blocks_second_owner(tmp_path, small_study_spec):
    budget = FitBudget(6, 4, 60.0, 1024**3)
    first, second = [training_view(p) for p in synthetic_partitions()]
    tune_outer_training(first, small_study_spec, build_hgb_tuning_arm, budget,
                        tmp_path / "first", campaign_root=tmp_path)
    with pytest.raises(BlockedExperimentError, match="reservation"):
        tune_outer_training(second, small_study_spec, build_hgb_tuning_arm, budget,
                            tmp_path / "second", campaign_root=tmp_path)


def test_campaign_cached_restores_counter_without_replaying(tmp_path, tiny_training_partition, small_study_spec):
    budget = FitBudget(6, 2, 60.0, 1024**3)
    counter = FitCounter(budget)
    tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, counter, tmp_path)
    tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, counter, tmp_path)
    assert (counter.used_model_fits, counter.reserved_model_fits, counter.reserved_trials) == (4, 6, 2)
    fresh = FitCounter(budget)
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, fresh, tmp_path)
    assert (fresh.used_model_fits, fresh.reserved_model_fits, fresh.reserved_trials) == (4, 6, 2)
    assert result.invocation_model_fits == 0


def test_factory_name_does_not_grant_permission(tmp_path, tiny_training_partition, small_study_spec):
    def failing_builder(params):
        return build_hgb_tuning_arm(params)

    with pytest.raises(TypeError, match="canonical"):
        tune_outer_training(tiny_training_partition, small_study_spec, failing_builder,
                            FitBudget(6, 2, 60.0, 1024**3), tmp_path)


@pytest.mark.parametrize("cap,fresh", [(6, True), (7, True), (20, False)])
def test_interrupted_owned_unit_reserves_retry_before_construction(
    tmp_path, tiny_training_partition, small_study_spec, monkeypatch, cap, fresh,
):
    counter = FitCounter(FitBudget(cap, 2, 60.0, 1024**3))
    original = _CandidateArm.fit

    def interrupt(arm, X, y=None, **params):
        raise KeyboardInterrupt("interrupted owned fit")

    monkeypatch.setattr(_CandidateArm, "fit", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tune_outer_training(tiny_training_partition, small_study_spec,
                            build_hgb_tuning_arm, counter, tmp_path)
    assert counter.used_model_fits == 1
    monkeypatch.setattr(_CandidateArm, "fit", original)
    resumed = FitCounter(counter.budget) if fresh else counter
    constructed = []
    builder = nested_tuning.build_hgb_tuning_arm

    def spy_builder(params):
        constructed.append(resumed.used_model_fits)
        return builder(params)

    monkeypatch.setattr(nested_tuning, "build_hgb_tuning_arm", spy_builder)
    if cap == 6:
        with pytest.raises(BlockedExperimentError, match="reservation"):
            tune_outer_training(tiny_training_partition, small_study_spec,
                                spy_builder, resumed, tmp_path)
        assert constructed == []
        assert (resumed.used_model_fits, resumed.reserved_model_fits, resumed.reserved_trials) == (1, 6, 2)
        return
    result = tune_outer_training(tiny_training_partition, small_study_spec,
                                 spy_builder, resumed, tmp_path)
    assert constructed == [2, 3, 4, 5]
    assert (resumed.used_model_fits, resumed.reserved_model_fits, resumed.reserved_trials) == (5, 7, 2)
    assert (result.study_charged_model_fits, result.invocation_model_fits,
            result.campaign_total_model_fits, result.owned_reserved_model_fits) == (5, 4, 5, 7)
    for unit in ("outer:default", "outer:selected"):
        attempt = resumed.before_owned_fit(result.owner_id, unit)
        resumed.finish_owned_fit(attempt, "completed")
    assert resumed.used_model_fits == 7


def test_constructor_failure_does_not_create_a_free_fit_token(
    tmp_path, tiny_training_partition, small_study_spec, monkeypatch,
):
    counter = FitCounter(FitBudget(10, 2, 60.0, 1024**3))

    def constructor(params):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(nested_tuning, "build_hgb_tuning_arm", constructor)
    with pytest.raises(BlockedExperimentError, match="no valid candidate"):
        tune_outer_training(tiny_training_partition, small_study_spec, constructor, counter, tmp_path)
    assert counter.used_model_fits == 2
    assert counter._precharged_model_fits == 0
    with pytest.raises(BlockedExperimentError, match="owner"):
        counter.before_fit()
    assert counter.used_model_fits == 2


def test_cached_result_separates_owner_and_campaign_totals(tmp_path, small_study_spec):
    budget = FitBudget(12, 4, 60.0, 1024**3)
    counter = FitCounter(budget)
    first, second = [training_view(p) for p in synthetic_partitions()]
    result = tune_outer_training(first, small_study_spec, build_hgb_tuning_arm, counter, tmp_path)
    for unit in ("outer:default", "outer:selected"):
        attempt = counter.before_owned_fit(result.owner_id, unit)
        counter.finish_owned_fit(attempt, "completed")
    tune_outer_training(second, small_study_spec, build_hgb_tuning_arm, counter, tmp_path)
    cached = tune_outer_training(first, small_study_spec, build_hgb_tuning_arm, FitCounter(budget), tmp_path)
    assert (cached.study_charged_model_fits, cached.invocation_model_fits,
            cached.campaign_total_model_fits, cached.planned_model_fits,
            cached.owned_reserved_model_fits) == (6, 0, 10, 6, 6)


def test_factory_identity_enforced_rejects_lambdas_fakes_and_instances(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """tune_outer_training requires arm_factory is build_hgb_tuning_arm exactly."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    def custom_factory(params):
        return build_hgb_tuning_arm(params)

    for invalid_factory in (
        lambda p: build_hgb_tuning_arm(p),
        custom_factory,
        build_hgb_tuning_arm(),
        None,
    ):
        with pytest.raises(TypeError, match="canonical build_hgb_tuning_arm"):
            tune_outer_training(
                training=tiny_training_partition,
                study_spec=small_study_spec,
                arm_factory=invalid_factory,  # type: ignore[arg-type]
                budget=budget,
                study_dir=tmp_path,
            )


def test_canonical_build_hgb_tuning_arm_validates_parameters():
    """build_hgb_tuning_arm accepts only the four approved tunables within STUDY_SPACE."""
    arm = build_hgb_tuning_arm({"learning_rate": 0.03, "max_leaf_nodes": 15})
    assert isinstance(arm, _CandidateArm)
    assert [name for name, _ in arm.steps] == ["schema", "preprocess", "model"]
    model = arm.named_steps["model"]
    assert model.learning_rate == 0.03
    assert model.max_leaf_nodes == 15
    assert model.l2_regularization == 0.0
    assert model.max_iter == 100
    assert model.early_stopping is False
    assert model.random_state == 42
    assert model.class_weight is None

    # Rejects unknown parameter outside 4 approved tunables
    with pytest.raises(ValueError, match="unapproved parameter"):
        build_hgb_tuning_arm({"n_estimators": 50})

    # Rejects out-of-space value for approved parameter
    with pytest.raises(ValueError, match="must be float in"):
        build_hgb_tuning_arm({"learning_rate": 0.5})

    with pytest.raises(ValueError, match="must be integer in"):
        build_hgb_tuning_arm({"max_iter": 50})

    # Rejects boolean for numeric parameter
    with pytest.raises(TypeError, match="cannot be boolean"):
        build_hgb_tuning_arm({"learning_rate": True})


# ==================== Finding 1: Canonical Structural & Param Validation Tests ====================


def test_validate_constructed_arm_rejects_native_wrong_prep():
    """Arm built with native=True (passthrough numeric instead of median) is rejected."""
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=42, class_weight=None, max_iter=100)
    # native=True uses passthrough numeric
    arm = _pipeline(CORE_FEATURE_POOL, CATEGORICAL_FEATURES, True, model, "nested_hgb_tuning", "natural")
    with pytest.raises((ValueError, TypeError), match="native_numeric|SimpleImputer"):
        _validate_constructed_arm(arm, {"max_iter": 100})


def test_validate_constructed_arm_rejects_mean_imputation():
    """Arm built with mean numeric imputation instead of median is rejected."""
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transforms = [
        ("numeric", SimpleImputer(strategy="mean", keep_empty_features=True), list(NUMERIC_FEATURES)),
        ("categorical", category, list(CATEGORICAL_FEATURES)),
    ]
    preprocess = ColumnTransformer(transforms, remainder="drop", sparse_threshold=0.0)
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=42, class_weight=None, max_iter=100)
    arm = _CandidateArm([
        ("schema", _PredictorSchema(CORE_FEATURE_POOL, CATEGORICAL_FEATURES)),
        ("preprocess", preprocess),
        ("model", model),
    ], comparison_factor="nested_hgb_tuning", weight_mode="natural")

    with pytest.raises(ValueError, match="strategy must be 'median'"):
        _validate_constructed_arm(arm, {"max_iter": 100})


def test_validate_constructed_arm_rejects_balanced_weights():
    """Arm built with weight_mode='balanced' instead of 'natural' is rejected."""
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=42, class_weight=None, max_iter=100)
    arm = _pipeline(CORE_FEATURE_POOL, CATEGORICAL_FEATURES, False, model, "regime_natural_vs_class_weighted", "balanced")
    with pytest.raises(ValueError, match="weight_mode must be 'natural'"):
        _validate_constructed_arm(arm, {"max_iter": 100})


def test_validate_constructed_arm_rejects_wrong_branch_selectors():
    """Arm with swapped or invalid column branch selectors in ColumnTransformer is rejected."""
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Swap numeric and categorical column lists
    transforms = [
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), list(CATEGORICAL_FEATURES)),
        ("categorical", category, list(NUMERIC_FEATURES)),
    ]
    preprocess = ColumnTransformer(transforms, remainder="drop", sparse_threshold=0.0)
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=42, class_weight=None, max_iter=100)
    arm = _CandidateArm([
        ("schema", _PredictorSchema(CORE_FEATURE_POOL, CATEGORICAL_FEATURES)),
        ("preprocess", preprocess),
        ("model", model),
    ], comparison_factor="nested_hgb_tuning", weight_mode="natural")

    with pytest.raises(ValueError, match="columns mismatch"):
        _validate_constructed_arm(arm, {"max_iter": 100})


def test_validate_constructed_arm_rejects_wrong_ohe_class_or_defaults():
    """Arm with OHE handle_unknown='error' or sparse_output=True is rejected."""
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("one_hot", OneHotEncoder(handle_unknown="error", sparse_output=True)),
    ])
    transforms = [
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), list(NUMERIC_FEATURES)),
        ("categorical", category, list(CATEGORICAL_FEATURES)),
    ]
    preprocess = ColumnTransformer(transforms, remainder="drop", sparse_threshold=0.0)
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=42, class_weight=None, max_iter=100)
    arm = _CandidateArm([
        ("schema", _PredictorSchema(CORE_FEATURE_POOL, CATEGORICAL_FEATURES)),
        ("preprocess", preprocess),
        ("model", model),
    ], comparison_factor="nested_hgb_tuning", weight_mode="natural")

    with pytest.raises(ValueError, match="handle_unknown='ignore'"):
        _validate_constructed_arm(arm, {"max_iter": 100})


def test_validate_constructed_arm_rejects_wrong_defaults_and_early_stopping():
    """Arm with early_stopping=True, class_weight='balanced', or wrong random_state is rejected."""
    for bad_kwarg in (
        {"early_stopping": True},
        {"random_state": 99},
        {"class_weight": "balanced"},
    ):
        model = HistGradientBoostingClassifier(early_stopping=False, random_state=42, class_weight=None, max_iter=100)
        for k, v in bad_kwarg.items():
            setattr(model, k, v)
        arm = _pipeline(CORE_FEATURE_POOL, CATEGORICAL_FEATURES, False, model, "nested_hgb_tuning", "natural")
        with pytest.raises(ValueError, match="model early_stopping|model random_state|model class_weight"):
            _validate_constructed_arm(arm, {"max_iter": 100})


# ==================== Finding 4: Freshness Negative Tests ====================


def test_fresh_arm_required_rejects_fitted_or_reused_arm(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """Constructed arm must be fresh and unfitted; reusing fitted estimator raises ValueError."""
    pre_fitted_arm = build_hgb_tuning_arm({"max_iter": 100})
    pre_fitted_arm.fit(tiny_training_partition.X_train, tiny_training_partition.y_train)

    # Directly calling validator with fitted arm raises freshness-specific ValueError
    with pytest.raises(ValueError, match="fresh unfitted estimator"):
        _validate_constructed_arm(pre_fitted_arm, {"max_iter": 100})

    # End-to-end: patched canonical builder returns pre-fitted arm, tune_outer_training raises ValueError
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    original_builder = nested_tuning.build_hgb_tuning_arm

    def patched_builder(params=None):
        arm = original_builder(params)
        arm.fit(tiny_training_partition.X_train, tiny_training_partition.y_train)
        return arm

    monkeypatch.setattr(nested_tuning, "build_hgb_tuning_arm", patched_builder)

    with pytest.raises(ValueError, match="fresh unfitted estimator"):
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=nested_tuning.build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )


# ==================== Finding 4: Attrs & Caller Immutability Tests ====================


def test_input_snapshot_clears_attrs_and_keeps_caller_unchanged(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """Attrs are strictly ABSENT at actual fit boundary, while caller attrs remain intact."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    tiny_training_partition.X_train.attrs["leak_key"] = "smuggled_x"
    tiny_training_partition.y_train.attrs["leak_key"] = "smuggled_y"
    tiny_training_partition.groups_train.attrs["leak_key"] = "smuggled_groups"

    observed_fit_attrs = []
    original_fit = nested_tuning._fit_and_score_fold

    def spy_fit(arm, X, y, groups, *args, **kwargs):
        observed_fit_attrs.append((
            dict(X.attrs),
            dict(y.attrs),
            dict(groups.attrs) if groups is not None else {},
        ))
        return original_fit(arm, X, y, groups, *args, **kwargs)

    monkeypatch.setattr(nested_tuning, "_fit_and_score_fold", spy_fit)

    result = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
    )
    assert isinstance(result, TuningResult)
    assert len(observed_fit_attrs) == 4  # 2 trials * 2 folds

    # At actual fit boundary, attrs must be strictly absent / empty
    for x_attrs, y_attrs, g_attrs in observed_fit_attrs:
        assert x_attrs == {}
        assert y_attrs == {}
        assert g_attrs == {}

    # Caller attrs remain intact and untouched
    assert tiny_training_partition.X_train.attrs["leak_key"] == "smuggled_x"
    assert tiny_training_partition.y_train.attrs["leak_key"] == "smuggled_y"
    assert tiny_training_partition.groups_train.attrs["leak_key"] == "smuggled_groups"


# ==================== Finding 3: Exact CORE_FEATURE_POOL Enforcement Tests ====================


def test_subset_of_core_columns_rejected_before_inventory_reservation_or_files(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Partition with a subset of CORE columns is rejected before file creation or fits."""
    # Drop one valid column ('hour')
    subset_cols = [c for c in CORE_COLUMNS if c != "hour"]
    bad_frame = tiny_training_partition.X_train[subset_cols].copy()
    bad_training = TrainingPartition(0, 0, bad_frame, tiny_training_partition.y_train, tiny_training_partition.groups_train)

    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    counter = FitCounter(budget)

    with pytest.raises(ValueError, match="exact ordered CORE feature pool"):
        tune_outer_training(
            training=bad_training,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=counter,
            study_dir=tmp_path,
        )

    # Zero fits used, zero fits reserved
    assert counter.used_model_fits == 0
    assert counter.reserved_model_fits == 0
    # Zero files created in study_dir
    assert list(tmp_path.iterdir()) == []


def test_reordered_core_columns_rejected_before_inventory_reservation_or_files(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Partition with reordered CORE columns is rejected before file creation or fits."""
    # Reorder columns: put 'primary_use' first
    reordered_cols = ["primary_use"] + [c for c in CORE_COLUMNS if c != "primary_use"]
    bad_frame = tiny_training_partition.X_train[reordered_cols].copy()
    bad_training = TrainingPartition(0, 0, bad_frame, tiny_training_partition.y_train, tiny_training_partition.groups_train)

    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    counter = FitCounter(budget)

    with pytest.raises(ValueError, match="exact ordered CORE feature pool"):
        tune_outer_training(
            training=bad_training,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=counter,
            study_dir=tmp_path,
        )

    assert counter.used_model_fits == 0
    assert counter.reserved_model_fits == 0
    assert list(tmp_path.iterdir()) == []


# ==================== Finding 2: Root Identity, Manifest & Reproducibility Tests ====================


def test_resolve_campaign_root_explicit_api(tmp_path: Path):
    """resolve_campaign_root validates directory existence and resolves path."""
    resolved = resolve_campaign_root(tmp_path)
    assert resolved == tmp_path.resolve()

    with pytest.raises(ValueError, match="campaign root must be an existing directory"):
        resolve_campaign_root(tmp_path / "nonexistent_root")


def test_deterministic_trial_space_and_reproducibility_with_independent_roots(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Independent roots yield identical sampled configs and winner, but NOT same campaign hash."""
    budget1 = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    budget2 = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    dir1 = tmp_path / "study1"
    dir2 = tmp_path / "study2"

    res1 = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget1, dir1, campaign_root=dir1)
    res2 = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget2, dir2, campaign_root=dir2)

    # Identical configurations and winner
    assert res1.selected_params == res2.selected_params
    assert res1.trial_id == res2.trial_id

    # NOT same campaign manifest hash due to independent campaign roots!
    assert res1.study_manifest_hash != res2.study_manifest_hash


def test_resume_fails_on_changed_campaign_root(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Resuming with a changed campaign root identity fails closed."""
    dir1 = tmp_path / "study1"
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()

    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, dir1, campaign_root=root1)

    # Resuming with changed root2 must fail closed
    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, dir1, campaign_root=root2)
    assert exc_info.value.code == "study_manifest_mismatch"


def test_resume_fails_on_changed_budget(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Resuming with a widened or altered budget fails closed."""
    dir1 = tmp_path / "study1"
    budget1 = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    budget2 = FitBudget(max_model_fits=20, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget1, dir1)

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget2, dir1)
    assert exc_info.value.code == "study_budget_mismatch"


def test_study_lock_contention_blocks_before_construction_or_fit(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """A held persistent study lock blocks before arm construction or fitting."""
    lock_path = tmp_path / ".lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fitted = []

    original_fit = _CandidateArm.fit

    def spy_fit(*args, **kwargs):
        fitted.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(_CandidateArm, "fit", spy_fit)

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockedExperimentError) as exc_info:
            tune_outer_training(
                tiny_training_partition,
                small_study_spec,
                build_hgb_tuning_arm,
                FitBudget(10, 5, 60.0, 1024**3),
                tmp_path,
            )
        assert exc_info.value.code == "study_busy"
        assert fitted == []
        assert not (tmp_path / "study_r0_f0.sqlite3").exists()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_fit_attempt_is_charged_before_arm_construction(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """Construction observes a charged fit attempt and failures are not refunded."""
    budget = FitBudget(10, 5, 60.0, 1024**3)
    counter = FitCounter(budget)
    observed_charges = []

    def failing_builder(params=None):
        observed_charges.append(counter.used_model_fits)
        with closing(sqlite3.connect(tmp_path / "study_r0_f0.sqlite3")) as conn:
            attempt_id, status = conn.execute(
                "SELECT owned_attempt_id, status FROM fit_attempt ORDER BY attempt_id DESC LIMIT 1"
            ).fetchone()
        with closing(sqlite3.connect(tmp_path / "fit_budget.sqlite3")) as conn:
            event = json.loads(conn.execute(
                "SELECT payload FROM accounting_event ORDER BY sequence DESC LIMIT 1"
            ).fetchone()[0])
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert status == "started"
        assert event["kind"] == "attempt_started"
        assert event["attempt_id"] == attempt_id
        raise RuntimeError("construction failed")

    monkeypatch.setattr(nested_tuning, "build_hgb_tuning_arm", failing_builder)

    with pytest.raises(BlockedExperimentError, match="no valid candidate trials"):
        tune_outer_training(
            tiny_training_partition,
            small_study_spec,
            nested_tuning.build_hgb_tuning_arm,
            counter,
            tmp_path,
        )

    assert observed_charges == [1, 2]
    assert counter.used_model_fits == 2
    with closing(sqlite3.connect(tmp_path / "study_r0_f0.sqlite3")) as conn:
        statuses = conn.execute(
            "SELECT status FROM fit_attempt ORDER BY attempt_id"
        ).fetchall()
    assert statuses == [("failed",), ("failed",)]


def test_cached_resume_reports_zero_new_and_preserves_cumulative_charge(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """A cached resume does not debit again but exposes study-cumulative accounting."""
    budget = FitBudget(10, 5, 60.0, 1024**3)
    first = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    second = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    assert first.used_model_fits == 4
    assert second.used_model_fits == 0
    assert second.study_charged_model_fits == 4
    with closing(sqlite3.connect(second.manifest_reference)) as conn:
        accounting = conn.execute(
            "SELECT charged_model_fits, last_invocation_model_fits FROM study"
        ).fetchone()
    assert accounting == (4, 0)


def test_mixed_success_resume_retries_failed_candidate_without_default_winner(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """A valid candidate can coexist with a failed candidate and remain resumable."""
    fail_first_trial = True
    original_fit = _CandidateArm.fit

    def fail_trial_zero_once(arm, X, y=None, **params):
        nonlocal fail_first_trial
        if fail_first_trial and arm.named_steps["model"].learning_rate == 0.03:
            raise RuntimeError("candidate trial failed")
        return original_fit(arm, X, y, **params)

    monkeypatch.setattr(_CandidateArm, "fit", fail_trial_zero_once)
    budget = FitBudget(10, 5, 60.0, 1024**3)
    first = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert first.trial_id == 1
    assert first.used_model_fits == 3

    fail_first_trial = False
    second = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert second.trial_id in (0, 1)
    assert second.used_model_fits == 5
    assert second.invocation_model_fits == 2
    assert second.study_charged_model_fits == 5
    with closing(sqlite3.connect(second.manifest_reference)) as conn:
        statuses = conn.execute(
            "SELECT trial_id, status FROM trial WHERE is_default = 0 ORDER BY trial_id"
        ).fetchall()
    assert statuses == [(0, "completed"), (1, "completed")]


def test_resume_fails_on_tampered_manifest_payload_or_registrydefs(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Tampering with stored manifest payload (e.g. registry_defs) fails closed on resume."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    res = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    db_path = res.manifest_reference
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("UPDATE study SET manifest_payload_json = '{\"tampered\": true}'")

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_manifest_mismatch"


def test_manifest_hash_distinguishes_hour_vs_weekday_with_same_values(
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Manifest hash distinguishes columns even when hour and weekday have identical values."""
    folds = build_inner_folds(
        tiny_training_partition.y_train,
        tiny_training_partition.groups_train,
        n_splits=small_study_spec.inner_folds,
        seed=small_study_spec.seed,
    )
    digest = inner_membership_digest(folds, tiny_training_partition.groups_train)

    frame1 = tiny_training_partition.X_train.copy()
    frame1["hour"] = 0
    frame1["weekday"] = 0
    p1 = TrainingPartition(0, 0, frame1, tiny_training_partition.y_train, tiny_training_partition.groups_train)
    hash1 = compute_study_manifest_hash(p1, small_study_spec, False, digest)

    frame2 = frame1.copy()
    cols = list(frame1.columns)
    idx_h, idx_w = cols.index("hour"), cols.index("weekday")
    cols[idx_h], cols[idx_w] = cols[idx_w], cols[idx_h]
    frame2 = frame2[cols]
    p2 = TrainingPartition(0, 0, frame2, tiny_training_partition.y_train, tiny_training_partition.groups_train)
    hash2 = compute_study_manifest_hash(p2, small_study_spec, False, digest)

    assert hash1 != hash2


def test_manifest_payload_contains_explicit_registry_defs_and_pipeline_structure(
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Manifest payload binds explicit raw FeatureMetadata records and canonical pipeline."""
    folds = build_inner_folds(
        tiny_training_partition.y_train,
        tiny_training_partition.groups_train,
        n_splits=small_study_spec.inner_folds,
        seed=small_study_spec.seed,
    )
    digest = inner_membership_digest(folds, tiny_training_partition.groups_train)
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    payload = build_study_manifest_payload(tiny_training_partition, small_study_spec, False, digest, budget=budget)

    # Verify registry_defs has all 9 features with truthful metadata records
    assert len(payload["registry_defs"]) == 9
    assert "meter_reading" in payload["registry_defs"]
    mr_meta = payload["registry_defs"]["meter_reading"]
    assert mr_meta["family"] == "measurement"
    assert mr_meta["prediction_time_availability"] == "available"
    assert mr_meta["target_dependence"] == "independent"

    # Verify canonical pipeline structural description
    pipeline_desc = payload["resolved_canonical_pipeline"]
    assert pipeline_desc["arm_class"] == "_CandidateArm"
    assert pipeline_desc["weight_mode"] == "natural"
    assert pipeline_desc["preprocess"]["numeric_branch"]["strategy"] == "median"
    assert pipeline_desc["preprocess"]["categorical_branch"]["steps"][0]["strategy"] == "most_frequent"
    assert pipeline_desc["model"]["early_stopping"] is False
    assert pipeline_desc["model"]["random_state"] == 42


# ==================== Core Workflow & Boundary Spy Tests ====================


def test_insufficient_budget_rejects_before_any_fits(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Budget with max_model_fits=5 must reject before creating a completed trial or fitting."""
    budget = FitBudget(
        max_model_fits=5,
        max_trials=2,
        max_wall_seconds=60.0,
        max_rss_bytes=1024 * 1024 * 1024,
    )

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )
    assert exc_info.value.code == "fit_budget_exhausted"


def test_training_partition_isolation_and_no_fold_partition(
    tmp_path: Path,
    small_study_spec: StudySpec,
):
    """tune_outer_training accepts ONLY TrainingPartition; FoldPartition is rejected."""
    partitions = synthetic_partitions()
    fold_partition = partitions[0]
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    with pytest.raises(TypeError, match="FoldPartition is prohibited"):
        tune_outer_training(
            training=fold_partition,  # type: ignore[arg-type]
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )

    fake_with_heldout = SimpleNamespace(
        repeat=0, fold=0, X_train=fold_partition.X_train, y_train=fold_partition.y_train,
        groups_train=fold_partition.groups_train, X_validation=fold_partition.X_validation,
    )
    with pytest.raises(TypeError):
        tune_outer_training(
            training=fake_with_heldout,  # type: ignore[arg-type]
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )


def test_canonical_spy_receives_only_train_group_subsets_and_no_outer_validation(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """Canonical fit boundary spy verifies fitting receives only train-group subsets."""
    budget = FitBudget(max_model_fits=20, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    fitted_subsets = []

    original_fit = nested_tuning._fit_and_score_fold

    def spy_fit(arm, X, y, groups, *args, **kwargs):
        fitted_subsets.append((X.index.tolist(), set(y.tolist()), set(groups.tolist()) if groups is not None else set()))
        return original_fit(arm, X, y, groups, *args, **kwargs)

    monkeypatch.setattr(nested_tuning, "_fit_and_score_fold", spy_fit)

    res = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
    )
    assert isinstance(res, TuningResult)
    assert len(fitted_subsets) == 4  # 2 trials * 2 folds

    training_indices = set(tiny_training_partition.X_train.index)
    training_groups = set(tiny_training_partition.groups_train)
    for indices, classes, groups in fitted_subsets:
        assert set(indices).issubset(training_indices)
        assert len(indices) < len(training_indices)
        assert classes == {0, 1}
        assert groups.issubset(training_groups)
        assert len(groups) < len(training_groups)


def test_actual_tiny_hgb_tuning_deterministic_and_persisted(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Actual tiny HGB study runs on 16 generated rows with SQLite persistence."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    result = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
    )

    assert isinstance(result, TuningResult)
    assert isinstance(result.selected_params, tuple)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in result.selected_params)
    assert result.trial_id in (0, 1)
    assert result.planned_model_fits == 6
    assert result.used_model_fits == 4
    assert Path(result.manifest_reference).is_file()

    with closing(sqlite3.connect(result.manifest_reference)) as conn:
        study_row = conn.execute("SELECT status, selected_trial_id, planned_model_fits FROM study").fetchone()
        assert study_row == ("completed", result.trial_id, 6)

        trial_rows = conn.execute("SELECT trial_id, status, mean_score FROM trial WHERE is_default = 0").fetchall()
        assert len(trial_rows) == 2
        for t_id, status, score in trial_rows:
            assert status == "completed"
            assert 0.0 <= score <= 1.0

        attempt_rows = conn.execute("SELECT status FROM fit_attempt").fetchall()
        assert len(attempt_rows) == 4
        assert all(a[0] == "completed" for a in attempt_rows)

        fold_rows = conn.execute("SELECT score, status FROM fold_score").fetchall()
        assert len(fold_rows) == 4
        assert all(f[1] == "completed" and 0.0 <= f[0] <= 1.0 for f in fold_rows)


def test_deterministic_tie_break_by_trial_id(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """When candidate trials tie on mean ROC-AUC, lowest trial_id wins deterministically."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    # Monkeypatch roc_auc_score inside nested_tuning to return identical score across all trials
    monkeypatch.setattr(nested_tuning, "roc_auc_score", lambda y_true, y_score: 0.5)

    result = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
    )
    assert result.trial_id == 0


def test_score_inner_default_evaluates_default_and_reserves_k_extra_fits(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """When score_inner_default=True, default HGB is scored across inner folds (extra K fits)."""
    budget = FitBudget(max_model_fits=15, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    result = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
        score_inner_default=True,
    )

    assert result.planned_model_fits == 8  # 2*2 + 2 + 2 = 8
    assert result.used_model_fits == 6     # 4 candidate fits + 2 default fits
    assert result.trial_id in (0, 1)       # Winner is chosen from candidate trials, NOT default!

    with closing(sqlite3.connect(result.manifest_reference)) as conn:
        default_row = conn.execute("SELECT status, is_default, mean_score FROM trial WHERE is_default = 1").fetchone()
        assert default_row is not None
        assert default_row[0] == "completed"
        assert default_row[1] == 1
        assert 0.0 <= default_row[2] <= 1.0


def test_failures_and_resume_within_cumulative_budget(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """Failed trial attempts are recorded/debited; matching resume completes remaining work."""
    fail_fits = True
    call_count = 0
    original_fit = _CandidateArm.fit

    def flaky_fit(arm, X, y=None, **params):
        nonlocal call_count
        call_count += 1
        if fail_fits:
            raise RuntimeError("simulated fold fit failure")
        return original_fit(arm, X, y, **params)

    monkeypatch.setattr(_CandidateArm, "fit", flaky_fit)

    budget1 = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    counter1 = FitCounter(budget1)

    with pytest.raises(BlockedExperimentError) as first_error:
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=counter1,
            study_dir=tmp_path,
        )
    assert first_error.value.code == "no_valid_trials"
    assert counter1.used_model_fits == 2

    # Resume with working fit
    fail_fits = False
    budget2 = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    counter2 = FitCounter(budget2)

    res = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=counter2,
        study_dir=tmp_path,
    )

    assert isinstance(res, TuningResult)
    assert counter2.used_model_fits == 6


def test_completed_study_resume_performs_zero_fits(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """Calling tune_outer_training on an already completed study performs zero model fits."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    res1 = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert res1.used_model_fits == 4

    fits_during_resume = 0

    def fail_fit(arm, X, y=None, **params):
        nonlocal fits_during_resume
        fits_during_resume += 1
        raise AssertionError("completed study must not fit models!")

    monkeypatch.setattr(_CandidateArm, "fit", fail_fit)

    res2 = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert fits_during_resume == 0
    assert res2.used_model_fits == 0
    assert res2.selected_params == res1.selected_params
    assert res2.trial_id == res1.trial_id


def test_tampered_database_fails_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Tampered study manifest or status in SQLite must raise BlockedExperimentError."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    res = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    db_path = res.manifest_reference
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("UPDATE study SET study_manifest_hash = 'tampered_hash'")

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_manifest_mismatch"


def test_tampered_trial_definition_fails_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Trial definitions are immutable across resume; a changed config hash blocks."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    result = tune_outer_training(
        tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path
    )
    with closing(sqlite3.connect(result.manifest_reference)) as conn, conn:
        conn.execute("UPDATE trial SET params_hash = 'tampered' WHERE trial_id = 0")
    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path
        )
    assert exc_info.value.code == "study_tampered"


def test_distinct_study_scopes_do_not_collide(
    tmp_path: Path,
    small_study_spec: StudySpec,
):
    """Distinct outer folds or repeats produce distinct study records without colliding."""
    partitions = synthetic_partitions()
    fold0 = training_view(partitions[0])
    fold1 = training_view(partitions[1])

    budget = FitBudget(max_model_fits=20, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    res0 = tune_outer_training(fold0, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    res1 = tune_outer_training(fold1, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    assert res0.study_manifest_hash != res1.study_manifest_hash
    assert res0.manifest_reference != res1.manifest_reference


def test_all_trials_failed_raises_no_valid_trials(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """When all candidate trials fail, raise BlockedExperimentError('no_valid_trials')."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    def failing_fit(arm, X, y=None, **params):
        raise ValueError("all fits fail")

    monkeypatch.setattr(_CandidateArm, "fit", failing_fit)

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )
    assert exc_info.value.code == "no_valid_trials"


def test_insufficient_group_support_fails_closed(tmp_path: Path, small_study_spec: StudySpec):
    """Training partition with fewer groups than inner folds raises BlockedExperimentError."""
    frame = core_frame().iloc[:4].copy()
    groups = pd.Series(["b1", "b1", "b1", "b1"], index=frame.index, name="building_id")
    target = binary_target(frame)

    training = TrainingPartition(0, 0, frame, target, groups)
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            training=training,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )
    assert exc_info.value.code in ("insufficient_inner_groups", "one_class_inner_partition")


# ==================== Chunk 3: Immutable Fold Prediction Evidence & Verified Resume ====================


def test_fold_prediction_evidence_files_created_with_required_fields(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Inner fold evaluation writes immutable prediction evidence JSON files."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    res = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
    )
    evidence_dir = tmp_path / "fold_evidence"
    assert evidence_dir.is_dir()

    required_fields = {
        "trial_id",
        "fold_index",
        "is_default",
        "params_hash",
        "validation_positions",
        "probabilities",
        "score",
        "sha256",
    }

    with closing(sqlite3.connect(res.manifest_reference)) as conn:
        fold_rows = conn.execute(
            "SELECT trial_id, fold_index, score, status, evidence_path, evidence_sha256 "
            "FROM fold_score ORDER BY trial_id, fold_index"
        ).fetchall()

    assert len(fold_rows) == 4  # 2 trials * 2 folds
    for trial_id, fold_idx, score, status, evidence_path_str, evidence_sha in fold_rows:
        assert status == "completed"
        assert evidence_path_str is not None
        assert evidence_sha is not None

        evidence_path = Path(evidence_path_str)
        assert evidence_path.is_file()
        assert evidence_path.name == f"trial_{trial_id}_fold_{fold_idx}.json"
        raw_bytes = evidence_path.read_bytes()
        assert hashlib.sha256(raw_bytes).hexdigest() == evidence_sha

        data = json.loads(raw_bytes.decode("utf-8"))
        assert required_fields.issubset(data.keys())
        assert data["trial_id"] == trial_id
        assert data["fold_index"] == fold_idx
        assert data["is_default"] == 0
        assert isinstance(data["params_hash"], str) and bool(data["params_hash"])
        assert isinstance(data["validation_positions"], list)
        assert len(data["validation_positions"]) == 4  # 8 rows per fold in 2 splits of 16 rows (groups_train)
        assert isinstance(data["probabilities"], list)
        assert len(data["probabilities"]) == len(data["validation_positions"])
        assert all(0.0 <= p <= 1.0 for p in data["probabilities"])
        assert 0.0 <= data["score"] <= 1.0
        assert abs(data["score"] - score) < 1e-12

        payload_without_sha = {k: v for k, v in data.items() if k != "sha256"}
        expected_sha = hashlib.sha256(
            json.dumps(payload_without_sha, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert data["sha256"] == expected_sha


def test_tampering_fold_prediction_file_causes_resume_to_fail_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Tampering with an evidence file's probabilities or score blocks resume."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    target_file = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    content = json.loads(target_file.read_text())

    # Tamper with probabilities
    tampered_content = dict(content)
    tampered_content["probabilities"] = [0.0] * len(tampered_content["probabilities"])
    target_file.write_text(json.dumps(tampered_content))

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"

    # Restore content then tamper with score
    tampered_score = dict(content)
    tampered_score["score"] = 0.999
    target_file.write_text(json.dumps(tampered_score))

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"


def test_deleting_fold_prediction_file_causes_resume_to_fail_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Deleting a fold prediction evidence file causes resume to fail closed."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    target_file = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    target_file.unlink()

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"


def test_tampering_with_selected_winner_causes_resume_to_fail_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Tampering with selected_trial_id in SQLite study table causes resume to fail closed."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    res = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    tampered_trial = 1 if res.trial_id == 0 else 0
    with closing(sqlite3.connect(res.manifest_reference)) as conn, conn:
        conn.execute("UPDATE study SET selected_trial_id = ?", (tampered_trial,))

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"


def test_recomputation_of_roc_auc_and_trial_mean_on_resume_matches_exactly(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Resume recomputes fold ROC-AUC and trial mean from evidence and matches exactly."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    res1 = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    res2 = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert res2.trial_id == res1.trial_id
    assert res2.selected_params == res1.selected_params
    assert res2.used_model_fits == 0

    with closing(sqlite3.connect(res1.manifest_reference)) as conn:
        for t_id in (0, 1):
            scores = []
            for f_idx in (0, 1):
                ev_file = tmp_path / "fold_evidence" / f"trial_{t_id}_fold_{f_idx}.json"
                ev_data = json.loads(ev_file.read_text())
                y_true = tiny_training_partition.y_train.iloc[ev_data["validation_positions"]].to_numpy()
                probs = pd.Series(ev_data["probabilities"]).to_numpy(dtype=float)
                recomputed_auc = float(nested_tuning.roc_auc_score(y_true, probs))
                assert abs(recomputed_auc - ev_data["score"]) < 1e-12
                scores.append(recomputed_auc)
            expected_mean = float(pd.Series(scores).mean())
            stored_mean = conn.execute(
                "SELECT mean_score FROM trial WHERE trial_id = ?", (t_id,)
            ).fetchone()[0]
            assert abs(expected_mean - stored_mean) < 1e-12


def test_completed_study_can_have_failed_candidate_and_failed_default(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """A study completes if at least one candidate trial completed; failed default is not winner."""
    original_fit = _CandidateArm.fit

    def flaky_fit(arm, X, y=None, **params):
        lr = arm.named_steps["model"].learning_rate
        # trial 1 has learning_rate 0.1; default has learning_rate 0.1
        # trial 0 has learning_rate 0.03
        if lr == 0.1:
            raise RuntimeError("simulated failure for lr 0.1 and default")
        return original_fit(arm, X, y, **params)

    monkeypatch.setattr(_CandidateArm, "fit", flaky_fit)
    budget = FitBudget(max_model_fits=20, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    res = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
        score_inner_default=True,
    )
    assert res.trial_id == 0  # Only candidate 0 completed, so candidate 0 must win!

    with closing(sqlite3.connect(res.manifest_reference)) as conn:
        study_status, sel_id = conn.execute("SELECT status, selected_trial_id FROM study").fetchone()
        assert study_status == "completed"
        assert sel_id == 0

        trial_statuses = dict(conn.execute("SELECT trial_id, status FROM trial").fetchall())
        assert trial_statuses[0] == "completed"
        assert trial_statuses[1] == "failed"
        assert trial_statuses[2] == "failed"  # Default trial failed!

    # Tampering selected_trial_id to point to failed trial 1 fails closed
    with closing(sqlite3.connect(res.manifest_reference)) as conn, conn:
        conn.execute("UPDATE study SET selected_trial_id = 1")
    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
            score_inner_default=True,
        )
    assert exc_info.value.code == "study_tampered"

    # Tampering selected_trial_id to point to default trial 2 fails closed
    with closing(sqlite3.connect(res.manifest_reference)) as conn, conn:
        conn.execute("UPDATE study SET selected_trial_id = 2")
    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
            score_inner_default=True,
        )
    assert exc_info.value.code == "study_tampered"


def test_in_progress_resume_with_tampered_or_missing_evidence_fails_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    monkeypatch: pytest.MonkeyPatch,
):
    """Resuming an in-progress study with tampered or deleted fold evidence fails closed."""
    call_count = 0
    orig_fit = _CandidateArm.fit

    class InterruptStudyError(Exception):
        pass

    def interrupt_after_first_fold(arm, X, y=None, **params):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise InterruptStudyError("interrupting after first fold committed")
        return orig_fit(arm, X, y, **params)

    monkeypatch.setattr(_CandidateArm, "fit", interrupt_after_first_fold)
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)

    # First attempt starts, completes trial 0 fold 0, then gets interrupted on fold 1
    with pytest.raises(InterruptStudyError):
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )

    # Verify study is still in_progress and trial 0 fold 0 was completed
    db_file = tmp_path / "study_r0_f0.sqlite3"
    with closing(sqlite3.connect(db_file)) as conn:
        status = conn.execute("SELECT status FROM study WHERE study_key = 'study_r0_f0'").fetchone()[0]
        assert status == "in_progress"
        f0_status = conn.execute(
            "SELECT status FROM fold_score WHERE study_key = 'study_r0_f0' AND trial_id = 0 AND fold_index = 0"
        ).fetchone()[0]
        assert f0_status == "completed"

    target_file = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    assert target_file.is_file()

    # Tamper with the completed fold evidence while study is in_progress
    content = json.loads(target_file.read_text())
    tampered_content = dict(content)
    tampered_content["score"] = 0.12345
    target_file.write_text(json.dumps(tampered_content))

    # Reset monkeypatch so fits would succeed if not blocked
    monkeypatch.setattr(_CandidateArm, "fit", orig_fit)

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )
    assert exc_info.value.code == "study_tampered"

    # Now delete the file completely
    target_file.unlink()
    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(
            training=tiny_training_partition,
            study_spec=small_study_spec,
            arm_factory=build_hgb_tuning_arm,
            budget=budget,
            study_dir=tmp_path,
        )
    assert exc_info.value.code == "study_tampered"


def test_direct_selected_params_json_tampering_fails_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """Directly tampering with selected_params_json in SQLite study table causes resume to fail closed."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    res = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    tampered_params = json.dumps({"learning_rate": 0.999, "max_leaf_nodes": 999})
    with closing(sqlite3.connect(res.manifest_reference)) as conn, conn:
        conn.execute("UPDATE study SET selected_params_json = ?", (tampered_params,))

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"


def test_orphan_uncommitted_evidence_file_recovered_on_retry(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """An orphan uncommitted evidence file from an interrupted attempt is cleaned up on retry."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    evidence_dir = tmp_path / "fold_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    orphan_file = evidence_dir / "trial_0_fold_0.json"
    orphan_file.write_text('{"orphan": "uncommitted"}')

    res = tune_outer_training(
        training=tiny_training_partition,
        study_spec=small_study_spec,
        arm_factory=build_hgb_tuning_arm,
        budget=budget,
        study_dir=tmp_path,
    )
    assert res.trial_id in (0, 1)
    content = json.loads(orphan_file.read_text())
    assert content["trial_id"] == 0
    assert content["fold_index"] == 0
    assert "orphan" not in content


def test_unconfined_evidence_path_outside_study_dir_fails_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """An evidence_path pointing outside study_dir fails closed with study_tampered."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    res = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    outside_dir = tmp_path.parent / f"outside_{tmp_path.name}"
    outside_dir.mkdir(parents=True, exist_ok=True)
    orig_ev = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    outside_ev = outside_dir / "trial_0_fold_0.json"
    outside_ev.write_bytes(orig_ev.read_bytes())
    outside_sha = hashlib.sha256(outside_ev.read_bytes()).hexdigest()

    with closing(sqlite3.connect(res.manifest_reference)) as conn, conn:
        conn.execute(
            "UPDATE fold_score SET evidence_path = ?, evidence_sha256 = ? WHERE trial_id = 0 AND fold_index = 0",
            (str(outside_ev), outside_sha),
        )

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"


def test_evidence_json_with_extra_keys_fails_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
):
    """An evidence JSON with extra unallowed keys fails closed with study_tampered."""
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    res = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    target_file = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    data = json.loads(target_file.read_text())
    data["extra_unauthorized_key"] = "tampered_value"

    payload_without_sha = {k: v for k, v in data.items() if k != "sha256"}
    data["sha256"] = hashlib.sha256(
        json.dumps(payload_without_sha, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    raw_bytes = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    target_file.write_bytes(raw_bytes)
    new_sha = hashlib.sha256(raw_bytes).hexdigest()

    with closing(sqlite3.connect(res.manifest_reference)) as conn, conn:
        conn.execute(
            "UPDATE fold_score SET evidence_sha256 = ? WHERE trial_id = 0 AND fold_index = 0",
            (new_sha,),
        )

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"


@pytest.mark.parametrize(
    "tamper_mutation",
    [
        ("non_int_positions", lambda d: d.update({"validation_positions": ["bad_pos", "bad_pos2"]})),
        ("non_numeric_probabilities", lambda d: d.update({"probabilities": ["not_a_number"] * len(d["probabilities"])})),
        ("nan_probabilities", lambda d: d.update({"probabilities": [float("nan")] * len(d["probabilities"])})),
        ("non_numeric_score", lambda d: d.update({"score": "not_a_score"})),
    ],
)
def test_malformed_evidence_value_types_fail_closed(
    tmp_path: Path,
    tiny_training_partition: TrainingPartition,
    small_study_spec: StudySpec,
    tamper_mutation: tuple[str, Any],
):
    """Malformed evidence value types fail closed with study_tampered, not raw TypeError/ValueError/IndexError."""
    _, mutate_fn = tamper_mutation
    budget = FitBudget(max_model_fits=10, max_trials=5, max_wall_seconds=60.0, max_rss_bytes=1024**3)
    res = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)

    target_file = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    data = json.loads(target_file.read_text())
    mutate_fn(data)

    payload_without_sha = {k: v for k, v in data.items() if k != "sha256"}
    data["sha256"] = hashlib.sha256(
        json.dumps(payload_without_sha, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    raw_bytes = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    target_file.write_bytes(raw_bytes)
    new_sha = hashlib.sha256(raw_bytes).hexdigest()

    with closing(sqlite3.connect(res.manifest_reference)) as conn, conn:
        conn.execute(
            "UPDATE fold_score SET evidence_sha256 = ? WHERE trial_id = 0 AND fold_index = 0",
            (new_sha,),
        )

    with pytest.raises(BlockedExperimentError) as exc_info:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert exc_info.value.code == "study_tampered"


@pytest.mark.parametrize("role", ["study", "campaign"])
def test_existing_symlink_ancestor_rejected_before_external_orphan_cleanup(
    tmp_path, tiny_training_partition, small_study_spec, role,
):
    external = tmp_path / "external"
    child = external / "existing_child"
    evidence_dir = child / "fold_evidence"
    evidence_dir.mkdir(parents=True)
    sentinel = evidence_dir / "trial_0_fold_0.json"
    sentinel.write_text("external sentinel")
    alias = tmp_path / "alias"
    alias.symlink_to(external, target_is_directory=True)
    unsafe = alias / "existing_child"
    kwargs = {"campaign_root": unsafe} if role == "campaign" else {}
    with pytest.raises(ValueError, match="symlink"):
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm,
                            FitBudget(10, 2, 60.0, 1024**3),
                            tmp_path / "safe_study" if role == "campaign" else unsafe, **kwargs)
    assert sentinel.read_text() == "external sentinel"


def test_unmanaged_regular_evidence_path_inside_root_is_rejected(
    tmp_path, tiny_training_partition, small_study_spec,
):
    budget = FitBudget(10, 2, 60.0, 1024**3)
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    canonical = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    unmanaged = tmp_path / "unmanaged.json"
    unmanaged.write_bytes(canonical.read_bytes())
    with closing(sqlite3.connect(result.manifest_reference)) as conn, conn:
        conn.execute("UPDATE fold_score SET evidence_path = ? WHERE trial_id = 0 AND fold_index = 0", (str(unmanaged),))
    with pytest.raises(BlockedExperimentError) as error:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert error.value.code == "study_tampered"
    assert unmanaged.read_bytes() == canonical.read_bytes()


@pytest.mark.parametrize("role", ["evidence", "database", "journal"])
def test_managed_hardlink_rejected_without_replacement_or_mutation(
    tmp_path, tiny_training_partition, small_study_spec, role,
):
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"external orphan" if role == "evidence" else b"")
    managed = {"evidence": tmp_path / "fold_evidence" / "trial_0_fold_0.json",
               "database": tmp_path / "study_r0_f0.sqlite3", "journal": tmp_path / "fit_budget.sqlite3"}[role]
    managed.parent.mkdir(exist_ok=True)
    os.link(sentinel, managed)
    original = sentinel.read_bytes()
    inode = managed.stat().st_ino
    with pytest.raises(ValueError, match="hardlink|regular|link"):
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm,
                            FitBudget(10, 2, 60.0, 1024**3), tmp_path)
    assert sentinel.read_bytes() == original
    assert managed.stat().st_ino == inode


def test_rank_preserving_probability_shift_rehashed_in_sql_is_rejected(
    tmp_path, tiny_training_partition, small_study_spec,
):
    budget = FitBudget(10, 2, 60.0, 1024**3)
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    path = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    data = json.loads(path.read_text())
    data["probabilities"] = [p + 2.0 for p in data["probabilities"]]
    data["sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in data.items() if k != "sha256"}, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    with closing(sqlite3.connect(result.manifest_reference)) as conn, conn:
        conn.execute("UPDATE fold_score SET evidence_sha256 = ? WHERE trial_id = 0 AND fold_index = 0",
                     (hashlib.sha256(path.read_bytes()).hexdigest(),))
    with pytest.raises(BlockedExperimentError) as error:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert error.value.code == "study_tampered"


@pytest.mark.parametrize("dtype", [bool, str, complex])
def test_probability_boundary_does_not_coerce_invalid_numeric_storage(
    tiny_training_partition, monkeypatch, dtype,
):
    arm = build_hgb_tuning_arm()
    arm.fit(tiny_training_partition.X_train, tiny_training_partition.y_train)
    original = _CandidateArm.predict_proba

    def malformed(arm, X):
        return original(arm, X).astype(dtype)

    monkeypatch.setattr(_CandidateArm, "predict_proba", malformed)
    with pytest.raises((TypeError, ValueError), match="numeric|boolean|real"):
        nested_tuning._extract_pos_proba(arm, tiny_training_partition.X_train)


@pytest.mark.parametrize("mutation", ["extra_failed_fold", "unknown_fold_status", "foreign_attempt", "missing_trial"])
def test_projection_inventory_cannot_invent_or_recreate_work(
    tmp_path, tiny_training_partition, small_study_spec, mutation,
):
    budget = FitBudget(10, 2, 60.0, 1024**3)
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    with closing(sqlite3.connect(result.manifest_reference)) as conn, conn:
        if mutation in ("extra_failed_fold", "unknown_fold_status"):
            conn.execute("INSERT INTO fold_score (study_key, trial_id, fold_index, status) VALUES (?, 999, 999, ?)",
                         ("study_r0_f0", "failed" if mutation == "extra_failed_fold" else "invented"))
        elif mutation == "foreign_attempt":
            conn.execute("INSERT INTO fit_attempt (study_key, trial_id, fold_index, status, timestamp) "
                         "VALUES ('invented_owner', 999, 999, 'unknown', 'invented')")
        else:
            conn.execute("DELETE FROM trial WHERE trial_id = 0")
            conn.execute("UPDATE study SET status = 'in_progress', selected_trial_id = NULL, selected_params_json = NULL")
    with pytest.raises(BlockedExperimentError) as error:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert error.value.code == "study_tampered"


def test_rehashed_valid_probability_evidence_must_match_journal_completion(
    tmp_path, tiny_training_partition, small_study_spec,
):
    budget = FitBudget(10, 2, 60.0, 1024**3)
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    path = tmp_path / "fold_evidence" / "trial_0_fold_0.json"
    data = json.loads(path.read_text())
    data["probabilities"] = [p / 2 + 0.125 for p in data["probabilities"]]
    data["sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in data.items() if k != "sha256"}, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    with closing(sqlite3.connect(result.manifest_reference)) as conn, conn:
        conn.execute("UPDATE fold_score SET evidence_sha256 = ? WHERE trial_id = 0 AND fold_index = 0",
                     (hashlib.sha256(path.read_bytes()).hexdigest(),))
        conn.execute("UPDATE fit_attempt SET evidence_sha256 = ? WHERE trial_id = 0 AND fold_index = 0",
                     (hashlib.sha256(path.read_bytes()).hexdigest(),))
    with pytest.raises(BlockedExperimentError) as error:
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    assert error.value.code == "study_tampered"


@pytest.mark.parametrize("fresh", [False, True])
def test_actual_crash_after_evidence_fsync_reconciles_started_attempt(
    tmp_path, tiny_training_partition, small_study_spec, fresh,
):
    code = """
import os, sys
from candidate_fixtures import synthetic_partitions
from project.experiments import nested_tuning as n
from project.experiments.candidate_contracts import FitBudget, StudySpec, training_view
def crash(*args, **kwargs):
    os._exit(77)
n._insert_fold_score = crash
n.tune_outer_training(training_view(synthetic_partitions()[0]),
    StudySpec(2, 2, 42, (("learning_rate", (.03, .1)), ("max_leaf_nodes", (15, 31)), ("max_iter", (100,)))),
    n.build_hgb_tuning_arm, FitBudget(7, 2, 60.0, 1024**3), sys.argv[1])
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[1]),
                                                     str(Path(nested_tuning.__file__).parents[2])])}
    crashed = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env,
                             capture_output=True, text=True, timeout=30, check=False)
    assert crashed.returncode == 77, crashed.stderr
    assert (tmp_path / "fold_evidence" / "trial_0_fold_0.json").is_file()
    counter = FitCounter(FitBudget(7, 2, 60.0, 1024**3))
    if not fresh:
        counter.bind_campaign(tmp_path)
        assert counter.used_model_fits == 1
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, counter, tmp_path)
    assert (counter.used_model_fits, counter.reserved_model_fits) == (5, 7)
    with closing(sqlite3.connect(result.manifest_reference)) as conn:
        statuses = [row[0] for row in conn.execute("SELECT status FROM fit_attempt ORDER BY attempt_id")]
    assert statuses == ["interrupted", "completed", "completed", "completed", "completed"]
    assert [status for _, _, status in counter.owner_attempts(result.owner_id)] == statuses


@pytest.mark.parametrize("mutation", ["deleted_attempts", "invented_attempt", "unknown_attempt_status", "legacy_hash"])
def test_attempt_deletion_invention_and_legacy_completion_fail_closed(
    tmp_path, tiny_training_partition, small_study_spec, monkeypatch, mutation,
):
    budget = FitBudget(10, 2, 60.0, 1024**3)
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, budget, tmp_path)
    with closing(sqlite3.connect(result.manifest_reference)) as conn, conn:
        if mutation == "deleted_attempts":
            conn.execute("DELETE FROM fit_attempt")
        elif mutation == "invented_attempt":
            conn.execute("INSERT INTO fit_attempt (study_key, trial_id, fold_index, status, timestamp, owned_attempt_id) "
                         "VALUES ('study_r0_f0', 999, 999, 'failed', 'invented', 'invented')")
        elif mutation == "unknown_attempt_status":
            conn.execute("UPDATE fit_attempt SET status = 'unknown' WHERE attempt_id = 1")
        else:
            conn.execute("ALTER TABLE fold_score DROP COLUMN evidence_sha256")

    def forbid_construction(params):
        pytest.fail("tampered projection must be blocked before construction")

    monkeypatch.setattr(nested_tuning, "build_hgb_tuning_arm", forbid_construction)
    counter = FitCounter(budget)
    with pytest.raises(BlockedExperimentError) as error:
        tune_outer_training(tiny_training_partition, small_study_spec, forbid_construction, counter, tmp_path)
    assert error.value.code == "study_tampered"
    if mutation == "deleted_attempts":
        assert counter.used_model_fits == 4
    if mutation == "legacy_hash":
        with closing(sqlite3.connect(result.manifest_reference)) as conn:
            assert "evidence_sha256" not in {row[1] for row in conn.execute("PRAGMA table_info(fold_score)")}


def test_inprocess_baseexception_rolls_back_partial_sql_completion(
    tmp_path, tiny_training_partition, small_study_spec, monkeypatch,
):
    counter = FitCounter(FitBudget(7, 2, 60.0, 1024**3))
    original = nested_tuning._insert_fold_score

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("after fsync before SQL completion")

    monkeypatch.setattr(nested_tuning, "_insert_fold_score", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, counter, tmp_path)
    with closing(sqlite3.connect(tmp_path / "study_r0_f0.sqlite3")) as conn:
        assert conn.execute("SELECT status, evidence_sha256 FROM fit_attempt").fetchall() == [("interrupted", None)]
        assert conn.execute("SELECT COUNT(*) FROM fold_score").fetchone()[0] == 0
    monkeypatch.setattr(nested_tuning, "_insert_fold_score", original)
    result = tune_outer_training(tiny_training_partition, small_study_spec, build_hgb_tuning_arm, counter, tmp_path)
    assert (result.study_charged_model_fits, result.invocation_model_fits, counter.reserved_model_fits) == (5, 4, 7)
