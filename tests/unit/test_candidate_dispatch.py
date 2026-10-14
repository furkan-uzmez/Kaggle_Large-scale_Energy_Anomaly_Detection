"""Synthetic dispatch and supervised fit routing contracts.

Only generated fixtures and TrainingPartition are used; no real IO,
network access, or output creation occurs here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from candidate_fixtures import (
    CORE_COLUMNS,
    synthetic_partitions,
)
from project.experiments.candidate_contracts import (
    BlockedExperimentError,
    CandidateSpec,
    FitBudget,
    FitCounter,
    TrainingPartition,
    training_view,
)
from project.experiments.candidate_dispatch import (
    ALL_SUPPORTED_EXPERIMENT_IDS,
    EXISTING_EXPERIMENT_IDS,
    NEW_EXPERIMENT_IDS,
    CandidateArm,
    build_experiment_arms,
    fit_arm,
    resolve_candidate_spec,
)
from project.features.provided import ProvidedSourceEvidence
from project.features.registry import (
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    ResourceCost,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)
from project.features.target_encoding import GroupCrossFittedTargetEncoder
from project.features.value_recipes import SentinelRule


def _make_counter(limit: int = 10) -> FitCounter:
    budget = FitBudget(
        max_model_fits=limit,
        max_trials=0,
        max_wall_seconds=30.0,
        max_rss_bytes=256 * 1024 * 1024,
        cpu_threads=1,
    )
    counter = FitCounter(budget)
    counter.reserve(limit)
    return counter


def _training_partition() -> TrainingPartition:
    parts = synthetic_partitions()
    return training_view(parts[0])


def _sentinel_rule(tmp_path: Path) -> SentinelRule:
    evidence_rel = "docs/evidence/sentinel_test.json"
    evidence_path = tmp_path / evidence_rel
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b'{"invalid": [-999.0]}'
    evidence_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return SentinelRule(
        column="meter_reading",
        invalid_values=(-999.0,),
        replacement_value=-1.0,
        evidence_reference=evidence_rel,
        evidence_sha256=digest,
    )


def test_supported_id_inventory() -> None:
    assert len(NEW_EXPERIMENT_IDS) == 10
    assert len(EXISTING_EXPERIMENT_IDS) == 8
    assert len(ALL_SUPPORTED_EXPERIMENT_IDS) == 18
    assert set(ALL_SUPPORTED_EXPERIMENT_IDS) == set(NEW_EXPERIMENT_IDS) | set(EXISTING_EXPERIMENT_IDS)


@pytest.mark.parametrize(
    ("exp_id", "code"),
    [
        ("prohibited_opaque_building_id_digit_extraction", "prohibited_experiment"),
        ("blocked_unverified_gte_family", "blocked_unverified_gte"),
        ("feat_group_aggregates_blocked", "blocked_group_aggregates"),
    ],
)
def test_prohibited_and_blocked_experiments_fail_closed(exp_id: str, code: str) -> None:
    spec = CandidateSpec(
        experiment_id=exp_id,
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    with pytest.raises(BlockedExperimentError) as exc_info:
        build_experiment_arms(spec)
    assert exc_info.value.code == code

    with pytest.raises(BlockedExperimentError) as exc_info_resolve:
        resolve_candidate_spec(spec)
    assert exc_info_resolve.value.code == code


def test_unknown_experiment_id_rejected() -> None:
    spec = CandidateSpec(
        experiment_id="completely_invented_recipe",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    with pytest.raises(ValueError, match="unsupported or unregistered experiment ID"):
        build_experiment_arms(spec)


def test_mismatched_model_family_rejected() -> None:
    spec = CandidateSpec(
        experiment_id="feat_meter_deviation_ratio",
        feature_pool=CORE_COLUMNS,
        model_family="logistic_l2",
        model_params=(),
    )
    with pytest.raises(ValueError, match="mismatched or unsupported model family"):
        build_experiment_arms(spec)


def test_feat_core_vs_provided_blocks_without_verified_addition() -> None:
    base_pool = build_lead_baseline_registry().names
    spec = CandidateSpec(
        experiment_id="feat_core_vs_provided",
        feature_pool=base_pool,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    with pytest.raises(BlockedExperimentError) as exc_info:
        build_experiment_arms(spec)
    assert exc_info.value.code == "no_verified_additional_features"


def test_feat_core_vs_provided_succeeds_with_verified_addition(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_feature = "temperature_mean"
    evidence = ProvidedSourceEvidence(
        source_notebook="tests/fixtures/provided_sources.ipynb",
        source_cell="cell-01",
        producer="synthetic_fixture",
    )
    monkeypatch.setattr(
        "project.features.provided._repository_producer_definitions",
        lambda: {fake_feature: evidence},
    )
    meta = FeatureMetadata(
        name=fake_feature,
        family="synthetic_fixture",
        source_columns=(fake_feature,),
        semantic_meaning="Synthetic mean temperature",
        prediction_time_availability=AvailabilityStatus.AVAILABLE,
        target_dependence=TargetDependence.INDEPENDENT,
        transform_fit_boundary=TransformFitBoundary.OUTER_TRAIN,
        fallback_behavior="Required value",
        resource_cost=ResourceCost.LOW,
        provenance_status=ProvenanceStatus.VERIFIED,
    )
    base_reg = build_lead_baseline_registry()
    combined_reg = FeatureRegistry([*(base_reg.get(n) for n in base_reg.names), meta])

    core_pool = base_reg.names
    spec = CandidateSpec(
        experiment_id="feat_core_vs_provided",
        feature_pool=(*core_pool, fake_feature),
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec, registry=combined_reg)
    assert set(arms.keys()) == {"control", "challenger"}
    assert arms["control"].raw_pool == core_pool
    assert arms["challenger"].raw_pool == (*core_pool, fake_feature)


def test_feat_preserve_raw_vs_invalid_sentinel_blocks_without_rule() -> None:
    spec = CandidateSpec(
        experiment_id="feat_preserve_raw_vs_invalid_sentinel",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    with pytest.raises(BlockedExperimentError) as exc_info:
        build_experiment_arms(spec)
    assert exc_info.value.code == "missing_evidence_rule"


def test_feat_preserve_raw_vs_invalid_sentinel_builds_arms_with_rule(tmp_path: Path) -> None:
    evidence_rel = "docs/evidence/sentinel_test.json"
    evidence_path = tmp_path / evidence_rel
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b'{"invalid": [-999.0]}'
    evidence_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    spec = CandidateSpec(
        experiment_id="feat_preserve_raw_vs_invalid_sentinel",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
        method_params=(
            ("column", "meter_reading"),
            ("invalid_values", (-999.0,)),
            ("replacement_value", -1.0),
            ("evidence_reference", evidence_rel),
            ("evidence_sha256", digest),
            ("evidence_root", str(tmp_path)),
        ),
    )
    resolved = resolve_candidate_spec(spec)
    assert set(resolved.arms.keys()) == {"control", "challenger"}
    assert resolved.score_kind == "positive_probability"


def test_feat_meter_deviation_ratio_fitted_preprocessor_includes_ratio_and_flag() -> None:
    spec = CandidateSpec(
        experiment_id="feat_meter_deviation_ratio",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "challenger"}

    part = _training_partition()
    counter = _make_counter(2)

    fitted_challenger = fit_arm(arms["challenger"], part, counter)
    assert counter.used_model_fits == 1

    preprocessor = fitted_challenger.named_steps["preprocess"]
    out_features = preprocessor.get_feature_names_out()
    assert any("meter_reading_per_square_foot" in f for f in out_features)
    assert any("meter_reading_per_square_foot_invalid" in f for f in out_features)

    preds = fitted_challenger.predict_proba(part.X_train)
    assert preds.shape == (len(part.X_train), 2)
    assert np.allclose(preds.sum(axis=1), 1.0)


def test_feat_histogram_quantile_produces_three_arms() -> None:
    spec = CandidateSpec(
        experiment_id="feat_histogram_quantile",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "bin", "quantile"}

    part = _training_partition()
    counter = _make_counter(3)

    for arm_id in ("control", "bin", "quantile"):
        fitted = fit_arm(arms[arm_id], part, counter)
        preds = fitted.predict_proba(part.X_train)
        assert preds.shape == (len(part.X_train), 2)
    assert counter.used_model_fits == 3


def test_feat_conditional_cross_fitted_target_encoding_integration() -> None:
    spec = CandidateSpec(
        experiment_id="feat_conditional_cross_fitted_target_encoding",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
        method_params=(("smoothing", 10.0), ("cv_splits", 3), ("seed", 42)),
    )
    resolved = resolve_candidate_spec(spec)
    assert set(resolved.arms.keys()) == {"control", "challenger"}

    challenger = resolved.arms["challenger"]
    assert challenger.raw_pool == CORE_COLUMNS
    meta = challenger.authoritative_registry.get("primary_use_target_mean")
    assert meta is not None
    assert meta.target_dependence is TargetDependence.TARGET_DERIVED

    part = _training_partition()
    counter = _make_counter(2)
    fitted = fit_arm(challenger, part, counter)
    assert counter.used_model_fits == 1

    te_step = fitted.named_steps["features"]
    assert isinstance(te_step, GroupCrossFittedTargetEncoder)
    assert getattr(te_step, "cross_fitted_", False) is True

    preds = fitted.predict_proba(part.X_train)
    assert preds.shape == (len(part.X_train), 2)


def test_feat_missingness_passthrough_comparison() -> None:
    spec = CandidateSpec(
        experiment_id="feat_missingness_passthrough_comparison",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "challenger"}

    part = _training_partition()
    counter = _make_counter(2)
    for arm in arms.values():
        fitted = fit_arm(arm, part, counter)
        preds = fitted.predict_proba(part.X_train)
        assert preds.shape == (len(part.X_train), 2)
    assert counter.used_model_fits == 2


def test_regime_natural_vs_class_weighted() -> None:
    spec = CandidateSpec(
        experiment_id="regime_natural_vs_class_weighted",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "challenger"}

    part = _training_partition()
    counter = _make_counter(2)
    for arm in arms.values():
        fitted = fit_arm(arm, part, counter)
        preds = fitted.predict_proba(part.X_train)
        assert preds.shape == (len(part.X_train), 2)
    assert counter.used_model_fits == 2


def test_model_extratrees_core_features() -> None:
    spec = CandidateSpec(
        experiment_id="model_extratrees_core_features",
        feature_pool=CORE_COLUMNS,
        model_family="extratrees",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "challenger"}
    assert isinstance(arms["challenger"].named_steps["model"], ExtraTreesClassifier)
    assert isinstance(arms["control"].named_steps["model"], HistGradientBoostingClassifier)

    part = _training_partition()
    counter = _make_counter(2)
    for arm in arms.values():
        fitted = fit_arm(arm, part, counter)
        preds = fitted.predict_proba(part.X_train)
        assert preds.shape == (len(part.X_train), 2)
    assert counter.used_model_fits == 2


def test_model_autogluon_core_features_fails_closed_without_supervisor() -> None:
    spec = CandidateSpec(
        experiment_id="model_autogluon_core_features",
        feature_pool=CORE_COLUMNS,
        model_family="autogluon",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "challenger"}

    part = _training_partition()
    counter = _make_counter(2)

    fit_arm(arms["control"], part, counter)
    assert counter.used_model_fits == 1

    with pytest.raises(BlockedExperimentError) as exc_info:
        fit_arm(arms["challenger"], part, counter)
    assert exc_info.value.code == "supervisor_unavailable"
    assert counter.used_model_fits == 2


def test_regime_same_family_hpo_vs_default_resolution_and_guard() -> None:
    spec = CandidateSpec(
        experiment_id="regime_same_family_hpo_vs_default",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "challenger"}

    part = _training_partition()
    counter = _make_counter(2)

    fitted_control = fit_arm(arms["control"], part, counter)
    assert counter.used_model_fits == 1
    assert fitted_control.predict_proba(part.X_train).shape == (len(part.X_train), 2)

    with pytest.raises(BlockedExperimentError) as exc_info:
        fit_arm(arms["challenger"], part, counter)
    assert exc_info.value.code == "hpo_not_ready"


@pytest.mark.parametrize(
    "exp_id",
    [
        "feat_missingness_indicators",
        "feat_safe_calendar_numeric_transforms",
        "feat_categorical_interactions",
        "feat_missingness_bitmask",
        "feat_frequency_encoding",
    ],
)
def test_existing_feature_and_encoding_recipes_dispatch_and_fit(exp_id: str) -> None:
    spec = CandidateSpec(
        experiment_id=exp_id,
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    assert set(arms.keys()) == {"control", "challenger"}

    part = _training_partition()
    counter = _make_counter(2)
    for arm in arms.values():
        fitted = fit_arm(arm, part, counter)
        preds = fitted.predict_proba(part.X_train)
        assert preds.shape == (len(part.X_train), 2)
    assert counter.used_model_fits == 2


def test_dim_pca_vs_passthrough_score_kind_margin() -> None:
    spec = CandidateSpec(
        experiment_id="dim_pca_vs_passthrough",
        feature_pool=("meter_reading", "square_feet", "primary_use"),
        model_family="svm",
        model_params=(),
    )
    resolved = resolve_candidate_spec(spec)
    assert resolved.score_kind == "decision_margin"
    for arm in resolved.arms.values():
        assert arm.score_kind == "decision_margin"

    part = _training_partition()
    reduced_X = part.X_train[["meter_reading", "square_feet", "primary_use"]].copy()
    train_part = TrainingPartition(
        repeat=part.repeat,
        fold=part.fold,
        X_train=reduced_X,
        y_train=part.y_train,
        groups_train=part.groups_train,
    )
    counter = _make_counter(2)
    for arm in resolved.arms.values():
        fitted = fit_arm(arm, train_part, counter)
        margins = fitted.decision_function(reduced_X)
        assert margins.shape == (len(reduced_X),)
    assert counter.used_model_fits == 2


def test_dim_fold_local_selector_vs_passthrough() -> None:
    spec = CandidateSpec(
        experiment_id="dim_fold_local_selector_vs_passthrough",
        feature_pool=("meter_reading", "square_feet", "primary_use"),
        model_family="logistic_l2",
        model_params=(),
    )
    resolved = resolve_candidate_spec(spec)
    assert resolved.score_kind == "positive_probability"

    part = _training_partition()
    reduced_X = part.X_train[["meter_reading", "square_feet", "primary_use"]].copy()
    train_part = TrainingPartition(
        repeat=part.repeat,
        fold=part.fold,
        X_train=reduced_X,
        y_train=part.y_train,
        groups_train=part.groups_train,
    )
    counter = _make_counter(3)
    for arm in resolved.arms.values():
        fitted = fit_arm(arm, train_part, counter)
        preds = fitted.predict_proba(reduced_X)
        assert preds.shape == (len(reduced_X), 2)
    assert counter.used_model_fits == 3


def test_dim_semantic_block_drop() -> None:
    spec = CandidateSpec(
        experiment_id="dim_semantic_block_drop",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
        method_params=(("drop_block", "calendar"),),
    )
    resolved = resolve_candidate_spec(spec)
    assert resolved.score_kind == "positive_probability"
    assert "passthrough" in resolved.arms
    assert "drop_calendar" in resolved.arms

    part = _training_partition()
    counter = _make_counter(2)
    for arm in resolved.arms.values():
        fitted = fit_arm(arm, part, counter)
        preds = fitted.predict_proba(part.X_train)
        assert preds.shape == (len(part.X_train), 2)
    assert counter.used_model_fits == 2


def test_dim_semantic_block_drop_requires_drop_block() -> None:
    spec = CandidateSpec(
        experiment_id="dim_semantic_block_drop",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    with pytest.raises(ValueError, match="semantic recipe requires an explicit drop_block"):
        build_experiment_arms(spec)


def test_fit_arm_rejects_fold_partition() -> None:
    spec = CandidateSpec(
        experiment_id="feat_meter_deviation_ratio",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    fold_part = synthetic_partitions()[0]
    counter = _make_counter(2)

    with pytest.raises(TypeError, match="TrainingPartition; FoldPartition is prohibited"):
        fit_arm(arms["control"], fold_part, counter)  # type: ignore[arg-type]


def test_fit_arm_enforces_budget_debit_and_fails_when_exhausted() -> None:
    spec = CandidateSpec(
        experiment_id="feat_meter_deviation_ratio",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    part = _training_partition()
    exhausted_counter = _make_counter(1)
    exhausted_counter.before_fit()

    with pytest.raises(BlockedExperimentError) as exc_info:
        fit_arm(arms["control"], part, exhausted_counter)
    assert exc_info.value.code == "fit_not_reserved"


def test_clone_starts_without_fitted_state_and_param_mutation_requires_refit() -> None:
    spec = CandidateSpec(
        experiment_id="feat_meter_deviation_ratio",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    part = _training_partition()
    counter = _make_counter(2)

    fitted_arm = fit_arm(arms["challenger"], part, counter)
    assert hasattr(fitted_arm.named_steps["features"], "output_features_")

    cloned_arm = clone(fitted_arm)
    assert not hasattr(cloned_arm.named_steps["features"], "output_features_")

    cloned_pipe = clone(fitted_arm.transformer_or_pipeline)
    assert not hasattr(cloned_pipe.named_steps["features"], "output_features_")


def test_target_encoding_clone_starts_unfitted_and_mutation_requires_refit() -> None:
    spec = CandidateSpec(
        experiment_id="feat_conditional_cross_fitted_target_encoding",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    part = _training_partition()
    counter = _make_counter(2)

    fitted_arm = fit_arm(arms["challenger"], part, counter)
    te_step = fitted_arm.named_steps["features"]
    assert hasattr(te_step, "full_training_map_")

    cloned = clone(fitted_arm)
    assert not hasattr(cloned.named_steps["features"], "full_training_map_")

    te_step.smoothing = 999.0
    with pytest.raises(ValueError, match="configuration changed; fit again before use"):
        fitted_arm.predict_proba(part.X_train)


def test_distribution_mutation_requires_refit() -> None:
    spec = CandidateSpec(
        experiment_id="feat_histogram_quantile",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    arms = build_experiment_arms(spec)
    part = _training_partition()
    counter = _make_counter(2)

    fitted_bin = fit_arm(arms["bin"], part, counter)
    bin_step = fitted_bin.named_steps["features"]
    bin_step.n_bins = 999
    with pytest.raises(ValueError, match="configuration changed; fit again before use"):
        fitted_bin.predict_proba(part.X_train)


def test_candidate_arm_validation_and_delegations() -> None:
    base_reg = build_lead_baseline_registry()
    pipe = Pipeline([("clf", HistGradientBoostingClassifier(random_state=42))])

    with pytest.raises(ValueError, match="candidate_id must be a nonempty string"):
        CandidateArm(
            candidate_id="",
            arm_id="control",
            raw_pool=CORE_COLUMNS,
            transformer_or_pipeline=pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )

    with pytest.raises(ValueError, match="arm_id must be a nonempty string"):
        CandidateArm(
            candidate_id="c1",
            arm_id="  ",
            raw_pool=CORE_COLUMNS,
            transformer_or_pipeline=pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )

    with pytest.raises(ValueError, match="raw_pool must be a nonempty tuple"):
        CandidateArm(
            candidate_id="c1",
            arm_id="control",
            raw_pool=(),
            transformer_or_pipeline=pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )

    with pytest.raises(ValueError, match="transformer_or_pipeline must not be None"):
        CandidateArm(
            candidate_id="c1",
            arm_id="control",
            raw_pool=CORE_COLUMNS,
            transformer_or_pipeline=None,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )

    with pytest.raises(TypeError, match="authoritative_registry must be an authoritative FeatureRegistry"):
        CandidateArm(
            candidate_id="c1",
            arm_id="control",
            raw_pool=CORE_COLUMNS,
            transformer_or_pipeline=pipe,
            authoritative_registry="not_a_registry",  # type: ignore[arg-type]
            score_kind="positive_probability",
        )

    with pytest.raises(ValueError, match="unsupported explicit score_kind"):
        CandidateArm(
            candidate_id="c1",
            arm_id="control",
            raw_pool=CORE_COLUMNS,
            transformer_or_pipeline=pipe,
            authoritative_registry=base_reg,
            score_kind="unknown_kind",  # type: ignore[arg-type]
        )

    arm = CandidateArm(
        candidate_id="c1",
        arm_id="control",
        raw_pool=CORE_COLUMNS,
        transformer_or_pipeline=pipe,
        authoritative_registry=base_reg,
        score_kind="positive_probability",
    )
    assert arm["clf"] is pipe.named_steps["clf"]


def test_resolved_candidate_properties() -> None:
    spec = CandidateSpec(
        experiment_id="feat_meter_deviation_ratio",
        feature_pool=CORE_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    resolved = resolve_candidate_spec(spec)
    assert resolved.experiment_id == "feat_meter_deviation_ratio"
    assert resolved.model_family == "hist_gradient_boosting"
    assert resolved.raw_pool == CORE_COLUMNS
    assert isinstance(resolved.authoritative_registry, FeatureRegistry)
    assert resolved.score_kind == "positive_probability"


@pytest.mark.parametrize("recipe,family", [("dim_pca_vs_passthrough", "svm"), ("dim_fold_local_selector_vs_passthrough", "logistic_l2")])
def test_selection_rejects_nonfrozen_model_params(recipe, family):
    with pytest.raises(ValueError):
        build_experiment_arms(CandidateSpec(recipe, CORE_COLUMNS, family, (("C", 7.),)))


@pytest.mark.parametrize("recipe", ["feat_meter_deviation_ratio", "feat_frequency_encoding", "feat_histogram_quantile"])
def test_unknown_method_parameter_rejected_before_assembly(recipe):
    with pytest.raises(ValueError):
        build_experiment_arms(CandidateSpec(recipe, CORE_COLUMNS, "hist_gradient_boosting", (), (("inventedkey", 1),)))


def test_ag_settings_do_not_reach_control():
    arms = build_experiment_arms(CandidateSpec("model_autogluon_core_features", CORE_COLUMNS, "autogluon", (("time_limit", 7.),)))
    assert arms["control"].named_steps["model"].max_iter == 100
    assert arms["challenger"].transformer_or_pipeline.settings.time_limit == 7.


def test_fitting_reuses_recipe_not_fitted_objects():
    arm = build_experiment_arms(CandidateSpec("feat_conditional_cross_fitted_target_encoding", CORE_COLUMNS, "hist_gradient_boosting", ()))["challenger"]
    counter = _make_counter(2)
    first = fit_arm(arm, training_view(synthetic_partitions()[0]), counter)
    second = fit_arm(arm, training_view(synthetic_partitions()[1]), counter)
    assert first.transformer_or_pipeline is not second.transformer_or_pipeline
    assert not hasattr(arm.named_steps["features"], "full_training_map_")


def test_model_mutation_invalidates_fitted_arm():
    arm = build_experiment_arms(CandidateSpec("feat_meter_deviation_ratio", CORE_COLUMNS, "hist_gradient_boosting", ()))["challenger"]
    fitted = fit_arm(arm, _training_partition(), _make_counter(1))
    fitted.named_steps["model"].max_iter = 3
    with pytest.raises(ValueError, match="configuration changed"):
        fitted.predict_proba(_training_partition().X_train)


def test_selector_requires_two_supervised_fits_before_start():
    arm = build_experiment_arms(CandidateSpec("dim_fold_local_selector_vs_passthrough", CORE_COLUMNS, "logistic_l2", ()))["selector"]
    counter = _make_counter(1)
    with pytest.raises(BlockedExperimentError):
        fit_arm(arm, _training_partition(), counter)
    assert counter.used_model_fits == 0


def test_configured_hpo_remains_blocked():
    from project.experiments.candidate_dispatch import HGBTuningStrategy
    strategy = HGBTuningStrategy(object(), "/unused", CandidateSpec("regime_same_family_hpo_vs_default", CORE_COLUMNS, "hist_gradient_boosting", ()))
    with pytest.raises(BlockedExperimentError) as error:
        strategy.tune(_training_partition(), _make_counter(1))
    assert error.value.code == "hpo_not_ready"


def test_selector_failure_debits_only_selector_attempt(monkeypatch):
    from sklearn.feature_selection import SelectFromModel
    arm = build_experiment_arms(CandidateSpec("dim_fold_local_selector_vs_passthrough", CORE_COLUMNS, "logistic_l2", ()))["selector"]
    def fail(*args, **kwargs):
        raise RuntimeError("selector failed")
    monkeypatch.setattr(SelectFromModel, "fit", fail)
    counter = _make_counter(2)
    with pytest.raises(RuntimeError, match="selector failed"):
        fit_arm(arm, _training_partition(), counter)
    assert counter.used_model_fits == 1


@pytest.mark.parametrize("key,value", [("n_bins", True), ("n_bins", 65), ("n_quantiles", 257), ("n_bins", 3.5)])
def test_distribution_rejects_invalid_typed_bounds(key, value):
    with pytest.raises(ValueError):
        build_experiment_arms(CandidateSpec("feat_histogram_quantile", CORE_COLUMNS, "hist_gradient_boosting", (), ((key, value),)))


@pytest.mark.parametrize("step,param,value", [("preprocess", "remainder", "passthrough"), ("features", "n_bins", 4), ("model", "random_state", 3)])
def test_full_graph_mutation_guard(step, param, value):
    arm = build_experiment_arms(CandidateSpec("feat_histogram_quantile", CORE_COLUMNS, "hist_gradient_boosting", ()))["bin"]
    fitted = fit_arm(arm, _training_partition(), _make_counter(1))
    setattr(fitted.named_steps[step], param, value)
    with pytest.raises(ValueError, match="configuration changed"):
        fitted.predict(_training_partition().X_train)


def test_margin_arm_does_not_expose_probability_method():
    arm = build_experiment_arms(CandidateSpec("dim_pca_vs_passthrough", CORE_COLUMNS, "svm", ()))["pca"]
    assert not hasattr(arm, "predict_proba")


@pytest.mark.parametrize("recipe", ALL_SUPPORTED_EXPERIMENT_IDS)
def test_every_recipe_rejects_unknown_method_keys(recipe):
    family = {"dim_pca_vs_passthrough": "svm", "dim_fold_local_selector_vs_passthrough": "logistic_l2", "model_autogluon_core_features": "autogluon", "model_extratrees_core_features": "extratrees"}.get(recipe, "hist_gradient_boosting")
    with pytest.raises(ValueError, match="unknown method"):
        build_experiment_arms(CandidateSpec(recipe, CORE_COLUMNS, family, (), (("invented", 1),)))


def test_selector_counts_each_attempt_in_order(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    arm = build_experiment_arms(CandidateSpec("dim_fold_local_selector_vs_passthrough", CORE_COLUMNS, "logistic_l2", ()))["selector"]
    counter = _make_counter(2)
    observed = []
    original = LogisticRegression.fit
    def recording(model, X, y, **kwargs):
        observed.append((model.solver, counter.used_model_fits))
        return original(model, X, y, **kwargs)
    monkeypatch.setattr(LogisticRegression, "fit", recording)
    fitted = fit_arm(arm, _training_partition(), counter)
    assert observed == [("liblinear", 1), ("lbfgs", 2)]
    assert fitted.fit_cost_metadata["supervised_model_fits"] == 2
    assert fitted.predict_proba(_training_partition().X_train).shape == (8, 2)
