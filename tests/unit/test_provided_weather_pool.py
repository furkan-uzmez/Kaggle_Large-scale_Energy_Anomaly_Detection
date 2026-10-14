"""TDD contracts for verified 7-weather provided feature pool."""
from __future__ import annotations

import pytest

from project.experiments.candidate_contracts import BlockedExperimentError, CandidateSpec
from project.experiments.candidate_dispatch import build_experiment_arms
from project.features.provided import WEATHER_FEATURES, build_provided_feature_registry, resolve_provided_pool
from project.features.registry import (
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    build_lead_baseline_registry,
)

CORE = build_lead_baseline_registry().names
EXPECTED_WEATHER = (
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
)


def test_weather_features_constant_matches_expected_seven():
    assert WEATHER_FEATURES == EXPECTED_WEATHER


def test_provided_feature_registry_contains_sixteen_features():
    registry = build_provided_feature_registry()
    assert isinstance(registry, FeatureRegistry)
    assert len(registry.names) == 16
    assert registry.names == (*CORE, *EXPECTED_WEATHER)
    for name in EXPECTED_WEATHER:
        meta = registry.get(name)
        assert isinstance(meta, FeatureMetadata)
        assert meta.name == name
        assert meta.family == "weather_physical"
        assert meta.source_columns == (name,)
        assert meta.provenance_status == ProvenanceStatus.VERIFIED
        assert meta.prediction_time_availability == AvailabilityStatus.AVAILABLE
        assert meta.target_dependence == TargetDependence.INDEPENDENT


def test_resolve_provided_pool_successfully_resolves_all_sixteen():
    registry = build_provided_feature_registry()
    pool = resolve_provided_pool(CORE, (*CORE, *WEATHER_FEATURES), registry)
    assert pool == (*CORE, *WEATHER_FEATURES)
    assert len(pool) == 16


@pytest.mark.parametrize("invalid_feature", [
    "air_temperature_mean_lag7",
    "air_temperature_max_lag73",
    "gte_building_hour",
    "site_id",
    "anomaly",
    "unverified_sensor",
])
def test_resolve_provided_pool_rejects_leaked_or_unverified_additions(invalid_feature):
    registry = build_provided_feature_registry()
    with pytest.raises(BlockedExperimentError):
        resolve_provided_pool(CORE, (*CORE, invalid_feature), registry)


def test_candidate_arms_built_with_sixteen_provided_features():
    registry = build_provided_feature_registry()
    spec = CandidateSpec(
        experiment_id="feat_core_vs_provided",
        feature_pool=(*CORE, *WEATHER_FEATURES),
        model_family="hist_gradient_boosting",
        model_params=(("early_stopping", False), ("random_state", 42)),
        method_params=(),
    )
    arms = build_experiment_arms(spec, registry=registry)
    assert set(arms) == {"control", "challenger"}
    assert arms["control"].raw_pool == CORE
    assert arms["challenger"].raw_pool == (*CORE, *WEATHER_FEATURES)
    assert arms["control"].score_kind == "positive_probability"
    assert arms["challenger"].score_kind == "positive_probability"
