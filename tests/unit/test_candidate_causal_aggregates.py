"""TDD contracts for feat_causal_group_aggregates candidate dispatch and pipeline."""
from __future__ import annotations



from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.candidate_dispatch import build_experiment_arms
from project.features.causal_aggregates import build_causal_aggregates_registry
from project.features.registry import build_lead_baseline_registry

CORE = build_lead_baseline_registry().names


def test_build_causal_aggregates_registry_contains_rolling_features():
    registry = build_causal_aggregates_registry()
    expected = [
        "meter_reading_roll_mean_24h",
        "meter_reading_roll_std_24h",
        "meter_reading_roll_mean_168h",
        "meter_reading_roll_std_168h",
    ]
    for name in expected:
        meta = registry.get(name)
        assert meta is not None
        assert meta.family == "causal_aggregates"
        assert meta.target_dependence.name == "INDEPENDENT"


def test_build_experiment_arms_for_causal_group_aggregates():
    registry = build_causal_aggregates_registry()
    spec = CandidateSpec(
        experiment_id="feat_causal_group_aggregates",
        feature_pool=CORE,
        model_family="hist_gradient_boosting",
        model_params=(("early_stopping", False), ("random_state", 42)),
        method_params=(("windows", (24, 168)),),
    )
    arms = build_experiment_arms(spec, registry=registry)
    assert set(arms) == {"control", "challenger"}
    assert arms["control"].score_kind == "positive_probability"
    assert arms["challenger"].score_kind == "positive_probability"

    # Verify challenger pipeline contains the causal transformer
    challenger_pipe = arms["challenger"].transformer_or_pipeline
    assert "features" in challenger_pipe.named_steps
