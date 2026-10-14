"""Synthetic factory wiring only; no model fitting or run admission."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from candidate_fixtures import CORE_COLUMNS, core_frame
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.candidate_dispatch import (
    _configuration,
    build_experiment_arms,
    resolve_candidate_spec,
)
from project.features.input_encoding import (
    ABSOLUTE_CALENDAR_YEAR,
    LEAD_YEAR_OFFSET_1900_UINT8_V1,
)
from project.features.registry import build_lead_baseline_registry
from project.models.candidate_factories import build_candidate_arms

ENCODED = LEAD_YEAR_OFFSET_1900_UINT8_V1


def config_hash(arm):
    return hashlib.sha256(json.dumps(
        _configuration(arm.transformer_or_pipeline), sort_keys=True, default=str,
    ).encode()).hexdigest()


@pytest.mark.parametrize("recipe", [
    "feat_missingness_indicators", "feat_missingness_bitmask",
    "feat_safe_calendar_numeric_transforms",
])
def test_dispatch_explicit_encoding_reaches_transformer_registry_and_hash(recipe):
    spec = CandidateSpec(recipe, CORE_COLUMNS, "hist_gradient_boosting", ())
    default = build_experiment_arms(spec)
    encoded = build_experiment_arms(spec, year_built_encoding=ENCODED)
    resolved = resolve_candidate_spec(spec, year_built_encoding=ENCODED)
    raw = core_frame().iloc[:4].copy()
    raw["year_built"] = pd.array([0, 100, 255, None], dtype="Float64")
    raw["year"] = 2020
    legacy_schema_output = default["control"].named_steps["schema"].fit_transform(raw)
    for arms in (encoded, resolved.arms):
        features = arms["challenger"].named_steps["features"]
        assert features.get_params()["year_built_encoding"] == ENCODED
        output = features.fit_transform(raw)
        pd.testing.assert_frame_equal(output[list(raw.columns)], raw)
        assert arms["control"].raw_pool == arms["challenger"].raw_pool == CORE_COLUMNS
        for arm in arms.values():
            assert ENCODED in arm.authoritative_registry.get("year_built").semantic_meaning
            pd.testing.assert_frame_equal(arm.named_steps["schema"].fit_transform(raw), legacy_schema_output)
        if recipe == "feat_missingness_indicators":
            assert output.year_built_missing.tolist() == [0, 0, 1, 1]
        elif recipe == "feat_missingness_bitmask":
            assert ((output.missingness_bitmask.to_numpy() >> 2) & 1).tolist() == [0, 0, 1, 1]
        else:
            np.testing.assert_allclose(output.building_age, [120, 20, np.nan, np.nan])
    assert default["challenger"].named_steps["features"].get_params()["year_built_encoding"] == ABSOLUTE_CALENDAR_YEAR
    assert config_hash(encoded["challenger"]) != config_hash(default["challenger"])
    assert config_hash(encoded["control"]) == config_hash(default["control"])
    if recipe == "feat_missingness_bitmask":
        legacy = default["challenger"].named_steps["features"].fit_transform(raw)
        assert ((legacy.missingness_bitmask.to_numpy() >> 2) & 1).tolist() == [0, 0, 0, 1]
    else:
        with pytest.raises(ValueError):
            default["challenger"].named_steps["features"].fit(raw)


def test_task7_factory_accepts_matching_explicit_encoding_metadata_without_raw_changes():
    spec = CandidateSpec("feat_missingness_passthrough_comparison", CORE_COLUMNS, "hist_gradient_boosting", ())
    registry = build_lead_baseline_registry(ENCODED)
    arms = build_candidate_arms(spec, registry, year_built_encoding=ENCODED)
    dispatched = build_experiment_arms(spec, year_built_encoding=ENCODED)
    raw = core_frame().assign(year_built=0)
    legacy = build_candidate_arms(spec, build_lead_baseline_registry())
    legacy_schema_output = legacy["control"].named_steps["schema"].fit_transform(raw)
    for arm_id, arm in arms.items():
        pd.testing.assert_frame_equal(arm.named_steps["schema"].fit_transform(raw), legacy_schema_output)
        assert _configuration(arm.named_steps["preprocess"]) == _configuration(legacy[arm_id].named_steps["preprocess"])
    assert all(ENCODED in arm.authoritative_registry.get("year_built").semantic_meaning
               for arm in dispatched.values())
    with pytest.raises(ValueError, match="authoritative"):
        build_candidate_arms(spec, registry)


def test_unknown_encoding_context_rejected_without_spec_method_parameters():
    spec = CandidateSpec("feat_missingness_indicators", CORE_COLUMNS, "hist_gradient_boosting", ())
    assert spec.method_params == ()
    with pytest.raises(ValueError, match="encoding"):
        build_experiment_arms(spec, year_built_encoding="guess")
