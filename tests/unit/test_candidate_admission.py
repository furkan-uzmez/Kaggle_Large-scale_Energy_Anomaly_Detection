"""Pure admission tests: generated configuration only, no training or dataset IO."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from candidate_fixtures import CORE_COLUMNS
from project.experiments.candidate_contracts import CandidateSpec, FitBudget, RunSpec


def api():
    from project.experiments import candidate_admission
    return candidate_admission


def budget(fits=4, trials=0):
    return FitBudget(fits, trials, 30., 256 * 1024**2, 1)


def packet(recipe="feat_meter_deviation_ratio", family="hist_gradient_boosting", mode="synthetic_smoke", caps=None, methods=()):
    caps = caps or budget()
    spec = RunSpec(1, (CandidateSpec(recipe, CORE_COLUMNS, family, (), methods),), caps,
                   mode, "notebooks/02_baseline_and_candidate_design.ipynb", "matrix-cell")
    campaign = {
        "schema_version": 1, "campaign_id": "task7_feature_campaign",
        "mode": "design_only", "execution_authorized": False,
        "baseline_config": "configs/experiments/baselines.yaml",
        "resource_budget": "configs/resource_budget.yaml",
        "prediction_contract": {"target": "anomaly", "positive_label": 1,
                                "folds": 4, "repeats": 3, "split_seeds": [42, 43, 44]},
        "experiment_matrix": [{"id": recipe, "model_family": family,
                               "status": "planned", "execution_status": "blocked_unapproved_compute"}],
    }
    digest = hashlib.sha256(json.dumps(campaign, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    source = {
        "campaign": campaign, "campaign_sha256": digest,
        "candidate_links": {recipe: {"source_notebook": spec.source_notebook,
                                    "source_cell_id": spec.source_cell_id,
                                    "campaign_sha256": digest}},
        "diagnostic": {"generator": "candidate_fixtures_v1", "max_rows": 16,
                       "groups": 4, "folds": 2, "repeats": 1, "cpu_threads": 1},
    }
    return spec, source, caps


def codes(report):
    return {b.code for b in report.blockers}


def test_valid_diagnostic_is_readiness_not_permission():
    report = api().validate_candidate_admission(*packet())
    assert report.design_valid and report.diagnostic_ready
    assert report.planned_model_fits == 4
    assert report.executed is False and report.execution_authorized is False
    assert report.n03_eligible is False and report.data_access_authorized is False
    assert report.source_provenance == "externally_supplied_unverified"
    json.dumps(report.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        report.executed = True


@pytest.mark.parametrize("authorize", [False, True])
def test_full_always_blocked_before_execution_boundaries(monkeypatch, authorize):
    import pathlib

    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    import project.experiments.candidate_dispatch as dispatch
    def forbidden(*args, **kwargs):
        pytest.fail("admission crossed an execution boundary")
    monkeypatch.setattr(dispatch, "resolve_candidate_spec", forbidden)
    monkeypatch.setattr(dispatch, "build_experiment_arms", forbidden)
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pathlib.Path, "mkdir", forbidden)
    spec, source, caps = packet(mode="full")
    source["campaign"]["execution_authorized"] = authorize
    report = api().validate_candidate_admission(spec, source, caps)
    assert "production_execution_blocked" in codes(report)
    assert not report.execution_authorized and not report.executed


def test_missing_source_returns_blockers_without_building(monkeypatch):
    import project.experiments.candidate_dispatch as dispatch
    monkeypatch.setattr(dispatch, "build_experiment_arms", lambda *a, **k: pytest.fail("built models"))
    spec, _, caps = packet()
    report = api().validate_candidate_admission(spec, {}, caps)
    assert "missing_source_design" in codes(report)
    assert not report.design_valid


@pytest.mark.parametrize("field,value", [("max_rows", 129), ("max_rows", True), ("groups", 3), ("folds", 4), ("repeats", 3), ("cpu_threads", 2), ("generator", "trust_my_synthetic")])
def test_diagnostic_protocol_strict(field, value):
    spec, source, caps = packet()
    source["diagnostic"][field] = value
    assert "invalid_diagnostic_protocol" in codes(api().validate_candidate_admission(spec, source, caps))


@pytest.mark.parametrize("key", ["data_path", "split_path", "prepared_data", "execution_authorized"])
def test_external_paths_or_authorization_flags_rejected(key):
    spec, source, caps = packet()
    source[key] = "caller-input"
    assert "unsupported_source_design_fields" in codes(api().validate_candidate_admission(spec, source, caps))


def test_source_hash_and_cell_linkage():
    spec, source, caps = packet()
    source["candidate_links"][spec.candidates[0].experiment_id]["source_cell_id"] = "forged-cell"
    assert "source_linkage_mismatch" in codes(api().validate_candidate_admission(spec, source, caps))
    source["campaign"]["campaign_id"] = "changed"
    assert "source_hash_mismatch" in codes(api().validate_candidate_admission(spec, source, caps))


def test_budget_equality_and_crossfold_affordability():
    spec, source, _caps = packet()
    assert "budget_mismatch" in codes(api().validate_candidate_admission(spec, source, budget(5)))
    small = budget(3)
    report = api().validate_candidate_admission(replace(spec, budget=small), source, small)
    assert report.planned_model_fits == 4
    assert "fit_budget_exhausted" in codes(report)


def test_selector_counts_control_plus_two_challenger_fits_per_fold():
    report = api().validate_candidate_admission(*packet("dim_fold_local_selector_vs_passthrough", "logistic_l2", caps=budget(6)))
    assert report.planned_model_fits == 6 and report.diagnostic_ready


@pytest.mark.parametrize("recipe,code", [("invented", "unknown_experiment"), ("prohibited_opaque_building_id_digit_extraction", "prohibited_experiment"), ("blocked_unverified_gte_family", "blocked_unverified_gte"), ("feat_group_aggregates_blocked", "blocked_group_aggregates"), ("feat_core_vs_provided", "no_verified_additional_features"), ("feat_preserve_raw_vs_invalid_sentinel", "missing_evidence_rule"), ("regime_same_family_hpo_vs_default", "hpo_not_ready"), ("model_autogluon_core_features", "supervisor_unavailable")])
def test_explicit_candidate_blockers(recipe, code):
    family = "autogluon" if recipe.startswith("model_autogluon") else "hist_gradient_boosting"
    assert code in codes(api().validate_candidate_admission(*packet(recipe, family)))


def test_configured_hpo_fit_arithmetic_does_not_authorize():
    methods = (("study_spec", (("n_trials", 2), ("inner_folds", 2))), ("study_dir", "unused"))
    report = api().validate_candidate_admission(*packet("regime_same_family_hpo_vs_default", caps=budget(12, 4), methods=methods))
    assert report.planned_model_fits == 12 and report.planned_trials == 4
    assert "hpo_not_ready" in codes(report) and not report.diagnostic_ready


@pytest.mark.parametrize("methods", [(("invented", 1),), (("n_bins", True),), (("n_bins", 65),)])
def test_invalid_recipe_methods_are_not_diagnostic_ready(methods):
    recipe = "feat_histogram_quantile" if methods[0][0] == "n_bins" else "feat_meter_deviation_ratio"
    report = api().validate_candidate_admission(*packet(recipe, caps=budget(6), methods=methods))
    assert "invalid_candidate_parameters" in codes(report)
    assert not report.design_valid and not report.diagnostic_ready


def test_nonfrozen_hgb_params_are_not_ready():
    spec, source, caps = packet()
    candidate = replace(spec.candidates[0], model_params=(("max_iter", 3),))
    report = api().validate_candidate_admission(replace(spec, candidates=(candidate,)), source, caps)
    assert "invalid_candidate_parameters" in codes(report)


@pytest.mark.parametrize("key,value", [("max_model_fits", True), ("max_trials", -1), ("max_wall_seconds", float("nan")), ("max_rss_bytes", 0), ("cpu_threads", False)])
def test_mutated_caps_revalidated_without_contract_internals(key, value):
    spec, source, caps = packet()
    object.__setattr__(caps, key, value)
    report = api().validate_candidate_admission(spec, source, caps)
    assert "invalid_budget" in codes(report) and not report.diagnostic_ready


def test_synthetic_validation_never_constructs_models_or_outputs(monkeypatch):
    import pathlib

    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    import project.experiments.candidate_dispatch as dispatch
    spec, source, caps = packet()
    def forbidden(*args, **kwargs):
        pytest.fail("pure admission performed IO or constructed/fitted a model")
    monkeypatch.setattr(dispatch, "resolve_candidate_spec", forbidden)
    monkeypatch.setattr(dispatch, "build_experiment_arms", forbidden)
    monkeypatch.setattr(HistGradientBoostingClassifier, "__init__", forbidden)
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pathlib.Path, "mkdir", forbidden)
    monkeypatch.setattr(pathlib.Path, "write_text", forbidden)
    monkeypatch.setattr(pathlib.Path, "open", forbidden)
    assert api().validate_candidate_admission(spec, source, caps).diagnostic_ready


def test_configured_sentinel_is_unknown_evidence_not_approval():
    methods = (("column", "meter_reading"), ("invalid_values", (-999.,)),
               ("replacement_value", -1.), ("evidence_reference", "unopened.json"),
               ("evidence_sha256", "0" * 64))
    report = api().validate_candidate_admission(*packet("feat_preserve_raw_vs_invalid_sentinel", methods=methods))
    assert "sentinel_evidence_unverified" in codes(report)
    assert not report.diagnostic_ready


@pytest.mark.parametrize("capability,code", [("unavailable", "optional_dependency_unavailable"), ("unsupported", "optional_capability_unsupported"), ("available", "supervisor_unavailable")])
def test_optional_capability_never_grants_supervision(capability, code):
    spec, source, caps = packet("model_autogluon_core_features", "autogluon")
    source["optional_capabilities"] = {"autogluon": capability}
    report = api().validate_candidate_admission(spec, source, caps)
    assert code in codes(report) and "supervisor_unavailable" in codes(report)
    assert not report.diagnostic_ready


def test_multiple_candidates_sum_all_folds_before_any_fit():
    spec, source, caps = packet(caps=budget(9))
    candidate = CandidateSpec("feat_histogram_quantile", CORE_COLUMNS, "hist_gradient_boosting", ())
    spec = replace(spec, candidates=(*spec.candidates, candidate))
    source["campaign"]["experiment_matrix"].append({"id": candidate.experiment_id, "model_family": candidate.model_family, "status": "planned", "execution_status": "blocked_unapproved_compute"})
    digest = hashlib.sha256(json.dumps(source["campaign"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    source["campaign_sha256"] = digest
    source["candidate_links"] = {c.experiment_id: {"source_notebook": spec.source_notebook, "source_cell_id": spec.source_cell_id, "campaign_sha256": digest} for c in spec.candidates}
    report = api().validate_candidate_admission(spec, source, caps)
    assert report.planned_model_fits == 10
    assert "fit_budget_exhausted" in codes(report)


def test_malformed_candidate_packet_fails_closed_with_report():
    spec, source, caps = packet()
    object.__setattr__(spec.candidates[0], "method_params", ("bad",))
    report = api().validate_candidate_admission(spec, source, caps)
    assert "invalid_candidate_parameters" in codes(report)
    assert not report.diagnostic_ready


def test_foreign_raw_source_and_group_history_are_blocked():
    spec, source, caps = packet()
    for pool, code in [((*CORE_COLUMNS, "temperature_mean"), "unverified_raw_source"), ((*CORE_COLUMNS, "history_meter_mean"), "blocked_group_aggregates")]:
        altered = replace(spec, candidates=(replace(spec.candidates[0], feature_pool=pool),))
        assert code in codes(api().validate_candidate_admission(altered, source, caps))


def test_source_payload_nonstring_json_keys_are_not_accepted():
    spec, source, caps = packet()
    source["campaign"]["extra"] = {1: "silently_coerced_key"}
    digest = hashlib.sha256(json.dumps(source["campaign"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    source["campaign_sha256"] = digest
    source["candidate_links"][spec.candidates[0].experiment_id]["campaign_sha256"] = digest
    report = api().validate_candidate_admission(spec, source, caps)
    assert "invalid_source_design" in codes(report)


def test_real_small_design_matrix_can_be_validated_as_supplied_mapping():
    from pathlib import Path

    import yaml

    spec, source, caps = packet()
    path = Path(__file__).resolve().parents[2] / "configs/experiments/feature_campaign.yaml"
    # Explicit small design configuration only. This never opens datasets/notebooks.
    campaign = yaml.safe_load(path.read_text())
    digest = hashlib.sha256(json.dumps(campaign, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    source["campaign"] = campaign
    source["campaign_sha256"] = digest
    source["candidate_links"][spec.candidates[0].experiment_id]["campaign_sha256"] = digest
    report = api().validate_candidate_admission(spec, source, caps)
    assert report.design_valid and report.diagnostic_ready
    assert report.source_provenance == "externally_supplied_unverified"
    detached = report.to_dict()
    detached["candidates"][0]["disposition"] = "forged"
    assert report.candidates[0].disposition == "diagnostic_config_ready"


def test_consistent_full_design_is_valid_but_never_execution_ready():
    spec, source, caps = packet(mode="full", caps=budget(24))
    source.pop("diagnostic")
    report = api().validate_candidate_admission(spec, source, caps)
    assert report.design_valid and report.partition_count == 12
    assert report.planned_model_fits == 24
    assert "production_execution_blocked" in codes(report)
    assert not report.diagnostic_ready and not report.execution_authorized


def test_full_and_diagnostic_protocols_cannot_be_mixed():
    report = api().validate_candidate_admission(*packet(mode="full"))
    assert "mixed_execution_protocol" in codes(report)


def test_ratio_requires_complete_raw_core_pool():
    spec, source, caps = packet()
    spec = replace(spec, candidates=(replace(spec.candidates[0], feature_pool=("meter_reading",)),))
    report = api().validate_candidate_admission(spec, source, caps)
    assert "invalid_candidate_parameters" in codes(report)
    assert not report.diagnostic_ready


def test_encoder_groups_must_support_each_outer_training_partition():
    report = api().validate_candidate_admission(*packet("feat_conditional_cross_fitted_target_encoding"))
    assert "insufficient_inner_groups" in codes(report)
    assert not report.diagnostic_ready
