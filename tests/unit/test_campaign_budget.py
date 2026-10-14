"""Pure admission checks cannot borrow baseline compute approval."""

import copy
import importlib
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def api():
    name = "project.experiments.campaign_budget"
    assert importlib.util.find_spec(name), "campaign admission API is missing"
    return importlib.import_module(name)


def configs():
    return tuple(yaml.safe_load((ROOT / path).read_text()) for path in
                 ("configs/experiments/feature_campaign.yaml", "configs/resource_budget.yaml"))


def admit(campaign, budget):
    baseline = yaml.safe_load((ROOT / "configs/experiments/baselines.yaml").read_text())
    return api().validate_admission(campaign, budget, baseline)


def smoke_budget():
    return {"scope": "campaign_smoke", "mode": "smoke", "execution_authorized": True,
            "host_preflight_status": "approved", "user_approval_status": "approved",
            "maximum_cardinality": 10000, "max_ram_gb": 1,
            "max_wall_time_seconds": 120, "max_model_fits": 3, "max_hpo_trials": 0,
            "max_seed_count": 1, "max_fold_count": 1, "cpu_threads": 2,
            "device": "cpu", "rss_measurement": "process_tree"}


def test_design_valid_but_not_executable():
    report = admit(*configs())
    assert report["design_valid"] is True
    assert report["executable"] is False
    assert report["status"] == "blocked_unapproved_compute"


def test_only_smoke_approval_admits_without_running():
    campaign, _ = configs()
    campaign.update(mode="smoke", execution_authorized=True)
    report = admit(campaign, smoke_budget())
    assert report["executable"] is True
    assert report["status"] == "admitted_smoke_contract_only"


@pytest.mark.parametrize("key,value", [
    ("maximum_cardinality", True), ("max_model_fits", 1.5),
    ("max_ram_gb", float("nan")), ("max_wall_time_seconds", float("inf")),
    ("max_seed_count", 0), ("max_hpo_trials", -1), ("max_hpo_trials", True),
    ("maximum_cardinality", 10001), ("max_ram_gb", 2), ("max_wall_time_seconds", 121),
    ("max_model_fits", 4), ("max_fold_count", 2), ("cpu_threads", 3),
    ("device", "cuda"), ("rss_measurement", "process"),
])
def test_bad_caps_fail_closed(key, value):
    campaign, _ = configs()
    campaign.update(mode="smoke", execution_authorized=True)
    budget = smoke_budget()
    budget[key] = value
    report = admit(campaign, budget)
    assert not report["executable"]
    assert report["errors"]


def test_baseline_approval_and_full_campaign_cannot_authorize_smoke():
    campaign, _ = configs()
    campaign.update(mode="smoke", execution_authorized=True)
    baseline = yaml.safe_load((ROOT / "configs/experiments/baseline_execution_budget.yaml").read_text())
    assert not admit(campaign, baseline)["executable"]
    budget = smoke_budget()
    budget.update(scope="campaign_full", mode="full")
    assert not admit(campaign, budget)["executable"]


@pytest.mark.parametrize("key,value", [("execution_authorized", "true"),
                                        ("host_preflight_status", "pending"),
                                        ("user_approval_status", "pending")])
def test_permission_gates_are_independent(key, value):
    campaign, _ = configs()
    campaign.update(mode="smoke", execution_authorized=True)
    budget = smoke_budget()
    budget[key] = value
    assert not admit(campaign, budget)["executable"]


def test_duplicate_ids_and_na_without_reason_are_invalid_design():
    campaign, budget = configs()
    campaign["experiment_matrix"].append(copy.deepcopy(campaign["experiment_matrix"][0]))
    assert not admit(campaign, budget)["design_valid"]
    campaign, budget = configs()
    row = next(r for r in campaign["experiment_matrix"] if r["status"] == "N/A")
    row.pop("rationale")
    assert not admit(campaign, budget)["design_valid"]


def test_active_split_hashes_memberships_and_support():
    report = api().validate_split_contract(ROOT, configs()[0]["prediction_contract"])
    assert report["valid"], report
    assert report["repeat_count"] == 3
    assert report["vault_building_count"] == 40


def test_split_contract_requires_metadata(tmp_path):
    contract = configs()[0]["prediction_contract"]
    report = api().validate_split_contract(tmp_path, contract)
    assert not report["valid"]


def test_invalid_closed_execution_status_is_not_design_valid():
    campaign, budget = configs()
    campaign["experiment_matrix"][0]["execution_status"] = "completed"
    assert not admit(campaign, budget)["design_valid"]


def test_unknown_model_family_is_not_design_valid():
    campaign, budget = configs()
    campaign["experiment_matrix"][0]["model_family"] = "invented"
    assert not admit(campaign, budget)["design_valid"]


@pytest.mark.parametrize("mutation", ["master_hash", "repeat_hash", "vault", "support", "building_support"])
def test_split_metadata_tampering_fails_closed(tmp_path, mutation):
    import json
    import shutil

    destination = tmp_path / "outputs/splits"
    destination.mkdir(parents=True)
    for source in (ROOT / "outputs/splits").iterdir():
        shutil.copyfile(source, destination / source.name)
    path = destination / "split_metadata_repeated.json"
    metadata = json.loads(path.read_text())
    if mutation == "master_hash":
        metadata["family_sha256"] = "0" * 64
    elif mutation == "repeat_hash":
        metadata["per_repeat"][0]["sha256"] = "0" * 64
    elif mutation == "vault":
        metadata["vault_building_ids"][0] = -100
    elif mutation == "support":
        metadata["per_repeat"][0]["class_support"]["development_folds"][0]["positive_rows"] += 1
    else:
        metadata["per_repeat"][0]["per_building_support"][0]["anomaly_count"] += 1
    path.write_text(json.dumps(metadata))
    report = api().validate_split_contract(tmp_path, configs()[0]["prediction_contract"])
    assert not report["valid"]


def test_non_mapping_configs_fail_closed():
    assert not api().validate_admission([], {})["design_valid"]
    assert not api().validate_admission(configs()[0], None)["executable"]


@pytest.mark.parametrize("key,value", [("target", "other"), ("positive_label", True),
                                     ("primary_metric", "accuracy"), ("guardrails", [])])
def test_prediction_contract_is_frozen(key, value):
    campaign, budget = configs()
    campaign["prediction_contract"][key] = value
    assert not admit(campaign, budget)["design_valid"]


@pytest.mark.parametrize("key", ["prediction_contract", "model_family_audit", "feature_sets"])
@pytest.mark.parametrize("value", [[], None, "bad"])
def test_nested_mapping_shapes_fail_closed(key, value):
    campaign, budget = configs()
    campaign[key] = value
    report = admit(campaign, budget)
    assert report["design_valid"] is False
    assert report["executable"] is False
    assert report["errors"] and report["design_errors"]


def test_split_contract_list_reports_error():
    report = api().validate_split_contract(ROOT, [])
    assert report["valid"] is False
    assert report["errors"]
