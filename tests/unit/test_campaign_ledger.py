"""Campaign provenance, completion lifecycle, and locked append regressions."""

import copy
import hashlib
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from project.features.registry import (
    AvailabilityStatus,
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    build_lead_baseline_registry,
)

ROOT = Path(__file__).resolve().parents[2]


def api():
    return importlib.import_module("project.experiments.campaign_ledger")


@pytest.fixture
def context():
    return yaml.safe_load((ROOT / "configs/experiments/feature_campaign.yaml").read_text())


@pytest.fixture
def project(tmp_path, context):
    (tmp_path / "outputs/runs").mkdir(parents=True)
    config = tmp_path / "configs/experiments/feature_campaign.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(context))
    notebook = tmp_path / "notebooks/02_baseline_and_candidate_design.ipynb"
    notebook.parent.mkdir()
    notebook.write_text(json.dumps({"cells": [{"id": "proposal-cell", "cell_type": "markdown",
                                               "source": ["Candidate design"]}]}))
    return tmp_path


@pytest.fixture
def proposal():
    return {
        "proposal_id": "proposal-missingness-01",
        "experiment_id": "feat_missingness_indicators",
        "config_reference": "configs/experiments/feature_campaign.yaml",
        "source_reference": {"path": "notebooks/02_baseline_and_candidate_design.ipynb",
                             "cell_id": "proposal-cell"},
        "criteria": "Fold-local only; compare against frozen baseline",
        "disposition": "planned", "reason": "Pending implementation and approval",
        "predictors": ["meter_reading"], "prediction_time": "2026-10-02T00:00:00+00:00",
        "run_id": None, "outcome": None, "artifacts": [], "review": None,
    }


def registry():
    return build_lead_baseline_registry()


def ledger(project, context):
    return api().CampaignLedger(project / "outputs/runs/campaign_ledger.jsonl", project, context)


def completed(project, proposal):
    run = project / "outputs/runs/real"
    run.mkdir()
    status = {"run_id": "real", "experiment_id": proposal["experiment_id"], "status": "completed"}
    status_path = run / "run_status.json"
    status_path.write_text(json.dumps(status))
    artifact = run / "metrics.json"
    artifact.write_text('{"roc_auc":0.5}')
    record = copy.deepcopy(proposal)
    record.update(run_id="real", outcome="completed",
                  run_reference="outputs/runs/real/run_status.json",
                  run_status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(),
                  artifacts=[{"path": "outputs/runs/real/metrics.json",
                              "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
                  review={"status": "pending"})
    return record


def test_append_preserves_baseline_and_rejects_duplicate_batch(project, context, proposal):
    target = ledger(project, context)
    inventory = project / "outputs/runs/experiment_inventory.jsonl"
    inventory.write_bytes(b'{"baseline":"keep"}\n')
    target.append([proposal], registry())
    original = target.path.read_bytes()
    assert json.loads(original)["experiment_id"] == proposal["experiment_id"]
    assert inventory.read_bytes() == b'{"baseline":"keep"}\n'
    with pytest.raises(ValueError, match="duplicate"):
        target.append([proposal], registry())
    new = dict(proposal, proposal_id="new")
    with pytest.raises(ValueError, match="duplicate"):
        target.append([new, new], registry())
    assert target.path.read_bytes() == original


@pytest.mark.parametrize("change,match", [
    ({"experiment_id": "N02"}, "experiment"),
    ({"disposition": "excluded", "reason": " "}, "reason"),
    ({"disposition": "N/A", "reason": None}, "reason|disposition"),
    ({"disposition": "completed"}, "disposition"),
    ({"outcome": "completed", "run_id": "fake"}, "completion"),
    ({"source_reference": {"path": "notebooks/nonexistent.ipynb", "cell_id": "x"}}, "source"),
    ({"predictors": ["gte_probe"]}, "predictor"),
    ({"source_reference": None}, "source"),
    ({"source_reference": {"path": "notebooks/02_baseline_and_candidate_design.ipynb"}}, "cell"),
    ({"config_reference": "pyproject.toml"}, "config"),
    ({"predictors": []}, "predictor"),
])
def test_invalid_proposals_never_create_file(project, context, proposal, change, match):
    (project / "pyproject.toml").write_text("[project]")
    proposal.update(change)
    target = ledger(project, context)
    with pytest.raises(ValueError, match=match):
        target.append([proposal], registry())
    assert not target.path.exists()


@pytest.mark.parametrize("entry", [{"verified": False, "provenance": "review"},
                                   {"verified": True, "provenance": " "},
                                   {"verified": True, "provenance": "invented"}])
def test_registry_requires_typed_verified_provenance(project, context, proposal, entry):
    with pytest.raises(ValueError, match="predictor|registry"):
        ledger(project, context).append([proposal], {"meter_reading": entry})


@pytest.mark.parametrize("name", ["anomaly", "building_id", "id", "fold", "repeat", "seed",
                                  "split", "source_vendor", "gte_probe"])
def test_reserved_predictors_cannot_be_laundered_with_typed_metadata(project, context, proposal, name):
    meta = replace(registry().get("meter_reading"), name=name, source_columns=(name,))
    proposal["predictors"] = [name]
    if name == "gte_probe":
        meta = replace(meta, provenance_status=ProvenanceStatus.UNVERIFIED)
    with pytest.raises(ValueError, match="predictor|provenance"):
        ledger(project, context).append([proposal], FeatureRegistry([meta]))


@pytest.mark.parametrize("changes", [
    {"prediction_time_availability": AvailabilityStatus.PENDING},
    {"target_dependence": TargetDependence.TARGET_DERIVED},
    {"provenance_status": ProvenanceStatus.UNVERIFIED},
    {"available_at": datetime(2027, 1, 1, tzinfo=UTC)},
    {"source_columns": ("anomaly",)}, {"family": "grouping"},
])
def test_typed_metadata_availability_and_lineage(project, context, proposal, changes):
    meta = replace(registry().get("meter_reading"), **changes)
    with pytest.raises(ValueError, match="predictor|provenance|target|prediction|available"):
        ledger(project, context).append([proposal], FeatureRegistry([meta]))


def test_campaign_addition_resolves_typed_metadata(project, context, proposal):
    meta = replace(registry().get("meter_reading"), name="meter_missing",
                   family="missingness", source_columns=("meter_reading",))
    proposal["predictors"] = [meta.name]
    ledger(project, context).append([proposal], FeatureRegistry([meta]))


def test_completed_evidence_links_to_run_and_checks_hash(project, context, proposal):
    # Corrected gen10 expectation: execution completes BEFORE N03 review.
    # Pending review is valid; run status/artifact authenticity remains required.
    record = completed(project, proposal)
    target = ledger(project, context)
    target.append([record], registry())
    bad = copy.deepcopy(record)
    bad["proposal_id"] = "tampered"
    bad["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact"):
        target.append([bad], registry())


def test_completion_run_status_hash_is_required(project, context, proposal):
    record = completed(project, proposal)
    record["run_status_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="completion|status"):
        ledger(project, context).append([record], registry())


def test_linked_review_event_is_append_only(project, context, proposal):
    record = completed(project, proposal)
    target = ledger(project, context)
    target.append([record], registry())
    before = target.path.read_bytes()
    reference = project / "outputs/runs/real/review.json"
    reference.write_text(json.dumps({"run_id": "real", "experiment_id": proposal["experiment_id"],
                                     "status": "reviewed"}))
    event = {"event_type": "review", "event_id": "review-event-01",
             "proposal_id": proposal["proposal_id"], "experiment_id": proposal["experiment_id"],
             "run_id": "real", "reason": "Independent N03 review",
             "review": {"status": "reviewed", "reference": "outputs/runs/real/review.json"}}
    target.append([event], registry())
    assert target.path.read_bytes().startswith(before)
    assert len(target.path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="duplicate"):
        target.append([event], registry())


@pytest.mark.parametrize("outcome", ["failed", "blocked", "rejected_by_gate"])
def test_noncompleted_run_outcomes_require_run_id(project, context, proposal, outcome):
    proposal["outcome"] = outcome
    with pytest.raises(ValueError, match="run ID"):
        ledger(project, context).append([proposal], registry())


def test_prohibited_matrix_cannot_be_planned(project, context, proposal):
    proposal["experiment_id"] = "prohibited_opaque_building_id_digit_extraction"
    with pytest.raises(ValueError, match="excluded"):
        ledger(project, context).append([proposal], registry())


def test_justified_exclusion_may_have_no_predictors(project, context, proposal):
    proposal.update(experiment_id="prohibited_opaque_building_id_digit_extraction",
                    disposition="excluded", reason="Opaque identifiers are prohibited", predictors=[])
    ledger(project, context).append([proposal], registry())


@pytest.mark.parametrize("relative", ["experiment_inventory.jsonl", "campaign_ledger.jsonl"])
def test_ledger_cannot_target_external_or_baseline_path(project, context, relative):
    with pytest.raises(ValueError, match="campaign_ledger.jsonl|confin"):
        api().CampaignLedger(project.parent / relative, project, context)


def test_symlink_cannot_redirect_append_after_construction(project, context, proposal):
    target = ledger(project, context)
    inventory = project / "outputs/runs/experiment_inventory.jsonl"
    inventory.write_text("baseline\n")
    target.path.symlink_to(inventory)
    with pytest.raises(ValueError, match="symlink|follow"):
        target.append([proposal], registry())
    assert inventory.read_text() == "baseline\n"


def test_parent_symlink_rejected(project, context, proposal, tmp_path):
    runs = project / "outputs/runs"
    runs.rmdir()
    external = tmp_path / "elsewhere"
    external.mkdir()
    runs.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|follow|confin"):
        ledger(project, context).append([proposal], registry())
    assert not (external / "campaign_ledger.jsonl").exists()


@pytest.mark.parametrize("history", ["not-json\n", "{}\n", "[]\n", "duplicate"])
def test_malformed_or_duplicate_history_blocks_append(project, context, proposal, history):
    target = ledger(project, context)
    if history == "duplicate":
        history = (json.dumps(proposal) + "\n") * 2
    target.path.write_text(history)
    before = target.path.read_bytes()
    with pytest.raises(ValueError, match="history|duplicate"):
        target.append([dict(proposal, proposal_id="new")], registry())
    assert target.path.read_bytes() == before


@pytest.mark.parametrize("failure", ["partial_write", "flush"])
def test_failed_append_rolls_back_to_exact_previous_bytes(project, context, proposal, monkeypatch, failure):
    target = ledger(project, context)
    target.append([proposal], registry())
    before = target.path.read_bytes()
    real_write, real_fsync = os.write, os.fsync
    calls = 0

    def fail_write(fd, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, payload[:7])
        raise OSError("injected partial write")

    def fail_flush(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected flush")
        return real_fsync(fd)

    monkeypatch.setattr(os, "write" if failure == "partial_write" else "fsync",
                        fail_write if failure == "partial_write" else fail_flush)
    with pytest.raises(OSError, match="injected"):
        target.append([dict(proposal, proposal_id="new")], registry())
    assert target.path.read_bytes() == before


def test_versioned_config_must_link_actual_experiment(project, context, proposal):
    path = project / "configs/experiments/candidate.v1.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "experiment_id": "not-registered"}))
    proposal["config_reference"] = "configs/experiments/candidate.v1.yaml"
    with pytest.raises(ValueError, match="config"):
        ledger(project, context).append([proposal], registry())
    path.write_text(yaml.safe_dump({"version": 1, "experiment_id": proposal["experiment_id"]}))
    ledger(project, context).append([proposal], registry())


def test_leaf_replaced_while_validating_cannot_redirect_or_claim_append(project, context, proposal, monkeypatch):
    target = ledger(project, context)
    target.append([proposal], registry())
    inventory = project / "outputs/runs/experiment_inventory.jsonl"
    inventory.write_text("baseline\n")
    original = target._validate_sequence
    replaced = False

    def replace_after_validation(records, features):
        nonlocal replaced
        original(records, features)
        if not replaced:
            replaced = True
            target.path.unlink()
            target.path.symlink_to(inventory)

    monkeypatch.setattr(target, "_validate_sequence", replace_after_validation)
    with pytest.raises(ValueError, match="changed|symlink|confin"):
        target.append([dict(proposal, proposal_id="new")], registry())
    assert inventory.read_text() == "baseline\n"


def test_concurrent_duplicate_submission_is_serialized(project, context, proposal):
    def submit():
        try:
            ledger(project, context).append([proposal], registry())
        except ValueError as error:
            return str(error)
        return "appended"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert results.count("appended") == 1
    assert any("duplicate" in result for result in results)
    assert len((project / "outputs/runs/campaign_ledger.jsonl").read_text().splitlines()) == 1


def test_review_of_unknown_proposal_is_rejected_without_append(project, context, proposal):
    target = ledger(project, context)
    record = completed(project, proposal)
    target.append([record], registry())
    reference = project / "outputs/runs/real/review.json"
    reference.write_text(json.dumps({"run_id": "real", "experiment_id": proposal["experiment_id"],
                                     "status": "reviewed"}))
    before = target.path.read_bytes()
    event = {"event_type": "review", "event_id": "review-unknown", "proposal_id": "unknown",
             "experiment_id": proposal["experiment_id"], "run_id": "real", "reason": "review",
             "review": {"status": "reviewed", "reference": "outputs/runs/real/review.json"}}
    with pytest.raises(ValueError, match="lifecycle"):
        target.append([event], registry())
    assert target.path.read_bytes() == before
