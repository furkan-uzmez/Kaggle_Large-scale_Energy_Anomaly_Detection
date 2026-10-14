"""Pure evidence gates on synthetic reviewed authorities; never a fit grant."""
import copy
import importlib.util
import json
from dataclasses import replace

import polars as pl
import pytest

from full_campaign_fixtures import write_admission_fixture
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_development_audits import AuditPlan, audit_full_development, write_audit_evidence
from project.experiments.full_run_contracts import _file_hash


def readiness_module():
    assert importlib.util.find_spec("project.experiments.full_development_readiness"), "readiness evidence gate is missing"
    from project.experiments import full_development_readiness
    return full_development_readiness


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    gate = readiness_module()
    fixture = write_admission_fixture(tmp_path / "reviewed")
    report = audit_full_development(fixture.root, fixture.spec, AuditPlan((2, 3, 4, 5), 42))
    path = fixture.root / gate.EVIDENCE_PATH
    digest = write_audit_evidence(path, report)
    # Unit-only reviewed fixture authority: production has a fixed immutable
    # actual-evidence anchor, not an injectable caller readiness boolean.
    monkeypatch.setattr(gate, "_REFERENCE_ROOT", fixture.root)
    monkeypatch.setattr(gate, "EVIDENCE_SHA256", digest)
    candidate = replace(fixture.spec.candidates[0], spec=CandidateSpec(
        gate.TARGET_ID, fixture.spec.candidates[0].spec.feature_pool,
        "hist_gradient_boosting", (("early_stopping", False), ("random_state", 42)),
        gate.TARGET_METHODS,
    ))
    return gate, fixture, replace(fixture.spec, candidates=(candidate,)), report


def test_authenticated_support_is_data_ready_not_execution_grant(reviewed):
    gate, fixture, spec, _ = reviewed
    evidence = gate.inspect_development_readiness(fixture.root, spec)
    assert evidence.target_encoding_disposition == "data_ready_pending_runtime_proof"
    assert evidence.sentinel_disposition == "not_applicable"
    assert evidence.execution_authorized is False
    assert evidence.n_splits == 3 and evidence.seed == 42
    assert len(evidence.inner_membership_sha256) == 12


@pytest.mark.parametrize("mutation", ["row_count", "one_class", "missing_partition", "missing_inner_fold", "seed", "k", "identity"])
def test_malformed_reviewed_evidence_rejected_before_any_debit(reviewed, monkeypatch, mutation):
    gate, fixture, spec, original = reviewed
    report = copy.deepcopy(original)
    plan = report["target_encoding"][0]["plans"][1]
    if mutation == "row_count":
        report["development_rows"] += 1
    elif mutation == "one_class":
        plan["folds"][0]["train"]["class_counts"][1] = 0
    elif mutation == "missing_partition":
        report["target_encoding"].pop()
    elif mutation == "missing_inner_fold":
        plan["folds"].pop()
    elif mutation == "seed":
        plan["seed"] = 43
    elif mutation == "k":
        plan["n_splits"] = 2
    else:
        report["development_identity_sha256"] = "unbound"
    path = fixture.root / gate.EVIDENCE_PATH
    path.write_text(json.dumps(report, allow_nan=False))
    monkeypatch.setattr(gate, "EVIDENCE_SHA256", _file_hash(path))
    with pytest.raises(ValueError):
        gate.inspect_development_readiness(fixture.root, spec)


@pytest.mark.parametrize("role", ["data", "split", "encoding"])
def test_changed_bound_sources_rejected(reviewed, role):
    gate, fixture, spec, _ = reviewed
    source = next(s for s in spec.sources if s.role == role)
    (fixture.root / source.path).write_bytes(b"changed source")
    with pytest.raises(ValueError):
        gate.inspect_development_readiness(fixture.root, spec)


def test_source_only_inspection_excludes_encoding_without_downgrading_dispositions(reviewed):
    # The source-only campaign inspector strips the encoding extension before
    # consulting this gate; encoding is authenticated separately upstream. The
    # omission must not downgrade the sentinel's not_applicable disposition.
    gate, fixture, spec, _ = reviewed
    source_only = replace(spec, sources=tuple(
        source for source in spec.sources if source.role != "encoding"))
    evidence = gate.inspect_development_readiness(fixture.root, source_only)
    assert evidence.sentinel_disposition == "not_applicable"
    assert evidence.target_encoding_disposition == "data_ready_pending_runtime_proof"


def test_declared_encoding_digest_still_strictly_bound(reviewed):
    # When the caller declares the encoding role, its declared digest must still
    # equal the authenticated evidence digest.
    gate, fixture, spec, _ = reviewed
    forged = replace(next(s for s in spec.sources if s.role == "encoding"), sha256="0" * 64)
    with pytest.raises(ValueError, match="binding mismatch"):
        gate.inspect_development_readiness(
            fixture.root, replace(spec, sources=tuple(
                forged if s.role == "encoding" else s for s in spec.sources)))


def test_stale_evidence_hash_rejected(reviewed):
    gate, fixture, spec, _ = reviewed
    path = fixture.root / gate.EVIDENCE_PATH
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        gate.inspect_development_readiness(fixture.root, spec)


def test_producer_mutation_requires_refresh_but_unrelated_integration_does_not(reviewed):
    gate, fixture, spec, _ = reviewed
    # These modules were only part of a diagnostic whole-tree inventory; the
    # audit never used their execution paths to compute dev support/sentinels.
    (fixture.root / "src/project/experiments/full_campaign_admission.py").write_text("# integration\n")
    assert gate.inspect_development_readiness(fixture.root, spec).execution_authorized is False
    (fixture.root / "src/project/experiments/inner_cv.py").write_text("# producer changed\n")
    with pytest.raises(ValueError, match="audit_refresh_required"):
        gate.inspect_development_readiness(fixture.root, spec)


def test_partition_actual_inner_digest_and_labels_verified(reviewed):
    gate, fixture, spec, _ = reviewed
    from project.experiments.full_candidate_io import build_development_index, load_full_training_partition
    index = build_development_index(fixture.root, spec, fixture.root / "fixture-index.sqlite")
    partition = load_full_training_partition(index, spec, 0, 0)
    result = gate.verify_target_encoding_partition(fixture.root, spec, partition)
    assert len(result) == 64
    bad_labels = partition.y_train.copy()
    bad_labels.iloc[0] = 1 - bad_labels.iloc[0]
    with pytest.raises(ValueError, match="label"):
        gate.verify_target_encoding_partition(fixture.root, spec, replace(partition, y_train=bad_labels))


def test_exact_default_parameters_only(reviewed):
    gate, fixture, spec, _ = reviewed
    snapshot = spec.candidates[0]
    changed = replace(snapshot, spec=replace(snapshot.spec, method_params=(("n_splits", 2),)))
    with pytest.raises(ValueError, match="parameters"):
        gate.inspect_development_readiness(fixture.root, replace(spec, candidates=(changed,)))


def test_all_source_rows_preserved_with_truthful_dispositions(reviewed):
    gate, fixture, spec, _ = reviewed
    from project.experiments.full_campaign_admission import FullCampaignIntent, inspect_full_campaign
    # Source-only inspector excludes encoding; the support gate still requires
    # and verifies encoding itself through the actual full snapshot extension.
    report = inspect_full_campaign(fixture.root, FullCampaignIntent(spec, True))
    assert len(report.rows) == 21
    target = next(r for r in report.rows if r.record.experiment_id == gate.TARGET_ID)
    sentinel = next(r for r in report.rows if r.record.experiment_id == gate.SENTINEL_ID)
    assert target.scientific_disposition == "data_ready_pending_runtime_proof"
    assert target.record.status != "ready"
    assert sentinel.scientific_disposition == "not_applicable"
    assert sentinel.disposition_reason == "no_observed_invalid_values"
    assert sentinel.planned_model_fits == 0
    assert sentinel.record.n03_eligible is False
