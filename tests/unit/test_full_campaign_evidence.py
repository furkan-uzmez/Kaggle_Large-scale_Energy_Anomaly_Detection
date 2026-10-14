"""Unit tests for full-campaign strict evidence and immutable manifests (T5)."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from full_campaign_fixtures import write_admission_fixture, write_fixture
from project.experiments import full_campaign_evidence as evidence
from project.experiments.full_campaign_evidence import (
    sign_parent_record,
    verify_interrupted_exit,
)
from project.experiments.full_campaign_worker import prepare_worker_request
from project.experiments.full_run_contracts import FullAttemptRecord, _file_hash
from project.experiments.full_snapshot_codec import (
    LEAD_YEAR_OFFSET_1900_UINT8_V1,
    PROVENANCE_PATH,
    encode_snapshot,
    semantic_digest,
)


def test_strict_json_rejects_nonfinite_and_unknown_types():
    with pytest.raises(ValueError, match="nonfinite"):
        evidence.strict_json_dumps({"bad": float("nan")})

    with pytest.raises(ValueError, match="nonfinite"):
        evidence.strict_json_dumps({"bad": float("inf")})

    with pytest.raises(TypeError, match="unsupported"):
        evidence.strict_json_dumps({"bad": object()})


def test_full_evidence_requires_actual_hashes_and_per_arm_metadata(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    dest = tmp_path / "evidence"

    # Missing per-arm metadata or sha256
    with pytest.raises(ValueError, match="arm_metadata|hashes"):
        evidence.write_full_evidence(
            spec=fixture.spec,
            destination=dest,
            status="completed",
            artifacts={},
            validation_payload=None,
        )


def test_full_evidence_rejects_artifact_overwrite(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    dest = tmp_path / "evidence"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text('{"existing": true}')

    with pytest.raises(FileExistsError, match="overwrite"):
        evidence.write_full_evidence(
            spec=fixture.spec,
            destination=dest,
            status="completed",
            artifacts={"oof_sha256": "a" * 64},
            validation_payload={"validated": True, "seal": "s" * 64, "metrics": {}},
        )


def test_comparative_completion_rejected_without_validated_seal(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    dest = tmp_path / "evidence"

    with pytest.raises(ValueError, match="seal|validation"):
        evidence.write_full_evidence(
            spec=fixture.spec,
            destination=dest,
            status="completed",
            artifacts={"oof_sha256": "a" * 64},
            validation_payload=None,
        )


def test_write_full_evidence_success(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    dest = tmp_path / "evidence"

    result = evidence.write_full_evidence(
        spec=fixture.spec,
        destination=dest,
        status="awaiting_validation",
        artifacts={"oof_sha256": "a" * 64, "oof_path": "feat_missingness_indicators_oof.parquet"},
        validation_payload={
            "validated": True,
            "seal": "s" * 64,
            "metrics": {
                "feat_missingness_indicators_challenger": {
                    "mean_repeat_metrics": {"roc_auc": 0.85, "brier_score": 0.12}
                }
            },
            "candidates": {
                "feat_missingness_indicators_challenger": {
                    "arms": ["control", "challenger"],
                    "hashes": {"config_sha256": fixture.spec.binding_sha256},
                }
            },
        },
    )
    assert result["status"] == "awaiting_validation"
    assert (dest / "manifest.json").is_file()
    assert (dest / "run_status.json").is_file()
    status_doc = json.loads((dest / "run_status.json").read_text())
    assert status_doc["status"] == "awaiting_validation"
    assert status_doc["validation_seal"] == "s" * 64


def test_truthy_or_self_hashed_validation_never_grants_completion(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    destination = tmp_path / "evidence"
    payload = {"validated": True, "seal": "s" * 64, "candidate_metrics": {}}
    with pytest.raises(ValueError, match="independent|completion"):
        evidence.write_full_evidence(fixture.spec, destination, "completed",
                                     {"oof_sha256": "a" * 64}, payload)
    assert not destination.exists()


def test_overwrite_flag_cannot_replace_immutable_evidence(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    destination = tmp_path / "evidence"
    evidence.write_full_evidence(fixture.spec, destination, "awaiting_validation", {})
    before = (destination / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        evidence.write_full_evidence(fixture.spec, destination, "awaiting_validation", {}, allow_overwrite=True)
    assert (destination / "manifest.json").read_bytes() == before


def _setup_interrupted_attempt(
    tmp_path: Path,
    *,
    exp_id: str = "dim_fold_local_selector_vs_passthrough",
    arm_id: str = "challenger",
    selector_arms: tuple[str, ...] = ("challenger",),
    attempt_unit: str | None = None,
    launch_unit_override: str | None = None,
    identity_unit_override: str | None = None,
    journal_attempt_id_override: str | None = None,
    binding_override: dict | None = None,
):
    source = tmp_path / "source"
    fixture = write_admission_fixture(source)
    cand = replace(
        fixture.spec.candidates[0],
        spec=replace(fixture.spec.candidates[0].spec, experiment_id=exp_id),
        selector_arms=selector_arms,
    )
    spec = replace(fixture.spec, candidates=(cand,))

    run_root = tmp_path / "run"
    attempt_id = "att_selector_001"
    attempt_dir = run_root / "journal" / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)

    model_work_id = f"{exp_id}/{arm_id}/r0/f0/model"
    unit = attempt_unit if attempt_unit is not None else f"{exp_id}/{arm_id}/r0/f0/selector"
    attempt_name = f"{exp_id}_{arm_id}_r0_f0_testnonce"

    prov_sha = next(s.sha256 for s in spec.sources if s.role == "encoding")
    sem_sha = semantic_digest(LEAD_YEAR_OFFSET_1900_UINT8_V1, prov_sha)

    worker_key = "k" * 32
    request_dir = tmp_path / "worker_staging" / attempt_name
    payload = {
        "version": 1,
        "project_root": str(source.resolve()),
        "spec": encode_snapshot(spec),
        "spec_sha256": spec.binding_sha256,
        "index_path": str((run_root / "dev_index.sqlite3").resolve()),
        "index_sha256": "1" * 64,
        "experiment_id": exp_id,
        "arm_id": arm_id,
        "repeat": 0,
        "fold": 0,
        "year_built_encoding": LEAD_YEAR_OFFSET_1900_UINT8_V1,
        "provenance_path": PROVENANCE_PATH,
        "provenance_sha256": prov_sha,
        "semantic_sha256": sem_sha,
        "work_id": model_work_id,
        "attempt_id": attempt_name,
        "output_path": "shard.parquet",
        "parent_pid": os.getpid(),
        "ack_timeout_seconds": 10.0,
        "nonce": "a" * 64,
    }
    signed_payload = prepare_worker_request(request_dir, payload, worker_key)
    shutil.copy2(request_dir / "worker-request.json", attempt_dir / "worker-request.json")

    dead_pid = 99999999
    binding = (
        binding_override
        if binding_override is not None
        else {
            "nonce": signed_payload["nonce"],
            "attempt_id": signed_payload["attempt_id"],
            "spec_sha256": spec.binding_sha256,
        }
    )
    launch_doc = {
        "kind": "worker_launch",
        "run_root": str(run_root),
        "spec_sha256": spec.binding_sha256,
        "unit": launch_unit_override if launch_unit_override is not None else model_work_id,
        "worker_request_sha256": _file_hash(attempt_dir / "worker-request.json"),
        "journal_attempt_id": journal_attempt_id_override if journal_attempt_id_override is not None else attempt_id,
        "identity_record": {
            "unit": identity_unit_override if identity_unit_override is not None else model_work_id,
            "pid": dead_pid,
            "identity": [1, dead_pid, dead_pid],
            "binding": binding,
        },
    }
    (attempt_dir / "worker-launch.json").write_text(json.dumps(sign_parent_record(source, launch_doc)))

    attempt = FullAttemptRecord(
        attempt_id=attempt_id,
        unit=unit,
        status="running",
        artifact_directory=f"attempts/{attempt_id}",
    )
    return source, run_root, spec, attempt


def test_interrupted_selector_exit_verifies_authentically(tmp_path):
    source, run_root, spec, attempt = _setup_interrupted_attempt(tmp_path)
    proof = verify_interrupted_exit(source, run_root, spec, attempt)
    assert proof["kind"] == "owned_attempt_exit"
    assert proof["journal_attempt_id"] == attempt.attempt_id
    assert proof["unit"] == attempt.unit
    assert proof["cleanup_confirmed"] is True


def test_interrupted_selector_exit_rejects_unapproved_arm(tmp_path):
    # Attempt claims selector for 'control', but 'control' is not in selector_arms
    source, run_root, spec, attempt = _setup_interrupted_attempt(
        tmp_path,
        arm_id="challenger",
        selector_arms=("challenger",),
        attempt_unit="dim_fold_local_selector_vs_passthrough/control/r0/f0/selector",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_interrupted_exit(source, run_root, spec, attempt)


def test_interrupted_selector_exit_rejects_arbitrary_suffix(tmp_path):
    source, run_root, spec, attempt = _setup_interrupted_attempt(
        tmp_path,
        attempt_unit="dim_fold_local_selector_vs_passthrough/challenger/r0/f0/arbitrary_phase",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_interrupted_exit(source, run_root, spec, attempt)


def test_interrupted_selector_exit_rejects_tampered_launch_unit(tmp_path):
    source, run_root, spec, attempt = _setup_interrupted_attempt(
        tmp_path,
        launch_unit_override="dim_fold_local_selector_vs_passthrough/challenger/r0/f0/tampered",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_interrupted_exit(source, run_root, spec, attempt)


def test_interrupted_selector_exit_rejects_tampered_identity_record_unit(tmp_path):
    source, run_root, spec, attempt = _setup_interrupted_attempt(
        tmp_path,
        identity_unit_override="dim_fold_local_selector_vs_passthrough/challenger/r0/f0/tampered",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_interrupted_exit(source, run_root, spec, attempt)


def test_interrupted_selector_exit_rejects_tampered_journal_attempt_id(tmp_path):
    source, run_root, spec, attempt = _setup_interrupted_attempt(
        tmp_path,
        journal_attempt_id_override="att_wrong_id",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_interrupted_exit(source, run_root, spec, attempt)


def test_interrupted_selector_exit_rejects_tampered_binding(tmp_path):
    source, run_root, spec, attempt = _setup_interrupted_attempt(
        tmp_path,
        binding_override={"nonce": "b" * 64, "attempt_id": "tampered", "spec_sha256": "c" * 64},
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_interrupted_exit(source, run_root, spec, attempt)

