"""Strict worker transport and authenticated lifecycle boundary regressions."""

import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from full_campaign_fixtures import write_fixture
from project.experiments.candidate_contracts import CandidateSpec, StudySpec
from project.experiments.full_campaign_worker import (
    ParentLifecycleService,
    prepare_worker_request,
)
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    SourceBinding,
    _file_hash,
)


def request_fixture(tmp_path):
    from project.experiments.full_snapshot_codec import encode_snapshot, semantic_digest

    fixture = write_fixture(tmp_path / "source")
    path = fixture.root / "docs/evidence/year_built_encoding_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"encoding_id":"lead_year_offset_1900_uint8_v1"}')
    binding = SourceBinding("encoding", str(path.relative_to(fixture.root)), _file_hash(path))
    snapshot = replace(fixture.spec.candidates[0], spec=CandidateSpec(
        "feat_missingness_indicators", fixture.spec.candidates[0].spec.feature_pool,
        "hist_gradient_boosting", (), ()))
    spec = replace(fixture.spec, sources=fixture.spec.sources + (binding,), candidates=(snapshot,),
                   development_rows=52, vault_rows=4)
    index = tmp_path / "run/index.sqlite3"
    index.parent.mkdir()
    index.write_bytes(b"index-test")
    payload = {
        "version": 1, "project_root": str(fixture.root), "spec": encode_snapshot(spec),
        "spec_sha256": spec.binding_sha256, "index_path": str(index),
        "index_sha256": _file_hash(index), "experiment_id": snapshot.spec.experiment_id,
        "arm_id": snapshot.arms[0], "repeat": 0, "fold": 0,
        "year_built_encoding": "lead_year_offset_1900_uint8_v1",
        "provenance_path": binding.path, "provenance_sha256": binding.sha256,
        "semantic_sha256": semantic_digest("lead_year_offset_1900_uint8_v1", binding.sha256),
        "work_id": spec.planned_units[0], "attempt_id": "attempt-one",
        "output_path": "shard.parquet", "parent_pid": os.getpid(),
        "ack_timeout_seconds": 10.0, "nonce": "a" * 64,
    }
    return fixture, spec, payload, tmp_path / "run/attempts/attempt-one"


def write_event(directory, request, seq=1, event="debit", phase="model", **changes):
    from project.experiments.full_snapshot_codec import canonical_json

    payload = {"event": event, "seq": seq, "phase": phase,
               "work_id": request["work_id"], "attempt_id": request["attempt_id"],
               "nonce": request["nonce"],
               "unit": request["work_id"].rsplit("/", 1)[0] + "/" + phase, **changes}
    signature = hmac.new(b"key", canonical_json(payload).encode(), hashlib.sha256).hexdigest()
    (directory / f"event-{seq:04d}.json").write_text(canonical_json({"payload": payload, "signature": signature}))


def test_request_rejects_unknown_schema_before_writing(tmp_path):
    directory = tmp_path / "attempt"
    with pytest.raises((TypeError, ValueError)):
        prepare_worker_request(directory, {"version": True, "work_id": "fake"}, "key")
    assert not directory.exists()


def test_request_refuses_reusing_populated_attempt_directory(tmp_path):
    _, _, payload, directory = request_fixture(tmp_path)
    directory.mkdir(parents=True)
    (directory / "prior-event.json").write_text("retained failed attempt")
    with pytest.raises(ValueError, match="fresh"):
        prepare_worker_request(directory, payload, "key")
    assert not (directory / "worker-request.json").exists()


def test_fit_configuration_codec_rejects_unknown_objects():
    from project.experiments import full_snapshot_codec as codec

    with pytest.raises(TypeError):
        codec.configuration_digest((object, object()))


def test_parent_rejects_signed_wrong_unit_before_debit(tmp_path):
    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    debits = []
    service = ParentLifecycleService(directory, "key", on_debit=lambda *args: debits.append(args))
    write_event(directory, request, work_id="wrong")
    with pytest.raises(ValueError):
        service.service_once()
    assert debits == []
    assert not (directory / "ack-0001.json").exists()


def test_snapshot_roundtrip_preserves_study_types_and_binding(tmp_path):
    from project.experiments.full_snapshot_codec import decode_snapshot, encode_snapshot

    _, spec, _, _ = request_fixture(tmp_path)
    study = StudySpec(2, 3, 42, (("max_iter", (100, 200)),))
    candidate = replace(spec.candidates[0].spec, experiment_id="regime_same_family_hpo_vs_default")
    snapshot = FullCandidateSnapshot(candidate, ("control", "challenger"),
                                     spec.candidates[0].source_notebook,
                                     spec.candidates[0].source_cell_id, study, "random_search")
    spec = replace(spec, candidates=(snapshot,))
    decoded = decode_snapshot(json.loads(json.dumps(encode_snapshot(spec))))
    assert decoded == spec
    assert type(decoded.candidates[0].study) is StudySpec
    assert type(decoded.candidates[0].study.parameter_lists[0][1]) is tuple
    assert decoded.binding_sha256 == spec.binding_sha256


@pytest.mark.parametrize("change", ["bool", "missing", "extra", "digest", "semantic"])
def test_strict_request_rejects_protocol_coercion_and_tampering(tmp_path, change):
    from project.experiments.full_snapshot_codec import validate_request

    _, _, payload, _ = request_fixture(tmp_path)
    if change == "bool":
        payload["repeat"] = False
    elif change == "missing":
        del payload["spec"]["spec"]["folds"]
    elif change == "extra":
        payload["spec"]["spec"]["unknown"] = 1
    elif change == "digest":
        payload["spec_sha256"] = "b" * 64
    else:
        payload["semantic_sha256"] = "b" * 64
    with pytest.raises((TypeError, ValueError)):
        validate_request(payload)


@pytest.mark.parametrize("changes", [{"nonce": "b" * 64}, {"phase": "selector"},
                                    {"seq": 2}, {"seq": True}])
def test_parent_rejects_wrong_nonce_phase_or_sequence(tmp_path, changes):
    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    calls = []
    service = ParentLifecycleService(directory, "key", on_debit=lambda *args: calls.append(args))
    write_event(directory, request, **changes)
    with pytest.raises(ValueError):
        service.service_once()
    assert calls == []


def test_parent_ack_binds_identity_and_rejects_replayed_sequence(tmp_path):
    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    calls = []
    service = ParentLifecycleService(directory, "key", on_debit=lambda *args: calls.append(args))
    write_event(directory, request)
    service.service_once()
    ack = json.loads((directory / "ack-0001.json").read_text())["payload"]
    assert ack["nonce"] == request["nonce"]
    assert ack["unit"] == request["work_id"]
    assert ack["seq"] == 1
    assert len(calls) == 1
    # Replayed payload in a different file must not cause another debit.
    (directory / "event-0002.json").write_text((directory / "event-0001.json").read_text())
    with pytest.raises(ValueError):
        service.service_once()
    assert len(calls) == 1


def test_parent_rejects_mutated_previously_serviced_event(tmp_path):
    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    service = ParentLifecycleService(directory, "key", on_debit=lambda *args: None)
    write_event(directory, request)
    service.service_once()
    write_event(directory, request, nonce="c" * 64)
    with pytest.raises(ValueError, match="changed"):
        service.service_once()


def test_parent_restart_never_redebits_existing_ack(tmp_path):
    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    calls = []
    write_event(directory, request)
    ParentLifecycleService(directory, "key", on_debit=lambda *args: calls.append(args)).service_once()
    with pytest.raises(ValueError):
        ParentLifecycleService(directory, "key", on_debit=lambda *args: calls.append(args)).service_once()
    assert len(calls) == 1


def test_parent_rejects_signed_shard_hash_mismatch_without_ack(tmp_path):
    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    service = ParentLifecycleService(directory, "key", on_debit=lambda *args: None)
    write_event(directory, request)
    service.service_once()
    path = directory / "shard.parquet"
    path.write_bytes(b"closed artifact")
    write_event(directory, request, seq=2, event="shard_closed", sha256="b" * 64,
                num_rows=1, path=str(path))
    with pytest.raises(ValueError, match="shard"):
        service.service_once()
    assert service.shard_evidence is None
    assert not (directory / "ack-0002.json").exists()


def test_request_cannot_be_replayed_in_another_attempt_directory(tmp_path):
    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    other = directory.parent / "another-attempt"
    other.mkdir()
    (other / "worker-request.json").write_bytes((directory / "worker-request.json").read_bytes())
    write_event(other, request)
    calls = []
    service = ParentLifecycleService(other, "key", on_debit=lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="attempt"):
        service.service_once()
    assert calls == []


def real_worker_fixture(tmp_path):
    import polars as pl

    from project.experiments.full_candidate_io import build_development_index
    from project.experiments.full_snapshot_codec import encode_snapshot

    fixture, spec, payload, directory = request_fixture(tmp_path)
    # Original fixture stores calendar years; this transport explicitly declares
    # offset encoding. Re-encode fixture values and rebind the changed bytes.
    fixture.full_frame.with_columns(pl.lit(100).alias("year_built")).write_parquet(fixture.data_path)
    spec = replace(spec, sources=tuple(replace(item, sha256=_file_hash(fixture.data_path))
                                      if item.role == "data" else item for item in spec.sources))
    index = tmp_path / "run/development.sqlite3"
    build_development_index(fixture.root, spec, index)
    payload.update(spec=encode_snapshot(spec), spec_sha256=spec.binding_sha256,
                   index_path=str(index), index_sha256=_file_hash(index))
    request = prepare_worker_request(directory, payload, "key")
    return request, directory


def launch_worker(request, directory):
    return subprocess.Popen(
        [sys.executable, "-B", "-c",
         "import sys; from project.experiments.full_campaign_worker import run; run(sys.argv[1],sys.argv[2])",
         request["work_id"], str(directory)],
        env=dict(os.environ, FULL_CAMPAIGN_WORKER_KEY="key"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_real_fixture_worker_consumes_codec_and_exact_parent_handshake(tmp_path):
    import pyarrow.parquet as pq

    request, directory = real_worker_fixture(tmp_path)
    debits = []
    service = ParentLifecycleService(directory, "key", on_debit=lambda *args: debits.append(args))
    process = launch_worker(request, directory)
    try:
        deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < deadline:
            service.service_once()
            time.sleep(0.01)
        stdout, stderr = process.communicate(timeout=2)
        assert process.returncode == 0, (stdout, stderr)
        service.service_once()
        assert len(debits) == 1
        assert debits[0][1]["unit"] == request["work_id"]
        shard = directory / "shard.parquet"
        assert service.shard_evidence["sha256"] == _file_hash(shard)
        assert service.shard_evidence["num_rows"] == pq.ParquetFile(shard).metadata.num_rows
        assert pq.ParquetFile(shard).metadata.num_rows > 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


@pytest.mark.parametrize("changes", [{"nonce": "b" * 64}, {"seq": True},
                                    {"unit": "wrong"}, {"event": "shard_closed"},
                                    {"attempt_id": "other-attempt"}])
def test_real_worker_rejects_signed_wrong_identity_ack_before_fit(tmp_path, changes):
    from project.experiments.full_snapshot_codec import canonical_json

    request, directory = real_worker_fixture(tmp_path)
    process = launch_worker(request, directory)
    try:
        event_path = directory / "event-0001.json"
        deadline = time.monotonic() + 20
        while not event_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert event_path.exists()
        ack = json.loads(event_path.read_text())["payload"]
        ack.update(status="authorized", **changes)
        signature = hmac.new(b"key", canonical_json(ack).encode(), hashlib.sha256).hexdigest()
        (directory / "ack-0001.json").write_text(canonical_json({"payload": ack, "signature": signature}))
        _, stderr = process.communicate(timeout=10)
        assert process.returncode != 0
        assert "invalid ack" in stderr
        assert not (directory / "shard.parquet").exists()
        assert not (directory / "event-0002.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


@pytest.mark.parametrize("field", ["spec_sha256", "semantic_sha256"])
def test_real_worker_rejects_signed_digest_tampering_before_index_or_debit(tmp_path, field):
    from project.experiments.full_snapshot_codec import canonical_json

    _, _, payload, directory = request_fixture(tmp_path)
    request = prepare_worker_request(directory, payload, "key")
    request[field] = "b" * 64
    signature = hmac.new(b"key", canonical_json(request).encode(), hashlib.sha256).hexdigest()
    (directory / "worker-request.json").write_text(canonical_json({"payload": request, "signature": signature}))
    # The dummy index is not a valid SQLite database. Digest rejection must
    # precede any attempt to read its content or perform a debit.
    process = launch_worker(request, directory)
    try:
        _, stderr = process.communicate(timeout=10)
        assert process.returncode != 0
        assert "digest mismatch" in stderr
        assert "index" not in stderr
        assert not list(directory.glob("event-*.json"))
        assert not (directory / "shard.parquet").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
