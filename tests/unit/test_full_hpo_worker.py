"""Dedicated transport integration with recording pipelines, never real fits."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import MethodType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_full_nested_tuning import _make_hpo_journal_fixture

from candidate_fixtures import synthetic_partitions
from project.experiments import full_hpo_worker as backend
from project.experiments.candidate_contracts import (
    _validate_managed_path,
    training_view,
)
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_hpo_lifecycle import WorkerHpoLifecycle, sign_message
from project.experiments.full_nested_tuning import (
    _write_inner_fold_evidence,
    _write_refit_evidence,
    full_hpo_study_spec,
)
from project.experiments.full_run_contracts import (
    ArtifactEvidence,
    FullFitJournal,
    SourceBinding,
)
from project.experiments.full_snapshot_codec import PROVENANCE_PATH, canonical_json
from project.experiments.inner_cv import build_inner_folds
from project.experiments.nested_tuning import build_hgb_tuning_arm
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1


@pytest.fixture
def authority(tmp_path):
    old, spec = _make_hpo_journal_fixture(tmp_path, full_hpo_study_spec())
    source_root = old.source_root
    old.close()
    encoding = source_root / PROVENANCE_PATH
    encoding.parent.mkdir(parents=True)
    encoding.write_text(canonical_json({"encoding_id": LEAD_YEAR_OFFSET_1900_UINT8_V1}))
    binding = SourceBinding("encoding", PROVENANCE_PATH, hashlib.sha256(encoding.read_bytes()).hexdigest())
    spec = replace(spec, sources=(*spec.sources, binding), environment=current_environment())
    journal = FullFitJournal(tmp_path / "transport-run", spec, source_root=source_root)
    training = training_view(synthetic_partitions()[0])
    yield journal, training
    journal.close()


def test_backend_has_dedicated_versioned_authenticated_request(authority):
    journal, training = authority
    assert callable(getattr(backend, "prepare_hpo_request", None)), "dedicated HPO request contract missing"
    directory = journal.root / "hpo_sessions" / ("a" * 32)
    request = backend.prepare_hpo_request(directory, journal, training, "fixture-key")
    actual, plan = backend.authenticate_hpo_request(directory, "fixture-key")
    assert actual == request
    assert actual["kind"] == "full_hpo"
    assert actual["version"] == 1
    assert plan.work_id.endswith("/hpo_session")
    assert actual["session_id"] not in journal.spec.planned_units
    assert journal.used_model_fits == 0


def test_dedicated_backend_runs_26_recording_fits_and_verified_resume(authority):
    journal, training = authority
    assert callable(getattr(backend, "make_parent_hpo_service", None)), "parent HPO evidence service missing"
    assert callable(getattr(backend, "execute_hpo_request", None)), "mechanical HPO backend missing"
    directory = journal.root / "hpo_sessions" / ("b" * 32)
    backend.prepare_hpo_request(directory, journal, training, "fixture-key")
    service = backend.make_parent_hpo_service(directory, journal, "fixture-key")
    calls = []
    def recording_factory(params):
        model = build_hgb_tuning_arm(params)  # construction only; genuine fit is replaced
        def record_fit(self, X, y):
            assert journal.used_model_fits == len(calls) + 1
            calls.append((tuple(X.index), tuple(y.index)))
            return self
        model.fit = MethodType(record_fit, model)
        model.predict_proba = MethodType(lambda self, X: np.full((len(X), 2), 0.5), model)
        return model
    result = backend.execute_hpo_request(directory, "fixture-key", service.lifecycle.handle,
                                         arm_factory=recording_factory)
    assert len(calls) == 26
    assert journal.used_model_fits == 26
    assert result.selected_trial_id == 0
    assert len(journal.verify_completed()) == 26
    resumed_dir = journal.root / "hpo_sessions" / ("c" * 32)
    backend.prepare_hpo_request(resumed_dir, journal, training, "fixture-key")
    resumed_service = backend.make_parent_hpo_service(resumed_dir, journal, "fixture-key")
    resumed = backend.execute_hpo_request(resumed_dir, "fixture-key", resumed_service.lifecycle.handle,
        arm_factory=lambda *_: pytest.fail("completed refit must not be reconstructed"))
    assert resumed.default_model is None and resumed.selected_model is None
    assert journal.used_model_fits == 26


@pytest.mark.parametrize("tamper", ["group_key", "row_key", "target", "train_membership", "config"])
def test_parent_rejects_self_consistent_but_false_scientific_evidence(authority, tamper):
    journal, _training = authority
    directory = journal.root / "hpo_sessions" / ("d" * 32)
    request = backend.prepare_hpo_request(directory, journal, _training, "fixture-key")
    _, plan = backend.authenticate_hpo_request(directory, "fixture-key")
    training = backend.load_hpo_training(directory, request)
    service = backend.make_parent_hpo_service(directory, journal, "fixture-key")
    worker = WorkerHpoLifecycle(plan, "fixture-key", service.lifecycle.handle)
    worker.run_fit(plan.default(), object, lambda _, record: _write_refit_evidence(
        journal.root / record.artifact_directory, "default_refit", dict(plan.default().parameters),
        len(training.X_train), unit=record.unit))
    train, valid = build_inner_folds(training.y_train, training.groups_train, n_splits=3, seed=42)[0]
    identity = plan.inner(0, 0)
    class RecordingPrediction:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)
    def forged_evidence(model, record):
        params = dict(identity.parameters)
        params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        _, proof = _write_inner_fold_evidence(journal.root / record.artifact_directory, 0, 0,
            params_hash, valid, model, training, parameters=params, unit=record.unit, train_positions=train)
        parquet_path = journal.root / proof[0].path
        metadata_path = journal.root / proof[1].path
        metadata = json.loads(metadata_path.read_text())
        if tamper in ("row_key", "group_key", "target"):
            table = pa.Table.from_batches(list(pq.ParquetFile(parquet_path).iter_batches(batch_size=3)))
            column = table.schema.get_field_index(tamper)
            values = ([1 - int(v) for v in table.column(column).to_pylist()] if tamper == "target"
                      else ["forged-key"] * len(table))
            table = table.set_column(column, table.schema.field(column), pa.array(values, type=table.schema.field(column).type))
            pq.write_table(table, parquet_path)
            metadata["parquet_sha256"] = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        elif tamper == "train_membership":
            metadata["train_positions_sha256"] = "1" * 64
        else:
            metadata["config_sha256"] = "1" * 64
        metadata["sha256"] = hashlib.sha256(canonical_json({k: v for k, v in metadata.items() if k != "sha256"}).encode()).hexdigest()
        metadata_path.write_text(canonical_json(metadata))
        return tuple(ArtifactEvidence(item.path, hashlib.sha256((journal.root / item.path).read_bytes()).hexdigest()) for item in proof)
    with pytest.raises(ValueError, match="key|membership|identity|target|probability"):
        worker.run_fit(identity, RecordingPrediction, forged_evidence)
    assert journal.used_model_fits == 2
    assert journal.attempts()[-1].status == "awaiting_validation"
    service.lifecycle.terminalize_owned("failed", reason="fixture scientific proof rejected after worker exit")
    assert journal.attempts()[-1].status == "failed"


def test_real_file_transport_is_exclusive_and_cannot_restart_same_session(authority, monkeypatch):
    journal, training = authority
    directory = journal.root / "hpo_sessions" / ("e" * 32)
    request = backend.prepare_hpo_request(directory, journal, training, "fixture-key", ack_timeout_seconds=3.0)
    service = backend.make_parent_hpo_service(directory, journal, "fixture-key")
    def recording_factory(params):
        model = build_hgb_tuning_arm(params)
        model.fit = MethodType(lambda self, X, y: self, model)
        model.predict_proba = MethodType(lambda self, X: np.full((len(X), 2), 0.5), model)
        return model
    def forbidden_collect(*args, **kwargs):
        pytest.fail("population read_table must not replace streamed HPO batches")
    monkeypatch.setattr(pq, "read_table", forbidden_collect)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend.execute_hpo_request, directory, "fixture-key", arm_factory=recording_factory)
        deadline = time.monotonic() + 20
        while not future.done() and time.monotonic() < deadline:
            service.service_once()
            time.sleep(0.005)
        result = future.result(timeout=4)
    assert result.selected_trial_id == 0
    assert len(tuple(directory.glob("hpo-event-*.json"))) == 52
    assert len(tuple(directory.glob("hpo-ack-*.json"))) == 52
    assert request["session_id"] not in {record.attempt_id for record in journal.attempts()}
    with pytest.raises(ValueError, match="fresh|session"):
        backend.make_parent_hpo_service(directory, journal, "fixture-key")
    first_event = directory / "hpo-event-0001.json"
    first_event.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        service.service_once()


@pytest.mark.parametrize("field,value", [
    ("version", True), ("kind", "ordinary"), ("repeat", True),
    ("environment_sha256", "1" * 64), ("semantic_sha256", "1" * 64),
    ("training_schema_sha256", "1" * 64), ("training_path", "../training.parquet"),
    ("session_id", "1" * 32), ("nonce", "short"),
])
def test_even_signed_request_mutations_fail_before_debit(authority, field, value):
    journal, training = authority
    directory = journal.root / "hpo_sessions" / ("f" * 32)
    request = backend.prepare_hpo_request(directory, journal, training, "fixture-key")
    request[field] = value
    (directory / "hpo-request.json").write_text(canonical_json(sign_message(request, "fixture-key")))
    with pytest.raises((ValueError, TypeError)):
        backend.authenticate_hpo_request(directory, "fixture-key")
    assert journal.used_model_fits == 0


def test_file_messages_are_bounded_and_exclusive(authority):
    journal, training = authority
    directory = journal.root / "hpo_sessions" / ("9" * 32)
    backend.prepare_hpo_request(directory, journal, training, "fixture-key")
    with pytest.raises(ValueError, match="bounded|large"):
        backend._publish(directory / "oversized.json", {"payload": "x" * 131073})
    assert not (directory / "oversized.json").exists()
    backend._publish(directory / "exclusive.json", {"payload": "once"})
    with pytest.raises(FileExistsError):
        backend._publish(directory / "exclusive.json", {"payload": "twice"})


def test_publication_never_exposes_a_transient_hardlink(authority, monkeypatch):
    journal, training = authority
    directory = journal.root / "hpo_sessions" / ("8" * 32)
    backend.prepare_hpo_request(directory, journal, training, "fixture-key")
    original_link = backend.os.link
    def observing_parent_at_link(source, destination):
        original_link(source, destination)
        # Deterministically observe the exact window seen by a concurrent parent.
        _validate_managed_path(destination, "concurrent HPO reader", root=directory)
    monkeypatch.setattr(backend.os, "link", observing_parent_at_link)
    destination = directory / "atomic.json"
    backend._publish(destination, {"payload": "complete"})
    assert destination.stat().st_nlink == 1
    assert _validate_managed_path(destination, "published HPO message", root=directory) == destination
