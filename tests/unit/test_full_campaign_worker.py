"""Real subprocess worker and parent-only debit acknowledgements on tiny fixtures."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from full_campaign_fixtures import (
    NOTEBOOK_RELATIVE,
    int_development_frame,
    make_spec,
    write_fixture,
    write_int_fixture,
)
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_candidate_io import build_development_index
from project.experiments.full_run_contracts import FullCandidateSnapshot, SourceBinding
from project.experiments.full_snapshot_codec import encode_snapshot
from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    candidate_oof_schema,
)

KEY = "ab" * 32
ENCODING = "lead_year_offset_1900_uint8_v1"


def test_fixed_production_target_is_importable():
    assert importlib.util.find_spec("project.experiments.full_campaign_worker") is not None


def worker_api():
    from project.experiments import full_campaign_worker
    return full_campaign_worker


def setup_request(tmp_path, recipe="feat_missingness_indicators", ack_timeout_seconds=3.0):
    fixture = write_fixture(tmp_path / "project")
    fixture.full_frame.with_columns(
        pl.when(pl.col("hour") % 3 == 0).then(0)
        .when(pl.col("hour") % 3 == 1).then(100).otherwise(255).alias("year_built"),
    ).write_parquet(fixture.data_path)
    spec = replace(make_spec(fixture.root), development_rows=52, vault_rows=4)
    candidate = CandidateSpec(recipe, spec.candidates[0].spec.feature_pool,
                              "logistic_l2" if "selector" in recipe else "hist_gradient_boosting", ())
    snapshot = FullCandidateSnapshot(candidate, ("control", "challenger"), NOTEBOOK_RELATIVE,
                                     "real-cell", selector_arms=("challenger",) if "selector" in recipe else ())
    provenance = fixture.root / "docs/evidence/year_built_encoding_v1.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"encoding_id": ENCODING, "contract": "test-only verified fixture"}))
    provenance_sha = hashlib.sha256(provenance.read_bytes()).hexdigest()
    spec = replace(spec, candidates=(snapshot,), sources=spec.sources + (
        SourceBinding("encoding", str(provenance.relative_to(fixture.root)), provenance_sha),))
    run_root = tmp_path / "run"
    run_root.mkdir()
    index = build_development_index(fixture.root, spec, run_root / "index.sqlite3")
    directory = run_root / "attempts/attempt-1"
    encoding = ENCODING
    semantic_sha = worker_api().semantic_digest(encoding, provenance_sha)
    payload = {
        "version": 1, "project_root": str(fixture.root), "spec": encode_snapshot(spec),
        "spec_sha256": spec.binding_sha256, "index_path": str(index),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "experiment_id": recipe, "arm_id": "challenger", "repeat": 0, "fold": 0,
        "year_built_encoding": encoding, "provenance_path": str(provenance.relative_to(fixture.root)),
        "provenance_sha256": provenance_sha, "semantic_sha256": semantic_sha,
        "work_id": f"{recipe}/challenger/r0/f0/model", "attempt_id": "attempt-1",
        "output_path": "shard.parquet", "parent_pid": os.getpid(), "ack_timeout_seconds": ack_timeout_seconds,
        "nonce": "a" * 64,
    }
    payload = worker_api().prepare_worker_request(directory, payload, KEY)
    return directory, payload, spec


def setup_int_request(tmp_path, recipe="feat_missingness_indicators", ack_timeout_seconds=3.0):
    """Mirror setup_request over an integer-building frame with no vault partition."""
    fixture = write_int_fixture(tmp_path / "project", vault=True)
    fixture.full_frame.with_columns(
        pl.when(pl.col("hour") % 3 == 0).then(0)
        .when(pl.col("hour") % 3 == 1).then(100).otherwise(255).alias("year_built"),
    ).write_parquet(fixture.data_path)
    spec = replace(make_spec(fixture.root), development_rows=int_development_frame().height,
                   vault_rows=4)
    candidate = CandidateSpec(recipe, spec.candidates[0].spec.feature_pool,
                              "logistic_l2" if "selector" in recipe else "hist_gradient_boosting", ())
    snapshot = FullCandidateSnapshot(candidate, ("control", "challenger"), NOTEBOOK_RELATIVE,
                                     "real-cell", selector_arms=("challenger",) if "selector" in recipe else ())
    provenance = fixture.root / "docs/evidence/year_built_encoding_v1.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"encoding_id": ENCODING, "contract": "test-only verified fixture"}))
    provenance_sha = hashlib.sha256(provenance.read_bytes()).hexdigest()
    spec = replace(spec, candidates=(snapshot,), sources=spec.sources + (
        SourceBinding("encoding", str(provenance.relative_to(fixture.root)), provenance_sha),))
    run_root = tmp_path / "run-int"
    run_root.mkdir()
    index = build_development_index(fixture.root, spec, run_root / "index.sqlite3")
    directory = run_root / "attempts/attempt-1"
    encoding = ENCODING
    semantic_sha = worker_api().semantic_digest(encoding, provenance_sha)
    payload = {
        "version": 1, "project_root": str(fixture.root), "spec": encode_snapshot(spec),
        "spec_sha256": spec.binding_sha256, "index_path": str(index),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "experiment_id": recipe, "arm_id": "challenger", "repeat": 0, "fold": 0,
        "year_built_encoding": encoding, "provenance_path": str(provenance.relative_to(fixture.root)),
        "provenance_sha256": provenance_sha, "semantic_sha256": semantic_sha,
        "work_id": f"{recipe}/challenger/r0/f0/model", "attempt_id": "attempt-1",
        "output_path": "shard.parquet", "parent_pid": os.getpid(), "ack_timeout_seconds": ack_timeout_seconds,
        "nonce": "a" * 64,
    }
    payload = worker_api().prepare_worker_request(directory, payload, KEY)
    return directory, payload, spec


def launch(directory, payload):
    env = {**os.environ, "FULL_CAMPAIGN_WORKER_KEY": KEY,
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1"}
    return subprocess.Popen([
        sys.executable, "-c",
        "from project.experiments.full_campaign_worker import run; import sys; run(sys.argv[1],sys.argv[2])",
        payload["work_id"], str(directory),
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def supervise(process, directory, *, deny=False, on_debit=None,
              on_selector_success=None, on_shard_closed=None):
    observed = []
    debits = on_debit or (lambda phase, event: observed.append(("debit", phase)))
    selector_success = on_selector_success or (lambda event: observed.append(("success", "selector")))
    shard_closed = on_shard_closed or (lambda event: observed.append(("closed", "model")))
    service = worker_api().ParentLifecycleService(
        directory, KEY,
        on_debit=debits,
        on_selector_success=selector_success,
        on_shard_closed=shard_closed,
    )
    deadline = time.monotonic() + 20
    try:
        while process.poll() is None and time.monotonic() < deadline:
            if not deny:
                service.service_once()
            time.sleep(.01)
        stdout, stderr = process.communicate(timeout=5)
        return observed, stdout.decode(), stderr.decode(), service
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("recipe", ["feat_missingness_indicators", "feat_missingness_bitmask",
                                     "feat_safe_calendar_numeric_transforms"])
def test_real_worker_closes_exact_shard_only_after_parent_debit(tmp_path, recipe):
    directory, payload, spec = setup_request(tmp_path, recipe)
    process = launch(directory, payload)
    phases, _, stderr, service = supervise(process, directory)
    assert process.returncode == 0, stderr
    assert phases == [("debit", "model"), ("closed", "model")]
    table = pq.read_table(directory / "shard.parquet")
    assert table.schema == CANDIDATE_OOF_SCHEMA
    assert table.num_rows == 7
    assert table.column("config_sha256").to_pylist() == [spec.binding_sha256] * 7
    assert table.column("execution_mode").to_pylist() == ["full"] * 7
    assert table.column("n03_eligible").to_pylist() == [False] * 7
    assert service.shard_evidence["sha256"] == hashlib.sha256((directory / "shard.parquet").read_bytes()).hexdigest()
    assert not list(directory.glob("*completed*"))


def test_real_worker_closes_integer_building_shard(tmp_path):
    """Regression: int64 building rows must produce an int64 OOF shard schema."""
    directory, payload, _spec = setup_int_request(tmp_path)
    process = launch(directory, payload)
    phases, _, stderr, service = supervise(process, directory)
    assert process.returncode == 0, stderr
    assert phases == [("debit", "model"), ("closed", "model")]
    table = pq.read_table(directory / "shard.parquet")
    expected = candidate_oof_schema(pa.int64(), pa.timestamp("ns"))
    assert table.schema.equals(expected, check_metadata=True)
    assert table.num_rows == 7
    assert all(isinstance(value, int) for value in table.column("building_id").to_pylist())
    assert service.shard_evidence["sha256"] == hashlib.sha256((directory / "shard.parquet").read_bytes()).hexdigest()


def test_selector_success_ack_precedes_model_debit(tmp_path):
    directory, payload, _ = setup_request(tmp_path, "dim_fold_local_selector_vs_passthrough")
    process = launch(directory, payload)
    phases, _, stderr, _ = supervise(process, directory)
    assert process.returncode == 0, stderr
    assert phases == [("debit", "selector"), ("success", "selector"),
                      ("debit", "model"), ("closed", "model")]


def test_selector_success_uses_generous_ack_budget(tmp_path):
    """The selector_success phase must survive a slow parent verify callback."""
    directory, payload, _ = setup_request(
        tmp_path, "dim_fold_local_selector_vs_passthrough", ack_timeout_seconds=1.0)
    process = launch(directory, payload)
    observed, _, stderr, _ = supervise(
        process, directory,
        on_selector_success=lambda event: time.sleep(3.0),
    )
    assert process.returncode == 0, stderr
    assert "acknowledgement timed out" not in stderr
    assert observed == [("debit", "selector"), ("debit", "model"), ("closed", "model")]


def test_slow_debit_still_times_out_on_tight_budget(tmp_path):
    """Debits must NOT inherit the generous selector_success budget."""
    directory, payload, _ = setup_request(
        tmp_path, "feat_missingness_indicators", ack_timeout_seconds=1.0)
    process = launch(directory, payload)
    _, _, stderr, _ = supervise(
        process, directory,
        on_debit=lambda phase, event: time.sleep(3.0),
    )
    assert process.returncode != 0
    assert "acknowledgement timed out" in stderr


def test_selector_ack_budget_is_finite_and_scoped():
    api = worker_api()
    assert api._ack_budget_seconds("debit", 3.0) == 3.0
    assert api._ack_budget_seconds("shard_closed", 3.0) == 3.0
    assert api._ack_budget_seconds("selector_success", 3.0) == api.SELECTOR_SUCCESS_ACK_TIMEOUT_SECONDS
    assert 0 < api.SELECTOR_SUCCESS_ACK_TIMEOUT_SECONDS <= 600.0


def test_missing_ack_blocks_fit_and_never_writes_shard(tmp_path):
    directory, payload, _ = setup_request(tmp_path)
    process = launch(directory, payload)
    phases, _, stderr, _ = supervise(process, directory, deny=True)
    assert process.returncode != 0
    assert "acknowledgement" in stderr
    assert phases == []
    assert not (directory / "shard.parquet").exists()
    assert len(list(directory.glob("event-*.json"))) == 1


def test_invalid_ack_does_not_authorize_fit(tmp_path):
    directory, payload, _ = setup_request(tmp_path)
    (directory / "ack-0001.json").write_text('{"payload":{},"signature":"bad"}')
    process = launch(directory, payload)
    _, _, stderr, _ = supervise(process, directory, deny=True)
    assert process.returncode != 0
    assert "signature" in stderr
    assert not (directory / "shard.parquet").exists()


def test_tampered_request_and_index_fail_before_debit(tmp_path):
    directory, payload, _ = setup_request(tmp_path)
    request = directory / "worker-request.json"
    envelope = json.loads(request.read_text())
    envelope["payload"]["year_built_encoding"] = "absolute_calendar_year"
    request.write_text(json.dumps(envelope))
    process = launch(directory, payload)
    _, _, stderr, _ = supervise(process, directory, deny=True)
    assert process.returncode != 0 and "signature" in stderr
    assert not list(directory.glob("event-*.json"))


def test_prediction_failure_never_requests_model_completion(tmp_path, monkeypatch):
    directory, payload, _ = setup_request(tmp_path)
    api = worker_api()
    def fail(*args):
        raise RuntimeError("prediction failed")
    monkeypatch.setattr(api, "extract_candidate_score", fail)
    # Real fit, same handshake; only the prediction boundary is fault-injected.
    import threading
    service = api.ParentLifecycleService(directory, KEY, on_debit=lambda *args: None,
                                         on_selector_success=lambda *args: None,
                                         on_shard_closed=lambda *args: pytest.fail("must not complete"))
    stopped = threading.Event()
    def parent():
        while not stopped.is_set():
            service.service_once()
            time.sleep(.01)
    thread = threading.Thread(target=parent)
    monkeypatch.setenv("FULL_CAMPAIGN_WORKER_KEY", KEY)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="prediction failed"):
            api.run(payload["work_id"], directory)
    finally:
        stopped.set()
        thread.join()
    assert service.shard_evidence is None
    assert len(list(directory.glob("event-*.json"))) == 1
