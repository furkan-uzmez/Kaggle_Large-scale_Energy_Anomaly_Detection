"""RBF transport/coverage boundary tests with recording models, never real fits."""

import os
import sqlite3
from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq
import pytest

from project.experiments import full_candidate_io as io
from project.experiments import full_rbf_svc_worker as worker
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_campaign_worker import (
    ParentLifecycleService,
    _publish_message,
)
from project.experiments.full_input_schema import CORE
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
    _file_hash,
)
from project.experiments.full_snapshot_codec import (
    PROVENANCE_PATH,
    canonical_json,
    encode_snapshot,
    semantic_digest,
)
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1 as ENCODING
from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA

KEY = "ab" * 32


def real_smoke_fixture_spec(fixture):
    """Synthetic smoke declaration, never production admission."""
    import polars as pl
    split = pl.read_parquet(fixture.split_path).filter(pl.col("repeat") == 0)
    counts = {}
    for name in ("dev", "vault"):
        members = split.filter(pl.col("split") == name).select("building_id")
        counts[name] = fixture.full_frame.join(members, on="building_id", how="inner").height
    return replace(fixture.spec, development_rows=counts["dev"], vault_rows=counts["vault"])


def test_real_smoke_declares_actual_fixture_population(tmp_path):
    import polars as pl

    from full_campaign_fixtures import write_fixture
    fixture = write_fixture(tmp_path / "synthetic")
    spec = real_smoke_fixture_spec(fixture)
    split = pl.read_parquet(fixture.split_path).filter(pl.col("repeat") == 0)
    counts = {}
    for name in ("dev", "vault"):
        members = split.filter(pl.col("split") == name).select("building_id")
        counts[name] = fixture.full_frame.join(members, on="building_id", how="inner").height
    assert spec.development_rows == counts["dev"] == 52
    assert spec.vault_rows == counts["vault"] == 4


def api(name):
    value = getattr(worker, name, None)
    assert callable(value), f"missing dedicated RBF worker behavior: {name}"
    return value


def setup(tmp_path, arm="challenger", repeat=0, fold=0, *, timestamp_unit="ns", building_dtype="string"):
    root = tmp_path / "source"
    sources = []
    paths = {"data": io.DATA_PATH, "split": io.SPLIT_PATH, "notebook": "notebooks/source.ipynb",
             "matrix": "configs/matrix.yaml", "producer": "src/producer.py", "encoding": PROVENANCE_PATH}
    for role, path in paths.items():
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        body = '{"cells":[{"id":"rbf-cell"}]}' if role == "notebook" else (
            canonical_json({"encoding_id": ENCODING}) if role == "encoding" else "source metadata fixture only")
        file.write_text(body)
        sources.append(SourceBinding(role, path, _file_hash(file)))
    snapshot = FullCandidateSnapshot(CandidateSpec("dim_pca_vs_passthrough", CORE, "svm", ()),
                                    ("control", "challenger"), paths["notebook"], "rbf-cell")
    spec = FullRunSpec("rbf-worker-fixture", (snapshot,), tuple(sources), current_environment(),
                       FullResourcePolicy(True), development_rows=16, vault_rows=2)
    run_root = tmp_path / "run"
    run_root.mkdir()
    index = run_root / "index.sqlite3"
    types = {name: "string" if name == "primary_use" else "float64" for name in CORE}
    conn = sqlite3.connect(index)
    io._create_schema(conn, CORE, building_dtype, types)
    conn.execute("DROP TABLE dev_row_stage")
    for building in range(8):
        for target in (0, 1):
            row = building * 2 + target
            values = tuple("office" if name == "primary_use" else float(row + 1) for name in CORE)
            identity = building if building_dtype == "int64" else f"b{building}"
            base = 1700000000000000 if timestamp_unit == "us" else 1700000000000000000
            conn.execute("INSERT INTO dev_row VALUES (" + ",".join("?" for _ in range(13)) + ")",
                         (row, identity, base + row, target, *values))
        for r in range(3):
            f = (building // 2) if r == 0 else ((building % 4) if r == 1 else ((building // 2 + building % 2) % 4))
            identity = building if building_dtype == "int64" else f"b{building}"
            conn.execute("INSERT INTO assign VALUES (?,?,?,?)", (identity, r, f, 42 + r))
    meta = io._IndexMeta(spec.binding_sha256, sources[0].sha256, sources[1].sha256,
                         str(root / io.DATA_PATH), timestamp_unit, None, building_dtype, types, 16, io._identity_digest(conn))
    io._store_meta(conn, meta, CORE, spec)
    conn.execute("INSERT INTO meta VALUES ('integrity_sha256', ?)", (io._integrity_digest(conn),))
    conn.commit()
    conn.close()
    directory = run_root / "attempts" / f"attempt-{arm}-{repeat}-{fold}"
    provenance = sources[-1]
    payload = {"version": 1, "project_root": str(root), "spec": encode_snapshot(spec),
               "spec_sha256": spec.binding_sha256, "index_path": str(index), "index_sha256": _file_hash(index),
               "experiment_id": snapshot.spec.experiment_id, "arm_id": arm, "repeat": repeat, "fold": fold,
               "year_built_encoding": ENCODING, "provenance_path": PROVENANCE_PATH,
               "provenance_sha256": provenance.sha256, "semantic_sha256": semantic_digest(ENCODING, provenance.sha256),
               "work_id": f"dim_pca_vs_passthrough/{arm}/r{repeat}/f{fold}/model",
               "attempt_id": directory.name, "output_path": "shard.parquet", "parent_pid": os.getppid(),
               "ack_timeout_seconds": 1.0, "nonce": "a" * 64}
    return directory, payload, spec


def recording_fits(monkeypatch, events, *, fail=False):
    from project.features.selection import FoldLocalPreprocessor, _SparseFrameSVC
    def pre_fit(self, X, y=None):
        events.append(("pre-fit", self.use_pca, tuple(X.index)))
        self.feature_names_in_ = np.array(CORE)
        self.n_features_in_ = len(CORE)
        return self
    def transform(self, X):
        return np.ones((len(X), 2))
    def fit(self, X, y, sample_weight=None):
        events.append(("model-fit", len(y)))
        if fail:
            raise RuntimeError("recorded fit failed")
        self.classes_ = np.array([0, 1])
        return self
    def decision(self, X):
        return np.arange(len(X), dtype=float) - 2.0
    monkeypatch.setattr(FoldLocalPreprocessor, "fit", pre_fit)
    monkeypatch.setattr(FoldLocalPreprocessor, "transform", transform)
    monkeypatch.setattr(_SparseFrameSVC, "fit", fit)
    monkeypatch.setattr(_SparseFrameSVC, "decision_function", decision)


def exchange_for(directory, events, *, deny=False):
    service = ParentLifecycleService(directory, KEY, on_debit=lambda phase, event: events.append(("debit", phase)),
                                     on_shard_closed=lambda event: events.append(("closed", event["num_rows"])))
    def exchange(envelope):
        sequence = envelope["payload"]["seq"]
        _publish_message(directory / f"event-{sequence:04d}.json", envelope)
        if deny:
            raise TimeoutError("parent debit denied")
        service.service_once()
        from project.experiments.full_snapshot_codec import parse_json
        return parse_json((directory / f"ack-{sequence:04d}.json").read_text())
    return exchange, service


@pytest.mark.parametrize("arm", ["control", "challenger"])
def test_signed_backend_debits_before_pipeline_construction_and_writes_margin_shard(tmp_path, monkeypatch, arm):
    directory, payload, spec = setup(tmp_path, arm)
    payload = api("prepare_rbf_request")(directory, payload, KEY)
    events = []
    recording_fits(monkeypatch, events)
    original = worker.build_full_rbf_svc_arms
    def construct(*args, **kwargs):
        events.append(("construct",))
        return original(*args, **kwargs)
    monkeypatch.setattr(worker, "build_full_rbf_svc_arms", construct)
    exchange, service = exchange_for(directory, events)
    result = api("execute_rbf_request")(directory, KEY, exchange=exchange)
    assert events[:2] == [("debit", "model"), ("construct",)]
    assert events[-1] == ("closed", 4)
    assert result["model_fits"] == 1 and result["planned_model_fits"] == 24
    table = pq.read_table(directory / "shard.parquet")
    assert table.schema == CANDIDATE_OOF_SCHEMA
    assert table.num_rows == 4
    assert set(table["score_kind"].to_pylist()) == {"decision_margin"}
    assert set(table["config_sha256"].to_pylist()) == {spec.binding_sha256}
    assert set(table["n03_eligible"].to_pylist()) == {False}
    assert service.shard_evidence["sha256"] == _file_hash(directory / "shard.parquet")
    assert not list(directory.glob("*completed*"))


def test_rejects_unsupported_config_before_request_or_debit(tmp_path):
    directory, payload, spec = setup(tmp_path)
    changed = replace(spec, candidates=(replace(spec.candidates[0], spec=replace(spec.candidates[0].spec,
                                                                             model_params=(("kernel", "linear"),))),))
    payload.update(spec=encode_snapshot(changed), spec_sha256=changed.binding_sha256)
    with pytest.raises(ValueError):
        api("prepare_rbf_request")(directory, payload, KEY)
    assert not directory.exists()


@pytest.mark.parametrize("fault", ["signature", "index", "environment", "attempt", "parent"])
def test_authentication_failures_precede_debit(tmp_path, monkeypatch, fault):
    directory, payload, _ = setup(tmp_path)
    payload = api("prepare_rbf_request")(directory, payload, KEY)
    if fault == "signature":
        (directory / "worker-request.json").write_text('{"payload":{},"signature":"bad"}')
    elif fault == "index":
        with open(payload["index_path"], "ab") as stream:
            stream.write(b"tamper")
    elif fault == "environment":
        monkeypatch.setattr(worker, "current_environment", lambda: (("python", "wrong"),))
    elif fault == "parent":
        monkeypatch.setattr(worker.os, "getppid", lambda: -1)
    else:
        directory = directory.rename(directory.with_name("wrong-attempt"))
    with pytest.raises((ValueError, FileNotFoundError)):
        api("execute_rbf_request")(directory, KEY, exchange=lambda _: pytest.fail("debit forbidden"))
    assert not list(directory.glob("event-*.json"))
    assert not (directory / "shard.parquet").exists()


def test_denied_ack_and_fit_failure_never_report_shard_completion(tmp_path, monkeypatch):
    directory, payload, _ = setup(tmp_path / "denied")
    api("prepare_rbf_request")(directory, payload, KEY)
    events = []
    recording_fits(monkeypatch, events)
    exchange, service = exchange_for(directory, events, deny=True)
    with pytest.raises(TimeoutError):
        api("execute_rbf_request")(directory, KEY, exchange=exchange)
    assert events == [] and service.shard_evidence is None
    directory, payload, _ = setup(tmp_path / "failed")
    api("prepare_rbf_request")(directory, payload, KEY)
    recording_fits(monkeypatch, events, fail=True)
    exchange, service = exchange_for(directory, events)
    with pytest.raises(RuntimeError, match="recorded fit failed"):
        api("execute_rbf_request")(directory, KEY, exchange=exchange)
    assert service.shard_evidence is None
    assert not (directory / "shard.parquet").exists()


def test_validation_coverage_failure_never_acknowledges_shard(tmp_path, monkeypatch):
    directory, payload, _ = setup(tmp_path)
    api("prepare_rbf_request")(directory, payload, KEY)
    events = []
    recording_fits(monkeypatch, events)
    original = io.iter_full_validation_batches
    def corrupt(*args):
        for batch in original(*args):
            batch["target"] = 1 - batch["target"]
            yield batch
    monkeypatch.setattr(io, "iter_full_validation_batches", corrupt)
    exchange, service = exchange_for(directory, events)
    with pytest.raises(ValueError, match="coverage|identity|target"):
        api("execute_rbf_request")(directory, KEY, exchange=exchange)
    assert service.shard_evidence is None


def test_all_24_planned_units_have_one_recorded_model_fit_and_full_oof_coverage(tmp_path, monkeypatch):
    events = []
    recording_fits(monkeypatch, events)
    units = []
    rows_by_arm = {"control": 0, "challenger": 0}
    for repeat in range(3):
        for fold in range(4):
            for arm in rows_by_arm:
                directory, payload, spec = setup(tmp_path / f"{arm}-{repeat}-{fold}", arm, repeat, fold)
                payload = api("prepare_rbf_request")(directory, payload, KEY)
                exchange, _ = exchange_for(directory, events)
                result = api("execute_rbf_request")(directory, KEY, exchange=exchange)
                assert result["model_fits"] == 1
                units.append(payload["work_id"])
                rows_by_arm[arm] += result["num_rows"]
    assert set(units) == set(spec.planned_units) and len(units) == 24
    assert sum(event[0] == "model-fit" for event in events) == 24
    assert sum(event == ("debit", "model") for event in events) == 24
    assert rows_by_arm == {"control": 48, "challenger": 48}


def test_truncated_training_packet_fails_before_debit(tmp_path, monkeypatch):
    directory, payload, _ = setup(tmp_path)
    api("prepare_rbf_request")(directory, payload, KEY)
    original = io.load_full_training_partition
    def sampled(*args):
        partition = original(*args)
        return replace(partition, X_train=partition.X_train.iloc[:-2], y_train=partition.y_train.iloc[:-2],
                       groups_train=partition.groups_train.iloc[:-2])
    monkeypatch.setattr(io, "load_full_training_partition", sampled)
    with pytest.raises(ValueError, match="incomplete training"):
        api("execute_rbf_request")(directory, KEY, exchange=lambda _: pytest.fail("no debit allowed"))


def test_ack_signature_and_nonce_do_not_authorize_construction(tmp_path, monkeypatch):
    import hashlib
    import hmac
    directory, payload, _ = setup(tmp_path)
    api("prepare_rbf_request")(directory, payload, KEY)
    monkeypatch.setattr(worker, "build_full_rbf_svc_arms", lambda *a, **k: pytest.fail("construction forbidden"))
    def forged(envelope):
        ack = dict(envelope["payload"], status="authorized", nonce="b" * 64)
        return {"payload": ack, "signature": hmac.new(KEY.encode(), canonical_json(ack).encode(), hashlib.sha256).hexdigest()}
    with pytest.raises(ValueError, match="acknowledgement"):
        api("execute_rbf_request")(directory, KEY, exchange=forged)


def test_existing_attempt_replay_and_index_escape_fail_closed(tmp_path):
    directory, payload, _ = setup(tmp_path / "replay")
    api("prepare_rbf_request")(directory, payload, KEY)
    (directory / "event-0001.json").write_text("prior interrupted event")
    with pytest.raises(ValueError, match="fresh RBF attempt"):
        api("execute_rbf_request")(directory, KEY, exchange=lambda _: pytest.fail("no replay debit"))
    directory, payload, _ = setup(tmp_path / "escape")
    outside = tmp_path / "outside.sqlite3"
    outside.write_bytes(b"not development")
    payload.update(index_path=str(outside), index_sha256=_file_hash(outside))
    with pytest.raises(ValueError):
        api("prepare_rbf_request")(directory, payload, KEY)
    assert not directory.exists()


def test_float_targets_cannot_be_silently_coerced_to_integer_labels(tmp_path, monkeypatch):
    directory, payload, _ = setup(tmp_path)
    api("prepare_rbf_request")(directory, payload, KEY)
    events = []
    recording_fits(monkeypatch, events)
    original = io.iter_full_validation_batches
    def floats(*args):
        for batch in original(*args):
            batch["target"] = batch["target"].astype(float)
            yield batch
    monkeypatch.setattr(io, "iter_full_validation_batches", floats)
    exchange, service = exchange_for(directory, events)
    with pytest.raises(ValueError, match="integer"):
        api("execute_rbf_request")(directory, KEY, exchange=exchange)
    assert service.shard_evidence is None


def test_integer_buildings_and_microseconds_preserve_raw_identity(tmp_path, monkeypatch):
    import pyarrow as pa
    directory, payload, _ = setup(tmp_path, timestamp_unit="us", building_dtype="int64")
    api("prepare_rbf_request")(directory, payload, KEY)
    events = []
    recording_fits(monkeypatch, events)
    exchange, _ = exchange_for(directory, events)
    api("execute_rbf_request")(directory, KEY, exchange=exchange)
    table = pq.read_table(directory / "shard.parquet")
    assert table.schema.field("building_id").type == pa.int64()
    assert table.schema.field("timestamp").type == pa.timestamp("ns")
    assert table["building_id"].to_pylist() == [0, 0, 1, 1]
    assert table["timestamp"].cast(pa.int64()).to_pylist() == [1700000000000000000 + i * 1000 for i in range(4)]


def test_submicrosecond_corruption_is_not_truncated_during_identity_check(tmp_path, monkeypatch):
    directory, payload, _ = setup(tmp_path, timestamp_unit="us")
    api("prepare_rbf_request")(directory, payload, KEY)
    events = []
    recording_fits(monkeypatch, events)
    original = io.iter_full_validation_batches
    def shifted(*args):
        for batch in original(*args):
            batch["timestamp"] = batch["timestamp"].astype("datetime64[ns]") + np.timedelta64(1, "ns")
            yield batch
    monkeypatch.setattr(io, "iter_full_validation_batches", shifted)
    exchange, service = exchange_for(directory, events)
    with pytest.raises(ValueError, match="timestamp|identity"):
        api("execute_rbf_request")(directory, KEY, exchange=exchange)
    assert service.shard_evidence is None


def test_interrupted_fit_retains_debit_without_shard_acknowledgement(tmp_path, monkeypatch):
    from project.features.selection import _SparseFrameSVC
    directory, payload, _ = setup(tmp_path)
    api("prepare_rbf_request")(directory, payload, KEY)
    events = []
    recording_fits(monkeypatch, events)
    def interrupt(self, X, y, sample_weight=None):
        raise KeyboardInterrupt("explicit interruption")
    monkeypatch.setattr(_SparseFrameSVC, "fit", interrupt)
    exchange, service = exchange_for(directory, events)
    with pytest.raises(KeyboardInterrupt):
        api("execute_rbf_request")(directory, KEY, exchange=exchange)
    assert events[0] == ("debit", "model")
    assert service.shard_evidence is None
    assert len(list(directory.glob("event-*.json"))) == 1
    assert not (directory / "shard.parquet").exists()


def manual_real_two_arm_smoke(tmp_path):
    """Parent-only callable; NOT collected or executed by pytest.

    Run only after ordinary fixture fitting has stopped and parent permits it.
    This proves tiny actual model transport, not full-scale feasibility/admission.
    """
    results = []
    for arm in ("control", "challenger"):
        directory, payload, _ = setup(tmp_path / arm, arm)
        api("prepare_rbf_request")(directory, payload, KEY)
        events = []
        exchange, service = exchange_for(directory, events)
        results.append(api("execute_rbf_request")(directory, KEY, exchange=exchange))
        assert service.shard_evidence is not None
    return results
