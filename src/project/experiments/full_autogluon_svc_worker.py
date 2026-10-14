"""Authenticated outer OOF transport to the isolated reviewed AutoGluon backend."""
from __future__ import annotations

import hashlib
import hmac
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from project.experiments import full_candidate_io as io
from project.experiments.candidate_contracts import _validate_managed_path
from project.experiments.candidate_dispatch import _configuration
from project.experiments.full_autogluon_contract import (
    EXPERIMENT_ID, IMPLEMENTATION_ROOT, APPROVAL_SHA256,
    build_reviewed_autogluon_arms, validate_full_autogluon_snapshot,
    verify_autogluon_environment, verify_autogluon_profile_source,
)
from project.experiments.full_campaign_runtime import run_work_unit, confirm_owned_attempt_exit
from project.experiments.full_campaign_worker import (
    ParentLifecycleService, _authenticate_request, _build_batch_table, _publish_message,
    prepare_worker_request,
)
from project.experiments.full_rbf_svc_worker import (
    RBFWorkerLifecycle, _verify_inputs, _verify_training, _verify_batch,
)
from project.experiments.full_run_contracts import _file_hash
from project.experiments.full_snapshot_codec import (
    canonical_json, configuration_digest, exact_fields, parse_json, validate_request,
)
from project.models import full_autogluon_bridge as bridge
from project.models import full_autogluon_worker as backend
from project.models.candidate_artifacts import MAX_BATCH_ROWS, candidate_oof_schema, write_candidate_oof_batches

BACKEND_TARGET = "project.experiments.full_autogluon_svc_worker:run"
INNER_TARGET = "project.models.full_autogluon_worker:run"
BACKEND_ARTIFACTS = ("worker-request.json", "backend-request.json", "scores.parquet", "response.json",
                     "backend-started.json", "backend-status.json", "event-0001.json", "ack-0001.json",
                     "worker-identity.json", "startup-authorized.json", "stdout.log", "stderr.log",
                     "train.parquet", "predict.parquet", "request.json", "supervisor.json")


def _validate_payload(request):
    spec = validate_request(request)
    if request["experiment_id"] != EXPERIMENT_ID or request["arm_id"] != "challenger":
        raise ValueError("reviewed AutoGluon challenger required")
    snapshot = next(item for item in spec.candidates if item.spec.experiment_id == EXPERIMENT_ID)
    validate_full_autogluon_snapshot(snapshot, year_built_encoding=request["year_built_encoding"])
    return spec, snapshot


def prepare_autogluon_request(directory: Path, payload: dict, key: str) -> dict:
    spec, _ = _validate_payload(payload)
    directory = _validate_managed_path(directory, "AutoGluon attempt", directory=True)
    _verify_inputs(directory, payload, spec)
    verify_autogluon_profile_source(spec, payload["project_root"])
    verify_autogluon_environment(payload["project_root"])
    return prepare_worker_request(directory, payload, key)


def _inputs(directory, request):
    spec, snapshot = _validate_payload(request)
    if directory.name != request["attempt_id"] or directory.parent.name != "attempts":
        raise ValueError("AutoGluon run-owned attempt identity required")
    index = _verify_inputs(directory, request, spec)
    verify_autogluon_profile_source(spec, request["project_root"])
    return spec, snapshot, index


def authenticate_autogluon_request(directory: Path, key: str):
    request, _ = _authenticate_request(directory, key)
    spec, snapshot, index = _inputs(directory, request)
    if request["parent_pid"] != os.getppid():
        raise ValueError("AutoGluon supervisor parent identity mismatch")
    return request, spec, snapshot, index


class _BackendLifecycleService(ParentLifecycleService):
    """Inner lifecycle has the same event protocol, but an outer-owned directory."""

    def _initialize(self):
        envelope = exact_fields(parse_json((self.directory / "worker-request.json").read_text()), {"payload", "signature"})
        request = envelope["payload"]
        signature = bridge.authenticate(request, self.key.encode())
        if not hmac.compare_digest(signature, envelope["signature"]):
            raise ValueError("invalid inner lifecycle request signature")
        _validate_payload(request)
        if (self.directory.name != "backend" or self.directory.parent.name != request["attempt_id"]
                or request["parent_pid"] != os.getpid()):
            raise ValueError("inner lifecycle directory/parent identity mismatch")
        self._request = request
        self._states = [("debit", "model")]


def _write_private_parquet(path, tables, schema):
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        with pq.ParquetWriter(stream, schema) as writer:
            for table in tables:
                writer.write_table(table, row_group_size=MAX_BATCH_ROWS)
        stream.flush()
        os.fsync(stream.fileno())


def _validation_cursor(connection, request):
    return connection.execute(
        "SELECT d.row_index,d.building_id,d.timestamp_ns,d.target FROM dev_row d JOIN assign a USING(building_id)"
        " WHERE a.repeat=? AND a.fold=? ORDER BY d.row_index", (request["repeat"], request["fold"]))


def _prediction_tables(index, spec, request, connection, meta, train_schema):
    expected = _validation_cursor(connection, request)
    total = 0
    for batch in io.iter_full_validation_batches(index, spec, request["repeat"], request["fold"]):
        _verify_batch(batch, expected, request, meta.timestamp_unit)
        features = pa.Table.from_pandas(batch["X"], schema=train_schema, preserve_index=False)
        total += features.num_rows
        yield features.add_column(0, "row_index", pa.array(batch["row_index"], type=pa.int64()))
    if expected.fetchone() is not None or total == 0:
        raise ValueError("AutoGluon incomplete validation coverage")


def _rows(path, columns=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=MAX_BATCH_ROWS, columns=columns):
        yield from zip(*(column.to_pylist() for column in batch.columns), strict=True)


def _canonical_batches(index, spec, snapshot, request, backend_dir, connection, meta):
    arm = build_reviewed_autogluon_arms(snapshot, year_built_encoding=request["year_built_encoding"])["challenger"]
    fit_sha = configuration_digest(_configuration(arm.transformer_or_pipeline))
    schema = candidate_oof_schema(pa.int64() if meta.building_dtype == "int64" else pa.string(),
                                  pa.timestamp(meta.timestamp_unit, tz=meta.timestamp_time_zone))
    scores = iter(_rows(backend_dir / "scores.parquet", ["row_index", "probability"]))
    expected = _validation_cursor(connection, request)
    for batch in io.iter_full_validation_batches(index, spec, request["repeat"], request["fold"]):
        _verify_batch(batch, expected, request, meta.timestamp_unit)
        values = []
        for identity in batch["row_index"]:
            score = next(scores, None)
            if score is None or score[0] != int(identity):
                raise ValueError("AutoGluon score/index mapping mismatch")
            values.append(score[1])
        yield _build_batch_table(batch, np.asarray(values, dtype=np.float64), snapshot.spec, arm, spec,
                                 request["repeat"], request["fold"], fit_sha, schema)
    if next(scores, None) is not None or expected.fetchone() is not None:
        raise ValueError("AutoGluon extra or missing canonical score coverage")


def _launch_backend(directory, request, spec, key, facts):
    backend_dir = directory / "backend"
    binding = {"run_id": spec.run_id, "attempt_id": request["attempt_id"], "config_hash": spec.binding_sha256,
               "data_hash": next(s.sha256 for s in spec.sources if s.role == "data"),
               "split_hash": next(s.sha256 for s in spec.sources if s.role == "split")}
    child_request = dict(request, parent_pid=os.getpid())
    _publish_message(backend_dir / "worker-request.json", {"payload": child_request,
                     "signature": bridge.authenticate(child_request, key.encode())})
    transport = bridge.prepare_request(backend_dir, binding, facts)
    transport["prediction_batch_rows"] = min(MAX_BATCH_ROWS, backend.BATCH_ROWS)
    transport = backend.bind_worker_request(backend_dir, transport)
    backend._publish(backend_dir, "backend-request.json", {"payload": transport,
                     "signature": bridge.authenticate(transport, key.encode())})
    interpreter = facts["executable"]
    bridge.supervisor_command(interpreter, Path(backend.__file__).absolute(),
        {"environment": facts, "worker_sha256": transport["backend_binding"]["worker_sha256"]})
    def acknowledged(phase, event):
        if phase != "model" or event["unit"] != request["work_id"] or (backend_dir / "backend-models").exists():
            raise ValueError("backend debit must precede construction after outer authorization")
    service = _BackendLifecycleService(backend_dir, key, on_debit=acknowledged)
    previous = sys.path
    try:
        # Runtime collects sys.path into PYTHONPATH. Never forward Python3.13 sites.
        sys.path = [str(IMPLEMENTATION_ROOT / "src")]
        outcome = run_work_unit(request["work_id"], INNER_TARGET, backend_dir,
            python_executable=interpreter, spec_binding=spec.binding_sha256,
            environment_binding={"PYTHONPATH": str(IMPLEMENTATION_ROOT / "src"), "FULL_CAMPAIGN_WORKER_KEY": key},
            expected_artifacts=("scores.parquet", "response.json", "backend-status.json"),
            poll_callback=service.service_once,
            attempt_binding={"attempt_id": request["attempt_id"], "nonce": request["nonce"],
                             "request_hash": hashlib.sha256(bridge._canonical(transport)).hexdigest()})
    finally:
        sys.path = previous
    _publish_message(backend_dir / "supervisor.json", asdict(outcome))
    if outcome.exit_code != 0 or outcome.status != "awaiting_validation" or not outcome.observation["cleanup_confirmed"]:
        raise ValueError("supervised AutoGluon backend failed; preserve debit and reconcile")
    confirm_owned_attempt_exit(parse_json((backend_dir / "worker-identity.json").read_text()))
    _verify_backend_envelopes(backend_dir, key)


def execute_autogluon_request(directory: Path, key: str, *, exchange: Callable | None = None) -> dict:
    directory = _validate_managed_path(directory, "AutoGluon attempt", directory=True)
    request, spec, snapshot, index = authenticate_autogluon_request(directory, key)
    if (list(directory.glob("event-*.json")) or list(directory.glob("ack-*.json"))
            or (directory / "shard.parquet").exists() or (directory / "backend").exists()):
        raise ValueError("fresh AutoGluon attempt required")
    if "AG_MEMORY_LIMIT_IN_GB" in os.environ:
        raise ValueError("inherited AutoGluon memory soft cap forbidden")
    facts = verify_autogluon_environment(request["project_root"])
    connection, meta = io._open_bound_index(index, spec)
    try:
        training = io.load_full_training_partition(index, spec, request["repeat"], request["fold"])
        _verify_training(connection, training, request)
        lifecycle = RBFWorkerLifecycle(directory, key, request, exchange)
        lifecycle.send("debit")
        backend_dir = directory / "backend"
        backend_dir.mkdir(mode=0o700)
        features = pa.Table.from_pandas(training.X_train, preserve_index=False).replace_schema_metadata(None)
        train = features.append_column("anomaly", pa.array(training.y_train, type=pa.int64()))
        _write_private_parquet(backend_dir / "train.parquet", (train,), train.schema)
        del train, training
        predict_schema = pa.schema([pa.field("row_index", pa.int64()), *features.schema])
        train_schema = features.schema
        del features
        _write_private_parquet(backend_dir / "predict.parquet",
            _prediction_tables(index, spec, request, connection, meta, train_schema), predict_schema)
        _launch_backend(directory, request, spec, key, facts)
        schema = candidate_oof_schema(pa.int64() if meta.building_dtype == "int64" else pa.string(),
                                     pa.timestamp(meta.timestamp_unit, tz=meta.timestamp_time_zone))
        output = directory / "shard.parquet"
        write_candidate_oof_batches(output, _canonical_batches(index, spec, snapshot, request, backend_dir, connection, meta), schema)
        proof = verify_autogluon_unit(directory, key, expected_request=request)
        with output.open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        lifecycle.send("shard_closed", sha256=proof["shard_sha256"], num_rows=proof["num_rows"], path=str(output))
        return proof
    finally:
        connection.close()


def _verify_backend_envelopes(backend_dir, key):
    for name in ("backend-request.json", "worker-request.json", "response.json", "event-0001.json", "ack-0001.json"):
        envelope = exact_fields(parse_json((backend_dir / name).read_text()), {"payload", "signature"})
        if not hmac.compare_digest(envelope["signature"], bridge.authenticate(envelope["payload"], key.encode())):
            raise ValueError("AutoGluon backend authentication failed: " + name)


def _verify_inventory(inventory):
    if (inventory.get("models") != ["ExtraTrees"] or inventory.get("model_type") != "XTModel"
            or inventory.get("n_estimators") != 64 or inventory.get("evidence_kind") != "runtime_instrumented"
            or inventory.get("learner_random_state") != 42 or inventory.get("independent_model_units") != 1
            or inventory.get("version") != "1.4.0" or inventory.get("approval_sha256") != APPROVAL_SHA256
            or canonical_json(inventory.get("resolved_fit_profile")) != canonical_json(backend.resolved_profile()["fit"])):
        raise ValueError("AutoGluon runtime profile/inventory mismatch")
    expected = {**bridge.full_xt_profile()["hyperparameters"]["XT"], **backend._NATIVE_FOREST_DEFAULTS}
    inventory_expected = {k: v for k, v in expected.items() if k not in {"criterion", "oob_score"}}
    if any(inventory.get("effective_model_params", {}).get(k) != v for k, v in inventory_expected.items()):
        raise ValueError("AutoGluon effective native forest profile mismatch")
    for name in ("trainer_attempts", "abstract_fit_attempts"):
        records = inventory.get("model_fit_attempts", {}).get(name)
        if (not isinstance(records, list) or len(records) != 1 or records[0].get("attempt") != 1
                or records[0].get("outcome") != "success" or records[0].get("model_type") != "XTModel"
                or records[0].get("rejected_attempts") != []):
            raise ValueError("AutoGluon exact one model attempt evidence required")
    calls = inventory.get("forest_fit_calls")
    if (not isinstance(calls, list) or [v.get("actual_tree_count") for v in calls] not in ([64], [4, 64])
            or any(v.get("n_estimators") != v.get("actual_tree_count")
                   or v.get("outcome") != "success" or v.get("rejected_attempts") != []
                   or any(canonical_json(v.get(k)) != canonical_json(value)
                          for k, value in expected.items() if k != "n_estimators")
                   for v in calls)
            or (len(calls) == 2 and any(v.get("warm_start") is not True for v in calls))):
        raise ValueError("AutoGluon forest growth evidence mismatch")
    expected_policy = {"max_wall_seconds": None, "max_rss_bytes": None, "wall_limit_enforced": False,
                       "rss_limit_enforced": False, "cpu_limit_enforced": False, "cpu_threads": None,
                       "informed_opt_out": True}
    if canonical_json(inventory.get("resource_policy")) != canonical_json(expected_policy):
        raise ValueError("AutoGluon campaign must remain uncapped")


def _verify_transport_tables(backend_dir, connection, request, snapshot):
    core = snapshot.spec.feature_pool
    train = connection.execute("SELECT " + ",".join('d."' + name + '"' for name in core) +
        ",d.target FROM dev_row d JOIN assign a USING(building_id) WHERE a.repeat=? AND a.fold<>? ORDER BY d.row_index",
        (request["repeat"], request["fold"]))
    predict = connection.execute("SELECT d.row_index," + ",".join('d."' + name + '"' for name in core) +
        " FROM dev_row d JOIN assign a USING(building_id) WHERE a.repeat=? AND a.fold=? ORDER BY d.row_index",
        (request["repeat"], request["fold"]))
    for filename, columns, expected in (("train.parquet", [*core, "anomaly"], train),
                                        ("predict.parquet", ["row_index", *core], predict)):
        for values in _rows(backend_dir / filename, columns):
            if values != expected.fetchone():
                raise ValueError("AutoGluon transport/index feature or label mapping mismatch")
        if expected.fetchone() is not None:
            raise ValueError("AutoGluon incomplete full partition transport")


def verify_autogluon_artifacts(directory: Path, *, expected_request: dict) -> dict:
    """Recompute exact byte/mapping proof for comparison to durable parent authentication.

    This deliberately does not authenticate a worker without its ephemeral key.
    Fresh-process completion MUST compare this result to the parent's signed proof.
    """
    directory = _validate_managed_path(directory, "AutoGluon attempt", directory=True)
    outer = exact_fields(parse_json((directory / "worker-request.json").read_text()), {"payload", "signature"})
    if canonical_json(outer["payload"]) != canonical_json(expected_request):
        raise ValueError("AutoGluon parent request binding mismatch")
    request = outer["payload"]
    spec, snapshot, index = _inputs(directory, request)
    backend_dir = _validate_managed_path(directory / "backend", "AutoGluon backend", root=directory, directory=True)
    transport = parse_json((backend_dir / "backend-request.json").read_text())["payload"]
    child_request = parse_json((backend_dir / "worker-request.json").read_text())["payload"]
    parent_pid = parse_json((backend_dir / "worker-identity.json").read_text())["parent_pid"]
    if canonical_json(child_request) != canonical_json(dict(request, parent_pid=parent_pid)):
        raise ValueError("AutoGluon inner/outer parent request binding mismatch")
    binding = transport["backend_binding"]
    expected_binding = {"version": 1, "worker_sha256": _file_hash(Path(backend.__file__)),
        "bridge_sha256": _file_hash(Path(bridge.__file__)),
        "lifecycle_sha256": _file_hash(backend_dir / "worker-request.json"),
        "approval_sha256": APPROVAL_SHA256, "resolved_profile": backend.resolved_profile()}
    if canonical_json(binding) != canonical_json(expected_binding):
        raise ValueError("AutoGluon executing backend source/profile binding mismatch")
    expected_run = {"run_id": spec.run_id, "attempt_id": request["attempt_id"], "config_hash": spec.binding_sha256,
                   "data_hash": next(s.sha256 for s in spec.sources if s.role == "data"),
                   "split_hash": next(s.sha256 for s in spec.sources if s.role == "split")}
    if transport["binding"] != expected_run:
        raise ValueError("AutoGluon backend parent snapshot binding mismatch")
    # Structural validation plus descriptors: live caller separately verifies HMAC.
    bridge.validate_profile(transport["profile"])
    bridge.validate_environment(transport["environment"])
    if (transport["environment"]["executable"] != str(IMPLEMENTATION_ROOT / ".venv-autogluon/bin/python")
            or transport["environment"]["prefix"] != str(IMPLEMENTATION_ROOT / ".venv-autogluon")):
        raise ValueError("AutoGluon fixed approved interpreter binding mismatch")
    for role in ("train", "predict"):
        if transport[role] != bridge._descriptor(backend_dir, role + ".parquet"):
            raise ValueError("AutoGluon transport descriptor mismatch")
    bridge._validate_tables(backend_dir, transport)
    response = parse_json((backend_dir / "response.json").read_text())["payload"]
    bridge.verify_response(backend_dir, transport, response)
    _verify_inventory(response["backend"])
    status = parse_json((backend_dir / "backend-status.json").read_text())
    started = parse_json((backend_dir / "backend-started.json").read_text())
    event = {"event": "debit", "seq": 1, "phase": "model", "work_id": request["work_id"],
             "attempt_id": request["attempt_id"], "nonce": request["nonce"], "unit": request["work_id"]}
    ack = dict(event, status="authorized")
    if (status != {"status": "awaiting_parent_validation", "request_hash": response["request_hash"]}
            or started != {"status": "running", "debit": ack, "request_hash": response["request_hash"]}
            or parse_json((backend_dir / "event-0001.json").read_text())["payload"] != event
            or parse_json((backend_dir / "ack-0001.json").read_text())["payload"] != ack
            or parse_json((directory / "event-0001.json").read_text())["payload"] != event
            or parse_json((directory / "ack-0001.json").read_text())["payload"] != ack):
        raise ValueError("AutoGluon debit/status/parent binding mismatch")
    supervisor = parse_json((backend_dir / "supervisor.json").read_text())
    if (supervisor["unit"] != request["work_id"] or supervisor["exit_code"] != 0
            or supervisor["status"] != "awaiting_validation" or supervisor["observation"]["cleanup_confirmed"] is not True):
        raise ValueError("AutoGluon backend supervision proof mismatch")
    confirm_owned_attempt_exit(parse_json((backend_dir / "worker-identity.json").read_text()))
    connection, meta = io._open_bound_index(index, spec)
    try:
        _verify_transport_tables(backend_dir, connection, request, snapshot)
        actual_rows = iter(_rows(directory / "shard.parquet"))
        total = 0
        for table in _canonical_batches(index, spec, snapshot, request, backend_dir, connection, meta):
            if pq.ParquetFile(directory / "shard.parquet").schema_arrow != table.schema:
                raise ValueError("AutoGluon canonical shard schema mismatch")
            for row in zip(*(column.to_pylist() for column in table.columns), strict=True):
                if row != next(actual_rows, None):
                    raise ValueError("AutoGluon canonical score mapping substitution")
                total += 1
        if next(actual_rows, None) is not None or total == 0:
            raise ValueError("AutoGluon canonical shard coverage mismatch")
    finally:
        connection.close()
    artifacts = {"backend/" + name: _file_hash(_validate_managed_path(backend_dir / name, "AutoGluon proof", root=backend_dir))
                 for name in BACKEND_ARTIFACTS}
    artifacts["worker-request.json"] = _file_hash(directory / "worker-request.json")
    artifacts["event-0001.json"] = _file_hash(directory / "event-0001.json")
    artifacts["ack-0001.json"] = _file_hash(directory / "ack-0001.json")
    arm = build_reviewed_autogluon_arms(snapshot, year_built_encoding=request["year_built_encoding"])["challenger"]
    return {"version": 1, "work_id": request["work_id"], "attempt_id": request["attempt_id"],
            "nonce": request["nonce"], "spec_sha256": spec.binding_sha256, "index_sha256": request["index_sha256"],
            "model_fit_config_sha256": configuration_digest(_configuration(arm.transformer_or_pipeline)),
            "shard_sha256": _file_hash(directory / "shard.parquet"), "num_rows": total,
            "backend": response["backend"], "environment": transport["environment"], "artifacts": artifacts}


def verify_autogluon_unit(directory: Path, key: str, *, expected_request: dict) -> dict:
    request, _ = _authenticate_request(directory, key)
    if canonical_json(request) != canonical_json(expected_request):
        raise ValueError("AutoGluon expected authenticated parent request mismatch")
    _verify_backend_envelopes(directory / "backend", key)
    envelope = parse_json((directory / "backend" / "backend-request.json").read_text())
    bridge.verify_request(directory / "backend", envelope["payload"], (key.encode(), envelope["signature"]))
    for name in ("event-0001.json", "ack-0001.json"):
        envelope = parse_json((directory / name).read_text())
        if not hmac.compare_digest(envelope["signature"], bridge.authenticate(envelope["payload"], key.encode())):
            raise ValueError("AutoGluon outer debit authentication failed")
    return verify_autogluon_artifacts(directory, expected_request=request)


def run(work_id: str, directory: str | Path) -> None:
    key = os.environ.get("FULL_CAMPAIGN_WORKER_KEY")
    if not key:
        raise ValueError("FULL_CAMPAIGN_WORKER_KEY required")
    directory = Path(directory)
    request, _, _, _ = authenticate_autogluon_request(directory, key)
    if request["work_id"] != work_id:
        raise ValueError("AutoGluon supervisor work identity mismatch")
    execute_autogluon_request(directory, key)
