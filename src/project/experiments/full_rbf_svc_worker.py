"""Dedicated authenticated exact-RBF backend; no campaign readiness promotion."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from project.experiments import full_candidate_io as io
from project.experiments.candidate_contracts import (
    _validate_managed_path,
    validate_partition,
)
from project.experiments.candidate_dispatch import _configuration
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_campaign_worker import (
    ParentLifecycleService,
    _authenticate_request,
    _publish_message,
    prepare_worker_request,
)
from project.experiments.full_input_schema import CORE
from project.experiments.full_rbf_svc_contract import (
    EXPERIMENT_ID,
    build_full_rbf_svc_arms,
    validate_full_rbf_svc_snapshot,
)
from project.experiments.full_run_contracts import _file_hash
from project.experiments.full_snapshot_codec import (
    canonical_json,
    configuration_digest,
    exact_fields,
    parse_json,
    validate_request,
)
from project.features.registry import build_lead_baseline_registry
from project.models.candidate_artifacts import (
    MAX_BATCH_ROWS,
    candidate_oof_schema,
    write_candidate_oof_batches,
)
from project.models.candidate_scores import extract_candidate_score

BACKEND_TARGET = "project.experiments.full_rbf_svc_worker:run"


def _validate_rbf_payload(payload):
    spec = validate_request(payload)
    if payload["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("dedicated RBF experiment required")
    snapshot = next(item for item in spec.candidates if item.spec.experiment_id == EXPERIMENT_ID)
    validate_full_rbf_svc_snapshot(snapshot, year_built_encoding=payload["year_built_encoding"])
    if len(snapshot.planned_units) != 24:
        raise ValueError("RBF exact 24 planned model units required")
    return spec, snapshot


def prepare_rbf_request(directory: Path, payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Parent transport preparation only; unsupported configs create no request.

    Caller still owns authenticated admission, development index preparation and
    durable debit callback. Signed declarations do not promote route readiness.
    """
    spec, _ = _validate_rbf_payload(payload)
    directory = _validate_managed_path(directory, "RBF attempt", directory=True)
    if directory.parent.name != "attempts":
        raise ValueError("run-owned attempts directory required")
    _verify_inputs(directory, payload, spec)
    return prepare_worker_request(directory, payload, key)


def _verify_inputs(directory, payload, spec):
    root = _validate_managed_path(payload["project_root"], "RBF source root", directory=True)
    spec.verify_bindings(root)  # Byte hashes only; never parse prepared/vault/test rows.
    if current_environment() != spec.environment:
        raise ValueError("RBF runtime environment binding mismatch")
    provenance = next(source for source in spec.sources if source.role == "encoding")
    if parse_json(provenance.verify(root).read_text()).get("encoding_id") != payload["year_built_encoding"]:
        raise ValueError("RBF encoding provenance identity mismatch")
    if directory.name != payload["attempt_id"] or directory.parent.name != "attempts":
        raise ValueError("RBF attempt directory identity mismatch")
    index = _validate_managed_path(payload["index_path"], "RBF development index", root=directory.parent.parent)
    if not index.is_file() or _file_hash(index) != payload["index_sha256"]:
        raise ValueError("RBF index hash mismatch")
    for suffix in ("-wal", "-shm", "-journal"):
        if Path(str(index) + suffix).exists():
            raise ValueError("immutable closed RBF index required; live sidecars forbidden")
    connection, meta = io._open_bound_index(index, spec)
    try:
        if Path(meta.data_path) != root / io.DATA_PATH:
            raise ValueError("RBF index declared source confinement mismatch")
        if tuple(meta.predictor_types) != CORE or meta.timestamp_unit not in ("ns", "us"):
            raise ValueError("RBF index predictor/timestamp policy mismatch")
    finally:
        connection.close()
    return index


def authenticate_rbf_request(directory: Path, key: str):
    """Authenticate exact signed request, source/environment/index and parent."""
    directory = _validate_managed_path(directory, "RBF attempt", directory=True)
    payload, spec = _authenticate_request(directory, key)
    spec, snapshot = _validate_rbf_payload(payload)
    if os.getppid() != payload["parent_pid"]:
        raise ValueError("RBF supervisor parent process identity mismatch")
    index = _verify_inputs(directory, payload, spec)
    return payload, spec, snapshot, index


class RBFWorkerLifecycle:
    """Same HMAC event/ack protocol as ParentLifecycleService; no child journal."""

    def __init__(self, directory, key, request, exchange=None):
        self.directory, self.key, self.request = directory, key, request
        self.exchange = exchange if exchange is not None else self._file_exchange
        self.sequence = 0

    def _file_exchange(self, envelope):
        sequence = envelope["payload"]["seq"]
        _publish_message(self.directory / f"event-{sequence:04d}.json", envelope)
        ack = self.directory / f"ack-{sequence:04d}.json"
        deadline = time.monotonic() + self.request["ack_timeout_seconds"]
        while not ack.is_file():
            if os.getppid() != self.request["parent_pid"]:
                raise RuntimeError("RBF supervisor disappeared; reconcile attempt")
            if time.monotonic() >= deadline:
                raise TimeoutError("parent RBF acknowledgement timed out; reconcile attempt")
            time.sleep(.02)
        ack = _validate_managed_path(ack, "RBF acknowledgement", root=self.directory)
        return parse_json(ack.read_text())

    def send(self, event, **evidence):
        expected = "debit" if self.sequence == 0 else "shard_closed" if self.sequence == 1 else None
        if event != expected:
            raise ValueError("RBF lifecycle requires exactly one debit then closed shard")
        self.sequence += 1
        payload = {"event": event, "seq": self.sequence, "phase": "model",
                   **{name: self.request[name] for name in ("work_id", "attempt_id", "nonce")},
                   "unit": self.request["work_id"], **evidence}
        signature = hmac.new(self.key.encode(), canonical_json(payload).encode(), hashlib.sha256).hexdigest()
        response = exact_fields(self.exchange({"payload": payload, "signature": signature}), {"payload", "signature"})
        ack = response["payload"]
        expected_signature = hmac.new(self.key.encode(), canonical_json(ack).encode(), hashlib.sha256).hexdigest()
        expected_ack = {name: payload[name] for name in
                        ("event", "seq", "phase", "work_id", "attempt_id", "nonce", "unit")}
        expected_ack["status"] = "authorized"
        if (type(response["signature"]) is not str
                or not hmac.compare_digest(response["signature"], expected_signature)
                or ack != expected_ack or type(ack.get("seq")) is not int):
            raise ValueError("invalid RBF acknowledgement signature or identity")


def make_parent_rbf_service(directory: Path, key: str, *, on_debit: Callable,
                            on_shard_closed: Callable | None = None) -> ParentLifecycleService:
    """Parent polling hooks; debit callback must durably append before returning.

    This callback protocol cannot prove journal durability by itself. The parent
    integration must authenticate journal/run/attempt bindings and terminal proofs.
    """
    payload, _ = _authenticate_request(directory, key)
    _validate_rbf_payload(payload)
    if not callable(on_debit):
        raise TypeError("durable parent debit callback required")
    return ParentLifecycleService(directory, key, on_debit=on_debit, on_shard_closed=on_shard_closed)


def _verify_training(connection, partition, request):
    if (partition.repeat != request["repeat"] or partition.fold != request["fold"]
            or tuple(partition.X_train.columns) != CORE):
        raise ValueError("RBF training partition identity/schema mismatch")
    validate_partition(partition, build_lead_baseline_registry())
    expected = connection.execute(
        "SELECT d.row_index, d.building_id, d.target FROM dev_row d JOIN assign a USING(building_id)"
        " WHERE a.repeat=? AND a.fold<>? ORDER BY d.row_index", (partition.repeat, partition.fold))
    for index, group, target in zip(partition.X_train.index, partition.groups_train, partition.y_train, strict=True):
        if expected.fetchone() != (int(index), group, int(target)):
            raise ValueError("RBF training coverage/label/group mismatch")
    if expected.fetchone() is not None:
        raise ValueError("RBF incomplete training packet; sampling forbidden")


def _verify_batch(batch, expected, request, timestamp_unit):
    X = batch["X"]
    for name in ("row_index", "target"):
        values = np.asarray(batch[name])
        if values.ndim != 1 or values.dtype.kind not in "iu" or len(values) != len(X):
            raise ValueError("RBF validation row identity and target require exact integer arrays")
    timestamps = np.asarray(batch["timestamp"])
    if timestamps.ndim != 1 or timestamps.dtype.kind != "M" or len(timestamps) != len(X):
        raise ValueError("RBF validation timestamps require exact datetime arrays")
    normalized = timestamps.astype(f"datetime64[{timestamp_unit}]")
    if np.isnat(timestamps).any() or np.any(normalized.astype(timestamps.dtype) != timestamps):
        raise ValueError("RBF timestamp identity must preserve source precision without truncation")
    if (tuple(X.columns) != CORE or not 0 < len(X) <= MAX_BATCH_ROWS
            or tuple(X.index) != tuple(batch["row_index"])
            or type(batch["repeat"]) is not int or batch["repeat"] != request["repeat"]
            or type(batch["fold"]) is not int or batch["fold"] != request["fold"]
            or type(batch["split_seed"]) is not int or batch["split_seed"] != 42 + request["repeat"]):
        raise ValueError("RBF validation coverage/schema/protocol mismatch")
    stamps = normalized.astype(np.int64)
    if timestamp_unit == "us" and (np.any(stamps < -(2**63) // 1000 + 1)
                                    or np.any(stamps > (2**63 - 1) // 1000)):
        raise ValueError("RBF timestamp nanosecond canonicalization would overflow")
    actual = list(zip(batch["row_index"], batch["building_id"], stamps, batch["target"], strict=True))
    if len(actual) != len(X) or actual != expected.fetchmany(len(X)):
        raise ValueError("RBF validation raw identity/target coverage mismatch")


def _table(batch, scores, context):
    request, spec, arm, fit_sha, schema = context
    size = len(scores)
    fixed = {"candidate_id": arm.candidate_id, "recipe_id": EXPERIMENT_ID, "model_family": "svm",
             "model_config_id": arm.candidate_id + "_config", "repeat": request["repeat"], "fold": request["fold"],
             "config_sha256": spec.binding_sha256, "run_id": spec.run_id, "split_seed": 42 + request["repeat"],
             "experiment_id": EXPERIMENT_ID, "arm_id": arm.arm_id, "score_kind": "decision_margin",
             "positive_label": 1, "execution_mode": "full", "non_comparative": False, "n03_eligible": False,
             "model_fit_config_sha256": fit_sha}
    for role in ("data", "split"):
        fixed[role + "_sha256"] = next(source.sha256 for source in spec.sources if source.role == role)
    columns = {name: [value] * size for name, value in fixed.items()}
    stamp_unit = schema.field("timestamp").type.unit
    columns.update(row_index=batch["row_index"], building_id=batch["building_id"],
                   timestamp=np.asarray(batch["timestamp"]).astype(f"datetime64[{stamp_unit}]"),
                   target=batch["target"], prediction=scores)
    return pa.Table.from_pydict(columns, schema=schema)


def execute_rbf_request(directory: Path, key: str, *, exchange: Callable | None = None) -> dict:
    """Execute one exact unit with full training and streamed ranking-only OOF.

    Parent acknowledgement must follow durable model debit BEFORE estimator
    construction. A closed shard is transport evidence, never fit completion.
    No journal is opened in this child. Failures propagate without a success seal.
    """
    directory = _validate_managed_path(directory, "RBF attempt", directory=True)
    request, spec, snapshot, index = authenticate_rbf_request(directory, key)
    if (list(directory.glob("event-*.json")) or list(directory.glob("ack-*.json"))
            or (directory / "shard.parquet").exists()):
        raise ValueError("fresh RBF attempt required; reconcile prior attempt explicitly")
    connection, meta = io._open_bound_index(index, spec)
    try:
        training = io.load_full_training_partition(index, spec, request["repeat"], request["fold"])
        _verify_training(connection, training, request)
        lifecycle = RBFWorkerLifecycle(directory, key, request, exchange)
        lifecycle.send("debit")
        arm = build_full_rbf_svc_arms(snapshot, year_built_encoding=request["year_built_encoding"])[request["arm_id"]]
        pipeline = arm.transformer_or_pipeline
        fit_sha = configuration_digest(_configuration(pipeline))
        pipeline.fit(training.X_train, training.y_train)
        del training
        if configuration_digest(_configuration(pipeline)) != fit_sha:
            raise ValueError("RBF estimator declared configuration changed during fit")
        building_type = pa.int64() if meta.building_dtype == "int64" else pa.string()
        schema = candidate_oof_schema(building_type, pa.timestamp(meta.timestamp_unit, tz=meta.timestamp_time_zone))
        expected = connection.execute(
            "SELECT d.row_index, d.building_id, d.timestamp_ns, d.target FROM dev_row d JOIN assign a USING(building_id)"
            " WHERE a.repeat=? AND a.fold=? ORDER BY d.row_index", (request["repeat"], request["fold"]))
        total = 0
        def batches():
            nonlocal total
            for batch in io.iter_full_validation_batches(index, spec, request["repeat"], request["fold"]):
                _verify_batch(batch, expected, request, meta.timestamp_unit)
                score = extract_candidate_score(arm, batch["X"], "decision_margin")
                total += len(score.values)
                yield _table(batch, score.values, (request, spec, arm, fit_sha, schema))
            if expected.fetchone() is not None or total == 0:
                raise ValueError("RBF incomplete validation coverage")
        output = _validate_managed_path(directory / "shard.parquet", "RBF shard", root=directory)
        write_candidate_oof_batches(output, batches(), schema)
        if hasattr(arm.transformer_or_pipeline, "named_steps"):
            model_step = arm.transformer_or_pipeline.named_steps.get("model")
            if hasattr(model_step, "close"):
                model_step.close()
        _verify_inputs(directory, request, spec)
        with output.open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        digest = _file_hash(output)
        lifecycle.send("shard_closed", sha256=digest, num_rows=total, path=str(output))
        return {"model_fits": 1, "planned_model_fits": 24, "num_rows": total,
                "sha256": digest, "model_fit_config_sha256": fit_sha, "status": "awaiting_validation"}
    finally:
        connection.close()


def run(work_id: str, directory: str | Path) -> None:
    """Generic supervisor target; FULL_CAMPAIGN_WORKER_KEY is per-attempt secret."""
    key = os.environ.get("FULL_CAMPAIGN_WORKER_KEY")
    if not key:
        raise ValueError("FULL_CAMPAIGN_WORKER_KEY required")
    directory = Path(directory)
    request, _, _, _ = authenticate_rbf_request(directory, key)
    if request["work_id"] != work_id:
        raise ValueError("RBF supervisor work identity mismatch")
    execute_rbf_request(directory, key)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dedicated authenticated exact RBF SVC backend")
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--directory", required=True)
    arguments = parser.parse_args()
    run(arguments.work_id, arguments.directory)
