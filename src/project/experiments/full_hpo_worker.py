"""Dedicated full HPO backend; ordinary workers deliberately refuse this route."""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from project.experiments import full_candidate_io as io
from project.experiments.candidate_dispatch import CandidateArm
from project.experiments.full_campaign_worker import _build_batch_table, _oof_schema, _write_oof_shard
from project.models.candidate_scores import extract_candidate_score
from project.experiments.candidate_contracts import (
    TrainingPartition,
    _validate_managed_path,
    validate_partition,
)
from project.experiments.full_development_readiness import verify_target_encoding_partition
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_hpo_lifecycle import (
    EXPERIMENT,
    HpoLifecyclePlan,
    ParentHpoLifecycle,
    WorkerHpoLifecycle,
    _authenticate,
    sign_message,
)
from project.experiments.full_nested_tuning import (
    MAX_BATCH_ROWS,
    FullTuningAuthority,
    _verify_fold_evidence,
    _verify_refit_evidence,
    compute_fit_config_sha256,
    full_hpo_study_spec,
    tune_full_outer_partition,
)
from project.experiments.full_run_contracts import FullFitJournal, _file_hash
from project.experiments.full_snapshot_codec import (
    PROVENANCE_PATH,
    canonical_json,
    decode_snapshot,
    encode_snapshot,
    exact_fields,
    parse_json,
    semantic_digest,
)
from project.experiments.inner_cv import build_inner_folds
from project.experiments.nested_tuning import build_hgb_tuning_arm
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1
from project.features.registry import build_lead_baseline_registry

REQUEST_FIELDS = {
    "version", "kind", "session_id", "work_id", "nonce", "spec", "spec_sha256",
    "study_spec", "source_root", "run_root", "repeat", "fold", "environment_sha256",
    "encoding", "provenance_sha256", "semantic_sha256", "training_path", "training_sha256",
    "training_schema_sha256", "training_rows", "index_name", "ack_timeout_seconds", "parent_pid",
    "index_path", "index_sha256", "index_integrity_sha256", "audited_membership_sha256",
}
AUXILIARY = ("__hpo_row_index", "__hpo_target", "__hpo_group")
MAX_MESSAGE_BYTES = 131072


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Linux atomic no-replace publication, with no transient second hardlink.

    Full journals already require Linux flock. Fail closed on systems without
    renameat2 rather than falling back to an overwrite or link/unlink window.
    """
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as error:
        raise RuntimeError("Linux renameat2 required for exclusive HPO publication") from error
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _publish(path: Path, value: dict[str, Any]) -> None:
    _validate_managed_path(path, "HPO transport file", root=path.parent)
    body = canonical_json(value)
    if len(body.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("bounded HPO metadata message required; data belongs in batch files")
    temporary = path.parent / (".hpo-message-" + secrets.token_hex(16))
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        _rename_exclusive(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_message(path: Path) -> dict[str, Any]:
    _validate_managed_path(path, "HPO message", root=path.parent)
    with path.open("rb") as stream:
        body = stream.read(MAX_MESSAGE_BYTES + 1)
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError("HPO metadata message too large")
    return parse_json(body.decode("utf-8"))


def _schema_hash(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _validate_hpo_index(path: str | Path, journal_root: str | Path) -> Path:
    root = _validate_managed_path(journal_root, "HPO journal root", directory=True)
    index = _validate_managed_path(path, "HPO development index")
    if index.is_relative_to(root):
        return index
    if root.name == "journal" and index == root.parent / "dev_index.sqlite3":
        return index
    raise ValueError("HPO development index must remain beneath its managed root")


def prepare_hpo_request(directory: Path, journal: FullFitJournal, training: TrainingPartition,
                        key: str, *, ack_timeout_seconds: float = 120.0,
                        index_path: Path | None = None) -> dict[str, Any]:
    """Parent writes one confined training-only batch file and signed fresh session.

    The caller must derive training from its bound development index; this helper
    is serialization, not admission or a substitute for authoritative membership.
    """
    plan = HpoLifecyclePlan(journal.spec, training.repeat, training.fold, "preparation", "preparation")
    validate_partition(training, build_lead_baseline_registry())
    destination = _validate_managed_path(directory, "HPO session", root=journal.root, directory=True)
    if destination != journal.root / "hpo_sessions" / directory.name or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
        raise ValueError("exclusive supervisor session directory required")
    if destination.exists():
        raise ValueError("fresh HPO session required")
    if current_environment() != journal.spec.environment:
        raise ValueError("exact HPO environment binding required")
    provenance = next((source for source in journal.spec.sources if source.role == "encoding"), None)
    if provenance is None or provenance.path != PROVENANCE_PATH:
        raise ValueError("immutable encoding provenance required")
    provenance_value = parse_json(provenance.verify(journal.source_root).read_text())
    if provenance_value.get("encoding_id") != LEAD_YEAR_OFFSET_1900_UINT8_V1:
        raise ValueError("encoding provenance identity mismatch")
    index_integrity = None
    audited_membership = None
    if index_path is not None:
        index_path = _validate_hpo_index(index_path, journal.root)
        index_integrity = io.verify_development_index(index_path, journal.spec)
        _require_training_equal(training, io.load_full_training_partition(
            index_path, journal.spec, training.repeat, training.fold))
        audited_membership = verify_target_encoding_partition(journal.source_root, journal.spec, training)
    destination.mkdir(parents=True)
    frame = training.X_train.copy(deep=False)
    frame = frame.assign(**{AUXILIARY[0]: training.X_train.index,
                            AUXILIARY[1]: training.y_train, AUXILIARY[2]: training.groups_train})
    schema = pa.Table.from_pandas(frame.iloc[:0], preserve_index=False).schema
    path = destination / "training.parquet"
    with path.open("xb") as stream, pq.ParquetWriter(stream, schema, compression="snappy") as writer:
        for start in range(0, len(frame), MAX_BATCH_ROWS):
            writer.write_table(pa.Table.from_pandas(frame.iloc[start:start + MAX_BATCH_ROWS], schema=schema, preserve_index=False))
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    payload = {
        "version": 1, "kind": "full_hpo", "session_id": destination.name,
        "work_id": f"{EXPERIMENT}/r{plan.repeat}/f{plan.fold}/hpo_session",
        "nonce": secrets.token_hex(32), "spec": encode_snapshot(journal.spec),
        "spec_sha256": journal.spec.binding_sha256, "study_spec": asdict(full_hpo_study_spec()),
        "source_root": str(journal.source_root), "run_root": str(journal.root),
        "repeat": plan.repeat, "fold": plan.fold,
        "environment_sha256": hashlib.sha256(canonical_json(journal.spec.environment).encode()).hexdigest(),
        "encoding": LEAD_YEAR_OFFSET_1900_UINT8_V1, "provenance_sha256": provenance.sha256,
        "semantic_sha256": semantic_digest(LEAD_YEAR_OFFSET_1900_UINT8_V1, provenance.sha256),
        "training_path": "training.parquet", "training_sha256": _file_hash(path),
        "training_schema_sha256": _schema_hash(schema), "training_rows": len(frame),
        "index_name": training.X_train.index.name, "ack_timeout_seconds": ack_timeout_seconds,
        "parent_pid": os.getpid(),
        "index_path": str(index_path) if index_path is not None else None,
        "index_sha256": _file_hash(index_path) if index_path is not None else None,
        "index_integrity_sha256": index_integrity,
        "audited_membership_sha256": audited_membership,
    }
    payload = parse_json(canonical_json(payload))
    _publish(destination / "hpo-request.json", sign_message(payload, key))
    authenticate_hpo_request(destination, key)
    return payload


def authenticate_hpo_request(directory: Path, key: str) -> tuple[dict[str, Any], HpoLifecyclePlan]:
    directory = _validate_managed_path(directory, "HPO session", directory=True)
    request_file = _validate_managed_path(directory / "hpo-request.json", "HPO request", root=directory)
    value = exact_fields(_authenticate(_read_message(request_file), key), REQUEST_FIELDS)
    if type(value["version"]) is not int or value["version"] != 1 or value["kind"] != "full_hpo":
        raise ValueError("unsupported dedicated HPO request version/kind")
    spec = decode_snapshot(value["spec"])
    if value["spec_sha256"] != spec.binding_sha256:
        raise ValueError("HPO request spec mismatch")
    if canonical_json(value["study_spec"]) != canonical_json(asdict(full_hpo_study_spec())):
        raise ValueError("exact approved StudySpec required")
    if (type(value["nonce"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["nonce"])
            or type(value["session_id"]) is not str or not re.fullmatch(r"[0-9a-f]{32}", value["session_id"])):
        raise ValueError("fresh session/nonce required")
    for name in ("run_root", "source_root"):
        if type(value[name]) is not str or not Path(value[name]).is_absolute():
            raise ValueError("explicit absolute HPO authority paths required")
        _validate_managed_path(value[name], name, directory=True)
    if directory != Path(value["run_root"]) / "hpo_sessions" / value["session_id"]:
        raise ValueError("supervisor session identity mismatch")
    plan = HpoLifecyclePlan(spec, value["repeat"], value["fold"], value["work_id"], value["nonce"],
                            hashlib.sha256(canonical_json(value).encode()).hexdigest())
    if value["work_id"] != f"{EXPERIMENT}/r{plan.repeat}/f{plan.fold}/hpo_session":
        raise ValueError("exact supervisor HPO session identity required")
    spec.verify_bindings(value["source_root"])
    if (spec.environment != current_environment()
            or value["environment_sha256"] != hashlib.sha256(canonical_json(spec.environment).encode()).hexdigest()):
        raise ValueError("HPO runtime environment mismatch")
    bindings = [source for source in spec.sources if source.role == "encoding"]
    if (len(bindings) != 1 or bindings[0].path != PROVENANCE_PATH
            or value["provenance_sha256"] != bindings[0].sha256
            or value["encoding"] != LEAD_YEAR_OFFSET_1900_UINT8_V1
            or value["semantic_sha256"] != semantic_digest(value["encoding"], bindings[0].sha256)):
        raise ValueError("HPO encoding/provenance binding mismatch")
    if parse_json(bindings[0].verify(Path(value["source_root"])).read_text()).get("encoding_id") != value["encoding"]:
        raise ValueError("HPO encoding identity mismatch")
    if (type(value["training_rows"]) is not int or value["training_rows"] <= 0
            or type(value["parent_pid"]) is not int or value["parent_pid"] <= 0
            or type(value["ack_timeout_seconds"]) is not float or not 0 < value["ack_timeout_seconds"] < float("inf")
            or value["training_path"] != "training.parquet"
            or (value["index_name"] is not None and type(value["index_name"]) is not str)):
        raise ValueError("invalid HPO training/session metadata")
    training_path = _validate_managed_path(directory / value["training_path"], "HPO training", root=directory)
    if _file_hash(training_path) != value["training_sha256"]:
        raise ValueError("HPO training file hash mismatch")
    parquet = pq.ParquetFile(training_path)
    if _schema_hash(parquet.schema_arrow) != value["training_schema_sha256"] or parquet.metadata.num_rows != value["training_rows"]:
        raise ValueError("HPO training schema/row hash mismatch")
    _verify_bound_index(directory, value, spec)
    return value, plan


def load_hpo_training(directory: Path, request: dict[str, Any]) -> TrainingPartition:
    """Materialize one outer training frame from bounded Parquet batches, not row lists."""
    path = _validate_managed_path(directory / request["training_path"], "HPO training", root=directory)
    if _file_hash(path) != request["training_sha256"]:
        raise ValueError("HPO training changed")
    parquet = pq.ParquetFile(path)
    if _schema_hash(parquet.schema_arrow) != request["training_schema_sha256"]:
        raise ValueError("HPO training schema changed")
    frame = pd.concat([batch.to_pandas() for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS)], ignore_index=True)
    if len(frame) != request["training_rows"]:
        raise ValueError("HPO training row count changed")
    index = pd.Index(frame.pop(AUXILIARY[0]), name=request["index_name"])
    target = frame.pop(AUXILIARY[1]).set_axis(index).rename("anomaly")
    groups = frame.pop(AUXILIARY[2]).set_axis(index).rename("building_id")
    training = TrainingPartition(request["repeat"], request["fold"], frame.set_axis(index), target, groups)
    validate_partition(training, build_lead_baseline_registry())
    return training


def _require_training_equal(actual: TrainingPartition, expected: TrainingPartition) -> None:
    for name in ("X_train", "y_train", "groups_train"):
        if not getattr(actual, name).equals(getattr(expected, name)):
            raise ValueError("HPO training differs from bound development index")


def _verify_bound_index(directory: Path, request: dict[str, Any], spec) -> None:
    if request["index_path"] is None:
        if any(request[name] is not None for name in
               ("index_sha256", "index_integrity_sha256", "audited_membership_sha256")):
            raise ValueError("partial HPO index authority")
        return
    index = _validate_hpo_index(request["index_path"], request["run_root"])
    if _file_hash(index) != request["index_sha256"]:
        raise ValueError("HPO index file hash mismatch")
    io.verify_development_index(index, spec, expected_integrity_sha256=request["index_integrity_sha256"])
    training = io.load_full_training_partition(index, spec, request["repeat"], request["fold"])
    _require_training_equal(load_hpo_training(directory, request), training)
    if verify_target_encoding_partition(request["source_root"], spec, training) != request["audited_membership_sha256"]:
        raise ValueError("HPO audited K3 membership mismatch")


class HpoFileParentService:
    """Poll-only supervisor bridge. Owns a session, not a placeholder fit attempt."""

    def __init__(self, directory: Path, lifecycle: ParentHpoLifecycle) -> None:
        self.directory, self.lifecycle = directory, lifecycle
        self._serviced: dict[str, str] = {}

    def service_once(self) -> None:
        for path in sorted(self.directory.glob("hpo-event-*.json")):
            _validate_managed_path(path, "HPO event", root=self.directory)
            digest = _file_hash(path)
            if path.name in self._serviced:
                if digest != self._serviced[path.name]:
                    raise ValueError("serviced HPO event changed")
                continue
            envelope = _read_message(path)
            payload = _authenticate(envelope, self.lifecycle.key)
            seq = payload.get("seq")
            if type(seq) is not int or path.name != f"hpo-event-{seq:04d}.json":
                raise ValueError("HPO event filename/sequence identity mismatch")
            ack_path = self.directory / f"hpo-ack-{seq:04d}.json"
            if ack_path.exists():
                raise ValueError("HPO acknowledgement replay; fresh supervisor session required")
            ack = self.lifecycle.handle(envelope)
            _publish(ack_path, ack)
            self._serviced[path.name] = digest


def make_parent_hpo_service(directory: Path, journal: FullFitJournal, key: str, *,
                            retry_reasons: Mapping[str, str] | None = None) -> HpoFileParentService:
    """Borrow the existing journal and independently verify each actual seal.

    Parent must call prepare_hpo_request with its authoritative index-derived
    training. Worker messages never supply authoritative labels or membership.
    """
    request, plan = authenticate_hpo_request(directory, key)
    if (any(directory.glob("hpo-event-*.json")) or any(directory.glob("hpo-ack-*.json"))
            or (directory / "hpo-result.json").exists()):
        raise ValueError("fresh HPO supervisor session required; reconcile prior owner explicitly")
    if (journal.spec.binding_sha256 != plan.spec.binding_sha256 or journal.root != Path(request["run_root"])
            or journal.source_root != Path(request["source_root"])):
        raise ValueError("parent HPO journal authority mismatch")
    training = load_hpo_training(directory, request)
    folds = build_inner_folds(training.y_train, training.groups_train, n_splits=3, seed=42)

    def verify(identity, record, evidence):
        actual = next((item for item in journal.attempts() if item.attempt_id == record.attempt_id), None)
        if actual != record or actual.evidence != evidence or actual.status not in ("awaiting_validation", "completed"):
            raise ValueError("actual parent journal seal required")
        if _file_hash(directory / request["training_path"]) != request["training_sha256"]:
            raise ValueError("authoritative HPO training file changed")
        if identity.phase == "inner":
            return _verify_fold_evidence(journal.root, actual, folds[identity.inner_fold][1],
                                         training.y_train.to_numpy(), training)
        _verify_refit_evidence(journal.root, actual, identity.phase, dict(identity.parameters), len(training.X_train))
        return None

    return HpoFileParentService(directory, ParentHpoLifecycle(
        journal, plan, key, verify_completion=verify, retry_reasons=retry_reasons))


class HpoFileExchange:
    """Exclusive bounded JSON messages; the data plane is streamed Parquet."""

    def __init__(self, directory: Path, timeout_seconds: float) -> None:
        self.directory = directory
        self.timeout_seconds = timeout_seconds

    def __call__(self, envelope: dict[str, Any]) -> dict[str, Any]:
        seq = envelope["payload"]["seq"]
        _publish(self.directory / f"hpo-event-{seq:04d}.json", envelope)
        acknowledgement = self.directory / f"hpo-ack-{seq:04d}.json"
        deadline = time.monotonic() + self.timeout_seconds
        while not acknowledgement.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("parent HPO acknowledgement timed out; reconcile owned attempt")
            time.sleep(0.02)
        _validate_managed_path(acknowledgement, "HPO acknowledgement", root=self.directory)
        return _read_message(acknowledgement)


def execute_hpo_request(directory: Path, key: str,
                        exchange: Callable[[dict[str, Any]], dict[str, Any]] | None = None, *,
                        arm_factory: Callable = build_hgb_tuning_arm):
    """Mechanically run the dedicated backend without a journal in this process.

    Reused completed refits deliberately return no model; producing outer OOF must
    happen while freshly authorized models exist, not by silently fitting again.
    """
    request, plan = authenticate_hpo_request(directory, key)
    training = load_hpo_training(directory, request)
    lifecycle = WorkerHpoLifecycle(plan, key, exchange if exchange is not None else HpoFileExchange(directory, request["ack_timeout_seconds"]))
    authority = FullTuningAuthority(Path(request["run_root"]), Path(request["source_root"]), plan.spec)
    return tune_full_outer_partition(training, full_hpo_study_spec(), None,
                                     authority.root / "hpo_studies" / f"r{plan.repeat}_f{plan.fold}",
                                     arm_factory=arm_factory, lifecycle=lifecycle, authority=authority)


def run(work_id: str, directory: str | Path) -> None:
    """CLI backend entrypoint; parent later supplies supervision and promotion."""
    directory = Path(directory)
    key = os.environ.get("FULL_HPO_WORKER_KEY")
    if not key:
        raise ValueError("FULL_HPO_WORKER_KEY required")
    request, plan = authenticate_hpo_request(directory, key)
    if work_id != plan.work_id or os.getppid() != request["parent_pid"]:
        raise ValueError("dedicated HPO supervisor work/process identity mismatch")
    result = execute_hpo_request(directory, key)
    shards = write_hpo_outer_shards(directory, request, result)
    summary = {"version": 1, "kind": "hpo_result", "work_id": plan.work_id,
               "session_id": request["session_id"], "nonce": plan.nonce,
               "spec_sha256": plan.spec.binding_sha256, "request_sha256": plan.request_sha256,
               "selected_trial_id": result.selected_trial_id, "selected_params": result.selected_params,
               "study_manifest_hash": result.study_manifest_hash,
               "inner_membership_digest": result.inner_membership_digest,
               "study_db_path": result.study_db_path, "shards": shards}
    _publish(directory / "hpo-result.json", sign_message(summary, key))


def write_hpo_outer_shards(directory: Path, request: dict[str, Any], result) -> dict[str, Any]:
    """Score the two existing outer refits; never reconstruct or refit models."""
    if request["index_path"] is None:
        raise ValueError("bound development index required for outer HPO OOF")
    if result.default_model is None or result.selected_model is None:
        raise ValueError("HPO resume requires verified existing partition proof; fresh models unavailable")
    spec = decode_snapshot(request["spec"])
    _verify_bound_index(directory, request, spec)
    connection, meta = io._open_bound_index(request["index_path"], spec)
    connection.close()
    schema = _oof_schema(meta)
    candidate = next(item.spec for item in spec.candidates if item.spec.experiment_id == EXPERIMENT)
    from project.experiments.nested_tuning import DEFAULT_HGB_PARAMS
    shards = {}
    for arm_id, model, parameters in (
            ("control", result.default_model, DEFAULT_HGB_PARAMS),
            ("challenger", result.selected_model, result.selected_params)):
        fit_hash = compute_fit_config_sha256(EXPERIMENT, arm_id, parameters)
        arm = CandidateArm(f"{EXPERIMENT}_{arm_id}", arm_id, candidate.feature_pool,
                           model, build_lead_baseline_registry(), "positive_probability")
        def batches(model=model, arm=arm, fit_hash=fit_hash):
            for batch in io.iter_full_validation_batches(
                    request["index_path"], spec, request["repeat"], request["fold"]):
                score = extract_candidate_score(model, batch["X"], "positive_probability")
                yield _build_batch_table(batch, score.values, candidate, arm, spec,
                                         request["repeat"], request["fold"], fit_hash, schema)
        path = directory / f"outer-{arm_id}.parquet"
        digest, rows = _write_oof_shard(path, batches(), schema)
        shards[arm_id] = {"path": path.name, "sha256": digest, "rows": rows,
                          "fit_config_sha256": fit_hash}
    return shards


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dedicated authenticated full HPO backend")
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--directory", required=True)
    arguments = parser.parse_args()
    run(arguments.work_id, arguments.directory)
