"""Strict full-campaign evidence persistence and validation verification (T5).

Unlike diagnostic evidence, full campaign evidence:
- Rejects nonfinite floats (NaN, Inf) and arbitrary unhandled types.
- Forbids overwriting existing manifests or run statuses.
- Strictly forbids emitting 'completed' status without a verified OOF validation seal.
- Persists exact sha256 checksums and per-arm candidate metadata.
"""

from __future__ import annotations

import ctypes
import datetime
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from project.experiments.candidate_contracts import _validate_managed_path
from project.experiments.full_run_contracts import (
    AdmittedFullRun,
    FullFitJournal,
    FullRunSpec,
    OrdinaryCompletionAdmission,
    _file_hash,
    _json,
)


def strict_json_safe(obj: Any) -> Any:
    """Strictly convert primitives to JSON-safe data; reject nonfinites and unknown objects."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"nonfinite float not allowed in strict full evidence: {obj}")
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"nonfinite float not allowed in strict full evidence: {val}")
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [strict_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): strict_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [strict_json_safe(x) for x in obj]
    raise TypeError(f"unsupported type for strict full evidence JSON: {type(obj)}")


def strict_json_dumps(obj: Any, indent: int = 2) -> str:
    """Serialize object with strict type checking and nonfinite rejection."""
    safe = strict_json_safe(obj)
    return json.dumps(safe, indent=indent, sort_keys=True)


def write_full_evidence(
    spec: FullRunSpec,
    destination: str | Path,
    status: str,
    artifacts: dict[str, Any],
    validation_payload: dict[str, Any] | None = None,
    *,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Persist immutable full campaign evidence, strict manifest, and status."""
    if type(spec) is not FullRunSpec:
        raise TypeError("exact FullRunSpec required")
    if allow_overwrite:
        raise ValueError("immutable evidence forbids overwrite")
    dest_path = _validate_managed_path(destination, "evidence destination", directory=True)

    manifest_path = dest_path / "manifest.json"
    status_path = dest_path / "run_status.json"

    if manifest_path.exists() or status_path.exists():
        raise FileExistsError("immutable evidence exists; overwrite rejected")

    if status == "completed":
        raise ValueError("comparative completion requires independent validation, actual hashes, arm_metadata, journal and supervisor proof; caller seal is insufficient")
    if status not in ("awaiting_validation", "failed", "interrupted", "cancelled", "finished_with_blockers"):
        raise ValueError("unsupported evidence status")

    manifest_data = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "execution_mode": "full",
        "spec_binding_sha256": spec.binding_sha256,
        "candidates": {
            s.spec.experiment_id: {
                "experiment_id": s.spec.experiment_id,
                "model_family": s.spec.model_family,
                "arms": list(s.arms),
                "source_notebook": s.source_notebook,
                "source_cell_id": s.source_cell_id,
            }
            for s in spec.candidates
        },
        "artifacts": artifacts,
    }
    if validation_payload:
        manifest_data["validation"] = validation_payload

    manifest_text = strict_json_dumps(manifest_data)

    status_data = {
        "run_id": spec.run_id,
        "status": status,
        "execution_mode": "full",
        "spec_binding_sha256": spec.binding_sha256,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "artifacts": artifacts,
    }
    if validation_payload:
        status_data["validation_seal"] = validation_payload.get("seal") or validation_payload.get("validation_sha256")
        status_data["metrics"] = validation_payload.get("metrics") or validation_payload.get("candidate_metrics")

    status_text = strict_json_dumps(status_data)
    dest_path.mkdir(parents=True, exist_ok=False)
    for path, text in ((manifest_path, manifest_text), (status_path, status_text)):
        with path.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    descriptor = os.open(dest_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "status_path": str(status_path),
    }


def _read_evidence_json(path: Path, root: Path) -> Any:
    from project.experiments.full_snapshot_codec import parse_json

    managed = _validate_managed_path(path, "ordinary receipt evidence", root=root)
    return parse_json(managed.read_text())


def sign_parent_record(source: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Same trusted-parent threat model as completion; never a worker HMAC."""
    if any(os.environ.get(name) for name in ("FULL_CAMPAIGN_WORKER_KEY", "FULL_HPO_WORKER_KEY")):
        raise PermissionError("worker cannot issue durable parent records")
    key = _parent_authority_key(source, create=True)
    return {"payload": payload, "signature": hmac.new(
        key, ("ordinary_parent_record_v1:" + _json(payload)).encode(), hashlib.sha256).hexdigest()}


def verify_parent_record(source: Path, path: Path, root: Path) -> dict[str, Any]:
    from project.experiments.full_snapshot_codec import exact_fields

    envelope = exact_fields(_read_evidence_json(path, root), {"payload", "signature"})
    expected = hmac.new(_parent_authority_key(source),
                        ("ordinary_parent_record_v1:" + _json(envelope["payload"])).encode(), hashlib.sha256).hexdigest()
    if type(envelope["signature"]) is not str or not hmac.compare_digest(envelope["signature"], expected):
        raise ValueError("forged or changed durable parent record")
    return envelope["payload"]


def resume_binding(admitted: AdmittedFullRun, root: Path) -> dict[str, Any]:
    return {"kind": "ordinary_run_binding", "source_root": admitted.source_root, "run_root": str(root),
            "spec_sha256": admitted.spec.binding_sha256,
            "implementation_sources": [asdict(item) for item in admitted.implementation_sources],
            "actual_environment": dict(admitted.actual_environment),
            "resource_opt_out_record": admitted.resource_opt_out_record}


def _training_state(preprocessor, partition) -> dict:
    import pandas as pd

    transformed = preprocessor.transform(partition.X_train)
    def digest(frame):
        return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
    return {"names": list(preprocessor.get_feature_names_out()), "rows": len(partition.X_train),
            "matrix_sha256": digest(transformed), "labels_sha256": digest(partition.y_train),
            "groups_sha256": digest(partition.groups_train)}


def selector_outcome(pipeline, partition, request: dict) -> dict:
    """Worker's actual fitted L1 outcome; never inferred from planned unit names."""
    selector = pipeline.named_steps["selector"]
    estimator = selector.estimator_
    from project.experiments.candidate_dispatch import _configuration
    from project.experiments.full_snapshot_codec import configuration_digest
    return {"kind": "fitted_selector_outcome", "spec_sha256": request["spec_sha256"],
            "config_sha256": configuration_digest(_configuration(pipeline)),
            "nonce": request["nonce"], "attempt_id": request["attempt_id"],
            "unit": request["work_id"].rsplit("/", 1)[0] + "/selector",
            "training": _training_state(pipeline.named_steps["preprocessor"], partition),
            "support": selector.get_support().tolist(), "coef": estimator.coef_.tolist(),
            "intercept": estimator.intercept_.tolist(), "classes": estimator.classes_.tolist(),
            "n_iter": estimator.n_iter_.tolist(), "threshold": float(selector.threshold_)}


def verify_selector_outcome(path: Path, authority: Path, spec: FullRunSpec, request: dict) -> dict:
    """Rebuild train-only preprocessing from source; validate actual coefficient mask.

    No selector or model is refitted here. The fixed authenticated worker supplies
    fitted coefficients; independent source reconstruction proves names, training
    population/labels, preprocessing and mask consistency, not optimizer replay.
    """
    from sklearn.base import clone

    from project.experiments import full_candidate_io as io
    from project.experiments.full_campaign_capabilities import (
        build_reviewed_ordinary_arms,
    )
    from project.experiments.full_snapshot_codec import exact_fields

    outcome = exact_fields(_read_evidence_json(path, path.parent), {
        "kind", "spec_sha256", "nonce", "attempt_id", "unit", "training", "support",
        "coef", "intercept", "classes", "n_iter", "threshold", "config_sha256"})
    identity = request["experiment_id"]
    snapshot = next(s for s in spec.candidates if s.spec.experiment_id == identity)
    if identity != "dim_fold_local_selector_vs_passthrough" or request["arm_id"] != "challenger":
        raise ValueError("selector outcome requires exact reviewed selector challenger")
    if any(outcome[key] != value for key, value in {
        "kind": "fitted_selector_outcome", "spec_sha256": spec.binding_sha256,
        "nonce": request["nonce"], "attempt_id": request["attempt_id"],
        "unit": request["work_id"].rsplit("/", 1)[0] + "/selector"}.items()):
        raise ValueError("selector outcome identity/spec/nonce mismatch")
    arms = build_reviewed_ordinary_arms(snapshot, year_built_encoding=request["year_built_encoding"])
    from project.experiments.candidate_dispatch import _configuration
    from project.experiments.full_snapshot_codec import configuration_digest
    if outcome["config_sha256"] != configuration_digest(_configuration(arms["challenger"].transformer_or_pipeline)):
        raise ValueError("selector fitted recipe configuration mismatch")
    partition = io.load_full_training_partition(authority, spec, request["repeat"], request["fold"])
    pre = clone(arms["challenger"].named_steps["preprocessor"])
    pre.fit(partition.X_train, partition.y_train)
    expected = _training_state(pre, partition)
    if outcome["training"] != expected:
        raise ValueError("selector source-derived training/preprocessing mismatch")
    coefficients = np.asarray(outcome["coef"], dtype=float)
    intercept = np.asarray(outcome["intercept"], dtype=float)
    support = outcome["support"]
    if (coefficients.shape != (1, len(expected["names"])) or intercept.shape != (1,)
            or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all()
            or outcome["classes"] != [0, 1] or outcome["threshold"] != 1e-5
            or not isinstance(outcome["n_iter"], list) or len(outcome["n_iter"]) != 1
            or type(outcome["n_iter"][0]) is not int or not 1 <= outcome["n_iter"][0] <= 1000
            or type(support) is not list or any(type(v) is not bool for v in support)
            or support != (np.abs(coefficients[0]) >= 1e-5).tolist() or not any(support)):
        raise ValueError("invalid fitted selector coefficient/support evidence")
    return outcome


def verify_completed_selector(source: Path, root: Path, spec: FullRunSpec, attempt,
                              authority: Path, index_sha: str) -> None:
    from project.experiments.full_snapshot_codec import validate_request

    directory = root / "journal" / attempt.artifact_directory
    names = {"selector-outcome.json", "worker-request.json", "worker-launch.json", "parent-auth.json"}
    files = {Path(item.path).name: item.sha256 for item in attempt.evidence}
    if attempt.status != "completed" or len(attempt.evidence) != len(names) or set(files) != names:
        raise ValueError("exact fitted selector journal evidence required")
    for name, digest in files.items():
        if _file_hash(_validate_managed_path(directory / name, "selector evidence", root=root)) != digest:
            raise ValueError("selector evidence hash mismatch")
    auth = verify_parent_record(source, directory / "parent-auth.json", root)
    if auth != {"kind": "completed_selector", "run_root": str(root), "spec_sha256": spec.binding_sha256,
                "journal_attempt_id": attempt.attempt_id, "unit": attempt.unit,
                "artifacts": {name: digest for name, digest in files.items() if name != "parent-auth.json"}}:
        raise ValueError("selector parent authentication mismatch")
    request = _read_evidence_json(directory / "worker-request.json", root)["payload"]
    if (validate_request(request) != spec or request["index_sha256"] != index_sha
            or request["project_root"] != str(source)
            or request["work_id"].rsplit("/", 1)[0] + "/selector" != attempt.unit):
        raise ValueError("selector request/config lineage mismatch")
    launch = verify_parent_record(source, directory / "worker-launch.json", root)
    if (launch.get("journal_attempt_id") != attempt.attempt_id or launch.get("unit") != request["work_id"]
            or launch.get("worker_request_sha256") != files["worker-request.json"]
            or launch["identity_record"]["binding"] != {"nonce": request["nonce"], "attempt_id": request["attempt_id"],
                                                       "spec_sha256": spec.binding_sha256}):
        raise ValueError("selector launch identity mismatch")
    verify_selector_outcome(directory / "selector-outcome.json", authority, spec, request)


def verify_completed_unit(source: Path, root: Path, spec: FullRunSpec, attempt: Any,
                          declaration: dict, authority: Path, index_sha: str) -> Path:
    """Authenticate partial parent evidence and independently revalidate a real shard."""
    from project.experiments.full_candidate_runner import _validate_shard
    from project.experiments.full_snapshot_codec import exact_fields, validate_request

    directory = root / "journal" / attempt.artifact_directory
    files = {Path(item.path).name: item.sha256 for item in attempt.evidence}
    names = {"shard.parquet", "parent_validation.json", "supervisor.json", "worker-request.json",
             "worker-launch.json", "parent-auth.json"}
    experiment, arm = attempt.unit.split("/")[:2]
    snapshot = next(s for s in spec.candidates if s.spec.experiment_id == experiment)
    selector_model = arm in snapshot.selector_arms
    if selector_model:
        names.add("selector-link.json")
    if experiment == "model_autogluon_core_features":
        names.add("autogluon-proof.json")
    if set(files) != names or len(attempt.evidence) != len(names) or attempt.status != "completed":
        raise ValueError("exact authenticated completed unit evidence required")
    for name, digest in files.items():
        if _file_hash(_validate_managed_path(directory / name, "completed unit evidence", root=root)) != digest:
            raise ValueError("completed unit evidence hash mismatch")
    auth = verify_parent_record(source, directory / "parent-auth.json", root)
    if auth != {"kind": "completed_unit", "run_root": str(root), "spec_sha256": spec.binding_sha256,
                "journal_attempt_id": attempt.attempt_id, "unit": attempt.unit,
                "artifacts": {name: digest for name, digest in files.items() if name != "parent-auth.json"}}:
        raise ValueError("completed unit parent authentication mismatch")
    request = exact_fields(_read_evidence_json(directory / "worker-request.json", root), {"payload", "signature"})["payload"]
    if (validate_request(request) != spec or request["work_id"] != attempt.unit
            or request["project_root"] != str(source) or request["index_path"] != str(root / "dev_index.sqlite3")
            or request["index_sha256"] != index_sha
            or attempt.unit != f"{request['experiment_id']}/{request['arm_id']}/r{request['repeat']}/f{request['fold']}/model"):
        raise ValueError("completed unit request/spec/config lineage mismatch")
    launch = verify_parent_record(source, directory / "worker-launch.json", root)
    if (launch.get("journal_attempt_id") != attempt.attempt_id or launch.get("unit") != attempt.unit
            or launch.get("run_root") != str(root) or launch.get("spec_sha256") != spec.binding_sha256
            or launch.get("worker_request_sha256") != files["worker-request.json"]
            or launch["identity_record"]["binding"] != {"nonce": request["nonce"], "attempt_id": request["attempt_id"],
                                                        "spec_sha256": spec.binding_sha256}):
        raise ValueError("replayed/mismatched completed worker launch identity")
    supervisor = _read_evidence_json(directory / "supervisor.json", root)
    expected = {"unit": attempt.unit, "status": "awaiting_validation", "exit_code": 0,
                "spec_sha256": spec.binding_sha256, "nonce": request["nonce"], "attempt_id": request["attempt_id"],
                "request_sha256": files["worker-request.json"], "shard_sha256": files["shard.parquet"], "index_sha256": index_sha}
    observation = supervisor.get("observation", {})
    identity = launch["identity_record"]
    if (any(supervisor.get(key) != value for key, value in expected.items())
            or observation.get("cleanup_confirmed") is not True or observation.get("reason") is not None
            or observation.get("status") != "awaiting_validation" or observation.get("exit_code") != 0
            or observation.get("process_identities", {}).get(str(identity["pid"])) != identity["identity"]):
        raise ValueError("completed supervisor cleanup/identity mismatch")
    shard = directory / "shard.parquet"
    checked = _validate_shard(shard, authority, spec, declaration,
                              f"{request['experiment_id']}_{request['arm_id']}", request["repeat"], request["fold"], _file_hash(authority))
    checked["index_sha256"] = index_sha
    if _read_evidence_json(directory / "parent_validation.json", root) != checked:
        raise ValueError("completed unit independent labels/coverage validation mismatch")
    if experiment == "model_autogluon_core_features":
        from project.experiments.full_autogluon_svc_worker import verify_autogluon_artifacts
        child = verify_parent_record(source, directory / "autogluon-proof.json", root)
        if child.get("kind") != "autogluon_child":
            raise ValueError("isolated child evidence requires authenticated parent proof")
        child_directory = _validate_managed_path(child["directory"], "isolated child attempt", root=root, directory=True)
        for name, digest in child["artifacts"].items():
            if _file_hash(_validate_managed_path(child_directory / name, "isolated child artifact", root=root)) != digest:
                raise ValueError("isolated child evidence changed")
        if verify_autogluon_artifacts(child_directory, expected_request=request) != child["proof"]:
            raise ValueError("independent isolated child backend verification mismatch")
    if selector_model:
        link = verify_parent_record(source, directory / "selector-link.json", root)
        with FullFitJournal.open_readonly(root / "journal", spec, source_root=source) as journal:
            history = journal.attempts()
        selected = next((a for a in history if a.attempt_id == link.get("selector_attempt_id")), None)
        if (selected is None or selected.unit != attempt.unit.rsplit("/", 1)[0] + "/selector"
                or history.index(selected) >= history.index(attempt)
                or link != {"kind": "selector_model_link", "spec_sha256": spec.binding_sha256,
                            "selector_attempt_id": selected.attempt_id, "model_attempt_id": attempt.attempt_id,
                            "nonce": request["nonce"]}):
            raise ValueError("model lacks authentic preceding selector completion")
        verify_completed_selector(source, root, spec, selected, authority, index_sha)
        selected_request = _read_evidence_json(root / "journal" / selected.artifact_directory / "worker-request.json", root)["payload"]
        if selected_request != request:
            raise ValueError("selector/model must share exact nonce/request/session")
    return shard


def verify_interrupted_exit(source: Path, root: Path, spec: FullRunSpec, attempt: Any,
                            *, require_receipt: bool = False) -> dict[str, Any]:
    """Verify this journal attempt's signed launch and prove its owned session exited."""
    from project.experiments.full_campaign_runtime import confirm_owned_attempt_exit
    from project.experiments.full_snapshot_codec import exact_fields, validate_request

    directory = root / "journal" / attempt.artifact_directory
    launch = verify_parent_record(source, directory / "worker-launch.json", root)
    request_path = directory / "worker-request.json"
    request = exact_fields(_read_evidence_json(request_path, root), {"payload", "signature"})["payload"]
    if validate_request(request) != spec or request["project_root"] != str(source):
        raise ValueError("interrupted attempt launch/request/journal identity mismatch")

    snapshot = next((item for item in spec.candidates if item.spec.experiment_id == request["experiment_id"]), None)
    is_selector = (
        snapshot is not None
        and request["arm_id"] in snapshot.selector_arms
        and attempt.unit == request["work_id"].rsplit("/", 1)[0] + "/selector"
    )
    if attempt.unit != request["work_id"] and not is_selector:
        raise ValueError("interrupted attempt launch/request/journal identity mismatch")

    if (launch.get("unit") != request["work_id"] or launch.get("journal_attempt_id") != attempt.attempt_id
            or launch.get("run_root") != str(root) or launch.get("spec_sha256") != spec.binding_sha256
            or launch.get("worker_request_sha256") != _file_hash(request_path)
            or launch["identity_record"]["unit"] != request["work_id"]
            or launch["identity_record"]["binding"] != {"nonce": request["nonce"], "attempt_id": request["attempt_id"],
                                                        "spec_sha256": spec.binding_sha256}):
        raise ValueError("interrupted attempt launch/request/journal identity mismatch")
    confirm_owned_attempt_exit(launch["identity_record"])
    proof = {"kind": "owned_attempt_exit", "run_root": str(root), "spec_sha256": spec.binding_sha256,
             "journal_attempt_id": attempt.attempt_id, "unit": attempt.unit,
             "launch_sha256": _file_hash(directory / "worker-launch.json"), "cleanup_confirmed": True}
    if require_receipt and verify_parent_record(source, directory / "recovery-exit.json", root) != proof:
        raise ValueError("historical retry lacks authentic owned-attempt exit proof")
    return proof


def _verify_prediction_union(oof: Path, shards: list[Path]) -> None:
    """OOF probabilities must be exactly the parent-sealed shard probabilities."""
    columns = ("candidate_id", "row_index", "repeat", "fold", "prediction")
    connection = sqlite3.connect("")
    try:
        for name in ("shards", "oof"):
            connection.execute(f"CREATE TABLE {name}(candidate TEXT, row INTEGER, repeat INTEGER, fold INTEGER, prediction REAL, PRIMARY KEY(candidate,row,repeat))")
            for path in (shards if name == "shards" else [oof]):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=64000, columns=list(columns)):
                    arrays = [batch.column(index).to_pylist() for index in range(len(columns))]
                    try:
                        connection.executemany(f"INSERT INTO {name} VALUES(?,?,?,?,?)", zip(*arrays, strict=True))
                    except sqlite3.IntegrityError as error:
                        raise ValueError("duplicate sealed prediction identity") from error
        for left, right in (("shards", "oof"), ("oof", "shards")):
            if connection.execute(f"SELECT 1 FROM (SELECT * FROM {left} EXCEPT SELECT * FROM {right}) LIMIT 1").fetchone():
                raise ValueError("OOF differs from actual parent-sealed shard predictions")
    finally:
        connection.close()


def verify_ordinary_campaign_receipt(
    source_root: str | Path, run_root: str | Path, admitted: AdmittedFullRun,
    *, expected_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _existing_source_lock(Path(source_root)):
        return _verify_receipt_locked(source_root, run_root, admitted, expected_receipt=expected_receipt)


@contextmanager
def _existing_source_lock(source: Path):
    """Acquire the existing stable inode without modifying lock bytes or creating files."""
    path = _validate_managed_path(source / "campaign.lock", "existing campaign lock", root=source)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _verify_receipt_locked(
    source_root: str | Path, run_root: str | Path, admitted: AdmittedFullRun,
    *, expected_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute ordinary evidence from source bytes, never from a validator seal.

    This is read-only with respect to the campaign. Fresh processes must acquire
    fresh authentic admission. The resulting receipt deliberately grants neither
    comparative eligibility nor permission to skip/refit work.
    """
    from project.experiments import full_candidate_io as io
    from project.experiments.full_campaign_capabilities import (
        build_reviewed_route_arms,
    )
    from project.experiments.full_candidate_runner import (
        _build_candidate_declaration,
        _campaign_status,
        _encoding_provenance,
        _source_lineage,
        _validate_shard,
    )
    from project.experiments.full_snapshot_codec import (
        encode_snapshot,
        exact_fields,
        validate_request,
    )
    from project.models.full_candidate_artifacts import validate_full_candidate_oof

    if type(admitted) is not AdmittedFullRun:
        raise TypeError("authentic exact AdmittedFullRun required")
    source = _validate_managed_path(source_root, "receipt source root", directory=True)
    root = _validate_managed_path(run_root, "receipt run root", directory=True)
    admitted.verify_bindings(source, phase="launch")
    spec = admitted.spec
    if _read_evidence_json(root / "spec.json", root) != encode_snapshot(spec):
        raise ValueError("immutable receipt snapshot mismatch")
    if verify_parent_record(source, root / "run-authority.json", root) != resume_binding(admitted, root):
        raise ValueError("immutable source/implementation/environment admission snapshot mismatch")
    immutable_files = {"spec.json": _file_hash(root / "spec.json"),
                       "run_status.json": _file_hash(root / "run_status.json"),
                       "run-authority.json": _file_hash(root / "run-authority.json")}
    status = _read_evidence_json(root / "run_status.json", root)
    if status.get("status") not in ("awaiting_validation", "failed", "interrupted", "cancelled") or status.get("n03_eligible") is not False:
        raise ValueError("campaign does not have complete awaiting-validation transport")
    index_path = _validate_managed_path(root / "dev_index.sqlite3", "receipt index", root=root)
    index_sha = _file_hash(index_path)
    if verify_parent_record(source, root / "index-authority.json", root) != {
        "kind": "development_index", "run_root": str(root), "spec_sha256": spec.binding_sha256, "index_sha256": index_sha
    }:
        raise ValueError("authenticated development index mismatch")
    immutable_files["index-authority.json"] = _file_hash(root / "index-authority.json")
    encoding, _relative, _provenance_hash = _encoding_provenance(source, spec)
    with FullFitJournal.open_readonly(
        root / "journal", spec, source_root=source
    ) as journal, tempfile.TemporaryDirectory(prefix="full-route-validation-") as temporary:
        history = journal.attempts()
        latest = {attempt.unit: attempt for attempt in history}
        attempts = tuple(latest.values())
        if (len(attempts) != len(spec.planned_units)
                or set(latest) != set(spec.planned_units)
                or any(attempt.status != "completed" for attempt in attempts)):
            raise ValueError("missing, extra, duplicate or unfinished required journal unit")
        for prior in history:
            if prior != latest[prior.unit]:
                if prior.status not in ("failed", "interrupted", "cancelled"):
                    raise ValueError("unverified or duplicated historical attempt")
                verify_interrupted_exit(source, root, spec, prior, require_receipt=True)
        # Rebuild authority from actual source/split bytes. A run-owned index
        # claiming the data hash is not independently authoritative for labels.
        authority = io.build_development_index(source, spec, Path(temporary) / "authority.sqlite3")
        authority_sha = _file_hash(authority)
        nonces: set[str] = set()
        workers: set[str] = set()
        candidates = []
        transport_receipts = []
        for snapshot in spec.candidates:
            experiment = snapshot.spec.experiment_id
            arms = build_reviewed_route_arms(snapshot, year_built_encoding=encoding)
            if set(arms) != set(snapshot.arms):
                raise ValueError("exact ordinary arm set mismatch")
            declaration = _build_candidate_declaration(snapshot, arms, spec)
            hpo_shards = []
            hpo_proofs = {}
            if snapshot.study is not None:
                from project.experiments.full_hpo_campaign import HpoPartitionContext, verify_hpo_partition
                for repeat in range(spec.repeats):
                    for fold in range(spec.folds):
                        name = f"{experiment}_r{repeat}_f{fold}_partition.json"
                        proof = _read_evidence_json(root / name, root)
                        context = HpoPartitionContext(source, root, spec, index_path, repeat, fold)
                        partition = verify_hpo_partition(context, proof, journal=journal)
                        immutable_files[name] = _file_hash(root / name)
                        hpo_proofs[f"{repeat}:{fold}"] = proof
                        for arm in snapshot.arms:
                            candidate = f"{experiment}_{arm}"
                            metadata = partition["candidates"][candidate]
                            if "fold_fit_config_sha256" in metadata:
                                existing = declaration["candidates"][candidate]
                                existing.pop("model_fit_config_sha256", None)
                                existing.setdefault("fold_fit_config_sha256", {}).update(metadata["fold_fit_config_sha256"])
                            else:
                                declaration["candidates"][candidate] = metadata
                            hpo_shards.append((Path(partition["shard_paths"][arm]), candidate, repeat, fold))
            if _read_evidence_json(root / f"{experiment}_declaration.json", root) != declaration:
                raise ValueError("candidate declaration differs from independent exact configuration")
            if _read_evidence_json(root / f"{experiment}_source_lineage.json", root) != _source_lineage(snapshot):
                raise ValueError("source matrix/variant/notebook lineage mismatch")
            for suffix in ("declaration.json", "transport_validation.json", "source_lineage.json"):
                name = f"{experiment}_{suffix}"
                immutable_files[name] = _file_hash(root / name)
            shards = []
            units = []
            supervisor_artifacts = {}
            for shard, candidate, repeat, fold in hpo_shards:
                _validate_shard(shard, authority, spec, declaration, candidate, repeat, fold, authority_sha)
                shards.append(shard)
            supervisor_artifacts.update(hpo_proofs)
            for attempt in attempts:
                if not attempt.unit.startswith(experiment + "/"):
                    continue
                if snapshot.study is not None:
                    units.append(attempt.unit)
                    continue
                directory = root / "journal" / attempt.artifact_directory
                if attempt.unit.endswith("/selector"):
                    verify_completed_selector(source, root, spec, attempt, authority, index_sha)
                    units.append(attempt.unit)
                    continue
                verify_completed_unit(source, root, spec, attempt, declaration, authority, index_sha)
                envelope = exact_fields(_read_evidence_json(directory / "worker-request.json", root), {"payload", "signature"})
                request = envelope["payload"]
                request_spec = validate_request(request)
                if (request_spec != spec or request["work_id"] != attempt.unit
                        or request["project_root"] != str(source)
                        or request["index_path"] != str(index_path)
                        or request["index_sha256"] != index_sha):
                    raise ValueError("worker request/spec/source/index mismatch")
                if request["nonce"] in nonces or request["attempt_id"] in workers:
                    raise ValueError("replayed worker nonce or attempt identity")
                nonces.add(request["nonce"])
                workers.add(request["attempt_id"])
                supervisor = _read_evidence_json(directory / "supervisor.json", root)
                shard = directory / "shard.parquet"
                expected_supervisor = {
                    "unit": attempt.unit, "status": "awaiting_validation", "exit_code": 0,
                    "spec_sha256": spec.binding_sha256, "nonce": request["nonce"],
                    "attempt_id": request["attempt_id"], "request_sha256": _file_hash(directory / "worker-request.json"),
                    "shard_sha256": _file_hash(shard), "index_sha256": index_sha,
                }
                if any(supervisor.get(key) != value for key, value in expected_supervisor.items()):
                    raise ValueError("supervisor spec/nonce/artifact/exit binding mismatch")
                observation = supervisor.get("observation", {})
                if (observation.get("cleanup_confirmed") is not True
                        or not observation.get("process_identities")
                        or observation.get("unit") != attempt.unit
                        or observation.get("status") != "awaiting_validation"
                        or observation.get("exit_code") != 0 or observation.get("reason") is not None):
                    raise ValueError("independent supervisor cleanup/process identity proof required")
                checked = _validate_shard(shard, authority, spec, declaration,
                                          f"{experiment}_{request['arm_id']}", request["repeat"], request["fold"], authority_sha)
                checked["index_sha256"] = index_sha
                if _read_evidence_json(directory / "parent_validation.json", root) != checked:
                    raise ValueError("sealed validator claim differs from independent source label/coverage validation")
                shards.append(shard)
                units.append(attempt.unit)
                supervisor_artifacts[attempt.unit] = _file_hash(directory / "supervisor.json")
            oof = _validate_managed_path(root / f"{experiment}_oof.parquet", "receipt OOF", root=root)
            oof_sha = _file_hash(oof)
            _verify_prediction_union(oof, shards)
            validation = validate_full_candidate_oof(oof, io.iter_full_expected_batches(authority, spec), declaration)
            if _read_evidence_json(root / f"{experiment}_transport_validation.json", root) != validation:
                raise ValueError("stored OOF validation differs from independent recomputation")
            if _file_hash(oof) != oof_sha:
                raise ValueError("OOF bytes changed during independent verification")
            candidates.append({"experiment_id": experiment, "arms": list(snapshot.arms),
                               "oof_sha256": oof_sha, "units": units, "validation": validation,
                               "supervisor_artifacts": supervisor_artifacts})
            immutable_files[str(oof.relative_to(root))] = oof_sha
            transport_receipts.append({"experiment_id": experiment, "status": "awaiting_validation",
                                       "oof_path": str(oof), "oof_sha256": oof_sha,
                                       "comparative_completion_available": False})
        expected_status = _campaign_status(admitted, transport_receipts, "awaiting_validation")
        if status["status"] == "awaiting_validation" and status != expected_status:
            raise ValueError("campaign status/inventory differs from actual independently verified work")
        if status["status"] != "awaiting_validation":
            recovery_status = verify_parent_record(source, root / "resumed-transport.json", root)
            if recovery_status != {"kind": "resumed_transport", "status": expected_status,
                                   "journal_sha256": _file_hash(root / "journal/full_fit.sqlite3")}:
                raise ValueError("resume transport completion lacks authenticated independent status")
            immutable_files["resumed-transport.json"] = _file_hash(root / "resumed-transport.json")
        admitted.verify_bindings(source, phase="launch")
        journal.verify_completed()
        if _file_hash(index_path) != index_sha or any(_file_hash(root / name) != digest for name, digest in immutable_files.items()):
            raise ValueError("immutable campaign evidence changed during receipt verification")
        receipt = {"status": "independently_verified_ordinary_receipt", "run_id": spec.run_id,
                   "spec_sha256": spec.binding_sha256, "source_root": str(source), "run_root": str(root),
                   "implementation_sources": [asdict(binding) for binding in admitted.implementation_sources],
                   "environment": dict(admitted.actual_environment),
                   "sources": [asdict(binding) for binding in spec.sources],
                   "resource_policy": asdict(spec.resource_policy),
                   "encoding": {"encoding_id": encoding, "path": _relative, "sha256": _provenance_hash},
                    "verified_model_fits": sum(not a.unit.endswith("/selector") for a in history),
                    "verified_selector_fits": sum(a.unit.endswith("/selector") for a in history), "candidates": candidates,
                   "journal_sha256": _file_hash(root / "journal/full_fit.sqlite3"),
                   "artifact_sha256": immutable_files, "index_sha256": index_sha,
                   "inventory": expected_status["inventory"], "n03_eligible": False,
                   "comparative_completion_available": False}
        if expected_receipt is not None and expected_receipt != receipt:
            raise ValueError("forged, replayed or mismatched ordinary receipt")
        return receipt


def issue_ordinary_campaign_completion(
    source_root: str | Path, run_root: str | Path, admitted: AdmittedFullRun, destination: str | Path,
) -> dict[str, Any]:
    """Parent-only ordinary publication after independent verification and durable MAC.

    Threat model: trusted fixed workers and trusted same-UID parent code, accidental
    corruption, stale/replayed evidence and concurrent publishers. This is NOT
    security isolation against a malicious same-UID process that can read the
    authority secret or rewrite the installation. The secret is never sent in a
    worker request/environment, receipt, manifest or admission snapshot.
    """
    if any(os.environ.get(name) for name in ("FULL_CAMPAIGN_WORKER_KEY", "FULL_HPO_WORKER_KEY")):
        raise PermissionError("worker transport cannot issue parent completion")
    source = _validate_managed_path(source_root, "completion source", directory=True)
    root = _validate_managed_path(run_root, "completion run", directory=True)
    destination = _validate_managed_path(destination, "completion publication", root=root, directory=True)
    if destination != root / "completion":
        raise ValueError("completion must use the fixed run-owned publication path")
    with _existing_source_lock(source):
        if destination.exists():
            raise FileExistsError("immutable completion publication already exists")
        receipt = _verify_receipt_locked(source, root, admitted)
        key = _parent_authority_key(source, create=True)
        nonce = secrets.token_hex(32)
        documents = _completion_documents(receipt, admitted, destination, nonce)
        publication = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in documents.items()}
        permit = OrdinaryCompletionAdmission(str(source), str(root), str(destination), admitted.spec.binding_sha256,
                                             nonce, hashlib.sha256(key).hexdigest(), _json(receipt), _json(publication), "")
        permit = replace(permit, signature=_completion_signature(key, permit))
        # Genuine admission performs fresh source/implementation/environment checks
        # AND recomputes the receipt, rather than accepting this MAC alone.
        admitted.verify_bindings(source, phase="completion", completion_admission=permit)
        documents["admission.json"] = _json(asdict(permit))
        stage = root / f".completion-stage-{nonce}"
        stage.mkdir(mode=0o700)
        for name, text in documents.items():
            with (stage / name).open("x") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(stage)
        # Recheck immediately before committing: a valid permit issued before
        # a source/artifact mutation is stale, not a license to publish it.
        admitted.verify_bindings(source, phase="completion", completion_admission=permit)
        if ({path.name for path in stage.iterdir()} != set(documents)
                or any(_file_hash(_validate_managed_path(stage / name, "staged completion", root=stage))
                       != hashlib.sha256(text.encode()).hexdigest() for name, text in documents.items())):
            raise ValueError("staged completion publication changed before commit")
        # A crash before this commit leaves only quarantined staging, never a
        # visible partial completed directory. No-replace is essential: ordinary
        # rename would overwrite a concurrently created empty destination.
        _publish_completion_directory(stage, destination)
        _fsync_directory(root)
        return json.loads(documents["run_status.json"])


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parent_authority_key(source: Path, *, create: bool = False) -> bytes:
    directory = _validate_managed_path(source / ".full_campaign_completion_authority", "parent authority", root=source, directory=True)
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError("parent authority directory must be owner-only")
    path = _validate_managed_path(directory / "key", "parent authority key", root=directory)
    if create and not path.exists():
        temporary = directory / f".key-stage-{secrets.token_hex(32)}"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                pass
            _fsync_directory(directory)
            _fsync_directory(source)
        finally:
            temporary.unlink()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise PermissionError("parent authority key must be an owner-only regular file")
        key = os.read(descriptor, 33)
    finally:
        os.close(descriptor)
    if len(key) != 32:
        raise ValueError("invalid durable parent authority")
    return key


def _completion_signature(key: bytes, permit: OrdinaryCompletionAdmission) -> str:
    payload = asdict(permit)
    payload.pop("signature")
    return hmac.new(key, ("ordinary_completion_admission_v1:" + _json(payload)).encode(), hashlib.sha256).hexdigest()


def verify_completion_admission(source: Path, admitted: AdmittedFullRun, permit: OrdinaryCompletionAdmission) -> dict[str, Any]:
    """Durable authentication plus actual revalidation; never a truthy seal check."""
    if type(permit) is not OrdinaryCompletionAdmission:
        raise TypeError("exact ordinary completion admission required")
    key = _parent_authority_key(source)
    if (type(permit.signature) is not str or not hmac.compare_digest(permit.signature, _completion_signature(key, permit))
            or permit.authority_id != hashlib.sha256(key).hexdigest()
            or permit.source_root != str(source) or permit.spec_sha256 != admitted.spec.binding_sha256):
        raise ValueError("forged, replayed or mismatched parent completion admission")
    from project.experiments.full_snapshot_codec import parse_json
    receipt = parse_json(permit.receipt_json)
    root = _validate_managed_path(permit.run_root, "completion bound run", directory=True)
    if permit.destination != str(root / "completion"):
        raise ValueError("replayed completion destination mismatch")
    return _verify_receipt_locked(source, root, admitted, expected_receipt=receipt)


def _completion_documents(receipt: dict, admitted: AdmittedFullRun, destination: Path, nonce: str) -> dict[str, str]:
    from project.experiments.full_run_contracts import (
        SEMANTIC_VARIANTS,
        source_matrix_identity,
    )

    inventory = json.loads(_json(receipt["inventory"]))
    experiments = []
    for candidate in receipt["candidates"]:
        experiment = candidate["experiment_id"]
        validation_sha = candidate["validation"]["validation_sha256"]
        supervisor_sha = hashlib.sha256(_json(candidate["supervisor_artifacts"]).encode()).hexdigest()
        row = next(row for row in inventory if row["experiment_id"] == source_matrix_identity(experiment))
        if experiment in SEMANTIC_VARIANTS:
            coverage = row.setdefault("variant_coverage", {v: "pending" for v in SEMANTIC_VARIANTS})
            coverage[experiment] = "completed"
            if all(coverage[v] == "completed" for v in SEMANTIC_VARIANTS):
                values = {c["experiment_id"]: c["validation"]["validation_sha256"] for c in receipt["candidates"]
                          if c["experiment_id"] in SEMANTIC_VARIANTS}
                row.update(status="completed", n03_eligible=True,
                           validation_sha256=hashlib.sha256(_json(values).encode()).hexdigest(),
                           supervisor_sha256=hashlib.sha256(_json({c["experiment_id"]: c["supervisor_artifacts"]
                               for c in receipt["candidates"] if c["experiment_id"] in SEMANTIC_VARIANTS}).encode()).hexdigest())
        else:
            row.update(status="completed", n03_eligible=True, validation_sha256=validation_sha, supervisor_sha256=supervisor_sha)
        experiments.append({"experiment_id": experiment, "status": "completed", "arms": candidate["arms"],
                            "oof_path": str(Path(receipt["run_root"]) / f"{experiment}_oof.parquet"),
                            "oof_sha256": candidate["oof_sha256"], "n03_eligible": True})
    complete = all(row["status"] == "completed" for row in inventory)
    status = {"run_id": admitted.spec.run_id, "spec_sha256": admitted.spec.binding_sha256,
              "status": "all_experiments_completed" if complete else "finished_with_blockers",
              "experiments": experiments, "inventory": inventory, "n03_eligible": complete,
              "comparative_completion_available": True, "completion_nonce": nonce,
              "completion_directory": str(destination)}
    manifest = {"schema_version": 1, "execution_mode": "full", "run_id": admitted.spec.run_id,
                "spec_sha256": admitted.spec.binding_sha256, "completion_nonce": nonce,
                "candidates": experiments, "receipt_sha256": hashlib.sha256(_json(receipt).encode()).hexdigest()}
    return {"manifest.json": _json(manifest), "run_status.json": _json(status), "receipt.json": _json(receipt)}


def _publish_completion_directory(stage: Path, destination: Path) -> None:
    """Linux atomic directory publication with RENAME_NOREPLACE; fail closed elsewhere."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise RuntimeError("atomic no-replace completion publication unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def verify_ordinary_campaign_completion(source_root: str | Path, run_root: str | Path, admitted: AdmittedFullRun) -> dict[str, Any]:
    """Fresh-process, read-only completion authentication and scientific revalidation."""
    from project.experiments.full_snapshot_codec import exact_fields, parse_json

    source = _validate_managed_path(source_root, "completion source", directory=True)
    root = _validate_managed_path(run_root, "completion run", directory=True)
    destination = _validate_managed_path(root / "completion", "completion publication", root=root, directory=True)
    with _existing_source_lock(source):
        if {path.name for path in destination.iterdir()} != {"manifest.json", "run_status.json", "receipt.json", "admission.json"}:
            raise ValueError("incomplete or unexpected completion publication")
        document = exact_fields(_read_evidence_json(destination / "admission.json", root), set(OrdinaryCompletionAdmission.__dataclass_fields__))
        permit = OrdinaryCompletionAdmission(**document)
        if permit.run_root != str(root) or permit.destination != str(destination):
            raise ValueError("replayed completion run/attempt destination")
        admitted.verify_bindings(source, phase="completion", completion_admission=permit)
        publication = parse_json(permit.publication_json)
        expected = _completion_documents(parse_json(permit.receipt_json), admitted, destination, permit.proof_nonce)
        if set(publication) != set(expected):
            raise ValueError("invalid signed completion artifact set")
        for name, text in expected.items():
            if (_file_hash(destination / name) != publication[name]
                    or hashlib.sha256(text.encode()).hexdigest() != publication[name]):
                raise ValueError("completion publication bytes differ from authenticated independent evidence")
        return parse_json(expected["run_status.json"])
