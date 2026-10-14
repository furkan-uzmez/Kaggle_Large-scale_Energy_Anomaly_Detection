"""Sequential full-campaign runner and journal-backed fit accounting.

Production orchestration for the full development campaign. It fits one
experiment at a time, one arm at a time, one outer partition at a time, using
isolated subprocess workers (``full_campaign_worker``), the authenticated
development index (``full_candidate_io``), and the frozen dispatch
(``candidate_dispatch``). This milestone supports only the fixed reviewed
ordinary route and obtains independent durable parent completion after fitting.
It never calls the synthetic runner, never derives
labels from row parity, streams OOF batches to Parquet, and never writes a
comparative completion claim before full OOF validation and sealed evidence.

``FullFitAccountant`` adapts the durable ``FullFitJournal`` to the shared
``FitAccounting`` protocol so ``fit_arm`` debits exactly once per estimator fit
without a fabricated bounded ``FitBudget``. The runner opens the correct planned
unit(s) before each worker fit: normal arms charge one ``/model`` unit.
Selector and HPO routing remain outside this milestone.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from project.experiments import full_candidate_io as io
from project.experiments.candidate_contracts import (
    CandidateSpec,
    _validate_managed_path,
)
from project.experiments.candidate_dispatch import (
    CandidateArm,
    _configuration,
)
from project.experiments.full_campaign_capabilities import (
    FullCampaignCapability,
    build_reviewed_route_arms,
    verify_full_campaign_capability,
)
from project.experiments.full_campaign_preflight import (
    FullDataPreflight,
    verify_admitted_full_run,
)
from project.experiments.full_campaign_runtime import (
    CampaignLock,
    run_work_unit,
)
from project.experiments.full_campaign_worker import (
    ParentLifecycleService,
    prepare_worker_request,
    semantic_digest,
)
from project.experiments.full_run_contracts import (
    AdmittedFullRun,
    ArtifactEvidence,
    FullAttemptRecord,
    FullFitJournal,
    FullRunSpec,
    _file_hash,
    _json,
)
from project.experiments.full_snapshot_codec import (
    configuration_digest,
    encode_snapshot,
)
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1
from project.models.candidate_artifacts import (
    candidate_oof_schema,
    write_candidate_oof_batches,
)
from project.models.full_candidate_artifacts import validate_full_candidate_oof


def _encoding_provenance(source: Path, spec: FullRunSpec) -> tuple[str, str, str]:
    """Resolve only the authoritative encoding evidence bound in the snapshot."""
    relative = "docs/evidence/year_built_encoding_v1.json"
    bindings = [binding for binding in spec.sources if binding.role == "encoding"]
    if len(bindings) != 1 or bindings[0].path != relative:
        raise ValueError("explicit authoritative encoding provenance binding required")
    binding = bindings[0]
    path = binding.verify(source)
    payload = json.loads(path.read_text())
    if type(payload) is not dict or payload.get("encoding_id") != LEAD_YEAR_OFFSET_1900_UINT8_V1:
        raise ValueError("unsupported encoding provenance contract")
    return payload["encoding_id"], relative, binding.sha256


@dataclass
class FullFitAccountant:
    """Journal-backed ``FitAccounting``; one durable debit per estimator fit.

    The runner sets the pending unit queue with :meth:`open_units` immediately
    before a ``fit_arm`` call. Each ``before_fit`` dequeues the next planned unit
    and opens a durable journal attempt for it. ``reserved_model_fits`` reflects
    the finite planned fit count, never a runtime cap.
    """

    journal: FullFitJournal
    planned_units: tuple[str, ...] = ()
    _pending: list[str] = field(default_factory=list)
    _active: FullAttemptRecord | None = None
    retry_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.journal, FullFitJournal):
            raise TypeError("typed FullFitJournal required")

    def open_units(self, units: tuple[str, ...]) -> None:
        if self._active is not None or self._pending:
            raise ValueError("previous fit units are still open")
        for unit in units:
            if unit not in self.planned_units:
                raise ValueError(f"unit is not part of the immutable plan: {unit}")
        self._pending = list(units)

    def before_fit(self, *, prepare_attempt: Callable[[FullAttemptRecord], None] | None = None) -> None:
        if self._active is not None:
            raise ValueError("a journal attempt is already open")
        if not self._pending:
            raise ValueError("no planned unit is open for this fit")
        unit = self._pending.pop(0)
        if any(attempt.unit == unit for attempt in self.journal.attempts()):
            if not self.retry_reason or not self.retry_reason.strip():
                raise ValueError("explicit retry reason required")
            self._active = self.journal.retry(unit, reason=self.retry_reason, prepare_attempt=prepare_attempt)
        else:
            self._active = self.journal.begin(unit, prepare_attempt=prepare_attempt)

    def finish_active(
        self,
        status: str,
        *,
        reason: str | None = None,
        evidence: tuple[ArtifactEvidence, ...] = (),
    ) -> None:
        if self._active is None:
            raise ValueError("no active journal attempt to finish")
        if status == "completed":
            if not evidence:
                raise ValueError("completed fit attempt requires sealed evidence")
            self.journal.seal(self._active.attempt_id, evidence)
        self.journal.finish(self._active.attempt_id, status, reason=reason)
        self._active = None

    @property
    def reserved_model_fits(self) -> int:
        return len(self.planned_units)

    @property
    def used_model_fits(self) -> int:
        return len(self.journal.attempts())

    @property
    def active_directory(self) -> str:
        if self._active is None:
            raise ValueError("no active journal attempt")
        return self._active.artifact_directory


def _run_hash(spec: FullRunSpec, role: str) -> str:
    for source in spec.sources:
        if source.role == role:
            return source.sha256
    raise ValueError(f"source role not bound: {role}")


def _oof_schema(meta: io._IndexMeta) -> pa.Schema:
    """Derive the canonical OOF schema from the authoritative index metadata."""
    building_type = pa.int64() if meta.building_dtype == "int64" else pa.string()
    return candidate_oof_schema(building_type, pa.timestamp(meta.timestamp_unit, tz=meta.timestamp_time_zone))


def _batch_table(
    batch: dict[str, Any],
    predictions: np.ndarray,
    spec: CandidateSpec,
    arm: CandidateArm,
    admitted_spec: FullRunSpec,
    repeat: int,
    fold: int,
    fit_sha: str,
    schema: pa.Schema | None = None,
) -> pa.Table:
    schema = candidate_oof_schema() if schema is None else schema
    size = len(batch["row_index"])
    columns = {
        "candidate_id": [f"{spec.experiment_id}_{arm.arm_id}"] * size,
        "recipe_id": [spec.experiment_id] * size,
        "model_family": [spec.model_family] * size,
        "model_config_id": [f"{spec.experiment_id}_{arm.arm_id}_config"] * size,
        "row_index": [int(value) for value in batch["row_index"]],
        "repeat": [repeat] * size,
        "fold": [fold] * size,
        "building_id": list(batch["building_id"]),
        "timestamp": pl.Series(batch["timestamp"])
        .cast(pl.Datetime(schema.field("timestamp").type.unit, schema.field("timestamp").type.tz))
        .to_arrow()
        .cast(schema.field("timestamp").type),
        "prediction": [float(value) for value in predictions],
        "target": [int(value) for value in batch["target"]],
        "split_sha256": [_run_hash(admitted_spec, "split")] * size,
        "data_sha256": [_run_hash(admitted_spec, "data")] * size,
        "config_sha256": [admitted_spec.binding_sha256] * size,
        "run_id": [admitted_spec.run_id] * size,
        "split_seed": [int(batch["split_seed"])] * size,
        "experiment_id": [spec.experiment_id] * size,
        "arm_id": [arm.arm_id] * size,
        "score_kind": [arm.score_kind] * size,
        "positive_label": [1] * size,
        "execution_mode": ["full"] * size,
        "non_comparative": [False] * size,
        "n03_eligible": [False] * size,
        "model_fit_config_sha256": [fit_sha] * size,
    }
    return pa.Table.from_pydict(columns, schema=schema)


def _build_candidate_declaration(
    snapshot: Any,
    arms: dict[str, CandidateArm],
    admitted_spec: FullRunSpec,
) -> dict[str, Any]:
    candidate_metas: dict[str, Any] = {}
    for arm_id in snapshot.arms:
        arm = arms.get(arm_id)
        if arm is None and arm_id == "control" and "passthrough" in arms:
            arm = arms["passthrough"]
        elif arm is None and arm_id == "challenger" and "passthrough" in arms:
            other = next(k for k in arms if k != "passthrough")
            arm = arms[other]
        if arm is None:
            continue
        underlying = arm.transformer_or_pipeline
        fit_sha = configuration_digest(_configuration(underlying))
        candidate_metas[f"{snapshot.spec.experiment_id}_{arm_id}"] = {
            "recipe_id": snapshot.spec.experiment_id,
            "model_family": snapshot.spec.model_family,
            "model_config_id": f"{snapshot.spec.experiment_id}_{arm_id}_config",
            "experiment_id": snapshot.spec.experiment_id,
            "arm_id": arm_id,
            "score_kind": arm.score_kind,
            "positive_label": 1,
            "model_fit_config_sha256": fit_sha,
        }
    return {
        "schema_version": 1,
        "source_notebook": snapshot.source_notebook,
        "source_cell_id": snapshot.source_cell_id,
        "vault_row_indices": [],
        "vault_building_ids": [],
        "protocol": {
            "repeat_ids": list(range(admitted_spec.repeats)),
            "fold_ids": list(range(admitted_spec.folds)),
            "split_seeds": {str(i): seed for i, seed in enumerate(admitted_spec.split_seeds)},
        },
        "candidates": candidate_metas,
        "split_sha256": _run_hash(admitted_spec, "split"),
        "data_sha256": _run_hash(admitted_spec, "data"),
        "config_sha256": admitted_spec.binding_sha256,
        "run_id": admitted_spec.run_id,
        "execution_mode": "full",
        "non_comparative": False,
        "n03_eligible": False,
    }


def _source_lineage(snapshot) -> dict:
    from project.experiments.full_run_contracts import (
        SEMANTIC_VARIANTS,
        source_matrix_identity,
    )
    from project.features.selection import SEMANTIC_BLOCKS

    lineage = {"experiment_id": snapshot.spec.experiment_id,
            "source_matrix_id": source_matrix_identity(snapshot.spec.experiment_id),
            "source_notebook": snapshot.source_notebook, "source_cell_id": snapshot.source_cell_id,
            "method_params": dict(snapshot.spec.method_params)}
    if snapshot.spec.experiment_id in SEMANTIC_VARIANTS:
        dropped = SEMANTIC_BLOCKS[dict(snapshot.spec.method_params)["drop_block"]]
        lineage["retained_raw_columns"] = {"control": list(snapshot.spec.feature_pool),
            "challenger": [name for name in snapshot.spec.feature_pool if name not in dropped]}
        lineage["approval_source"] = "docs/superpowers/specs/2026-10-04-full-campaign-decisions.md#3-semantic-block-drop-dim_semantic_block_drop"
    return lineage


def _stream_shards(shard_paths: list[Path]) -> Iterator[pa.RecordBatch]:
    for path in shard_paths:
        file = pq.ParquetFile(path)
        yield from file.iter_batches(batch_size=io.MAX_BATCH_ROWS)


def _exclusive_json(path: Path, payload: object) -> None:
    with path.open("x") as stream:
        stream.write(_json(payload))
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _campaign_status(admitted: AdmittedFullRun, receipts: list[dict[str, Any]], status: str) -> dict[str, Any]:
    from project.experiments.full_run_contracts import SEMANTIC_VARIANTS

    actual = {receipt["experiment_id"]: receipt for receipt in receipts}
    inventory = []
    for row in admitted.rows:
        entry = asdict(row.record)
        entry["blocker_codes"] = list(entry["blocker_codes"])
        receipt = actual.get(entry["experiment_id"])
        if receipt is not None:
            entry.update(status=receipt["status"])
            if "reason" in receipt:
                entry["reason"] = receipt["reason"]
        if entry["experiment_id"] == "dim_semantic_block_drop" and any(
                s.spec.experiment_id in SEMANTIC_VARIANTS for s in admitted.spec.candidates):
            coverage = {identity: actual.get(identity, {}).get("status", "pending") for identity in SEMANTIC_VARIANTS}
            entry["variant_coverage"] = coverage
            states = set(coverage.values())
            entry["status"] = next(iter(states)) if len(states) == 1 else "blocked"
        inventory.append(entry)
    return {
        "run_id": admitted.spec.run_id,
        "status": status,
        "experiments": receipts,
        "n03_eligible": False,
        "comparative_completion_available": False,
        "inventory": inventory,
    }


def _copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _validate_shard(
    path: Path, index_path: Path, spec: FullRunSpec, declaration: dict,
    candidate_id: str, repeat: int, fold: int, index_sha256: str,
) -> dict[str, Any]:
    """Parent-owned exact partition coverage against the independently bound index.

    Expected and observed keys live in temporary disk SQLite, not a Python
    population map. This per-fit check is distinct from final 12-fold OOF review.
    """
    path = _validate_managed_path(path, "parent validation shard", root=path.parent)
    index_path = _validate_managed_path(index_path, "parent validation index")
    if _file_hash(index_path) != index_sha256:
        raise ValueError("development index changed before parent validation")
    authority, _meta = io._open_bound_index(index_path, spec)
    expected_schema = _oof_schema(_meta)
    connection = sqlite3.connect("")
    try:
        connection.execute("CREATE TABLE expected(row_index INTEGER PRIMARY KEY, building, timestamp_ns INTEGER, target INTEGER, seed INTEGER)")
        connection.execute("CREATE TABLE observed(row_index INTEGER PRIMARY KEY, building, timestamp_ns INTEGER, target INTEGER, seed INTEGER)")
        cursor = authority.execute(
            "SELECT d.row_index,d.building_id,d.timestamp_ns,d.target,a.seed FROM dev_row d "
            "JOIN assign a ON a.building_id=d.building_id WHERE a.repeat=? AND a.fold=?", (repeat, fold))
        while rows := cursor.fetchmany(io.MAX_BATCH_ROWS):
            connection.executemany("INSERT INTO expected VALUES(?,?,?,?,?)", rows)
        expected_rows = connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0]
        if expected_rows == 0:
            raise ValueError("parent expected partition is empty")
        metadata = dict(declaration["candidates"][candidate_id])
        fold_hashes = metadata.pop("fold_fit_config_sha256", None)
        if fold_hashes is not None:
            metadata.pop("model_fit_config_sha256", None)
            try:
                metadata["model_fit_config_sha256"] = fold_hashes[f"{repeat}:{fold}"]
            except KeyError as error:
                raise ValueError("authenticated selected fold configuration missing") from error
        metadata.update({name: declaration[name] for name in
                         ("data_sha256", "split_sha256", "config_sha256", "run_id",
                          "execution_mode", "non_comparative", "n03_eligible")})
        metadata.update(candidate_id=candidate_id, repeat=repeat, fold=fold,
                        split_seed=spec.split_seeds[repeat])
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
            raise ValueError("shard schema mismatch")
        for batch in parquet.iter_batches(batch_size=io.MAX_BATCH_ROWS):
            frame = pl.from_arrow(batch).with_columns(pl.col("timestamp").cast(pl.Int64))
            records = []
            for row in frame.iter_rows(named=True):
                if any(row[name] != value for name, value in metadata.items()):
                    raise ValueError("shard identity/configuration/lineage mismatch")
                prediction = row["prediction"]
                if (prediction is None or not np.isfinite(prediction)
                        or metadata["score_kind"] == "positive_probability" and not 0 <= prediction <= 1):
                    raise ValueError("shard prediction is invalid")
                records.append((row["row_index"], row["building_id"], row["timestamp"],
                                row["target"], row["split_seed"]))
            try:
                connection.executemany("INSERT INTO observed VALUES(?,?,?,?,?)", records)
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate shard row") from exc
        for left, right in (("expected", "observed"), ("observed", "expected")):
            if connection.execute(f"SELECT 1 FROM (SELECT * FROM {left} EXCEPT SELECT * FROM {right}) LIMIT 1").fetchone():
                raise ValueError("shard missing/extra row or authoritative label/key mismatch")
        return {"status": "validated_shard", "rows": expected_rows, "sha256": _file_hash(path),
                "spec_sha256": spec.binding_sha256, "index_sha256": index_sha256,
                "candidate_id": candidate_id, "repeat": repeat, "fold": fold,
                "fit_config_sha256": metadata["model_fit_config_sha256"]}
    finally:
        authority.close()
        connection.close()


def resume_full_campaign(
    admitted_spec: FullRunSpec, source_root: str | Path, run_root: str | Path, *,
    data_preflight: AdmittedFullRun, capability: FullCampaignCapability | None = None,
    retry_reason: str | None = None,
) -> dict[str, Any]:
    """Verify completed evidence or recover exact exited ordinary attempts explicitly."""
    from project.experiments.full_campaign_evidence import (
        verify_ordinary_campaign_completion,
    )

    if type(admitted_spec) is not FullRunSpec or type(data_preflight) is not AdmittedFullRun:
        raise TypeError("exact snapshot and fresh authentic admission required")
    verify_admitted_full_run(source_root, data_preflight, spec=admitted_spec, phase="resume")
    verify_full_campaign_capability(source_root, capability)
    from project.experiments.full_campaign_capabilities import HPO_ROUTE_ID, AG_ROUTE_ID
    if capability.route_id in (HPO_ROUTE_ID, AG_ROUTE_ID) and not (Path(run_root) / "completion").exists():
        raise ValueError("unsupported recovery for selected route before mutation; authenticated completed receipt required")
    root = _validate_managed_path(run_root, "resume run", directory=True)
    if not (root / "completion").exists():
        if not (root / "spec.json").is_file():
            raise ValueError("immutable snapshot missing: proven process exit and explicit journal retry required")
        return run_full_campaign(admitted_spec, source_root, root, data_preflight=data_preflight,
                                 capability=capability, _resume=True, _retry_reason=retry_reason)
    if retry_reason is not None:
        raise ValueError("completed evidence cannot authorize a new retry/refit")
    return verify_ordinary_campaign_completion(source_root, root, data_preflight)


def _verify_resume_state(source: Path, root: Path, admitted: AdmittedFullRun,
                         retry_reason: str | None) -> tuple[dict[str, Path], list[tuple[FullAttemptRecord, dict]]]:
    from project.experiments.full_campaign_evidence import (
        _read_evidence_json,
        resume_binding,
        verify_completed_unit,
        verify_interrupted_exit,
        verify_parent_record,
    )

    spec = admitted.spec
    admitted.verify_bindings(source, phase="resume")
    if (_read_evidence_json(root / "spec.json", root) != encode_snapshot(spec)
            or verify_parent_record(source, root / "run-authority.json", root) != resume_binding(admitted, root)):
        raise ValueError("immutable resume snapshot/source/implementation/environment mismatch")
    encoding, _, _ = _encoding_provenance(source, spec)
    from project.experiments.full_campaign_runtime import confirm_owned_attempt_exit
    from project.experiments.full_snapshot_codec import exact_fields, validate_request
    for directory in (root / "attempts").iterdir():
        _validate_managed_path(directory, "owned worker attempt", root=root, directory=True)
        launch = verify_parent_record(source, directory / "worker-launch.json", root)
        envelope = exact_fields(_read_evidence_json(directory / "worker-request.json", root), {"payload", "signature"})
        request = envelope["payload"]
        if (validate_request(request) != spec or request["attempt_id"] != directory.name
                or launch.get("unit") != request["work_id"] or launch.get("run_root") != str(root)
                or launch.get("spec_sha256") != spec.binding_sha256
                or launch.get("worker_request_sha256") != _file_hash(directory / "worker-request.json")
                or launch["identity_record"]["binding"] != {"nonce": request["nonce"], "attempt_id": directory.name,
                                                            "spec_sha256": spec.binding_sha256}):
            raise ValueError("worker attempt identity/nonce/spec mismatch")
        confirm_owned_attempt_exit(launch["identity_record"])
    index_sha = _file_hash(root / "dev_index.sqlite3")
    if verify_parent_record(source, root / "index-authority.json", root) != {
        "kind": "development_index", "run_root": str(root), "spec_sha256": spec.binding_sha256, "index_sha256": index_sha
    }:
        raise ValueError("authenticated development index mismatch")
    completed = {}
    recoveries = []
    with FullFitJournal.open_readonly(root / "journal", spec, source_root=source) as journal, tempfile.TemporaryDirectory(
        prefix="ordinary-resume-"
    ) as temporary:
        history = journal.attempts()
        latest = {attempt.unit: attempt for attempt in history}
        if any(attempt.status != "completed" for attempt in latest.values()) and (
            type(retry_reason) is not str or not retry_reason.strip()
        ):
            raise ValueError("explicit retry reason required for interrupted attempted work")
        authority = io.build_development_index(source, spec, Path(temporary) / "authority.sqlite3")
        declarations = {}
        for snapshot in spec.candidates:
            arms = build_reviewed_route_arms(snapshot, year_built_encoding=encoding)
            if snapshot.study is not None or snapshot.spec.model_family == "autogluon" or set(arms) != set(snapshot.arms):
                raise ValueError("resume supports only exact reviewed ordinary arms")
            declarations[snapshot.spec.experiment_id] = _build_candidate_declaration(snapshot, arms, spec)
        for attempt in history:
            experiment = attempt.unit.split("/", 1)[0]
            if attempt.status == "completed":
                if attempt.unit.endswith("/selector"):
                    from project.experiments.full_campaign_evidence import (
                        verify_completed_selector,
                    )
                    verify_completed_selector(source, root, spec, attempt, authority, index_sha)
                    model = latest.get(attempt.unit.rsplit("/", 1)[0] + "/model")
                    if model is None or model.status != "completed":
                        raise ValueError("partial selector/model recovery requires explicit fitted-state recovery; no selector refit")
                    continue
                if attempt.unit in completed:
                    raise ValueError("duplicate completed unit")
                completed[attempt.unit] = verify_completed_unit(source, root, spec, attempt,
                                                                 declarations[experiment], authority, index_sha)
            else:
                proof = verify_interrupted_exit(source, root, spec, attempt)
                recoveries.append((attempt, proof))
        admitted.verify_bindings(source, phase="resume")
    return completed, recoveries


def _persist_or_verify_json(path: Path, payload: dict) -> None:
    from project.experiments.full_snapshot_codec import parse_json

    if path.exists():
        if path.is_symlink() or parse_json(path.read_text()) != payload:
            raise ValueError("immutable existing transport artifact differs from independent recomputation")
    else:
        _exclusive_json(path, payload)


def run_full_campaign(
    admitted_spec: FullRunSpec,
    source_root: str | Path,
    run_root: str | Path,
    *,
    data_preflight: Any = None,
    capability: Any = None,
    journal: FullFitJournal | None = None,
    registry: Any | None = None,
    publish_completion: bool = True,
    _resume: bool = False,
    _retry_reason: str | None = None,
) -> dict[str, Any]:
    """Require genuine admission, fit serially, then publish verified ordinary completion.

    publish_completion=False is a transport-only diagnostic/test boundary; True
    does not grant permission and still requires the independent durable gate.
    """
    if type(admitted_spec) is not FullRunSpec:
        raise TypeError("admitted_spec must be a typed FullRunSpec")
    if type(publish_completion) is not bool:
        raise TypeError("publication selection must be an exact boolean; it never grants admission")

    if data_preflight is None:
        raise TypeError("authentic FullDataPreflight or AdmittedFullRun required")
    if isinstance(data_preflight, FullDataPreflight):
        raise TypeError("data evidence alone never launches: authentic AdmittedFullRun required")
    if type(data_preflight) is not AdmittedFullRun:
        raise TypeError("authentic exact AdmittedFullRun required")
    verify_admitted_full_run(source_root, data_preflight, spec=admitted_spec, phase="launch")
    verify_full_campaign_capability(source_root, capability)
    from project.experiments.full_campaign_capabilities import route_for_experiments
    if capability.route_id != route_for_experiments(tuple(
            snapshot.spec.experiment_id for snapshot in admitted_spec.candidates)):
        raise ValueError("capability does not match selected route")
    if _resume and capability.route_id in ("reviewed_full_hpo_transport_v1", "reviewed_autogluon_xt_transport_v1"):
        raise ValueError("unsupported recovery for selected route before mutation")

    if registry is not None:
        raise TypeError("fixed ordinary route does not accept caller registry substitution")
    source = _validate_managed_path(source_root, "source root", directory=True)
    root = _validate_managed_path(run_root, "run destination", directory=True)
    if root.exists() and not _resume:
        raise FileExistsError("fresh campaign destination already exists; resume is not yet available")
    if journal is not None:
        raise TypeError("fixed ordinary route owns its fresh journal")
    admitted_spec.verify_bindings(source)
    encoding, prov_rel, prov_sha = _encoding_provenance(source, admitted_spec)
    from project.experiments.full_campaign_evidence import (
        _existing_source_lock,
        _verify_prediction_union,
        resume_binding,
        sign_parent_record,
        verify_parent_record,
    )
    lock = _existing_source_lock(source) if _resume else CampaignLock(root, source_root=source)
    with lock:
        completed_units = {}
        recoveries = []
        if _resume:
            completed_units, recoveries = _verify_resume_state(source, root, data_preflight, _retry_reason)
            index_path = root / "dev_index.sqlite3"
        else:
            root.mkdir(parents=True, exist_ok=False)
            _exclusive_json(root / "spec.json", encode_snapshot(admitted_spec))
            _exclusive_json(root / "run-authority.json", sign_parent_record(source, resume_binding(data_preflight, root)))
            index_path = io.build_development_index(source, admitted_spec, root / "dev_index.sqlite3")
        index_sha256 = _file_hash(index_path)
        if not _resume:
            _exclusive_json(root / "index-authority.json", sign_parent_record(source, {
                "kind": "development_index", "run_root": str(root), "spec_sha256": admitted_spec.binding_sha256,
                "index_sha256": index_sha256}))

        index_connection, index_meta = io._open_bound_index(index_path, admitted_spec)
        index_connection.close()
        oof_schema = _oof_schema(index_meta)

        sem_sha = semantic_digest(encoding, prov_sha)

        worker_key = secrets.token_hex(32)

        fit_journal = FullFitJournal(
            root / "journal", admitted_spec, source_root=source
        )
        accountant = FullFitAccountant(fit_journal, admitted_spec.planned_units)
        accountant.retry_reason = _retry_reason
        receipts: list[dict[str, Any]] = []
        queue_stopped = False

        try:
            for previous, proof in recoveries:
                proof_path = root / "journal" / previous.artifact_directory / "recovery-exit.json"
                _persist_or_verify_json(proof_path, sign_parent_record(source, proof))
                if previous.status in ("running", "awaiting_validation"):
                    fit_journal.finish(previous.attempt_id, "interrupted", reason=_retry_reason)
            if _resume and not (root / "run_status.json").exists():
                _exclusive_json(root / "run_status.json", _campaign_status(data_preflight, [], "interrupted"))
            for snapshot in admitted_spec.candidates:
                exp_id = snapshot.spec.experiment_id
                if queue_stopped:
                    receipts.append({
                        "experiment_id": exp_id,
                        "status": "skipped",
                        "reason": "queue stopped after previous runtime failure",
                    })
                    continue

                arms = build_reviewed_route_arms(snapshot, year_built_encoding=encoding)
                if set(arms) != set(snapshot.arms):
                    raise ValueError("selected route requires exact resolved reviewed arms")
                declaration = _build_candidate_declaration(snapshot, arms, admitted_spec)

                oof_path = root / f"{exp_id}_oof.parquet"

                shard_paths: list[Path] = []
                candidate_failed = False
                arm_ids = tuple(snapshot.arms)

                for repeat in range(admitted_spec.repeats):
                    if candidate_failed:
                        break
                    for fold in range(admitted_spec.folds):
                        if candidate_failed:
                            break
                        if snapshot.study is not None:
                            from project.experiments.full_hpo_campaign import HpoPartitionContext, run_hpo_partition
                            context = HpoPartitionContext(source, root, admitted_spec, index_path, repeat, fold)
                            partition = run_hpo_partition(context, fit_journal, worker_key)
                            for name in snapshot.arms:
                                candidate = f"{exp_id}_{name}"
                                metadata = partition["candidates"][candidate]
                                if "fold_fit_config_sha256" in metadata:
                                    existing = declaration["candidates"][candidate]
                                    existing.pop("model_fit_config_sha256", None)
                                    existing.setdefault("fold_fit_config_sha256", {}).update(metadata["fold_fit_config_sha256"])
                                else:
                                    declaration["candidates"][candidate] = metadata
                                shard = Path(partition["shard_paths"][name])
                                _validate_shard(shard, index_path, admitted_spec, declaration,
                                                candidate, repeat, fold, index_sha256)
                                shard_paths.append(shard)
                            _exclusive_json(root / f"{exp_id}_r{repeat}_f{fold}_partition.json", partition["proof"])
                            continue
                        for arm_id in arm_ids:
                            work_id = f"{exp_id}/{arm_id}/r{repeat}/f{fold}/model"
                            if work_id in completed_units:
                                shard_paths.append(completed_units[work_id])
                                continue
                            attempt_name = f"{exp_id}_{arm_id}_r{repeat}_f{fold}_{secrets.token_hex(16)}"
                            attempt_dir = root / "attempts" / attempt_name
                            selector_unit = work_id.rsplit("/", 1)[0] + "/selector"
                            selector_attempt = []
                            accountant.open_units((selector_unit, work_id) if arm_id in snapshot.selector_arms else (work_id,))

                            payload = {
                                "version": 1,
                                "project_root": str(source),
                                "spec": encode_snapshot(admitted_spec),
                                "spec_sha256": admitted_spec.binding_sha256,
                                "index_path": str(index_path.resolve()),
                                "index_sha256": index_sha256,
                                "experiment_id": exp_id,
                                "arm_id": arm_id,
                                "repeat": repeat,
                                "fold": fold,
                                "year_built_encoding": encoding,
                                "provenance_path": str(prov_rel),
                                "provenance_sha256": prov_sha,
                                "semantic_sha256": sem_sha,
                                "work_id": work_id,
                                "attempt_id": attempt_name,
                                "output_path": "shard.parquet",
                                "parent_pid": os.getpid(),
                                "ack_timeout_seconds": 10.0,
                                "nonce": secrets.token_hex(32),
                            }
                            prepare_request = prepare_worker_request
                            target = "project.experiments.full_campaign_worker:run"
                            if exp_id == "dim_pca_vs_passthrough":
                                from project.experiments.full_rbf_svc_worker import BACKEND_TARGET, prepare_rbf_request
                                target, prepare_request = BACKEND_TARGET, prepare_rbf_request
                            elif exp_id == "model_autogluon_core_features":
                                from project.experiments.full_autogluon_svc_worker import BACKEND_TARGET, prepare_autogluon_request
                                target, prepare_request = BACKEND_TARGET, prepare_autogluon_request
                            payload = prepare_request(attempt_dir, payload, worker_key)

                            def _on_started(identity_record: dict[str, Any], directory: Path = attempt_dir,
                                            unit: str = work_id) -> None:
                                _exclusive_json(directory / "worker-launch.json", sign_parent_record(source, {
                                    "kind": "worker_launch", "run_root": str(root), "spec_sha256": admitted_spec.binding_sha256,
                                    "unit": unit, "worker_request_sha256": _file_hash(directory / "worker-request.json"),
                                    "identity_record": identity_record}))

                            def _on_debit(phase: str, event: dict[str, Any], expected_unit: str = work_id,
                                          directory: Path = attempt_dir) -> None:
                                expected_phase_unit = expected_unit.rsplit("/", 1)[0] + "/" + phase
                                if phase not in ("model", "selector") or event["unit"] != expected_phase_unit:
                                    raise ValueError("debit does not match exact planned unit")
                                def prepare_attempt(record: FullAttemptRecord) -> None:
                                    launch = verify_parent_record(source, directory / "worker-launch.json", root)
                                    launch["journal_attempt_id"] = record.attempt_id
                                    journal_directory = root / "journal" / record.artifact_directory
                                    _exclusive_json(journal_directory / "worker-launch.json", sign_parent_record(source, launch))
                                    _copy_exclusive(directory / "worker-request.json", journal_directory / "worker-request.json")

                                # Persist the association BEFORE the journal's begin event:
                                # a killed parent can never leave a debited unit without identity.
                                accountant.before_fit(prepare_attempt=prepare_attempt)

                            def _on_selector_success(event: dict[str, Any], directory: Path = attempt_dir,
                                                     request: dict = payload, records: list = selector_attempt) -> None:
                                from project.experiments.full_campaign_evidence import (
                                    verify_selector_outcome,
                                )

                                if accountant._active is None or accountant._active.unit != event["unit"]:
                                    raise ValueError("selector success has no matching durable debit")
                                outcome_file = directory / "selector-outcome.json"
                                verify_selector_outcome(outcome_file, index_path, admitted_spec, request)
                                journal_directory = root / "journal" / accountant.active_directory
                                target = journal_directory / "selector-outcome.json"
                                _copy_exclusive(outcome_file, target)
                                paths = (target, journal_directory / "worker-request.json", journal_directory / "worker-launch.json")
                                authentication = journal_directory / "parent-auth.json"
                                record = accountant._active
                                _exclusive_json(authentication, sign_parent_record(source, {
                                    "kind": "completed_selector", "run_root": str(root), "spec_sha256": admitted_spec.binding_sha256,
                                    "journal_attempt_id": record.attempt_id, "unit": record.unit,
                                    "artifacts": {p.name: _file_hash(p) for p in paths}}))
                                accountant.finish_active("completed", evidence=tuple(
                                    ArtifactEvidence(str(p.relative_to(root / "journal")), _file_hash(p)) for p in (*paths, authentication)))
                                records.append(record)

                            service = ParentLifecycleService(
                                attempt_dir,
                                worker_key,
                                on_debit=_on_debit,
                                on_selector_success=_on_selector_success,
                                on_shard_closed=lambda event: None,
                            )

                            outcome = run_work_unit(
                                unit=work_id,
                                target=target,
                                work_dir=attempt_dir,
                                expected_artifacts=("shard.parquet",),
                                poll_callback=service.service_once,
                                environment_binding={"FULL_CAMPAIGN_WORKER_KEY": worker_key},
                                attempt_binding={"nonce": payload["nonce"], "attempt_id": payload["attempt_id"],
                                                 "spec_sha256": admitted_spec.binding_sha256},
                                on_started=_on_started,
                            )

                            if outcome.status != "awaiting_validation" or outcome.exit_code != 0 or outcome.reason:
                                if accountant._active is not None:
                                    accountant.finish_active("failed", reason=outcome.reason)
                                receipts.append({
                                    "experiment_id": exp_id,
                                    "status": "failed",
                                    "reason": outcome.reason or f"worker exited with status {outcome.status}",
                                })
                                candidate_failed = True
                                queue_stopped = True
                                break

                            shard_file = attempt_dir / "shard.parquet"
                            if service.shard_evidence is None or len(outcome.artifacts) != 1:
                                raise ValueError("worker lacks exact closed shard/supervisor evidence")
                            validated = _validate_shard(
                                shard_file, index_path, admitted_spec, declaration,
                                f"{exp_id}_{arm_id}", repeat, fold, index_sha256)
                            if (outcome.artifacts[0].path != "shard.parquet"
                                    or outcome.artifacts[0].sha256 != validated["sha256"]
                                    or service.shard_evidence["sha256"] != validated["sha256"]
                                    or service.shard_evidence["num_rows"] != validated["rows"]):
                                raise ValueError("parent validation differs from supervisor/closed shard")
                            attempt_journal_dir = root / "journal" / accountant.active_directory
                            dest_shard = attempt_journal_dir / "shard.parquet"
                            _copy_exclusive(shard_file, dest_shard)
                            shard_sha = _file_hash(dest_shard)
                            if shard_sha != validated["sha256"]:
                                raise ValueError("copied shard hash changed")
                            validation_path = attempt_journal_dir / "parent_validation.json"
                            _exclusive_json(validation_path, validated)
                            supervisor_path = attempt_journal_dir / "supervisor.json"
                            request_copy = attempt_journal_dir / "worker-request.json"
                            # The immutable request was persisted with the debit before its ACK.
                            _exclusive_json(supervisor_path, {"unit": work_id, "status": outcome.status,
                                                             "exit_code": outcome.exit_code,
                                                             "spec_sha256": admitted_spec.binding_sha256,
                                                             "nonce": payload["nonce"],
                                                             "attempt_id": payload["attempt_id"],
                                                             "request_sha256": _file_hash(request_copy),
                                                             "shard_sha256": shard_sha,
                                                             "index_sha256": index_sha256,
                                                             "observation": outcome.observation})
                            launch_copy = attempt_journal_dir / "worker-launch.json"
                            authenticated_paths = (dest_shard, validation_path, supervisor_path, request_copy, launch_copy)
                            if exp_id == "model_autogluon_core_features":
                                from project.experiments.full_autogluon_svc_worker import verify_autogluon_unit
                                child_proof = verify_autogluon_unit(attempt_dir, worker_key, expected_request=payload)
                                # Child artifacts remain in the immutable run-owned attempt;
                                # journal sealing binds them and independently recomputed proof.
                                child_paths = tuple(path for path in attempt_dir.iterdir()
                                                    if path.is_file() and path.name not in ("worker-request.json", "worker-launch.json"))
                                child_record = attempt_journal_dir / "autogluon-proof.json"
                                _exclusive_json(child_record, sign_parent_record(source, {
                                    "kind": "autogluon_child", "directory": str(attempt_dir),
                                    "proof": child_proof, "artifacts": {path.name: _file_hash(path) for path in child_paths}}))
                                authenticated_paths += (child_record,)
                            if arm_id in snapshot.selector_arms:
                                if len(selector_attempt) != 1:
                                    raise ValueError("model fit lacks prior verified fitted selector completion")
                                selector_link = attempt_journal_dir / "selector-link.json"
                                _exclusive_json(selector_link, sign_parent_record(source, {
                                    "kind": "selector_model_link", "spec_sha256": admitted_spec.binding_sha256,
                                    "selector_attempt_id": selector_attempt[0].attempt_id,
                                    "model_attempt_id": accountant._active.attempt_id, "nonce": payload["nonce"]}))
                                authenticated_paths += (selector_link,)
                            auth_path = attempt_journal_dir / "parent-auth.json"
                            _exclusive_json(auth_path, sign_parent_record(source, {
                                "kind": "completed_unit", "run_root": str(root), "spec_sha256": admitted_spec.binding_sha256,
                                "journal_attempt_id": accountant._active.attempt_id, "unit": work_id,
                                "artifacts": {path.name: _file_hash(path) for path in authenticated_paths}}))
                            ev = tuple(ArtifactEvidence(str(path.relative_to(root / "journal")), _file_hash(path))
                                       for path in (*authenticated_paths, auth_path))
                            accountant.finish_active("completed", evidence=ev)
                            shard_paths.append(dest_shard)

                if not candidate_failed:
                    if oof_path.exists():
                        _validate_managed_path(oof_path, "existing OOF", root=root)
                        _verify_prediction_union(oof_path, shard_paths)
                    else:
                        write_candidate_oof_batches(oof_path, _stream_shards(shard_paths), oof_schema)
                    expected = io.iter_full_expected_batches(index_path, admitted_spec)
                    val_payload = validate_full_candidate_oof(oof_path, expected, declaration)

                    # Actual parent coverage/metrics were computed, but a
                    # self-hash cannot grant independent comparative permission.
                    _persist_or_verify_json(root / f"{exp_id}_transport_validation.json", val_payload)
                    _persist_or_verify_json(root / f"{exp_id}_declaration.json", declaration)
                    _persist_or_verify_json(root / f"{exp_id}_source_lineage.json", _source_lineage(snapshot))
                    receipts.append({
                        "experiment_id": exp_id,
                        "status": "awaiting_validation",
                        "oof_path": str(oof_path),
                        "oof_sha256": _file_hash(oof_path),
                        "comparative_completion_available": False,
                    })
        except BaseException as exc:
            terminal_status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            if accountant._active is not None:
                accountant.finish_active(terminal_status,
                                          reason=f"{type(exc).__name__}: {exc}")
            if "exp_id" in locals() and not any(receipt["experiment_id"] == exp_id for receipt in receipts):
                receipts.append({"experiment_id": exp_id, "status": terminal_status,
                                 "reason": f"{type(exc).__name__}: {exc}"})
            for snapshot in admitted_spec.candidates:
                if not any(receipt["experiment_id"] == snapshot.spec.experiment_id for receipt in receipts):
                    receipts.append({"experiment_id": snapshot.spec.experiment_id, "status": "skipped",
                                     "reason": "queue stopped before experiment started"})
            status_path = root / "run_status.json" if not _resume else root / f"resume-failure-{secrets.token_hex(16)}.json"
            _exclusive_json(status_path, _campaign_status(data_preflight, receipts, terminal_status))
            raise
        finally:
            fit_journal.close()

        final_status = "failed" if any(receipt["status"] == "failed" for receipt in receipts) else "awaiting_validation"
        result = _campaign_status(data_preflight, receipts, final_status)
        if _resume:
            if final_status == "awaiting_validation":
                _persist_or_verify_json(root / "resumed-transport.json", sign_parent_record(source, {
                    "kind": "resumed_transport", "status": result,
                    "journal_sha256": _file_hash(root / "journal/full_fit.sqlite3")}))
        else:
            _exclusive_json(root / "run_status.json", result)
    if publish_completion and final_status == "awaiting_validation":
        from project.experiments.full_campaign_evidence import (
            issue_ordinary_campaign_completion,
        )
        return issue_ordinary_campaign_completion(source, root, data_preflight, root / "completion")
    return result
