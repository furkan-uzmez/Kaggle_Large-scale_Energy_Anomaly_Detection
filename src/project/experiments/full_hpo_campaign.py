"""Dedicated parent HPO partition execution and independent scientific verification."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from project.experiments import full_candidate_io as io
from project.experiments import full_hpo_worker as backend
from project.experiments.candidate_contracts import _validate_managed_path
from project.experiments.candidate_dispatch import CandidateArm
from project.experiments.full_campaign_evidence import sign_parent_record, verify_parent_record
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_campaign_runtime import confirm_owned_attempt_exit, run_work_unit
from project.experiments.full_campaign_worker import _build_batch_table, _oof_schema
from project.experiments.full_hpo_lifecycle import EXPERIMENT, HpoLifecyclePlan, _authenticate
from project.experiments.full_nested_tuning import (
    MAX_BATCH_ROWS,
    _verify_fold_evidence,
    _verify_refit_evidence,
    compute_fit_config_sha256,
    full_hpo_study_spec,
    partition_planned_units,
)
from project.experiments.full_run_contracts import FullFitJournal, FullRunSpec, _file_hash
from project.experiments.full_snapshot_codec import canonical_json, encode_snapshot, exact_fields, parse_json
from project.experiments.inner_cv import build_inner_folds
from project.experiments.nested_tuning import DEFAULT_HGB_PARAMS, _generate_trial_params
from project.features.registry import build_lead_baseline_registry


@dataclass(frozen=True)
class HpoPartitionContext:
    source_root: Path
    run_root: Path
    spec: FullRunSpec
    index_path: Path
    repeat: int
    fold: int

    def __post_init__(self) -> None:
        for name in ("source_root", "run_root", "index_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())
        HpoLifecyclePlan(self.spec, self.repeat, self.fold, "context", "context")


def _authority(context: HpoPartitionContext) -> str:
    _validate_managed_path(context.source_root, "HPO source root", directory=True)
    _validate_managed_path(context.run_root, "HPO run root", directory=True)
    _validate_managed_path(context.index_path, "HPO development index", root=context.run_root)
    context.spec.verify_bindings(context.source_root)
    if current_environment() != context.spec.environment:
        raise ValueError("HPO environment authority changed")
    return io.verify_development_index(context.index_path, context.spec)


def _journal_authority(context: HpoPartitionContext, journal: FullFitJournal) -> None:
    valid_roots = (context.run_root, context.run_root / "journal")
    if (type(journal) is not FullFitJournal or journal.root not in valid_roots
            or journal.source_root != context.source_root
            or journal.spec.binding_sha256 != context.spec.binding_sha256):
        raise ValueError("exact parent HPO journal context required")


def prepare_hpo_partition(context: HpoPartitionContext, journal: FullFitJournal,
                          key: str):
    """Authenticate the source/index/environment before allocating any fit."""
    _authority(context)
    _journal_authority(context, journal)
    units = partition_planned_units(EXPERIMENT, context.repeat, context.fold, full_hpo_study_spec())
    if any(record.unit in units for record in journal.attempts()):
        raise ValueError("HPO resume requires verified existing partition proof; partial/reused session rejected")
    training = io.load_full_training_partition(context.index_path, context.spec, context.repeat, context.fold)
    directory = journal.root / "hpo_sessions" / secrets.token_hex(16)
    request = backend.prepare_hpo_request(directory, journal, training, key, index_path=context.index_path)
    service = backend.make_parent_hpo_service(directory, journal, key)
    return directory, request, service


def _candidate_metadata(context: HpoPartitionContext, selected_params) -> dict[str, Any]:
    candidates = {}
    for arm, params in (("control", DEFAULT_HGB_PARAMS), ("challenger", selected_params)):
        candidate_id = f"{EXPERIMENT}_{arm}"
        value = {"recipe_id": EXPERIMENT, "model_family": "hist_gradient_boosting",
                 "model_config_id": f"{candidate_id}_config", "experiment_id": EXPERIMENT,
                 "arm_id": arm, "score_kind": "positive_probability", "positive_label": 1}
        digest = compute_fit_config_sha256(EXPERIMENT, arm, params)
        if arm == "challenger":
            value["fold_fit_config_sha256"] = {f"{context.repeat}:{context.fold}": digest}
        else:
            value["model_fit_config_sha256"] = digest
        candidates[candidate_id] = value
    return candidates


def run_hpo_partition(context: HpoPartitionContext, journal: FullFitJournal,
                      key: str) -> dict[str, Any]:
    """Supervise exactly one dedicated 26-fit worker and parent-authenticate proof."""
    directory, request, service = prepare_hpo_partition(context, journal, key)
    try:
        outcome = run_work_unit(
            request["work_id"], "project.experiments.full_hpo_worker:run", directory,
            journal=journal, expected_artifacts=("hpo-result.json", "outer-control.parquet", "outer-challenger.parquet"),
            spec_binding=context.spec.binding_sha256,
            environment_binding={"FULL_HPO_WORKER_KEY": key}, poll_callback=service.service_once)
        if (outcome.status != "awaiting_validation" or outcome.exit_code != 0
                or not outcome.observation.get("cleanup_confirmed", False) or outcome.reason is not None):
            raise RuntimeError(f"HPO partition worker failed: {outcome.status}: {outcome.reason}")
        request_envelope = backend._read_message(directory / "hpo-request.json")
        result_envelope = backend._read_message(directory / "hpo-result.json")
        if canonical_json(_authenticate(request_envelope, key)) != canonical_json(request):
            raise ValueError("HPO original authenticated request changed")
        result = _authenticate(result_envelope, key)
        payload = {
            "version": 1, "kind": "hpo_partition_parent", "spec_sha256": context.spec.binding_sha256,
            "source_root": str(context.source_root), "run_root": str(context.run_root),
            "index_path": str(context.index_path), "index_integrity_sha256": _authority(context),
            "repeat": context.repeat, "fold": context.fold, "session_path": str(directory),
            "request": request_envelope, "result": result_envelope,
            "request_file_sha256": _file_hash(directory / "hpo-request.json"),
            "result_file_sha256": _file_hash(directory / "hpo-result.json"),
            "study_sha256": _file_hash(_study_path(context, result, journal_root=journal.root)),
            "attempt_ids": [record.attempt_id for record in _partition_records(context, journal)],
            "outcome": {"status": outcome.status, "exit_code": outcome.exit_code,
                        "cleanup_confirmed": True},
        }
        _verify_partition_payload(context, payload, journal)
        proof_path = directory / "hpo-partition-proof.json"
        backend._publish(proof_path, sign_parent_record(context.source_root, payload))
        proof = {"path": str(proof_path), "sha256": _file_hash(proof_path)}
        verified = verify_hpo_partition(context, proof, journal=journal)
        return {**verified, "outcome": payload["outcome"]}
    except BaseException as error:
        identity_path = directory / "worker-identity.json"
        if identity_path.exists():
            confirm_owned_attempt_exit(parse_json(identity_path.read_text()))
        service.lifecycle.terminalize_owned("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                            reason=f"{type(error).__name__}: {error}")
        raise


def _study_path(context: HpoPartitionContext, result: dict[str, Any], *, journal_root: Path | None = None) -> Path:
    base = journal_root if journal_root is not None else (
        context.run_root / "journal" if (context.run_root / "journal").exists() else context.run_root
    )
    expected = base / "hpo_studies" / f"r{context.repeat}_f{context.fold}" / f"full_study_r{context.repeat}_f{context.fold}.sqlite3"
    if result.get("study_db_path") != str(expected):
        alt = context.run_root / "hpo_studies" / f"r{context.repeat}_f{context.fold}" / f"full_study_r{context.repeat}_f{context.fold}.sqlite3"
        if result.get("study_db_path") == str(alt):
            expected = alt
        else:
            raise ValueError("HPO study path identity mismatch")
    return _validate_managed_path(expected, "HPO study", root=context.run_root)

def _partition_records(context: HpoPartitionContext, journal: FullFitJournal):
    units = partition_planned_units(EXPERIMENT, context.repeat, context.fold, full_hpo_study_spec())
    journal.verify_completed()
    history = [record for record in journal.attempts() if record.unit in units]
    if len(history) != 26 or any(record.status != "completed" for record in history):
        raise ValueError("exactly26 completed HPO partition fits required; retries/resume unsupported")
    records = {record.unit: record for record in history}
    if set(records) != set(units):
        raise ValueError("all26 actual HPO journal units required")
    return tuple(records[unit] for unit in units)


def _frame_digest(frame) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(frame), MAX_BATCH_ROWS):
        digest.update(pd.util.hash_pandas_object(frame.iloc[start:start + MAX_BATCH_ROWS], index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _verify_study(context, request, result, records, training, *, journal_root: Path | None = None) -> None:
    base = journal_root if journal_root is not None else (
        context.run_root / "journal" if (context.run_root / "journal").exists() else context.run_root
    )
    path = _study_path(context, result, journal_root=base)
    folds = build_inner_folds(training.y_train, training.groups_train, n_splits=3, seed=42)
    membership = hashlib.sha256()
    for train, valid in folds:
        for positions in (train, valid):
            membership.update(len(positions).to_bytes(8, "big"))
            membership.update(np.asarray(positions, dtype="<i8").tobytes())
    if result["inner_membership_digest"] != membership.hexdigest():
        raise ValueError("HPO inner membership digest mismatch")
    plan = HpoLifecyclePlan(context.spec, context.repeat, context.fold, request["work_id"], request["nonce"])
    _verify_refit_evidence(base, records[0], "default_refit", DEFAULT_HGB_PARAMS, len(training.X_train))
    trials = _generate_trial_params(full_hpo_study_spec())
    scores = []
    expected_fold_rows = []
    for trial, params in enumerate(trials):
        fold_scores = []
        for inner, (_, valid) in enumerate(folds):
            record = records[1 + trial * 3 + inner]
            if record.selected_parameters != plan.inner(trial, inner).parameters:
                raise ValueError("HPO canonical sampled fit parameters mismatch")
            score = _verify_fold_evidence(base, record, valid, training.y_train.to_numpy(), training)
            fold_scores.append(score)
            expected_fold_rows.append((trial, inner, score, "completed", None, record.attempt_id,
                                       record.evidence[1].sha256, record.evidence[0].sha256))
        scores.append(math.fsum(fold_scores) / 3)
    winner = min(range(8), key=lambda trial: (-scores[trial], trial))
    if (type(result["selected_trial_id"]) is not int or result["selected_trial_id"] != winner
            or canonical_json(result["selected_params"]) != canonical_json(tuple(sorted(trials[winner].items())))):
        raise ValueError("HPO winner differs from sealed inner scores only")
    _verify_refit_evidence(base, records[-1], "selected_refit", trials[winner], len(training.X_train))
    key = f"full_study_r{context.repeat}_f{context.fold}"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        study_rows = connection.execute(
            "SELECT study_key, repeat, fold, study_manifest_hash, manifest_payload_json, status, "
            "inner_membership_digest, training_rows_count, n_trials, inner_folds, seed, planned_model_fits, "
            "selected_trial_id, selected_params_json, default_params_json, failure_reason FROM study").fetchall()
        if len(study_rows) != 1:
            raise ValueError("one exact HPO study required")
        row = study_rows[0]
        expected = (key, context.repeat, context.fold, result["study_manifest_hash"], row[4], "completed",
                    membership.hexdigest(), len(training.X_train), 8, 3, 42, 26, winner,
                    json.dumps(trials[winner], sort_keys=True), json.dumps(DEFAULT_HGB_PARAMS, sort_keys=True), None)
        if row != expected or hashlib.sha256(row[4].encode()).hexdigest() != result["study_manifest_hash"]:
            raise ValueError("HPO study/selection/manifest identity mismatch")
        stored_study_spec = connection.execute("SELECT study_spec_json FROM study").fetchone()[0]
        if canonical_json(parse_json(stored_study_spec)) != canonical_json(asdict(full_hpo_study_spec())):
            raise ValueError("HPO SQL frozen study specification mismatch")
        manifest = parse_json(row[4])
        snapshot = next(item for item in context.spec.candidates if item.spec.experiment_id == EXPERIMENT)
        expected_manifest = {
            "source_root": str(context.source_root), "spec_sha256": context.spec.binding_sha256,
            "sources": [asdict(binding) for binding in context.spec.sources], "candidate_spec": asdict(snapshot),
            "environment": dict(context.spec.environment), "resource_policy": asdict(context.spec.resource_policy),
            "feature_encoding_digest": {binding.path: binding.sha256 for binding in context.spec.sources if binding.role == "encoding"},
            "training_digests": {"X": _frame_digest(training.X_train), "y": _frame_digest(training.y_train), "groups": _frame_digest(training.groups_train)},
            "columns_info": [{"name": name, "dtype": str(dtype)} for name, dtype in training.X_train.dtypes.items()],
            "study_spec": asdict(full_hpo_study_spec()), "inner_membership_digest": membership.hexdigest(),
            "repeat": context.repeat, "fold": context.fold, "deterministic_trials": trials,
        }
        if canonical_json(manifest) != canonical_json(expected_manifest):
            raise ValueError("HPO manifest training/source/study authority mismatch")
        trial_rows = connection.execute("SELECT study_key, trial_id, params_json, params_hash, status, mean_score, failure_reason FROM trial ORDER BY trial_id").fetchall()
        expected_trials = [(key, trial, json.dumps(params, sort_keys=True),
                            hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
                            "completed", scores[trial], None) for trial, params in enumerate(trials)]
        if trial_rows != expected_trials:
            raise ValueError("HPO SQL trial scores differ from sealed inner OOF")
        fold_rows = connection.execute("SELECT trial_id, fold_index, score, status, failure_reason, attempt_id, evidence_json_sha256, parquet_sha256 FROM fold_score WHERE study_key=? ORDER BY trial_id, fold_index", (key,)).fetchall()
        if fold_rows != expected_fold_rows or connection.execute("SELECT COUNT(*) FROM fold_score").fetchone()[0] != 24:
            raise ValueError("HPO SQL fold scores differ from actual sealed journal")


def _verify_outer(context, directory, result, candidates) -> dict[str, str]:
    connection, meta = io._open_bound_index(context.index_path, context.spec)
    connection.close()
    schema = _oof_schema(meta)
    candidate = next(item.spec for item in context.spec.candidates if item.spec.experiment_id == EXPERIMENT)
    if set(result["shards"]) != {"control", "challenger"}:
        raise ValueError("both genuine outer HPO shards required")
    paths = {}
    for arm_id in ("control", "challenger"):
        item = exact_fields(result["shards"][arm_id], {"path", "sha256", "rows", "fit_config_sha256"})
        if item["path"] != f"outer-{arm_id}.parquet":
            raise ValueError("canonical HPO shard path required")
        path = _validate_managed_path(directory / item["path"], "HPO outer shard", root=directory)
        if _file_hash(path) != item["sha256"]:
            raise ValueError("HPO outer shard hash mismatch")
        metadata = candidates[f"{EXPERIMENT}_{arm_id}"]
        fit_hash = (metadata["model_fit_config_sha256"] if arm_id == "control" else
                    metadata["fold_fit_config_sha256"][f"{context.repeat}:{context.fold}"])
        if item["fit_config_sha256"] != fit_hash:
            raise ValueError("HPO outer fit configuration mismatch")
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(schema, check_metadata=True):
            raise ValueError("canonical HPO outer schema required")
        actual = iter(parquet.iter_batches(batch_size=MAX_BATCH_ROWS))
        count = 0
        arm = CandidateArm(f"{EXPERIMENT}_{arm_id}", arm_id, candidate.feature_pool,
                           object(), build_lead_baseline_registry(), "positive_probability")
        for expected in io.iter_full_validation_batches(context.index_path, context.spec, context.repeat, context.fold):
            batch = next(actual, None)
            if batch is None or batch.num_rows != len(expected["target"]):
                raise ValueError("HPO outer index/label coverage mismatch")
            reference = _build_batch_table(expected, np.zeros(batch.num_rows), candidate, arm, context.spec,
                                           context.repeat, context.fold, fit_hash, schema)
            for name in schema.names:
                if name != "prediction" and not batch.column(schema.get_field_index(name)).equals(reference[name].combine_chunks()):
                    raise ValueError(f"HPO outer expected index/labels/lineage mismatch: {name}")
            predictions = batch.column(schema.get_field_index("prediction")).to_numpy()
            if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
                raise ValueError("HPO outer probability must be finite within [0,1]")
            count += batch.num_rows
        if next(actual, None) is not None or type(item["rows"]) is not int or item["rows"] != count or count == 0:
            raise ValueError("HPO outer shard exact row count mismatch")
        paths[arm_id] = str(path)
    return paths


def _verify_partition_payload(context, payload, journal) -> dict[str, Any]:
    exact_fields(payload, {"version", "kind", "spec_sha256", "source_root", "run_root", "index_path",
                           "index_integrity_sha256", "repeat", "fold", "session_path", "request", "result",
                           "request_file_sha256", "result_file_sha256", "study_sha256", "attempt_ids", "outcome"})
    if (type(payload["version"]) is not int or payload["version"] != 1 or payload["kind"] != "hpo_partition_parent"
            or payload["spec_sha256"] != context.spec.binding_sha256
            or payload["source_root"] != str(context.source_root) or payload["run_root"] != str(context.run_root)
            or payload["index_path"] != str(context.index_path)
            or type(payload["repeat"]) is not int or payload["repeat"] != context.repeat
            or type(payload["fold"]) is not int or payload["fold"] != context.fold
            or payload["index_integrity_sha256"] != _authority(context)
            or payload["outcome"] != {"status": "awaiting_validation", "exit_code": 0, "cleanup_confirmed": True}):
        raise ValueError("HPO partition parent authority mismatch")
    _journal_authority(context, journal)
    directory = _validate_managed_path(payload["session_path"], "HPO proof session", root=context.run_root, directory=True)
    request = exact_fields(payload["request"], {"payload", "signature"})["payload"]
    result = exact_fields(payload["result"], {"payload", "signature"})["payload"]
    exact_fields(request, backend.REQUEST_FIELDS)
    exact_fields(result, {"version", "kind", "work_id", "session_id", "nonce", "spec_sha256", "request_sha256",
                          "selected_trial_id", "selected_params", "study_manifest_hash", "inner_membership_digest", "study_db_path", "shards"})
    plan = HpoLifecyclePlan(context.spec, context.repeat, context.fold, request["work_id"], request["nonce"])
    if ((directory != journal.root / "hpo_sessions" / request["session_id"]
            and directory != context.run_root / "hpo_sessions" / request["session_id"])
            or request["repeat"] != context.repeat or request["fold"] != context.fold
            or request["source_root"] != str(context.source_root)
            or request["run_root"] not in (str(context.run_root), str(journal.root))
            or request["spec_sha256"] != context.spec.binding_sha256
            or canonical_json(request["spec"]) != canonical_json(encode_snapshot(context.spec))
            or request["index_path"] != str(context.index_path)
            or result["version"] != 1 or result["kind"] != "hpo_result"
            or any(result[name] != request[name] for name in ("work_id", "session_id", "nonce", "spec_sha256"))
            or result["request_sha256"] != hashlib.sha256(canonical_json(request).encode()).hexdigest()):
        raise ValueError("HPO signed request/result context mismatch")
    for name in ("request", "result"):
        path = directory / f"hpo-{name}.json"
        if (_file_hash(path) != payload[f"{name}_file_sha256"]
                or canonical_json(backend._read_message(path)) != canonical_json(payload[name])):
            raise ValueError("HPO authenticated request/result file changed")
    backend._verify_bound_index(directory, request, context.spec)
    records = _partition_records(context, journal)
    if payload["attempt_ids"] != [record.attempt_id for record in records]:
        raise ValueError("HPO actual journal attempt association mismatch")
    if _file_hash(_study_path(context, result, journal_root=journal.root)) != payload["study_sha256"]:
        raise ValueError("HPO study hash changed")
    training = io.load_full_training_partition(context.index_path, context.spec, context.repeat, context.fold)
    backend._require_training_equal(backend.load_hpo_training(directory, request), training)
    _verify_study(context, request, result, records, training, journal_root=journal.root)
    candidates = _candidate_metadata(context, result["selected_params"])
    paths = _verify_outer(context, directory, result, candidates)
    return {"shard_paths": paths, "candidates": candidates, "request": request, "result": result,
            "selected_refit_unit": plan.selected(result["selected_trial_id"]).unit}


def verify_hpo_partition(context: HpoPartitionContext, proof: dict[str, Any], *,
                         journal: FullFitJournal | None = None) -> dict[str, Any]:
    """Fresh-process completion/resume verification; no persisted worker key."""
    exact_fields(proof, {"path", "sha256"})
    path = _validate_managed_path(proof["path"], "HPO parent proof", root=context.run_root)
    if path.name != "hpo-partition-proof.json" or _file_hash(path) != proof["sha256"]:
        raise ValueError("HPO parent partition proof hash mismatch")
    payload = verify_parent_record(context.source_root, path, context.run_root)
    if journal is not None:
        verified = _verify_partition_payload(context, payload, journal)
    else:
        journal_root = context.run_root / "journal" if (context.run_root / "journal").exists() else context.run_root
        with FullFitJournal.open_readonly(journal_root, context.spec, source_root=context.source_root) as readonly:
            verified = _verify_partition_payload(context, payload, readonly)
    return {**verified, "proof": proof, "outcome": payload["outcome"]}
