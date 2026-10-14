"""Bounded full-campaign nested HGB tuning lane with FullFitJournal accounting.

Implements sequential HGB hyperparameter optimization for full development
campaigns under approved specifications:
- Finite ParameterSampler random search (seed 42, T=8 trials, K=3 inner folds)
- Durable journal debit before estimator construction
- Group-safe inner validation splits (no leakage across folds)
- Streamed inner validation predictions (chunked Parquet + compact metadata)
- Memory cleanup: estimator freed immediately after each trial fold
- Deterministic tie breaking (highest mean ROC-AUC, lowest trial_id on tie)
- Exact single outer default refit (control) and single outer selected refit (challenger)
- Total accounting: 12 * (T * K + 2) model fits, 12 * T trials across 12 outer partitions
- Strict input verification: rejects FoldPartition and held-out validation fields
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline

from project.experiments.candidate_contracts import (
    STUDY_SPACE,
    BlockedExperimentError,
    FoldPartition,
    StudySpec,
    TrainingPartition,
    _validate_managed_path,
    validate_partition,
)
from project.experiments.full_input_schema import canonical_feature_order
from project.experiments.full_run_contracts import (
    ArtifactEvidence,
    FullAttemptRecord,
    FullCandidateSnapshot,
    FullFitJournal,
    FullRunSpec,
    _file_hash,
)
from project.experiments.full_snapshot_codec import (
    canonical_json,
    exact_fields,
    parse_json,
)
from project.experiments.inner_cv import build_inner_folds
from project.experiments.nested_tuning import (
    CORE_FEATURE_POOL,
    DEFAULT_HGB_PARAMS,
    _extract_pos_proba,
    _generate_trial_params,
    _validate_constructed_arm,
    build_hgb_tuning_arm,
    get_hgb_signature_defaults,
)
from project.features.registry import build_lead_baseline_registry

if TYPE_CHECKING:
    from project.experiments.full_hpo_lifecycle import WorkerHpoLifecycle

FULL_HPO_DEFAULT_T: int = 8
FULL_HPO_DEFAULT_K: int = 3
FULL_HPO_DEFAULT_SEED: int = 42
MAX_BATCH_ROWS = 64000

FULL_HPO_PARAMETER_LISTS: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("l2_regularization", (0.0, 1.0)),
    ("learning_rate", (0.03, 0.1)),
    ("max_iter", (100, 200)),
    ("max_leaf_nodes", (15, 31)),
)


def full_hpo_study_spec(
    n_trials: int = FULL_HPO_DEFAULT_T,
    inner_folds: int = FULL_HPO_DEFAULT_K,
    seed: int = FULL_HPO_DEFAULT_SEED,
) -> StudySpec:
    """Construct the approved StudySpec for full campaign HGB tuning."""
    return StudySpec(
        n_trials=n_trials,
        inner_folds=inner_folds,
        seed=seed,
        parameter_lists=FULL_HPO_PARAMETER_LISTS,
    )


def required_full_fits(n_trials: int, inner_folds: int, outer_partitions: int = 12) -> int:
    """Calculate total model fits: outer_partitions * (n_trials * inner_folds + 2)."""
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError("n_trials must be a positive integer; booleans are invalid")
    if type(inner_folds) is not int or not (2 <= inner_folds <= 5):
        raise ValueError("inner_folds must be an integer in [2, 5]; booleans are invalid")
    if type(outer_partitions) is not int or not (1 <= outer_partitions <= 12):
        raise ValueError("outer_partitions must be an integer in [1, 12]; booleans are invalid")
    return outer_partitions * (n_trials * inner_folds + 2)


def required_full_trials(n_trials: int, outer_partitions: int = 12) -> int:
    """Calculate total trial count: outer_partitions * n_trials."""
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError("n_trials must be a positive integer; booleans are invalid")
    if type(outer_partitions) is not int or not (1 <= outer_partitions <= 12):
        raise ValueError("outer_partitions must be an integer in [1, 12]; booleans are invalid")
    return outer_partitions * n_trials


def partition_planned_units(
    experiment_id: str,
    repeat: int,
    fold: int,
    study_spec: StudySpec,
) -> tuple[str, ...]:
    """Compute the ordered planned journal units for one outer partition."""
    units: list[str] = [
        f"{experiment_id}/control/r{repeat}/f{fold}/default_refit",
    ]
    for trial in range(study_spec.n_trials):
        for inner in range(study_spec.inner_folds):
            units.append(
                f"{experiment_id}/challenger/r{repeat}/f{fold}/inner/trial{trial}/fold{inner}"
            )
    units.append(
        f"{experiment_id}/challenger/r{repeat}/f{fold}/selected_refit"
    )
    return tuple(units)


def compute_fit_config_sha256(
    experiment_id: str,
    arm_id: str,
    params: dict[str, Any] | tuple[tuple[str, object], ...],
) -> str:
    """Compute deterministic model_fit_config_sha256 for a given configuration."""
    params_dict = dict(params) if not isinstance(params, dict) else params
    canonical_payload = {
        "experiment_id": experiment_id,
        "arm_id": arm_id,
        "parameters": {k: params_dict[k] for k in sorted(params_dict.keys())},
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_full_hpo_factory(snapshot: FullCandidateSnapshot) -> None:
    """Reject declarations that the approved fixed HGB constructor would ignore."""
    expected = {**get_hgb_signature_defaults(), **DEFAULT_HGB_PARAMS,
                "random_state": 42, "early_stopping": False, "class_weight": None}
    if (snapshot.spec.model_family != "hist_gradient_boosting"
            or canonical_feature_order(snapshot.spec.feature_pool) != CORE_FEATURE_POOL or snapshot.selector_arms
            or snapshot.arms != ("control", "challenger")
            or any(name not in expected or canonical_json(value) != canonical_json(expected[name])
                   for name, value in snapshot.spec.model_params)):
        raise ValueError("HPO fixed model configuration/parameters must match actual approved factory")


@dataclass(frozen=True)
class FullTuningResult:
    """Immutable tuning result holding fitted models, evidence, and winner parameters."""

    repeat: int
    fold: int
    selected_trial_id: int
    selected_params: tuple[tuple[str, object], ...]
    selected_params_hash: str
    study_manifest_hash: str
    inner_membership_digest: str
    default_model: Pipeline | None
    selected_model: Pipeline | None
    default_evidence: tuple[ArtifactEvidence, ...]
    selected_refit_evidence: tuple[ArtifactEvidence, ...]
    study_db_path: str


@dataclass(frozen=True)
class FullTuningAuthority:
    """Read-only worker paths/snapshot, not a journal or mutation capability."""

    root: Path
    source_root: Path
    spec: FullRunSpec


def _reject_symlink_path(path: Path, label: str) -> None:
    _validate_managed_path(path, label, directory=path.is_dir())


def _validate_run_study_dir(study_dir: str | Path, journal_root: Path) -> Path:
    candidate = Path(study_dir)
    if not candidate.absolute().is_relative_to(journal_root) or candidate.absolute() == journal_root:
        raise ValueError("study directory must be strictly under journal root")
    _reject_symlink_path(candidate, "study directory")
    absolute = candidate.absolute()
    if absolute.exists() and not absolute.is_dir():
        raise ValueError(f"study directory must be a directory: {absolute}")
    absolute.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(absolute, "study directory")
    return absolute


def _init_full_sqlite_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS study (
            study_key TEXT PRIMARY KEY,
            repeat INTEGER NOT NULL,
            fold INTEGER NOT NULL,
            study_manifest_hash TEXT NOT NULL,
            manifest_payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            study_spec_json TEXT NOT NULL,
            inner_membership_digest TEXT NOT NULL,
            training_rows_count INTEGER NOT NULL,
            n_trials INTEGER NOT NULL,
            inner_folds INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            planned_model_fits INTEGER NOT NULL,
            selected_trial_id INTEGER,
            selected_params_json TEXT,
            default_params_json TEXT NOT NULL,
            failure_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trial (
            study_key TEXT NOT NULL,
            trial_id INTEGER NOT NULL,
            params_json TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            mean_score REAL,
            failure_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (study_key, trial_id)
        );

        CREATE TABLE IF NOT EXISTS fold_score (
            study_key TEXT NOT NULL,
            trial_id INTEGER NOT NULL,
            fold_index INTEGER NOT NULL,
            score REAL,
            status TEXT NOT NULL,
            failure_reason TEXT,
            attempt_id TEXT,
            evidence_json_sha256 TEXT,
            parquet_sha256 TEXT,
            PRIMARY KEY (study_key, trial_id, fold_index)
        );
    """)
    conn.commit()


def _open_or_retry_unit(
    journal: FullFitJournal,
    unit: str,
    selected_parameters: tuple[tuple[str, object], ...],
    retry_reason: str | None,
) -> FullAttemptRecord:
    """Open or retry a journal unit, handling prior attempts cleanly."""
    history = [a for a in journal.attempts() if a.unit == unit]
    if any(a.status == "completed" for a in history):
        raise ValueError(f"unit {unit} is already completed in journal")
    if not history:
        return journal.begin(unit, selected_parameters=selected_parameters)
    if not retry_reason:
        raise BlockedExperimentError("explicit_retry_required", "explicit_retry_required: failed work needs parent authorization")
    return journal.retry(
        unit,
        reason=retry_reason,
        selected_parameters=selected_parameters,
    )


def _terminalize_owned_attempt(journal: FullFitJournal, attempt_id: str, error: BaseException) -> None:
    record = next(record for record in journal.attempts() if record.attempt_id == attempt_id)
    if record.status in ("running", "awaiting_validation"):
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        journal.finish(attempt_id, status, reason=f"{type(error).__name__}: {error}")


INNER_SCHEMA = pa.schema([
    ("validation_position", pa.int64()), ("probability", pa.float64()),
    ("target", pa.int64()), ("row_key", pa.string()), ("group_key", pa.string()),
])


def _key(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positions_digest(positions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(positions), MAX_BATCH_ROWS):
        digest.update(np.asarray(positions[start:start + MAX_BATCH_ROWS], dtype="<i8").tobytes())
    return digest.hexdigest()


def _prediction_auc(path: Path, positions: np.ndarray, labels: np.ndarray,
                    training: TrainingPartition | None = None) -> float:
    """Verify batches against training-only authority and rank ties on disk."""
    connection = sqlite3.connect("")
    offset = 0
    try:
        connection.execute("CREATE TABLE scores(probability REAL NOT NULL, target INTEGER NOT NULL)")
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(INNER_SCHEMA, check_metadata=True):
            raise BlockedExperimentError("study_tampered", "exact inner prediction schema required")
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
            actual_positions = batch.column(0).to_numpy()
            probabilities = batch.column(1).to_numpy()
            targets = batch.column(2).to_numpy()
            expected = positions[offset:offset + len(actual_positions)]
            if (not np.array_equal(actual_positions, expected)
                    or not np.array_equal(targets, labels[expected])
                    or not np.isfinite(probabilities).all()
                    or np.any((probabilities < 0) | (probabilities > 1))):
                raise BlockedExperimentError("study_tampered", "study_tampered: invalid inner prediction identity/target/probability")
            if training is not None:
                row_keys = [_key(value) for value in training.X_train.index[expected]]
                group_keys = [_key(value) for value in training.groups_train.iloc[expected]]
                if (batch.column(3).to_pylist() != row_keys or batch.column(4).to_pylist() != group_keys):
                    raise BlockedExperimentError("study_tampered", "inner row/group keys differ from training-only authority")
            connection.executemany("INSERT INTO scores VALUES (?, ?)",
                                   ((float(p), int(y)) for p, y in zip(probabilities, targets, strict=True)))
            offset += len(actual_positions)
        if offset != len(positions):
            raise BlockedExperimentError("study_tampered", "study_tampered: missing inner predictions")
        positive, total = connection.execute("SELECT SUM(target), COUNT(*) FROM scores").fetchone()
        negative = total - positive
        if not positive or not negative:
            raise ValueError("both classes required for inner AUC")
        connection.execute("CREATE INDEX score_order ON scores(probability)")
        below = 0
        favorable = 0.0
        for pos, count in connection.execute("SELECT SUM(target), COUNT(*) FROM scores GROUP BY probability ORDER BY probability"):
            neg = count - pos
            favorable += pos * (below + neg / 2)
            below += neg
        return favorable / (positive * negative)
    finally:
        connection.close()


def _write_inner_fold_evidence(
    attempt_dir: Path,
    trial_id: int,
    fold_idx: int,
    params_hash: str,
    valid_positions: np.ndarray,
    arm: Pipeline,
    training: TrainingPartition,
    *,
    parameters: dict[str, Any],
    unit: str,
    train_positions: np.ndarray,
) -> tuple[float, tuple[ArtifactEvidence, ...]]:
    """Persist inner validation predictions in Parquet and compact metadata in JSON."""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = attempt_dir / "inner_predictions.parquet"
    json_path = attempt_dir / "evidence.json"

    # 1. Parquet storage
    schema = INNER_SCHEMA
    with parquet_path.open("xb") as stream, pq.ParquetWriter(stream, schema, compression="snappy") as writer:
        for start in range(0, len(valid_positions), MAX_BATCH_ROWS):
            positions = valid_positions[start:start + MAX_BATCH_ROWS]
            probabilities = _extract_pos_proba(arm, training.X_train.iloc[positions])
            writer.write_table(pa.Table.from_arrays([
                pa.array(positions, type=pa.int64()),
                pa.array(probabilities, type=pa.float64()),
                pa.array(training.y_train.iloc[positions].to_numpy(), type=pa.int64()),
                pa.array([_key(value) for value in training.X_train.index[positions]], type=pa.string()),
                pa.array([_key(value) for value in training.groups_train.iloc[positions]], type=pa.string()),
            ], schema=schema))
    with parquet_path.open("rb") as f:
        os.fsync(f.fileno())
    parquet_sha = _file_hash(parquet_path)
    score = _prediction_auc(parquet_path, valid_positions, training.y_train.to_numpy(), training)

    # 2. Compact metadata JSON (no giant lists)
    positions_hash = hashlib.sha256()
    for start in range(0, len(valid_positions), MAX_BATCH_ROWS):
        positions_hash.update(np.asarray(valid_positions[start:start + MAX_BATCH_ROWS], dtype="<i8").tobytes())
    positions_sha = positions_hash.hexdigest()

    meta = {
        "trial_id": int(trial_id),
        "fold_index": int(fold_idx),
        "params_hash": str(params_hash),
        "score": float(score),
        "row_count": len(valid_positions),
        "positions_sha256": positions_sha,
        "parquet_sha256": parquet_sha,
        "train_positions_sha256": _positions_digest(train_positions),
        "unit": unit,
        "config_sha256": compute_fit_config_sha256(unit.split("/")[0], "challenger", parameters),
    }
    canonical_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))
    meta_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    meta["sha256"] = meta_sha

    raw_json_bytes = json.dumps(meta, indent=2, sort_keys=True).encode("utf-8")
    with json_path.open("xb") as f:
        f.write(raw_json_bytes)
        f.flush()
        os.fsync(f.fileno())

    json_file_sha = hashlib.sha256(raw_json_bytes).hexdigest()

    # Artifact evidence relative to journal root
    return score, (
        ArtifactEvidence(
            path=str(parquet_path.relative_to(attempt_dir.parent.parent)),
            sha256=parquet_sha,
        ),
        ArtifactEvidence(
            path=str(json_path.relative_to(attempt_dir.parent.parent)),
            sha256=json_file_sha,
        ),
    )


def _write_refit_evidence(
    attempt_dir: Path,
    phase: str,
    params: dict[str, Any],
    row_count: int,
    *,
    unit: str,
) -> tuple[ArtifactEvidence, ...]:
    """Persist outer refit metadata in JSON."""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    json_path = attempt_dir / f"fit_{phase}.json"

    payload = {
        "phase": phase,
        "parameters": {k: params[k] for k in sorted(params.keys())},
        "row_count": int(row_count),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "unit": unit,
        "config_sha256": compute_fit_config_sha256(unit.split("/")[0], unit.split("/")[1], params),
    }
    raw_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with json_path.open("xb") as f:
        f.write(raw_bytes)
        f.flush()
        os.fsync(f.fileno())

    file_sha = hashlib.sha256(raw_bytes).hexdigest()
    return (
        ArtifactEvidence(
            path=str(json_path.relative_to(attempt_dir.parent.parent)),
            sha256=file_sha,
        ),
    )


def _verify_seals(root: Path, record: FullAttemptRecord, names: tuple[str, ...]) -> None:
    if record.status not in ("awaiting_validation", "completed"):
        raise BlockedExperimentError("study_tampered", "actual journal seal required before scientific validation")
    expected = tuple(str(Path(record.artifact_directory) / name) for name in names)
    if tuple(item.path for item in record.evidence) != expected:
        raise BlockedExperimentError("study_tampered", "exclusive exact journal evidence identities required")
    for item in record.evidence:
        path = _validate_managed_path(root / item.path, "sealed HPO evidence", root=root)
        if not path.is_relative_to(root / record.artifact_directory) or _file_hash(path) != item.sha256:
            raise BlockedExperimentError("study_tampered", "HPO evidence differs from actual journal seal")


def _verify_refit_evidence(root: Path, record: FullAttemptRecord, phase: str,
                           parameters: dict[str, Any], row_count: int) -> None:
    _verify_seals(root, record, (f"fit_{phase}.json",))
    meta = exact_fields(parse_json((root / record.evidence[0].path).read_text()),
                        {"phase", "parameters", "row_count", "timestamp", "unit", "config_sha256"})
    config_sha = compute_fit_config_sha256(record.unit.split("/")[0], record.unit.split("/")[1], parameters)
    if (meta["phase"] != phase or canonical_json(meta["parameters"]) != canonical_json(parameters)
            or type(meta["row_count"]) is not int or meta["row_count"] != row_count
            or type(meta["timestamp"]) is not str or not meta["timestamp"]
            or meta["unit"] != record.unit or meta["config_sha256"] != config_sha
            or record.selected_parameters != tuple(sorted(parameters.items()))):
        raise BlockedExperimentError("study_tampered", "outer refit configuration/row/phase identity mismatch")


def _verify_fold_evidence(
    journal_root: Path,
    attempt_record: FullAttemptRecord,
    expected_positions: np.ndarray,
    expected_y: np.ndarray,
    training: TrainingPartition | None = None,
) -> float:
    """Verify evidence files exist, match SHA-256 seals, and recompute ROC-AUC score."""
    _verify_seals(journal_root, attempt_record, ("inner_predictions.parquet", "evidence.json"))
    attempt_dir = journal_root / attempt_record.artifact_directory
    parquet_path = attempt_dir / "inner_predictions.parquet"
    json_path = attempt_dir / "evidence.json"

    if not parquet_path.is_file() or not json_path.is_file():
        raise BlockedExperimentError("study_tampered", "missing inner fold evidence file")

    actual_parquet_sha = _file_hash(parquet_path)

    meta = exact_fields(parse_json(json_path.read_text()), {
        "trial_id", "fold_index", "params_hash", "score", "row_count", "positions_sha256",
        "parquet_sha256", "sha256", "train_positions_sha256", "unit", "config_sha256",
    })
    match = re.fullmatch(r"([^/]+)/challenger/r(\d+)/f(\d+)/inner/trial(\d+)/fold(\d+)", attempt_record.unit)
    if match is None:
        raise BlockedExperimentError("study_tampered", "invalid inner unit identity")
    experiment, repeat, outer_fold, trial, inner_fold = match.groups()
    params = dict(attempt_record.selected_parameters)
    params_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    unsigned = {name: value for name, value in meta.items() if name != "sha256"}
    if (type(meta["trial_id"]) is not int or meta["trial_id"] != int(trial)
            or type(meta["fold_index"]) is not int or meta["fold_index"] != int(inner_fold)
            or type(meta["row_count"]) is not int or meta["row_count"] != len(expected_positions)
            or meta["positions_sha256"] != _positions_digest(expected_positions)
            or meta["params_hash"] != params_hash or meta["unit"] != attempt_record.unit
            or meta["config_sha256"] != compute_fit_config_sha256(experiment, "challenger", params)
            or meta["sha256"] != hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):
        raise BlockedExperimentError("study_tampered", "inner evidence metadata/config identity mismatch")
    if training is not None:
        if training.repeat != int(repeat) or training.fold != int(outer_fold):
            raise BlockedExperimentError("study_tampered", "training outer partition identity mismatch")
        # The supplied expected validation positions are already derived by the
        # caller from the exact StudySpec; train membership is its complement.
        train_positions = np.setdiff1d(np.arange(len(training.y_train)), expected_positions)
        if meta["train_positions_sha256"] != _positions_digest(train_positions):
            raise BlockedExperimentError("study_tampered", "inner train-only membership mismatch")
        if set(training.groups_train.iloc[train_positions]) & set(training.groups_train.iloc[expected_positions]):
            raise BlockedExperimentError("study_tampered", "inner group membership leakage")
    if meta.get("parquet_sha256") != actual_parquet_sha:
        raise BlockedExperimentError("study_tampered", "inner predictions parquet hash mismatch")

    recomputed_auc = _prediction_auc(parquet_path, expected_positions, expected_y, training)
    if not math.isfinite(recomputed_auc):
        raise BlockedExperimentError("study_tampered", "recomputed ROC-AUC is non-finite")

    if type(meta["score"]) is not float or not math.isfinite(meta["score"]) or abs(recomputed_auc - meta["score"]) > 1e-12:
        raise BlockedExperimentError("study_tampered", "recomputed ROC-AUC score mismatch")

    return recomputed_auc


def tune_full_outer_partition(
    training: TrainingPartition,
    study_spec: StudySpec,
    journal: FullFitJournal | None,
    study_dir: str | Path,
    experiment_id: str = "regime_same_family_hpo_vs_default",
    arm_factory: Callable[[dict[str, Any]], Pipeline] = build_hgb_tuning_arm,
    *,
    retry_reason: str | None = None,
    lifecycle: WorkerHpoLifecycle | None = None,
    authority: FullTuningAuthority | None = None,
) -> FullTuningResult:
    """Execute complete bounded HGB nested tuning for a single outer partition.

    Accounting:
    - Exactly 1 default refit on outer training data (control)
    - Exactly T*K inner fits on inner group folds (challenger)
    - Exactly 1 selected refit on outer training data (challenger)
    - Total per partition: T*K + 2 model fits, T trials.
    """
    # Strict validation of input partition
    if not isinstance(training, TrainingPartition) or isinstance(training, FoldPartition):
        raise TypeError("training must be a TrainingPartition; FoldPartition is prohibited")
    if hasattr(training, "X_validation") or hasattr(training, "validation_keys"):
        raise TypeError("training partition must not contain held-out validation fields")
    if not isinstance(study_spec, StudySpec):
        raise TypeError("study_spec must be a typed StudySpec")
    if lifecycle is None:
        if not isinstance(journal, FullFitJournal):
            raise TypeError("journal must be a typed FullFitJournal")
        context = journal
    else:
        from project.experiments.full_hpo_lifecycle import WorkerHpoLifecycle
        if (type(lifecycle) is not WorkerHpoLifecycle or journal is not None
                or type(authority) is not FullTuningAuthority or retry_reason is not None):
            raise TypeError("worker requires lifecycle and read-only authority, never a journal/retry grant")
        if (lifecycle.plan.spec.binding_sha256 != authority.spec.binding_sha256 or lifecycle.plan.repeat != training.repeat
                or lifecycle.plan.fold != training.fold
                or canonical_json(asdict(study_spec)) != canonical_json(asdict(full_hpo_study_spec()))):
            raise ValueError("worker lifecycle training/study authority mismatch")
        context = authority

    snapshots = [item for item in context.spec.candidates if item.spec.experiment_id == experiment_id]
    if (len(snapshots) != 1 or snapshots[0].study is None
            or canonical_json(asdict(snapshots[0].study)) != canonical_json(asdict(study_spec))
            or snapshots[0].study_method != "random_search"):
        raise BlockedExperimentError("study_spec_mismatch", "study_spec_mismatch: study must equal the immutable journal snapshot")
    validate_full_hpo_factory(snapshots[0])
    if journal is not None and any(record.status in ("running", "awaiting_validation") for record in journal.attempts()):
        raise BlockedExperimentError("journal_occupied", "journal_occupied: parent must reconcile active work explicitly")
    try:
        if journal is not None:
            journal.verify_completed()
    except ValueError as exc:
        raise BlockedExperimentError("study_tampered", f"study_tampered: {exc}") from exc

    # Deep copy inputs and validate features
    X_clean = training.X_train.copy(deep=True)
    X_clean.attrs = {}
    y_clean = training.y_train.copy(deep=True)
    y_clean.attrs = {}
    groups_clean = training.groups_train.copy(deep=True)
    groups_clean.attrs = {}

    clean_training = TrainingPartition(
        repeat=int(training.repeat),
        fold=int(training.fold),
        X_train=X_clean,
        y_train=y_clean,
        groups_train=groups_clean,
    )
    validate_partition(clean_training, build_lead_baseline_registry())

    if tuple(clean_training.X_train.columns) != CORE_FEATURE_POOL:
        raise ValueError(
            f"training predictors must match exact ordered CORE feature pool: "
            f"expected {CORE_FEATURE_POOL}, got {tuple(clean_training.X_train.columns)}"
        )

    # Validate study spec
    if type(study_spec.inner_folds) is not int or not (2 <= study_spec.inner_folds <= 5):
        raise ValueError("study_spec inner_folds must be integer in [2, 5]")
    if type(study_spec.seed) is not int or not (0 <= study_spec.seed <= 2**32 - 1):
        raise ValueError("study_spec seed must be integer in [0, 2**32 - 1]")

    for name, values in study_spec.parameter_lists:
        if name not in STUDY_SPACE:
            raise ValueError(f"study parameter not in allowed HGB space: {name}")
        for v in values:
            if v not in STUDY_SPACE[name]:
                raise ValueError(f"study parameter value not in allowed HGB space: {name}={v}")

    study_dir_path = _validate_run_study_dir(study_dir, context.root)
    db_path = study_dir_path / f"full_study_r{clean_training.repeat}_f{clean_training.fold}.sqlite3"
    for suffix in ("", "-journal", "-wal", "-shm"):
        _validate_managed_path(Path(str(db_path) + suffix), "study database", root=context.root)

    # Inner group folds
    folds = build_inner_folds(
        clean_training.y_train,
        clean_training.groups_train,
        n_splits=study_spec.inner_folds,
        seed=study_spec.seed,
    )
    membership_hash = hashlib.sha256()
    for train_positions, valid_positions in folds:
        for positions in (train_positions, valid_positions):
            membership_hash.update(len(positions).to_bytes(8, "big"))
            for start in range(0, len(positions), MAX_BATCH_ROWS):
                membership_hash.update(np.asarray(positions[start:start + MAX_BATCH_ROWS], dtype="<i8").tobytes())
    membership_digest = membership_hash.hexdigest()
    trial_param_list = _generate_trial_params(study_spec)

    study_key = f"full_study_r{clean_training.repeat}_f{clean_training.fold}"
    planned_fits = required_full_fits(
        study_spec.n_trials, study_spec.inner_folds, outer_partitions=1
    )

    def frame_digest(frame: pd.DataFrame | pd.Series) -> str:
        digest = hashlib.sha256()
        for start in range(0, len(frame), MAX_BATCH_ROWS):
            digest.update(pd.util.hash_pandas_object(frame.iloc[start:start + MAX_BATCH_ROWS], index=True).to_numpy().tobytes())
        return digest.hexdigest()

    manifest_payload = {
        "source_root": str(context.source_root),
        "spec_sha256": context.spec.binding_sha256,
        "sources": [asdict(binding) for binding in context.spec.sources],
        "candidate_spec": asdict(snapshots[0]),
        "environment": dict(context.spec.environment),
        "resource_policy": asdict(context.spec.resource_policy),
        "feature_encoding_digest": {binding.path: binding.sha256 for binding in context.spec.sources if binding.role == "encoding"},
        "training_digests": {"X": frame_digest(X_clean), "y": frame_digest(y_clean), "groups": frame_digest(groups_clean)},
        "columns_info": [{"name": name, "dtype": str(dtype)} for name, dtype in X_clean.dtypes.items()],
        "study_spec": asdict(study_spec),
        "inner_membership_digest": membership_digest,
        "repeat": clean_training.repeat, "fold": clean_training.fold,
        "deterministic_trials": trial_param_list,
    }
    manifest_json = json.dumps(manifest_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    study_manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()

    def execute_fit(unit, phase, params, perform, *, trial_id=None, inner_fold=None):
        """One constructor boundary for direct-parent and isolated-worker callers."""
        parameters = tuple(sorted(params.items()))
        def construct():
            model = arm_factory(dict(params))
            _validate_constructed_arm(model, params)
            return model
        if lifecycle is not None:
            identity = (lifecycle.plan.default() if phase == "default_refit" else
                        lifecycle.plan.inner(trial_id, inner_fold) if phase == "inner" else
                        lifecycle.plan.selected(trial_id))
            if identity.unit != unit or identity.parameters != parameters:
                raise ValueError("tuning lifecycle exact fit identity mismatch")
            model = lifecycle.run_fit(identity, construct, perform)
            return model, lifecycle.completed_proof
        completed = next((record for record in journal.attempts()
                          if record.unit == unit and record.status == "completed"), None)
        if completed is not None:
            if completed.selected_parameters != parameters:
                raise BlockedExperimentError("study_tampered", "completed fit parameters mismatch")
            if phase != "inner":
                _verify_refit_evidence(context.root, completed, phase, params, len(clean_training.X_train))
            return None, completed
        record = _open_or_retry_unit(journal, unit, parameters, retry_reason)
        try:
            model = construct()
            evidence = perform(model, record)
            journal.seal(record.attempt_id, evidence)
            sealed = next(item for item in journal.attempts() if item.attempt_id == record.attempt_id)
            if phase == "inner":
                _verify_fold_evidence(context.root, sealed, folds[inner_fold][1], clean_training.y_train.to_numpy(), clean_training)
            else:
                _verify_refit_evidence(context.root, sealed, phase, params, len(clean_training.X_train))
            journal.finish(record.attempt_id, "completed")
            return model, next(item for item in journal.attempts() if item.attempt_id == record.attempt_id)
        except BaseException as error:
            _terminalize_owned_attempt(journal, record.attempt_id, error)
            raise

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA synchronous = FULL")
        _init_full_sqlite_tables(conn)

        study_row = conn.execute(
            "SELECT study_manifest_hash, status, selected_trial_id, selected_params_json, manifest_payload_json FROM study WHERE study_key = ?",
            (study_key,),
        ).fetchone()

        if study_row is not None:
            existing_hash, _status, _sel_id, _sel_json, existing_manifest = study_row
            if existing_hash != study_manifest_hash:
                raise BlockedExperimentError(
                    "study_manifest_mismatch", "study manifest hash does not match stored specification"
                )
            if existing_manifest != manifest_json:
                raise BlockedExperimentError("study_tampered", "study_tampered: stored manifest bytes differ from actual binding")
            if conn.execute("SELECT COUNT(*) FROM study").fetchone()[0] != 1:
                raise BlockedExperimentError("study_tampered", "study_tampered: unexpected study rows")
            for key, trial_id, params_json, params_hash in conn.execute("SELECT study_key, trial_id, params_json, params_hash FROM trial"):
                if key != study_key or not 0 <= trial_id < len(trial_param_list):
                    raise BlockedExperimentError("study_tampered", "study_tampered: unplanned trial row")
                canonical_params = json.dumps(trial_param_list[trial_id], sort_keys=True)
                if params_json != canonical_params or params_hash != hashlib.sha256(canonical_params.encode()).hexdigest():
                    raise BlockedExperimentError("study_tampered", "study_tampered: mutable trial differs from canonical sampling")

        now = datetime.datetime.now(datetime.UTC).isoformat()
        if study_row is None:
            study_spec_dict = {
                "n_trials": study_spec.n_trials,
                "inner_folds": study_spec.inner_folds,
                "seed": study_spec.seed,
                "parameter_lists": [list(item) for item in study_spec.parameter_lists],
            }
            conn.execute(
                """INSERT INTO study (
                    study_key, repeat, fold, study_manifest_hash, manifest_payload_json,
                    status, study_spec_json, inner_membership_digest, training_rows_count,
                    n_trials, inner_folds, seed, planned_model_fits, default_params_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    study_key,
                    clean_training.repeat,
                    clean_training.fold,
                    study_manifest_hash,
                    manifest_json,
                    json.dumps(study_spec_dict, sort_keys=True),
                    membership_digest,
                    len(clean_training.X_train),
                    study_spec.n_trials,
                    study_spec.inner_folds,
                    study_spec.seed,
                    planned_fits,
                    json.dumps(DEFAULT_HGB_PARAMS, sort_keys=True),
                    now,
                    now,
                ),
            )
            conn.commit()

        # Step 1: Default outer refit (Control)
        default_unit = f"{experiment_id}/control/r{clean_training.repeat}/f{clean_training.fold}/default_refit"
        def perform_default(model, record):
            model.fit(clean_training.X_train, clean_training.y_train)
            return _write_refit_evidence(context.root / record.artifact_directory,
                                        "default_refit", DEFAULT_HGB_PARAMS, len(clean_training.X_train), unit=record.unit)
        default_model, default_record = execute_fit(default_unit, "default_refit", DEFAULT_HGB_PARAMS, perform_default)
        default_evidence = default_record.evidence

        # Step 2: Inner trial fits (Challenger)
        y_train_np = clean_training.y_train.to_numpy()

        for t_idx, params in enumerate(trial_param_list):
            p_json = json.dumps(params, sort_keys=True)
            p_hash = hashlib.sha256(p_json.encode("utf-8")).hexdigest()

            trial_row = conn.execute(
                "SELECT status, mean_score, params_json, params_hash FROM trial WHERE study_key = ? AND trial_id = ?",
                (study_key, t_idx),
            ).fetchone()
            if trial_row is not None and (trial_row[2] != p_json or trial_row[3] != p_hash):
                raise BlockedExperimentError("study_tampered", "study_tampered: trial parameters differ from canonical sampling")
            if trial_row is None:
                conn.execute(
                    "INSERT INTO trial VALUES (?, ?, ?, ?, 'in_progress', NULL, NULL, ?, ?)",
                    (study_key, t_idx, p_json, p_hash, now, now),
                )
                conn.commit()

            fold_scores: list[float] = []

            for f_idx, (train_pos, valid_pos) in enumerate(folds):
                inner_unit = f"{experiment_id}/challenger/r{clean_training.repeat}/f{clean_training.fold}/inner/trial{t_idx}/fold{f_idx}"
                params_tuple = tuple(sorted(params.items()))

                work = (t_idx, f_idx, p_hash, valid_pos, params, inner_unit, train_pos)
                def perform_inner(model, record, work=work):
                    trial, inner, param_hash, positions, parameters, unit, train = work
                    model.fit(clean_training.X_train.iloc[train], clean_training.y_train.iloc[train])
                    _, proof = _write_inner_fold_evidence(
                        context.root / record.artifact_directory, trial, inner, param_hash,
                        positions, model, clean_training, parameters=parameters, unit=unit,
                        train_positions=train)
                    return proof
                try:
                    arm, completed_attempt = execute_fit(inner_unit, "inner", params, perform_inner,
                                                        trial_id=t_idx, inner_fold=f_idx)
                    score = _verify_fold_evidence(context.root, completed_attempt, valid_pos, y_train_np, clean_training)
                    if completed_attempt.selected_parameters != params_tuple:
                        raise BlockedExperimentError("study_tampered", "journal trial configuration mismatch")
                    score_row = conn.execute(
                        "SELECT score, status, attempt_id, evidence_json_sha256, parquet_sha256 FROM fold_score "
                        "WHERE study_key=? AND trial_id=? AND fold_index=?", (study_key, t_idx, f_idx)).fetchone()
                    expected_row = (score, "completed", completed_attempt.attempt_id,
                                    completed_attempt.evidence[1].sha256, completed_attempt.evidence[0].sha256)
                    if arm is None and score_row is not None and score_row != expected_row:
                        raise BlockedExperimentError("study_tampered", "SQL score differs from sealed journal and independently recomputed AUC")
                    conn.execute(
                        "INSERT OR REPLACE INTO fold_score VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                        (study_key, t_idx, f_idx, *expected_row),
                    )
                    conn.commit()
                    fold_scores.append(score)
                except BaseException as exc:
                    conn.execute(
                        "UPDATE trial SET status = 'failed', failure_reason = ?, updated_at = ? WHERE study_key = ? AND trial_id = ?",
                        (str(exc), datetime.datetime.now(datetime.UTC).isoformat(), study_key, t_idx),
                    )
                    conn.commit()
                    raise
                finally:
                    if "arm" in locals():
                        del arm

            if len(fold_scores) == study_spec.inner_folds:
                mean_score = math.fsum(fold_scores) / len(fold_scores)
                conn.execute(
                    "UPDATE trial SET status = 'completed', mean_score = ?, updated_at = ? WHERE study_key = ? AND trial_id = ?",
                    (mean_score, datetime.datetime.now(datetime.UTC).isoformat(), study_key, t_idx),
                )
                conn.commit()

        # Step 3: Deterministic tie-breaking selection
        completed_trial_rows = conn.execute(
            "SELECT trial_id, params_json, mean_score FROM trial WHERE study_key = ? AND status = 'completed' ORDER BY mean_score DESC, trial_id ASC",
            (study_key,),
        ).fetchall()

        if not completed_trial_rows:
            conn.execute(
                "UPDATE study SET status = 'failed', failure_reason = 'no_valid_trials' WHERE study_key = ?",
                (study_key,),
            )
            conn.commit()
            raise BlockedExperimentError("no_valid_trials", "study completed with no valid candidate trials")

        best_trial_id, best_params_json, _best_mean = completed_trial_rows[0]
        best_params_dict = json.loads(best_params_json)
        selected_params_tuple = tuple(sorted(best_params_dict.items()))
        selected_fit_sha = compute_fit_config_sha256(
            experiment_id, "challenger", selected_params_tuple
        )

        conn.execute(
            """UPDATE study SET status = 'selection_finished', selected_trial_id = ?,
                                selected_params_json = ?, updated_at = ?
               WHERE study_key = ?""",
            (best_trial_id, best_params_json, datetime.datetime.now(datetime.UTC).isoformat(), study_key),
        )
        conn.commit()

        # Step 4: Outer selected refit (Challenger)
        selected_unit = f"{experiment_id}/challenger/r{clean_training.repeat}/f{clean_training.fold}/selected_refit"
        def perform_selected(model, record):
            model.fit(clean_training.X_train, clean_training.y_train)
            return _write_refit_evidence(context.root / record.artifact_directory,
                                        "selected_refit", best_params_dict, len(clean_training.X_train), unit=record.unit)
        selected_model, selected_record = execute_fit(selected_unit, "selected_refit", best_params_dict,
                                                     perform_selected, trial_id=best_trial_id)
        selected_refit_evidence = selected_record.evidence

        conn.execute("UPDATE study SET status = 'completed' WHERE study_key = ?", (study_key,))
        conn.commit()

        return FullTuningResult(
            repeat=clean_training.repeat,
            fold=clean_training.fold,
            selected_trial_id=best_trial_id,
            selected_params=selected_params_tuple,
            selected_params_hash=selected_fit_sha,
            study_manifest_hash=study_manifest_hash,
            inner_membership_digest=membership_digest,
            default_model=default_model,
            selected_model=selected_model,
            default_evidence=default_evidence,
            selected_refit_evidence=selected_refit_evidence,
            study_db_path=str(db_path),
        )
    finally:
        conn.close()
