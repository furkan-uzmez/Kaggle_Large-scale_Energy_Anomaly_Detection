"""Typed candidate foundation: no data IO, fitting, or execution authorization.

Configuration tuples remain unchanged. Partition packets own defensive pandas
copies, stripped of attrs that could smuggle held-out data into nested fitting.
Their bindings are frozen, not their pandas contents: validate again at fitting
boundaries. Scores own immutable float64 buffers. Runtime watchdogs and real
manifest/hash binding belong to admission/IO/runner tasks, not these types.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from project.features.registry import FeatureRegistry, build_lead_baseline_registry

ScoreKind = Literal["positive_probability", "decision_margin"]
KEY_COLUMNS = ("row_index", "building_id", "timestamp", "repeat", "fold")
SOURCE_NOTEBOOKS = {
    "notebooks/01_data_understanding.ipynb",
    "notebooks/02_baseline_and_candidate_design.ipynb",
}
STUDY_SPACE = {
    "learning_rate": (.03, .1), "max_leaf_nodes": (15, 31),
    "l2_regularization": (0., 1.), "max_iter": (100, 200),
}
NON_PREDICTORS = {
    "anomaly", "target", "label", "y", "id", "row_index", "timestamp",
    "repeat", "fold", "seed", "split", "support_stratum",
}


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integer(value: object, label: str, bounds: tuple[int, int | None] = (1, None)) -> None:
    low, high = bounds
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{label} must be an integer in [{low}, {high}]; booleans are invalid")


def _forbidden(name: str) -> bool:
    lowered = name.casefold()
    return (lowered in NON_PREDICTORS or lowered.endswith("_id")
            or lowered.startswith(("gte_", "source_", "fold_")))


def _pool(names: object) -> None:
    if (not isinstance(names, tuple) or not names
            or any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in names)
            or len(set(names)) != len(names) or any(_forbidden(n) for n in names)):
        raise ValueError("feature_pool requires unique nonempty predictor names, not ID/target/gte fields")


def _immutable(value: object, label: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, tuple):
        for item in value:
            _immutable(item, label)
        return
    raise ValueError(f"{label} values must be finite immutable scalars/tuples, not callables or mutable collections")


def _pairs(pairs: object, label: str) -> None:
    if not isinstance(pairs, tuple):
        raise TypeError(f"{label} parameters must be an immutable tuple of pairs")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{label} parameters require (name, value) tuple pairs")
        name, value = pair
        _text(name, label)
        if name != name.strip() or name in seen:
            raise ValueError(f"{label} parameter names must be unique and unpadded")
        seen.add(name)
        _immutable(value, label)


def _column(frame: pd.DataFrame, name: str, categorical: bool) -> None:
    values = frame[name]
    dtype = values.dtype
    if categorical:
        if isinstance(dtype, pd.CategoricalDtype):
            if any(not isinstance(v, str) for v in values.cat.categories):
                raise ValueError(f"{name} must declare string categories")
        elif not (is_object_dtype(dtype) or is_string_dtype(dtype)):
            raise ValueError(f"{name} must contain string/null categorical values")
        if any(not isinstance(v, str) for v in values.dropna()):
            raise ValueError(f"{name} must contain string/null categorical values")
    else:
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError(f"{name} requires a real nonboolean numeric dtype")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{name} contains nonfinite nonmissing numeric values")


def _predictors(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas predictor DataFrame")
    _pool(tuple(frame.columns))
    if len(frame) == 0:
        raise ValueError(f"{label} requires nonempty rows")
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError(f"{label} index must be unique and nonnull")
    baseline = build_lead_baseline_registry()
    for name in frame:
        metadata = baseline.get(name)
        categorical = (metadata.family == "categorical" if metadata is not None else
                       not is_numeric_dtype(frame[name].dtype))
        _column(frame, name, categorical)


def _aligned(series: pd.Series, frame: pd.DataFrame, label: str) -> None:
    if not isinstance(series, pd.Series) or not series.index.equals(frame.index):
        raise ValueError(f"{label} Series index must align exactly with predictor index")
    if series.isna().any():
        raise ValueError(f"{label} must not contain null values")


def _groups(groups: pd.Series) -> None:
    values = groups.tolist()
    strings = all(isinstance(v, str) and bool(v.strip()) for v in values)
    integers = all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values)
    if groups.isna().any() or not values or not (strings or integers):
        raise ValueError("group metadata must be nonnull, consistently typed opaque strings or integer IDs")


def _training(partition: TrainingPartition | FoldPartition) -> None:
    _integer(partition.repeat, "repeat", (0, 2))
    _integer(partition.fold, "fold", (0, 3))
    _predictors(partition.X_train, "training predictors")
    _aligned(partition.y_train, partition.X_train, "training target")
    _aligned(partition.groups_train, partition.X_train, "training groups")
    target = partition.y_train
    if (not is_integer_dtype(target.dtype) or is_bool_dtype(target.dtype)
            or set(target.tolist()) != {0, 1}):
        raise ValueError("training target must have both integer binary classes 0 and 1")
    _groups(partition.groups_train)


def _validation(partition: FoldPartition) -> None:
    frame, keys = partition.X_validation, partition.validation_keys
    _predictors(frame, "validation predictors")
    if tuple(frame.columns) != tuple(partition.X_train.columns):
        raise ValueError("validation schema must match the exact ordered training schema")
    if not isinstance(keys, pd.DataFrame) or tuple(keys.columns) != KEY_COLUMNS:
        raise ValueError("validation key columns must be row_index/building_id/timestamp/repeat/fold only")
    if not keys.index.equals(frame.index) or not keys.index.is_unique:
        raise ValueError("validation key index must align exactly with validation predictors")
    for name in ("row_index", "repeat", "fold"):
        if not is_integer_dtype(keys[name].dtype) or is_bool_dtype(keys[name].dtype) or keys[name].isna().any():
            raise ValueError(f"validation key {name} must be nonnull integer metadata")
    if (keys.row_index < 0).any() or keys.row_index.duplicated().any():
        raise ValueError("validation row_index keys must be unique and nonnegative")
    if not keys.repeat.eq(partition.repeat).all() or not keys.fold.eq(partition.fold).all():
        raise ValueError("validation repeat/fold keys disagree with partition identity")
    if not is_datetime64_any_dtype(keys.timestamp.dtype) or keys.timestamp.isna().any():
        raise ValueError("validation timestamp keys must be nonnull Datetime values")
    _groups(keys.building_id)
    if keys.duplicated(["building_id", "timestamp"]).any():
        raise ValueError("duplicate validation natural keys")
    if set(partition.groups_train) & set(keys.building_id):
        raise ValueError("training and validation building groups must be disjoint")
    if not partition.X_train.index.intersection(frame.index).empty:
        raise ValueError("training and validation row indices overlap")


def _snapshot(partition: object, names: tuple[str, ...]) -> None:
    for name in names:
        copy = getattr(partition, name).copy(deep=True)
        copy.attrs = {}
        object.__setattr__(partition, name, copy)


class BlockedExperimentError(ValueError):
    """An explicit readiness/admission/accounting blocker, not a fallback."""

    def __init__(self, code: str, reason: str) -> None:
        _text(code, "blocker code")
        _text(reason, "blocker reason")
        self.code = code
        super().__init__(reason)


def _validate_managed_path(path: str | Path, label: str, *, root: str | Path | None = None,
                           directory: bool = False) -> Path:
    """Validate every lexical ancestor before resolving or opening managed paths."""
    candidate = Path(path).absolute()
    if ".." in candidate.parts:
        raise ValueError(f"{label} must not contain parent-path aliases")
    if root is not None and not candidate.is_relative_to(Path(root).absolute()):
        raise ValueError(f"{label} must remain beneath its managed root")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} must not contain symlink components: {current}")
        if current != candidate or directory:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"{label} ancestors must be directories")
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"{label} must be a regular file without hardlinks")
    return candidate


@dataclass(frozen=True)
class CandidateScore:
    kind: ScoreKind
    values: np.ndarray
    positive_label: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in ("positive_probability", "decision_margin"):
            raise ValueError("unsupported explicit score kind")
        if type(self.positive_label) is not int or self.positive_label != 1:
            raise ValueError("positive_label must be integer 1")
        if (not isinstance(self.values, np.ndarray) or self.values.ndim != 1
                or self.values.dtype.kind not in "iuf"):
            raise ValueError("score values must be a one-dimensional real numeric array")
        with np.errstate(over="ignore", invalid="ignore"):
            values = self.values.astype(np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("score values must be finite float64 values")
        if self.kind == "positive_probability" and ((values < 0) | (values > 1)).any():
            raise ValueError("positive probability scores must be in [0,1]")
        object.__setattr__(self, "values", np.frombuffer(values.tobytes(), dtype=np.float64))


@dataclass(frozen=True)
class TrainingPartition:
    """Isolated fitting view; deliberately no held-out fields or attrs."""
    repeat: int
    fold: int
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series

    def __post_init__(self) -> None:
        _training(self)
        _snapshot(self, ("X_train", "y_train", "groups_train"))


@dataclass(frozen=True)
class FoldPartition:
    """Predictor/key separation; validation targets belong to scoring only."""
    repeat: int
    fold: int
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    X_validation: pd.DataFrame
    validation_keys: pd.DataFrame

    def __post_init__(self) -> None:
        _training(self)
        _validation(self)
        _snapshot(self, ("X_train", "y_train", "groups_train", "X_validation", "validation_keys"))


@dataclass(frozen=True)
class FitBudget:
    """Required declared caps; no runtime enforcement or authorization inferred."""
    max_model_fits: int
    max_trials: int
    max_wall_seconds: float
    max_rss_bytes: int
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        for name in ("max_model_fits", "max_rss_bytes", "cpu_threads"):
            _integer(getattr(self, name), name)
        _integer(self.max_trials, "max_trials", (0, None))
        try:
            valid_wall = (type(self.max_wall_seconds) in (int, float)
                          and math.isfinite(self.max_wall_seconds) and self.max_wall_seconds > 0)
        except OverflowError:
            valid_wall = False
        if not valid_wall:
            raise ValueError("max_wall_seconds must be positive and finite, not boolean")


@dataclass(frozen=True)
class StudySpec:
    n_trials: int
    inner_folds: int
    seed: int
    parameter_lists: tuple[tuple[str, tuple], ...]

    def __post_init__(self) -> None:
        _integer(self.n_trials, "n_trials")
        _integer(self.inner_folds, "inner_folds", (2, 5))
        _integer(self.seed, "seed", (0, 2**32 - 1))
        _pairs(self.parameter_lists, "study parameter_lists")
        if not self.parameter_lists:
            raise ValueError("study parameter space must be nonempty")
        capacity = 1
        for name, values in self.parameter_lists:
            if name not in STUDY_SPACE or not isinstance(values, tuple) or not values:
                raise ValueError("study parameter space must use nonempty tuples of allowed HGB subsets")
            if len(set(values)) != len(values):
                raise ValueError("study parameter values must be unique")
            for value in values:
                if (type(value) not in (int, float) or value not in STUDY_SPACE[name]
                        or (name in ("max_iter", "max_leaf_nodes") and type(value) is not int)):
                    raise ValueError(f"study parameter outside frozen HGB space: {name}")
            capacity *= len(values)
        if self.n_trials > capacity:
            raise ValueError("n_trials exceeds finite unique study parameter space")


@dataclass(frozen=True)
class CandidateSpec:
    experiment_id: str
    feature_pool: tuple[str, ...]
    model_family: str
    model_params: tuple[tuple[str, object], ...]
    method_params: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _text(self.experiment_id, "experiment_id")
        _text(self.model_family, "model_family")
        _pool(self.feature_pool)
        _pairs(self.model_params, "model_params")
        _pairs(self.method_params, "method_params")


@dataclass(frozen=True)
class RunSpec:
    """Declarative run intent; full mode alone is never an execution grant."""
    version: int
    candidates: tuple[CandidateSpec, ...]
    budget: FitBudget
    execution_mode: Literal["synthetic_smoke", "full"]
    source_notebook: str
    source_cell_id: str

    def __post_init__(self) -> None:
        _integer(self.version, "version", (1, 1))
        if (not isinstance(self.candidates, tuple) or not self.candidates
                or any(not isinstance(c, CandidateSpec) for c in self.candidates)
                or len({c.experiment_id for c in self.candidates}) != len(self.candidates)):
            raise ValueError("candidates must be a nonempty immutable tuple with unique experiment IDs")
        if not isinstance(self.budget, FitBudget):
            raise TypeError("run budget must be a typed FitBudget")
        if not isinstance(self.execution_mode, str) or self.execution_mode not in ("synthetic_smoke", "full"):
            raise ValueError("unsupported execution_mode")
        if self.execution_mode == "synthetic_smoke" and self.budget.cpu_threads != 1:
            raise ValueError("synthetic diagnostic protocol requires one CPU thread")
        if not isinstance(self.source_notebook, str) or self.source_notebook not in SOURCE_NOTEBOOKS:
            raise ValueError("source_notebook must reference approved Notebook 01/02")
        _text(self.source_cell_id, "source_cell_id")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    artifact_paths: tuple[str, ...]
    non_comparative: bool
    n03_eligible: bool

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        if not isinstance(self.status, str) or self.status not in ("in_progress", "completed", "failed", "blocked"):
            raise ValueError("unsupported run status")
        if (not isinstance(self.artifact_paths, tuple)
                or any(not isinstance(p, str) or not p.strip() for p in self.artifact_paths)
                or len(set(self.artifact_paths)) != len(self.artifact_paths)
                or (self.status == "completed" and not self.artifact_paths)):
            raise ValueError("artifact_paths must be unique immutable references; completed requires artifacts")
        if type(self.non_comparative) is not bool or type(self.n03_eligible) is not bool:
            raise ValueError("non_comparative/n03_eligible must be exact booleans")
        if self.n03_eligible and (self.non_comparative or self.status != "completed"):
            raise ValueError("N03 eligible evidence must be completed and comparative")


def validate_partition(partition: FoldPartition | TrainingPartition, registry: FeatureRegistry) -> None:
    """Revalidate mutable pandas contents and strictly authoritative raw sources.

    Validation natural keys are explicit. Training row identity is its unique
    index; this signature has no training timestamps or production split hashes.
    Actual source-cell, training-key and manifest binding belongs to Task 12 IO.
    """
    if not isinstance(partition, (FoldPartition, TrainingPartition)):
        raise TypeError("partition must be a typed FoldPartition or TrainingPartition")
    if not isinstance(registry, FeatureRegistry):
        raise TypeError("raw source registry must be an authoritative FeatureRegistry")
    _training(partition)
    if isinstance(partition, FoldPartition):
        _validation(partition)
    authoritative = build_lead_baseline_registry()
    for name in partition.X_train:
        expected, submitted = authoritative.get(name), registry.get(name)
        if expected is None or submitted != expected or submitted.source_columns != (name,):
            raise BlockedExperimentError("unverified_raw_source", f"raw source lacks verified repository producer definition: {name}")
        _column(partition.X_train, name, expected.family == "categorical")
        if isinstance(partition, FoldPartition):
            _column(partition.X_validation, name, expected.family == "categorical")


def training_view(partition: FoldPartition) -> TrainingPartition:
    """Snapshot only training fields; cannot grant source/compute approval."""
    if not isinstance(partition, FoldPartition):
        raise TypeError("training_view requires a FoldPartition")
    return TrainingPartition(partition.repeat, partition.fold, partition.X_train,
                             partition.y_train, partition.groups_train)


class _OwnerState(TypedDict):
    manifest: str
    units: list[str]
    trials: int
    allowances: dict[str, int]


class _OwnedAttempt(TypedDict):
    owner: str
    unit: str
    status: str
    evidence_sha256: str | None


class FitCounter:
    """Atomic cumulative planned capacity and debits before model fit attempts.

    reserved_* includes consumed work and is never recycled. Failed attempts
    remain used. Trial reservations are cumulative; actual trial lifecycle is
    recorded by the study owner. Bound counters use an owner-isolated persistent
    campaign journal; unowned methods retain their original in-memory behavior.
    This counter does not measure wall time/RSS.
    """

    def __init__(self, budget: FitBudget) -> None:
        if not isinstance(budget, FitBudget):
            raise TypeError("FitCounter requires a typed FitBudget")
        self._budget = budget
        self._reserved_model_fits = 0
        self._reserved_trials = 0
        self._used_model_fits = 0
        self._precharged_model_fits = 0
        self._campaign_root: Path | None = None
        self._owners: dict[str, _OwnerState] = {}
        self._attempts: dict[str, _OwnedAttempt] = {}
        self._lock = Lock()

    @property
    def budget(self) -> FitBudget:
        return self._budget

    @property
    def reserved_model_fits(self) -> int:
        with self._lock:
            return self._reserved_model_fits

    @property
    def reserved_trials(self) -> int:
        with self._lock:
            return self._reserved_trials

    @property
    def used_model_fits(self) -> int:
        with self._lock:
            return self._used_model_fits

    def reserve(self, model_fits: int, trials: int = 0) -> None:
        _integer(model_fits, "model_fits")
        _integer(trials, "trials", (0, None))
        with self._lock:
            self._require_unowned()
            if self._reserved_model_fits + model_fits > self._budget.max_model_fits:
                raise BlockedExperimentError("fit_budget_exhausted", "cumulative fit reservation exceeds budget")
            if self._reserved_trials + trials > self._budget.max_trials:
                raise BlockedExperimentError("trial_budget_exhausted", "cumulative trial reservation exceeds budget")
            self._reserved_model_fits += model_fits
            self._reserved_trials += trials

    def before_fit(self) -> None:
        with self._lock:
            self._require_unowned()
            if self._precharged_model_fits:
                self._precharged_model_fits -= 1
                return
            if self._used_model_fits >= self._reserved_model_fits:
                raise BlockedExperimentError("fit_not_reserved", "model fit is not reserved or fit budget is consumed")
            self._used_model_fits += 1

    def precharge_fit(self) -> None:
        """Charge before construction; the stable fit boundary consumes the marker."""
        with self._lock:
            self._require_unowned()
            if self._used_model_fits >= self._reserved_model_fits:
                raise BlockedExperimentError("fit_not_reserved", "model fit is not reserved or fit budget is consumed")
            self._used_model_fits += 1
            self._precharged_model_fits += 1

    def cancel_precharge(self) -> None:
        """Remove an unconsumed construction precharge after a failed construction."""
        with self._lock:
            if self._precharged_model_fits:
                self._precharged_model_fits -= 1

    def _require_unowned(self) -> None:
        if self._campaign_root is not None:
            raise BlockedExperimentError("fit_owner_required", "campaign work requires an explicit owner")

    @property
    def campaign_root(self) -> Path | None:
        return self._campaign_root

    def bind_campaign(self, root: str | Path) -> None:
        """Bind once and restore every owner's reservations and historical debits."""
        root = _validate_managed_path(root, "campaign root", directory=True).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("campaign root must be a directory")
        with self._lock:
            if self._campaign_root is None:
                if any((self._reserved_model_fits, self._reserved_trials,
                        self._used_model_fits, self._precharged_model_fits)):
                    raise BlockedExperimentError("campaign_unknown_work", "unbound counter must be pristine")
            elif self._campaign_root != root:
                raise BlockedExperimentError("campaign_binding_mismatch", "counter is bound to one campaign root")
            with self._journal(root):
                pass
            self._campaign_root = root

    @contextmanager
    def _journal(self, root: Path | None = None):
        """Serialize durable journal transitions; SQLite FULL commits precede construction."""
        root = root or self._campaign_root
        if root is None:
            raise BlockedExperimentError("campaign_unbound", "bind campaign before owned work")
        lock_path, db_path = root / ".fit_budget.lock", root / "fit_budget.sqlite3"
        _validate_managed_path(root, "campaign root", directory=True)
        for path in (lock_path, db_path, Path(str(db_path) + "-journal"),
                     Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
            _validate_managed_path(path, "campaign journal", root=root)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        conn = None
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BlockedExperimentError("campaign_busy", "campaign accounting lock is held") from error
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA synchronous = FULL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS campaign (
                    identity TEXT PRIMARY KEY, budget TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL, head_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS accounting_event (
                    sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL);
            """)
            budget_json = json.dumps(asdict(self._budget), sort_keys=True, separators=(",", ":"))
            stored = conn.execute("SELECT identity, budget FROM campaign").fetchall()
            if not stored:
                if conn.execute("SELECT COUNT(*) FROM accounting_event").fetchone()[0]:
                    raise BlockedExperimentError("campaign_tampered", "journal lacks immutable budget binding")
                initial_hash = hashlib.sha256((str(root) + budget_json).encode()).hexdigest()
                conn.execute("INSERT INTO campaign VALUES (?, ?, 0, ?)", (str(root), budget_json, initial_hash))
                conn.commit()
            elif stored != [(str(root), budget_json)]:
                raise BlockedExperimentError("study_budget_mismatch", "campaign exact fit budget or root differs")
            self._restore_journal(conn, str(root), budget_json)
            yield conn
        finally:
            if conn is not None:
                conn.close()
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _restore_journal(self, conn: sqlite3.Connection, root: str, budget_json: str) -> None:
        owners: dict[str, _OwnerState] = {}
        attempts: dict[str, _OwnedAttempt] = {}
        previous = hashlib.sha256((root + budget_json).encode()).hexdigest()
        sequence = 0
        try:
            for expected, (sequence, payload, prev_hash, event_hash) in enumerate(
                conn.execute("SELECT sequence, payload, previous_hash, event_hash FROM accounting_event ORDER BY sequence"), 1
            ):
                digest = hashlib.sha256((previous + payload).encode()).hexdigest()
                if sequence != expected or prev_hash != previous or event_hash != digest:
                    raise ValueError("accounting journal hash/sequence mismatch")
                event = json.loads(payload)
                kind = event["kind"]
                if kind == "owner_claimed":
                    owner = event["owner"]
                    if owner in owners or len(set(event["units"])) != len(event["units"]):
                        raise ValueError("duplicate owner or fit units")
                    owners[owner] = {"manifest": event["manifest"], "units": event["units"],
                                     "trials": event["trials"], "allowances": dict.fromkeys(event["units"], 1)}
                elif kind == "owner_retry_reserved":
                    owner = owners[event["owner"]]
                    for unit in event["units"]:
                        owner["allowances"][unit] += 1
                elif kind == "attempt_started":
                    owner, unit, attempt = event["owner"], event["unit"], event["attempt_id"]
                    count = sum(a["owner"] == owner and a["unit"] == unit for a in attempts.values())
                    if attempt in attempts or count >= owners[owner]["allowances"][unit]:
                        raise ValueError("attempt exceeds owned reservation")
                    attempts[attempt] = {"owner": owner, "unit": unit, "status": "started", "evidence_sha256": None}
                elif kind == "attempt_evidence_sealed":
                    attempt = attempts[event["attempt_id"]]
                    proof = event["evidence_sha256"]
                    if (attempt["status"] != "started" or attempt["evidence_sha256"] is not None
                            or not _is_sha256(proof)):
                        raise ValueError("invalid immutable evidence seal")
                    attempt["evidence_sha256"] = proof
                elif kind == "attempt_finished":
                    attempt = attempts[event["attempt_id"]]
                    if attempt["status"] != "started" or event["status"] not in ("completed", "failed", "interrupted"):
                        raise ValueError("invalid attempt transition")
                    if event["status"] == "completed" and attempt["unit"].startswith("inner:") and attempt["evidence_sha256"] is None:
                        raise ValueError("legacy inner completion lacks sealed hash provenance")
                    attempt["status"] = event["status"]
                else:
                    raise ValueError("unknown accounting event")
                previous = digest
            if conn.execute("SELECT last_sequence, head_hash FROM campaign").fetchone() != (sequence, previous):
                raise ValueError("accounting journal head mismatch")
            reserved = sum(sum(o["allowances"].values()) for o in owners.values())
            trials = sum(o["trials"] for o in owners.values())
            if reserved > self._budget.max_model_fits or trials > self._budget.max_trials:
                raise ValueError("journal exceeds immutable budget")
        except (KeyError, TypeError, ValueError) as error:
            raise BlockedExperimentError("campaign_tampered", str(error)) from error
        self._owners, self._attempts = owners, attempts
        self._reserved_model_fits, self._reserved_trials = reserved, trials
        self._used_model_fits = len(attempts)

    def _append_event(self, conn: sqlite3.Connection, event: dict) -> None:
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
        last = conn.execute("SELECT sequence, event_hash FROM accounting_event ORDER BY sequence DESC LIMIT 1").fetchone()
        root, budget_json = conn.execute("SELECT identity, budget FROM campaign").fetchone()
        sequence, previous = (last[0] + 1, last[1]) if last else (1, hashlib.sha256((root + budget_json).encode()).hexdigest())
        digest = hashlib.sha256((previous + payload).encode()).hexdigest()
        conn.execute("INSERT INTO accounting_event VALUES (?, ?, ?, ?)", (sequence, payload, previous, digest))
        conn.execute("UPDATE campaign SET last_sequence = ?, head_hash = ?", (sequence, digest))
        conn.commit()
        self._restore_journal(conn, root, budget_json)

    def claim_owner(self, owner: str, manifest: str, units: tuple[str, ...], trials: int,
                    retry_units: tuple[str, ...] = ()) -> None:
        """Idempotent initial plan plus preflight reservations for exhausted unfinished units."""
        _text(owner, "owner")
        _text(manifest, "manifest")
        _integer(trials, "trials", (0, None))
        if not isinstance(units, tuple) or not units or len(set(units)) != len(units):
            raise ValueError("owned units must be a unique nonempty tuple")
        for unit in units:
            _text(unit, "unit")
        if not isinstance(retry_units, tuple) or len(set(retry_units)) != len(retry_units) or not set(retry_units) <= set(units):
            raise ValueError("retry units must be unique owned units")
        with self._lock, self._journal() as conn:
            existing = self._owners.get(owner)
            if existing is not None and (existing["manifest"], existing["units"], existing["trials"]) != (manifest, list(units), trials):
                raise BlockedExperimentError("study_manifest_mismatch", "owner manifest or plan changed")
            extras = []
            if existing is not None:
                for unit in retry_units:
                    history = [a for a in self._attempts.values() if a["owner"] == owner and a["unit"] == unit]
                    if any(a["status"] == "completed" for a in history):
                        raise BlockedExperimentError("campaign_tampered", "completed owned unit cannot be retried")
                    if len(history) >= existing["allowances"][unit]:
                        extras.append(unit)
            additional_fits = len(extras) if existing else len(units)
            if self._reserved_model_fits + additional_fits > self._budget.max_model_fits:
                raise BlockedExperimentError("fit_budget_exhausted", "owned campaign fit reservation exceeds budget")
            if self._reserved_trials + (0 if existing else trials) > self._budget.max_trials:
                raise BlockedExperimentError("trial_budget_exhausted", "owned campaign trial reservation exceeds budget")
            if existing is None:
                self._append_event(conn, {"kind": "owner_claimed", "owner": owner, "manifest": manifest,
                                          "units": units, "trials": trials})
            if extras:
                self._append_event(conn, {"kind": "owner_retry_reserved", "owner": owner, "units": extras})

    def before_owned_fit(self, owner: str, unit: str) -> str:
        """Durably debit one unique attempt before construction, without a free-fit marker."""
        with self._lock, self._journal() as conn:
            owned = self._owners.get(owner)
            count = sum(a["owner"] == owner and a["unit"] == unit for a in self._attempts.values())
            if owned is None or unit not in owned["allowances"] or count >= owned["allowances"][unit]:
                raise BlockedExperimentError("fit_not_reserved", "fit has no remaining owned reservation")
            if any(a["owner"] == owner and a["unit"] == unit and a["status"] == "completed" for a in self._attempts.values()):
                raise BlockedExperimentError("fit_already_completed", "owned fit unit already completed")
            attempt_id = uuid.uuid4().hex
            self._append_event(conn, {"kind": "attempt_started", "owner": owner, "unit": unit,
                                      "attempt_id": attempt_id})
            return attempt_id

    def finish_owned_fit(self, attempt_id: str, status: str) -> None:
        """Record an attempt outcome; failures never refund the durable debit."""
        if status not in ("completed", "failed", "interrupted"):
            raise ValueError("invalid owned attempt status")
        with self._lock, self._journal() as conn:
            attempt = self._attempts.get(attempt_id)
            if attempt is None or attempt["status"] != "started":
                raise BlockedExperimentError("campaign_tampered", "owned attempt is unknown or already finished")
            if status == "completed" and attempt["unit"].startswith("inner:") and attempt["evidence_sha256"] is None:
                raise BlockedExperimentError("campaign_tampered", "inner completion requires sealed evidence provenance")
            self._append_event(conn, {"kind": "attempt_finished", "attempt_id": attempt_id, "status": status})

    def seal_owned_evidence(self, attempt_id: str, evidence_sha256: str) -> None:
        """Durably bind fsynced evidence to a started attempt before SQL completion."""
        if not _is_sha256(evidence_sha256):
            raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")
        with self._lock, self._journal() as conn:
            attempt = self._attempts.get(attempt_id)
            if attempt is None or attempt["status"] != "started" or attempt["evidence_sha256"] is not None:
                raise BlockedExperimentError("campaign_tampered", "attempt evidence can only be sealed once while started")
            self._append_event(conn, {"kind": "attempt_evidence_sealed", "attempt_id": attempt_id,
                                      "evidence_sha256": evidence_sha256})

    def owner_attempt_evidence(self, owner: str) -> tuple[tuple[str, str | None], ...]:
        """Read immutable attempt/hash bindings without altering the existing attempt API."""
        with self._lock, self._journal():
            return tuple((key, a["evidence_sha256"]) for key, a in self._attempts.items() if a["owner"] == owner)

    def owner_attempts(self, owner: str) -> tuple[tuple[str, str, str], ...]:
        """Return immutable (attempt_id, unit, status) journal records for one owner."""
        with self._lock, self._journal():
            return tuple((key, a["unit"], a["status"]) for key, a in self._attempts.items() if a["owner"] == owner)

    def owner_reserved_model_fits(self, owner: str) -> int:
        with self._lock, self._journal():
            if owner not in self._owners:
                raise BlockedExperimentError("fit_owner_unknown", "unknown owner")
            return sum(self._owners[owner]["allowances"].values())
