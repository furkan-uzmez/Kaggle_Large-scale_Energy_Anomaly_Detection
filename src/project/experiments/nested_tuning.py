"""Bounded HGB random search with SQLite study evidence.

Initial family is HGB only; objective is mean inner ROC-AUC across fixed
verified group folds. Sample a predeclared finite ParameterSampler space
deterministically without an optional Optuna dependency.
"""

from __future__ import annotations

import datetime
import errno
import fcntl
import hashlib
import inspect
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project.experiments.candidate_contracts import (
    STUDY_SPACE,
    BlockedExperimentError,
    FitBudget,
    FitCounter,
    FoldPartition,
    StudySpec,
    TrainingPartition,
    _validate_managed_path,
    validate_partition,
)
from project.experiments.inner_cv import (
    _typed_identity,
    build_inner_folds,
    inner_membership_digest,
)
from project.features.registry import FeatureMetadata, build_lead_baseline_registry
from project.models.candidate_factories import (
    _CandidateArm,
    _pipeline,
    _PredictorSchema,
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "meter_reading",
    "square_feet",
    "year_built",
    "floor_count",
    "hour",
    "weekday",
    "month",
    "year",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("primary_use",)
CORE_FEATURE_POOL: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_EVIDENCE_FIELDS: frozenset[str] = frozenset({
    "trial_id",
    "fold_index",
    "is_default",
    "params_hash",
    "validation_positions",
    "probabilities",
    "score",
    "sha256",
})

DEFAULT_HGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.0,
    "max_iter": 100,
}


def resolve_campaign_root(root: str | Path | None = None) -> Path:
    """Resolve and validate the actual campaign root directory."""
    if root is None:
        candidate = Path(__file__).resolve().parents[3]
        if candidate.is_dir() and (candidate / "src").is_dir():
            root = candidate
        else:
            root = Path.cwd()
    candidate = Path(root)
    _reject_symlink_path(candidate, "campaign root")
    resolved = candidate.absolute()
    if not resolved.is_dir():
        raise ValueError(f"campaign root must be an existing directory: {resolved}")
    return resolved


def _reject_symlink_path(path: Path, label: str) -> None:
    """Reject symlink components so managed paths cannot escape their root."""
    _validate_managed_path(path, label, directory=path.is_dir())


def _validate_study_dir(study_dir: str | Path) -> Path:
    """Validate and create the temporary study root without following links."""
    candidate = Path(study_dir)
    _reject_symlink_path(candidate, "study directory")
    absolute = candidate.absolute()
    temp_root = Path(tempfile.gettempdir()).absolute()
    try:
        absolute.relative_to(temp_root)
    except ValueError as error:
        raise ValueError("study_dir must be confined beneath the system temporary directory") from error

    if absolute.exists() and not absolute.is_dir():
        raise ValueError(f"study_dir must be a directory: {absolute}")
    absolute.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(absolute, "study directory")
    return absolute


def _validate_database_path(study_dir: Path, db_path: Path) -> None:
    """Validate the SQLite path before SQLite can create or open it."""
    _validate_managed_path(db_path, "study database", root=study_dir)
    try:
        db_path.absolute().relative_to(study_dir.absolute())
    except ValueError as error:
        raise ValueError("study database must remain beneath study_dir") from error
    if db_path.exists() and not db_path.is_file():
        raise ValueError(f"study database must be a regular file: {db_path}")
    for suffix in ("-journal", "-wal", "-shm"):
        _validate_managed_path(Path(str(db_path) + suffix), "study database sidecar", root=study_dir)


def _canonical_evidence_path(study_dir: Path, study_key: str, trial_id: int, fold_index: int) -> Path:
    managed = study_dir / "fold_evidence"
    if study_key != "study_r0_f0":
        managed /= study_key
    _validate_managed_path(managed, "fold evidence directory", root=study_dir, directory=True)
    path = managed / f"trial_{trial_id}_fold_{fold_index}.json"
    return _validate_managed_path(path, "fold evidence file", root=managed)


@contextmanager
def _study_lock(study_dir: Path, name: str = ".lock"):
    """Hold a persistent nonblocking lock for the complete study operation."""
    lock_path = study_dir / name
    _validate_database_path(study_dir, lock_path)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError("study lock must not be a symlink") from error
        raise
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise BlockedExperimentError("study_busy", "study lock is held by another invocation") from error
            raise
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def get_hgb_signature_defaults() -> dict[str, Any]:
    """Extract default constructor parameters for HistGradientBoostingClassifier."""
    sig = inspect.signature(HistGradientBoostingClassifier.__init__)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if k != "self" and v.default is not inspect.Parameter.empty
    }


def get_canonical_arm_description(params: dict[str, Any]) -> dict[str, Any]:
    """Shared canonical resolver for expected structural and parameter specification."""
    resolved_params = {**DEFAULT_HGB_PARAMS, **params}
    hgb_defaults = get_hgb_signature_defaults()
    for k in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter"):
        hgb_defaults[k] = resolved_params[k]
    hgb_defaults["early_stopping"] = False
    hgb_defaults["random_state"] = 42
    hgb_defaults["class_weight"] = None

    return {
        "arm_class": "_CandidateArm",
        "weight_mode": "natural",
        "comparison_factor": "nested_hgb_tuning",
        "native_numeric": [],
        "steps": ["schema", "preprocess", "model"],
        "schema": {
            "class": "_PredictorSchema",
            "columns": list(CORE_FEATURE_POOL),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "preprocess": {
            "class": "ColumnTransformer",
            "remainder": "drop",
            "sparse_threshold": 0.0,
            "numeric_branch": {
                "name": "numeric",
                "class": "SimpleImputer",
                "strategy": "median",
                "keep_empty_features": True,
                "columns": list(NUMERIC_FEATURES),
            },
            "categorical_branch": {
                "name": "categorical",
                "class": "Pipeline",
                "columns": list(CATEGORICAL_FEATURES),
                "steps": [
                    {
                        "name": "imputer",
                        "class": "SimpleImputer",
                        "strategy": "most_frequent",
                        "keep_empty_features": True,
                    },
                    {
                        "name": "one_hot",
                        "class": "OneHotEncoder",
                        "handle_unknown": "ignore",
                        "sparse_output": False,
                    },
                ],
            },
        },
        "model": {
            "class": "HistGradientBoostingClassifier",
            "early_stopping": False,
            "random_state": 42,
            "class_weight": None,
            "sampled_tunables": {
                k: resolved_params[k]
                for k in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter")
            },
            "resolved_parameters": hgb_defaults,
        },
    }


def required_model_fits(n_trials: int, inner_folds: int, score_inner_default: bool = False) -> int:
    """Calculate required model fits: T * K + 2 (+ K if score_inner_default)."""
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError("n_trials must be a positive integer; booleans are invalid")
    if type(inner_folds) is not int or not (2 <= inner_folds <= 5):
        raise ValueError("inner_folds must be an integer in [2, 5]; booleans are invalid")
    if type(score_inner_default) is not bool:
        raise TypeError("score_inner_default must be an exact boolean")

    fits = n_trials * inner_folds + 2
    if score_inner_default:
        fits += inner_folds
    return fits


@dataclass(frozen=True)
class TuningResult:
    """Immutable tuning outcome with selected params, hashes, and fit accounting."""

    selected_params: tuple[tuple[str, object], ...]
    trial_id: int
    study_manifest_hash: str
    planned_model_fits: int
    used_model_fits: int
    manifest_reference: str
    study_charged_model_fits: int = 0
    invocation_model_fits: int = 0
    campaign_total_model_fits: int = 0
    owned_reserved_model_fits: int = 0
    owner_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.selected_params, tuple):
            raise TypeError("selected_params must be an immutable tuple of parameter pairs")
        for pair in self.selected_params:
            if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str):
                raise ValueError("selected_params entries must be (name, value) pairs")
        if type(self.trial_id) is not int or self.trial_id < 0:
            raise ValueError("trial_id must be a nonnegative integer")
        if not isinstance(self.study_manifest_hash, str) or not self.study_manifest_hash:
            raise ValueError("study_manifest_hash must be a nonempty string")
        if type(self.planned_model_fits) is not int or self.planned_model_fits < 1:
            raise ValueError("planned_model_fits must be a positive integer")
        if type(self.used_model_fits) is not int or self.used_model_fits < 0:
            raise ValueError("used_model_fits must be a nonnegative integer")
        if not isinstance(self.manifest_reference, str) or not self.manifest_reference:
            raise ValueError("manifest_reference must be a nonempty string")


def build_hgb_tuning_arm(
    params: dict[str, Any] | tuple[tuple[str, object], ...] | None = None,
) -> Pipeline:
    """Build fresh natural HGB arm using Task 7 schema/preprocess and four approved tunables."""
    if params is None:
        params_dict: dict[str, Any] = {}
    elif isinstance(params, (tuple, dict)):
        params_dict = dict(params)
    else:
        raise TypeError("params must be a mapping, tuple of pairs, or None")

    approved_tunables = {"learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter"}
    for k in params_dict:
        if k not in approved_tunables:
            raise ValueError(f"unapproved parameter for HGB tuning arm: {k}")

    resolved = {**DEFAULT_HGB_PARAMS, **params_dict}

    for name, allowed in STUDY_SPACE.items():
        val = resolved[name]
        if type(val) in (bool, np.bool_):
            raise TypeError(f"parameter {name} cannot be boolean")
        if name in ("max_iter", "max_leaf_nodes"):
            if type(val) not in (int, np.integer) or val not in allowed:
                raise ValueError(f"parameter {name} must be integer in {allowed}")
            resolved[name] = int(val)
        else:
            if type(val) not in (int, float, np.floating) or val not in allowed:
                raise ValueError(f"parameter {name} must be float in {allowed}")
            resolved[name] = float(val)

    model = HistGradientBoostingClassifier(
        early_stopping=False,
        random_state=42,
        class_weight=None,
        **resolved,
    )
    return _pipeline(
        CORE_FEATURE_POOL,
        CATEGORICAL_FEATURES,
        False,
        model,
        "nested_hgb_tuning",
        "natural",
    )


def _generate_trial_params(study_spec: StudySpec) -> list[dict[str, Any]]:
    """Sample finite ParameterSampler space deterministically without replacement."""
    param_grid = {name: sorted(values) for name, values in sorted(study_spec.parameter_lists)}
    sampler = ParameterSampler(param_grid, n_iter=study_spec.n_trials, random_state=study_spec.seed)
    trials: list[dict[str, Any]] = []
    for sampled in sampler:
        full = {**DEFAULT_HGB_PARAMS, **sampled}
        trials.append(full)
    return trials


def serialize_feature_metadata(meta: FeatureMetadata) -> dict[str, Any]:
    """Serialize raw FeatureMetadata into truthful JSON dictionary."""
    return {
        "name": meta.name,
        "family": meta.family,
        "source_columns": list(meta.source_columns),
        "semantic_meaning": meta.semantic_meaning,
        "prediction_time_availability": meta.prediction_time_availability.value,
        "target_dependence": meta.target_dependence.value,
        "transform_fit_boundary": meta.transform_fit_boundary.value,
        "fallback_behavior": meta.fallback_behavior,
        "resource_cost": meta.resource_cost.value,
        "provenance_status": meta.provenance_status.value,
        "available_at": meta.available_at.isoformat() if meta.available_at is not None else None,
    }


def build_study_manifest_payload(
    training: TrainingPartition,
    study_spec: StudySpec,
    score_inner_default: bool,
    membership_digest: str,
    budget: FitBudget | FitCounter | None = None,
    deterministic_trials: list[dict[str, Any]] | None = None,
    campaign_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build full immutable manifest payload binding all dimensions and configurations."""
    baseline_reg = build_lead_baseline_registry()
    registry_defs = {}
    columns_info = []
    for col in training.X_train.columns:
        meta = baseline_reg.get(col)
        if meta is None:
            raise ValueError(f"unregistered feature in training predictors: {col}")
        registry_defs[col] = serialize_feature_metadata(meta)
        columns_info.append({
            "name": col,
            "family": meta.family,
            "dtype": str(training.X_train[col].dtype),
            "storage": training.X_train[col].dtype.name,
        })

    X_bytes = pd.util.hash_pandas_object(training.X_train, index=True).to_numpy().tobytes()
    values_hash = hashlib.sha256(X_bytes).hexdigest()

    y_bytes = training.y_train.to_numpy(dtype=np.int64).tobytes()
    labels_hash = hashlib.sha256(y_bytes).hexdigest()

    row_identities = [_typed_identity(idx) for idx in training.X_train.index.tolist()]
    group_identities = [_typed_identity(g) for g in training.groups_train.tolist()]

    if budget is not None:
        fit_budget = budget.budget if isinstance(budget, FitCounter) else budget
        budget_dict = {
            "max_model_fits": fit_budget.max_model_fits,
            "max_trials": fit_budget.max_trials,
            "max_wall_seconds": fit_budget.max_wall_seconds,
            "max_rss_bytes": fit_budget.max_rss_bytes,
            "cpu_threads": fit_budget.cpu_threads,
        }
    else:
        budget_dict = {
            "max_model_fits": 0,
            "max_trials": 0,
            "max_wall_seconds": 0.0,
            "max_rss_bytes": 0,
            "cpu_threads": 1,
        }

    if deterministic_trials is None:
        deterministic_trials = _generate_trial_params(study_spec)

    root_path = resolve_campaign_root(campaign_root)
    resolved_canonical_pipeline = get_canonical_arm_description(DEFAULT_HGB_PARAMS)

    payload = {
        "recipe": "nested_hgb_tuning",
        "version": "nested_tuning_hgb_v1",
        "repeat": training.repeat,
        "outer_fold": training.fold,
        "ordered_columns": list(training.X_train.columns),
        "columns_info": columns_info,
        "registry_defs": registry_defs,
        "values_hash": values_hash,
        "labels_hash": labels_hash,
        "typed_row_identities": row_identities,
        "typed_group_identities": group_identities,
        "v2_inner_membership_digest": membership_digest,
        "deterministic_trial_list": deterministic_trials,
        "exact_study_spec": {
            "n_trials": study_spec.n_trials,
            "inner_folds": study_spec.inner_folds,
            "seed": study_spec.seed,
            "parameter_lists": [list(item) for item in sorted(study_spec.parameter_lists)],
        },
        "score_inner_default": score_inner_default,
        "default_config": DEFAULT_HGB_PARAMS,
        "objective": "mean_inner_roc_auc",
        "positive_label": 1,
        "tie_break_strategy": "deterministic_trial_id",
        "env_versions": {
            "protocol": "nested_tuning_hgb_v2",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "original_exact_fit_budget": budget_dict,
        "campaign_root_identity": str(root_path),
        "resolved_canonical_pipeline": resolved_canonical_pipeline,
        "resolved_fixed_config": {
            "early_stopping": False,
            "random_state": 42,
            "class_weight": None,
            "comparison_factor": "nested_hgb_tuning",
            "weight_mode": "natural",
        },
    }
    return payload


def compute_study_manifest_hash(
    training: TrainingPartition,
    study_spec: StudySpec,
    score_inner_default: bool,
    membership_digest: str,
    budget: FitBudget | FitCounter | None = None,
    deterministic_trials: list[dict[str, Any]] | None = None,
    campaign_root: str | Path | None = None,
) -> str:
    """Compute deterministic SHA-256 study manifest hash from full manifest payload."""
    payload = build_study_manifest_payload(
        training=training,
        study_spec=study_spec,
        score_inner_default=score_inner_default,
        membership_digest=membership_digest,
        budget=budget,
        deterministic_trials=deterministic_trials,
        campaign_root=campaign_root,
    )
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _init_sqlite_tables(conn: sqlite3.Connection) -> None:
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
            env_version TEXT NOT NULL,
            n_trials INTEGER NOT NULL,
            inner_folds INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            planned_model_fits INTEGER NOT NULL,
            score_inner_default INTEGER NOT NULL,
            default_params_json TEXT NOT NULL,
            selected_trial_id INTEGER,
            selected_params_json TEXT,
            failure_reason TEXT,
            charged_model_fits INTEGER NOT NULL DEFAULT 0,
            last_invocation_model_fits INTEGER NOT NULL DEFAULT 0,
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
            is_default INTEGER NOT NULL,
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
            evidence_path TEXT,
            evidence_sha256 TEXT,
            PRIMARY KEY (study_key, trial_id, fold_index)
        );

        CREATE TABLE IF NOT EXISTS fit_attempt (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_key TEXT NOT NULL,
            trial_id INTEGER NOT NULL,
            fold_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            failure_reason TEXT,
            timestamp TEXT NOT NULL,
            UNIQUE (study_key, trial_id, fold_index, attempt_id)
        );
    """)
    study_columns = {row[1] for row in conn.execute("PRAGMA table_info(study)")}
    if "charged_model_fits" not in study_columns:
        conn.execute("ALTER TABLE study ADD COLUMN charged_model_fits INTEGER NOT NULL DEFAULT 0")
    if "last_invocation_model_fits" not in study_columns:
        conn.execute("ALTER TABLE study ADD COLUMN last_invocation_model_fits INTEGER NOT NULL DEFAULT 0")
    fold_columns = {row[1] for row in conn.execute("PRAGMA table_info(fold_score)")}
    if (not {"evidence_path", "evidence_sha256", "owned_attempt_id"} <= fold_columns
            and conn.execute("SELECT 1 FROM fold_score WHERE status = 'completed' LIMIT 1").fetchone()):
        raise BlockedExperimentError("study_tampered", "legacy completed fold schema lacks hash/attempt provenance")
    if "evidence_path" not in fold_columns:
        conn.execute("ALTER TABLE fold_score ADD COLUMN evidence_path TEXT")
    if "evidence_sha256" not in fold_columns:
        conn.execute("ALTER TABLE fold_score ADD COLUMN evidence_sha256 TEXT")
    attempt_columns = {row[1] for row in conn.execute("PRAGMA table_info(fit_attempt)")}
    if (not {"owned_attempt_id", "evidence_sha256"} <= attempt_columns
            and conn.execute("SELECT 1 FROM fit_attempt WHERE status = 'completed' LIMIT 1").fetchone()):
        raise BlockedExperimentError("study_tampered", "legacy completed attempt schema lacks hash provenance")
    if "owned_attempt_id" not in attempt_columns:
        conn.execute("ALTER TABLE fit_attempt ADD COLUMN owned_attempt_id TEXT")
    if "evidence_sha256" not in attempt_columns:
        conn.execute("ALTER TABLE fit_attempt ADD COLUMN evidence_sha256 TEXT")
    if "owned_attempt_id" not in fold_columns:
        conn.execute("ALTER TABLE fold_score ADD COLUMN owned_attempt_id TEXT")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS owned_attempt_identity ON fit_attempt(owned_attempt_id)")
    conn.commit()


def _insert_fold_score(
    conn: sqlite3.Connection,
    study_key: str,
    trial_id: int,
    fold_idx: int,
    score: float | None,
    status: str,
    failure_reason: str | None = None,
    evidence_path: str | None = None,
    evidence_sha256: str | None = None,
    owned_attempt_id: str | None = None,
) -> None:
    """Insert one immutable fold outcome; conflicting duplicates fail closed."""
    if status not in {"completed", "failed", "interrupted"}:
        raise BlockedExperimentError("study_tampered", f"unknown fold status: {status}")
    existing = conn.execute(
        "SELECT score, status, failure_reason, evidence_path, evidence_sha256, owned_attempt_id FROM fold_score "
        "WHERE study_key = ? AND trial_id = ? AND fold_index = ?",
        (study_key, trial_id, fold_idx),
    ).fetchone()
    if existing is not None:
        if existing[1] in {"failed", "interrupted"} and status in {"completed", "failed", "interrupted"}:
            conn.execute(
                "UPDATE fold_score SET score = ?, status = ?, failure_reason = ?, evidence_path = ?, evidence_sha256 = ?, owned_attempt_id = ? "
                "WHERE study_key = ? AND trial_id = ? AND fold_index = ?",
                (score, status, failure_reason, evidence_path, evidence_sha256, owned_attempt_id, study_key, trial_id, fold_idx),
            )
            return
        if existing != (score, status, failure_reason, evidence_path, evidence_sha256, owned_attempt_id):
            raise BlockedExperimentError("study_tampered", "conflicting fold score already exists")
        return
    conn.execute(
        "INSERT INTO fold_score "
        "(study_key, trial_id, fold_index, score, status, failure_reason, evidence_path, evidence_sha256, owned_attempt_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (study_key, trial_id, fold_idx, score, status, failure_reason, evidence_path, evidence_sha256, owned_attempt_id),
    )


def _extract_pos_proba(arm: Any, X: pd.DataFrame) -> np.ndarray:
    proba = arm.predict_proba(X)
    if isinstance(proba, pd.DataFrame):
        proba = proba.to_numpy()
    proba = np.asarray(proba)
    if proba.dtype.kind not in "iuf":
        raise TypeError("inner positive-class probabilities must use real nonboolean numeric storage")
    if proba.ndim == 2 and proba.shape[1] == 2:
        classes = getattr(arm, "classes_", None)
        if classes is not None:
            classes_list = list(classes)
            if len(classes_list) != 2 or set(classes_list) != {0, 1}:
                raise ValueError("inner model classes must identify binary labels 0 and 1")
            proba = proba[:, classes_list.index(1)]
        else:
            proba = proba[:, 1]
    if proba.ndim != 1 or len(proba) != len(X):
        raise ValueError("inner positive-class probabilities must be one-dimensional and row-aligned")
    if not np.isfinite(proba).all() or ((proba < 0.0) | (proba > 1.0)).any():
        raise ValueError("inner positive-class probabilities must be finite values in [0, 1]")
    return proba


def _validate_constructed_arm(arm: Any, params: dict[str, Any]) -> None:
    """Ensure constructed arm is fresh, unfitted, and conforms to Task 7 canonical structural description."""
    if not isinstance(arm, Pipeline) or not isinstance(arm, _CandidateArm):
        raise TypeError("constructed arm must be a Task 7 canonical _CandidateArm Pipeline")

    # Step names
    if [name for name, _ in arm.steps] != ["schema", "preprocess", "model"]:
        raise ValueError("constructed arm steps must strictly match ['schema', 'preprocess', 'model']")

    # Candidate arm attributes
    if arm.weight_mode != "natural":
        raise ValueError(f"arm weight_mode must be 'natural', got {arm.weight_mode!r}")
    if arm.comparison_factor != "nested_hgb_tuning":
        raise ValueError(f"arm comparison_factor must be 'nested_hgb_tuning', got {arm.comparison_factor!r}")
    if arm.native_numeric != ():
        raise ValueError("arm native_numeric must be empty () indicating non-native median numeric imputation")

    # Schema validation
    schema = arm.named_steps["schema"]
    if not isinstance(schema, _PredictorSchema):
        raise TypeError(f"schema step must be _PredictorSchema, got {type(schema).__name__}")
    if getattr(schema, "columns", None) != CORE_FEATURE_POOL:
        raise ValueError(f"schema columns mismatch: expected {CORE_FEATURE_POOL}, got {getattr(schema, 'columns', None)}")
    if getattr(schema, "categorical", None) != CATEGORICAL_FEATURES:
        raise ValueError(f"schema categorical columns mismatch: expected {CATEGORICAL_FEATURES}, got {getattr(schema, 'categorical', None)}")

    # Preprocess validation
    preprocess = arm.named_steps["preprocess"]
    if not isinstance(preprocess, ColumnTransformer):
        raise TypeError(f"preprocess step must be ColumnTransformer, got {type(preprocess).__name__}")
    if preprocess.remainder != "drop" or preprocess.sparse_threshold != 0.0:
        raise ValueError("ColumnTransformer remainder must be 'drop' and sparse_threshold must be 0.0")

    transformers = preprocess.transformers
    if len(transformers) != 2:
        raise ValueError(f"ColumnTransformer must declare exactly 2 branches, got {len(transformers)}")

    # Numeric branch
    num_name, num_trans, num_cols = transformers[0]
    if num_name != "numeric":
        raise ValueError(f"first branch name must be 'numeric', got {num_name!r}")
    if tuple(num_cols) != NUMERIC_FEATURES:
        raise ValueError(f"numeric branch columns mismatch: expected {NUMERIC_FEATURES}, got {tuple(num_cols)}")
    if not isinstance(num_trans, SimpleImputer):
        raise TypeError(f"numeric transformer must be SimpleImputer, got {type(num_trans).__name__}")
    if num_trans.strategy != "median":
        raise ValueError(f"numeric SimpleImputer strategy must be 'median', got {num_trans.strategy!r}")
    if num_trans.keep_empty_features is not True:
        raise ValueError("numeric SimpleImputer keep_empty_features must be True")

    # Categorical branch
    cat_name, cat_trans, cat_cols = transformers[1]
    if cat_name != "categorical":
        raise ValueError(f"second branch name must be 'categorical', got {cat_name!r}")
    if tuple(cat_cols) != CATEGORICAL_FEATURES:
        raise ValueError(f"categorical branch columns mismatch: expected {CATEGORICAL_FEATURES}, got {tuple(cat_cols)}")
    if not isinstance(cat_trans, Pipeline):
        raise TypeError(f"categorical transformer must be Pipeline, got {type(cat_trans).__name__}")
    if [name for name, _ in cat_trans.steps] != ["imputer", "one_hot"]:
        raise ValueError(f"categorical branch steps mismatch: expected ['imputer', 'one_hot'], got {[name for name, _ in cat_trans.steps]}")

    cat_imp = cat_trans.named_steps["imputer"]
    if not isinstance(cat_imp, SimpleImputer) or cat_imp.strategy != "most_frequent" or cat_imp.keep_empty_features is not True:
        raise ValueError("categorical imputer must be SimpleImputer with strategy='most_frequent' and keep_empty_features=True")

    cat_ohe = cat_trans.named_steps["one_hot"]
    if not isinstance(cat_ohe, OneHotEncoder) or cat_ohe.handle_unknown != "ignore" or cat_ohe.sparse_output is not False:
        raise ValueError("categorical encoder must be OneHotEncoder with handle_unknown='ignore' and sparse_output=False")

    # Model validation
    model = arm.named_steps["model"]
    if not isinstance(model, HistGradientBoostingClassifier):
        raise TypeError(f"arm model must be HistGradientBoostingClassifier, got {type(model).__name__}")
    if model.early_stopping is not False:
        raise ValueError(f"model early_stopping must be False, got {model.early_stopping!r}")
    if model.random_state != 42:
        raise ValueError(f"model random_state must be 42, got {model.random_state!r}")
    if model.class_weight is not None:
        raise ValueError(f"model class_weight must be None, got {model.class_weight!r}")

    resolved_params = {**DEFAULT_HGB_PARAMS, **params}
    for k in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter"):
        if getattr(model, k) != resolved_params[k]:
            raise ValueError(f"model parameter {k} mismatch: expected {resolved_params[k]}, got {getattr(model, k)}")

    signature_defaults = get_hgb_signature_defaults()
    for k, expected_default in signature_defaults.items():
        if k not in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_iter", "early_stopping", "random_state", "class_weight"):
            actual_val = getattr(model, k)
            if actual_val != expected_default:
                raise ValueError(f"model parameter {k} mismatch from signature default: expected {expected_default!r}, got {actual_val!r}")

    # Unfitted markers check across ALL components
    components_to_check = [
        ("arm", arm),
        ("schema", schema),
        ("preprocess", preprocess),
        ("num_imputer", num_trans),
        ("cat_pipeline", cat_trans),
        ("cat_imputer", cat_imp),
        ("cat_encoder", cat_ohe),
        ("model", model),
    ]
    for comp_name, comp in components_to_check:
        fitted_attrs = [attr for attr in vars(comp) if attr.endswith("_") and not attr.startswith("_")]
        if fitted_attrs:
            raise ValueError(
                f"constructed arm must be a fresh unfitted estimator, found fitted {comp_name} attributes: {fitted_attrs}"
            )


def _fit_and_score_fold(
    arm: Pipeline,
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    groups_train_fold: pd.Series,
    X_valid_fold: pd.DataFrame,
    y_valid_fold: pd.Series,
    fit_counter: FitCounter,
    conn: sqlite3.Connection,
    study_key: str,
    trial_id: int,
    fold_idx: int,
    is_def: int,
    params_hash: str,
    valid_pos: list[int] | np.ndarray,
    study_dir_path: Path,
) -> float:
    owned_attempt_id = conn.execute(
        "SELECT owned_attempt_id FROM fit_attempt WHERE study_key = ? AND trial_id = ? AND fold_index = ? "
        "AND status = 'started' ORDER BY attempt_id DESC LIMIT 1", (study_key, trial_id, fold_idx),
    ).fetchone()[0]
    try:
        arm.fit(X_train_fold, y_train_fold)

        pos_proba = _extract_pos_proba(arm, X_valid_fold)
        score = float(roc_auc_score(y_valid_fold.to_numpy(), pos_proba))
        if not math.isfinite(score):
            raise ValueError(f"non-finite ROC-AUC score computed on fold: {score}")

        evidence_path = _canonical_evidence_path(study_dir_path, study_key, trial_id, fold_idx)
        evidence_dir = evidence_path.parent
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _reject_symlink_path(evidence_dir, "fold evidence directory")

        evidence_path = _canonical_evidence_path(study_dir_path, study_key, trial_id, fold_idx)

        evidence_payload = {
            "trial_id": int(trial_id),
            "fold_index": int(fold_idx),
            "is_default": int(is_def),
            "params_hash": str(params_hash),
            "validation_positions": [int(p) for p in valid_pos],
            "probabilities": [float(p) for p in pos_proba],
            "score": float(score),
        }
        canonical_payload = json.dumps(evidence_payload, sort_keys=True, separators=(",", ":"))
        payload_sha256 = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        evidence_payload["sha256"] = payload_sha256

        raw_bytes = json.dumps(evidence_payload, indent=2, sort_keys=True).encode("utf-8")
        file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        if evidence_path.exists():
            is_referenced = conn.execute(
                "SELECT 1 FROM fold_score WHERE evidence_path = ? LIMIT 1",
                (str(evidence_path),),
            ).fetchone() is not None
            if not is_referenced:
                _canonical_evidence_path(study_dir_path, study_key, trial_id, fold_idx)
                evidence_path.unlink()

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(evidence_path, flags, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw_bytes)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError as err:
            raise BlockedExperimentError("study_tampered", f"evidence file already exists: {evidence_path}") from err

        fit_counter.seal_owned_evidence(owned_attempt_id, file_sha256)
        conn.execute(
            "UPDATE fit_attempt SET status = 'completed', evidence_sha256 = ? WHERE study_key = ? AND trial_id = ? AND fold_index = ? "
            "AND status = 'started' AND attempt_id = (SELECT MAX(attempt_id) FROM fit_attempt "
            "WHERE study_key = ? AND trial_id = ? AND fold_index = ?)",
            (file_sha256, study_key, trial_id, fold_idx, study_key, trial_id, fold_idx),
        )
        _insert_fold_score(
            conn,
            study_key,
            trial_id,
            fold_idx,
            score,
            "completed",
            evidence_path=str(evidence_path),
            evidence_sha256=file_sha256,
            owned_attempt_id=owned_attempt_id,
        )
        conn.commit()
        return score
    except Exception as exc:
        conn.rollback()
        reason = str(exc) or exc.__class__.__name__
        conn.execute(
            "UPDATE fit_attempt SET status = 'failed', failure_reason = ? WHERE study_key = ? AND trial_id = ? "
            "AND fold_index = ? AND status = 'started' AND attempt_id = (SELECT MAX(attempt_id) FROM fit_attempt "
            "WHERE study_key = ? AND trial_id = ? AND fold_index = ?)",
            (reason, study_key, trial_id, fold_idx, study_key, trial_id, fold_idx),
        )
        _insert_fold_score(conn, study_key, trial_id, fold_idx, None, "failed", reason,
                           owned_attempt_id=owned_attempt_id)
        conn.commit()
        raise


def _verify_projection_inventory(conn: sqlite3.Connection, study_key: str,
                                 expected_by_id: dict, inner_folds: int) -> None:
    """Reject unknown owners, units, statuses and missing parents before resume writes."""
    for table in ("study", "trial", "fold_score", "fit_attempt"):
        if conn.execute(f"SELECT 1 FROM {table} WHERE study_key != ? LIMIT 1", (study_key,)).fetchone():
            raise BlockedExperimentError("study_tampered", f"unexpected owner in {table} inventory")
    trial_rows = conn.execute("SELECT trial_id, status, mean_score FROM trial WHERE study_key = ?", (study_key,)).fetchall()
    trial_ids = {row[0] for row in trial_rows}
    folds = conn.execute("SELECT trial_id, fold_index, status, score, evidence_path, evidence_sha256, owned_attempt_id "
                         "FROM fold_score WHERE study_key = ?", (study_key,)).fetchall()
    attempts = conn.execute("SELECT trial_id, fold_index, status, owned_attempt_id, evidence_sha256 "
                            "FROM fit_attempt WHERE study_key = ?", (study_key,)).fetchall()
    if (trial_ids or folds or attempts) and trial_ids != set(expected_by_id):
        raise BlockedExperimentError("study_tampered", "trial inventory is incomplete; cannot recreate missing work")
    for trial, status, mean in trial_rows:
        if type(trial) is not int or trial not in expected_by_id or status not in {"completed", "failed", "in_progress"}:
            raise BlockedExperimentError("study_tampered", "unexpected trial identity or status")
        if status == "completed":
            if type(mean) not in (int, float) or not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
                raise BlockedExperimentError("study_tampered", "completed trial has an invalid mean score")
        elif mean is not None:
            raise BlockedExperimentError("study_tampered", "unfinished trial has a completed mean score")
    for trial, fold, status, score, path, proof, attempt in folds:
        if (type(trial) is not int or trial not in trial_ids or type(fold) is not int
                or not 0 <= fold < inner_folds or status not in {"completed", "failed", "interrupted"}):
            raise BlockedExperimentError("study_tampered", "unexpected fold identity or status")
        if status == "completed":
            if not attempt or not proof or not path:
                raise BlockedExperimentError("study_tampered", "legacy completed fold lacks attempt/hash provenance")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise BlockedExperimentError("study_tampered", "completed fold has an invalid score")
        elif score is not None or path is not None or proof is not None or not attempt:
            raise BlockedExperimentError("study_tampered", "failed/interrupted fold has invalid evidence provenance")
    for trial, fold, status, attempt, proof in attempts:
        if (type(trial) is not int or trial not in trial_ids or type(fold) is not int
                or not 0 <= fold < inner_folds or status not in {"started", "completed", "failed", "interrupted"}
                or not isinstance(attempt, str) or not attempt):
            raise BlockedExperimentError("study_tampered", "unexpected attempt identity or status")
        if status == "completed" and not proof:
            raise BlockedExperimentError("study_tampered", "legacy completed attempt lacks hash provenance")
        if status != "completed" and proof is not None:
            raise BlockedExperimentError("study_tampered", "unfinished SQL attempt has fabricated completion proof")


def _reconcile_owned_attempts(conn: sqlite3.Connection, counter: FitCounter, owner: str,
                              study_key: str, units: tuple[str, ...]) -> None:
    """Treat journal events as authority, preserving debits across bounded crash windows."""
    journal = {attempt: (unit, status) for attempt, unit, status in counter.owner_attempts(owner)}
    sealed = dict(counter.owner_attempt_evidence(owner))
    local = conn.execute("SELECT owned_attempt_id, trial_id, fold_index, status, evidence_sha256 "
                         "FROM fit_attempt WHERE study_key = ? ORDER BY attempt_id", (study_key,)).fetchall()
    folds = {f"inner:{trial}:{fold}": (status, attempt, proof) for trial, fold, status, attempt, proof in conn.execute(
        "SELECT trial_id, fold_index, status, owned_attempt_id, evidence_sha256 FROM fold_score WHERE study_key = ?",
        (study_key,),
    )}
    local_ids = {row[0] for row in local}
    for attempt, (unit, status) in journal.items():
        if unit not in units:
            raise BlockedExperimentError("study_tampered", "unexpected journal unit for study owner")
        if not unit.startswith("inner:") or attempt in local_ids:
            continue
        if status != "started" or sealed[attempt] is not None or unit in folds:
            raise BlockedExperimentError("study_tampered", "journal work is missing from SQL attempt inventory")
        _, trial, fold = unit.split(":")
        if not conn.execute("SELECT 1 FROM trial WHERE trial_id = ? AND study_key = ?", (int(trial), study_key)).fetchone():
            raise BlockedExperimentError("study_tampered", "journal attempt has no trial parent")
        conn.execute("INSERT INTO fit_attempt (study_key, trial_id, fold_index, status, timestamp, owned_attempt_id) "
                     "VALUES (?, ?, ?, 'interrupted', ?, ?)",
                     (study_key, int(trial), int(fold), datetime.datetime.now(datetime.UTC).isoformat(), attempt))
        conn.commit()
        counter.finish_owned_fit(attempt, "interrupted")
    for attempt, trial, fold, status, proof in local:
        unit = f"inner:{trial}:{fold}"
        if attempt not in journal or journal[attempt][0] != unit:
            raise BlockedExperimentError("study_tampered", "SQL attempt does not bind an owned journal identity")
        journal_status = journal[attempt][1]
        fold_record = folds.get(unit)
        if status == "completed":
            if (journal_status not in {"started", "completed"} or sealed[attempt] != proof
                    or fold_record != ("completed", attempt, proof) or proof is None):
                raise BlockedExperimentError("study_tampered", "completion does not match its immutable journal seal")
            if journal_status == "started":
                counter.finish_owned_fit(attempt, "completed")
            continue
        if journal_status == "completed" or (fold_record and fold_record[0] == "completed" and fold_record[1] == attempt):
            raise BlockedExperimentError("study_tampered", "completed journal/fold work contradicts SQL attempt")
        if journal_status == "started":
            counter.finish_owned_fit(attempt, "interrupted")
            journal_status = "interrupted"
        elif status != journal_status and status != "started":
            raise BlockedExperimentError("study_tampered", "SQL and journal terminal statuses differ")
        if status != journal_status:
            conn.execute("UPDATE fit_attempt SET status = ? WHERE owned_attempt_id = ?", (journal_status, attempt))
            if fold_record and fold_record[1] == attempt:
                conn.execute("UPDATE fold_score SET status = ? WHERE owned_attempt_id = ?", (journal_status, attempt))
            conn.commit()
    local = {attempt: (f"inner:{trial}:{fold}", status, proof) for attempt, trial, fold, status, proof in conn.execute(
        "SELECT owned_attempt_id, trial_id, fold_index, status, evidence_sha256 FROM fit_attempt WHERE study_key = ?",
        (study_key,),
    )}
    for unit, (status, attempt, proof) in folds.items():
        expected = local.get(attempt)
        if expected is None or expected[0] != unit:
            raise BlockedExperimentError("study_tampered", "fold has no matching owned attempt")
        actual_status = conn.execute("SELECT status FROM fold_score WHERE owned_attempt_id = ?", (attempt,)).fetchone()[0]
        if expected[1] != actual_status:
            raise BlockedExperimentError("study_tampered", "fold and owned attempt statuses differ")
        if status == "completed" and sum(a[0] == unit and a[1] == "completed" for a in local.values()) != 1:
            raise BlockedExperimentError("study_tampered", "fold requires exactly one hash-bound successful attempt")


def _verify_study_evidence(
    conn: sqlite3.Connection,
    study_key: str,
    clean_training: TrainingPartition,
    folds: list[tuple[np.ndarray, np.ndarray]],
    study_spec: StudySpec,
    expected_by_id: dict[int, tuple[dict[str, Any], int, str]],
    is_completed_study: bool,
    sel_trial: int | None = None,
    sel_params_json: str | None = None,
    study_dir_path: Path | None = None,
) -> None:
    """Verify immutable fold prediction evidence, recalculate AUC & trial means, and verify winner."""
    fold_rows = conn.execute(
        "SELECT trial_id, fold_index, score, status, failure_reason, evidence_path, evidence_sha256 "
        "FROM fold_score WHERE study_key = ?",
        (study_key,),
    ).fetchall()

    verified_fold_scores: dict[tuple[int, int], float] = {}

    for t_id, f_idx, score, status, _failure_reason, ev_path_str, ev_sha in fold_rows:
        if status != "completed":
            continue
        if score is None or not math.isfinite(score):
            raise BlockedExperimentError(
                "study_tampered", f"completed fold ({t_id}, {f_idx}) has non-finite score: {score}"
            )
        if not ev_path_str or not ev_sha:
            raise BlockedExperimentError(
                "study_tampered", f"completed fold ({t_id}, {f_idx}) missing evidence path or hash"
            )

        if study_dir_path is None:
            raise BlockedExperimentError(
                "study_tampered", "study_dir_path must be provided to verify evidence path confinement"
            )
        try:
            ev_path = _canonical_evidence_path(study_dir_path, study_key, t_id, f_idx)
            if ev_path_str != str(ev_path):
                raise ValueError("evidence path is not the exact managed canonical path")
        except (TypeError, ValueError, OSError) as error:
            raise BlockedExperimentError("study_tampered", str(error)) from error

        if not ev_path.is_file():
            raise BlockedExperimentError(
                "study_tampered", f"missing fold prediction evidence file: {ev_path}"
            )

        file_bytes = ev_path.read_bytes()
        actual_file_sha = hashlib.sha256(file_bytes).hexdigest()
        if actual_file_sha != ev_sha:
            raise BlockedExperimentError(
                "study_tampered", f"fold evidence file content hash mismatch for ({t_id}, {f_idx})"
            )

        try:
            ev_data = json.loads(file_bytes.decode("utf-8"))
        except Exception as err:
            raise BlockedExperimentError(
                "study_tampered", f"corrupted JSON in evidence file {ev_path}: {err}"
            ) from err

        if not isinstance(ev_data, dict) or set(ev_data.keys()) != set(REQUIRED_EVIDENCE_FIELDS):
            raise BlockedExperimentError(
                "study_tampered",
                f"evidence file {ev_path} schema mismatch: expected exact fields {sorted(REQUIRED_EVIDENCE_FIELDS)}",
            )

        try:
            if not isinstance(ev_data["trial_id"], int) or isinstance(ev_data["trial_id"], bool):
                raise BlockedExperimentError("study_tampered", f"trial_id must be integer in {ev_path}")
            if not isinstance(ev_data["fold_index"], int) or isinstance(ev_data["fold_index"], bool):
                raise BlockedExperimentError("study_tampered", f"fold_index must be integer in {ev_path}")

            if ev_data["trial_id"] != t_id or ev_data["fold_index"] != f_idx:
                raise BlockedExperimentError(
                    "study_tampered", f"evidence file identity mismatch: {ev_path}"
                )

            exp_info = expected_by_id.get(t_id)
            if exp_info is None:
                raise BlockedExperimentError(
                    "study_tampered", f"unexpected trial id {t_id} in fold evidence"
                )
            exp_params, exp_is_def, exp_phash = exp_info

            if type(ev_data["is_default"]) is not int or ev_data["is_default"] != exp_is_def or ev_data["params_hash"] != exp_phash:
                raise BlockedExperimentError(
                    "study_tampered", f"evidence metadata mismatch in {ev_path}"
                )

            payload_copy = {k: v for k, v in ev_data.items() if k != "sha256"}
            calc_sha = hashlib.sha256(
                json.dumps(payload_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if ev_data["sha256"] != calc_sha:
                raise BlockedExperimentError(
                    "study_tampered", f"evidence payload sha256 mismatch in {ev_path}"
                )

            if f_idx >= len(folds):
                raise BlockedExperimentError(
                    "study_tampered", f"fold index {f_idx} out of range"
                )

            raw_pos = ev_data["validation_positions"]
            if not isinstance(raw_pos, list) or not all(
                isinstance(p, int) and not isinstance(p, bool) for p in raw_pos
            ):
                raise BlockedExperimentError(
                    "study_tampered", f"validation positions must be list of integers in {ev_path}"
                )

            expected_valid_pos = [int(p) for p in folds[f_idx][1]]
            if raw_pos != expected_valid_pos:
                raise BlockedExperimentError(
                    "study_tampered", f"validation positions mismatch in {ev_path}"
                )

            valid_positions = raw_pos
            y_true = clean_training.y_train.iloc[valid_positions].to_numpy()

            raw_probs = ev_data["probabilities"]
            if not isinstance(raw_probs, list):
                raise BlockedExperimentError(
                    "study_tampered", f"probabilities must be a list in {ev_path}"
                )
            if not all(
                type(p) in (int, float) and math.isfinite(p) and 0.0 <= p <= 1.0
                for p in raw_probs
            ):
                raise BlockedExperimentError(
                    "study_tampered", f"probabilities must contain finite nonboolean numbers in [0, 1] in {ev_path}"
                )

            probs = np.asarray(raw_probs, dtype=np.float64)
            if len(probs) != len(y_true):
                raise BlockedExperimentError(
                    "study_tampered", f"probabilities length mismatch in {ev_path}"
                )

            recomputed_auc = float(roc_auc_score(y_true, probs))
            if not math.isfinite(recomputed_auc):
                raise BlockedExperimentError(
                    "study_tampered", f"recomputed ROC-AUC is non-finite in {ev_path}"
                )

            raw_score = ev_data["score"]
            if (
                not isinstance(raw_score, (int, float))
                or isinstance(raw_score, bool)
                or not math.isfinite(raw_score)
            ):
                raise BlockedExperimentError(
                    "study_tampered", f"score must be a finite number in {ev_path}"
                )

            if abs(recomputed_auc - float(raw_score)) > 1e-12:
                raise BlockedExperimentError(
                    "study_tampered",
                    f"recomputed ROC-AUC {recomputed_auc} does not match evidence file score {raw_score}",
                )
            if abs(recomputed_auc - float(score)) > 1e-12:
                raise BlockedExperimentError(
                    "study_tampered",
                    f"recomputed ROC-AUC {recomputed_auc} does not match fold_score {score}",
                )

            verified_fold_scores[(t_id, f_idx)] = recomputed_auc
        except BlockedExperimentError:
            raise
        except (TypeError, ValueError, IndexError, ArithmeticError) as err:
            raise BlockedExperimentError(
                "study_tampered", f"malformed evidence values in {ev_path}: {err}"
            ) from err

    trial_rows = conn.execute(
        "SELECT trial_id, params_json, params_hash, status, mean_score, is_default "
        "FROM trial WHERE study_key = ?",
        (study_key,),
    ).fetchall()

    verified_trial_means: dict[int, float] = {}

    for t_id, params_json, params_hash, status, mean_score, is_default in trial_rows:
        exp_info = expected_by_id.get(t_id)
        if exp_info is None:
            raise BlockedExperimentError("study_tampered", f"unexpected trial {t_id}")
        exp_params, exp_is_def, exp_phash = exp_info
        exp_json = json.dumps(exp_params, sort_keys=True)
        if params_json != exp_json or params_hash != exp_phash or is_default != exp_is_def:
            raise BlockedExperimentError("study_tampered", f"trial {t_id} specification mismatch")
        if status not in ("in_progress", "completed", "failed"):
            raise BlockedExperimentError("study_tampered", f"trial {t_id} invalid status {status}")

        if status == "completed":
            for f in range(study_spec.inner_folds):
                if (t_id, f) not in verified_fold_scores:
                    raise BlockedExperimentError(
                        "study_tampered", f"completed trial {t_id} missing verified fold {f}"
                    )
            expected_mean = float(
                np.mean([verified_fold_scores[(t_id, f)] for f in range(study_spec.inner_folds)])
            )
            if mean_score is None or abs(expected_mean - float(mean_score)) > 1e-12:
                raise BlockedExperimentError(
                    "study_tampered",
                    f"trial {t_id} mean score mismatch: expected {expected_mean}, stored {mean_score}",
                )
            verified_trial_means[t_id] = expected_mean

    completed_candidates = [
        t_id for t_id in verified_trial_means
        if expected_by_id[t_id][1] == 0
    ]

    if is_completed_study:
        if not completed_candidates:
            raise BlockedExperimentError(
                "study_tampered", "completed study has no completed candidate trials"
            )
        if sel_trial is None:
            raise BlockedExperimentError(
                "study_tampered", "completed study lacks selected trial id"
            )

    if completed_candidates and sel_trial is not None:
        sorted_candidates = sorted(completed_candidates, key=lambda t: (-verified_trial_means[t], t))
        deterministic_winner_id = sorted_candidates[0]
        expected_winner_params = expected_by_id[deterministic_winner_id][0]
        expected_winner_json = json.dumps(expected_winner_params, sort_keys=True)

        if sel_trial != deterministic_winner_id:
            raise BlockedExperimentError(
                "study_tampered",
                f"selected winner trial {sel_trial} does not match deterministic winner {deterministic_winner_id}",
            )
        if sel_params_json is not None and sel_params_json != expected_winner_json:
            raise BlockedExperimentError(
                "study_tampered",
                "selected winner params JSON does not match deterministic winner params",
            )


def tune_outer_training(
    training: TrainingPartition,
    study_spec: StudySpec,
    arm_factory: Callable[[dict[str, Any]], Any],
    budget: FitBudget | FitCounter,
    study_dir: str | Path,
    score_inner_default: bool = False,
    campaign_root: str | Path | None = None,
) -> TuningResult:
    """Bounded HGB random search on outer training partition with SQLite evidence."""
    # Identity enforcement: require exact canonical factory before any reserve or output
    if arm_factory is not build_hgb_tuning_arm:
        raise TypeError("arm_factory must be the canonical build_hgb_tuning_arm constructor")

    # Strict partition separation: never accept FoldPartition or outer validation fields
    if not isinstance(training, TrainingPartition) or isinstance(training, FoldPartition):
        raise TypeError("training must be a TrainingPartition; FoldPartition is prohibited")
    if hasattr(training, "X_validation") or hasattr(training, "validation_keys"):
        raise TypeError("training partition must not contain held-out validation fields")
    if not isinstance(study_spec, StudySpec):
        raise TypeError("study_spec must be a typed StudySpec")
    if type(score_inner_default) is not bool:
        raise TypeError("score_inner_default must be an exact boolean")

    # Defensive input snapshot: reconstruct five fields, clear attrs, caller unchanged
    X_copy = training.X_train.copy(deep=True)
    X_copy.attrs = {}
    y_copy = training.y_train.copy(deep=True)
    y_copy.attrs = {}
    groups_copy = training.groups_train.copy(deep=True)
    groups_copy.attrs = {}

    clean_training = TrainingPartition(
        repeat=int(training.repeat),
        fold=int(training.fold),
        X_train=X_copy,
        y_train=y_copy,
        groups_train=groups_copy,
    )

    # Revalidate snapshot against repository-owned raw-source contracts
    validate_partition(clean_training, build_lead_baseline_registry())

    # EXACT CORE_FEATURE_POOL enforcement immediately after snapshot rawvalidation
    # Rejects subsets and reordered columns BEFORE trial inventory, reservation, or file creation
    if tuple(clean_training.X_train.columns) != CORE_FEATURE_POOL:
        raise ValueError(
            f"training predictors must match exact ordered CORE feature pool: "
            f"expected {CORE_FEATURE_POOL}, got {tuple(clean_training.X_train.columns)}"
        )

    # Authoritative bounds check on snapshot
    if not (0 <= clean_training.repeat <= 2):
        raise ValueError("training repeat must be integer in [0, 2]")
    if not (0 <= clean_training.fold <= 3):
        raise ValueError("training fold must be integer in [0, 3]")
    if type(study_spec.inner_folds) is not int or not (2 <= study_spec.inner_folds <= 5):
        raise ValueError("study_spec inner_folds must be integer in [2, 5]")
    if type(study_spec.seed) is not int or not (0 <= study_spec.seed <= 2**32 - 1):
        raise ValueError("study_spec seed must be integer in [0, 2**32 - 1]")

    # Check parameter_lists valid subsets
    for name, values in study_spec.parameter_lists:
        if name not in STUDY_SPACE:
            raise ValueError(f"study parameter not in allowed HGB space: {name}")
        for v in values:
            if v not in STUDY_SPACE[name]:
                raise ValueError(f"study parameter value not in allowed HGB space: {name}={v}")

    # Budget setup and reservation
    if isinstance(budget, FitBudget):
        fit_counter = FitCounter(budget)
    elif isinstance(budget, FitCounter):
        fit_counter = budget
    else:
        raise TypeError("budget must be a typed FitBudget or FitCounter")

    study_dir_path = _validate_study_dir(study_dir)
    actual_campaign_root = resolve_campaign_root(campaign_root if campaign_root is not None else study_dir_path)
    folds = build_inner_folds(
        clean_training.y_train,
        clean_training.groups_train,
        n_splits=study_spec.inner_folds,
        seed=study_spec.seed,
    )
    membership_digest = inner_membership_digest(folds, clean_training.groups_train)
    trial_param_list = _generate_trial_params(study_spec)
    db_path = study_dir_path / f"study_r{clean_training.repeat}_f{clean_training.fold}.sqlite3"
    _validate_database_path(study_dir_path, db_path)
    for t_id in range(study_spec.n_trials + int(score_inner_default)):
        for f_idx in range(study_spec.inner_folds):
            _canonical_evidence_path(study_dir_path, f"study_r{clean_training.repeat}_f{clean_training.fold}", t_id, f_idx)

    with _study_lock(actual_campaign_root, ".campaign_hpo.lock"), _study_lock(study_dir_path):
        return _tune_locked(
            clean_training,
            study_spec,
            arm_factory,
            fit_counter,
            study_dir_path,
            actual_campaign_root,
            score_inner_default,
            folds,
            membership_digest,
            trial_param_list,
            db_path,
            budget,
        )


def _tune_locked(
    clean_training: TrainingPartition,
    study_spec: StudySpec,
    arm_factory: Callable[..., Pipeline],
    fit_counter: FitCounter,
    study_dir_path: Path,
    actual_campaign_root: Path,
    score_inner_default: bool,
    folds: list[tuple[np.ndarray, np.ndarray]],
    membership_digest: str,
    trial_param_list: list[dict[str, Any]],
    db_path: Path,
    requested_budget: FitBudget | FitCounter,
) -> TuningResult:
    """Run a validated study while its persistent lock is held."""
    planned_fits = required_model_fits(
        study_spec.n_trials, study_spec.inner_folds, score_inner_default=score_inner_default
    )

    manifest_payload = build_study_manifest_payload(
        training=clean_training,
        study_spec=study_spec,
        score_inner_default=score_inner_default,
        membership_digest=membership_digest,
        budget=requested_budget,
        deterministic_trials=trial_param_list,
        campaign_root=actual_campaign_root,
    )
    manifest_payload_json = json.dumps(manifest_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    study_manifest_hash = hashlib.sha256(manifest_payload_json.encode("utf-8")).hexdigest()

    study_key = f"study_r{clean_training.repeat}_f{clean_training.fold}"
    owner_id = hashlib.sha256(json.dumps(
        [str(actual_campaign_root), "nested_hgb_tuning", clean_training.repeat, clean_training.fold],
        separators=(",", ":"),
    ).encode()).hexdigest()

    now = datetime.datetime.now(datetime.UTC).isoformat()
    default_params_json = json.dumps(DEFAULT_HGB_PARAMS, sort_keys=True)

    all_trials_to_run: list[tuple[int, dict[str, Any], int]] = []
    for t_idx, params in enumerate(trial_param_list):
        all_trials_to_run.append((t_idx, params, 0))
    if score_inner_default:
        all_trials_to_run.append((study_spec.n_trials, DEFAULT_HGB_PARAMS, 1))

    expected_trials: list[tuple[int, dict[str, Any], int, str]] = []
    for t_id, params, is_def in all_trials_to_run:
        p_json = json.dumps(params, sort_keys=True)
        p_hash = hashlib.sha256(p_json.encode("utf-8")).hexdigest()
        expected_trials.append((t_id, params, is_def, p_hash))
    expected_by_id = {t_id: (p, is_def, p_hash) for t_id, p, is_def, p_hash in expected_trials}

    invocation_start = charged_total = 0
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA synchronous = FULL")
        _init_sqlite_tables(conn)

        charged_total = conn.execute(
            "SELECT COUNT(*) FROM fit_attempt WHERE study_key = ?", (study_key,)
        ).fetchone()[0] if db_path.exists() else 0
        invocation_start = charged_total

        row = conn.execute(
            """SELECT study_manifest_hash, manifest_payload_json, status, selected_trial_id,
                      selected_params_json, planned_model_fits
               FROM study WHERE study_key = ?""",
            (study_key,),
        ).fetchone()

        _verify_projection_inventory(conn, study_key, expected_by_id, study_spec.inner_folds)
        if row is not None:
            existing_hash, stored_payload_json, existing_status, sel_trial, sel_params_json, ex_planned = row
            if existing_hash != study_manifest_hash or stored_payload_json != manifest_payload_json:
                stored_payload = json.loads(stored_payload_json)
                if (
                    "original_exact_fit_budget" in stored_payload
                    and stored_payload.get("original_exact_fit_budget") != manifest_payload.get("original_exact_fit_budget")
                ):
                    raise BlockedExperimentError(
                        "study_budget_mismatch",
                        "existing study is bound to a different exact fit budget",
                    )
                raise BlockedExperimentError(
                    "study_manifest_mismatch",
                    "existing study manifest hash or payload does not match current specification",
                )
            if existing_status not in ("in_progress", "completed", "failed"):
                raise BlockedExperimentError(
                    "study_tampered", f"existing study status is corrupted: {existing_status}"
                )

            _verify_study_evidence(
                conn=conn,
                study_key=study_key,
                clean_training=clean_training,
                folds=folds,
                study_spec=study_spec,
                expected_by_id=expected_by_id,
                is_completed_study=(existing_status == "completed"),
                sel_trial=sel_trial,
                sel_params_json=sel_params_json,
                study_dir_path=study_dir_path,
            )

        fit_counter.bind_campaign(actual_campaign_root)
        units = tuple(f"inner:{t_id}:{f_idx}" for t_id, _, _ in all_trials_to_run
                      for f_idx in range(study_spec.inner_folds)) + ("outer:default", "outer:selected")
        completed_units = {f"inner:{t_id}:{f_idx}" for t_id, f_idx in conn.execute(
            "SELECT trial_id, fold_index FROM fold_score WHERE study_key = ? AND status = 'completed'",
            (study_key,),
        )}
        _reconcile_owned_attempts(conn, fit_counter, owner_id, study_key, units)
        journal_attempts = fit_counter.owner_attempts(owner_id)
        retry_units = tuple(sorted({unit for _, unit, _ in journal_attempts
                                    if unit.startswith("inner:") and unit not in completed_units}))
        fit_counter.claim_owner(owner_id, study_manifest_hash, units, study_spec.n_trials, retry_units)
        charged_total = len(fit_counter.owner_attempts(owner_id))
        invocation_start = charged_total

        if row is not None:

            incomplete_or_failed = conn.execute(
                "SELECT COUNT(*) FROM trial WHERE study_key = ? AND status != 'completed'",
                (study_key,),
            ).fetchone()[0]
            if existing_status == "completed" and not incomplete_or_failed:
                if sel_trial is None or sel_params_json is None:
                    raise BlockedExperimentError(
                        "study_tampered", "completed study lacks selected trial or params"
                    )
                stored_completed = conn.execute(
                    """SELECT trial_id, params_json, params_hash, status, is_default
                       FROM trial WHERE study_key = ?""",
                    (study_key,),
                ).fetchall()
                if len(stored_completed) != len(expected_by_id):
                    raise BlockedExperimentError("study_tampered", "completed study trial inventory changed")
                for trial_id, params_json, params_hash, status, is_default in stored_completed:
                    expected_info = expected_by_id.get(trial_id)
                    expected_json = json.dumps(expected_info[0], sort_keys=True) if expected_info else None
                    expected_hash = expected_info[2] if expected_info else None
                    if (
                        expected_info is None
                        or params_json != expected_json
                        or params_hash != expected_hash
                        or status not in ("completed", "failed")
                        or is_default != expected_info[1]
                    ):
                        raise BlockedExperimentError(
                            "study_tampered", f"completed trial {trial_id} contract/status changed"
                        )
                sel_dict = json.loads(sel_params_json)
                selected_record = next(
                    (record for record in stored_completed if record[0] == sel_trial), None
                )
                if (
                    selected_record is None
                    or selected_record[3] != "completed"
                    or selected_record[4] != 0
                    or selected_record[1] != json.dumps(sel_dict, sort_keys=True)
                ):
                    raise BlockedExperimentError(
                        "study_tampered", "selected trial does not match its stored completed configuration"
                    )
                sel_tuple = tuple(sorted(sel_dict.items()))
                conn.execute(
                    "UPDATE study SET charged_model_fits = ?, last_invocation_model_fits = 0, updated_at = ? "
                    "WHERE study_key = ?",
                    (charged_total, datetime.datetime.now(datetime.UTC).isoformat(), study_key),
                )
                conn.commit()
                return TuningResult(
                    selected_params=sel_tuple,
                    trial_id=sel_trial,
                    study_manifest_hash=study_manifest_hash,
                    planned_model_fits=ex_planned,
                    used_model_fits=0,
                    manifest_reference=str(db_path),
                    study_charged_model_fits=charged_total,
                    invocation_model_fits=0,
                    campaign_total_model_fits=fit_counter.used_model_fits,
                    owned_reserved_model_fits=fit_counter.owner_reserved_model_fits(owner_id),
                    owner_id=owner_id,
                )

        if row is None:
            study_spec_dict = {
                "n_trials": study_spec.n_trials,
                "inner_folds": study_spec.inner_folds,
                "seed": study_spec.seed,
                "parameter_lists": [list(item) for item in study_spec.parameter_lists],
            }
            conn.execute(
                """INSERT INTO study (
                    study_key, repeat, fold, study_manifest_hash, manifest_payload_json, status,
                    study_spec_json, inner_membership_digest, training_rows_count,
                    env_version, n_trials, inner_folds, seed, planned_model_fits,
                    score_inner_default, default_params_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?, 'nested_tuning_hgb_v2', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    study_key,
                    clean_training.repeat,
                    clean_training.fold,
                    study_manifest_hash,
                    manifest_payload_json,
                    json.dumps(study_spec_dict, sort_keys=True),
                    membership_digest,
                    len(clean_training.X_train),
                    study_spec.n_trials,
                    study_spec.inner_folds,
                    study_spec.seed,
                    planned_fits,
                    1 if score_inner_default else 0,
                    default_params_json,
                    now,
                    now,
                ),
            )
            conn.commit()

        expected_trial_ids = {trial_id for trial_id, _, _ in all_trials_to_run}
        stored_trial_ids = {
            record[0]
            for record in conn.execute(
                "SELECT trial_id FROM trial WHERE study_key = ?", (study_key,)
            ).fetchall()
        }
        if not stored_trial_ids.issubset(expected_trial_ids):
            raise BlockedExperimentError("study_tampered", "study contains unexpected trial IDs")

        for t_id, params, is_def in all_trials_to_run:
            p_json = json.dumps(params, sort_keys=True)
            p_hash = hashlib.sha256(p_json.encode("utf-8")).hexdigest()
            t_row = conn.execute(
                "SELECT params_json, params_hash, status, is_default FROM trial WHERE study_key = ? AND trial_id = ?",
                (study_key, t_id),
            ).fetchone()
            if t_row is None:
                conn.execute(
                    """INSERT INTO trial (study_key, trial_id, params_json, params_hash,
                                          status, is_default, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 'in_progress', ?, ?, ?)""",
                    (study_key, t_id, p_json, p_hash, is_def, now, now),
                )
            elif (
                t_row[0] != p_json
                or t_row[1] != p_hash
                or t_row[3] != is_def
                or t_row[2] not in ("in_progress", "failed", "completed")
            ):
                raise BlockedExperimentError(
                    "study_tampered", f"stored trial contract/status is invalid for trial {t_id}"
                )
        conn.commit()

        # Run trials across folds
        for t_id, params, is_def in all_trials_to_run:
            p_json = json.dumps(params, sort_keys=True)
            p_hash = hashlib.sha256(p_json.encode("utf-8")).hexdigest()
            t_row = conn.execute(
                "SELECT status FROM trial WHERE study_key = ? AND trial_id = ?",
                (study_key, t_id),
            ).fetchone()
            if t_row is not None and t_row[0] == "completed":
                continue

            fold_scores: list[float] = []

            for f_idx, (train_pos, valid_pos) in enumerate(folds):
                f_row = conn.execute(
                    "SELECT score, status FROM fold_score WHERE study_key = ? AND trial_id = ? AND fold_index = ?",
                    (study_key, t_id, f_idx),
                ).fetchone()
                if f_row is not None and f_row[1] == "completed" and f_row[0] is not None:
                    fold_scores.append(f_row[0])
                    continue

                X_tr = clean_training.X_train.iloc[train_pos]
                y_tr = clean_training.y_train.iloc[train_pos]
                groups_tr = clean_training.groups_train.iloc[train_pos]
                X_va = clean_training.X_train.iloc[valid_pos]
                y_va = clean_training.y_train.iloc[valid_pos]

                attempt_id = None
                owned_attempt_id = fit_counter.before_owned_fit(owner_id, f"inner:{t_id}:{f_idx}")
                try:
                    now = datetime.datetime.now(datetime.UTC).isoformat()
                    attempt_id = conn.execute(
                        "INSERT INTO fit_attempt (study_key, trial_id, fold_index, status, timestamp, owned_attempt_id) "
                        "VALUES (?, ?, ?, 'started', ?, ?)",
                        (study_key, t_id, f_idx, now, owned_attempt_id),
                    ).lastrowid
                    conn.commit()
                    arm = arm_factory(params)
                    _validate_constructed_arm(arm, params)
                except ValueError as exc:
                    fit_counter.finish_owned_fit(owned_attempt_id, "failed")
                    reason = str(exc) or exc.__class__.__name__
                    if attempt_id is not None:
                        conn.execute(
                            "UPDATE fit_attempt SET status = 'failed', failure_reason = ? WHERE attempt_id = ?",
                            (reason, attempt_id),
                        )
                    conn.execute(
                        "UPDATE trial SET status = 'failed', failure_reason = ?, updated_at = ? "
                        "WHERE study_key = ? AND trial_id = ?",
                        (reason, datetime.datetime.now(datetime.UTC).isoformat(), study_key, t_id),
                    )
                    conn.commit()
                    if "fresh unfitted estimator" in reason:
                        raise
                    break
                except (ArithmeticError, RuntimeError, TypeError) as exc:
                    fit_counter.finish_owned_fit(owned_attempt_id, "failed")
                    reason = str(exc) or exc.__class__.__name__
                    if attempt_id is not None:
                        conn.execute(
                            "UPDATE fit_attempt SET status = 'failed', failure_reason = ? WHERE attempt_id = ?",
                            (reason, attempt_id),
                        )
                    conn.execute(
                        "UPDATE trial SET status = 'failed', failure_reason = ?, updated_at = ? "
                        "WHERE study_key = ? AND trial_id = ?",
                        (reason, datetime.datetime.now(datetime.UTC).isoformat(), study_key, t_id),
                    )
                    conn.commit()
                    break
                except BaseException:
                    conn.rollback()
                    if attempt_id is not None:
                        conn.execute("UPDATE fit_attempt SET status = 'interrupted' WHERE attempt_id = ?", (attempt_id,))
                        conn.commit()
                    fit_counter.finish_owned_fit(owned_attempt_id, "interrupted")
                    raise

                try:
                    score = _fit_and_score_fold(
                        arm,
                        X_tr,
                        y_tr,
                        groups_tr,
                        X_va,
                        y_va,
                        fit_counter,
                        conn,
                        study_key,
                        t_id,
                        f_idx,
                        is_def=is_def,
                        params_hash=p_hash,
                        valid_pos=valid_pos,
                        study_dir_path=study_dir_path,
                    )
                    fit_counter.finish_owned_fit(owned_attempt_id, "completed")
                    fold_scores.append(score)
                except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                    fit_counter.finish_owned_fit(owned_attempt_id, "failed")
                    t_now = datetime.datetime.now(datetime.UTC).isoformat()
                    reason = str(exc) or exc.__class__.__name__
                    conn.execute(
                        """UPDATE trial SET status = 'failed', failure_reason = ?, updated_at = ?
                           WHERE study_key = ? AND trial_id = ?""",
                        (reason, t_now, study_key, t_id),
                    )
                    conn.commit()
                    break
                except BaseException:
                    conn.rollback()
                    status = conn.execute("SELECT status FROM fit_attempt WHERE attempt_id = ?", (attempt_id,)).fetchone()[0]
                    if status == "completed":
                        fit_counter.finish_owned_fit(owned_attempt_id, "completed")
                        raise
                    conn.execute("UPDATE fit_attempt SET status = 'interrupted' WHERE attempt_id = ?", (attempt_id,))
                    conn.execute("UPDATE fold_score SET status = 'interrupted' WHERE owned_attempt_id = ?", (owned_attempt_id,))
                    conn.commit()
                    fit_counter.finish_owned_fit(owned_attempt_id, "interrupted")
                    raise

            t_now = datetime.datetime.now(datetime.UTC).isoformat()
            if len(fold_scores) != study_spec.inner_folds:
                continue

            mean_s = float(np.mean(fold_scores))
            conn.execute(
                """UPDATE trial SET status = 'completed', mean_score = ?, updated_at = ?
                   WHERE study_key = ? AND trial_id = ?""",
                (mean_s, t_now, study_key, t_id),
            )
            conn.commit()

        # Selection of best candidate trial (deterministic highest mean ROC-AUC, tie-break by trial_id)
        candidate_rows = conn.execute(
            """SELECT trial_id, params_json, mean_score
               FROM trial
               WHERE study_key = ? AND is_default = 0 AND status = 'completed'
               ORDER BY mean_score DESC, trial_id ASC""",
            (study_key,),
        ).fetchall()

        if not candidate_rows:
            conn.execute(
                """UPDATE study SET status = 'failed', failure_reason = 'no_valid_trials', updated_at = ?
                       WHERE study_key = ?""",
                (now, study_key),
            )
            conn.commit()
            raise BlockedExperimentError(
                "no_valid_trials", "study completed with no valid candidate trials"
            )

        best_trial_id, best_params_json, _ = candidate_rows[0]
        best_params_dict = json.loads(best_params_json)
        selected_params_tuple = tuple(sorted(best_params_dict.items()))

        conn.execute(
            """UPDATE study SET status = 'completed', selected_trial_id = ?,
                                selected_params_json = ?, updated_at = ?
               WHERE study_key = ?""",
            (best_trial_id, best_params_json, now, study_key),
        )
        conn.commit()

        charged_total = len(fit_counter.owner_attempts(owner_id))
        conn.execute(
            "UPDATE study SET charged_model_fits = ?, last_invocation_model_fits = ?, updated_at = ? "
            "WHERE study_key = ?",
            (charged_total, charged_total - invocation_start, datetime.datetime.now(datetime.UTC).isoformat(), study_key),
        )
        conn.commit()
        return TuningResult(
            selected_params=selected_params_tuple,
            trial_id=best_trial_id,
            study_manifest_hash=study_manifest_hash,
            planned_model_fits=planned_fits,
            used_model_fits=fit_counter.used_model_fits,
            manifest_reference=str(db_path),
            study_charged_model_fits=charged_total,
            invocation_model_fits=charged_total - invocation_start,
            campaign_total_model_fits=fit_counter.used_model_fits,
            owned_reserved_model_fits=fit_counter.owner_reserved_model_fits(owner_id),
            owner_id=owner_id,
        )
    finally:
        conn.close()
