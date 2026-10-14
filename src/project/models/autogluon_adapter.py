"""Optional, supervised, XT-only AutoGluon adapter; no execution grant.

Reviewed API: AutoGluon 1.4.0, especially fit/predict_proba and kwarg validation:
https://github.com/autogluon/autogluon/blob/1.4.0/tabular/src/autogluon/tabular/predictor/predictor.py

AutoGluon's time_limit and memory_limit (GiB) are soft backend hints. Hard wall
time/RSS limits must be enforced by Task 12's external supervisor. The private
runtime verifier is deliberately unavailable until that worker integration is
implemented. This Python boundary protects against accidental unsupervised
execution; it is not a tamperproof security sandbox.

Fitted evidence is kept on the estimator for Task 12 to serialize in its bundle;
this module writes no independent evidence files or model output directories.
The backend writes models only at the path supplied by the verified worker.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from project.experiments.candidate_contracts import BlockedExperimentError
from project.features.registry import build_lead_baseline_registry

_LABEL_COLUMN = "__candidate_autogluon_label__"
_REVIEWED_VERSION = "1.4.0"
_GIB = 1024**3
_FIT_KWARGS = {
    "num_bag_folds": 0, "num_stack_levels": 0, "auto_stack": False,
    "calibrate": False, "hyperparameter_tune_kwargs": None,
}
_FIT_EXPLICIT = (
    "train_data", "time_limit", "presets", "hyperparameters",
    "fit_weighted_ensemble", "dynamic_stacking", "calibrate_decision_threshold",
    "num_cpus", "num_gpus", "fit_strategy", "memory_limit",
)


def _positive_finite(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


@dataclass(frozen=True)
class AutoGluonSettings:
    """Required declared limits; only one CPU and the fixed XT profile are allowed."""

    time_limit: float
    max_rss_bytes: int
    num_cpus: int = 1

    def __post_init__(self) -> None:
        if not _positive_finite(self.time_limit):
            raise ValueError("time_limit must be positive finite seconds, not boolean")
        if type(self.max_rss_bytes) is not int or not _positive_finite(self.max_rss_bytes):
            raise ValueError("max_rss_bytes must be a positive finite integer byte count")
        if type(self.num_cpus) is not int or self.num_cpus != 1:
            raise ValueError("num_cpus must be integer 1")


@dataclass(frozen=True)
class _SupervisedRuntime:
    """Task 12 worker-owned projection of a verified live supervisor lease.

    The verifier, not construction of this record, must establish actual watcher
    proof and worker-owned lifecycle. supervisor_pid is the worker's parent PID;
    deadline_monotonic is the live lease's absolute local monotonic deadline.
    artifact_path is an absolute, confined, unique backend directory allocated by
    the runner, never an estimator parameter. It need not exist before creation.
    """

    worker_pid: int
    supervisor_pid: int
    time_limit: float
    max_rss_bytes: int
    num_cpus: int
    deadline_monotonic: float
    artifact_path: str
    lease_id: str


def _verify_supervised_runtime(settings: AutoGluonSettings) -> _SupervisedRuntime:
    """Fail closed until Task 12 implements its repository-owned lease verifier.

    Task 12 must resolve a process-local active context, verify the external wall
    time/RSS watcher and admitted bounds, and return _SupervisedRuntime. Context
    ownership must end with the worker lifecycle; reject missing, expired or
    inherited leases. A caller boolean or production monkeypatch is not a
    binding mechanism. Tests fake this external boundary only.

    Binding contract Task 12 must satisfy when it replaces this body:

    * Signature stays ``(settings: AutoGluonSettings) -> _SupervisedRuntime``.
      Return the typed record; never a dict, bool or duck-typed stand-in.
    * Raise ``BlockedExperimentError('supervisor_unavailable', reason)`` whenever
      no live verified lease exists. Returning ``None`` is also treated as a
      block, but raising carries the diagnostic reason.
    * Field obligations, each re-asserted by ``_require_runtime``:
      ``worker_pid == os.getpid()``; ``supervisor_pid == os.getppid()`` and the
      parent must be the live watcher process; ``time_limit``, ``max_rss_bytes``
      and ``num_cpus`` must equal the caller's ``AutoGluonSettings`` exactly, so
      a worker cannot widen limits the candidate declared;
      ``deadline_monotonic`` is a local ``time.monotonic()`` absolute deadline
      strictly in the future (finite, not ``inf``); ``artifact_path`` is an
      absolute, confined, per-fit unique backend directory the runner allocates;
      ``lease_id`` is a nonempty unpadded identifier stable for one lease.
    * The lease must stay valid across the whole fit. ``fit`` re-verifies
      immediately before ``backend.fit`` and blocks if the record changed, so the
      verifier must return an equal record while a lease is live and a different
      one (or raise) once it is revoked, expired or inherited by another process.
    * The watcher, not this adapter, enforces wall time and RSS by killing the
      worker. AutoGluon's ``time_limit``/``memory_limit`` are soft hints only.
    """
    raise BlockedExperimentError(
        "supervisor_unavailable",
        "AutoGluon fit requires Task 12's verified active wall-time/RSS supervisor context",
    )


def _require_runtime(settings: AutoGluonSettings) -> _SupervisedRuntime:
    runtime = _verify_supervised_runtime(settings)
    if not isinstance(runtime, _SupervisedRuntime):
        raise BlockedExperimentError("supervisor_unavailable", "supervisor returned no typed live lease")
    valid_identity = (
        type(runtime.worker_pid) is int and runtime.worker_pid == os.getpid()
        and type(runtime.supervisor_pid) is int and runtime.supervisor_pid == os.getppid()
    )
    valid_bounds = (
        _positive_finite(runtime.time_limit) and runtime.time_limit == settings.time_limit
        and type(runtime.max_rss_bytes) is int and runtime.max_rss_bytes == settings.max_rss_bytes
        and type(runtime.num_cpus) is int and runtime.num_cpus == settings.num_cpus
    )
    valid_lifecycle = (
        _positive_finite(runtime.deadline_monotonic)
        and runtime.deadline_monotonic > time.monotonic()
        and isinstance(runtime.lease_id, str) and bool(runtime.lease_id.strip())
        and runtime.lease_id == runtime.lease_id.strip()
    )
    valid_path = (
        isinstance(runtime.artifact_path, str) and bool(runtime.artifact_path.strip())
        and Path(runtime.artifact_path).is_absolute()
    )
    if not (valid_identity and valid_bounds and valid_lifecycle and valid_path):
        raise BlockedExperimentError("supervisor_unavailable", "supervisor identity/lifecycle/bounds/path mismatch")
    return runtime


def _import_tabular() -> Any:
    """Private optional-import boundary; importing this adapter never calls it."""
    return importlib.import_module("autogluon.tabular")


def _require_signature(method: Any, names: tuple[str, ...]) -> inspect.Signature:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as error:
        raise BlockedExperimentError("optional_dependency_unsupported", "backend signature is unavailable") from error
    for name in names:
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        ):
            raise BlockedExperimentError("optional_dependency_unsupported", f"backend explicit guard unsupported: {name}")
    return signature


def _backend_definition() -> tuple[str, type]:
    try:
        module = _import_tabular()
    except ImportError as error:
        raise BlockedExperimentError("optional_dependency_unavailable", "AutoGluon Tabular dependency is unavailable") from error
    version = getattr(module, "__version__", None)
    predictor_type = getattr(module, "TabularPredictor", None)
    if version != _REVIEWED_VERSION or not isinstance(predictor_type, type):
        raise BlockedExperimentError("optional_dependency_unsupported", f"reviewed AutoGluon {_REVIEWED_VERSION} API required; found {version!r}")
    _require_signature(predictor_type.__init__, ("label", "problem_type", "eval_metric", "path", "positive_class"))
    fit_signature = _require_signature(getattr(predictor_type, "fit", None), _FIT_EXPLICIT)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in fit_signature.parameters.values()):
        _require_signature(predictor_type.fit, tuple(_FIT_KWARGS))
    _require_signature(getattr(predictor_type, "predict_proba", None), ("data", "as_pandas", "as_multiclass"))
    _require_signature(getattr(predictor_type, "_validate_fit_kwargs", None), ("kwargs",))
    for name in ("info", "model_names"):
        if not callable(getattr(predictor_type, name, None)):
            raise BlockedExperimentError("optional_dependency_unsupported", f"backend evidence API missing: {name}")
    return version, predictor_type


def autogluon_capability() -> str:
    """Return the supported backend version, or an explicit dependency/API blocker.

    Imports the optional package only on invocation; never constructs a backend,
    creates an output directory, verifies supervision or grants fit permission.
    """
    version, _ = _backend_definition()
    return version


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("AutoGluon predictors must be a pandas DataFrame")
    if _LABEL_COLUMN in frame.columns:
        raise ValueError("reserved temporary label column collision")
    if frame.columns.has_duplicates or not len(frame.columns):
        raise ValueError("predictor schema must contain unique nonempty columns")
    if not frame.index.is_unique or frame.index.to_frame(index=False).isna().any().any():
        raise ValueError("predictor index must be unique and nonnull")
    approved = build_lead_baseline_registry().names
    if any(not isinstance(name, str) or name not in approved for name in frame.columns):
        raise ValueError("predictor schema must contain approved raw core columns only")
    numeric = {}
    domains = {"hour": (0, 23), "weekday": (0, 6), "month": (1, 12)}
    for name in frame:
        values = frame[name]
        if name == "primary_use":
            categorical = isinstance(values.dtype, pd.CategoricalDtype)
            if (not (categorical or is_object_dtype(values.dtype) or is_string_dtype(values.dtype))
                    or any(not isinstance(v, str) for v in values.dropna())
                    or (categorical and any(not isinstance(v, str) for v in values.cat.categories))):
                raise ValueError("primary_use must contain string categories or nulls")
            continue
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype) or is_complex_dtype(values.dtype):
            raise ValueError(f"{name} must have a real nonboolean numeric dtype")
        numeric[name] = values.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(numeric[name]).any():
            raise ValueError(f"{name} must contain finite numeric values or nulls")
        if name in domains:
            low, high = domains[name]
            known = numeric[name][~np.isnan(numeric[name])]
            if ((known < low) | (known > high) | (known != np.floor(known))).any():
                raise ValueError(f"{name} must be integer-valued in [{low}, {high}] or null")
    if "year_built" in numeric and "year" in numeric and (numeric["year_built"] > numeric["year"]).any():
        raise ValueError("year_built must not exceed the known observation year")


def _validate_training(frame: pd.DataFrame, target: pd.Series) -> None:
    _validate_frame(frame)
    if not len(frame):
        raise ValueError("fit requires nonempty training rows")
    if not isinstance(target, pd.Series) or not target.index.equals(frame.index):
        raise ValueError("training target Series index must align exactly with predictors")
    if (not is_integer_dtype(target.dtype) or is_bool_dtype(target.dtype)
            or target.isna().any() or set(target.tolist()) != {0, 1}):
        raise ValueError("training target must contain both integer binary classes 0 and 1")


def _fit_params(settings: AutoGluonSettings) -> dict[str, Any]:
    return {
        **_FIT_KWARGS,
        "time_limit": settings.time_limit, "memory_limit": settings.max_rss_bytes / _GIB,
        "presets": None, "hyperparameters": {"XT": {
            "n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2,
        }},
        "dynamic_stacking": False, "fit_weighted_ensemble": False,
        "calibrate_decision_threshold": False, "num_cpus": 1, "num_gpus": 0,
        "fit_strategy": "sequential",
    }


def _create_backend(predictor_type: type, runtime: _SupervisedRuntime) -> Any:
    """Private construction boundary, with output location owned by the worker."""
    return predictor_type(label=_LABEL_COLUMN, problem_type="binary", eval_metric="roc_auc",
                          path=runtime.artifact_path, positive_class=1)


def _check_fit_kwargs(backend: Any) -> None:
    try:
        validated = backend._validate_fit_kwargs(kwargs=deepcopy(_FIT_KWARGS))
    except (TypeError, ValueError, AssertionError) as error:
        raise BlockedExperimentError("optional_dependency_unsupported", "backend rejects required fit guards") from error
    if (not isinstance(validated, dict) or any(
            name not in validated or type(validated[name]) is not type(value) or validated[name] != value
            for name, value in _FIT_KWARGS.items())):
        raise BlockedExperimentError("optional_dependency_unsupported", "backend drops or changes required fit guards")


def _class_order(backend: Any) -> tuple[int, int]:
    classes = getattr(backend, "classes_", None)
    if (not isinstance(classes, (list, tuple, np.ndarray)) or len(classes) != 2
            or any(not isinstance(c, (int, np.integer)) or isinstance(c, (bool, np.bool_)) for c in classes)
            or set(classes) != {0, 1}):
        raise ValueError("backend classes_ must identify integer binary classes 0 and 1")
    positive = getattr(backend, "positive_class", None)
    if not isinstance(positive, (int, np.integer)) or isinstance(positive, (bool, np.bool_)) or positive != 1:
        raise ValueError("backend positive class must be integer 1")
    return tuple(int(c) for c in classes)


class _DiagnosticFeatureMetadataBoundary:
    """Private boundary fake class for FeatureMetadata diagnostic projection testing."""

    def __init__(
        self,
        type_map_raw: dict[str, str],
        type_group_map_special: dict[str, list[str]] | None = None,
    ) -> None:
        self.type_map_raw = type_map_raw
        self.type_group_map_special = (
            type_group_map_special if type_group_map_special is not None else {}
        )


def _is_feature_metadata(value: Any) -> bool:
    """Recognize the private fake or the lazily imported supported backend class."""
    if isinstance(value, _DiagnosticFeatureMetadataBoundary):
        return True
    try:
        metadata = importlib.import_module('autogluon.common.features.feature_metadata')
        expected = metadata.FeatureMetadata
    except (ImportError, AttributeError):
        return False
    return isinstance(expected, type) and isinstance(value, expected)


def _project_feature_metadata(value: Any) -> dict[str, Any]:
    raw = getattr(value, "type_map_raw", None)
    if not isinstance(raw, Mapping):
        raise TypeError("FeatureMetadata type_map_raw must be a mapping")
    projected_raw: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("FeatureMetadata type_map_raw keys and values must be strings")
        projected_raw[k] = v

    special = getattr(value, "type_group_map_special", None)
    if special is None:
        special = {}
    if not isinstance(special, Mapping):
        raise TypeError("FeatureMetadata type_group_map_special must be a mapping")
    projected_special: dict[str, list[str]] = {}
    for k, v in special.items():
        if not isinstance(k, str):
            raise TypeError("FeatureMetadata type_group_map_special keys must be strings")
        if not isinstance(v, (list, tuple, set)):
            raise TypeError("FeatureMetadata type_group_map_special values must be sequences of strings")
        items: list[str] = []
        for item in v:
            if not isinstance(item, str):
                raise TypeError("FeatureMetadata type_group_map_special elements must be strings")
            items.append(item)
        projected_special[k] = items

    return {
        "type_map_raw": projected_raw,
        "type_group_map_special": projected_special,
    }


_MAX_DIAGNOSTIC_DEPTH = 16


def _project_diagnostic_value(
    val: Any,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    if depth > _MAX_DIAGNOSTIC_DEPTH:
        raise ValueError(f"diagnostic mapping exceeds maximum nesting depth of {_MAX_DIAGNOSTIC_DEPTH}")

    if val is None:
        return None

    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    if isinstance(val, (int, np.integer)):
        return int(val)

    if isinstance(val, (float, np.floating)):
        f_val = float(val)
        if not math.isfinite(f_val):
            raise ValueError(f"non-finite numeric value in diagnostic info: {val!r}")
        return f_val

    if isinstance(val, str):
        return str(val)

    if _is_feature_metadata(val):
        return _project_feature_metadata(val)

    if seen is None:
        seen = set()

    obj_id = id(val)
    if obj_id in seen:
        raise ValueError("circular reference detected in diagnostic mapping")

    if isinstance(val, Mapping):
        seen.add(obj_id)
        try:
            result_map: dict[str, Any] = {}
            for k, v in val.items():
                if not isinstance(k, str):
                    raise TypeError(f"diagnostic mapping keys must be strings, got {type(k).__name__}")
                result_map[k] = _project_diagnostic_value(v, depth=depth + 1, seen=seen)
            return result_map
        finally:
            seen.remove(obj_id)

    if isinstance(val, (list, tuple)):
        seen.add(obj_id)
        try:
            return [_project_diagnostic_value(item, depth=depth + 1, seen=seen) for item in val]
        finally:
            seen.remove(obj_id)

    raise TypeError(f"unsupported diagnostic object of type {type(val).__name__}")


class AutoGluonCandidate(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible fixed binary XT candidate, admitted only in a worker.

    fit accepts training predictors/labels only. The backend may hold out rows
    internally from this isolated training table. Output probability columns are
    always [0, 1], aligned to input row order, regardless of backend class order.
    """

    def __init__(self, settings: AutoGluonSettings) -> None:
        self.settings = settings

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AutoGluonCandidate:
        for name in tuple(vars(self)):
            if name.endswith("_") and not name.startswith("_"):
                delattr(self, name)
        if not isinstance(self.settings, AutoGluonSettings):
            raise TypeError("settings must be a typed AutoGluonSettings")
        self.settings.__post_init__()
        _validate_training(X, y)
        runtime = _require_runtime(self.settings)
        version, predictor_type = _backend_definition()
        backend = _create_backend(predictor_type, runtime)
        _check_fit_kwargs(backend)
        table = X.copy(deep=True)
        table.attrs = {}
        table[_LABEL_COLUMN] = y.to_numpy(copy=True)
        params = _fit_params(self.settings)
        if _require_runtime(self.settings) != runtime:
            raise BlockedExperimentError("supervisor_unavailable", "supervisor lease changed before backend fit")
        backend.fit(train_data=table, **deepcopy(params))
        class_order = _class_order(backend)
        model_names = backend.model_names()
        if not isinstance(model_names, list) or not model_names or any(not isinstance(n, str) or not n for n in model_names):
            raise ValueError("backend model inventory must contain named fitted models")
        info = backend.info()
        if not isinstance(info, dict):
            raise TypeError("backend info must be a diagnostic mapping")
        # info() is experimental: record it as diagnostics, never use it for admission.
        projected = _project_diagnostic_value(info)
        evidence = json.loads(json.dumps(projected, allow_nan=False))
        self.backend_ = backend
        self.backend_version_ = version
        self.backend_class_order_ = class_order
        self.classes_ = np.frombuffer(np.array([0, 1], dtype=np.int64).tobytes(), dtype=np.int64)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.settings_ = self.settings
        self.resolved_fit_params_ = deepcopy(params)
        self.backend_model_names_ = tuple(model_names)
        self.backend_info_ = evidence
        self.supervisor_lease_id_ = runtime.lease_id
        return self

    def _check_settings(self) -> None:
        check_is_fitted(self, "backend_")
        if self.settings != self.settings_:
            raise ValueError("settings changed; fit again before use")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        self._check_settings()
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match the ordered fit schema")
        return self.feature_names_in_.copy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_settings()
        _validate_frame(X)
        if tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError("prediction schema must match the exact ordered fit schema")
        if not len(X):
            return np.empty((0, 2), dtype=np.float64)
        isolated = X.copy(deep=True)
        isolated.attrs = {}
        result = self.backend_.predict_proba(data=isolated, as_pandas=True, as_multiclass=True)
        if _class_order(self.backend_) != self.backend_class_order_:
            raise ValueError("backend fitted class identity changed")
        if not isinstance(result, pd.DataFrame) or result.shape != (len(X), 2):
            raise ValueError("backend probabilities must be a two-column DataFrame with matching row count")
        if not result.index.equals(X.index):
            raise ValueError("backend probability index must preserve exact input row order")
        columns = tuple(result.columns)
        if (columns != self.backend_class_order_
                or any(not isinstance(c, (int, np.integer)) or isinstance(c, (bool, np.bool_)) for c in columns)):
            raise ValueError("backend probability columns must match fitted integer class identities")
        for dtype in result.dtypes:
            if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
                raise TypeError("backend probabilities must have real nonboolean numeric dtypes")
        probabilities = result.loc[:, [0, 1]].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("backend probabilities must be finite in [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1., rtol=1e-7, atol=1e-8):
            raise ValueError("backend class probabilities must sum to one")
        return probabilities
