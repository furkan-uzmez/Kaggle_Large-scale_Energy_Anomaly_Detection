"""One authorized XT backend unit, compatible with parent debit/ACK messages.

No journal, runner, admission bypass, bounded lease, or model pickle transport.
Only backend-owned files under ``backend-models`` may contain AutoGluon pickles;
neither requests nor responses can select model/code paths. The parent translates
scores to the canonical OOF shard and performs the existing shard_closed handshake.
SIGKILL/OS crashes leave incomplete files; only the parent can classify those.

Installed 1.4.0 source reviewed: predictor.py fit/_validate_fit_kwargs; RFModel
_fit; XTModel. RFModel may make two warm-start fit calls on the SAME forest and
may reduce trees under native memory heuristics. Calls are recorded, and reduced
trees fail closed. Tree-level threading is a library default, not concurrent
campaign model fits or a campaign CPU cap.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import importlib.metadata
import inspect
import json
import os
import secrets
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Direct -I execution uses only the repository containing this reviewed worker,
# never a request-selected sys.path or PYTHONPATH. Parent binds its source files.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from project.models import full_autogluon_bridge as bridge

BATCH_ROWS = 64000
_NATIVE_FOREST_DEFAULTS = {"bootstrap": True, "max_leaf_nodes": 15000,
                           "n_jobs": -1, "criterion": "gini", "oob_score": False}
_BINDING_FIELDS = {"version", "worker_sha256", "bridge_sha256", "lifecycle_sha256",
                   "approval_sha256", "resolved_profile"}


def _approval_path() -> Path:
    return Path(__file__).resolve().parents[3] / "docs/evidence/autogluon_full_profile_v1.json"


def resolved_profile() -> dict[str, Any]:
    fit = bridge.full_xt_profile()
    fit["memory_limit"] = "auto"
    return {"constructor": {"label": "anomaly", "problem_type": "binary", "eval_metric": "roc_auc",
                            "positive_class": 1, "learner_kwargs": {"random_state": 42}}, "fit": fit}


def _verify_approval() -> None:
    path = _approval_path()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("approval source symlink forbidden")
    approval = _read_json(path)
    profile = resolved_profile()
    resource_policy = {"informed_opt_out": True, "rss_limit_enforced": False,
                       "max_rss_bytes": None, "cpu_limit_enforced": False,
                       "cpu_threads": None, "wall_limit_enforced": False,
                       "max_wall_seconds": None}
    fit_policy = {name: value for name, value in profile["fit"].items()
                  if name not in {"hyperparameters", "memory_limit", "num_cpus", "num_gpus"}}
    if (approval.get("record_kind") != "approval_not_execution_proof"
            or approval.get("native_memory_protection") != "keep"
            or bridge._canonical(approval.get("constructor")) != bridge._canonical(profile["constructor"])
            or bridge._canonical(approval.get("xt_hyperparameters")) != bridge._canonical(profile["fit"]["hyperparameters"]["XT"])
            or approval.get("memory_limit_backend") != "auto"
            or bridge._canonical(approval.get("resource_policy")) != bridge._canonical(resource_policy)
            or bridge._canonical(approval.get("preserved_native_xt_defaults")) != bridge._canonical(_NATIVE_FOREST_DEFAULTS)
            or bridge._canonical(approval.get("fit_policy")) != bridge._canonical(fit_policy)
            or approval.get("execution_evidence") is not None):
        raise ValueError("approved profile declaration mismatch; not execution proof")


def _read_json(path: Path) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for name, value in values:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result
    def invalid(_value):
        raise ValueError("nonfinite JSON")
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=invalid)


def _publish(root: Path, name: str, payload: dict[str, Any]) -> None:
    path = bridge.confined_path(root, name)
    temporary = bridge.confined_path(root, f"message-{secrets.token_hex(16)}.json")
    try:
        with temporary.open("xb") as stream:
            stream.write(bridge._canonical(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def bind_worker_request(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Parent-side mechanical binding; parent must authenticate/admit snapshot."""
    if "backend_binding" in request:
        raise ValueError("backend binding already exists")
    _verify_approval()
    return dict(request, backend_binding={
        "version": 1, "worker_sha256": bridge._file_hash(Path(__file__)),
        "bridge_sha256": bridge._file_hash(Path(bridge.__file__)),
        "lifecycle_sha256": bridge._file_hash(bridge.confined_path(root, "worker-request.json")),
        "approval_sha256": bridge._file_hash(_approval_path()), "resolved_profile": resolved_profile()})


def _live_environment() -> dict[str, Any]:
    names = ("pandas", "pyarrow", "autogluon.tabular", "scikit-learn")
    return dict({name: importlib.metadata.version(name) for name in names},
                python=sys.version.split()[0], executable=sys.executable, prefix=sys.prefix)


def _lifecycle(root: Path, request: dict[str, Any], key: bytes) -> dict[str, Any]:
    binding = request.get("backend_binding")
    if (not isinstance(binding, dict) or set(binding) != _BINDING_FIELDS
            or type(binding["version"]) is not int or binding["version"] != 1):
        raise ValueError("authenticated backend source binding required")
    paths = {"worker_sha256": Path(__file__), "bridge_sha256": Path(bridge.__file__),
             "approval_sha256": _approval_path(),
             "lifecycle_sha256": bridge.confined_path(root, "worker-request.json")}
    for field, path in paths.items():
        if bridge._file_hash(path) != binding[field]:
            raise ValueError("worker/bridge/lifecycle source hash mismatch")
    _verify_approval()
    if bridge._canonical(binding["resolved_profile"]) != bridge._canonical(resolved_profile()):
        raise ValueError("authenticated resolved backend profile mismatch")
    envelope = _read_json(paths["lifecycle_sha256"])
    if set(envelope) != {"payload", "signature"}:
        raise ValueError("invalid lifecycle envelope")
    payload = envelope["payload"]
    if not hmac.compare_digest(bridge.authenticate(payload, key), envelope["signature"]):
        raise ValueError("lifecycle authentication failed")
    from project.experiments.full_snapshot_codec import validate_request
    spec = validate_request(payload)
    approvals = [(source.path, source.sha256) for source in spec.sources if source.role == "autogluon_profile"]
    if approvals != [("docs/evidence/autogluon_full_profile_v1.json", request["backend_binding"]["approval_sha256"])]:
        raise ValueError("approval must be hash-bound in the authenticated configuration snapshot")
    snapshot = next(item for item in spec.candidates if item.spec.experiment_id == payload["experiment_id"])
    binding = request["binding"]
    data = [source.sha256 for source in spec.sources if source.role == "data"]
    split = [source.sha256 for source in spec.sources if source.role == "split"]
    train_features = [item["name"] for item in request["train"]["schema"] if item["name"] != "anomaly"]
    if (spec.run_id != binding["run_id"] or spec.binding_sha256 != binding["config_hash"]
            or data != [binding["data_hash"]] or split != [binding["split_hash"]]
            or snapshot.spec.model_family != "autogluon" or snapshot.study is not None or snapshot.study_method is not None
            or snapshot.spec.model_params or snapshot.spec.method_params or snapshot.selector_arms or payload["arm_id"] != "challenger"
            or tuple(train_features) != snapshot.spec.feature_pool):
        raise ValueError("bridge transport does not match authenticated snapshot/source/schema")
    if (payload.get("attempt_id") != request["binding"]["attempt_id"]
            or not (root.name == payload["attempt_id"]
                    or (root.name == "backend" and root.parent.name == payload["attempt_id"]
                        and root.parent.parent.name == "attempts"))
            or payload.get("experiment_id") != "model_autogluon_core_features"
            or type(payload.get("parent_pid")) is not int
            or payload["parent_pid"] != os.getppid()
            or not isinstance(payload.get("spec"), dict)
            or not isinstance(payload.get("spec_sha256"), str)
            or len(payload["spec_sha256"]) != 64
            or not isinstance(payload.get("work_id"), str)
            or not payload["work_id"].startswith("model_autogluon_core_features/")
            or not payload["work_id"].endswith("/model")
            or not isinstance(payload.get("nonce"), str)
            or len(payload["nonce"]) != 64
            or any(c not in "0123456789abcdef" for c in payload["nonce"])):
        raise ValueError("lifecycle snapshot/parent/unit/nonce identity mismatch")
    # The full snapshot's validation and live source/config bindings remain the
    # parent service's responsibility; this request signs its entire envelope hash.
    return payload


def _debit(lifecycle: dict[str, Any], key: bytes, exchange: Callable) -> dict[str, Any]:
    event = {"event": "debit", "seq": 1, "phase": "model",
             "work_id": lifecycle["work_id"], "attempt_id": lifecycle["attempt_id"],
             "nonce": lifecycle["nonce"], "unit": lifecycle["work_id"]}
    ack = exchange(event)
    expected = dict(event, status="authorized")
    if (not isinstance(ack, dict) or set(ack) != {"payload", "signature"}
            or bridge._canonical(ack["payload"]) != bridge._canonical(expected)
            or not isinstance(ack["signature"], str)
            or not hmac.compare_digest(bridge.authenticate(expected, key), ack["signature"])):
        raise ValueError("invalid parent debit ack")
    return expected


def _backend_type():
    from autogluon.tabular import TabularPredictor, __version__
    if __version__ != "1.4.0":
        raise ValueError("reviewed AutoGluon1.4.0 API required")
    signature = inspect.signature(TabularPredictor.fit)
    for name in ("fit_strategy", "time_limit", "memory_limit", "hyperparameters",
                 "dynamic_stacking", "fit_weighted_ensemble", "calibrate_decision_threshold"):
        if name not in signature.parameters:
            raise ValueError("reviewed explicit fit guard missing")
    return TabularPredictor


@contextlib.contextmanager
def _model_audit(trainer_type: type, xt_type: type, owner: list | None = None):
    """Count attempts before calls, including failed/non-surviving models."""
    original = trainer_type._train_single
    signature = inspect.signature(original)
    records = []
    violations = []
    owner = [] if owner is None else owner
    lock = threading.Lock()
    def recorded(trainer, *args, **kwargs):
        if not lock.acquire(blocking=False):
            violations.append("overlap")
            raise ValueError("overlapping model attempt forbidden")
        try:
            model = signature.bind(trainer, *args, **kwargs).arguments["model"]
            if records or type(model) is not xt_type or (owner and owner[0] is not model):
                violations.append("extra_or_wrong_model")
                raise ValueError("exactly one XT model attempt/object required")
            if not owner:
                owner.append(model)
            record = {"attempt": 1, "model_type": xt_type.__name__, "outcome": "running",
                      "rejected_attempts": violations}
            records.append(record)
            try:
                result = original(trainer, *args, **kwargs)
                if result is not model:
                    raise ValueError("model attempt returned a different object")
                record["outcome"] = "success"
                return result
            except BaseException:
                record["outcome"] = "failed"
                raise
        finally:
            lock.release()
    trainer_type._train_single = recorded
    try:
        yield records
        if violations or len(records) != 1 or records[0]["outcome"] != "success":
            raise ValueError("model attempt audit failed; swallowed failures are not success")
    finally:
        trainer_type._train_single = original


@contextlib.contextmanager
def _abstract_fit_audit(abstract_type: type, xt_type: type, owner: list):
    """Also cover direct AbstractModel.fit calls outside trainer inventory."""
    original = abstract_type.fit
    records = []
    violations = []
    lock = threading.Lock()
    def recorded(model, *args, **kwargs):
        if not lock.acquire(blocking=False):
            violations.append("overlap")
            raise ValueError("overlapping abstract model fit attempt")
        try:
            if records or type(model) is not xt_type or len(owner) != 1 or owner[0] is not model:
                violations.append("extra_or_unrouted")
                raise ValueError("unrouted or extra abstract model fit attempt")
            record = {"attempt": 1, "model_type": "XTModel", "outcome": "running",
                      "rejected_attempts": violations}
            records.append(record)
            try:
                result = original(model, *args, **kwargs)
                if result is not model:
                    raise ValueError("abstract fit changed model identity")
                record["outcome"] = "success"
                return result
            except BaseException:
                record["outcome"] = "failed"
                raise
        finally:
            lock.release()
    abstract_type.fit = recorded
    try:
        yield records
        if violations or len(records) != 1 or records[0]["outcome"] != "success":
            raise ValueError("abstract model attempt audit failed")
    finally:
        abstract_type.fit = original


@contextlib.contextmanager
def _reviewed_model_audit():
    from autogluon.core.models.abstract.abstract_model import AbstractModel
    from autogluon.tabular.models.xt.xt_model import XTModel
    from autogluon.tabular.trainer.abstract_trainer import AbstractTabularTrainer
    if (XTModel.fit is not AbstractModel.fit
            or "model" not in inspect.signature(AbstractTabularTrainer._train_single).parameters):
        raise ValueError("reviewed model-fit entrypoints unsupported")
    owner = []
    with (_model_audit(AbstractTabularTrainer, XTModel, owner) as trainer_attempts,
          _abstract_fit_audit(AbstractModel, XTModel, owner) as abstract_attempts):
        yield {"trainer_attempts": trainer_attempts, "abstract_fit_attempts": abstract_attempts}


@contextlib.contextmanager
def _forest_audit():
    """Observe sequential growth on one estimator, not independent auxiliary fits."""
    from sklearn.ensemble import ExtraTreesClassifier
    original = ExtraTreesClassifier.fit
    records = []
    owner = []
    violations = []
    lock = threading.Lock()
    def recorded(model, *args, **kwargs):
        if not lock.acquire(blocking=False):
            violations.append("overlap")
            raise ValueError("concurrent/reentrant forest fit forbidden")
        try:
            if (owner and owner[0] is not model) or len(records) >= 2:
                violations.append("extra_forest")
                raise ValueError("extra independent forest fit forbidden")
            if not owner:
                owner.append(model)
            expected = {"random_state": 42, "max_depth": 12, "min_samples_leaf": 2,
                        **_NATIVE_FOREST_DEFAULTS}
            if any(bridge._canonical(getattr(model, name, None)) != bridge._canonical(value)
                   for name, value in expected.items()):
                violations.append("forest_profile")
                raise ValueError("forest seed/native profile mismatch")
            if type(model.n_estimators) is not int or model.n_estimators not in (4, 64):
                violations.append("tree_parameter")
                raise ValueError("integer 4/64 forest growth parameter required")
            if ((not records and model.n_estimators == 4 and model.warm_start is not True)
                    or (records and (records[0]["outcome"] != "success"
                                     or records[0]["n_estimators"] != 4
                                     or model.n_estimators != 64 or model.warm_start is not True))):
                violations.append("growth_profile")
                raise ValueError("forest warm-start growth profile mismatch")
            record = {"n_estimators": model.n_estimators, "actual_tree_count": None,
                      "outcome": "running", "warm_start": bool(model.warm_start),
                      "n_jobs": model.n_jobs, "random_state": model.random_state,
                      "max_leaf_nodes": model.max_leaf_nodes, "rejected_attempts": violations}
            record.update({name: getattr(model, name) for name in expected})
            records.append(record)
            try:
                result = original(model, *args, **kwargs)
                record["actual_tree_count"] = len(model.estimators_)
                if (result is not model or len(model.estimators_) != record["n_estimators"]
                        or model.n_estimators != record["n_estimators"]
                        or any(bridge._canonical(getattr(model, name, None)) != bridge._canonical(value)
                               for name, value in expected.items())):
                    raise ValueError("actual forest tree count/identity/seed/native profile mismatch")
                record["outcome"] = "success"
                return result
            except BaseException:
                record["outcome"] = "failed"
                violations.append("failed_forest_call")
                raise
        finally:
            lock.release()
    ExtraTreesClassifier.fit = recorded
    try:
        yield records
        if (violations or [record["actual_tree_count"] for record in records] not in ([64], [4, 64])
                or any(record["outcome"] != "success" for record in records)):
            raise ValueError("forest audit failed; final actual tree count must be 64")
    finally:
        ExtraTreesClassifier.fit = original


def _inventory(backend: Any) -> dict[str, Any]:
    names = backend.model_names()
    if names != ["ExtraTrees"]:
        raise ValueError("backend inventory must contain exactly ExtraTrees")
    classes = backend.classes_
    if (len(classes) != 2 or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in classes)
            or list(classes) != [0, 1] or backend.positive_class != 1
            or not isinstance(backend.positive_class, (int, np.integer))
            or isinstance(backend.positive_class, (bool, np.bool_))):
        raise ValueError("backend positive class/order mismatch")
    info = backend.info()["model_info"]["ExtraTrees"]
    params = info["hyperparameters"]
    if (info["model_type"] != "XTModel" or
            any(bridge._canonical(params.get(k)) != bridge._canonical(v) for k, v in bridge.full_xt_profile()["hyperparameters"]["XT"].items())
            or info["hyperparameters_fit"].get("n_estimators") != 64):
        raise ValueError("backend fitted XT profile changed or trees reduced")
    return {"models": names, "model_type": "XTModel", "n_estimators": 64,
            "effective_model_params": {name: params.get(name) for name in
                ("n_estimators", "max_depth", "min_samples_leaf", "n_jobs", "random_state",
                 "max_leaf_nodes", "bootstrap", "criterion", "oob_score")},
            "native_memory_heuristics": "retained; reduced trees rejected",
            "feature_generation": "train-only unsupervised backend stages",
            "independent_model_units": 1, "version": "1.4.0"}


def _write_scores(root: Path, backend: Any, batch_rows: int = BATCH_ROWS) -> int:
    schema = pa.schema([pa.field("row_index", pa.int64(), nullable=False),
                        pa.field("probability", pa.float64(), nullable=False)])
    output = bridge.confined_path(root, "scores.parquet")
    prediction = pq.ParquetFile(bridge.confined_path(root, "predict.parquet"))
    with output.open("xb") as stream:
        with pq.ParquetWriter(stream, schema) as writer:
            batches = 0
            for batch in prediction.iter_batches(batch_size=batch_rows):
                batches += 1
                table = pa.Table.from_batches([batch])
                rows = table.column("row_index")
                if rows.null_count:
                    raise ValueError("null prediction row identity")
                frame = table.drop(["row_index"]).to_pandas()
                result = backend.predict_proba(data=frame, as_pandas=True, as_multiclass=True)
                if (not isinstance(result, pd.DataFrame) or not result.index.equals(frame.index)
                        or result.columns.tolist() != [0, 1] or result.shape != (len(frame), 2)
                        or any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                               for value in result.columns)):
                    raise ValueError("probability index/class order mismatch")
                if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                       or pd.api.types.is_complex_dtype(dtype) for dtype in result.dtypes):
                    raise ValueError("probability columns require real nonboolean numeric dtypes")
                values = result.to_numpy(dtype=np.float64)
                if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                        or not np.allclose(values.sum(axis=1), 1., rtol=1e-7, atol=1e-8)):
                    raise ValueError("invalid class probabilities")
                writer.write_table(pa.Table.from_arrays([rows.cast(pa.int64()), pa.array(values[:, 1])], schema=schema))
        stream.flush()
        os.fsync(stream.fileno())
    return batches


def run_backend(root: Path, authentication: tuple[dict, bytes, str], exchange: Callable,
                *, backend_type: type | None = None) -> dict[str, Any]:
    """Backend injection is a Python test boundary, never JSON/CLI selectable."""
    request, key, signature = authentication
    model_attempts = None
    forest_calls = None
    bridge.confined_path(root, "request.json")
    if root.stat().st_uid != os.geteuid() or root.stat().st_mode & 0o077:
        raise ValueError("private owner-only backend attempt directory required")
    for name in ("scores.parquet", "response.json", "backend-status.json", "backend-started.json"):
        if bridge.confined_path(root, name).exists():
            raise ValueError("fresh backend attempt required; no overwrite/replay")
    try:
        bridge.verify_request(root, request, (key, signature))
        lifecycle = _lifecycle(root, request, key)
        facts = _live_environment()
        bridge.validate_environment(facts)
        if facts != request["environment"]:
            raise ValueError("actual backend environment/interpreter mismatch")
        if "AG_MEMORY_LIMIT_IN_GB" in os.environ:
            raise ValueError("inherited AutoGluon memory soft cap forbidden under opt-out")
        real_backend = backend_type is None
        predictor_type = _backend_type() if real_backend else backend_type
        model_path = root / "backend-models"
        if model_path.exists() or model_path.is_symlink():
            raise ValueError("exclusive backend model directory required")
        debit = _debit(lifecycle, key, exchange)
        _publish(root, "backend-started.json", {"status": "running", "debit": debit,
                 "request_hash": hashlib.sha256(bridge._canonical(request)).hexdigest()})
        # Materialize precisely one admitted train-only partition, never prediction labels.
        train = pq.read_table(bridge.confined_path(root, "train.parquet")).to_pandas()
        profile = resolved_profile()
        backend = predictor_type(path=str(model_path), **profile["constructor"])
        if type(backend._learner.random_state) is not int or backend._learner.random_state != 42:
            raise ValueError("backend learner seed must remain integer 42")
        params = profile["fit"]
        # Installed 1.4.0 rejects None. 'auto' sets no AG_MEMORY_LIMIT_IN_GB
        # override; native library memory heuristics remain observable, not caps.
        params["memory_limit"] = "auto"
        fit_signature = inspect.signature(backend.fit)
        extras = {k: v for k, v in params.items() if k not in fit_signature.parameters}
        resolved = backend._validate_fit_kwargs(kwargs=extras.copy())
        if any(bridge._canonical(resolved.get(k)) != bridge._canonical(v) or k not in resolved for k, v in extras.items()):
            raise ValueError("backend drops/changes required fit guards")
        bridge.verify_request(root, request, (key, signature))
        _lifecycle(root, request, key)
        audit = _forest_audit() if real_backend else contextlib.nullcontext(None)
        model_audit = _reviewed_model_audit() if real_backend else contextlib.nullcontext(None)
        with model_audit as model_attempts, audit as forest_calls:
            backend.fit(train_data=train, **params)
        if type(backend._learner.random_state) is not int or backend._learner.random_state != 42:
            raise ValueError("backend learner seed changed during fitting")
        del train
        inventory = _inventory(backend)
        if real_backend and (not forest_calls or forest_calls[-1]["n_estimators"] != 64
                             or (len(forest_calls) == 2 and [v["n_estimators"] for v in forest_calls] != [4, 64])):
            raise ValueError("unexpected backend forest growth calls")
        inventory.update(evidence_kind="runtime_instrumented" if real_backend else "recording_fake",
                         forest_fit_calls=forest_calls, resolved_fit_profile=params,
                         model_fit_attempts=model_attempts, learner_random_state=42,
                         approval_sha256=request["backend_binding"]["approval_sha256"],
                         resource_policy={"max_wall_seconds": None, "max_rss_bytes": None,
                                          "wall_limit_enforced": False, "rss_limit_enforced": False,
                                          "cpu_limit_enforced": False, "cpu_threads": None,
                                          "informed_opt_out": True},
                         effective_thread_environment={name: os.environ.get(name) for name in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        bridge.verify_request(root, request, (key, signature))
        inventory["prediction_batches"] = _write_scores(root, backend, request.get("prediction_batch_rows", BATCH_ROWS))
        bridge.verify_request(root, request, (key, signature))
        _lifecycle(root, request, key)
        response = bridge.prepare_response(root, request)
        response["backend"] = inventory
        bridge.verify_response(root, request, response)
        _publish(root, "response.json", {"payload": response, "signature": bridge.authenticate(response, key)})
        _publish(root, "backend-status.json", {"status": "awaiting_parent_validation",
                 "request_hash": response["request_hash"]})
        return response
    except BaseException as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        if not bridge.confined_path(root, "backend-status.json").exists():
            _publish(root, "backend-status.json", {"status": status, "error_type": type(error).__name__,
                     "model_fit_attempts": model_attempts, "forest_fit_calls": forest_calls,
                     "evidence_kind": "runtime_instrumented" if backend_type is None else "recording_fake"})
        raise


def _file_exchange(root: Path, lifecycle: dict[str, Any], key: bytes) -> Callable:
    timeout = lifecycle.get("ack_timeout_seconds")
    if type(timeout) is not float or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("finite positive parent acknowledgement timeout required")
    def exchange(event):
        ack = bridge.confined_path(root, "ack-0001.json")
        if ack.exists():
            raise ValueError("stale acknowledgement forbidden")
        _publish(root, "event-0001.json", {"payload": event, "signature": bridge.authenticate(event, key)})
        deadline = time.monotonic() + timeout
        while not ack.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("parent debit acknowledgement timed out; not a fit time limit")
            time.sleep(.02)
        return _read_json(bridge.confined_path(root, "ack-0001.json"))
    return exchange


def run(unit: str, directory: str) -> None:
    """Existing supervisor target; consume a separate signed backend envelope."""
    root = Path(directory)
    secret = os.environ.pop("FULL_CAMPAIGN_WORKER_KEY", None)
    if not secret:
        raise ValueError("parent authentication key required")
    key = secret.encode()
    envelope = _read_json(bridge.confined_path(root, "backend-request.json"))
    if set(envelope) != {"payload", "signature"}:
        raise ValueError("signed backend request required")
    request = envelope["payload"]
    bridge.verify_request(root, request, (key, envelope["signature"]))
    lifecycle = _lifecycle(root, request, key)
    if lifecycle["work_id"] != unit:
        raise ValueError("supervisor unit mismatch")
    run_backend(root, (request, key, envelope["signature"]), _file_exchange(root, lifecycle, key))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-directory", required=True, type=Path)
    args = parser.parse_args()
    root = args.attempt_directory
    secret = os.environ.pop("FULL_CAMPAIGN_WORKER_KEY", None)
    if not secret:
        raise ValueError("parent authentication key required")
    key = secret.encode()
    envelope = _read_json(bridge.confined_path(root, "request.json"))
    if set(envelope) != {"payload", "signature"}:
        raise ValueError("signed backend request required")
    request = envelope["payload"]
    bridge.verify_request(root, request, (key, envelope["signature"]))
    lifecycle = _lifecycle(root, request, key)
    def cancelled(_signal, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        run_backend(root, (request, key, envelope["signature"]), _file_exchange(root, lifecycle, key))
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
