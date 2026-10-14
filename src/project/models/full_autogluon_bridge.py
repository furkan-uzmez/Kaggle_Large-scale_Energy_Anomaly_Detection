"""Versioned, authenticated Parquet transport; NOT a fit or launch grant.

The parent owns admission, process groups, cancellation, accounting and sealing.
No bounded adapter lease is synthesized. Only a repository-owned worker may use
the validated profile; this module never unpickles or loads backend models.
Run directories must be private, exclusively allocated by the supervisor. These
checks protect accidental/path payload misuse, not hostile concurrent filesystem
mutation by another process running as the same user.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROTOCOL_VERSION = 1
SCORE_SEMANTICS = {"score_type": "probability", "positive_label": 1,
                   "class_order": [0, 1], "row_order": "input"}


def full_xt_profile() -> dict[str, Any]:
    """Preserve the frozen CPU model profile, without campaign wall/RSS caps."""
    return {"hyperparameters": {"XT": {"n_estimators": 64, "max_depth": 12,
                                      "min_samples_leaf": 2, "random_state": 42}},
            "presets": None, "time_limit": None, "memory_limit": None,
            "num_bag_folds": 0, "num_stack_levels": 0, "auto_stack": False,
            "calibrate": False, "hyperparameter_tune_kwargs": None,
            "dynamic_stacking": False, "fit_weighted_ensemble": False,
            "calibrate_decision_threshold": False, "num_cpus": 1,
            "num_gpus": 0, "fit_strategy": "sequential",
            "refit_full": False, "set_best_to_refit_full": False,
            "feature_prune_kwargs": None, "num_bag_sets": 1,
            "learning_curves": False, "raise_on_model_failure": True,
            "raise_on_no_models_fitted": True}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_profile(profile: dict[str, Any]) -> None:
    if _canonical(profile) != _canonical(full_xt_profile()):
        raise ValueError("unknown or changed AutoGluon XT profile")


def confined_path(root: Path, name: str) -> Path:
    root = Path(root)
    relative = Path(name)
    if (not root.is_absolute() or relative.is_absolute() or len(relative.parts) != 1
            or relative.name in ("", ".", "..")
            or relative.suffix not in (".json", ".parquet")):
        raise ValueError("transport requires a confined JSON/Parquet basename")
    for part in (root, *root.parents):
        if part.is_symlink():
            raise ValueError("symlink in run path")
    if not root.is_dir():
        raise ValueError("run root must already exist")
    path = root / relative
    if path.is_symlink():
        raise ValueError("symlink transport forbidden")
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError("transport must be an unlinked regular file")
    return path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(root: Path, name: str) -> dict[str, Any]:
    path = confined_path(root, name)
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if len(set(schema.names)) != len(schema.names):
        raise ValueError("duplicate transport fields")
    for field in schema:
        if not (pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
                or pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise ValueError("unsupported transport type; no binary/object payloads")
    return {"path": name, "sha256": _file_hash(path), "rows": parquet.metadata.num_rows,
            "schema": [{"name": f.name, "type": str(f.type), "nullable": f.nullable}
                       for f in schema]}


def _validate_binding(binding: dict[str, Any]) -> None:
    if set(binding) != {"run_id", "attempt_id", "config_hash", "data_hash", "split_hash"}:
        raise ValueError("exact run/attempt/config/data/split binding required")
    for name, value in binding.items():
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("invalid binding identity")
        if name.endswith("hash") and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("invalid binding hash")


def prepare_request(root: Path, binding: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    _validate_binding(binding)
    validate_environment(environment)
    request = {"protocol_version": PROTOCOL_VERSION, "binding": binding,
               "environment": environment, "profile": full_xt_profile(),
               "score_semantics": dict(SCORE_SEMANTICS),
               "train": _descriptor(root, "train.parquet"),
               "predict": _descriptor(root, "predict.parquet")}
    _validate_tables(root, request)
    return json.loads(_canonical(request))


def _validate_tables(root: Path, request: dict[str, Any]) -> None:
    train = pq.ParquetFile(confined_path(root, "train.parquet"))
    predict = pq.ParquetFile(confined_path(root, "predict.parquet"))
    from project.features.registry import build_lead_baseline_registry
    features = [f for f in train.schema_arrow if f.name != "anomaly"]
    if (not features or any(f.name not in build_lead_baseline_registry().names for f in features)
            or train.schema_arrow.names.count("anomaly") != 1
            or not pa.types.is_integer(train.schema_arrow.field("anomaly").type)):
        raise ValueError("approved core training schema and integer anomaly required")
    expected = [(f.name, str(f.type)) for f in features]
    observed = [(f.name, str(f.type)) for f in predict.schema_arrow if f.name != "row_index"]
    if expected != observed or "row_index" not in predict.schema_arrow.names:
        raise ValueError("prediction schema must match training features with row_index")
    if not pa.types.is_integer(predict.schema_arrow.field("row_index").type):
        raise ValueError("integer row_index required")
    labels = set()
    for batch in train.iter_batches(columns=["anomaly"], batch_size=64000):
        labels.update(batch.column(0).to_pylist())
    if labels != {0, 1}:
        raise ValueError("training requires both nonnull binary integer classes")


def authenticate(payload: dict[str, Any], key: bytes) -> str:
    """Supervisor-held secret stays out of JSON; authentication is not admission."""
    if not isinstance(key, bytes) or not key:
        raise ValueError("authentication key required")
    return hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()


def verify_request(root: Path, request: dict[str, Any], authentication: tuple[bytes, str]) -> None:
    key, signature = authentication
    if not hmac.compare_digest(authenticate(request, key), signature):
        raise ValueError("request authentication failed")
    names = {"protocol_version", "binding", "environment", "profile", "score_semantics", "train", "predict"}
    if not names <= set(request) or not (set(request) - names) <= {"backend_binding", "prediction_batch_rows"}:
        raise ValueError("unknown request fields")
    if "prediction_batch_rows" in request and (type(request["prediction_batch_rows"]) is not int
            or not 1 <= request["prediction_batch_rows"] <= 64000):
        raise ValueError("invalid prediction batch rows")
    if type(request["protocol_version"]) is not int or request["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    validate_profile(request["profile"])
    _validate_binding(request["binding"])
    validate_environment(request["environment"])
    if _canonical(request["score_semantics"]) != _canonical(SCORE_SEMANTICS):
        raise ValueError("score semantics mismatch")
    for role in ("train", "predict"):
        if request[role] != _descriptor(root, role + ".parquet"):
            raise ValueError("transport hash/schema mismatch")
    _validate_tables(root, request)


def write_request(root: Path, request: dict[str, Any]) -> None:
    path = confined_path(root, "request.json")
    with path.open("xb") as stream:
        stream.write(_canonical(request))
        stream.flush()
        os.fsync(stream.fileno())


def prepare_response(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    response = {"protocol_version": PROTOCOL_VERSION,
                "request_hash": hashlib.sha256(_canonical(request)).hexdigest(),
                "score_semantics": dict(SCORE_SEMANTICS),
                "scores": _descriptor(root, "scores.parquet")}
    verify_response(root, request, response)
    return response


def verify_response(root: Path, request: dict[str, Any], response: dict[str, Any]) -> None:
    names = {"protocol_version", "request_hash", "score_semantics", "scores"}
    if (set(response) not in (names, names | {"backend"})
            or response["protocol_version"] != PROTOCOL_VERSION
            or response["request_hash"] != hashlib.sha256(_canonical(request)).hexdigest()):
        raise ValueError("response request/version binding mismatch")
    if _canonical(response["score_semantics"]) != _canonical(SCORE_SEMANTICS):
        raise ValueError("response score semantics mismatch")
    if response["scores"] != _descriptor(root, "scores.parquet"):
        raise ValueError("response hash/schema mismatch")
    scores = pq.ParquetFile(confined_path(root, "scores.parquet"))
    prediction = pq.ParquetFile(confined_path(root, "predict.parquet"))
    if (scores.schema_arrow.names != ["row_index", "probability"]
            or not pa.types.is_integer(scores.schema_arrow.field("row_index").type)
            or not pa.types.is_floating(scores.schema_arrow.field("probability").type)
            or scores.metadata.num_rows != prediction.metadata.num_rows):
        raise ValueError("invalid probability response schema/count")
    def rows(parquet: Any, columns: list[str]):
        for batch in parquet.iter_batches(columns=columns, batch_size=64000):
            yield from zip(*(column.to_pylist() for column in batch.columns))
    for score, identity in zip(rows(scores, ["row_index", "probability"]), rows(prediction, ["row_index"]), strict=True):
        row, probability = score
        if row is None or row != identity[0] or probability is None or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability or exact input row order mismatch")


def inspect_environment(interpreter: str) -> dict[str, Any]:
    """Read-only isolated probe. This does not launch a training worker."""
    path = Path(interpreter)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("exact absolute environment interpreter required")
    code = ("import sys,json,importlib.metadata as m; "
            "names=['pandas','pyarrow','autogluon.tabular','scikit-learn']; "
            "v={}; "
            "exec('for n in names:\\n try: v[n]=m.version(n)\\n except m.PackageNotFoundError: v[n]=None'); "
            "print(json.dumps(dict(v,python=sys.version.split()[0],executable=sys.executable,prefix=sys.prefix)))")
    result = subprocess.run([str(path), "-I", "-c", code], check=True, capture_output=True, text=True)
    facts = json.loads(result.stdout)
    if facts["executable"] != str(path):
        raise ValueError("interpreter identity mismatch")
    return facts


def validate_environment(facts: dict[str, Any]) -> None:
    from packaging.version import Version
    try:
        valid = (Version(facts["python"]).release[:2] == (3, 12)
                 and Version("2") <= Version(facts["pandas"]) < Version("2.4")
                 and facts["autogluon.tabular"] == "1.4.0"
                 and all(isinstance(facts.get(n), str) and facts[n]
                         for n in ("pyarrow", "scikit-learn", "executable", "prefix"))
                 and Path(facts["executable"]).is_absolute()
                 and Path(facts["prefix"]).is_absolute())
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("AutoGluon requires verified Python3.12/pandas<2.4/Tabular1.4.0 dependencies")


def supervisor_command(interpreter: str, worker: Path, binding: dict[str, Any]) -> tuple[str, ...]:
    """Build argv only; parent authenticates binding and owns actual launch.

    The caller supplies a reviewed repository worker, not a request-selected code
    path. A file hash binds its bytes to admission. Request/authentication transport
    and process-group setup are deliberately the supervisor's responsibility.
    """
    if set(binding) != {"environment", "worker_sha256"}:
        raise ValueError("exact worker/environment binding required")
    validate_environment(binding["environment"])
    facts = inspect_environment(interpreter)
    validate_environment(facts)
    if facts != binding["environment"]:
        raise ValueError("live environment differs from authenticated binding")
    worker = Path(worker)
    if not worker.is_absolute() or worker.suffix != ".py" or not worker.is_file():
        raise ValueError("reviewed absolute Python worker required")
    if any(part.is_symlink() for part in (worker, *worker.parents)):
        raise ValueError("worker symlink forbidden")
    if _file_hash(worker) != binding["worker_sha256"]:
        raise ValueError("worker hash mismatch")
    return (interpreter, "-I", str(worker))
