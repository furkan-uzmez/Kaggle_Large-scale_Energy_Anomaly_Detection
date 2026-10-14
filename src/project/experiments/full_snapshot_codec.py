"""Versioned exact-type transport; snapshots are declarations, never admission."""

from __future__ import annotations

import hashlib
import json
import math
import re
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

from project.experiments.candidate_contracts import CandidateSpec, StudySpec
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1

_CLASSES = (FullRunSpec, FullCandidateSnapshot, CandidateSpec, StudySpec,
            FullResourcePolicy, SourceBinding)
PROVENANCE_PATH = "docs/evidence/year_built_encoding_v1.json"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_json(text: str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def exact_fields(value: object, names: set[str]) -> dict:
    if type(value) is not dict or set(value) != names:
        raise ValueError("exact schema fields required")
    return value


def _encode(value):
    if type(value) in _CLASSES:
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if is_dataclass(value):
        raise TypeError("unsupported snapshot object")
    if type(value) is tuple:
        return {"$tuple": [_encode(item) for item in value]}
    if type(value) is list:
        return {"$list": [_encode(item) for item in value]}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("string mapping keys required")
        return {"$map": {key: _encode(item) for key, item in value.items()}}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"unsupported or nonfinite snapshot value: {type(value).__module__}.{type(value).__qualname__}")


def _dynamic(value):
    if type(value) is dict:
        if set(value) == {"$tuple"} and type(value["$tuple"]) is list:
            return tuple(_dynamic(item) for item in value["$tuple"])
        if set(value) == {"$list"} and type(value["$list"]) is list:
            return [_dynamic(item) for item in value["$list"]]
        if set(value) == {"$map"} and type(value["$map"]) is dict:
            return {key: _dynamic(item) for key, item in value["$map"].items()}
        raise ValueError("unknown value encoding")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("invalid snapshot primitive")


def _decode(value, expected):
    origin, args = get_origin(expected), get_args(expected)
    if expected in _CLASSES:
        value = exact_fields(value, {field.name for field in fields(expected)})
        hints = get_type_hints(expected)
        return expected(**{name: _decode(item, hints[name]) for name, item in value.items()})
    if origin is types.UnionType:
        for option in args:
            try:
                return _decode(value, option)
            except (TypeError, ValueError):
                pass
        raise TypeError("invalid union field type")
    if origin is Literal:
        if not any(type(value) is type(option) and value == option for option in args):
            raise ValueError("invalid literal field")
        return value
    if origin is tuple or expected is tuple:
        items = exact_fields(value, {"$tuple"})["$tuple"]
        if type(items) is not list:
            raise TypeError("tuple wire items required")
        if not args:
            return tuple(_dynamic(item) for item in items)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, args[0]) for item in items)
        if len(items) != len(args):
            raise ValueError("tuple arity mismatch")
        return tuple(_decode(item, kind) for item, kind in zip(items, args, strict=True))
    if expected is object:
        return _dynamic(value)
    if expected is None:
        expected = type(None)
    if type(value) is not expected:
        raise TypeError("exact field type required")
    return _dynamic(value)


def encode_snapshot(spec: FullRunSpec) -> dict:
    if type(spec) is not FullRunSpec:
        raise TypeError("exact FullRunSpec required")
    envelope = {"codec_version": 1, "spec": _encode(spec), "spec_sha256": spec.binding_sha256}
    # Encoding must itself satisfy the decoder; no lossy coercion is allowed.
    decode_snapshot(envelope)
    return envelope


def decode_snapshot(envelope: object) -> FullRunSpec:
    value = exact_fields(envelope, {"codec_version", "spec", "spec_sha256"})
    if type(value["codec_version"]) is not int or value["codec_version"] != 1:
        raise ValueError("unsupported snapshot codec version")
    spec = _decode(value["spec"], FullRunSpec)
    if type(value["spec_sha256"]) is not str or value["spec_sha256"] != spec.binding_sha256:
        raise ValueError("snapshot digest mismatch")
    return spec


def semantic_digest(encoding: str, provenance_sha256: str) -> str:
    return hashlib.sha256(f"{encoding}:{provenance_sha256}".encode()).hexdigest()


def configuration_digest(graph: object) -> str:
    """Encode the explicit type-sensitive dispatch graph without string fallback."""
    def encode(item):
        if isinstance(item, Enum):
            return {"enum": type(item).__module__ + "." + type(item).__qualname__,
                    "name": item.name, "value": _encode(item.value)}
        if isinstance(item, type):
            return {"type": item.__module__ + "." + item.__qualname__}
        if type(item) is types.FunctionType:
            if "<" in item.__qualname__:
                raise TypeError("anonymous configuration function unsupported")
            return {"function": item.__module__ + "." + item.__qualname__}
        if type(item) is tuple:
            return {"tuple": [encode(value) for value in item]}
        # sklearn's declared missing_values=np.nan is a real parameter, not a
        # JSON number. Represent its type/sign explicitly, never emit NaN JSON.
        if type(item) is float and not math.isfinite(item):
            return {"float": "nan" if math.isnan(item) else ("+inf" if item > 0 else "-inf")}
        return _encode(item)

    return hashlib.sha256(canonical_json(encode(graph)).encode()).hexdigest()


REQUEST_FIELDS = {"version", "project_root", "spec", "spec_sha256", "index_path",
                  "index_sha256", "experiment_id", "arm_id", "repeat", "fold",
                  "year_built_encoding", "provenance_path", "provenance_sha256",
                  "semantic_sha256", "work_id", "attempt_id", "output_path",
                  "parent_pid", "ack_timeout_seconds", "nonce"}


def validate_request(payload: object) -> FullRunSpec:
    value = exact_fields(payload, REQUEST_FIELDS)
    for name in REQUEST_FIELDS - {"spec", "version", "repeat", "fold", "parent_pid", "ack_timeout_seconds"}:
        if type(value[name]) is not str or not value[name]:
            raise TypeError(f"exact nonempty string required: {name}")
    for name in ("version", "repeat", "fold", "parent_pid"):
        if type(value[name]) is not int:
            raise TypeError(f"exact integer required: {name}")
    if value["version"] != 1 or value["parent_pid"] <= 0:
        raise ValueError("invalid request version or parent identity")
    if type(value["ack_timeout_seconds"]) is not float or not math.isfinite(value["ack_timeout_seconds"]) or value["ack_timeout_seconds"] <= 0:
        raise ValueError("positive finite float acknowledgement timeout required")
    if not re.fullmatch(r"[0-9a-f]{64}", value["nonce"]):
        raise ValueError("fresh 256-bit request nonce required")
    for name in ("index_sha256", "spec_sha256", "provenance_sha256", "semantic_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", value[name]) or value[name] == "0" * 64:
            raise ValueError("non-placeholder SHA-256 required")
    spec = decode_snapshot(value["spec"])
    if spec.binding_sha256 != value["spec_sha256"]:
        raise ValueError("request spec digest mismatch")
    snapshot = next((item for item in spec.candidates if item.spec.experiment_id == value["experiment_id"]), None)
    if snapshot is None or value["arm_id"] not in snapshot.arms:
        raise ValueError("request candidate/arm absent from snapshot")
    if not 0 <= value["repeat"] < spec.repeats or not 0 <= value["fold"] < spec.folds:
        raise ValueError("request partition outside frozen protocol")
    prefix = f"{value['experiment_id']}/{value['arm_id']}/r{value['repeat']}/f{value['fold']}"
    if value["work_id"] != prefix + "/model" or value["work_id"] not in spec.planned_units:
        raise ValueError("ordinary worker requires exact planned model unit; HPO route separate")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value["attempt_id"]) or value["output_path"] != "shard.parquet":
        raise ValueError("invalid attempt/output identity")
    if not Path(value["project_root"]).is_absolute() or not Path(value["index_path"]).is_absolute():
        raise ValueError("explicit absolute source and index paths required")
    bindings = [item for item in spec.sources if item.role == "encoding"]
    if (len(bindings) != 1 or bindings[0].path != PROVENANCE_PATH
            or value["provenance_path"] != PROVENANCE_PATH
            or bindings[0].sha256 != value["provenance_sha256"]
            or value["year_built_encoding"] != LEAD_YEAR_OFFSET_1900_UINT8_V1):
        raise ValueError("explicit immutable encoding provenance binding required")
    if value["semantic_sha256"] != semantic_digest(value["year_built_encoding"], value["provenance_sha256"]):
        raise ValueError("semantic digest mismatch")
    return spec
