"""Pinned development evidence gates; deliberately not fit/launch permission.

The immutable v1 audit binds its entire at-execution source inventory. Admission
checks the narrower *producer dependency closure* below against that pinned
inventory and both the installed producer and requested root. Unused ordinary,
HPO, RBF and AutoGluon transport modules remain historical context, not inputs to
the dev scanner/inner splitter. Changing a producer dependency requires a fresh
immutable audit version and a separately reviewed authority update. There is no
caller-supplied producer hash, readiness boolean, or runtime-promotion switch.

Candidate selection may extend the audited snapshot: data-stage proof transfers
only when all audited scientific/input source bindings, encoding, environment,
row cardinality and exact training-only inner plans still match. It does not
transfer the old snapshot's launch/configuration authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from project.experiments.candidate_contracts import TrainingPartition, _validate_managed_path
from project.experiments.full_campaign_preflight import _assignments, current_environment
from project.experiments.full_input_schema import canonical_feature_order
from project.experiments.full_run_contracts import FullRunSpec, _file_hash
from project.experiments.full_snapshot_codec import parse_json
from project.experiments.inner_cv import build_inner_folds, inner_membership_digest

EVIDENCE_PATH = "docs/evidence/full_development_audit_v3.json"
EVIDENCE_SHA256 = "bb858b1968316c2224b88e15fc9ba180c4840d1f18b8d217c3c63e23f0c9f5e0"
TARGET_ID = "feat_conditional_cross_fitted_target_encoding"
SENTINEL_ID = "feat_preserve_raw_vs_invalid_sentinel"
TARGET_METHODS = (("smoothing", 10.0), ("n_splits", 3), ("seed", 42), ("max_categories", 4096))
_REFERENCE_ROOT = Path(__file__).resolve().parents[3]
_PRODUCER_CLOSURE = (
    "scripts/audit_full_development.py",
    "src/project/experiments/full_development_audits.py",
    "src/project/experiments/inner_cv.py",
    "src/project/experiments/candidate_contracts.py",
    "src/project/experiments/full_input_schema.py",
    "src/project/experiments/full_candidate_io.py",
    "src/project/experiments/full_campaign_preflight.py",
    "src/project/experiments/full_candidate_runner.py",
    "src/project/experiments/full_run_contracts.py",
    "src/project/experiments/full_snapshot_codec.py",
    "src/project/experiments/full_campaign_cli.py",
    "src/project/features/input_encoding.py",
    "src/project/features/registry.py",
    "src/project/features/target_encoding.py",
)


@dataclass(frozen=True)
class DevelopmentReadiness:
    evidence_sha256: str
    development_identity_sha256: str
    inner_membership_sha256: tuple[tuple[int, int, str], ...]
    producer_dependencies: tuple[tuple[str, str], ...]
    sentinel_disposition: str
    target_encoding_disposition: str = "data_ready_pending_runtime_proof"
    n_splits: int = 3
    seed: int = 42
    execution_authorized: bool = field(default=False, init=False)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value: object) -> None:
    if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("evidence digest is missing or malformed")


def _read_evidence(root: Path) -> dict:
    from project.features.value_recipes import MAX_EVIDENCE_BYTES
    path = _validate_managed_path(root / EVIDENCE_PATH, "development audit evidence", root=root)
    reference = _validate_managed_path(_REFERENCE_ROOT / EVIDENCE_PATH, "reviewed development evidence", root=_REFERENCE_ROOT)
    for bound in (reference, path):
        if bound.stat().st_size > MAX_EVIDENCE_BYTES or _file_hash(bound) != EVIDENCE_SHA256:
            raise ValueError("development audit evidence authority/hash mismatch")
    report = parse_json(path.read_text())
    if type(report) is not dict or report.get("schema_version") != 1:
        raise ValueError("development evidence schema mismatch")
    if (report.get("audit_kind") != "full_development_only_no_fit"
            or report.get("execution_authorized") is not False
            or report.get("vault_content_processed") is not False
            or report.get("test_content_processed") is not False
            or type(report.get("model_fits")) is not int or report["model_fits"] != 0):
        raise ValueError("development evidence must be dev-only no-fit")
    _sha(report.get("development_identity_sha256"))
    return report


def _verify_producer(root: Path, report: dict) -> tuple[tuple[str, str], ...]:
    inventory = report["bindings"]["implementation"]
    checked = []
    for relative in _PRODUCER_CLOSURE:
        expected = inventory.get(relative)
        _sha(expected)
        for location in (_REFERENCE_ROOT, root):
            path = _validate_managed_path(location / relative, "audit producer dependency", root=location)
            if _file_hash(path) != expected:
                raise ValueError(f"audit_refresh_required: producer dependency changed: {relative}")
        checked.append((relative, expected))
    if (inventory.get("executing_audit") != inventory["src/project/experiments/full_development_audits.py"]
            or inventory.get("executing_inner_cv") != inventory["src/project/experiments/inner_cv.py"]):
        raise ValueError("executing audit producer differs from recorded producer")
    return tuple(checked)


def _verify_bindings(root: Path, spec: FullRunSpec, report: dict) -> None:
    if type(spec) is not FullRunSpec:
        raise TypeError("exact FullRunSpec required")
    bindings = report["bindings"]
    if dict(spec.environment) != bindings["environment"] or tuple(sorted(spec.environment)) != current_environment():
        raise ValueError("development evidence environment mismatch")
    expected_all = {(s["role"], s["path"], s["sha256"]) for s in bindings["sources"]}
    if len(expected_all) != len(bindings["sources"]):
        raise ValueError("development evidence scientific/input binding mismatch")
    # The source-only campaign inspector deliberately omits the encoding
    # extension (it is authenticated separately upstream by the full snapshot
    # path before any readiness disposition is consumed). When the caller
    # declares encoding, its digest is still strictly bound below.
    supplied_roles = {s.role for s in spec.sources}
    expected = expected_all if "encoding" in supplied_roles else {
        binding for binding in expected_all if binding[0] != "encoding"}
    actual = {(s.role, s.path, s.sha256) for s in spec.sources}
    if expected != actual:
        raise ValueError("development evidence scientific/input binding mismatch")
    if type(report["development_rows"]) is not int or report["development_rows"] != spec.development_rows:
        raise ValueError("development evidence row count mismatch")
    if "encoding" in supplied_roles:
        if next(s.sha256 for s in spec.sources if s.role == "encoding") != bindings["encoding_sha256"]:
            raise ValueError("development evidence encoding mismatch")
    spec.verify_bindings(root)


def _verify_methods(spec: FullRunSpec) -> None:
    candidates = [c for c in spec.candidates if c.spec.experiment_id == TARGET_ID]
    if not candidates:
        return
    snapshot = candidates[0]
    candidate = snapshot.spec
    canonical_feature_order(candidate.feature_pool)
    if (candidate.model_family != "hist_gradient_boosting" or snapshot.arms != ("control", "challenger")
            or snapshot.study is not None or snapshot.selector_arms
            or candidate.method_params != TARGET_METHODS
            or dict(candidate.model_params) != {"early_stopping": False, "random_state": 42}
            or any(type(v) is not type({"early_stopping": False, "random_state": 42}[k])
                   for k, v in candidate.model_params)):
        raise ValueError("target encoding exact approved default parameters required")


def _positive_counts(side: dict) -> None:
    for key in ("rows", "group_count"):
        if type(side.get(key)) is not int or side[key] <= 0:
            raise ValueError("inner support row/group counts must be positive integers")
    for key in ("class_counts", "class_supporting_groups"):
        values = side.get(key)
        if not isinstance(values, list) or len(values) != 2 or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("inner train/valid both-class support incomplete")
    if sum(side["class_counts"]) != side["rows"] or any(v > side["group_count"] for v in side["class_supporting_groups"]):
        raise ValueError("inner class support cardinality inconsistent")


def _verify_plans(report: dict, manifest: pl.DataFrame) -> tuple[tuple[int, int, str], ...]:
    partitions = report["target_encoding"]
    if len(partitions) != 12 or {(p["repeat"], p["fold"]) for p in partitions} != {
            (r, f) for r in range(3) for f in range(4)}:
        raise ValueError("exact twelve outer partitions required")
    checked = []
    for part in partitions:
        repeat, fold = part["repeat"], part["fold"]
        train_groups = manifest.filter((pl.col("repeat") == repeat) & (pl.col("split") == "dev")
                                       & (pl.col("fold") != fold))["building_id"].to_list()
        if (type(repeat) is not int or type(fold) is not int or part["split_seed"] != 42 + repeat
                or part["outer_validation_groups_used"] is not False
                or part["training_group_count"] != len(train_groups)
                or part["training_groups_sha256"] != _digest(train_groups)):
            raise ValueError("inner support not bound to authoritative outer-training groups")
        _sha(part["training_labels_sha256"])
        plans = [p for p in part["plans"] if p["n_splits"] == 3]
        if len(plans) != 1:
            raise ValueError("exact selected K3 inner plan required")
        plan = plans[0]
        if plan["status"] != "supported" or type(plan["seed"]) is not int or plan["seed"] != 42 or len(plan["folds"]) != 3:
            raise ValueError("selected K3 seed42 plan support incomplete")
        _sha(plan["membership_sha256"])
        covered = covered_groups = 0
        for sides in plan["folds"]:
            if set(sides) != {"train", "validation"}:
                raise ValueError("both inner sides required")
            for side in sides.values():
                _positive_counts(side)
            train, valid = sides["train"], sides["validation"]
            if (train["rows"] + valid["rows"] != part["training_rows"]
                    or train["group_count"] + valid["group_count"] != len(train_groups)):
                raise ValueError("inner partition coverage mismatch")
            covered += valid["rows"]
            covered_groups += valid["group_count"]
        if covered != part["training_rows"] or covered_groups != len(train_groups):
            raise ValueError("inner validation coverage mismatch")
        checked.append((repeat, fold, plan["membership_sha256"]))
    return tuple(checked)


def _sentinel_disposition(report: dict) -> str:
    sentinel = report["sentinel"]
    if (sentinel["column"] != "meter_reading" or sentinel["approved_invalid_rule"] != "meter_reading < 0"
            or sentinel["enumeration_complete"] is not True
            or sentinel["replacement_value"] is not None):
        raise ValueError("approved negative-only sentinel evidence is incomplete")
    counts = ("negative_count", "finite_negative_count", "unique_negative_count", "null_count", "nan_count", "nonfinite_count")
    if any(type(sentinel[k]) is not int or sentinel[k] < 0 for k in counts):
        raise ValueError("exact sentinel counts required")
    if sentinel["negative_count"] == sentinel["unique_negative_count"] == 0 and sentinel["invalid_values"] == []:
        if sentinel["finite_negative_count"] != 0 or sentinel["nonfinite_count"] != 0:
            raise ValueError("empty sentinel evidence has inconsistent counts")
        return "not_applicable"
    return "replacement_approval_pending"


def inspect_development_readiness(root: str | Path, spec: FullRunSpec) -> DevelopmentReadiness:
    """Authenticate evidence and sources; no source-config or launch promotion."""
    root = _validate_managed_path(root, "readiness source root", directory=True)
    report = _read_evidence(root)
    producer = _verify_producer(root, report)
    _verify_bindings(root, spec, report)
    _verify_methods(spec)
    split = next(s for s in spec.sources if s.role == "split")
    plans = _verify_plans(report, _assignments(split.verify(root)))
    sentinel = _sentinel_disposition(report)
    _verify_bindings(root, spec, report)
    if _verify_producer(root, report) != producer or _read_evidence(root) != report:
        raise ValueError("development evidence/producer mutated during admission")
    return DevelopmentReadiness(EVIDENCE_SHA256, report["development_identity_sha256"], plans, producer, sentinel)


def verify_target_encoding_partition(root: str | Path, spec: FullRunSpec, partition: TrainingPartition) -> str:
    """Pre-debit consumer hook: recompute actual K3 folds and exact label binding.

    A future genuine runtime integration must call this *before* debit/fit and
    also compare fitted GroupCrossFittedTargetEncoder membership evidence. This
    function alone is not runtime proof or permission to bypass ordinary gates.
    """
    evidence = inspect_development_readiness(root, spec)
    report = _read_evidence(Path(root))
    part = next(p for p in report["target_encoding"] if (p["repeat"], p["fold"]) == (partition.repeat, partition.fold))
    if len(partition.y_train) != part["training_rows"] or _digest(partition.y_train.tolist()) != part["training_labels_sha256"]:
        raise ValueError("outer-training labels/cardinality differ from audit")
    folds = build_inner_folds(partition.y_train, partition.groups_train, 3, 42)
    actual = inner_membership_digest(folds, partition.groups_train)
    if actual != next(digest for r, f, digest in evidence.inner_membership_sha256 if (r, f) == (partition.repeat, partition.fold)):
        raise ValueError("outer-training row/group inner membership differs from audit")
    return actual
