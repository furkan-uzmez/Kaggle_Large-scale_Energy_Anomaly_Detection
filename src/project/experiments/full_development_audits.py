"""Read-only, authenticated development evidence; never campaign admission.

Arrow's scanner applies authoritative building membership before Python sees
labels/features. Only labels, keys and groups are retained for actual inner CV;
meter values are reduced in bounded batches. No estimator is constructed/fitted.
The inner splitter still has population-scale memory needs, explicitly bounded
by the caller's row ceiling rather than disguised as a constant-memory audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.dataset as ds

from project.experiments.candidate_contracts import (
    BlockedExperimentError,
    _validate_managed_path,
)
from project.experiments.full_campaign_preflight import (
    MAX_BATCH_ROWS,
    _assignments,
    _verify_encoding_source,
    current_environment,
)
from project.experiments.full_candidate_io import DATA_PATH, SPLIT_PATH
from project.experiments.full_input_schema import (
    canonical_feature_order,
    validate_input_dtypes,
)
from project.experiments.full_run_contracts import FullRunSpec, _file_hash
from project.experiments.inner_cv import build_inner_folds, inner_membership_digest


@dataclass(frozen=True)
class AuditPlan:
    """Enumerate declared inner plans, not score-based search or seed retries."""
    inner_splits: tuple[int, ...] = (2, 3, 4, 5)
    inner_seed: int = 42
    max_development_rows: int = 1_399_631
    max_invalid_values: int = 256

    def __post_init__(self):
        if (not isinstance(self.inner_splits, tuple) or not self.inner_splits
                or len(set(self.inner_splits)) != len(self.inner_splits)
                or any(type(k) is not int or not 2 <= k <= 5 for k in self.inner_splits)):
            raise ValueError("inner splits must be distinct declared integers 2..5")
        if type(self.inner_seed) is not int or self.inner_seed != 42:
            raise ValueError("audit inner seed must remain the frozen estimator seed 42")
        if any(type(v) is not int or v < 1 for v in
               (self.max_development_rows, self.max_invalid_values)):
            raise ValueError("positive explicit audit bounds required")


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _authenticate(root: Path, spec: FullRunSpec) -> dict:
    if type(spec) is not FullRunSpec:
        raise TypeError("exact FullRunSpec required")
    sources = {source.role: source for source in spec.sources}
    for role, expected in (("data", DATA_PATH), ("split", SPLIT_PATH)):
        if sum(s.role == role for s in spec.sources) != 1 or sources[role].path != expected:
            raise ValueError(f"exact confined {role} source required")
    if spec.environment != current_environment():
        raise ValueError("exact environment mismatch")
    spec.verify_bindings(root)
    encoding = _verify_encoding_source(root, spec)
    for candidate in spec.candidates:
        canonical_feature_order(candidate.spec.feature_pool)
    paths = sorted({str(p.relative_to(root)) for directory in ("src/project", "scripts")
                    for p in (root / directory).rglob("*.py")})
    implementation = {}
    for relative in paths:
        path = _validate_managed_path(root / relative, "audit implementation", root=root)
        implementation[relative] = _file_hash(path)
    # Also bind the executing audit/splitter implementation, not just a caller's
    # copied source tree. Changes during a scan invalidate the report.
    implementation["executing_audit"] = _file_hash(Path(__file__))
    from project.experiments import inner_cv
    implementation["executing_inner_cv"] = _file_hash(Path(inner_cv.__file__))
    return {"spec_sha256": spec.binding_sha256,
            "sources": [{"role": s.role, "path": s.path, "sha256": s.sha256} for s in spec.sources],
            "encoding_sha256": encoding.sha256, "environment": dict(spec.environment),
            "implementation": implementation}


def _scan_development(data_path: Path, membership: pl.DataFrame, plan: AuditPlan):
    buildings = membership["building_id"].to_list()
    scanner = ds.dataset(data_path, format="parquet").scanner(
        columns=["building_id", "timestamp", "anomaly", "meter_reading", "year_built"],
        filter=ds.field("building_id").isin(buildings), batch_size=MAX_BATCH_ROWS,
        batch_readahead=1, fragment_readahead=1, use_threads=False,
    )
    frames = []
    count = nulls = nans = nonfinite = negatives = finite_negatives = 0
    invalid: set[float] = set()
    overflow = False
    low = 0.0
    for batch in scanner.to_batches():
        frame = pl.from_arrow(batch)
        if not set(frame["building_id"].to_list()) <= set(buildings):
            raise ValueError("scanner leaked non-development membership")
        count += frame.height
        if count > plan.max_development_rows:
            raise ValueError("development row memory bound exceeded")
        if frame["anomaly"].null_count() or frame.filter(~pl.col("anomaly").is_in([0, 1])).height:
            raise ValueError("development target must be binary 0/1")
        if frame["timestamp"].null_count():
            raise ValueError("development timestamp must be nonnull")
        year = pl.col("year_built")
        if frame.filter(year.is_not_null() & (
                ~year.is_finite() | (year < 0) | (year > 255) | (year != year.floor()))).height:
            raise ValueError("development year_built violates authoritative offset encoding")
        for value in frame["meter_reading"].to_list():
            if value is None:
                nulls += 1
            elif math.isnan(value):
                nans += 1
            elif not math.isfinite(value):
                nonfinite += 1
                negatives += int(value < 0)
            elif value < 0:
                negatives += 1
                finite_negatives += 1
                low = min(low, value)
                if value not in invalid:
                    if len(invalid) < plan.max_invalid_values:
                        invalid.add(value)
                    else:
                        overflow = True
        frames.append(frame.select("building_id", "timestamp", "anomaly"))
    if not frames:
        raise ValueError("empty development population")
    rows = pl.concat(frames).sort("building_id", "timestamp").with_row_index("row_index")
    if rows.select(pl.struct("building_id", "timestamp").n_unique()).item() != count:
        raise ValueError("duplicate development natural key")
    if set(rows["building_id"].unique().to_list()) != set(buildings):
        raise ValueError("declared development membership missing rows")
    candidate = math.nextafter(low, -math.inf)
    sentinel = {
        "column": "meter_reading", "approved_invalid_rule": "meter_reading < 0",
        "negative_count": negatives, "finite_negative_count": finite_negatives,
        "null_count": nulls, "nan_count": nans,
        "nonfinite_count": nonfinite,
        "unique_negative_count": None if overflow or negatives != finite_negatives else len(invalid),
        "unique_negative_count_lower_bound": len(invalid) + int(overflow) + int(negatives != finite_negatives),
        "invalid_values": None if overflow or negatives != finite_negatives else sorted(invalid),
        "enumeration_complete": not overflow and negatives == finite_negatives,
        "enumeration_limit": plan.max_invalid_values,
        "proposed_replacement_value": candidate if math.isfinite(candidate) else None,
        "replacement_value": None, "requires_parent_replacement_approval": True,
        "status": "enumeration_overrun" if overflow else (
            "nonfinite_readings_unresolved" if nonfinite else (
                "replacement_approval_pending" if invalid else "no_invalid_values_observed")),
    }
    return rows, sentinel


def _side_support(labels: pd.Series, groups: pd.Series, positions) -> dict:
    side_labels, side_groups = labels.iloc[positions], groups.iloc[positions]
    return {"rows": len(positions), "group_count": side_groups.nunique(),
            "class_counts": [int((side_labels == label).sum()) for label in (0, 1)],
            "class_supporting_groups": [side_groups[side_labels == label].nunique() for label in (0, 1)]}


def _audit_inner(rows: pl.DataFrame, manifest: pl.DataFrame, plan: AuditPlan) -> list[dict]:
    evidence = []
    for repeat in range(3):
        assignments = manifest.filter((pl.col("repeat") == repeat) & (pl.col("split") == "dev"))
        for fold in range(4):
            train_groups = assignments.filter(pl.col("fold") != fold)["building_id"].to_list()
            training = rows.filter(pl.col("building_id").is_in(train_groups))
            validation = rows.filter(~pl.col("building_id").is_in(train_groups))
            if any(side["anomaly"].unique().sort().to_list() != [0, 1] for side in (training, validation)):
                raise ValueError("outer partition class support missing")
            # Same sorted global identity used by full_candidate_io, so this
            # membership fingerprint is directly comparable at fit time.
            index = pd.Index(training["row_index"].to_numpy(), name="row_index")
            labels = pd.Series(training["anomaly"].to_numpy(), index=index)
            groups = pd.Series(training["building_id"].to_list(), index=index)
            partition = {"repeat": repeat, "fold": fold, "split_seed": 42 + repeat,
                         "training_rows": len(training), "training_group_count": len(train_groups),
                         "training_groups_sha256": _digest(train_groups),
                         "training_labels_sha256": _digest(labels.tolist()),
                         "outer_validation_groups_used": False, "plans": []}
            for k in plan.inner_splits:
                item = {"n_splits": k, "seed": plan.inner_seed}
                try:
                    folds = build_inner_folds(labels, groups, k, plan.inner_seed)
                    item.update(status="supported", membership_sha256=inner_membership_digest(folds, groups),
                                folds=[{"train": _side_support(labels, groups, tr),
                                        "validation": _side_support(labels, groups, va)} for tr, va in folds])
                except BlockedExperimentError as error:
                    item.update(status="blocked", reason=str(error))
                partition["plans"].append(item)
            evidence.append(partition)
    return evidence


def _development_identity_digest(rows: pl.DataFrame) -> str:
    """Stream full-I/O-compatible sorted natural key/label identity."""
    digest = hashlib.sha256()
    identity = rows.select("row_index", "building_id",
                           pl.col("timestamp").cast(pl.Int64).alias("timestamp_ns"), "anomaly")
    for row_index, building, timestamp, target in identity.iter_rows():
        building_bytes = str(building).encode("utf-8")
        digest.update(struct.pack("<q", row_index))
        digest.update(struct.pack("<q", len(building_bytes)))
        digest.update(building_bytes)
        digest.update(struct.pack("<q", timestamp))
        digest.update(struct.pack("<q", target))
    return digest.hexdigest()


def audit_full_development(root: str | Path, spec: FullRunSpec, plan: AuditPlan | None = None) -> dict:
    """Authenticate frozen sources, produce actual dev-only statistics, recheck.

    This neither resolves a replacement nor promotes a recipe. The parent must
    bind the published report hash and exact chosen inner k/seed to readiness.
    """
    plan = AuditPlan() if plan is None else plan
    root = _validate_managed_path(root, "audit source root", directory=True)
    before = _authenticate(root, spec)
    if spec.development_rows > plan.max_development_rows:
        raise ValueError("declared development rows exceed explicit audit memory bound")
    manifest = _assignments(root / SPLIT_PATH)
    membership = manifest.filter((pl.col("repeat") == 0) & (pl.col("split") == "dev"))
    schema = pl.read_parquet_schema(root / DATA_PATH)
    validate_input_dtypes(schema)
    if schema["building_id"] != manifest.schema["building_id"]:
        raise ValueError("split/data building dtype mismatch")
    timestamp = schema.get("timestamp")
    if not isinstance(timestamp, pl.Datetime) or timestamp.time_unit not in ("ns", "us"):
        raise ValueError("lossless ns/us timestamp required")
    if "anomaly" not in schema or not schema["anomaly"].is_integer():
        raise ValueError("integer development target required")
    rows, sentinel = _scan_development(root / DATA_PATH, membership, plan)
    if len(rows) != spec.development_rows:
        raise ValueError("authenticated development cardinality mismatch")
    report = {"schema_version": 1, "audit_kind": "full_development_only_no_fit",
              "execution_authorized": False, "bindings": before,
              "development_rows": len(rows), "timestamp_unit": timestamp.time_unit,
              "development_identity_sha256": _development_identity_digest(rows),
              "timestamp_time_zone": timestamp.time_zone,
              "vault_content_processed": False, "test_content_processed": False,
              "model_fits": 0, "sentinel": sentinel,
              "target_encoding": _audit_inner(rows, manifest, plan)}
    if _authenticate(root, spec) != before:
        raise ValueError("source or environment changed during development audit")
    return report


def write_audit_evidence(destination: str | Path, report: dict) -> str:
    """Publish strict JSON exclusively; existing evidence is never overwritten."""
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    from project.features.value_recipes import MAX_EVIDENCE_BYTES
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ValueError("audit exceeds SentinelRule evidence byte limit")
    path = _validate_managed_path(destination, "audit evidence", directory=False)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()
