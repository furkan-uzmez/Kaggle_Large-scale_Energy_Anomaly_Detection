"""Disk-backed complete-OOF coverage validation and exact repeated metrics.

The diagnostic validator in ``candidate_artifacts`` collects every OOF row into
Python sets/maps, which cannot scale to a complete development campaign
(``3 * n_dev`` records per arm, currently 4,198,893). This module keeps the same
canonical schema and manifest contract but streams expected and observed rows
into indexed SQLite tables, checks complete coverage with anti-joins, and only
then computes exact metrics from score-grouped counts.

It never derives expected membership from predictions, never collects the whole
OOF frame in memory, and never pools repeated rows: candidate summaries are the
equal mean of per-repeat metrics. Margin candidates report ranking metrics only;
probability guardrails stay null with an explicit reason. Validation never grants
Notebook03 eligibility by itself; only a later supervisor/parent proof may.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from project.models import candidate_artifacts as _artifacts

MAX_BATCH_ROWS = _artifacts.MAX_BATCH_ROWS
ROW_COLUMNS = _artifacts.ROW_COLUMNS
CANDIDATE_COLUMNS = _artifacts.CANDIDATE_COLUMNS
RUN_COLUMNS = _artifacts.RUN_COLUMNS
PROBABILITY_KINDS = ("positive_probability",)
VALIDATION_SCHEMA_VERSION = 1


class CoverageError(ValueError):
    """Raised when observed OOF evidence does not fully cover expected rows."""


@dataclass(frozen=True)
class _KeyPolicy:
    building_dtype: pl.DataType
    timestamp_dtype: pl.Datetime


def validate_full_candidate_oof(
    oof_path: str | Path,
    expected_batches: Iterable[Any],
    manifest: Mapping[str, Any],
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a complete full-run OOF artifact and compute exact metrics.

    ``expected_batches`` is an iterable of Polars/Arrow frames as produced by
    ``full_candidate_io.iter_full_expected_batches`` and must independently
    describe every development row/repeat. ``oof_path`` is a candidate schema v1
    Parquet file. Coverage failures raise before any metric is computed. The
    returned payload is JSON-serialisable and carries a self-hash; it is a
    validation proof, not a launch authorization or eligibility grant.
    """
    declaration = _artifacts._manifest(dict(manifest))
    path = _existing_file(oof_path)
    connection = _connect(database_path)
    try:
        _create_tables(connection)
        expected_count, key_policy = _ingest_expected(connection, expected_batches, declaration)
        _ingest_observed(connection, path, declaration, key_policy)
        _verify_coverage(connection, declaration, expected_count)
        return _build_payload(connection, declaration, expected_count)
    finally:
        connection.close()


def _connect(database_path: str | Path | None) -> sqlite3.Connection:
    if database_path is None:
        # Empty filename creates a temporary on-disk database, not an all-OOF
        # in-memory key set.
        return sqlite3.connect("")
    database = Path(database_path)
    if not database.is_absolute() or database.is_symlink():
        raise ValueError("validation database path must be absolute and symlink-free")
    return sqlite3.connect(database)


def _existing_file(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("OOF artifact must be an existing absolute symlink-free file")
    return path


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        "CREATE TABLE expected ("
        "row_index INTEGER NOT NULL, repeat INTEGER NOT NULL, fold INTEGER NOT NULL,"
        "building_id TEXT NOT NULL, timestamp_ns INTEGER NOT NULL, target INTEGER NOT NULL,"
        "split_seed INTEGER NOT NULL, PRIMARY KEY(row_index, repeat),"
        " UNIQUE(repeat, building_id, timestamp_ns))"
    )
    connection.execute(
        "CREATE TABLE observed ("
        "candidate_id TEXT NOT NULL, row_index INTEGER NOT NULL, repeat INTEGER NOT NULL,"
        "fold INTEGER NOT NULL, building_id TEXT NOT NULL, timestamp_ns INTEGER NOT NULL,"
        "target INTEGER NOT NULL, split_seed INTEGER NOT NULL, prediction REAL NOT NULL,"
        "score_kind TEXT NOT NULL, model_fit_config_sha256 TEXT NOT NULL,"
        "recipe_id TEXT NOT NULL, model_family TEXT NOT NULL, model_config_id TEXT NOT NULL,"
        "experiment_id TEXT NOT NULL, arm_id TEXT NOT NULL, positive_label INTEGER NOT NULL,"
        "split_sha256 TEXT NOT NULL, data_sha256 TEXT NOT NULL, config_sha256 TEXT NOT NULL,"
        "run_id TEXT NOT NULL, execution_mode TEXT NOT NULL, non_comparative INTEGER NOT NULL,"
        "n03_eligible INTEGER NOT NULL,"
        "PRIMARY KEY(candidate_id, row_index, repeat))"
    )
    connection.execute(
        "CREATE INDEX observed_arm ON observed(candidate_id, repeat)"
    )
    connection.execute("CREATE TABLE vault_rows (row_index INTEGER PRIMARY KEY)")
    connection.execute("CREATE TABLE vault_buildings (building_id TEXT PRIMARY KEY)")


def _key_policy(batch: Any) -> _KeyPolicy:
    if isinstance(batch, (pa.Table, pa.RecordBatch)):
        batch = pl.from_arrow(batch)
    if isinstance(batch, pl.DataFrame):
        if not {"building_id", "timestamp"} <= set(batch.columns):
            raise ValueError("authoritative keys are missing")
        building, timestamp = batch.schema["building_id"], batch.schema["timestamp"]
    elif isinstance(batch, Mapping):
        # Legacy row mappings declare physical canonical ns ticks. They cannot
        # represent a source timezone/unit; use Arrow/Polars for source schemas.
        building = pl.Int64 if type(batch.get("building_id")) is int else pl.String
        timestamp = pl.Datetime("ns")
    else:
        raise TypeError("expected batches must be Polars/Arrow frames or row mappings")
    if building not in (pl.String, pl.Int64):
        raise ValueError("authoritative building dtype must be opaque string/int64")
    if not isinstance(timestamp, pl.Datetime) or timestamp.time_unit not in ("ns", "us"):
        raise ValueError("authoritative timestamp policy requires lossless ns/us")
    return _KeyPolicy(building, timestamp)


def _canonical_ticks(value: object, unit: str) -> int:
    if type(value) is not int:
        raise ValueError("timestamp must be nonnull physical integer ticks")
    ticks = value * (1000 if unit == "us" else 1)
    if not -(2**63) <= ticks < 2**63:
        raise ValueError("canonical ns timestamp conversion would overflow")
    return ticks


def _iter_rows(batch: Any) -> Iterable[dict[str, Any]]:
    if isinstance(batch, (pa.Table, pa.RecordBatch)):
        batch = pl.from_arrow(batch)
    if isinstance(batch, pl.DataFrame):
        if "timestamp" in batch.columns:
            batch = batch.with_columns(pl.col("timestamp").cast(pl.Int64))
        return batch.iter_rows(named=True)
    if isinstance(batch, Mapping):
        return (dict(batch),)
    raise TypeError("expected batches must be Polars/Arrow frames or row mappings")


def _ingest_expected(connection: sqlite3.Connection, expected_batches: Iterable[Any], declaration: dict) -> tuple[int, _KeyPolicy]:
    protocol = declaration["protocol"]
    repeats = protocol["repeat_ids"]
    folds = set(protocol["fold_ids"])
    seeds = {int(k): int(v) for k, v in protocol["split_seeds"].items()}
    connection.executemany("INSERT INTO vault_rows VALUES (?)", ((value,) for value in declaration["vault_row_indices"]))
    connection.executemany("INSERT INTO vault_buildings VALUES (?)", ((str(value),) for value in declaration["vault_building_ids"]))
    policy = None
    count = 0
    for batch in expected_batches:
        current_policy = _key_policy(batch)
        if policy is not None and current_policy != policy:
            raise ValueError("authoritative key schema policy differs across batches")
        policy = current_policy
        rows = []
        for row in _iter_rows(batch):
            missing = [name for name in ("row_index", "repeat", "fold", "building_id",
                                         "timestamp", "target", "split_seed") if name not in row]
            if missing:
                raise ValueError(f"expected row missing columns: {missing}")
            repeat = row["repeat"]
            if any(type(row[name]) is not int for name in ("row_index", "repeat", "fold", "target", "split_seed")):
                raise ValueError("expected identity, protocol and targets require exact integers")
            if row["row_index"] < 0 or repeat not in repeats or row["fold"] not in folds:
                raise ValueError("expected row repeat/fold is outside the declared protocol")
            if row["target"] not in (0, 1):
                raise ValueError("expected target must be integer binary 0 or 1")
            if row["split_seed"] != seeds[repeat]:
                raise ValueError("expected split seed differs from protocol")
            for name in RUN_COLUMNS[:4]:
                if row.get(name) != declaration[name]:
                    raise ValueError(f"expected lineage differs from manifest: {name}")
            building = row["building_id"]
            if (policy.building_dtype == pl.Int64 and type(building) is not int
                    or policy.building_dtype == pl.String and (not isinstance(building, str) or not building)):
                raise ValueError("building identity differs from declared dtype policy")
            rows.append((row["row_index"], repeat, row["fold"], building,
                          _canonical_ticks(row["timestamp"], policy.timestamp_dtype.time_unit),
                          row["target"], row["split_seed"]))
            if len(rows) == MAX_BATCH_ROWS:
                _insert_expected(connection, rows)
                count += len(rows)
                rows.clear()
        if rows:
            _insert_expected(connection, rows)
            count += len(rows)
    if count == 0:
        raise ValueError("authoritative expected rows are empty")
    if connection.execute(
        "SELECT 1 FROM expected GROUP BY building_id, repeat HAVING COUNT(DISTINCT fold) > 1 LIMIT 1"
    ).fetchone():
        raise ValueError("authoritative building spans multiple folds within a repeat")
    if connection.execute(
        "SELECT 1 FROM expected e WHERE EXISTS (SELECT 1 FROM vault_rows v WHERE v.row_index = e.row_index)"
        " OR EXISTS (SELECT 1 FROM vault_buildings v WHERE v.building_id = e.building_id) LIMIT 1"
    ).fetchone():
        raise ValueError("vault membership is forbidden in development evidence")
    _verify_repeat_population(connection, repeats)
    return count, policy


def _insert_expected(connection, rows) -> None:
    try:
        connection.executemany("INSERT INTO expected VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    except sqlite3.IntegrityError as exc:
        raise ValueError("duplicate authoritative row/repeat or natural key") from exc


def _verify_repeat_population(connection, repeats) -> None:
    reference = repeats[0]
    for repeat in repeats[1:]:
        for left, right in ((repeat, reference), (reference, repeat)):
            if connection.execute(
                "SELECT 1 FROM expected e LEFT JOIN expected r ON r.row_index = e.row_index AND r.repeat = ?"
                " WHERE e.repeat = ? AND (r.row_index IS NULL OR e.building_id <> r.building_id"
                " OR e.timestamp_ns <> r.timestamp_ns OR e.target <> r.target) LIMIT 1", (right, left),
            ).fetchone():
                raise ValueError("authoritative development population differs across repeats")


def _ingest_observed(connection: sqlite3.Connection, path: Path, declaration: dict, policy: _KeyPolicy) -> None:
    candidates = declaration["candidates"]
    parquet = pq.ParquetFile(path)
    try:
        actual_schema = parquet.schema_arrow
        building = pa.int64() if policy.building_dtype == pl.Int64 else pa.string()
        timestamp = actual_schema.field("timestamp").type
        if (not pa.types.is_timestamp(timestamp) or timestamp.unit not in ("ns", "us")
                or timestamp.tz != policy.timestamp_dtype.time_zone):
            raise ValueError("OOF timestamp timezone/unit differs from authoritative key policy")
        schema = _artifacts.candidate_oof_schema(building, timestamp)
        if not parquet.schema_arrow.equals(schema, check_metadata=True):
            raise ValueError("OOF artifact schema must exactly match candidate schema v1")
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
            frame = pl.from_arrow(batch)
            _insert_observed_batch(connection, frame, candidates, declaration)
    finally:
        del parquet


def _insert_observed_batch(connection, frame: pl.DataFrame, candidates: dict, declaration: dict) -> None:
    unit = frame.schema["timestamp"].time_unit
    frame = frame.with_columns(pl.col("timestamp").cast(pl.Int64))
    rows = []
    for row in frame.iter_rows(named=True):
        candidate = row["candidate_id"]
        if candidate not in candidates:
            raise ValueError("OOF artifact contains an unexpected candidate arm")
        meta = candidates[candidate]
        _validate_observed_row(row, meta, declaration)
        rows.append((
            candidate, row["row_index"], row["repeat"], row["fold"], str(row["building_id"]),
            _canonical_ticks(row["timestamp"], unit), row["target"], row["split_seed"], float(row["prediction"]),
            row["score_kind"], row["model_fit_config_sha256"], row["recipe_id"],
            row["model_family"], row["model_config_id"], row["experiment_id"], row["arm_id"],
            row["positive_label"], row["split_sha256"], row["data_sha256"], row["config_sha256"],
            row["run_id"], row["execution_mode"], int(row["non_comparative"]), int(row["n03_eligible"]),
        ))
    if not rows:
        return
    try:
        connection.executemany(
            "INSERT INTO observed VALUES (" + ", ".join("?" * 24) + ")", rows
        )
    except sqlite3.IntegrityError as exc:
        raise CoverageError("duplicate candidate,row_index,repeat key across streamed batches") from exc


def _validate_observed_row(row: dict, meta: dict, declaration: dict) -> None:
    for name in (*CANDIDATE_COLUMNS, *RUN_COLUMNS):
        value = row[name]
        if name in CANDIDATE_COLUMNS:
            if name == "positive_label":
                if value != 1:
                    raise ValueError("candidate positive_label must be integer 1")
            elif value != meta[name]:
                raise ValueError(f"candidate metadata mismatch: {name}")
        elif name in ("non_comparative", "n03_eligible"):
            if bool(value) != declaration[name]:
                raise ValueError(f"run lineage mismatch: {name}")
        elif value != declaration[name]:
            raise ValueError(f"run lineage mismatch: {name}")
    if row["score_kind"] not in ("positive_probability", "decision_margin"):
        raise ValueError("unsupported score kind")
    prediction = float(row["prediction"])
    if not math.isfinite(prediction):
        raise ValueError("prediction must be finite")
    if row["score_kind"] == "positive_probability" and not 0.0 <= prediction <= 1.0:
        raise ValueError("probability prediction must be within [0, 1]")
    if row["target"] not in (0, 1):
        raise ValueError("target must be integer binary 0 or 1")
    if declaration["execution_mode"] == "full" and not (0 <= row["repeat"] <= 2 and 0 <= row["fold"] <= 3):
        raise ValueError("observed row repeat/fold is outside the full protocol")


def _verify_coverage(connection: sqlite3.Connection, declaration: dict, expected_count: int) -> None:
    repeats = set(declaration["protocol"]["repeat_ids"])
    folds = set(declaration["protocol"]["fold_ids"])
    for candidate, meta in declaration["candidates"].items():
        _verify_arm(connection, candidate, meta, repeats, folds, expected_count, declaration)
        _verify_fold_fit_hashes(connection, candidate, meta)
    # Every declared arm must exist and be complete.
    observed_candidates = {row[0] for row in connection.execute("SELECT DISTINCT candidate_id FROM observed")}
    if observed_candidates != set(declaration["candidates"]):
        raise ValueError("missing or unexpected candidate arms")
    if not repeats:
        raise ValueError("expected protocol declares no repeats")


def _verify_arm(connection, candidate: str, meta: dict, repeats: set, folds: set,
                expected_count: int, declaration: dict) -> None:
    for repeat in sorted(repeats):
        mismatch = connection.execute(
            "SELECT COUNT(*) FROM observed o JOIN expected e"
            " ON o.row_index = e.row_index AND o.repeat = e.repeat"
            " WHERE o.candidate_id = ? AND o.repeat = ? AND ("
            " o.target <> e.target OR o.fold <> e.fold OR o.split_seed <> e.split_seed"
            " OR o.building_id <> e.building_id OR o.timestamp_ns <> e.timestamp_ns)",
            (candidate, repeat),
        ).fetchone()[0]
        if mismatch:
            raise ValueError("OOF target/key/fold/seed alignment differs from authoritative rows")
        missing = connection.execute(
            "SELECT COUNT(*) FROM expected e WHERE e.repeat = ? AND NOT EXISTS ("
            " SELECT 1 FROM observed o WHERE o.candidate_id = ? AND o.row_index = e.row_index"
            " AND o.repeat = e.repeat)",
            (repeat, candidate),
        ).fetchone()[0]
        if missing:
            raise ValueError("candidate has incomplete expected development coverage")
        extra = connection.execute(
            "SELECT COUNT(*) FROM observed o WHERE o.candidate_id = ? AND o.repeat = ? AND NOT EXISTS ("
            " SELECT 1 FROM expected e WHERE e.row_index = o.row_index AND e.repeat = o.repeat)",
            (candidate, repeat),
        ).fetchone()[0]
        if extra:
            raise ValueError("candidate OOF contains rows absent from authoritative expected")
        observed_folds = {row[0] for row in connection.execute(
            "SELECT DISTINCT fold FROM observed WHERE candidate_id = ? AND repeat = ?", (candidate, repeat)
        )}
        if observed_folds != folds:
            raise ValueError("candidate OOF does not cover every declared fold")
        observed_count = connection.execute(
            "SELECT COUNT(*) FROM observed WHERE candidate_id = ? AND repeat = ?", (candidate, repeat)
        ).fetchone()[0]
        if observed_count != expected_count // len(repeats):
            raise ValueError("candidate repeat record count differs from authoritative count")
    if connection.execute(
        "SELECT 1 FROM observed o JOIN vault_rows v USING (row_index) WHERE o.candidate_id = ? LIMIT 1",
        (candidate,),
    ).fetchone():
        raise ValueError("candidate OOF retained a vault row")


def _verify_fold_fit_hashes(connection, candidate: str, meta: dict) -> None:
    if "fold_fit_config_sha256" in meta:
        declared = meta["fold_fit_config_sha256"]
        for repeat, fold, actual in connection.execute(
            "SELECT repeat, fold, model_fit_config_sha256 FROM observed WHERE candidate_id = ?"
            " GROUP BY repeat, fold, model_fit_config_sha256", (candidate,),
        ):
            if declared.get(f"{repeat}:{fold}") != actual:
                raise ValueError("candidate fit hash differs from declared fold configuration")
    else:
        expected_hash = meta["model_fit_config_sha256"]
        if connection.execute(
            "SELECT 1 FROM observed WHERE candidate_id = ? AND model_fit_config_sha256 <> ? LIMIT 1",
            (candidate, expected_hash),
        ).fetchone():
            raise ValueError("candidate fit hash differs from declared fixed configuration")


def _build_payload(connection, declaration: dict, expected_count: int) -> dict[str, Any]:
    names = ("roc_auc", "average_precision", "log_loss", "brier_score")
    repeat_metrics, fold_metrics, candidate_metrics = [], [], {}
    for candidate, meta in declaration["candidates"].items():
        score_kind = meta["score_kind"]
        repeats = []
        for repeat in declaration["protocol"]["repeat_ids"]:
            metrics = _repeat_metrics(connection, candidate, repeat, score_kind)
            repeats.append(metrics)
            repeat_metrics.append({"candidate_id": candidate, "repeat": repeat, **metrics})
            for fold in declaration["protocol"]["fold_ids"]:
                fold_metrics.append({
                    "candidate_id": candidate, "repeat": repeat, "fold": fold,
                    **_slice_metrics(connection, candidate, repeat, fold, score_kind),
                })
        candidate_metrics[candidate] = {
            "mean_repeat_metrics": {
                name: None if repeats[0][name] is None
                else float(np.mean([row[name] for row in repeats])) for name in names
            },
            "guardrail_status": repeats[0]["guardrail_status"],
            "probability_pool_eligible": False,
        }
    payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "validated",
        "oof_record_count": connection.execute("SELECT COUNT(*) FROM observed").fetchone()[0],
        "expected_row_repeat_count": expected_count,
        "summary_basis": "equal_mean_of_repeats_not_pooled_rows",
        "data_sha256": declaration["data_sha256"],
        "split_sha256": declaration["split_sha256"],
        "config_sha256": declaration["config_sha256"],
        "run_id": declaration["run_id"],
        "repeat_metrics": repeat_metrics,
        "fold_metrics": fold_metrics,
        "candidate_metrics": candidate_metrics,
    }
    payload["validation_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload


def _grouped_scores(connection, candidate: str, repeat: int, fold: int | None, order: str):
    where = "candidate_id = ? AND repeat = ?"
    params: list[Any] = [candidate, repeat]
    if fold is not None:
        where += " AND fold = ?"
        params.append(fold)
    return connection.execute(
        "SELECT prediction, SUM(target), COUNT(*) - SUM(target) FROM observed"
        f" WHERE {where} GROUP BY prediction ORDER BY prediction {order}", params
    )


def _support(connection, candidate: str, repeat: int, fold: int | None) -> tuple[int, int]:
    where = "candidate_id = ? AND repeat = ?"
    params: list[Any] = [candidate, repeat]
    if fold is not None:
        where += " AND fold = ?"
        params.append(fold)
    positive, negative = connection.execute(
        f"SELECT COALESCE(SUM(target), 0), COUNT(*) - COALESCE(SUM(target), 0)"
        f" FROM observed WHERE {where}", params
    ).fetchone()
    return int(positive), int(negative)


def _repeat_metrics(connection, candidate: str, repeat: int, score_kind: str) -> dict[str, Any]:
    positive, negative = _support(connection, candidate, repeat, None)
    if positive == 0 or negative == 0:
        raise ValueError("a required outer repeat must contain both target classes")
    return _metrics_from_groups(connection, candidate, repeat, None, score_kind, positive, negative)


def _slice_metrics(connection, candidate: str, repeat: int, fold: int, score_kind: str) -> dict[str, Any]:
    positive, negative = _support(connection, candidate, repeat, fold)
    if positive == 0 or negative == 0:
        names = ("roc_auc", "average_precision", "log_loss", "brier_score")
        return {name: None for name in names} | {
            "guardrail_status": "unavailable_single_class_slice",
            "positive_support": positive, "negative_support": negative,
        }
    return _metrics_from_groups(connection, candidate, repeat, fold, score_kind, positive, negative)


def _metrics_from_groups(connection, candidate, repeat, fold, score_kind, positive, negative):
    total = positive + negative
    auc = _auc_from_groups(connection, candidate, repeat, fold, positive, negative)
    ap = _average_precision_from_groups(connection, candidate, repeat, fold, positive)
    if score_kind == "positive_probability":
        log_loss, brier = _probability_losses(connection, candidate, repeat, fold, total)
        status = "computed"
    else:
        log_loss, brier = None, None
        status = "requires_calibrated_probability"
    return {
        "roc_auc": auc,
        "average_precision": ap,
        "log_loss": log_loss,
        "brier_score": brier,
        "guardrail_status": status,
        "positive_support": positive,
        "negative_support": negative,
        "prevalence": positive / total,
    }


def _auc_from_groups(connection, candidate, repeat, fold, positive, negative) -> float:
    negative_before = 0.0
    numerator = 0.0
    for score, pos, neg in _grouped_scores(connection, candidate, repeat, fold, "ASC"):
        # Ties share average rank: strictly-lower negatives plus half the tied ones.
        numerator += pos * (negative_before + 0.5 * neg)
        negative_before += neg
    return float(numerator / (positive * negative))


def _average_precision_from_groups(connection, candidate, repeat, fold, positive) -> float:
    true_positive = 0.0
    false_positive = 0.0
    average_precision = 0.0
    for score, pos, neg in _grouped_scores(connection, candidate, repeat, fold, "DESC"):
        true_positive += pos
        false_positive += neg
        precision = true_positive / (true_positive + false_positive)
        average_precision += (pos / positive) * precision
    return float(average_precision)


def _probability_losses(connection, candidate, repeat, fold, total) -> tuple[float, float]:
    epsilon = float(np.finfo(np.float64).eps)
    log_loss_sum = 0.0
    brier_sum = 0.0
    for score, pos, neg in _grouped_scores(connection, candidate, repeat, fold, "ASC"):
        clipped = min(max(float(score), epsilon), 1.0 - epsilon)
        log_loss_sum += pos * (-math.log(clipped)) + neg * (-math.log(1.0 - clipped))
        brier_sum += pos * (score - 1.0) ** 2 + neg * score ** 2
    return float(log_loss_sum / total), float(brier_sum / total)
