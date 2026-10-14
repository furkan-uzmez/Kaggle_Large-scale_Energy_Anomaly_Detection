"""Full-run projected development I/O and a stable disk-backed row identity.

This module binds the explicit confined ``train_joined.parquet`` path, authenticates
its bytes through the frozen full spec, streams projected Parquet batches, filters
to the authoritative development building membership before retaining predictors or
labels, and persists a canonical sorted identity into SQLite. It never opens
``test_joined.parquet``, never samples rows, and never synthesizes parity targets,
placeholder hashes or default fallbacks.

Fitting materializes at most one outer-training partition; validation and expected
lineage are streamed in ``MAX_BATCH_ROWS`` batches.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from project.experiments.candidate_contracts import _validate_managed_path
from project.experiments.full_input_schema import CORE, canonical_feature_order
from project.experiments.full_run_contracts import FullRunSpec
from project.models.candidate_artifacts import MAX_BATCH_ROWS

DATA_PATH = "outputs/prepared/train_joined.parquet"
SPLIT_PATH = "outputs/splits/building_folds_repeated.parquet"
ROW_UNIT = "ns"
ROW_UNITS = ("ns", "us")
PREDICTOR_INT = "int64"
PREDICTOR_FLOAT = "float64"
PREDICTOR_STRING = "string"
_INDEX_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _IndexMeta:
    run_binding: str
    data_sha256: str
    split_sha256: str
    data_path: str
    timestamp_unit: str
    timestamp_time_zone: str | None
    building_dtype: str
    predictor_types: dict[str, str]
    dev_row_count: int
    identity_sha256: str


def build_development_index(
    root: str | Path,
    admitted_spec: FullRunSpec,
    destination: str | Path,
) -> Path:
    """Stream projected dev rows into a canonical, durable SQLite index.

    The admitted spec must be a typed ``FullRunSpec`` whose declared and actual
    data/split bytes match; missing, unknown, unassigned, duplicated, malformed or
    non-``ns``/``us`` inputs fail closed before a finished index is published.
    Every outer training/validation partition is checked before any fit can start.
    """
    if not isinstance(admitted_spec, FullRunSpec):
        raise TypeError("admitted_spec must be a typed FullRunSpec")
    source_root = _validate_managed_path(root, "source root", directory=True)
    admitted_spec.verify_bindings(source_root)
    _require_declared_path(admitted_spec, "data", DATA_PATH)
    _require_declared_path(admitted_spec, "split", SPLIT_PATH)
    data_path = _confined(source_root, DATA_PATH, "data")
    split_path = _confined(source_root, SPLIT_PATH, "split")

    schema = pl.scan_parquet(data_path).collect_schema()
    predictors = _predictors_from_schema(schema, admitted_spec)
    building_dtype = _building_dtype(schema)
    timestamp_dtype = schema["timestamp"]
    if not isinstance(timestamp_dtype, pl.Datetime) or timestamp_dtype.time_unit not in ROW_UNITS:
        raise ValueError("development timestamp must use a lossless ns or us precision unit")
    timestamp_time_zone = timestamp_dtype.time_zone

    dev_buildings, vault_buildings, assignments = _read_assignments(split_path)
    data_sha256 = _source_hash(admitted_spec, "data")
    split_sha256 = _source_hash(admitted_spec, "split")

    index_path = Path(destination).absolute()
    _validate_managed_path(index_path.parent, "index parent", directory=True)
    _validate_managed_path(index_path, "index", directory=False)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    existed = index_path.exists()
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        if existed:
            connection.execute("BEGIN")
            _require_matching_meta(connection, admitted_spec, data_sha256, split_sha256)
            connection.close()
            return index_path
        _create_schema(connection, predictors, building_dtype, _predictor_types(schema, predictors))
        meta = _IndexMeta(
            run_binding=admitted_spec.binding_sha256, data_sha256=data_sha256,
            split_sha256=split_sha256, data_path=str(data_path.resolve()),
            timestamp_unit=timestamp_dtype.time_unit, timestamp_time_zone=timestamp_time_zone,
            building_dtype=building_dtype, predictor_types=_predictor_types(schema, predictors),
            dev_row_count=0, identity_sha256="",
        )
        _store_assignments(connection, assignments)
        inserted, class_values = _stream_dev_rows(
            connection, data_path, schema, predictors, dev_buildings, vault_buildings
        )
        if class_values != {0, 1}:
            raise ValueError("development population must contain both target classes 0 and 1")
        _materialize_sorted_index(connection, predictors)
        _verify_index_cardinality(connection, admitted_spec.development_rows)
        _verify_population_and_support(connection, admitted_spec)
        identity_sha256 = _identity_digest(connection)
        meta = _with_row_count(meta, inserted)
        meta = _with_identity(meta, identity_sha256)
        _store_meta(connection, meta, predictors, admitted_spec)
        connection.execute("INSERT INTO meta VALUES ('integrity_sha256', ?)",
                           (_integrity_digest(connection),))
        admitted_spec.verify_bindings(source_root)
        connection.commit()
    except BaseException:
        connection.close()
        if not existed:
            index_path.unlink(missing_ok=True)
        raise
    connection.close()
    return index_path


def load_full_training_partition(
    index_path: str | Path,
    admitted_spec: FullRunSpec,
    repeat: int,
    fold: int,
):
    """Materialize exactly one outer-training partition from the disk index."""
    from project.experiments.candidate_contracts import TrainingPartition

    _require_repeat_fold(repeat, fold)
    connection, meta = _open_bound_index(index_path, admitted_spec)
    try:
        rows = connection.execute(
            _projection_sql(meta) + " WHERE a.repeat = ? AND a.fold <> ? ORDER BY d.row_index",
            (repeat, fold),
        ).fetchall()
    finally:
        connection.close()
    frame = _rows_to_frame(rows, meta)
    if frame.height == 0:
        raise ValueError("outer-training partition is empty")
    predictors = frame.select(list(meta.predictor_types))
    target = frame.get_column("target")
    if target.unique().sort().to_list() != [0, 1]:
        raise ValueError("outer-training partition must contain both target classes")
    index = pd.Index(frame.get_column("row_index").to_numpy(), name="row_index")
    groups = pd.Series(frame.get_column("building_id").to_list(), index=index, name="building_id")
    return TrainingPartition(
        repeat=repeat, fold=fold, X_train=predictors.to_pandas().set_axis(index, axis=0),
        y_train=pd.Series(target.to_numpy(), index=index, name="anomaly"), groups_train=groups,
    )


def iter_full_validation_batches(
    index_path: str | Path,
    admitted_spec: FullRunSpec,
    repeat: int,
    fold: int,
) -> Iterator[dict[str, object]]:
    """Stream validation predictors with scoring-only real labels and keys.

    Each batch carries predictors, exact keys and the real target outside the
    estimator input; no batch is reused or resampled.
    """
    _require_repeat_fold(repeat, fold)
    connection, meta = _open_bound_index(index_path, admitted_spec)
    try:
        seed = connection.execute(
            "SELECT DISTINCT seed FROM assign WHERE repeat = ?", (repeat,)
        ).fetchall()
        if len(seed) != 1:
            raise ValueError("validation repeat must declare exactly one split seed")
        support = connection.execute(
            "SELECT COUNT(DISTINCT target) FROM dev_row d JOIN assign a ON a.building_id = d.building_id"
            " WHERE a.repeat = ? AND a.fold = ?",
            (repeat, fold),
        ).fetchone()[0]
        if support != 2:
            raise ValueError("validation partition must contain both target classes")
        offset = 0
        while True:
            rows = connection.execute(
                _projection_sql(meta) + " WHERE a.repeat = ? AND a.fold = ?"
                " ORDER BY d.row_index LIMIT ? OFFSET ?",
                (repeat, fold, MAX_BATCH_ROWS, offset),
            ).fetchall()
            if not rows:
                return
            frame = _rows_to_frame(rows, meta)
            index = pd.Index(frame.get_column("row_index").to_numpy(), name="row_index")
            predictors = frame.select(meta.predictor_types.keys()).to_pandas().set_axis(index, axis=0)
            if len(predictors) > MAX_BATCH_ROWS:
                raise ValueError("validation batch exceeds the writer bound")
            yield {
                "X": predictors,
                "row_index": frame.get_column("row_index").to_numpy(),
                "building_id": frame.get_column("building_id").to_list(),
                "timestamp": frame.get_column("timestamp").to_numpy(),
                "target": frame.get_column("target").to_numpy(),
                "repeat": repeat,
                "fold": fold,
                "split_seed": int(seed[0][0]),
            }
            offset += len(rows)
    finally:
        connection.close()


def iter_full_expected_batches(
    index_path: str | Path,
    admitted_spec: FullRunSpec,
) -> Iterator[pl.DataFrame]:
    """Stream authoritative development expected rows per repeat in row order."""
    connection, meta = _open_bound_index(index_path, admitted_spec)
    try:
        repeats = [row[0] for row in connection.execute("SELECT DISTINCT repeat FROM assign ORDER BY repeat")]
        for repeat in repeats:
            seed = connection.execute(
                "SELECT DISTINCT seed FROM assign WHERE repeat = ?", (repeat,)
            ).fetchone()[0]
            offset = 0
            while True:
                rows = connection.execute(
                    "SELECT d.row_index, d.building_id, d.timestamp_ns, d.target, a.fold"
                    " FROM dev_row d JOIN assign a ON a.building_id = d.building_id"
                    " WHERE a.repeat = ? ORDER BY d.row_index LIMIT ? OFFSET ?",
                    (repeat, MAX_BATCH_ROWS, offset),
                ).fetchall()
                if not rows:
                    break
                frame = _lineage_frame(rows, meta)
                yield frame.select(
                    "row_index",
                    pl.lit(repeat, dtype=pl.Int64).alias("repeat"),
                    pl.lit(int(seed), dtype=pl.Int64).alias("split_seed"),
                    pl.col("fold"),
                    "building_id",
                    "timestamp",
                    pl.col("target"),
                    pl.lit(meta.data_sha256).alias("data_sha256"),
                    pl.lit(meta.split_sha256).alias("split_sha256"),
                    pl.lit(meta.run_binding).alias("config_sha256"),
                    pl.lit(admitted_spec.run_id).alias("run_id"),
                )
                offset += len(rows)
    finally:
        connection.close()


def _confined(root: Path, relative: str, label: str) -> Path:
    return _validate_managed_path(root / relative, label, root=root)


def _source_hash(spec: FullRunSpec, role: str) -> str:
    matches = [(source.role, source.sha256) for source in spec.sources if source.role == role]
    if len(matches) != 1:
        raise ValueError(f"full spec must declare exactly one {role} source")
    return matches[0][1]


def _declared_path(spec: FullRunSpec, role: str) -> str:
    matches = [source.path for source in spec.sources if source.role == role]
    if len(matches) != 1:
        raise ValueError(f"full spec must declare exactly one {role} source")
    return matches[0]


def _require_declared_path(spec: FullRunSpec, role: str, expected: str) -> None:
    if _declared_path(spec, role) != expected:
        raise ValueError(f"declared {role} path must be the confined {expected!r}; discovery is not allowed")


def _building_dtype(schema: pl.Schema) -> str:
    if "building_id" not in schema:
        raise ValueError("development data is missing building_id")
    dtype = schema["building_id"]
    if dtype == pl.String:
        return PREDICTOR_STRING
    if dtype == pl.Int64:
        return PREDICTOR_INT
    raise ValueError("building identity must be opaque string or int64")


def _predictors_from_schema(schema: pl.Schema, spec: FullRunSpec) -> tuple[str, ...]:
    predictors = CORE
    missing = [name for name in ("building_id", "timestamp", "anomaly", *predictors) if name not in schema]
    if missing:
        raise ValueError(f"development data is missing required columns: {missing}")
    for candidate in spec.candidates:
        canonical_feature_order(candidate.spec.feature_pool)
    return predictors


def _predictor_types(schema: pl.Schema, predictors: tuple[str, ...]) -> dict[str, str]:
    types: dict[str, str] = {}
    for name in predictors:
        dtype = schema[name]
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            types[name] = PREDICTOR_INT
        elif dtype in (pl.Float32, pl.Float64):
            types[name] = PREDICTOR_FLOAT
        elif dtype == pl.String:
            types[name] = PREDICTOR_STRING
        else:
            raise ValueError(f"unsupported predictor dtype for {name}: {dtype}")
    return types


def _read_assignments(split_path: Path):
    manifest = pl.read_parquet(split_path)
    required = ("building_id", "repeat", "seed", "split", "fold")
    if any(name not in manifest.columns for name in required):
        raise ValueError("split manifest is missing required columns")
    vault_manifest = manifest.filter(pl.col("split") == "vault")
    vault_ids = set(vault_manifest["building_id"].to_list())
    dev_manifest = manifest.filter(pl.col("split") == "dev")
    if dev_manifest.height == 0:
        raise ValueError("split manifest has no development buildings")
    if dev_manifest["fold"].null_count():
        raise ValueError("development folds must be nonnull")
    if vault_manifest.height and vault_manifest["fold"].null_count() != vault_manifest.height:
        raise ValueError("vault folds must be null")
    dev_ids = set(dev_manifest["building_id"].to_list())
    if dev_ids & vault_ids:
        raise ValueError("development and vault building membership overlap")
    reference = None
    assignments = []
    for repeat, expected_seed in enumerate((42, 43, 44)):
        part = dev_manifest.filter(pl.col("repeat") == repeat)
        if part["seed"].unique().to_list() != [expected_seed]:
            raise ValueError("development split seed does not match the frozen protocol")
        if sorted(part["fold"].unique().to_list()) != [0, 1, 2, 3]:
            raise ValueError("development manifest must declare exactly four folds")
        if set(part["building_id"].to_list()) != dev_ids:
            raise ValueError("development membership differs across repeats")
        for building in part["building_id"].to_list():
            folds = part.filter(pl.col("building_id") == building)["fold"].unique().to_list()
            if len(folds) != 1:
                raise ValueError("a building spans multiple folds within one repeat")
        if reference is not None and _fold_partition(part) == reference:
            raise ValueError("split repetitions are copied partitions or fold-label permutations")
        reference = _fold_partition(part)
        assignments.extend(
            (building, repeat, int(fold), expected_seed)
            for building, fold in zip(part["building_id"].to_list(), part["fold"].to_list())
        )
    return dev_ids, vault_ids, tuple(assignments)


def _fold_partition(part: pl.DataFrame):
    return tuple(sorted((int(fold), tuple(sorted(part.filter(pl.col("fold") == fold)["building_id"].to_list())))
                        for fold in part["fold"].unique().to_list()))


def _create_schema(
    connection: sqlite3.Connection,
    predictors: tuple[str, ...],
    building_dtype: str,
    predictor_types: dict[str, str],
) -> None:
    building_type = "INTEGER" if building_dtype == PREDICTOR_INT else "TEXT"
    affinities = {PREDICTOR_INT: "INTEGER", PREDICTOR_FLOAT: "REAL", PREDICTOR_STRING: "TEXT"}
    predictor_columns = ", ".join(f'"{name}" {affinities[predictor_types[name]]}' for name in predictors)
    connection.execute(
        f"CREATE TABLE dev_row_stage (building_id {building_type} NOT NULL, timestamp_ns INTEGER NOT NULL,"
        f" target INTEGER NOT NULL, {predictor_columns}, UNIQUE(building_id, timestamp_ns))"
    )
    connection.execute(
        f"CREATE TABLE dev_row (row_index INTEGER PRIMARY KEY, building_id {building_type} NOT NULL,"
        " timestamp_ns INTEGER NOT NULL, target INTEGER NOT NULL,"
        f" {predictor_columns}, UNIQUE(building_id, timestamp_ns))"
    )
    connection.execute(
        f"CREATE TABLE assign (building_id {building_type} NOT NULL, repeat INTEGER NOT NULL,"
        " fold INTEGER NOT NULL, seed INTEGER NOT NULL, PRIMARY KEY(building_id, repeat))"
    )
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE predictor_order (position INTEGER PRIMARY KEY, name TEXT NOT NULL)")


def _materialize_sorted_index(connection: sqlite3.Connection, predictors: tuple[str, ...]) -> None:
    """Assign the baseline's sorted development identity once, disk-backed."""
    names = ", ".join(f'"{name}"' for name in predictors)
    connection.execute(
        "INSERT INTO dev_row SELECT ROW_NUMBER() OVER (ORDER BY building_id, timestamp_ns) - 1,"
        f" building_id, timestamp_ns, target, {names} FROM dev_row_stage"
    )
    connection.execute("DROP TABLE dev_row_stage")


def _store_meta(connection: sqlite3.Connection, meta: _IndexMeta, predictors: tuple[str, ...], spec: FullRunSpec) -> None:
    values = {
        "schema_version": json.dumps(_INDEX_SCHEMA_VERSION),
        "run_binding": meta.run_binding,
        "data_sha256": meta.data_sha256,
        "split_sha256": meta.split_sha256,
        "data_path": meta.data_path,
        "timestamp_unit": meta.timestamp_unit,
        "timestamp_time_zone": json.dumps(meta.timestamp_time_zone),
        "building_dtype": meta.building_dtype,
        "predictor_types": json.dumps(meta.predictor_types, sort_keys=True),
        "dev_row_count": json.dumps(meta.dev_row_count),
        "identity_sha256": meta.identity_sha256,
        "protocol": json.dumps({"repeats": spec.repeats, "folds": spec.folds,
                                "split_seeds": list(spec.split_seeds)}),
    }
    connection.executemany("INSERT INTO meta VALUES (?, ?)", list(values.items()))
    connection.executemany("INSERT INTO predictor_order VALUES (?, ?)",
                           list(enumerate(predictors)))


def _store_assignments(connection: sqlite3.Connection, assignments) -> None:
    connection.executemany(
        "INSERT INTO assign VALUES (?, ?, ?, ?)",
        [(building, repeat, fold, seed) for building, repeat, fold, seed in assignments],
    )


def _stream_dev_rows(connection, data_path, schema, predictors, dev_buildings, vault_buildings):
    columns = ["building_id", "timestamp", "anomaly", *predictors]
    predictor_types = _predictor_types(schema, predictors)
    building_dtype = _building_dtype(schema)
    seen_buildings: set[object] = set()
    inserted = 0
    class_values: set[int] = set()
    insert_sql = _insert_sql(predictors)
    parquet = pq.ParquetFile(data_path)
    try:
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, columns=columns):
            frame = pl.from_arrow(batch)
            seen_buildings.update(frame.get_column("building_id").unique().to_list())
            filtered = frame.filter(pl.col("building_id").is_in(list(dev_buildings)))
            if filtered.height == 0:
                continue
            rows = list(_dev_row_tuples(filtered, predictors, predictor_types, building_dtype))
            class_values.update(value[2] for value in rows)
            if not class_values <= {0, 1}:
                raise ValueError("development labels must be integer binary 0/1")
            try:
                connection.executemany(insert_sql, rows)
            except sqlite3.IntegrityError as error:
                raise ValueError("development natural keys or physical row identity are duplicated") from error
            inserted += len(rows)
    finally:
        del parquet
    unknown = seen_buildings - dev_buildings - vault_buildings
    if unknown:
        raise ValueError(f"development data contains unknown buildings: {sorted(unknown)}")
    if dev_buildings - seen_buildings:
        raise ValueError("declared development building membership has missing rows")
    return inserted, class_values


def _dev_row_tuples(filtered: pl.DataFrame, predictors, predictor_types, building_dtype):
    building_series = filtered.get_column("building_id")
    timestamp_ns = filtered.get_column("timestamp").cast(pl.Int64).to_list()
    targets = filtered.get_column("anomaly").to_list()
    stored_buildings = [_encode_building(value, building_dtype) for value in building_series.to_list()]
    predictor_values = [_encode_predictor(filtered.get_column(name).to_list(), predictor_types[name])
                        for name in predictors]
    for position in range(filtered.height):
        yield (stored_buildings[position], timestamp_ns[position], targets[position],
               *[column[position] for column in predictor_values])


def _insert_sql(predictors: tuple[str, ...]) -> str:
    columns = ", ".join(f'"{name}"' for name in predictors)
    placeholders = ", ".join("?" for _ in range(3 + len(predictors)))
    return f"INSERT INTO dev_row_stage (building_id, timestamp_ns, target, {columns}) VALUES ({placeholders})"


def _encode_building(value, building_dtype):
    if value is None:
        raise ValueError("development building identity must not be null")
    if building_dtype == PREDICTOR_INT:
        if type(value) is not int:
            raise ValueError("integer building identity required")
        return value
    if not isinstance(value, str) or not value:
        raise ValueError("opaque string building identity required")
    return value


def _encode_predictor(values, kind):
    if kind == PREDICTOR_STRING:
        if any(value is not None and not isinstance(value, str) for value in values):
            raise ValueError("categorical predictor must contain string or null values")
        return values
    if any(value is not None and not isinstance(value, (int, float)) for value in values):
        raise ValueError("numeric predictor must contain numeric or null values")
    return values


def _projection_sql(meta: _IndexMeta) -> str:
    predictor_columns = ", ".join(f'd."{name}"' for name in meta.predictor_types)
    prefix = "SELECT d.row_index, d.building_id, d.timestamp_ns, d.target, a.fold, "
    return prefix + predictor_columns + " FROM dev_row d JOIN assign a ON a.building_id = d.building_id"


def _rows_to_frame(rows, meta: _IndexMeta) -> pl.DataFrame:
    building_dtype = _building_polars_type(meta)
    timestamp_dtype = pl.Datetime(meta.timestamp_unit, meta.timestamp_time_zone)
    predictor_polars = {PREDICTOR_INT: pl.Int64, PREDICTOR_FLOAT: pl.Float64, PREDICTOR_STRING: pl.String}
    columns: dict[str, list] = {
        "row_index": [], "building_id": [], "timestamp_ns": [], "target": [], "fold": [],
    }
    for name in meta.predictor_types:
        columns[name] = []
    schema = {
        "row_index": pl.Int64, "building_id": building_dtype, "timestamp_ns": pl.Int64,
        "target": pl.Int64, "fold": pl.Int64,
    }
    for name, kind in meta.predictor_types.items():
        # Explicit typing keeps all-null predictor columns numeric instead of
        # collapsing to pl.Null (which becomes object dtype in pandas).
        schema[name] = predictor_polars[kind]
    decode = _predictor_decoder(meta.predictor_types)
    for row in rows:
        columns["row_index"].append(row[0])
        columns["building_id"].append(row[1])
        columns["timestamp_ns"].append(row[2])
        columns["target"].append(row[3])
        columns["fold"].append(row[4])
        for position, name in enumerate(meta.predictor_types, start=5):
            columns[name].append(decode[name](row[position]))
    frame = pl.DataFrame(columns, schema=schema)
    return frame.with_columns(
        pl.col("timestamp_ns").cast(timestamp_dtype).alias("timestamp"),
    ).drop("timestamp_ns")


def _predictor_decoder(predictor_types):
    return {name: _decoder_for(kind) for name, kind in predictor_types.items()}


def _decoder_for(kind):
    if kind == PREDICTOR_INT:
        return lambda value: None if value is None else int(value)
    if kind == PREDICTOR_FLOAT:
        return lambda value: None if value is None else float(value)
    return lambda value: value


def _building_polars_type(meta: _IndexMeta):
    return pl.Int64 if meta.building_dtype == PREDICTOR_INT else pl.String


def _lineage_frame(rows, meta: _IndexMeta) -> pl.DataFrame:
    """Build only identity/target columns; predictors are not needed here."""
    return pl.DataFrame(
        {
            "row_index": [row[0] for row in rows],
            "building_id": [row[1] for row in rows],
            "timestamp_ns": [row[2] for row in rows],
            "target": [row[3] for row in rows],
            "fold": [row[4] for row in rows],
        },
        schema={
            "row_index": pl.Int64, "building_id": _building_polars_type(meta),
            "timestamp_ns": pl.Int64, "target": pl.Int64, "fold": pl.Int64,
        },
    ).with_columns(
        pl.col("timestamp_ns").cast(pl.Datetime(meta.timestamp_unit, meta.timestamp_time_zone)).alias("timestamp"),
    ).drop("timestamp_ns")


def _open_bound_index(index_path, admitted_spec: FullRunSpec):
    if not isinstance(admitted_spec, FullRunSpec):
        raise TypeError("admitted_spec must be a typed FullRunSpec")
    path = _validate_managed_path(index_path, "index", directory=False)
    if not path.exists():
        raise FileNotFoundError("development index does not exist")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = None
    try:
        # Hold one read snapshot from integrity verification through consumer
        # queries, so an intervening SQLite writer cannot swap fitting inputs.
        connection.execute("BEGIN")
        meta = _load_meta(connection)
        if meta.run_binding != admitted_spec.binding_sha256:
            raise ValueError("development index binding does not match the admitted spec")
        _require_declared_path(admitted_spec, "data", DATA_PATH)
        _require_declared_path(admitted_spec, "split", SPLIT_PATH)
        if meta.data_sha256 != _source_hash(admitted_spec, "data") or meta.split_sha256 != _source_hash(admitted_spec, "split"):
            raise ValueError("development index source hashes do not match the admitted spec")
        _verify_index_cardinality(connection, admitted_spec.development_rows)
        _verify_population_and_support(connection, admitted_spec)
    except BaseException:
        connection.close()
        raise
    return connection, meta


def _load_meta(connection: sqlite3.Connection) -> _IndexMeta:
    stored = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    if int(stored.get("schema_version", "0")) != _INDEX_SCHEMA_VERSION:
        raise ValueError("development index schema version mismatch")
    required = {"schema_version", "run_binding", "data_sha256", "split_sha256", "data_path",
                "timestamp_unit", "timestamp_time_zone", "building_dtype", "predictor_types",
                 "protocol", "dev_row_count", "identity_sha256", "integrity_sha256"}
    if set(stored) != required:
        raise ValueError("development index metadata is incomplete")
    if _integrity_digest(connection) != stored["integrity_sha256"]:
        raise ValueError("development index integrity mismatch")
    order = [row[1] for row in connection.execute("SELECT position, name FROM predictor_order ORDER BY position")]
    if tuple(order) != CORE:
        raise ValueError("development index predictor order is not the canonical approved core")
    types = json.loads(stored["predictor_types"])
    if set(order) != set(types):
        raise ValueError("development index predictor order is inconsistent")
    if _identity_digest(connection) != stored["identity_sha256"]:
        raise ValueError("development index row identity was renumbered or tampered")
    return _IndexMeta(
        run_binding=stored["run_binding"], data_sha256=stored["data_sha256"],
        split_sha256=stored["split_sha256"], data_path=stored["data_path"],
        timestamp_unit=stored["timestamp_unit"],
        timestamp_time_zone=json.loads(stored["timestamp_time_zone"]),
        building_dtype=stored["building_dtype"],
        predictor_types={name: types[name] for name in order},
        dev_row_count=int(stored["dev_row_count"]),
        identity_sha256=stored["identity_sha256"],
    )


def _with_row_count(meta: _IndexMeta, count: int) -> _IndexMeta:
    return _replace_meta(meta, dev_row_count=count)


def _with_identity(meta: _IndexMeta, identity_sha256: str) -> _IndexMeta:
    return _replace_meta(meta, identity_sha256=identity_sha256)


def _replace_meta(meta: _IndexMeta, **changes) -> _IndexMeta:
    values = {
        "run_binding": meta.run_binding, "data_sha256": meta.data_sha256,
        "split_sha256": meta.split_sha256, "data_path": meta.data_path,
        "timestamp_unit": meta.timestamp_unit, "timestamp_time_zone": meta.timestamp_time_zone,
        "building_dtype": meta.building_dtype, "predictor_types": meta.predictor_types,
        "dev_row_count": meta.dev_row_count, "identity_sha256": meta.identity_sha256,
    }
    values.update(changes)
    return _IndexMeta(**values)


def _identity_digest(connection: sqlite3.Connection) -> str:
    """Hash the persisted (row_index, building, timestamp_ns, target) identity once."""
    digest = hashlib.sha256()
    for row_index, building, timestamp_ns, target in connection.execute(
        "SELECT row_index, building_id, timestamp_ns, target FROM dev_row ORDER BY row_index"
    ):
        building_bytes = str(building).encode("utf-8")
        digest.update(struct.pack("<q", row_index))
        digest.update(struct.pack("<q", len(building_bytes)))
        digest.update(building_bytes)
        digest.update(struct.pack("<q", timestamp_ns))
        digest.update(struct.pack("<q", target))
    return digest.hexdigest()


def _verify_index_cardinality(connection: sqlite3.Connection, expected: int) -> None:
    """Metadata validation, never permission to resample or repair the split."""
    minimum, maximum, count = connection.execute(
        "SELECT MIN(row_index), MAX(row_index), COUNT(*) FROM dev_row"
    ).fetchone()
    if count != expected or minimum != 0 or maximum != expected - 1:
        raise ValueError("index cardinality differs from authenticated development rows")


def _verify_population_and_support(connection: sqlite3.Connection, spec: FullRunSpec) -> None:
    """SQL anti-joins and aggregates, never population-scale Python maps."""
    if connection.execute(
        "SELECT 1 FROM assign a LEFT JOIN dev_row d USING (building_id)"
        " WHERE d.building_id IS NULL LIMIT 1"
    ).fetchone():
        raise ValueError("development assignment membership has no rows")
    if connection.execute(
        "SELECT 1 FROM dev_row d LEFT JOIN assign a USING (building_id)"
        " WHERE a.building_id IS NULL LIMIT 1"
    ).fetchone():
        raise ValueError("development row has no assignment membership")
    if connection.execute(
        "SELECT building_id FROM assign GROUP BY building_id HAVING COUNT(*) <> 3"
        " OR MIN(repeat) <> 0 OR MAX(repeat) <> 2 LIMIT 1"
    ).fetchone():
        raise ValueError("development assignment membership differs across repeats")
    if connection.execute("SELECT 1 FROM dev_row WHERE target NOT IN (0, 1) LIMIT 1").fetchone():
        raise ValueError("development target must be binary")
    # Twelve compact aggregate rows replace 24 repeated population scans.
    support_rows = connection.execute(
        "SELECT a.repeat, a.fold, SUM(d.target = 0), SUM(d.target = 1)"
        " FROM dev_row d JOIN assign a USING (building_id) GROUP BY a.repeat, a.fold"
    ).fetchall()
    support = {(repeat, fold): (normal, anomaly) for repeat, fold, normal, anomaly in support_rows}
    for repeat, seed in enumerate(spec.split_seeds):
        if connection.execute(
            "SELECT 1 FROM assign WHERE repeat = ? AND (seed <> ? OR fold NOT BETWEEN 0 AND 3) LIMIT 1",
            (repeat, seed),
        ).fetchone():
            raise ValueError("development assignment protocol mismatch")
        totals = tuple(sum(support.get((repeat, fold), (0, 0))[label] for fold in range(spec.folds))
                       for label in (0, 1))
        for fold in range(spec.folds):
            validation = support.get((repeat, fold), (0, 0))
            training = tuple(total - held_out for total, held_out in zip(totals, validation))
            if any(count == 0 for count in (*validation, *training)):
                raise ValueError(f"outer partition class support missing: repeat={repeat}, fold={fold}")


def _hash_value(digest, value: object) -> None:
    """Typed, length-delimited SQLite values preserve float64 and nulls."""
    if value is None:
        payload = b"N"
    elif type(value) is int:
        payload = b"I" + struct.pack("<q", value)
    elif type(value) is float:
        payload = b"F" + struct.pack("<d", value)
    elif isinstance(value, str):
        payload = b"S" + value.encode("utf-8")
    elif isinstance(value, bytes):
        payload = b"B" + value
    else:
        raise ValueError("unsupported persisted index value")
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _integrity_digest(connection: sqlite3.Connection) -> str:
    """Stream the entire finished fitting/index state in deterministic order.

    Covers schema, metadata, predictor order and values, exact labels/keys, and
    every repeat/fold/seed assignment. Only bounded result batches are retained.
    The stored digest detects cache corruption; parent evidence should bind the
    returned digest externally to reject an attacker recomputing local metadata.
    """
    digest = hashlib.sha256()
    queries = (
        ("schema", "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"),
        ("meta", "SELECT key, value FROM meta WHERE key <> 'integrity_sha256' ORDER BY key"),
        ("predictor_order", "SELECT position, name FROM predictor_order ORDER BY position"),
        ("assign", "SELECT * FROM assign ORDER BY building_id, repeat"),
        ("dev_row", "SELECT * FROM dev_row ORDER BY row_index"),
    )
    for name, query in queries:
        _hash_value(digest, name)
        cursor = connection.execute(query)
        for column in cursor.description:
            _hash_value(digest, column[0])
        while rows := cursor.fetchmany(MAX_BATCH_ROWS):
            for row in rows:
                digest.update(b"R")
                for value in row:
                    _hash_value(digest, value)
    return digest.hexdigest()


def verify_development_index(
    index_path: str | Path, admitted_spec: FullRunSpec, *,
    expected_integrity_sha256: str | None = None,
) -> str:
    """Verify finished integrity/population/all-fold support without fitting.

    Parent should persist the returned digest in authenticated run evidence and
    supply it on resume, before first fit, and at completion. Existing consumers
    also automatically verify the stored integrity on each read-only snapshot.
    """
    connection, _ = _open_bound_index(index_path, admitted_spec)
    try:
        digest = connection.execute("SELECT value FROM meta WHERE key = 'integrity_sha256'").fetchone()[0]
        if expected_integrity_sha256 is not None and digest != expected_integrity_sha256:
            raise ValueError("development index external integrity binding mismatch")
        return digest
    finally:
        connection.close()


def _require_matching_meta(connection, spec, data_sha256, split_sha256):
    try:
        meta = _load_meta(connection)
    except (sqlite3.DatabaseError, ValueError) as error:
        raise ValueError("existing index is not a valid development index") from error
    if (meta.run_binding, meta.data_sha256, meta.split_sha256) != (spec.binding_sha256, data_sha256, split_sha256):
        raise ValueError("existing development index has a different immutable source binding")
    _verify_index_cardinality(connection, spec.development_rows)
    _verify_population_and_support(connection, spec)


def _require_repeat_fold(repeat: int, fold: int) -> None:
    for name, value, upper in (("repeat", repeat, 2), ("fold", fold, 3)):
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError(f"{name} must be an integer within the frozen protocol")
