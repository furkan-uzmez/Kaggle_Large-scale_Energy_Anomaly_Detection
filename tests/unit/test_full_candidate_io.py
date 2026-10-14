"""Behavioral tests for full-run projected development I/O.

The tests use a real temporary projected Parquet fixture with non-parity labels,
opaque building ids, exact nanosecond timestamps and vault poison rows. Only the
filesystem/model boundaries are spied on; the I/O implementation itself is real.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from full_campaign_fixtures import (
    DATA_RELATIVE,
    DEV_BUILDINGS,
    VAULT_BUILDINGS,
    VAULT_METER_READING,
    development_frame,
    int_development_frame,
)
from full_campaign_fixtures import (
    make_spec as shared_make_spec,
)
from full_campaign_fixtures import (
    write_fixture as shared_write_fixture,
)
from full_campaign_fixtures import (
    write_int_fixture as shared_write_int_fixture,
)
from project.experiments import full_candidate_io as io
from project.experiments.candidate_contracts import TrainingPartition
from project.experiments.full_input_schema import CORE

REPEATS = (0, 1, 2)
FOLDS = (0, 1, 2, 3)
PREDICTOR_COLUMNS = CORE
EXPECTED_COLUMNS = {
    "row_index", "repeat", "split_seed", "fold", "building_id", "timestamp", "target",
    "data_sha256", "split_sha256", "config_sha256", "run_id",
}


def make_spec(root):
    """Explicit small fixture counts; never silently inherit real-data defaults."""
    return replace(shared_make_spec(root), development_rows=52, vault_rows=4)


def write_fixture(root):
    fixture = shared_write_fixture(root)
    return replace(fixture, spec=make_spec(fixture.root))


def write_int_fixture(root):
    fixture = shared_write_int_fixture(root)
    # FullRunSpec requires a nonempty declared vault. Add a small, disjoint
    # vault only in this owned helper rather than weakening shared contracts.
    vault = fixture.full_frame.head(4).with_columns(pl.lit(100, dtype=pl.Int64).alias("building_id"))
    frame = pl.concat([fixture.full_frame, vault])
    frame.write_parquet(fixture.data_path)
    manifest = pl.read_parquet(fixture.split_path)
    vault_assignments = pl.DataFrame({
        "building_id": [100] * 3, "repeat": [0, 1, 2], "seed": [42, 43, 44],
        "split": ["vault"] * 3, "fold": [None] * 3,
    }, schema=manifest.schema)
    pl.concat([manifest, vault_assignments]).write_parquet(fixture.split_path)
    return replace(fixture, spec=make_spec(fixture.root), full_frame=frame)


@pytest.fixture
def campaign(tmp_path: Path):
    return write_fixture(tmp_path / "campaign")


@pytest.fixture
def index(campaign):
    return io.build_development_index(campaign.root, campaign.spec, campaign.root / "dev_index.sqlite3")


def index_rows(index_path: Path) -> dict[int, tuple[str, int, int]]:
    with sqlite3.connect(index_path) as connection:
        return {
            int(row_index): (building, int(timestamp_ns), int(target))
            for row_index, building, timestamp_ns, target in connection.execute(
                "SELECT row_index, building_id, timestamp_ns, target FROM dev_row ORDER BY row_index"
            )
        }


def fold_buildings(index_path: Path, repeat: int, fold: int) -> set[str]:
    with sqlite3.connect(index_path) as connection:
        return {
            row[0] for row in connection.execute(
                "SELECT building_id FROM assign WHERE repeat = ? AND fold = ?", (repeat, fold)
            )
        }


def all_fold_buildings(index_path: Path, repeat: int) -> set[str]:
    with sqlite3.connect(index_path) as connection:
        return {
            row[0] for row in connection.execute(
                "SELECT building_id FROM assign WHERE repeat = ?", (repeat,)
            )
        }


def test_all_development_rows_are_indexed_once(campaign, index):
    rows = index_rows(index)
    assert len(rows) == development_frame().height
    assert set(rows) == set(range(len(rows)))
    assert {building for building, _, _ in rows.values()} == set(DEV_BUILDINGS)


def test_exact_four_by_three_membership(index):
    with sqlite3.connect(index) as connection:
        repeats = connection.execute(
            "SELECT repeat, COUNT(*) FROM assign GROUP BY repeat ORDER BY repeat"
        ).fetchall()
        folds = connection.execute(
            "SELECT repeat, fold, COUNT(*) FROM assign GROUP BY repeat, fold ORDER BY repeat, fold"
        ).fetchall()
    assert repeats == [(repeat, len(DEV_BUILDINGS)) for repeat in REPEATS]
    assert folds == [(repeat, fold, len(DEV_BUILDINGS) // 4) for repeat in REPEATS for fold in FOLDS]


def test_training_partition_is_isolated_and_validation_streams_real_labels(campaign, index):
    seen_validation: set[str] = set()
    for repeat in REPEATS:
        for fold in FOLDS:
            partition = io.load_full_training_partition(index, campaign.spec, repeat, fold)
            assert isinstance(partition, TrainingPartition)
            assert (partition.repeat, partition.fold) == (repeat, fold)
            assert set(partition.y_train.unique()) == {0, 1}
            assert tuple(partition.X_train.columns) == tuple(PREDICTOR_COLUMNS)
            validation_buildings = fold_buildings(index, repeat, fold)
            assert set(partition.groups_train).isdisjoint(validation_buildings)
            assert set(partition.groups_train) == all_fold_buildings(index, repeat) - validation_buildings
            batches = list(io.iter_full_validation_batches(index, campaign.spec, repeat, fold))
            total = sum(len(batch["X"]) for batch in batches)
            assert total >= 1
            labels = {int(label) for batch in batches for label in batch["target"]}
            assert labels <= {0, 1}
            for batch in batches:
                assert tuple(batch["X"].columns) == tuple(PREDICTOR_COLUMNS)
                assert len(batch["X"]) <= io.MAX_BATCH_ROWS
                assert batch["repeat"] == repeat and batch["fold"] == fold
                assert batch["split_seed"] == 42 + repeat
            assert {row for batch in batches for row in batch["building_id"]} == validation_buildings
            seen_validation |= validation_buildings
    assert seen_validation == set(DEV_BUILDINGS)


def test_expected_batches_cover_identical_population_with_real_non_parity_labels(campaign, index):
    frames = list(io.iter_full_expected_batches(index, campaign.spec))
    assert [frame["repeat"][0] for frame in frames] == list(REPEATS)
    populations = []
    for frame in frames:
        assert EXPECTED_COLUMNS <= set(frame.columns)
        assert frame.height == development_frame().height
        assert frame.schema["timestamp"] == pl.Datetime("ns")
        assert frame["split_seed"].unique().to_list() == [42 + frame["repeat"][0]]
        assert set(frame["target"].unique()) == {0, 1}
        populations.append(set(zip(frame["building_id"], frame["timestamp"].cast(pl.Int64))))
    assert populations[0] == populations[1] == populations[2]
    combined = pl.concat(frames).sort("row_index")
    labels = combined["target"].to_numpy()
    parity = (combined["row_index"].to_numpy() % 2).astype(labels.dtype)
    assert not (labels == parity).all(), "fixture labels must not be row-parity placeholders"


def test_expected_lineage_uses_real_hashes_and_run_id(campaign, index):
    source_hash = {source.role: source.sha256 for source in campaign.spec.sources}
    frame = pl.concat(io.iter_full_expected_batches(index, campaign.spec))
    assert frame["data_sha256"].unique().to_list() == [source_hash["data"]]
    assert frame["split_sha256"].unique().to_list() == [source_hash["split"]]
    assert frame["config_sha256"].unique().to_list() == [campaign.spec.binding_sha256]
    assert frame["run_id"].unique().to_list() == ["fixture-run"]


def test_row_identity_and_expected_batches_stable_across_chunk_sizes(campaign, monkeypatch):
    default_index = io.build_development_index(campaign.root, campaign.spec, campaign.root / "default.sqlite3")
    baseline_rows = index_rows(default_index)
    baseline_expected = pl.concat(io.iter_full_expected_batches(default_index, campaign.spec))
    monkeypatch.setattr(io, "MAX_BATCH_ROWS", 3)
    small_index = io.build_development_index(campaign.root, campaign.spec, campaign.root / "small.sqlite3")
    assert index_rows(small_index) == baseline_rows
    small_expected = pl.concat(io.iter_full_expected_batches(small_index, campaign.spec))
    assert small_expected.equals(baseline_expected)


def test_row_identity_stable_across_physical_row_order(campaign):
    original_index = io.build_development_index(campaign.root, campaign.spec, campaign.root / "first.sqlite3")
    original = index_rows(original_index)
    development_frame().sort("anomaly").write_parquet(campaign.data_path)
    reordered_index = io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "reordered.sqlite3")
    assert index_rows(reordered_index) == original


def test_zero_vault_rows_retained_and_test_file_never_opened(campaign, monkeypatch):
    opened: list[str] = []
    real_parquet_file = io.pq.ParquetFile

    def spy(where, *args, **kwargs):
        opened.append(str(where))
        return real_parquet_file(where, *args, **kwargs)

    monkeypatch.setattr(io.pq, "ParquetFile", spy)
    index = io.build_development_index(campaign.root, campaign.spec, campaign.root / "vault.sqlite3")
    assert not any("test_joined" in path for path in opened)
    assert all("train_joined" in path for path in opened)
    with sqlite3.connect(index) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM dev_row WHERE meter_reading = ?", (VAULT_METER_READING,)
        ).fetchone()[0] == 0
    assert {building for building, _, _ in index_rows(index).values()}.isdisjoint(set(VAULT_BUILDINGS))


def test_rejects_malformed_label(campaign):
    malformed = development_frame().with_columns(
        pl.when(pl.col("building_id") == DEV_BUILDINGS[0]).then(7).otherwise(pl.col("anomaly")).alias("anomaly")
    )
    malformed.write_parquet(campaign.data_path)
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "malformed.sqlite3")


def test_rejects_missing_label(campaign):
    missing = development_frame().with_columns(
        pl.when(pl.col("building_id") == DEV_BUILDINGS[0]).then(None).otherwise(pl.col("anomaly")).alias("anomaly")
    )
    missing.write_parquet(campaign.data_path)
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "missing.sqlite3")


def test_rejects_both_class_support_failure(campaign):
    single = development_frame().with_columns(pl.lit(0, dtype=pl.Int64).alias("anomaly"))
    single.write_parquet(campaign.data_path)
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "single.sqlite3")


def test_rejects_duplicate_natural_keys(campaign):
    frame = development_frame()
    duplicate = frame.head(1).with_columns((pl.col("anomaly") + 1) % 2)
    pl.concat([frame, duplicate]).write_parquet(campaign.data_path)
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "dup.sqlite3")


def test_rejects_unknown_building(campaign):
    intruder = development_frame().head(1).with_columns(pl.lit("bldg_unknown").alias("building_id"))
    pl.concat([development_frame(), intruder]).write_parquet(campaign.data_path)
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "unknown.sqlite3")


def test_rejects_timestamp_precision_drift(campaign):
    drifted = development_frame().with_columns(
        pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms")).alias("timestamp")
    )
    drifted.write_parquet(campaign.data_path)
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "drift.sqlite3")


def test_rejects_changed_source_hash(campaign):
    io.build_development_index(campaign.root, campaign.spec, campaign.root / "bound.sqlite3")
    (campaign.root / DATA_RELATIVE).write_bytes((campaign.root / DATA_RELATIVE).read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, campaign.spec, campaign.root / "changed.sqlite3")


def test_rejects_per_fold_renumbered_row_index(campaign, index):
    with sqlite3.connect(index) as connection:
        connection.execute("ALTER TABLE dev_row RENAME TO original")
        connection.execute(
            "CREATE TABLE dev_row (row_index INTEGER PRIMARY KEY, building_id TEXT NOT NULL,"
            " timestamp_ns INTEGER NOT NULL, target INTEGER NOT NULL, meter_reading REAL,"
            " primary_use TEXT, square_feet REAL, year_built INTEGER, floor_count INTEGER,"
            " hour INTEGER, weekday INTEGER, month INTEGER, year INTEGER,"
            " UNIQUE(building_id, timestamp_ns))"
        )
        connection.execute(
            "INSERT INTO dev_row"
            " SELECT ROW_NUMBER() OVER (ORDER BY a.fold, d.building_id DESC, d.timestamp_ns DESC) - 1,"
            " d.building_id, d.timestamp_ns, d.target, d.meter_reading, d.primary_use,"
            " d.square_feet, d.year_built, d.floor_count, d.hour, d.weekday, d.month, d.year"
            " FROM original d JOIN (SELECT building_id, fold FROM assign WHERE repeat = 0) a"
            " USING (building_id)"
        )
        connection.execute("DROP TABLE original")
        connection.commit()
    with pytest.raises(ValueError):
        io.load_full_training_partition(index, campaign.spec, 0, 0)


def test_declared_data_and_split_paths_must_match_the_confined_constants(campaign):
    # A spec may not authenticate one file while I/O silently reads another.
    other_data = campaign.root / "outputs/prepared/other.parquet"
    other_data.write_bytes((campaign.root / DATA_RELATIVE).read_bytes())
    misdirected = tuple(
        replace(source, path="outputs/prepared/other.parquet",
                sha256=io.hashlib.sha256(other_data.read_bytes()).hexdigest())
        if source.role == "data" else source
        for source in campaign.spec.sources
    )
    spec = replace(campaign.spec, sources=misdirected)
    with pytest.raises(ValueError):
        io.build_development_index(campaign.root, spec, campaign.root / "misdirected.sqlite3")


def test_integer_building_identity_uses_numeric_order_not_text(campaign):
    int_root = campaign.root / "int_campaign"
    fixture = write_int_fixture(int_root)
    int_index = io.build_development_index(int_root, fixture.spec, int_root / "int_index.sqlite3")
    with sqlite3.connect(int_index) as connection:
        ordered = [int(row[0]) for row in connection.execute(
            "SELECT building_id FROM dev_row ORDER BY row_index"
        )]
    first_row_index = {}
    for position, building in enumerate(ordered):
        first_row_index.setdefault(building, position)
    # Building 10 must appear after buildings 1..9, not directly after building 1
    # as a lexicographic TEXT ordering would place it.
    assert first_row_index[10] > first_row_index[9]
    assert first_row_index[20] > first_row_index[19]
    assert int_development_frame().height == len(ordered)


def test_float64_predictors_round_trip_without_precision_loss(campaign, index):
    sentinel = 123.45678901234567
    frame = development_frame().with_columns(
        pl.when(pl.col("building_id") == DEV_BUILDINGS[0])
        .then(pl.lit(sentinel))
        .otherwise(pl.col("meter_reading"))
        .alias("meter_reading")
    )
    frame.write_parquet(campaign.data_path)
    fresh_index = io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "precise.sqlite3")
    with sqlite3.connect(fresh_index) as connection:
        stored = connection.execute(
            "SELECT meter_reading FROM dev_row WHERE building_id = ? ORDER BY timestamp_ns LIMIT 1",
            (DEV_BUILDINGS[0],),
        ).fetchone()[0]
    assert float(stored) == sentinel


def test_idempotent_index_rebuild_closes_connection(campaign, monkeypatch):
    path = io.build_development_index(campaign.root, campaign.spec, campaign.root / "reuse.sqlite3")
    closed = {"count": 0}
    real_connect = io.sqlite3.connect

    class TrackingConnection:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            closed["count"] += 1
            return self._real.close()

    def tracking_connect(*args, **kwargs):
        return TrackingConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(io.sqlite3, "connect", tracking_connect)
    again = io.build_development_index(campaign.root, campaign.spec, path)
    assert again == path
    assert closed["count"] >= 1, "idempotent index rebuild leaked its SQLite connection"


def test_all_null_predictor_column_keeps_numeric_dtype(campaign):
    frame = development_frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("floor_count"))
    frame.write_parquet(campaign.data_path)
    fresh_index = io.build_development_index(campaign.root, make_spec(campaign.root), campaign.root / "nullcol.sqlite3")
    partition = io.load_full_training_partition(fresh_index, make_spec(campaign.root), 0, 0)
    assert partition.X_train["floor_count"].dtype.kind == "f"
    for batch in io.iter_full_validation_batches(fresh_index, make_spec(campaign.root), 0, 0):
        assert batch["X"]["floor_count"].dtype.kind == "f"


def test_corrupt_index_open_closes_connection(campaign):
    index = io.build_development_index(campaign.root, campaign.spec, campaign.root / "corrupt.sqlite3")
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE meta SET value = ? WHERE key = 'identity_sha256'", ("0" * 64,))
        connection.commit()
    closed = {"count": 0}
    real_connect = io.sqlite3.connect

    class TrackingConnection:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            closed["count"] += 1
            return self._real.close()

    def tracking_connect(*args, **kwargs):
        return TrackingConnection(real_connect(*args, **kwargs))

    original = io.sqlite3.connect
    io.sqlite3.connect = tracking_connect
    try:
        with pytest.raises(ValueError):
            io.load_full_training_partition(index, campaign.spec, 0, 0)
    finally:
        io.sqlite3.connect = original
    assert closed["count"] == 1, "corrupt-index open leaked its SQLite connection"


@pytest.mark.parametrize("omission", ["row", "building"])
def test_declared_development_population_cannot_be_silently_shortened(campaign, omission):
    frame = development_frame()
    shortened = frame.slice(1) if omission == "row" else frame.filter(pl.col("building_id") != DEV_BUILDINGS[0])
    shortened.write_parquet(campaign.data_path)
    destination = campaign.root / "short.sqlite3"
    with pytest.raises(ValueError, match="development|membership"):
        io.build_development_index(campaign.root, make_spec(campaign.root), destination)
    assert not destination.exists()


@pytest.mark.parametrize("tamper", ["predictor", "assignment", "seed", "order", "metadata"])
@pytest.mark.parametrize("consumer", ["reuse", "training", "validation", "expected"])
def test_cached_fitting_inputs_are_integrity_checked(campaign, index, tamper, consumer):
    with sqlite3.connect(index) as connection:
        sql = {
            "predictor": "UPDATE dev_row SET meter_reading = meter_reading + 100 WHERE row_index = 0",
            "assignment": "UPDATE assign SET fold = (fold + 1) % 4 WHERE repeat = 2",
            "seed": "UPDATE assign SET seed = 45 WHERE repeat = 2",
            "order": "UPDATE predictor_order SET position = position + 20",
            "metadata": "UPDATE meta SET value = 'us' WHERE key = 'timestamp_unit'",
        }[tamper]
        connection.execute(sql)
    with pytest.raises(ValueError):
        if consumer == "reuse":
            io.build_development_index(campaign.root, campaign.spec, index)
        elif consumer == "training":
            io.load_full_training_partition(index, campaign.spec, 0, 0)
        elif consumer == "validation":
            next(io.iter_full_validation_batches(index, campaign.spec, 0, 0))
        else:
            next(io.iter_full_expected_batches(index, campaign.spec))


def test_later_single_class_partition_is_rejected_during_build(campaign):
    frame = development_frame()
    # Repeat 2/fold 3 has fir/garnet; earlier partitions and global labels
    # still have both classes. Waiting until that fold is loaded is too late.
    frame.with_columns(
        pl.when(pl.col("building_id").is_in(["bldg_fir", "bldg_garnet"]))
        .then(0).otherwise(pl.col("anomaly")).alias("anomaly")
    ).write_parquet(campaign.data_path)
    destination = campaign.root / "unsupported.sqlite3"
    with pytest.raises(ValueError, match="class support missing: repeat=2, fold=3"):
        io.build_development_index(campaign.root, make_spec(campaign.root), destination)
    assert not destination.exists()


def test_integrity_verifier_exposes_external_binding_and_is_batch_invariant(campaign, monkeypatch):
    verifier = getattr(io, "verify_development_index", None)
    assert callable(verifier), "finished index integrity boundary is missing"
    path = io.build_development_index(campaign.root, campaign.spec, campaign.root / "finished.sqlite3")
    digest = verifier(path, campaign.spec)
    assert len(digest) == 64
    monkeypatch.setattr(io, "MAX_BATCH_ROWS", 3)
    assert verifier(path, campaign.spec, expected_integrity_sha256=digest) == digest
    with pytest.raises(ValueError, match="integrity"):
        verifier(path, campaign.spec, expected_integrity_sha256="0" * 64)


def test_approved_feature_order_is_canonical_without_mutating_spec(campaign):
    from project.experiments.full_campaign_admission import CORE
    candidate = campaign.spec.candidates[0]
    spec = replace(campaign.spec, candidates=(replace(candidate, spec=replace(candidate.spec, feature_pool=CORE)),))
    original_binding = spec.binding_sha256
    path = io.build_development_index(campaign.root, spec, campaign.root / "canonical.sqlite3")
    assert tuple(io.load_full_training_partition(path, spec, 0, 0).X_train.columns) == CORE
    assert spec.binding_sha256 == original_binding


def test_legacy_registry_order_cache_is_not_reused_as_canonical(campaign, index):
    legacy = ("meter_reading", "primary_use", "square_feet", "year_built", "floor_count",
              "hour", "weekday", "month", "year")
    with sqlite3.connect(index) as connection:
        connection.execute("DELETE FROM predictor_order")
        connection.executemany("INSERT INTO predictor_order VALUES (?, ?)", enumerate(legacy))
        # A formerly legitimate schema-v2 index can have a valid local digest
        # while following the superseded registry-order policy.
        connection.execute("UPDATE meta SET value = ? WHERE key = 'integrity_sha256'",
                           (io._integrity_digest(connection),))
    with pytest.raises(ValueError, match="canonical|valid development index"):
        io.build_development_index(campaign.root, campaign.spec, index)
