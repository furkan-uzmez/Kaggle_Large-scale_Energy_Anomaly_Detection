from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from project.data.join import build_feature_joins
from project.data.load import (
    load_data_contract,
    scan_raw_table,
)
from project.data.validate import (
    compute_file_sha256,
    count_file_rows,
    validate_building_disjointness,
    validate_duplicate_column_equality,
    validate_keys_uniqueness,
    validate_schema_types,
    validate_target_presence,
)


def test_tiny_fixtures_contract(tiny_lead_tables: dict[str, pl.DataFrame]):
    train = tiny_lead_tables["train"]
    test = tiny_lead_tables["test"]
    sample_sub = tiny_lead_tables["sample_submission"]

    # Target presence in train / absence in test
    assert "anomaly" in train.columns
    assert "anomaly" not in test.columns
    assert "anomaly" in sample_sub.columns

    # Expected key columns
    assert "building_id" in train.columns
    assert "timestamp" in train.columns
    assert "row_id" in test.columns
    assert "row_id" in sample_sub.columns


def test_build_feature_joins_success(tiny_lead_tables: dict[str, pl.DataFrame]):
    train_joined_lf, test_joined_lf = build_feature_joins(tiny_lead_tables)
    assert isinstance(train_joined_lf, pl.LazyFrame), "build_feature_joins must return LazyFrame for train"
    assert isinstance(test_joined_lf, pl.LazyFrame), "build_feature_joins must return LazyFrame for test"

    train_joined = train_joined_lf.collect()
    test_joined = test_joined_lf.collect()

    # Unchanged row counts
    assert len(train_joined) == len(tiny_lead_tables["train"])
    assert len(test_joined) == len(tiny_lead_tables["test"])

    # Target presence / absence
    assert "anomaly" in train_joined.columns
    assert "anomaly" not in test_joined.columns

    # Redundant duplicate copies dropped
    assert "temperature_mean" in train_joined.columns
    assert "temperature_mean" in test_joined.columns
    assert "meter_reading_right" not in train_joined.columns
    assert "anomaly_right" not in train_joined.columns

    # Train/test building disjointness
    train_buildings = set(train_joined.get_column("building_id").unique().to_list())
    test_buildings = set(test_joined.get_column("building_id").unique().to_list())
    assert train_buildings.isdisjoint(test_buildings)

    # Order preserved matching sample submission
    expected_order = tiny_lead_tables["sample_submission"].get_column("row_id").to_list()
    assert test_joined.get_column("row_id").to_list() == expected_order


def test_build_feature_joins_with_test_feature_duplicate_columns(
    tiny_lead_tables_with_duplicate_test_features: dict[str, pl.DataFrame],
):
    _train_joined_lf, test_joined_lf = build_feature_joins(tiny_lead_tables_with_duplicate_test_features)
    assert isinstance(test_joined_lf, pl.LazyFrame), "build_feature_joins must return LazyFrame for test"
    test_joined = test_joined_lf.collect()

    assert len(test_joined) == len(tiny_lead_tables_with_duplicate_test_features["test"])
    assert "meter_reading_right" not in test_joined.columns
    assert "building_id_right" not in test_joined.columns
    assert "timestamp_right" not in test_joined.columns
    expected_order = tiny_lead_tables_with_duplicate_test_features["sample_submission"].get_column("row_id").to_list()
    assert test_joined.get_column("row_id").to_list() == expected_order


def test_join_rejects_duplicate_training_keys(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    tables["train"] = pl.concat([tables["train"], tables["train"].head(1)])
    with pytest.raises(ValueError, match="train key.*not unique"):
        build_feature_joins(tables)


def test_join_rejects_duplicate_test_keys(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    tables["test"] = pl.concat([tables["test"], tables["test"].head(1)])
    with pytest.raises(ValueError, match="test key.*not unique"):
        build_feature_joins(tables)


def test_join_rejects_target_in_test(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    tables["test"] = tables["test"].with_columns(pl.lit(0).alias("anomaly"))
    with pytest.raises(ValueError, match="Target column 'anomaly' must not be present in test"):
        build_feature_joins(tables)


def test_join_rejects_missing_train_target(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    tables["train"] = tables["train"].drop("anomaly")
    with pytest.raises(ValueError, match="Target column 'anomaly' missing from train"):
        build_feature_joins(tables)


def test_join_rejects_train_key_set_mismatch(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    # Drop one row from train_features to create key set mismatch
    tables["train_features"] = tables["train_features"].slice(1)
    with pytest.raises(ValueError, match="Key set mismatch between train and train_features"):
        build_feature_joins(tables)


def test_join_rejects_test_key_set_mismatch(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    # Drop one row from test_features to create key set mismatch
    tables["test_features"] = tables["test_features"].slice(1)
    with pytest.raises(ValueError, match="Key set mismatch between test and test_features"):
        build_feature_joins(tables)


def test_join_rejects_differing_duplicate_meter_reading(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    # Corrupt meter_reading in train_features
    corrupted_meter = tables["train_features"].get_column("meter_reading") + 999.0
    tables["train_features"] = tables["train_features"].with_columns(corrupted_meter.alias("meter_reading"))
    with pytest.raises(ValueError, match="Duplicate column 'meter_reading' values in train_features do not match train"):
        build_feature_joins(tables)


def test_join_rejects_differing_duplicate_anomaly(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    # Flip anomaly in train_features
    flipped_anomaly = 1 - tables["train_features"].get_column("anomaly")
    tables["train_features"] = tables["train_features"].with_columns(flipped_anomaly.alias("anomaly"))
    with pytest.raises(ValueError, match="Duplicate column 'anomaly' values in train_features do not match train"):
        build_feature_joins(tables)


def test_join_rejects_differing_test_duplicate_meter_reading(
    tiny_lead_tables_with_duplicate_test_features: dict[str, pl.DataFrame],
):
    tables = dict(tiny_lead_tables_with_duplicate_test_features)
    corrupted_meter = tables["test_features"].get_column("meter_reading") + 10.0
    tables["test_features"] = tables["test_features"].with_columns(corrupted_meter.alias("meter_reading"))
    with pytest.raises(ValueError, match="Duplicate column 'meter_reading' values in test_features do not match test"):
        build_feature_joins(tables)


def test_join_rejects_overlapping_train_test_buildings(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    # Set one test building to overlap with train building 0
    tables["test"] = tables["test"].with_columns(
        pl.Series("building_id", [0, 101])
    )
    with pytest.raises(ValueError, match="Train and test share .* building_id"):
        build_feature_joins(tables)


def test_validate_keys_uniqueness_raises_on_duplicate():
    df = pl.DataFrame({"b": [1, 1], "t": ["2022-01-01", "2022-01-01"]})
    with pytest.raises(ValueError, match="b, t.*not unique"):
        validate_keys_uniqueness(df, ["b", "t"], "sample_table")


def test_validate_target_presence_checks():
    df_with_target = pl.DataFrame({"anomaly": [0, 1]})
    df_without_target = pl.DataFrame({"other": [1, 2]})

    # Expect present, is present -> ok
    validate_target_presence(df_with_target, "anomaly", should_be_present=True, table_name="train")

    # Expect absent, is absent -> ok
    validate_target_presence(df_without_target, "anomaly", should_be_present=False, table_name="test")

    # Expect present, is absent -> raises
    with pytest.raises(ValueError, match="Target column 'anomaly' missing from train"):
        validate_target_presence(df_without_target, "anomaly", should_be_present=True, table_name="train")

    # Expect absent, is present -> raises
    with pytest.raises(ValueError, match="Target column 'anomaly' must not be present in test"):
        validate_target_presence(df_with_target, "anomaly", should_be_present=False, table_name="test")


def test_validate_building_disjointness_checks():
    train = pl.DataFrame({"building_id": [1, 2, 3]})
    test_ok = pl.DataFrame({"building_id": [4, 5, 6]})
    test_overlap = pl.DataFrame({"building_id": [3, 4, 5]})

    validate_building_disjointness(train, test_ok)

    with pytest.raises(ValueError, match="overlap"):
        validate_building_disjointness(train, test_overlap)


def test_validate_duplicate_column_equality_checks():
    df1 = pl.DataFrame({"k": [1, 2], "val": [10.0, None]})
    df2 = pl.DataFrame({"k": [1, 2], "val": [10.0, None]})
    df_diff = pl.DataFrame({"k": [1, 2], "val": [10.0, 5.0]})

    validate_duplicate_column_equality(df1, df2, "val", ["k"], "df1", "df2")

    with pytest.raises(ValueError, match="do not match"):
        validate_duplicate_column_equality(df1, df_diff, "val", ["k"], "df1", "df_diff")


def test_compute_file_sha256_and_count_rows(tmp_path: Path):
    test_file = tmp_path / "sample.csv"
    test_file.write_text("a,b\n1,2\n3,4\n")

    assert count_file_rows(test_file) == 2
    sha = compute_file_sha256(test_file)
    assert isinstance(sha, str)
    assert len(sha) == 64


def test_count_file_rows_with_quoted_newline_and_no_final_newline(tmp_path: Path):
    # 1. Valid CSV without trailing newline
    no_trailing = tmp_path / "no_trailing.csv"
    no_trailing.write_text("col1,col2\nval1,val2")
    assert count_file_rows(no_trailing) == 1

    # 2. Valid CSV with quoted multiline field
    quoted = tmp_path / "quoted.csv"
    quoted.write_text('col1,col2\n"line1\nline2\nline3",val2\n')
    assert count_file_rows(quoted) == 1


def test_validate_schema_types_fails_on_missing_column():
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Configured type column 'b' missing from table"):
        validate_schema_types(df, {"b": "Int64"}, table_name="table")


def test_validate_schema_types_fails_on_unsupported_type_name():
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Unsupported expected type 'UnregisteredType'"):
        validate_schema_types(df, {"a": "UnregisteredType"}, table_name="table")


def test_load_data_contract_reads_yaml():
    contract = load_data_contract()
    assert "raw_files" in contract
    assert "train.csv" in contract["raw_files"]
    assert "test.csv" in contract["raw_files"]
    assert contract["raw_files"]["train.csv"]["row_count"] == 1749494
    assert contract["raw_files"]["test.csv"]["row_count"] == 1800567


def test_scan_raw_table_reads_lazy(tmp_path: Path):
    test_csv = tmp_path / "train.csv"
    test_csv.write_text("building_id,timestamp,meter_reading,anomaly\n1,2022-01-01 00:00:00,10.5,0\n")
    lf = scan_raw_table(test_csv)
    assert isinstance(lf, pl.LazyFrame)
    schema = lf.collect_schema()
    assert schema["building_id"] == pl.Int64
    assert schema["timestamp"] == pl.Datetime(time_unit="us")
    assert schema["meter_reading"] == pl.Float64
    assert schema["anomaly"] == pl.Int64


def test_prepare_data_script_cli(
    tiny_lead_csv_dir: Path, tmp_path: Path, tiny_lead_tables: dict[str, pl.DataFrame]
):
    import json
    import subprocess
    import sys

    # Test invoking scripts/prepare_data.py
    script_path = Path("scripts/prepare_data.py").resolve()
    assert script_path.is_file(), "scripts/prepare_data.py does not exist yet"

    out_dir = tmp_path / "outputs"

    # Create mini-contract
    mini_contract = tmp_path / "mini_contract.yaml"
    import hashlib

    def get_hash(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    contract_content = f"""
raw_files:
  train.csv:
    sha256: "{get_hash(tiny_lead_csv_dir / 'train.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train.csv')}
    primary_keys: ["building_id", "timestamp"]
    required_columns: ["building_id", "timestamp", "meter_reading", "anomaly"]
  test.csv:
    sha256: "{get_hash(tiny_lead_csv_dir / 'test.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test.csv')}
    primary_keys: ["row_id"]
    required_columns: ["row_id", "building_id", "timestamp", "meter_reading"]
  sample_submission.csv:
    sha256: "{get_hash(tiny_lead_csv_dir / 'sample_submission.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'sample_submission.csv')}
    primary_keys: ["row_id"]
    required_columns: ["row_id", "anomaly"]
  train_features.csv:
    sha256: "{get_hash(tiny_lead_csv_dir / 'train_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train_features.csv')}
    primary_keys: ["building_id", "timestamp"]
    duplicate_feature_copies: ["meter_reading", "anomaly"]
    required_columns: ["building_id", "timestamp", "meter_reading", "anomaly", "temperature_mean"]
  test_features.csv:
    sha256: "{get_hash(tiny_lead_csv_dir / 'test_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test_features.csv')}
    primary_keys: ["row_id"]
    duplicate_feature_copies: ["building_id", "timestamp", "meter_reading"]
    required_columns: ["row_id", "building_id", "timestamp", "meter_reading", "temperature_mean"]
"""
    mini_contract.write_text(contract_content)

    # 1. Neither flag provided -> exit code 1 or error
    res_none = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--project-root",
            str(tmp_path),
            "--data-dir",
            str(tiny_lead_csv_dir),
            "--output-dir",
            str(out_dir),
            "--contract",
            str(mini_contract),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_none.returncode != 0

    # 2. Smoke mode with 2 rows
    res_smoke = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--smoke",
            "--sample-rows",
            "2",
            "--project-root",
            str(tmp_path),
            "--data-dir",
            str(tiny_lead_csv_dir),
            "--output-dir",
            str(out_dir),
            "--contract",
            str(mini_contract),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_smoke.returncode == 0, f"Smoke run failed: {res_smoke.stderr}"
    assert "Smoke completed successfully" in res_smoke.stdout

    # In smoke mode: row_lineage_smoke.json should exist, but no parquet in prepared and no row_lineage.json
    smoke_lineage_file = out_dir / "lineage" / "row_lineage_smoke.json"
    assert smoke_lineage_file.is_file()
    assert not (out_dir / "lineage" / "row_lineage.json").exists()
    lineage_data = json.loads(smoke_lineage_file.read_text())
    assert lineage_data["mode"] == "smoke"
    assert lineage_data["sample_rows"] == 2
    assert lineage_data["validations"]["unique_keys_validated"] is True
    assert lineage_data["validations"]["one_to_one_joins_validated"] is True
    assert lineage_data["validations"]["train_test_building_overlap"] == 0

    prepared_dir = out_dir / "prepared"
    train_parquet = prepared_dir / "train_joined.parquet"
    test_parquet = prepared_dir / "test_joined.parquet"
    assert not train_parquet.exists()
    assert not test_parquet.exists()

    # 3. Materialize mode with custom mini-contract
    mini_contract = tmp_path / "mini_contract.yaml"

    def get_hash(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    res_mat = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--materialize",
            "--project-root",
            str(tmp_path),
            "--data-dir",
            str(tiny_lead_csv_dir),
            "--output-dir",
            str(out_dir),
            "--contract",
            str(mini_contract),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_mat.returncode == 0, f"Materialize run failed: {res_mat.stderr}"
    assert train_parquet.is_file()
    assert test_parquet.is_file()

    # Verify materialized parquet files
    tr_df = pl.read_parquet(train_parquet)
    ts_df = pl.read_parquet(test_parquet)
    assert len(tr_df) == 20
    assert len(ts_df) == 2
    assert "temperature_mean" in tr_df.columns
    assert "temperature_mean" in ts_df.columns
    # Minor 5: Assert persisted test parquet row_id sequence matches sample submission
    assert ts_df.get_column("row_id").to_list() == [910, 110]
    assert (
        ts_df.get_column("row_id").to_list()
        == tiny_lead_tables["sample_submission"].get_column("row_id").to_list()
    )

    # Verify updated lineage file
    materialize_lineage_file = out_dir / "lineage" / "row_lineage.json"
    assert materialize_lineage_file.is_file()
    lineage_data_mat = json.loads(materialize_lineage_file.read_text())
    assert lineage_data_mat["mode"] == "materialize"
    assert lineage_data_mat["validations"]["hashes_validated"] is True
    assert lineage_data_mat["validations"]["row_counts_validated"] is True
    assert lineage_data_mat["joined_tables"]["train"]["output_path"] == str(train_parquet)
    assert lineage_data_mat["joined_tables"]["test"]["output_path"] == str(test_parquet)
    assert "sha256" in lineage_data_mat["joined_tables"]["train"]
    assert "sha256" in lineage_data_mat["joined_tables"]["test"]


def test_lazy_join_returns_lazy_and_sinks_streaming(tiny_lead_tables: dict[str, pl.DataFrame], tmp_path: Path):
    lazy_tables = {k: v.lazy() for k, v in tiny_lead_tables.items()}
    train_joined_lf, test_joined_lf = build_feature_joins(lazy_tables)

    assert isinstance(train_joined_lf, pl.LazyFrame), "Expected LazyFrame output for train"
    assert isinstance(test_joined_lf, pl.LazyFrame), "Expected LazyFrame output for test"

    train_sink_path = tmp_path / "train_sink.parquet"
    test_sink_path = tmp_path / "test_sink.parquet"
    train_joined_lf.sink_parquet(train_sink_path, engine="streaming")
    test_joined_lf.sink_parquet(test_sink_path, engine="streaming", maintain_order=True)

    assert train_sink_path.is_file()
    assert test_sink_path.is_file()
    assert pl.read_parquet(train_sink_path).shape == (20, 5)

    test_sink_df = pl.read_parquet(test_sink_path)
    assert test_sink_df.shape == (2, 5)
    # Assert exact row_id sequence equality with sample submission
    assert (
        test_sink_df.get_column("row_id").to_list()
        == tiny_lead_tables["sample_submission"].get_column("row_id").to_list()
    )


def test_join_rejects_missing_declared_duplicate_copy(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    # Drop declared duplicate meter_reading from train_features
    tables["train_features"] = tables["train_features"].drop("meter_reading")
    with pytest.raises(ValueError, match="missing declared duplicate copy.*meter_reading"):
        build_feature_joins(tables)


def test_join_rejects_missing_test_declared_duplicate_copy(tiny_lead_tables: dict[str, pl.DataFrame]):
    tables = dict(tiny_lead_tables)
    # Drop declared duplicate meter_reading from test_features
    tables["test_features"] = tables["test_features"].drop("meter_reading")
    with pytest.raises(ValueError, match="missing declared duplicate copy.*meter_reading"):
        build_feature_joins(tables)


def test_prepare_data_rejects_nonpositive_sample_rows():
    import subprocess
    import sys

    script_path = Path("scripts/prepare_data.py").resolve()
    for bad_n in ["0", "-1", "-100"]:
        res = subprocess.run(
            [sys.executable, str(script_path), "--smoke", "--sample-rows", bad_n],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode != 0, f"Expected nonpositive sample-rows={bad_n} to fail"
        assert "sample-rows must be a positive integer" in (res.stderr + res.stdout)


def test_prepare_data_rejects_output_dir_outside_outputs():
    import subprocess
    import sys

    script_path = Path("scripts/prepare_data.py").resolve()
    res = subprocess.run(
        [sys.executable, str(script_path), "--smoke", "--output-dir", "/tmp/unauthorized_escape"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "must be within" in (res.stderr + res.stdout)


def test_prepare_data_lineage_contains_schemas_dropped_duplicates_and_qualified_memory(
    tiny_lead_csv_dir: Path, tmp_path: Path
):
    import json
    import subprocess
    import sys

    script_path = Path("scripts/prepare_data.py").resolve()
    # Output dir must be within outputs/
    proj_root = tmp_path / "mock_project"
    proj_root.mkdir()
    out_dir = proj_root / "outputs"
    out_dir.mkdir()

    mini_contract = tmp_path / "mini_contract.yaml"
    contract_content = f"""
raw_files:
  train.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train.csv')}
    primary_keys: ["building_id", "timestamp"]
    required_columns: ["building_id", "timestamp", "meter_reading", "anomaly"]
  test.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test.csv')}
    primary_keys: ["row_id"]
    required_columns: ["row_id", "building_id", "timestamp", "meter_reading"]
  sample_submission.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'sample_submission.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'sample_submission.csv')}
    primary_keys: ["row_id"]
    required_columns: ["row_id", "anomaly"]
  train_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train_features.csv')}
    primary_keys: ["building_id", "timestamp"]
    duplicate_feature_copies: ["meter_reading", "anomaly"]
    required_columns: ["building_id", "timestamp", "meter_reading", "anomaly", "temperature_mean"]
  test_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test_features.csv')}
    primary_keys: ["row_id"]
    duplicate_feature_copies: ["building_id", "timestamp", "meter_reading"]
    required_columns: ["row_id", "building_id", "timestamp", "meter_reading", "temperature_mean"]
"""
    mini_contract.write_text(contract_content)

    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--smoke",
            "--sample-rows",
            "5",
            "--project-root",
            str(proj_root),
            "--data-dir",
            str(tiny_lead_csv_dir),
            "--output-dir",
            str(out_dir),
            "--contract",
            str(mini_contract),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Run failed: {res.stderr}"

    # Must NOT claim safe full materialization based on smoke
    assert "Full materialization is estimated safe based on Polars performance" not in (res.stdout + res.stderr)

    lineage_file = out_dir / "lineage" / "row_lineage_smoke.json"
    lineage = json.loads(lineage_file.read_text())

    assert "source_schemas" in lineage
    assert "train.csv" in lineage["source_schemas"]
    assert "dropped_duplicate_columns" in lineage
    assert "train_features" in lineage["dropped_duplicate_columns"]
    assert "test_features" in lineage["dropped_duplicate_columns"]
    assert "meter_reading" in lineage["dropped_duplicate_columns"]["train_features"]

    assert "python_traced_peak_memory_mb" in lineage["resource_metrics"]
    assert "memory_note" in lineage["resource_metrics"]


def test_prepare_data_rejects_contract_raw_files_coverage_mismatch(tiny_lead_csv_dir: Path, tmp_path: Path):
    import subprocess
    import sys

    script_path = Path("scripts/prepare_data.py").resolve()
    bad_contract = tmp_path / "bad_contract.yaml"
    bad_contract.write_text("""
raw_files:
  train.csv:
    sha256: "dummy"
    row_count: 1
""")
    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--smoke",
            "--project-root",
            str(tmp_path),
            "--data-dir",
            str(tiny_lead_csv_dir),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--contract",
            str(bad_contract),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "must exactly match expected files" in (res.stderr + res.stdout)


def test_prepare_data_atomic_publish_rollback_on_failure(tiny_lead_csv_dir: Path, tmp_path: Path):
    import importlib.util
    import json
    from unittest.mock import patch

    spec = importlib.util.spec_from_file_location("prepare_data", "scripts/prepare_data.py")
    prepare_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_data)

    proj_root = tmp_path / "mock_project"
    proj_root.mkdir()
    out_dir = proj_root / "outputs"
    prepared_dir = out_dir / "prepared"
    lineage_dir = out_dir / "lineage"
    prepared_dir.mkdir(parents=True)
    lineage_dir.mkdir(parents=True)

    dummy_train = prepared_dir / "train_joined.parquet"
    dummy_test = prepared_dir / "test_joined.parquet"
    dummy_lineage = lineage_dir / "row_lineage.json"

    dummy_train.write_text("old_train")
    dummy_test.write_text("old_test")
    dummy_lineage.write_text(json.dumps({"old": "lineage"}))

    mini_contract = tmp_path / "mini_contract.yaml"
    contract_content = f"""
raw_files:
  train.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train.csv')}
    primary_keys: ["building_id", "timestamp"]
    required_columns: ["building_id", "timestamp", "meter_reading", "anomaly"]
  test.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test.csv')}
    primary_keys: ["row_id"]
    required_columns: ["row_id", "building_id", "timestamp", "meter_reading"]
  sample_submission.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'sample_submission.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'sample_submission.csv')}
    primary_keys: ["row_id"]
    required_columns: ["row_id", "anomaly"]
  train_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train_features.csv')}
    primary_keys: ["building_id", "timestamp"]
    duplicate_feature_copies: ["meter_reading", "anomaly"]
    required_columns: ["building_id", "timestamp", "meter_reading", "anomaly", "temperature_mean"]
  test_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test_features.csv')}
    primary_keys: ["row_id"]
    duplicate_feature_copies: ["building_id", "timestamp", "meter_reading"]
    required_columns: ["row_id", "building_id", "timestamp", "meter_reading", "temperature_mean"]
"""
    mini_contract.write_text(contract_content)

    import os
    original_replace = os.replace
    promote_attempted = False

    def faulty_replace(src, dst):
        nonlocal promote_attempted
        # Fail specifically when promoting staged test parquet to final destination
        if ".staging_" in str(src) and "test_joined.parquet" in str(dst):
            promote_attempted = True
            raise OSError("Simulated disk error during atomic publication of test parquet")
        return original_replace(src, dst)

    args = [
        "--materialize",
        "--project-root",
        str(proj_root),
        "--data-dir",
        str(tiny_lead_csv_dir),
        "--output-dir",
        str(out_dir),
        "--contract",
        str(mini_contract),
    ]

    with patch("os.replace", side_effect=faulty_replace):
        ret = prepare_data.main(args)
        assert ret != 0
        assert promote_attempted, "Expected promotion of test parquet to be attempted"

    # Ensure pre-existing artifacts are intact!
    assert dummy_train.read_text() == "old_train"
    assert dummy_test.read_text() == "old_test"
    assert json.loads(dummy_lineage.read_text()) == {"old": "lineage"}


def test_prepare_data_rejects_child_dir_symlink_outside_outputs(tiny_lead_csv_dir: Path, tmp_path: Path):
    import subprocess
    import sys

    script_path = Path("scripts/prepare_data.py").resolve()
    proj_root = tmp_path / "mock_project"
    proj_root.mkdir()
    out_dir = proj_root / "outputs"
    out_dir.mkdir()

    # Create an external directory outside outputs
    external_dir = tmp_path / "unauthorized_external"
    external_dir.mkdir()

    # Symlink prepared/ to external_dir
    symlink_prepared = out_dir / "prepared"
    symlink_prepared.symlink_to(external_dir, target_is_directory=True)

    mini_contract = tmp_path / "mini_contract.yaml"
    mini_contract.write_text(f"""
raw_files:
  train.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train.csv')}
  test.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test.csv')}
  sample_submission.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'sample_submission.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'sample_submission.csv')}
  train_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train_features.csv')}
  test_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test_features.csv')}
""")

    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--smoke",
            "--sample-rows",
            "2",
            "--project-root",
            str(proj_root),
            "--data-dir",
            str(tiny_lead_csv_dir),
            "--output-dir",
            str(out_dir),
            "--contract",
            str(mini_contract),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "outside" in (res.stderr + res.stdout).lower() or "symlink" in (res.stderr + res.stdout).lower()
    # Ensure nothing was written into external_dir
    assert list(external_dir.iterdir()) == []


def test_prepare_data_smoke_preserves_full_lineage(tiny_lead_csv_dir: Path, tmp_path: Path):
    import json
    import subprocess
    import sys

    script_path = Path("scripts/prepare_data.py").resolve()
    proj_root = tmp_path / "mock_project"
    proj_root.mkdir()
    out_dir = proj_root / "outputs"
    lineage_dir = out_dir / "lineage"
    lineage_dir.mkdir(parents=True)

    sentinel_lineage = lineage_dir / "row_lineage.json"
    sentinel_lineage.write_text(json.dumps({"mode": "materialize", "sentinel": "untouched"}))

    mini_contract = tmp_path / "mini_contract.yaml"
    mini_contract.write_text(f"""
raw_files:
  train.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train.csv')}
  test.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test.csv')}
  sample_submission.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'sample_submission.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'sample_submission.csv')}
  train_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train_features.csv')}
  test_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test_features.csv')}
""")

    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--smoke",
            "--sample-rows",
            "2",
            "--project-root",
            str(proj_root),
            "--data-dir",
            str(tiny_lead_csv_dir),
            "--output-dir",
            str(out_dir),
            "--contract",
            str(mini_contract),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Smoke failed: {res.stderr}"

    # row_lineage.json must remain UNCHANGED
    assert json.loads(sentinel_lineage.read_text()) == {"mode": "materialize", "sentinel": "untouched"}

    # Smoke lineage must be written to row_lineage_smoke.json
    smoke_lineage = lineage_dir / "row_lineage_smoke.json"
    assert smoke_lineage.is_file()
    assert json.loads(smoke_lineage.read_text())["mode"] == "smoke"


def test_prepare_data_rollback_failure_preserves_recovery_artifacts(tiny_lead_csv_dir: Path, tmp_path: Path):
    import importlib.util
    import json
    from unittest.mock import patch

    spec = importlib.util.spec_from_file_location("prepare_data", "scripts/prepare_data.py")
    prepare_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_data)

    proj_root = tmp_path / "mock_project"
    proj_root.mkdir()
    out_dir = proj_root / "outputs"
    prepared_dir = out_dir / "prepared"
    lineage_dir = out_dir / "lineage"
    prepared_dir.mkdir(parents=True)
    lineage_dir.mkdir(parents=True)

    dummy_train = prepared_dir / "train_joined.parquet"
    dummy_test = prepared_dir / "test_joined.parquet"
    dummy_lineage = lineage_dir / "row_lineage.json"

    dummy_train.write_text("old_train")
    dummy_test.write_text("old_test")
    dummy_lineage.write_text(json.dumps({"old": "lineage"}))

    mini_contract = tmp_path / "mini_contract.yaml"
    mini_contract.write_text(f"""
raw_files:
  train.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train.csv')}
  test.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test.csv')}
  sample_submission.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'sample_submission.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'sample_submission.csv')}
  train_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'train_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'train_features.csv')}
  test_features.csv:
    sha256: "{compute_file_sha256(tiny_lead_csv_dir / 'test_features.csv')}"
    row_count: {count_file_rows(tiny_lead_csv_dir / 'test_features.csv')}
""")

    import os
    original_replace = os.replace

    def faulty_replace(src, dst):
        # 1. Fail during promotion of test_joined.parquet
        if ".staging_" in str(src) and "test_joined.parquet" in str(dst):
            raise OSError("Promotion failure on test parquet")
        # 2. Also fail during rollback restore of train_joined.parquet from backup
        if ".bak_" in str(src) and "train_joined.parquet" in str(dst):
            raise OSError("Restore failure on train backup")
        return original_replace(src, dst)

    args = [
        "--materialize",
        "--project-root",
        str(proj_root),
        "--data-dir",
        str(tiny_lead_csv_dir),
        "--output-dir",
        str(out_dir),
        "--contract",
        str(mini_contract),
    ]

    import io
    err_capture = io.StringIO()
    with patch("os.replace", side_effect=faulty_replace), patch("sys.stderr", err_capture):
        ret = prepare_data.main(args)
        assert ret != 0

    err_output = err_capture.getvalue()
    assert "recovery" in err_output.lower() or "bak" in err_output.lower()
    # Check that backup or staging directory remains available on disk
    backups = list(prepared_dir.glob("*.bak*"))
    staging_dirs = list(out_dir.glob(".staging_*"))
    assert len(backups) > 0 or len(staging_dirs) > 0, "Expected recovery artifacts to be preserved on rollback failure"




