from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import pathlib
from datetime import datetime
from typing import Any

import polars as pl
import pytest
import yaml


def _load_train_module():
    """Dynamically load the scripts.train module.

    Must be loaded inside test bodies so test collection succeeds even when
    scripts/train.py has not yet been authored.
    """
    try:
        return importlib.import_module("scripts.train")
    except ModuleNotFoundError:
        script_path = (
            pathlib.Path(__file__).parents[2] / "scripts" / "train.py"
        )
        if not script_path.is_file():
            pytest.fail(f"scripts/train.py is not present at {script_path}")
        spec = importlib.util.spec_from_file_location("scripts.train", script_path)
        if spec is None or spec.loader is None:
            pytest.fail(f"Could not load module spec for {script_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _load_training_validator():
    """Dynamically load validate_baseline_training_frame from scripts.train.

    Must be loaded inside test bodies so test collection succeeds even when
    scripts/train.py has not yet been authored.
    """
    mod = _load_train_module()
    fn = getattr(mod, "validate_baseline_training_frame", None)
    if fn is None:
        pytest.fail("scripts.train does not define validate_baseline_training_frame")
    return fn


def _load_baseline_config() -> dict[str, Any]:
    """Load baseline experiment configuration."""
    config_path = (
        pathlib.Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "baselines.yaml"
    )
    assert config_path.exists(), f"baselines.yaml must exist at {config_path}"
    with config_path.open() as fh:
        config = yaml.safe_load(fh)
    assert isinstance(config, dict), "baselines.yaml must parse as a mapping"
    return config


def _build_valid_training_frame() -> pl.DataFrame:
    """Build a deterministic synthetic Polars training frame with all 9 approved features and keys.

    Approved design calendar semantics:
      - weekday is 0=Monday through 6=Sunday
      - hour (0-23), weekday (0-6), month (1-12), year are derived/checked at observed timestamp
    """
    # 2022-01-03 08:00:00 is Monday -> weekday=0, month=1, year=2022, hour=8
    # 2022-01-04 15:00:00 is Tuesday -> weekday=1, month=1, year=2022, hour=15
    # 2022-05-15 23:00:00 is Sunday -> weekday=6, month=5, year=2022, hour=23
    timestamps = [
        datetime.fromisoformat("2022-01-03T08:00:00"),
        datetime.fromisoformat("2022-01-04T15:00:00"),
        datetime.fromisoformat("2022-05-15T23:00:00"),
    ]
    return pl.DataFrame(
        {
            # Required row identifiers and target
            "building_id": ["bld_1", "bld_1", "bld_2"],
            "timestamp": timestamps,
            "anomaly": [0, 1, 0],
            # 8 approved numeric predictors
            "meter_reading": [105.4, 210.2, 55.0],
            "square_feet": [50000.0, 50000.0, 12000.0],
            "year_built": [1980.0, 1980.0, 2005.0],
            "floor_count": [4.0, 4.0, 2.0],
            "hour": [8, 15, 23],
            "weekday": [0, 1, 6],
            "month": [1, 1, 5],
            "year": [2022, 2022, 2022],
            # 1 approved categorical predictor
            "primary_use": ["Education", "Education", "Office"],
        }
    )


def test_validate_baseline_training_frame_valid_synthetic():
    """Valid synthetic Polars frame returns configured predictor tuple in deterministic order."""
    validator = _load_training_validator()
    config = _load_baseline_config()
    frame = _build_valid_training_frame()

    features = validator(frame, config)

    expected = (
        "meter_reading",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
        "primary_use",
    )
    assert isinstance(features, (tuple, list)), "Returned feature list must be a sequence"
    assert tuple(features) == expected, (
        f"Expected predictor tuple {expected}, got {tuple(features)}"
    )


@pytest.mark.parametrize(
    "missing_feature",
    [
        "meter_reading",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
        "primary_use",
    ],
)
def test_validate_baseline_training_frame_rejects_missing_approved_feature(missing_feature: str):
    """Missing approved predictor causes validation to fail with ValueError."""
    validator = _load_training_validator()
    config = _load_baseline_config()
    frame = _build_valid_training_frame().drop(missing_feature)

    with pytest.raises((ValueError, KeyError), match=rf"{missing_feature}|missing|column"):
        validator(frame, config)


@pytest.mark.parametrize(
    "missing_key",
    [
        "building_id",
        "timestamp",
        "anomaly",
    ],
)
def test_validate_baseline_training_frame_rejects_missing_key_column(missing_key: str):
    """Missing key or target column causes validation to fail with ValueError."""
    validator = _load_training_validator()
    config = _load_baseline_config()
    frame = _build_valid_training_frame().drop(missing_key)

    with pytest.raises((ValueError, KeyError), match=rf"{missing_key}|missing|column"):
        validator(frame, config)


@pytest.mark.parametrize(
    "col,invalid_type",
    [
        ("meter_reading", pl.String),
        ("square_feet", pl.String),
        ("hour", pl.String),
        ("weekday", pl.String),
        ("primary_use", pl.Float64),
        ("primary_use", pl.Int64),
    ],
)
def test_validate_baseline_training_frame_rejects_dtype_mismatch(col: str, invalid_type: pl.DataType):
    """Incompatible numeric or categorical dtypes cause validation to fail with TypeError or ValueError."""
    validator = _load_training_validator()
    config = _load_baseline_config()
    if col == "primary_use":
        frame = _build_valid_training_frame().with_columns(
            pl.Series(col, [1, 2, 3], dtype=invalid_type)
        )
    else:
        frame = _build_valid_training_frame().with_columns(pl.col(col).cast(invalid_type))

    with pytest.raises((TypeError, ValueError), match=rf"{col}|dtype|type|schema"):
        validator(frame, config)


@pytest.mark.parametrize(
    "calendar_col,corrupted_values",
    [
        ("hour", [9, 15, 23]),          # row 0 hour should be 8, corrupted to 9
        ("weekday", [1, 1, 6]),       # row 0 weekday should be 0 (Monday), corrupted to 1
        ("weekday", [1, 2, 7]),       # 1-indexed attempt (1=Monday, 7=Sunday) should fail 0-indexed check
        ("month", [2, 1, 5]),         # row 0 month should be 1, corrupted to 2
        ("year", [2021, 2022, 2022]), # row 0 year should be 2022, corrupted to 2021
    ],
)
def test_validate_baseline_training_frame_rejects_calendar_mismatches(
    calendar_col: str, corrupted_values: list[int]
):
    """Calendar component mismatch against observed timestamp causes validation to fail."""
    validator = _load_training_validator()
    config = _load_baseline_config()
    frame = _build_valid_training_frame().with_columns(
        pl.Series(calendar_col, corrupted_values)
    )

    with pytest.raises(ValueError, match=rf"{calendar_col}|calendar|timestamp|mismatch"):
        validator(frame, config)


@pytest.mark.parametrize(
    "unapproved_predictor",
    [
        "gte_probe",
        "gte_vendor_raw",
        "gte_reading_smooth",
    ],
)
def test_validate_baseline_training_frame_rejects_unapproved_gte_in_config(
    unapproved_predictor: str,
):
    """Config attempting to add unapproved gte_* predictor fails closed before fitting."""
    validator = _load_training_validator()
    config = copy.deepcopy(_load_baseline_config())
    config["features"]["numeric"].append(unapproved_predictor)

    frame = _build_valid_training_frame().with_columns(
        pl.Series(unapproved_predictor, [1.0, 2.0, 3.0])
    )

    with pytest.raises(
        ValueError,
        match=rf"{unapproved_predictor}|unapproved|not allowed|provenance|pending|allowlist",
    ):
        validator(frame, config)


@pytest.mark.parametrize(
    "unapproved_predictor",
    [
        "air_temperature",
        "is_holiday",
        "meter_reading_lag_1",
    ],
)
def test_validate_baseline_training_frame_rejects_unapproved_general_predictor_in_config(
    unapproved_predictor: str,
):
    """Config attempting to add other unapproved predictors outside 9 approved fields fails closed."""
    validator = _load_training_validator()
    config = copy.deepcopy(_load_baseline_config())
    config["features"]["numeric"].append(unapproved_predictor)

    frame = _build_valid_training_frame().with_columns(
        pl.Series(unapproved_predictor, [1.0, 2.0, 3.0])
    )

    with pytest.raises(
        ValueError,
        match=rf"{unapproved_predictor}|unapproved|not allowed|allowlist|unverified",
    ):
        validator(frame, config)


def test_run_rejects_non_mapping_config(tmp_path: pathlib.Path):
    """Baseline training run fails closed with TypeError when YAML config is not a mapping."""
    mod = _load_train_module()
    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "baselines.yaml"
    config_path.write_text("- item1\n- item2\n", encoding="utf-8")

    args = argparse.Namespace(
        config=pathlib.Path("configs/experiments/baselines.yaml"),
        project_root=tmp_path,
        smoke=True,
        full=False,
        max_buildings=8,
    )
    with pytest.raises(TypeError, match="mapping"):
        mod._run(args)


def test_run_rejects_unapproved_seed(tmp_path: pathlib.Path):
    """Baseline training run fails closed with ValueError when seed differs from frozen 42."""
    mod = _load_train_module()
    config = copy.deepcopy(_load_baseline_config())
    config["seed"] = 123

    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "baselines.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    args = argparse.Namespace(
        config=pathlib.Path("configs/experiments/baselines.yaml"),
        project_root=tmp_path,
        smoke=True,
        full=False,
        max_buildings=8,
    )
    with pytest.raises(ValueError, match=r"(?i)seed.*42"):
        mod._run(args)


@pytest.mark.parametrize(
    "candidate_id,mutated_params",
    [
        ("dummy_prior", {"strategy": "uniform"}),
        (
            "logistic_l2",
            {
                "C": 2.0,
                "l1_ratio": 0.0,
                "class_weight": None,
                "max_iter": 1000,
            },
        ),
        ("hist_gradient_boosting", {"early_stopping": True}),
    ],
)
def test_run_rejects_mutated_candidate_params(
    tmp_path: pathlib.Path, candidate_id: str, mutated_params: dict[str, Any]
):
    """_run rejects configs with mutated baseline params before accessing prepared data."""
    mod = _load_train_module()
    config = copy.deepcopy(_load_baseline_config())
    config["models"][candidate_id]["params"] = mutated_params

    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "baselines.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    args = argparse.Namespace(
        config=pathlib.Path("configs/experiments/baselines.yaml"),
        project_root=tmp_path,
        smoke=True,
        full=False,
        max_buildings=8,
    )
    with pytest.raises(ValueError, match=rf"{candidate_id}.*params"):
        mod._run(args)


@pytest.mark.parametrize(
    "candidate_id",
    [
        "dummy_prior",
        "logistic_l2",
        "hist_gradient_boosting",
    ],
)
def test_run_rejects_non_mapping_candidate_params(
    tmp_path: pathlib.Path, candidate_id: str
):
    """_run rejects non-mapping candidate params before accessing prepared data."""
    mod = _load_train_module()
    config = copy.deepcopy(_load_baseline_config())
    config["models"][candidate_id]["params"] = "invalid_string_params"

    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "baselines.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    args = argparse.Namespace(
        config=pathlib.Path("configs/experiments/baselines.yaml"),
        project_root=tmp_path,
        smoke=True,
        full=False,
        max_buildings=8,
    )
    with pytest.raises(TypeError, match=rf"{candidate_id}.*params"):
        mod._run(args)


def _setup_synthetic_experiment_environment(tmp_path: pathlib.Path) -> argparse.Namespace:
    """Create tiny synthetic parquet data and manifest for runner-level tests."""
    config = copy.deepcopy(_load_baseline_config())
    config["split_manifest"] = "outputs/splits/building_folds.parquet"

    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "baselines.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    splits_dir = tmp_path / "outputs" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifest = pl.DataFrame(
        {
            "building_id": ["bld_0", "bld_1", "bld_2", "bld_3", "bld_vault"],
            "split": ["dev", "dev", "dev", "dev", "vault"],
            "fold": [0, 1, 2, 3, None],
        },
        schema={
            "building_id": pl.String,
            "split": pl.String,
            "fold": pl.Int64,
        },
    )
    manifest.write_parquet(splits_dir / "building_folds.parquet")

    rows: list[dict[str, Any]] = []
    base_time = datetime.fromisoformat("2022-01-03T08:00:00")
    for building_id in ("bld_0", "bld_1", "bld_2", "bld_3", "bld_vault"):
        for label in (0, 1):
            ts = base_time.replace(hour=8 + label)
            rows.append(
                {
                    "building_id": building_id,
                    "timestamp": ts,
                    "anomaly": label,
                    "meter_reading": 100.0 + label,
                    "square_feet": 50000.0,
                    "year_built": 1980.0,
                    "floor_count": 4.0,
                    "hour": ts.hour,
                    "weekday": 0,
                    "month": 1,
                    "year": 2022,
                    "primary_use": "Education" if label == 0 else "Office",
                }
            )
    prepared_dir = tmp_path / "outputs" / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    train_frame = pl.DataFrame(rows)
    train_frame.write_parquet(prepared_dir / "train_joined.parquet")

    return argparse.Namespace(
        config=pathlib.Path("configs/experiments/baselines.yaml"),
        project_root=tmp_path,
        smoke=True,
        full=False,
        max_buildings=8,
    )


def test_run_success_writes_completed_status_and_artifacts(tmp_path: pathlib.Path):
    """Successful _run writes run_status completed, expected artifacts, and inventory records."""
    mod = _load_train_module()
    args = _setup_synthetic_experiment_environment(tmp_path)

    exit_code = mod._run(args)
    assert exit_code == 0

    runs_dir = tmp_path / "outputs" / "runs"
    assert runs_dir.is_dir()
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run_id = run_dir.name

    # Verify run_status.json
    status_path = run_dir / "run_status.json"
    assert status_path.is_file()
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_data == {"run_id": run_id, "status": "completed"}

    # Verify the five existing expected run artifacts
    expected_artifacts = [
        "baselines.yaml",
        "oof_predictions.parquet",
        "metrics.json",
        "resource_profile.json",
        "model_card.md",
    ]
    for artifact_name in expected_artifacts:
        assert (run_dir / artifact_name).is_file(), f"Missing expected artifact {artifact_name}"

    # Verify model card includes status sentinel instruction
    model_card_text = (run_dir / "model_card.md").read_text(encoding="utf-8")
    assert "run_status.json" in model_card_text
    assert "completed" in model_card_text

    # Verify exactly 3 inventory records reference this run_id
    inventory_path = runs_dir / "experiment_inventory.jsonl"
    assert inventory_path.is_file()
    inventory_records = [
        json.loads(line)
        for line in inventory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching_records = [
        record for record in inventory_records if record.get("run_id") == run_id
    ]
    assert len(matching_records) == 3


def test_run_failure_leaves_in_progress_status_sentinel(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Failed _run leaves run_status.json in_progress and does not complete."""
    mod = _load_train_module()
    args = _setup_synthetic_experiment_environment(tmp_path)

    def _failing_fits(*_args: Any, **_kwargs: Any):
        raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(mod, "_run_candidate_fits", _failing_fits)

    with pytest.raises(RuntimeError, match="simulated fit failure"):
        mod._run(args)

    runs_dir = tmp_path / "outputs" / "runs"
    assert runs_dir.is_dir()
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run_id = run_dir.name

    status_path = run_dir / "run_status.json"
    assert status_path.is_file()
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_data == {"run_id": run_id, "status": "in_progress"}

    # Outputs after candidate fits must not have been created
    assert not (run_dir / "oof_predictions.parquet").exists()


def test_run_inventory_failure_leaves_in_progress_status_sentinel(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Inventory failure cannot mark an otherwise complete artifact set as complete."""
    mod = _load_train_module()
    args = _setup_synthetic_experiment_environment(tmp_path)

    def _failing_inventory(*_args: Any, **_kwargs: Any):
        raise RuntimeError("simulated inventory append failure")

    monkeypatch.setattr(mod, "append_inventory_records", _failing_inventory)

    with pytest.raises(RuntimeError, match="simulated inventory append failure"):
        mod._run(args)

    runs_dir = tmp_path / "outputs" / "runs"
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run_id = run_dir.name

    status_data = json.loads(
        (run_dir / "run_status.json").read_text(encoding="utf-8")
    )
    assert status_data == {"run_id": run_id, "status": "in_progress"}
    assert (run_dir / "oof_predictions.parquet").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "resource_profile.json").is_file()
    assert (run_dir / "model_card.md").is_file()

