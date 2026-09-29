from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime, timedelta
from math import log
from pathlib import Path
from types import ModuleType

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def _validation_module() -> ModuleType:
    script_path = Path("scripts/adversarial_validate.py")
    assert script_path.is_file(), "scripts/adversarial_validate.py is required"
    return importlib.import_module("scripts.adversarial_validate")


def _hourly_table(
    domain: int,
    building_ids: range,
    *,
    include_target: bool,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for building_offset, building_id in enumerate(building_ids):
        row_count = 2 + building_offset % 3
        for hour in range(row_count):
            row: dict[str, object] = {
                "building_id": building_id,
                "row_id": building_id * 10 + hour,
                "timestamp": start + timedelta(hours=hour),
                "meter_reading": (
                    None
                    if hour == 1 and building_offset % 2 == 0
                    else float(domain * 10 + building_offset + hour)
                ),
                "meter_type": ("z", "a", "z", "a")[hour],
                "gte_probe": float(domain * 3 + building_offset + hour),
                "fold": building_offset % 5,
                "split": "train" if domain == 0 else "test",
                "domain_label": domain,
                "source_file": "train.parquet" if domain == 0 else "test.parquet",
                "source_meter": float(domain),
            }
            if include_target:
                row["anomaly"] = hour % 2
            rows.append(row)
    return pl.DataFrame(rows)


@pytest.fixture
def prepared_project(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    prepared_dir = project_root / "outputs" / "prepared"
    prepared_dir.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'domain-shift-test'\n", encoding="utf-8"
    )
    _hourly_table(0, range(10), include_target=True).write_parquet(
        prepared_dir / "train_joined.parquet"
    )
    _hourly_table(1, range(100, 110), include_target=False).write_parquet(
        prepared_dir / "test_joined.parquet"
    )
    return project_root, prepared_dir


def _config(module: ModuleType, project_root: Path, prepared_dir: Path):
    return module.ValidationConfig(
        prepared_dir=prepared_dir,
        output_dir=project_root / "outputs" / "shift",
        folds=5,
        seed=42,
        project_root=project_root,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_building_summary_reports_mean_median_missingness_and_stable_mode():
    module = _validation_module()
    hourly = pl.DataFrame(
        {
            "building_id": [1, 1, 1, 2, 2],
            "reading": [1.0, 3.0, None, 5.0, None],
            "meter_type": ["z", "z", "a", "z", "a"],
        }
    )

    selection = module.select_predictor_columns(hourly.schema, hourly.schema)
    summary = module.summarize_buildings(hourly, selection).sort("building_id")

    assert summary.height == 2
    assert summary.get_column("building_id").to_list() == [1, 2]
    assert summary.get_column("hourly_row_count").to_list() == [3, 2]
    assert summary.get_column("reading__mean").to_list() == [2.0, 5.0]
    assert summary.get_column("reading__median").to_list() == [2.0, 5.0]
    assert summary.get_column("reading__missing_rate").to_list() == pytest.approx(
        [1 / 3, 1 / 2]
    )
    assert summary.get_column("meter_type__mode").to_list() == ["z", "a"]
    assert selection.categorical_mode_tie_break == "lexical_ascending"


def test_domain_features_exclude_target_ids_source_split_and_metadata_predictors():
    module = _validation_module()
    train = _hourly_table(0, range(10), include_target=True)
    test = _hourly_table(1, range(100, 110), include_target=False)

    dataset = module.build_domain_dataset(train, test)

    assert dataset.metadata.height == 20
    assert dataset.metadata.get_column("building_id").n_unique() == 20
    assert dataset.metadata.get_column("domain_label").to_list() == [0] * 10 + [1] * 10
    assert dataset.metadata.get_column("hourly_row_count").to_list()[:3] == [2, 3, 4]
    assert "hourly_row_count" not in dataset.features.columns

    forbidden = {
        "building_id",
        "row_id",
        "timestamp",
        "anomaly",
        "fold",
        "split",
        "domain_label",
        "source_file",
        "source_meter",
    }
    assert forbidden.isdisjoint(dataset.features.columns)
    assert set(dataset.features.columns) == {
        "meter_reading__mean",
        "meter_reading__median",
        "meter_reading__missing_rate",
        "meter_type__mode",
        "gte_probe__mean",
        "gte_probe__median",
        "gte_probe__missing_rate",
    }
    assert dataset.selection.numeric_columns == ("gte_probe", "meter_reading")
    assert dataset.selection.categorical_columns == ("meter_type",)
    assert dataset.selection.excluded_fields["anomaly"] == "target"
    assert dataset.selection.excluded_fields["source_file"] == "source_metadata"
    assert "gte_probe" in dataset.selection.gte_shift_only_fields


def test_domain_dataset_rejects_building_overlap_between_train_and_test():
    module = _validation_module()
    train = _hourly_table(0, range(10), include_target=True)
    test = _hourly_table(1, range(100, 110), include_target=False).with_columns(
        pl.when(pl.col("building_id") == 100)
        .then(0)
        .otherwise(pl.col("building_id"))
        .alias("building_id")
    )

    with pytest.raises(ValueError, match="building.*overlap"):
        module.build_domain_dataset(train, test)


def test_oof_run_writes_metrics_baselines_schemas_and_verified_hashes(
    prepared_project: tuple[Path, Path],
):
    module = _validation_module()
    project_root, prepared_dir = prepared_project

    report = module.run_adversarial_validation(
        _config(module, project_root, prepared_dir)
    )

    output_dir = project_root / "outputs" / "shift"
    oof_path = output_dir / "domain_oof.parquet"
    importance_path = output_dir / "fold_averaged_importance.csv"
    report_path = output_dir / "adversarial_validation.json"
    oof = pl.read_parquet(oof_path)
    importance = pl.read_csv(importance_path)
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert oof.columns == [
        "building_id",
        "domain_label",
        "fold",
        "domain_test_probability",
        "hourly_row_count",
    ]
    assert oof.height == 20
    assert oof.get_column("building_id").n_unique() == 20
    assert set(oof.get_column("fold").to_list()) == set(range(5))
    assert np.isfinite(oof.get_column("domain_test_probability").to_numpy()).all()
    assert oof.get_column("domain_test_probability").min() >= 0.0
    assert oof.get_column("domain_test_probability").max() <= 1.0

    assert importance.columns == [
        "feature",
        "importance_mean",
        "importance_std",
        "fold_count",
    ]
    assert importance.get_column("fold_count").to_list() == [5] * importance.height
    assert importance.get_column("feature").to_list() == persisted_report[
        "features"
    ]["classifier_predictors"]

    labels = oof.get_column("domain_label").to_numpy()
    probabilities = oof.get_column("domain_test_probability").to_numpy()
    prevalence = float(np.mean(labels == 1))
    pooled = persisted_report["metrics"]["pooled_oof"]
    assert pooled["roc_auc"] == pytest.approx(roc_auc_score(labels, probabilities))
    assert pooled["average_precision"] == pytest.approx(
        average_precision_score(labels, probabilities)
    )
    assert pooled["log_loss"] == pytest.approx(
        log_loss(labels, probabilities, labels=[0, 1])
    )
    assert len(persisted_report["metrics"]["fold_metrics"]) == 5
    assert set(persisted_report["metrics"]["fold_mean_std"]) == {
        "roc_auc",
        "average_precision",
        "log_loss",
    }
    assert persisted_report["null_baselines"]["domain_prevalence_p"] == prevalence
    assert persisted_report["null_baselines"]["average_precision"] == prevalence
    expected_entropy = -prevalence * log(prevalence) - (1 - prevalence) * log(
        1 - prevalence
    )
    assert persisted_report["null_baselines"]["binary_entropy_log_loss"] == pytest.approx(
        expected_entropy
    )
    assert persisted_report["counts"]["train_buildings"] == 10
    assert persisted_report["counts"]["test_buildings"] == 10
    assert persisted_report["configuration"]["folds"] == 5
    assert persisted_report["configuration"]["seed"] == 42
    assert persisted_report["categorical_mode_tie_break"] == "lexical_ascending"
    assert "diagnostic only" in persisted_report["features"]["importance_use"]
    assert "not production feature selection" in persisted_report["features"]["gte_note"]

    assert report == persisted_report
    for artifact_name, artifact_path in (
        ("domain_oof_parquet", oof_path),
        ("fold_averaged_importance_csv", importance_path),
    ):
        artifact = persisted_report["artifacts"][artifact_name]
        assert Path(artifact["path"]) == artifact_path.resolve()
        assert artifact["sha256"] == _sha256(artifact_path)
    assert Path(persisted_report["artifacts"]["report_json"]["path"]) == report_path.resolve()
    # Serialization contract: every artifact path must be a plain string so that
    # json.dumps(allow_nan=False) never raises TypeError for PosixPath values.
    for artifact_name, artifact_info in report["artifacts"].items():
        assert isinstance(artifact_info["path"], str), (
            f"artifact '{artifact_name}' path must be str, got {type(artifact_info['path']).__name__}"
        )


def test_same_seed_reproduces_oof_predictions_and_artifact_hashes(
    prepared_project: tuple[Path, Path],
):
    module = _validation_module()
    project_root, prepared_dir = prepared_project
    config = _config(module, project_root, prepared_dir)

    first_report = module.run_adversarial_validation(config)
    first_oof = pl.read_parquet(project_root / "outputs" / "shift" / "domain_oof.parquet")
    second_report = module.run_adversarial_validation(config)
    second_oof = pl.read_parquet(project_root / "outputs" / "shift" / "domain_oof.parquet")

    assert first_oof.equals(second_oof)
    assert first_report["metrics"] == second_report["metrics"]
    assert first_report["artifacts"] == second_report["artifacts"]


def test_run_rejects_output_directory_escaping_through_symlink(
    prepared_project: tuple[Path, Path], tmp_path: Path
):
    module = _validation_module()
    project_root, prepared_dir = prepared_project
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    output_dir = project_root / "outputs" / "shift"
    output_dir.symlink_to(external_dir, target_is_directory=True)
    config = module.ValidationConfig(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        folds=5,
        seed=42,
        project_root=project_root,
    )

    with pytest.raises(ValueError, match="outside.*outputs|under.*outputs"):
        module.run_adversarial_validation(config)

    assert list(external_dir.iterdir()) == []


def test_run_requires_both_prepared_parquet_artifacts_and_exactly_five_folds(
    prepared_project: tuple[Path, Path], tmp_path: Path
):
    module = _validation_module()
    project_root, prepared_dir = prepared_project
    missing_dir = tmp_path / "missing-prepared"
    missing_config = module.ValidationConfig(
        prepared_dir=missing_dir,
        output_dir=project_root / "outputs" / "missing-shift",
        folds=5,
        seed=42,
        project_root=project_root,
    )
    with pytest.raises(FileNotFoundError, match="train_joined.parquet"):
        module.run_adversarial_validation(missing_config)

    non_five_fold_config = module.ValidationConfig(
        prepared_dir=prepared_dir,
        output_dir=project_root / "outputs" / "non-five-shift",
        folds=4,
        seed=42,
        project_root=project_root,
    )
    with pytest.raises(ValueError, match="exactly 5"):
        module.run_adversarial_validation(non_five_fold_config)


def test_cli_exposes_prepared_output_fold_and_seed_options():
    parsed = _validation_module().parse_args(
        [
            "--prepared-dir",
            "outputs/prepared",
            "--output-dir",
            "outputs/shift",
            "--folds",
            "5",
            "--seed",
            "7",
        ]
    )

    assert parsed.prepared_dir == Path("outputs/prepared")
    assert parsed.output_dir == Path("outputs/shift")
    assert parsed.folds == 5
    assert parsed.seed == 7
