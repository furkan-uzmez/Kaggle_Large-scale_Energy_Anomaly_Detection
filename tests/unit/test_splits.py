from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest


def _split_module() -> ModuleType:
    module_name = "project.data.splits"
    module_spec = importlib.util.find_spec(module_name)
    assert module_spec is not None, (
        "project.data.splits is required for building split manifests"
    )
    return importlib.import_module(module_name)


def _training_rows(
    building_count: int = 200,
    positive_counts: list[int] | None = None,
    rows_per_building: int = 10,
) -> pl.DataFrame:
    if positive_counts is None:
        positive_counts = [
            building_id % rows_per_building for building_id in range(building_count)
        ]

    rows = [
        {"building_id": building_id, "anomaly": int(row_index < positive_count)}
        for building_id, positive_count in enumerate(positive_counts)
        for row_index in range(rows_per_building)
    ]
    return pl.DataFrame(rows)


def _run_prepare_data(prepare_data: ModuleType, args: list[str]) -> int:
    try:
        return prepare_data.main(args)
    except SystemExit as error:
        return int(error.code)


def test_build_split_manifest_assigns_each_building_once_and_is_seed_deterministic():
    build_split_manifest = _split_module().build_split_manifest
    train = _training_rows().lazy()

    manifest = build_split_manifest(train, seed=42)
    repeated_manifest = build_split_manifest(train, seed=42)

    assert manifest.equals(repeated_manifest)
    assert manifest.get_column("building_id").n_unique() == 200
    assert manifest.height == 200
    assert {
        "building_id",
        "split",
        "fold",
        "row_count",
        "anomaly_count",
        "normal_count",
        "anomaly_prevalence",
        "support_stratum",
    }.issubset(manifest.columns)

    vault = manifest.filter(pl.col("split") == "vault")
    dev = manifest.filter(pl.col("split") == "dev")
    assert vault.height == 40
    assert dev.height == 160
    assert vault.get_column("fold").null_count() == 40
    assert set(dev.get_column("fold").unique().to_list()) == {0, 1, 2, 3}
    assert dev.group_by("fold").len().get_column("len").to_list() == [40] * 4

    building_zero = manifest.filter(pl.col("building_id") == 0).row(0, named=True)
    assert building_zero["row_count"] == 10
    assert building_zero["anomaly_count"] == 0
    assert building_zero["normal_count"] == 10
    assert building_zero["anomaly_prevalence"] == 0.0


def test_every_partition_has_both_target_classes_and_zero_building_overlap():
    manifest = _split_module().build_split_manifest(_training_rows())
    training_summary = (
        _training_rows()
        .group_by("building_id")
        .agg(
            pl.len().alias("row_count"),
            pl.col("anomaly").sum().alias("anomaly_count"),
            (pl.col("anomaly") == 0).sum().alias("normal_count"),
        )
    )
    checked = manifest.join(training_summary, on="building_id", suffix="_expected")

    assert (
        checked.get_column("row_count").to_list()
        == checked.get_column("row_count_expected").to_list()
    )
    assert (
        checked.get_column("anomaly_count").to_list()
        == checked.get_column("anomaly_count_expected").to_list()
    )
    assert (
        checked.get_column("normal_count").to_list()
        == checked.get_column("normal_count_expected").to_list()
    )
    assert checked.group_by("building_id").len().get_column("len").max() == 1

    support = manifest.group_by(["split", "fold"], maintain_order=True).agg(
        pl.col("anomaly_count").sum().alias("positive_rows"),
        pl.col("normal_count").sum().alias("negative_rows"),
        pl.col("row_count").sum().alias("rows"),
        pl.len().alias("buildings"),
    )
    assert support.get_column("positive_rows").min() > 0
    assert support.get_column("negative_rows").min() > 0
    assert (support.get_column("rows") > 0).all()
    assert (support.get_column("buildings") == 40).all()


def test_sparse_exact_support_bins_are_merged_into_feasible_strata():
    support_counts = list(range(25))
    train = _training_rows(
        building_count=len(support_counts),
        positive_counts=support_counts,
        rows_per_building=26,
    )

    manifest = _split_module().build_split_manifest(train)

    stratum_sizes = (
        manifest.group_by("support_stratum").len().get_column("len").to_list()
    )
    assert len(stratum_sizes) == 5
    assert stratum_sizes == [5] * 5
    assert set(manifest.get_column("support_stratum").to_list()) == set(range(5))


def test_insufficient_building_level_positive_support_fails_closed():
    train = _training_rows(
        building_count=25,
        positive_counts=[1, 1, 1, 1] + [0] * 21,
        rows_per_building=2,
    )

    with pytest.raises(ValueError, match=r"[Pp]ositive.*4.*5.*building-level"):
        _split_module().build_split_manifest(train)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"building_id": 1, "anomaly": 2}], "binary"),
        ([{"building_id": None, "anomaly": 0}], "building_id.*null"),
        ([{"building_id": 1, "anomaly": None}], "anomaly.*null"),
    ],
)
def test_invalid_split_inputs_fail_with_diagnostics(
    rows: list[dict[str, int | None]], message: str
):
    with pytest.raises(ValueError, match=message):
        _split_module().build_split_manifest(pl.DataFrame(rows))


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"n_dev_folds": 4.5}, "n_dev_folds must be an integer"),
        ({"seed": True}, "seed must be an integer"),
    ],
)
def test_noninteger_split_parameters_fail_with_precise_errors(
    parameters: dict[str, int | float | bool], message: str
):
    with pytest.raises(TypeError, match=message):
        _split_module().build_split_manifest(_training_rows(), **parameters)


def test_build_splits_cli_persists_manifest_and_provenance(tmp_path: Path):
    prepare_data_spec = importlib.util.spec_from_file_location(
        "prepare_data_for_split_tests", Path("scripts/prepare_data.py").resolve()
    )
    assert prepare_data_spec is not None and prepare_data_spec.loader is not None
    prepare_data = importlib.util.module_from_spec(prepare_data_spec)
    prepare_data_spec.loader.exec_module(prepare_data)

    project_root = tmp_path / "project"
    raw_dir = tmp_path / "raw"
    config_dir = project_root / "configs"
    output_dir = project_root / "outputs"
    config_dir.mkdir(parents=True)
    raw_dir.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='split-test'\n", encoding="utf-8"
    )
    (config_dir / "split.yaml").write_text(
        "building_col: site_id\ntarget_col: anomaly\n"
        "vault_fraction: 0.2\nn_dev_folds: 4\nseed: 42\n",
        encoding="utf-8",
    )
    _training_rows().rename({"building_id": "site_id"}).write_csv(
        raw_dir / "train.csv"
    )

    exit_code = _run_prepare_data(
        prepare_data,
        [
            "--build-splits",
            "--project-root",
            str(project_root),
            "--data-dir",
            str(raw_dir),
        ],
    )

    split_dir = output_dir / "splits"
    manifest_path = split_dir / "building_folds.parquet"
    metadata_path = split_dir / "split_metadata.json"
    assert exit_code == 0
    assert manifest_path.is_file()
    assert metadata_path.is_file()

    manifest = pl.read_parquet(manifest_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert manifest.height == 200
    assert metadata["seed"] == 42
    assert len(metadata["hashes"]["train_csv_sha256"]) == 64
    assert len(metadata["hashes"]["split_config_sha256"]) == 64
    assert metadata["stratification_method"]
    assert metadata["rationale"]
    assert metadata["class_support"]["vault"]["positive_rows"] > 0
    assert metadata["class_support"]["vault"]["negative_rows"] > 0
    assert metadata["artifacts"]["building_folds_sha256"]
    assert "site_id" in manifest.columns
    assert "site_id" in metadata["per_building_support"][0]


def test_build_splits_ignores_nonpositive_sample_rows():
    prepare_data_spec = importlib.util.spec_from_file_location(
        "prepare_data_sample_rows_tests", Path("scripts/prepare_data.py").resolve()
    )
    assert prepare_data_spec is not None and prepare_data_spec.loader is not None
    prepare_data = importlib.util.module_from_spec(prepare_data_spec)
    prepare_data_spec.loader.exec_module(prepare_data)

    parsed = prepare_data.parse_args(["--build-splits", "--sample-rows", "0"])

    assert parsed.build_splits is True
    assert parsed.sample_rows == 0


@pytest.mark.parametrize(
    ("config_text", "invalid_field"),
    [
        ("n_dev_folds: 4.5\n", "n_dev_folds"),
        ("seed: 42.5\n", "seed"),
        ("n_dev_folds: true\nvault_fraction: 0.5\n", "n_dev_folds"),
        ("n_dev_fold: 4\n", "n_dev_fold"),
    ],
)
def test_build_splits_cli_rejects_invalid_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_text: str,
    invalid_field: str,
):
    prepare_data_spec = importlib.util.spec_from_file_location(
        "prepare_data_invalid_config_tests", Path("scripts/prepare_data.py").resolve()
    )
    assert prepare_data_spec is not None and prepare_data_spec.loader is not None
    prepare_data = importlib.util.module_from_spec(prepare_data_spec)
    prepare_data_spec.loader.exec_module(prepare_data)

    project_root = tmp_path / "project"
    raw_dir = tmp_path / "raw"
    config_dir = project_root / "configs"
    output_dir = project_root / "outputs"
    config_dir.mkdir(parents=True)
    raw_dir.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='split-test'\n", encoding="utf-8"
    )
    (config_dir / "split.yaml").write_text(config_text, encoding="utf-8")
    _training_rows().write_csv(raw_dir / "train.csv")

    exit_code = _run_prepare_data(
        prepare_data,
        [
            "--build-splits",
            "--project-root",
            str(project_root),
            "--data-dir",
            str(raw_dir),
        ],
    )

    assert exit_code != 0
    assert invalid_field in capsys.readouterr().err
    assert not (output_dir / "splits" / "building_folds.parquet").exists()


def test_build_splits_cli_rejects_split_directory_symlink_outside_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    prepare_data_spec = importlib.util.spec_from_file_location(
        "prepare_data_symlink_tests", Path("scripts/prepare_data.py").resolve()
    )
    assert prepare_data_spec is not None and prepare_data_spec.loader is not None
    prepare_data = importlib.util.module_from_spec(prepare_data_spec)
    prepare_data_spec.loader.exec_module(prepare_data)

    project_root = tmp_path / "project"
    raw_dir = tmp_path / "raw"
    config_dir = project_root / "configs"
    output_dir = project_root / "outputs"
    external_dir = tmp_path / "external"
    config_dir.mkdir(parents=True)
    raw_dir.mkdir()
    output_dir.mkdir()
    external_dir.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='split-test'\n", encoding="utf-8"
    )
    (config_dir / "split.yaml").write_text(
        "vault_fraction: 0.2\nn_dev_folds: 4\nseed: 42\n", encoding="utf-8"
    )
    _training_rows().write_csv(raw_dir / "train.csv")
    (output_dir / "splits").symlink_to(external_dir, target_is_directory=True)

    exit_code = _run_prepare_data(
        prepare_data,
        [
            "--build-splits",
            "--project-root",
            str(project_root),
            "--data-dir",
            str(raw_dir),
        ],
    )

    assert exit_code != 0
    assert "outside" in capsys.readouterr().err.lower()
    assert list(external_dir.iterdir()) == []
