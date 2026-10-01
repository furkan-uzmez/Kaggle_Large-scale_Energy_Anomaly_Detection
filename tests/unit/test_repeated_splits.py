from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

from project.data.splits import build_split_manifest
from project.data.validate import compute_file_sha256


def module():
    name = "project.data.repeated_splits"
    assert importlib.util.find_spec(name) is not None, "Repeated split generation is required"
    return importlib.import_module(name)


@pytest.fixture
def inputs():
    train = pl.DataFrame(
        {"building_id": building, "anomaly": int(row < building % 10)}
        for building in range(200)
        for row in range(10)
    )
    return train, build_split_manifest(train)


def signature(frame):
    return frozenset(
        frozenset(frame.filter(pl.col("fold") == fold)["building_id"].to_list())
        for fold in range(4)
    )


def test_frozen_vault_determinism_and_genuinely_distinct_partitions(inputs):
    train, original = inputs
    build = module().build_repeated_split_manifest
    result = build(train.lazy(), original, [42, 43, 44])
    assert result.equals(build(train.reverse(), original.reverse(), [42, 43, 44]))
    assert result.height == 600
    assert result.select("repeat", "building_id").unique().height == 600
    assert result.columns == [
        "building_id", "repeat", "seed", "split", "fold", "row_count",
        "anomaly_count", "normal_count", "anomaly_prevalence", "support_stratum",
    ]
    signatures = []
    vault = original.filter(pl.col("split") == "vault").sort("building_id")
    for repeat, seed in enumerate([42, 43, 44]):
        part = result.filter(pl.col("repeat") == repeat)
        assert part["seed"].unique().to_list() == [seed]
        assert part.filter(pl.col("split") == "vault").drop("repeat", "seed").equals(vault.select(result.drop("repeat", "seed").columns))
        dev = part.filter(pl.col("split") == "dev")
        assert dev.height == 160
        assert dev.group_by("fold").len()["len"].to_list() == [40] * 4
        support = dev.group_by("fold").agg(pl.col("anomaly_count").sum(), pl.col("normal_count").sum())
        assert support["anomaly_count"].min() > 0
        assert support["normal_count"].min() > 0
        signatures.append(signature(dev))
    assert len(set(signatures)) == 3


def test_vault_targets_never_validated_or_aggregated(inputs):
    train, original = inputs
    vault_ids = original.filter(pl.col("split") == "vault")["building_id"]
    poisoned = train.with_columns(
        pl.when(pl.col("building_id").is_in(vault_ids.implode()))
        .then(pl.lit(None, dtype=pl.Int64)).otherwise(pl.col("anomaly")).alias("anomaly")
    )
    result = module().build_repeated_split_manifest(poisoned.lazy(), original, [42, 43, 44])
    assert result.equals(module().build_repeated_split_manifest(train, original, [42, 43, 44]))


@pytest.mark.parametrize("seeds", [[], [42, 42], [True], [-1], [2**32]])
def test_reject_invalid_seeds(inputs, seeds):
    train, original = inputs
    with pytest.raises((ValueError, TypeError), match="seed"):
        module().build_repeated_split_manifest(train, original, seeds)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "extra", "null", "split", "support"])
def test_reject_manifest_coverage_and_stale_dev_support(inputs, fault):
    train, original = inputs
    if fault == "duplicate":
        original = pl.concat([original, original.head(1)])
    elif fault == "missing":
        original = original.slice(1)
    elif fault == "extra":
        train = train.filter(pl.col("building_id") != 199)
    elif fault == "null":
        train = pl.concat([train, pl.DataFrame({"building_id": [None], "anomaly": [0]}, schema=train.schema)])
    elif fault == "split":
        original = original.with_columns(pl.lit("other").alias("split"))
    else:
        original = original.with_columns((pl.col("row_count") + 1).alias("row_count"))
    with pytest.raises(ValueError):
        module().build_repeated_split_manifest(train, original, [42, 43, 44])


def test_reject_insufficient_class_support(inputs):
    train, original = inputs
    train = train.with_columns(pl.lit(0).alias("anomaly"))
    original = original.with_columns(pl.lit(0).alias("anomaly_count"), pl.col("row_count").alias("normal_count"), pl.lit(0.0).alias("anomaly_prevalence"))
    with pytest.raises(ValueError, match="support|classes"):
        module().build_repeated_split_manifest(train, original, [42, 43, 44])


def test_reject_fold_label_permutations(inputs, monkeypatch):
    train, original = inputs
    repeated = module()
    real_splitter = repeated.StratifiedKFold

    def fixed_splitter(**kwargs):
        seed = kwargs["random_state"]
        kwargs["random_state"] = 42
        splitter = real_splitter(**kwargs)

        class PermutedSplitter:
            def split(self, x, y):
                parts = list(splitter.split(x, y))
                offset = seed % len(parts)
                return iter(parts[offset:] + parts[:offset])

        return PermutedSplitter()

    monkeypatch.setattr(repeated, "StratifiedKFold", fixed_splitter)
    with pytest.raises(ValueError, match="distinct|partition"):
        repeated.build_repeated_split_manifest(train, original, [42, 43, 44])


def test_writer_provenance_idempotence_and_refusal(inputs, tmp_path):
    train, original = inputs
    split_dir = tmp_path / "outputs" / "splits"
    split_dir.mkdir(parents=True)
    source = split_dir / "building_folds.parquet"
    original.write_parquet(source)
    source_hash = compute_file_sha256(source)
    write = module().write_repeated_split_artifacts
    metadata = write(train.lazy(), source, tmp_path, [42, 43, 44])
    master = split_dir / "building_folds_repeated.parquet"
    marker = split_dir / "split_metadata_repeated.json"
    assert metadata == json.loads(marker.read_text())
    assert metadata["family_sha256"] == compute_file_sha256(master)
    assert metadata["artifacts"]["building_folds_sha256"] == metadata["family_sha256"]
    assert metadata["source_vault_manifest_sha256"] == source_hash
    assert metadata["summary"]["building_count"] == 200
    assert metadata["summary"]["row_count"] == 2000
    assert metadata["split_seeds"] == [42, 43, 44]
    assert metadata["repeat_count"] == 3
    assert metadata["repeats"] == 3
    assert metadata["source_manifest"] == {"path": "outputs/splits/building_folds.parquet", "sha256": source_hash}
    assert metadata["vault_building_ids"] == original.filter(pl.col("split") == "vault").sort("building_id")["building_id"].to_list()
    for repeat in metadata["per_repeat"]:
        path = tmp_path / repeat["path"]
        assert compute_file_sha256(path) == repeat["sha256"]
        assert repeat["class_support"]["vault"]["building_count"] == 40
        assert len(repeat["per_building_support"]) == 200
        assert not Path(repeat["path"]).is_absolute()
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in split_dir.iterdir()}
    assert metadata == write(train, source, tmp_path, [42, 43, 44])
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in split_dir.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite|different"):
        write(train, source, tmp_path, [45, 46, 47])
    assert compute_file_sha256(source) == source_hash


def test_repeated_cli_uses_prepared_data_and_isolated_config(inputs, tmp_path):
    train, original = inputs
    spec = importlib.util.spec_from_file_location("prepare_repeated", "scripts/prepare_data.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs/splits").mkdir(parents=True)
    (tmp_path / "outputs/prepared").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='repeated-test'\n")
    train.write_parquet(tmp_path / "outputs/prepared/train_joined.parquet")
    original.write_parquet(tmp_path / "outputs/splits/building_folds.parquet")
    config = tmp_path / "configs/custom.yaml"
    config.write_text("source_manifest: outputs/splits/building_folds.parquet\nprepared_train: outputs/prepared/train_joined.parquet\noutput_dir: outputs/splits\nn_dev_folds: 4\nsplit_seeds: [42, 43, 44]\n")
    try:
        code = prepare.main(["--build-repeated-splits", "--split-config", str(config), "--project-root", str(tmp_path)])
    except SystemExit as exc:
        code = exc.code
    assert code == 0, "CLI must generate repeated splits from prepared data"
    assert pl.read_parquet(tmp_path / "outputs/splits/building_folds_repeated.parquet").height == 600


@pytest.mark.parametrize("folds", [True, 1, 4.5, 3, 161])
def test_reject_invalid_fold_counts(inputs, folds):
    train, original = inputs
    with pytest.raises((ValueError, TypeError), match="fold"):
        module().build_repeated_split_manifest(train, original, [42, 43, 44], folds)


def test_optional_stratum_and_exact_dev_support(inputs):
    train, original = inputs
    result = module().build_repeated_split_manifest(train, original.drop("support_stratum"), [42, 43, 44])
    assert "support_stratum" not in result.columns
    expected = original.filter(pl.col("split") == "dev").select("building_id", "row_count", "anomaly_count", "normal_count", "anomaly_prevalence")
    for repeat in range(3):
        assert result.filter((pl.col("repeat") == repeat) & (pl.col("split") == "dev")).select(expected.columns).equals(expected)


def test_publication_metadata_last_and_failure_rolls_back(inputs, tmp_path, monkeypatch):
    train, original = inputs
    split_dir = tmp_path / "outputs/splits"
    split_dir.mkdir(parents=True)
    source = split_dir / "building_folds.parquet"
    original.write_parquet(source)
    old_metadata = split_dir / "split_metadata.json"
    old_metadata.write_text("original-marker")
    before = {p.name: p.read_bytes() for p in split_dir.iterdir()}
    repeated = module()
    real_link = repeated.os.link
    calls = []

    def fail_marker(staged, final):
        calls.append(final.name)
        assert not (split_dir / "split_metadata_repeated.json").exists()
        if final.name == "split_metadata_repeated.json":
            assert pl.read_parquet(split_dir / "building_folds_repeated.parquet").height == 600
            raise OSError("publication failure")
        real_link(staged, final)

    monkeypatch.setattr(repeated.os, "link", fail_marker)
    with pytest.raises(OSError, match="publication failure"):
        repeated.write_repeated_split_artifacts(train, source, tmp_path, [42, 43, 44])
    assert calls[-1] == "split_metadata_repeated.json"
    assert before == {p.name: p.read_bytes() for p in split_dir.iterdir()}


def test_refuse_tampered_existing_family_before_any_publication(inputs, tmp_path):
    train, original = inputs
    split_dir = tmp_path / "outputs/splits"
    split_dir.mkdir(parents=True)
    source = split_dir / "building_folds.parquet"
    original.write_parquet(source)
    marker = split_dir / "split_metadata_repeated.json"
    marker.write_text("unexpected existing metadata")
    before = {p.name: p.read_bytes() for p in split_dir.iterdir()}
    with pytest.raises(FileExistsError):
        module().write_repeated_split_artifacts(train, source, tmp_path, [42, 43, 44])
    assert before == {p.name: p.read_bytes() for p in split_dir.iterdir()}


def test_refuse_archived_or_external_destinations(inputs, tmp_path):
    train, original = inputs
    split_dir = tmp_path / "outputs/splits"
    split_dir.mkdir(parents=True)
    source = split_dir / "building_folds.parquet"
    original.write_parquet(source)
    with pytest.raises(ValueError, match="output_dir"):
        module().write_repeated_split_artifacts(train, source, tmp_path, [42, 43, 44], output_dir=tmp_path / "outputs/runs/20261001T103441Z-b07d3c4b")


def test_concurrent_conflicting_publication_is_not_treated_as_identical(inputs, tmp_path, monkeypatch):
    train, original = inputs
    split_dir = tmp_path / "outputs/splits"
    split_dir.mkdir(parents=True)
    source = split_dir / "building_folds.parquet"
    original.write_parquet(source)
    repeated = module()
    real_link = repeated.os.link
    conflict = split_dir / "building_folds_r1_seed43.parquet"

    def competing_writer(staged, final):
        real_link(staged, final)
        if final.name == "building_folds_repeated.parquet":
            conflict.write_bytes(b"different concurrent writer")

    monkeypatch.setattr(repeated.os, "link", competing_writer)
    with pytest.raises(FileExistsError, match="different|overwrite"):
        repeated.write_repeated_split_artifacts(train, source, tmp_path, [42, 43, 44])
    assert conflict.read_bytes() == b"different concurrent writer"
    assert not (split_dir / "split_metadata_repeated.json").exists()
    assert not (split_dir / "building_folds_repeated.parquet").exists()
