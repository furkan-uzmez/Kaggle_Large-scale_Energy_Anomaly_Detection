"""Repeated OOF must retain lineage while bounding materialization to one arm."""
from __future__ import annotations

import importlib
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from project.models.metrics import evaluate_binary_metrics


def fixture_frames():
    expected = pl.DataFrame({
        "row_index": list(range(8)), "building_id": [b for b in range(4) for _ in range(2)],
        "timestamp": [datetime(2022, 1, 1, h, tzinfo=UTC) for _ in range(4) for h in (0, 1)],
        "fold": [b for b in range(4) for _ in range(2)], "target": [0, 1] * 4,
    }).with_columns(*(pl.lit("hash").alias(k) for k in
                     ("split_sha256", "data_sha256", "config_sha256", "run_id")))
    expected_by_repeat = []
    parts = []
    # Every repeat has perfect ranking, but cross-repeat score scales overlap.
    for repeat, (negative, positive) in enumerate(((0.1, 0.2), (0.8, 0.9), (0.4, 0.5))):
        exp = expected.with_columns(pl.lit(42 + repeat).alias("split_seed"))
        expected_by_repeat.append((repeat, exp))
        for candidate in ("a", "b", "c"):
            parts.append(exp.with_columns(
                pl.lit(candidate).alias("candidate_id"), pl.lit(candidate).alias("recipe_id"),
                pl.lit(candidate).alias("model_config_id"), pl.lit(candidate).alias("model_family"),
                pl.lit(repeat).alias("repeat"), pl.lit("full").alias("execution_mode"),
                pl.lit(False).alias("non_comparative"),
                pl.when(pl.col("target") == 1).then(positive).otherwise(negative).alias("prediction"),
            ))
    return expected_by_repeat, pl.concat(parts)


def validator():
    module = importlib.import_module("project.models.artifacts")
    assert hasattr(module, "validate_oof_parquet"), "missing bounded file validator"
    return module.validate_oof_parquet


def test_file_validation_is_per_repeat_not_duplicate_pool(tmp_path, monkeypatch):
    expected, frame = fixture_frames()
    path = tmp_path / "oof.parquet"
    frame.write_parquet(path, row_group_size=3)
    module = importlib.import_module("project.models.artifacts")
    real_validate = module.validate_oof_artifact
    sizes = []

    def bounded_validate(oof, *args, **kwargs):
        sizes.append(oof.height)
        assert oof.height <= 8
        return real_validate(oof, *args, **kwargs)

    monkeypatch.setattr(module, "validate_oof_artifact", bounded_validate)
    monkeypatch.setattr(pl, "read_parquet", lambda *a, **k: pytest.fail("unbounded Parquet read"))
    result = validator()(path, iter(expected), expected_candidate_ids=("a", "b", "c"))
    assert sizes == [8] * 9
    assert result["repeat_count"] == 3
    assert result["oof_record_count"] == 72
    assert len(result["repeat_metrics"]) == 9
    assert len(result["fold_metrics"]) == 36
    assert result["candidate_metrics"]["a"]["pooled"]["roc_auc"] == 1.0
    all_a = frame.filter(pl.col("candidate_id") == "a")
    assert evaluate_binary_metrics(all_a["target"].to_numpy(), all_a["prediction"].to_numpy())["roc_auc"] < 1


@pytest.mark.parametrize("corruption", ["missing_repeat", "missing_fold", "missing_candidate",
    "duplicate_recipe", "hash", "fold", "seed", "target", "vault", "nan", "range", "smoke",
    "extra_repeat", "extra_candidate"])
def test_file_validator_rejects_corruption(tmp_path, corruption):
    expected, frame = fixture_frames()
    if corruption.startswith("missing_"):
        column, value = {"missing_repeat": ("repeat", 2), "missing_fold": ("fold", 3),
                         "missing_candidate": ("candidate_id", "a")}[corruption]
        frame = frame.filter(pl.col(column) != value)
    elif corruption == "duplicate_recipe":
        frame = pl.concat([frame, frame.head(1).with_columns(pl.lit("other").alias("recipe_id"))])
    else:
        column, value = {"hash": ("data_sha256", "wrong"), "fold": ("fold", 9),
            "seed": ("split_seed", 99), "target": ("target", 3), "vault": ("building_id", 999),
            "nan": ("prediction", float("nan")), "range": ("prediction", 1.1),
            "smoke": ("execution_mode", "smoke"), "extra_repeat": ("repeat", 3),
            "extra_candidate": ("candidate_id", "extra")}[corruption]
        frame = frame.with_columns(pl.lit(value).cast(frame.schema[column]).alias(column))
    path = tmp_path / "oof.parquet"
    frame.write_parquet(path)
    with pytest.raises(ValueError):
        validator()(path, iter(expected), expected_candidate_ids=("a", "b", "c"))


def test_small_frame_api_accepts_explicit_repeat_and_rejects_disguised_duplicates():
    module = importlib.import_module("project.models.artifacts")
    expected, frame = fixture_frames()
    part = frame.filter(pl.col("repeat") == 1)
    result = module.validate_oof_artifact(part, expected[1][1], expected_repeat=1,
                                         expected_candidate_ids=("a", "b", "c"))
    assert set(result) == {"a", "b", "c"}
    with pytest.raises(ValueError):
        module.validate_oof_artifact(pl.concat([part, part.head(1).with_columns(
            pl.lit(99, dtype=pl.Int64).alias("fold"), pl.lit("other").alias("recipe_id"))]),
            expected[1][1], expected_repeat=1)


def test_columnwise_writer_preserves_float64_and_limits_batch_size(tmp_path, monkeypatch):
    from scripts import train

    assert hasattr(train, "_write_prediction_blocks"), "missing column-wise bounded writer"
    expected, _ = fixture_frames()
    keys = expected[0][1].rename({"target": "anomaly"})
    scores = np.array([np.nextafter(0.5, 1)] * 8, dtype=np.float64)
    path = tmp_path / "oof.parquet"
    calls = []
    real_write = pq.ParquetWriter.write_table

    def bounded_write(self, table, *args, **kwargs):
        calls.append(table.num_rows)
        assert table.num_rows <= 3
        return real_write(self, table, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", bounded_write)
    metadata = {"candidate_id": "a", "recipe_id": "a", "model_family": "a", "model_config_id": "a",
                "repeat": 0, "split_seed": 42, "execution_mode": "full", "non_comparative": False}
    train._write_prediction_blocks(path, [(keys, scores, metadata)], target_name="anomaly", batch_size=3)
    restored = pl.read_parquet(path)
    assert calls == [3, 3, 2]
    assert restored["prediction"].dtype == pl.Float64
    np.testing.assert_array_equal(restored["prediction"].to_numpy(), scores)
    assert restored["row_index"].to_list() == list(range(8))
    assert not path.with_suffix(".parquet.partial").exists()


def test_approved_three_repeat_config_and_budget():
    from pathlib import Path

    import yaml
    from scripts import train

    from project.experiments.baseline_budget import validate_baseline_budget

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/experiments/baselines.yaml").read_text())
    budget = yaml.safe_load((root / "configs/experiments/baseline_execution_budget.yaml").read_text())
    assert config["repeats"] == 3 and config["split_seeds"] == [42, 43, 44]
    assert budget["planned_fits"] == 36 and budget["repeat_ids"] == [0, 1, 2]
    assert config["seed"] == budget["seed"] == 42
    train._validate_execution_config(config)
    validate_baseline_budget(budget, config)
