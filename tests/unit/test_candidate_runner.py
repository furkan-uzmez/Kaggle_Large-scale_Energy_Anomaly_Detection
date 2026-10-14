"""Unit tests for candidate IO and runner (Task 12)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from candidate_fixtures import CORE_COLUMNS, synthetic_partitions
from project.experiments.candidate_contracts import (
    CandidateScore,
    CandidateSpec,
    FitBudget,
    FoldPartition,
    RunResult,
    RunSpec,
)
from project.experiments.candidate_dispatch import build_experiment_arms


def make_spec(recipe: str = "feat_meter_deviation_ratio", family: str = "hist_gradient_boosting", mode: str = "synthetic_smoke", fits: int = 8, trials: int = 0) -> RunSpec:
    budget = FitBudget(fits, trials, 30.0, 256 * 1024**2, 1)
    return RunSpec(
        1,
        (CandidateSpec(recipe, CORE_COLUMNS, family, (), ()),),
        budget,
        mode,
        "notebooks/02_baseline_and_candidate_design.ipynb",
        "candidate-cell",
    )


# ===========================================================================
# candidate_io tests
# ===========================================================================

def test_load_candidate_partitions_synthetic(tmp_path: Path):
    from project.experiments.candidate_io import load_candidate_partitions

    spec = make_spec(mode="synthetic_smoke")
    partitions = list(load_candidate_partitions(tmp_path, spec))
    assert len(partitions) == 2
    for p in partitions:
        assert isinstance(p, FoldPartition)
        assert p.repeat == 0
        assert p.fold in (0, 1)
        assert len(p.X_train) + len(p.X_validation) <= 128
        assert tuple(p.X_train.columns) == tuple(CORE_COLUMNS)
        assert tuple(p.X_validation.columns) == tuple(CORE_COLUMNS)
        # Validation keys has KEY_COLUMNS
        assert list(p.validation_keys.columns) == ["row_index", "building_id", "timestamp", "repeat", "fold"]


def test_load_candidate_partitions_parquet_confined_and_vault_filtered(tmp_path: Path):
    from project.experiments.candidate_io import load_candidate_partitions

    # Create synthetic parquet dataset and repeated split manifest
    split_path = tmp_path / "splits.parquet"
    data_path = tmp_path / "data.parquet"

    # 4 buildings: b0, b1 in dev; b2, b3 in vault
    manifest_df = pl.DataFrame({
        "building_id": ["b0", "b1", "b2", "b3"],
        "repeat": [0, 0, 0, 0],
        "seed": [42, 42, 42, 42],
        "split": ["dev", "dev", "vault", "vault"],
        "fold": [0, 1, None, None],
    })
    manifest_df.write_parquet(split_path)

    # Data frame with all 4 buildings
    rows = []
    for b in ["b0", "b1", "b2", "b3"]:
        for h in [0, 1]:
            rows.append({
                "building_id": b,
                "timestamp": pd.Timestamp("2024-01-01") + pd.to_timedelta(h, unit="h"),
                "meter_reading": 10.0,
                "square_feet": 1000.0,
                "year_built": 1990,
                "floor_count": 2,
                "hour": h,
                "weekday": 0,
                "month": 1,
                "year": 2024,
                "primary_use": "office",
                "anomaly": h,
                "extra_column_to_drop": 999.0,
            })
    data_df = pd.DataFrame(rows)
    data_df.index.name = "row_index"
    data_df.to_parquet(data_path)

    spec = make_spec(mode="full")
    partitions = list(load_candidate_partitions(tmp_path, spec, data_path=data_path, split_path=split_path))
    assert len(partitions) == 2  # 2 folds in dev

    for p in partitions:
        assert isinstance(p, FoldPartition)
        # Verify vault buildings b2, b3 NEVER appear in train or validation
        all_train_buildings = set(p.groups_train.unique())
        all_val_buildings = set(p.validation_keys["building_id"].unique())
        assert "b2" not in all_train_buildings and "b2" not in all_val_buildings
        assert "b3" not in all_train_buildings and "b3" not in all_val_buildings
        # Verify extra_column_to_drop was projected out
        assert "extra_column_to_drop" not in p.X_train.columns
        assert "extra_column_to_drop" not in p.X_validation.columns


def test_load_candidate_partitions_unconfined_path_rejected(tmp_path: Path):
    from project.experiments.candidate_io import load_candidate_partitions

    spec = make_spec(mode="full")
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    outside_file = tmp_path / "outside.parquet"
    outside_file.touch()

    # Pass outside_file which is outside sub_dir
    with pytest.raises(ValueError, match="managed root|parent-path aliases"):
        list(load_candidate_partitions(sub_dir, spec, data_path=outside_file, split_path=sub_dir / "split.parquet"))


# ===========================================================================
# candidate_runner tests
# ===========================================================================

def test_run_partition_arms_sees_training_view_only():
    from project.experiments.candidate_contracts import FitCounter
    from project.experiments.candidate_runner import run_partition_arms

    partitions = synthetic_partitions()
    partition = partitions[0]
    spec = CandidateSpec("feat_meter_deviation_ratio", CORE_COLUMNS, "hist_gradient_boosting", (), ())
    arms = build_experiment_arms(spec)

    counter = FitCounter(FitBudget(4, 0, 30.0, 256 * 1024**2, 1))
    counter.reserve(len(arms), 0)

    result = run_partition_arms(partition, arms, counter)
    assert hasattr(result, "scores")
    assert len(result.scores) == len(arms)
    for score in result.scores.values():
        assert isinstance(score, CandidateScore)
        assert score.kind == "positive_probability"
        assert len(score.values) == len(partition.X_validation)
        assert np.isfinite(score.values).all()
        assert ((score.values >= 0) & (score.values <= 1)).all()

    # Counter must have debited exactly the number of arms
    assert counter.used_model_fits == len(arms)


def test_run_synthetic_campaign_success(tmp_path: Path):
    from project.experiments.candidate_runner import run_synthetic_campaign

    spec = make_spec("feat_meter_deviation_ratio", fits=8)
    result = run_synthetic_campaign(spec, tmp_path)

    assert isinstance(result, RunResult)
    assert result.status == "completed"
    assert result.non_comparative is True
    assert result.n03_eligible is False
    assert len(result.artifact_paths) > 0

    # Ensure files exist in tmp_path
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "run_status.json").exists()
    assert (tmp_path / "candidate_inventory.jsonl").exists()
    assert (tmp_path / "candidate_oof.parquet").exists()


def test_run_synthetic_campaign_autogluon_supervisor_lease(tmp_path: Path, monkeypatch):
    from project.experiments.candidate_runner import (
        bind_autogluon_supervisor_lease,
        run_synthetic_campaign,
    )
    from project.models import autogluon_adapter

    # 1. Without supervisor lease, AutoGluon is blocked
    ag_spec = make_spec("model_autogluon_core_features", family="autogluon", fits=4)
    result_blocked = run_synthetic_campaign(ag_spec, tmp_path / "ag_blocked")
    assert result_blocked.status == "blocked"

    # 2. With supervisor lease bound and fake backend, AutoGluon completes
    class FakeFeatureMetadata:
        def __init__(self, type_map_raw=None, type_group_map_special=None):
            self.type_map_raw = dict(type_map_raw or {})
            self.type_group_map_special = dict(type_group_map_special or {})

    class FakePredictor:
        def __init__(self, label, problem_type=None, eval_metric=None, path=None, positive_class=None):
            self.classes_ = [1, 0]
            self.positive_class = positive_class

        def fit(self, train_data, tuning_data=None, time_limit=None, presets=None,
                hyperparameters=None, fit_weighted_ensemble=True,
                dynamic_stacking=False, calibrate_decision_threshold="auto",
                num_cpus="auto", num_gpus="auto", fit_strategy="sequential",
                memory_limit="auto", **kwargs):
            return self

        def predict_proba(self, data, model=None, as_pandas=True, as_multiclass=True,
                          transform_features=True):
            return pd.DataFrame({1: np.full(len(data), 0.7), 0: np.full(len(data), 0.3)}, index=data.index)

        def info(self):
            return {
                "model_info": {
                    "ExtraTrees": {
                        "hyperparameters": {},
                        "feature_metadata": FakeFeatureMetadata(),
                    }
                },
                "feature_metadata_in": FakeFeatureMetadata(),
            }

        def model_names(self):
            return ["ExtraTrees"]

        def _validate_fit_kwargs(self, *, kwargs):
            return kwargs.copy()

    monkeypatch.setattr(autogluon_adapter, "_import_tabular", lambda: SimpleNamespace(
        __version__="1.4.0", TabularPredictor=FakePredictor, FeatureMetadata=FakeFeatureMetadata,
    ))

    metadata_module = "autogluon.common.features.feature_metadata"
    original_import = autogluon_adapter.importlib.import_module

    def import_boundary(name, package=None):
        if name == metadata_module:
            return SimpleNamespace(FeatureMetadata=FakeFeatureMetadata)
        return original_import(name, package)

    monkeypatch.setattr(autogluon_adapter.importlib, "import_module", import_boundary)

    with bind_autogluon_supervisor_lease(tmp_path / "ag_success"):
        result_success = run_synthetic_campaign(ag_spec, tmp_path / "ag_success")
        assert result_success.status == "completed"


def test_run_synthetic_campaign_hpo_respects_blocker(tmp_path: Path):
    from project.experiments.candidate_runner import run_synthetic_campaign

    spec = make_spec("regime_same_family_hpo_vs_default", fits=10, trials=2)
    result = run_synthetic_campaign(spec, tmp_path)
    assert result.status == "blocked"


def test_run_synthetic_campaign_insufficient_budget(tmp_path: Path):
    from project.experiments.candidate_runner import run_synthetic_campaign

    # 2 folds * 2 arms = 4 fits needed, give only 1 fit
    spec = make_spec("feat_meter_deviation_ratio", fits=1)
    result = run_synthetic_campaign(spec, tmp_path)
    assert result.status == "blocked"
