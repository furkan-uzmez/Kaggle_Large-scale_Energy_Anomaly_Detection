"""Unit tests for GBDT Hyperparameter Optimization (HPO) module, CLI runner, and configuration.

Verifies:
1. Strict inner split isolation: inner_val has zero building intersection with inner_train,
   and outer_val is never accessed or leaked during HPO trials.
2. Feature allowlist guard: strictly accepts verified 31 features and rejects forbidden columns
   (gte_*, *_id, building_hour, building_meter, timestamp, anomaly).
3. Fast execution in dry-run/mock mode without requiring Optuna fits.
4. CLI argument parsing and configuration loading.
5. Schema conformity of output OOF records to 24-field CANDIDATE_OOF_SCHEMA and transport validation JSON.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml
from scripts.train_gbdt_hpo import (
    DEFAULT_CONFIG_PATH,
    load_config,
    parse_args,
    run_gbdt_hpo,
)

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.gbdt_hpo import (
    SUPPORTED_ARMS,
    SUPPORTED_FAMILIES,
    VERIFIED_FEATURE_POOL,
    build_inner_split,
    is_forbidden_predictor,
    normalize_arm,
    normalize_family,
    optimize_hyperparameters,
    train_hpo_partition,
    validate_feature_pool,
)


@pytest.fixture
def synthetic_data() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Generate synthetic train and validation DataFrames."""
    rng = np.random.default_rng(42)
    n_train = 80
    n_val = 30

    train_anomaly = [1 if i % 5 == 0 else 0 for i in range(n_train)]
    val_anomaly = [1 if i % 4 == 0 else 0 for i in range(n_val)]

    # 4 distinct buildings in train (b1, b2, b3, b4), 2 in val (b5, b6)
    train_buildings = ["b1"] * 20 + ["b2"] * 20 + ["b3"] * 20 + ["b4"] * 20
    val_buildings = ["b5"] * 15 + ["b6"] * 15

    train_dict: dict[str, Any] = {
        "building_id": train_buildings,
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(hours=i)
            for i in range(n_train)
        ],
        "anomaly": train_anomaly,
    }
    val_dict: dict[str, Any] = {
        "building_id": val_buildings,
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(hours=i)
            for i in range(n_val)
        ],
        "anomaly": val_anomaly,
    }

    feature_cols = [
        "meter_reading",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
        "primary_use",
    ]

    for col in feature_cols:
        if col == "primary_use":
            train_dict[col] = ["Education" if i % 2 == 0 else "Office" for i in range(n_train)]
            val_dict[col] = ["Education" if i % 2 == 0 else "Office" for i in range(n_val)]
        else:
            train_dict[col] = [float(rng.uniform(10.0, 100.0)) for _ in range(n_train)]
            val_dict[col] = [float(rng.uniform(10.0, 100.0)) for _ in range(n_val)]

    return pl.DataFrame(train_dict), pl.DataFrame(val_dict), feature_cols


class TestInnerSplitIsolation:
    """Test building-grouped inner validation split isolation."""

    def test_inner_split_building_disjointness(
        self, synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """Assert set of buildings in inner_val has zero intersection with inner_train."""
        train_df, _, _ = synthetic_data

        inner_train_df, inner_val_df = build_inner_split(train_df, seed=42, val_fraction=0.25)

        train_buildings = set(inner_train_df["building_id"].unique().to_list())
        val_buildings = set(inner_val_df["building_id"].unique().to_list())

        assert len(train_buildings) > 0
        assert len(val_buildings) > 0
        assert train_buildings.isdisjoint(val_buildings), (
            f"Inner split building leakage detected! Overlap: {train_buildings & val_buildings}"
        )
        assert len(inner_train_df) + len(inner_val_df) == len(train_df)

    def test_inner_split_never_accesses_outer_val(
        self,
        synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assert outer validation partition is never passed or accessed."""
        train_df, val_df, feature_cols = synthetic_data

        # build_inner_split signature only accepts train_df
        inner_train_1, inner_val_1 = build_inner_split(train_df, seed=42, val_fraction=0.25)

        # Mutate outer val_df in arbitrary ways; inner split must remain identical
        poisoned_val = val_df.with_columns(
            (pl.col("anomaly") * 0).alias("anomaly"),
            pl.lit("corrupted").alias("building_id"),
        )
        assert poisoned_val.height == val_df.height

        inner_train_2, inner_val_2 = build_inner_split(train_df, seed=42, val_fraction=0.25)
        assert inner_train_1.equals(inner_train_2)
        assert inner_val_1.equals(inner_val_2)

        # Spy on optimize_hyperparameters inside train_hpo_partition(arm="challenger")
        import project.models.gbdt_hpo as gbdt_hpo_module

        seen_inner_train_buildings: set[str] = set()
        seen_inner_val_buildings: set[str] = set()

        def spy_optimize_hyperparameters(
            family: str,
            inner_train_df: Any,
            inner_val_df: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            tr_b = (
                inner_train_df["building_id"].unique().to_list()
                if isinstance(inner_train_df, pl.DataFrame)
                else list(inner_train_df["building_id"].unique())
            )
            va_b = (
                inner_val_df["building_id"].unique().to_list()
                if isinstance(inner_val_df, pl.DataFrame)
                else list(inner_val_df["building_id"].unique())
            )
            seen_inner_train_buildings.update(tr_b)
            seen_inner_val_buildings.update(va_b)
            return {
                "learning_rate": 0.05,
                "max_depth": 5,
                "n_estimators": 5,
                "num_leaves": 31,
                "is_fallback": False,
            }

        monkeypatch.setattr(
            gbdt_hpo_module,
            "optimize_hyperparameters",
            spy_optimize_hyperparameters,
        )

        # Mock LGBMClassifier to avoid heavy model fit
        import lightgbm as lgb

        def mock_fit(self: Any, X: Any, y: Any, **fit_kwargs: Any) -> Any:
            return self

        def mock_predict_proba(self: Any, X: Any) -> np.ndarray:
            n = len(X)
            p = np.full(n, 0.5)
            return np.column_stack([1.0 - p, p])

        monkeypatch.setattr(lgb.LGBMClassifier, "fit", mock_fit)
        monkeypatch.setattr(lgb.LGBMClassifier, "predict_proba", mock_predict_proba)

        train_hpo_partition(
            model_family="lightgbm",
            arm="challenger",
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col="anomaly",
            n_trials=2,
            seed=42,
            dry_run=False,
        )

        train_buildings = set(train_df["building_id"].unique().to_list())
        val_buildings = set(val_df["building_id"].unique().to_list())

        all_seen_buildings = seen_inner_train_buildings | seen_inner_val_buildings
        assert len(all_seen_buildings) > 0
        assert all_seen_buildings.issubset(train_buildings)
        assert all_seen_buildings.isdisjoint(val_buildings)
        assert seen_inner_train_buildings.isdisjoint(seen_inner_val_buildings)
    def test_inner_split_handles_polars_and_pandas(
        self, synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """Assert build_inner_split works with both Polars and Pandas DataFrames."""
        train_df, _, _ = synthetic_data
        train_pandas = train_df.to_pandas()

        inner_tr_pl, inner_va_pl = build_inner_split(train_df, seed=42, val_fraction=0.25)
        inner_tr_pd, inner_va_pd = build_inner_split(train_pandas, seed=42, val_fraction=0.25)

        assert isinstance(inner_tr_pl, pl.DataFrame)
        assert isinstance(inner_va_pl, pl.DataFrame)
        import pandas as pd
        assert isinstance(inner_tr_pd, pd.DataFrame)
        assert isinstance(inner_va_pd, pd.DataFrame)
        assert len(inner_tr_pl) == len(inner_tr_pd)
        assert len(inner_va_pl) == len(inner_va_pd)

    def test_inner_split_insufficient_buildings(self) -> None:
        """Assert ValueError raised if train_df has fewer than 2 buildings."""
        single_b_df = pl.DataFrame({"building_id": ["b1", "b1"], "anomaly": [0, 1]})
        with pytest.raises(ValueError, match="fewer than 2 unique buildings"):
            build_inner_split(single_b_df)


class TestFeatureAllowlist:
    """Test feature pool allowlist and strict anti-leakage guards."""

    def test_accepts_verified_safe_features(self) -> None:
        """Assert validate_feature_pool accepts all 31 verified safe features."""
        assert len(VERIFIED_FEATURE_POOL) == 31
        validated = validate_feature_pool(VERIFIED_FEATURE_POOL)
        assert validated == list(VERIFIED_FEATURE_POOL)

    @pytest.mark.parametrize(
        "forbidden_col",
        [
            "gte_hour",
            "gte_building_id",
            "gte_meter_primary_use",
            "site_id",
            "building_id",
            "building_hour",
            "building_meter",
            "timestamp",
            "anomaly",
            "target",
            "user_id",
        ],
    )
    def test_rejects_forbidden_columns(self, forbidden_col: str) -> None:
        """Assert forbidden columns violate anti-leakage contract and raise ValueError."""
        assert is_forbidden_predictor(forbidden_col)
        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            validate_feature_pool(["meter_reading", forbidden_col])

    def test_rejects_empty_or_duplicate(self) -> None:
        """Assert empty or duplicate feature lists are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_feature_pool([])

        with pytest.raises(ValueError, match="Duplicate predictor column"):
            validate_feature_pool(["meter_reading", "meter_reading"])


class TestHyperparameterOptimization:
    """Test Optuna HPO routine and partition execution."""

    def test_dry_run_fast_execution(
        self, synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """Assert dry_run completes instantly without Optuna trial fits."""
        train_df, val_df, feature_cols = synthetic_data

        for fam in SUPPORTED_FAMILIES:
            for arm in SUPPORTED_ARMS:
                preds, params = train_hpo_partition(
                    model_family=fam,
                    arm=arm,
                    train_df=train_df,
                    val_df=val_df,
                    feature_cols=feature_cols,
                    target_col="anomaly",
                    dry_run=True,
                )
                assert preds.shape == (val_df.height,)
                assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
                assert isinstance(params, dict)
                assert "learning_rate" in params

    def test_optimize_hyperparameters_optuna(
        self,
        synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assert Optuna optimizes hyperparameters on inner split in milliseconds without real training."""
        train_df, _, feature_cols = synthetic_data
        inner_train, inner_val = build_inner_split(train_df, seed=42, val_fraction=0.25)

        import lightgbm as lgb
        import xgboost as xgb

        def mock_fit(self: Any, X: Any, y: Any, **fit_kwargs: Any) -> Any:
            return self

        def mock_predict_proba(self: Any, X: Any) -> np.ndarray:
            n = len(X)
            p = (np.arange(n) % 10) / 10.0 + 0.05
            return np.column_stack([1.0 - p, p])

        monkeypatch.setattr(lgb.LGBMClassifier, "fit", mock_fit)
        monkeypatch.setattr(lgb.LGBMClassifier, "predict_proba", mock_predict_proba)
        monkeypatch.setattr(xgb.XGBClassifier, "fit", mock_fit)
        monkeypatch.setattr(xgb.XGBClassifier, "predict_proba", mock_predict_proba)

        for fam in ("lightgbm", "xgboost"):
            best_params = optimize_hyperparameters(
                family=fam,
                inner_train_df=inner_train,
                inner_val_df=inner_val,
                feature_cols=feature_cols,
                target_col="anomaly",
                n_trials=2,
                seed=42,
                dry_run=False,
                n_jobs=2,
            )
            assert isinstance(best_params, dict)
            assert "learning_rate" in best_params
            assert "max_depth" in best_params
            assert best_params.get("is_fallback") is False
            if fam == "lightgbm":
                assert "num_leaves" in best_params
            elif fam == "xgboost":
                assert "min_child_weight" in best_params

    def test_optimize_hyperparameters_fallback_flag(
        self, synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assert is_fallback flag is True on Optuna failures, dry-run, or mock mode."""
        train_df, val_df, feature_cols = synthetic_data
        inner_train, inner_val = build_inner_split(train_df, seed=42, val_fraction=0.25)

        # 1. dry-run mode
        dry_params = optimize_hyperparameters(
            family="lightgbm",
            inner_train_df=inner_train,
            inner_val_df=inner_val,
            feature_cols=feature_cols,
            dry_run=True,
        )
        assert dry_params["is_fallback"] is True

        # 2. train_hpo_partition with mock=True
        _, mock_params = train_hpo_partition(
            model_family="lightgbm",
            arm="challenger",
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            params={"mock": True},
        )
        assert mock_params["is_fallback"] is True

        # 3. Failing Optuna study falls back to defaults with is_fallback=True
        import lightgbm as lgb

        def failing_fit(self: Any, X: Any, y: Any, **fit_kwargs: Any) -> Any:
            raise RuntimeError("Engine failure")

        monkeypatch.setattr(lgb.LGBMClassifier, "fit", failing_fit)
        fallback_params = optimize_hyperparameters(
            family="lightgbm",
            inner_train_df=inner_train,
            inner_val_df=inner_val,
            feature_cols=feature_cols,
            n_trials=2,
            dry_run=False,
        )
        assert fallback_params["is_fallback"] is True
    def test_normalize_family_and_arm(self) -> None:
        """Test family and arm normalization and error handling."""
        assert normalize_family("lightgbm") == "lightgbm"
        assert normalize_family("LGBM") == "lightgbm"
        assert normalize_family("xgboost") == "xgboost"
        assert normalize_family("xgb") == "xgboost"
        with pytest.raises(ValueError, match="Unsupported GBDT family"):
            normalize_family("catboost")

        assert normalize_arm("control") == "control"
        assert normalize_arm("default") == "control"
        assert normalize_arm("challenger") == "challenger"
        assert normalize_arm("hpo") == "challenger"
        with pytest.raises(ValueError, match="Unsupported arm"):
            normalize_arm("invalid_arm")


class TestCliAndConfig:
    """Test CLI argument parsing and configuration file loading."""

    def test_parse_args_defaults_and_custom(self) -> None:
        """Assert CLI flags are correctly parsed."""
        args = parse_args(["--family", "lightgbm", "--arm", "control", "--n-trials", "5", "--smoke", "--dry-run"])
        assert args.family == "lightgbm"
        assert args.arm == "control"
        assert args.n_trials == 5
        assert args.smoke is True
        assert args.dry_run is True

    def test_load_config_validity(self) -> None:
        """Assert gbdt_hpo_campaign.yaml loads with valid schema."""
        if not DEFAULT_CONFIG_PATH.is_file():
            pytest.skip(f"Config file {DEFAULT_CONFIG_PATH} does not exist yet")

        config = load_config(DEFAULT_CONFIG_PATH)
        assert "models" in config
        assert "features" in config
        assert "lightgbm" in config["models"]
        assert "xgboost" in config["models"]
        assert config.get("n_trials", 15) == 15
        assert config.get("smoke_n_trials", 2) == 2


class TestSchemaConformity:
    """Test that generated OOF records strictly conform to CANDIDATE_OOF_SCHEMA."""

    def test_dry_run_generates_conforming_artifacts(self, tmp_path: Path) -> None:
        """Run dry-run execution and verify 24-field schema and transport validation."""
        output_dir = tmp_path / "artifacts"
        config_path = DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            pytest.skip(f"Config file {config_path} does not exist yet")

        config = load_config(config_path)
        oof_paths = run_gbdt_hpo(
            family="lightgbm",
            arm="both",
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            smoke=True,
            dry_run=True,
            n_trials=2,
        )

        assert len(oof_paths) == 2  # control and challenger
        for p in oof_paths:
            assert p.is_file()
            table = pq.read_table(p)
            assert table.schema.names == CANDIDATE_OOF_SCHEMA.names
            assert len(table.schema.names) == 24

            # Zero nulls
            for col in CANDIDATE_OOF_SCHEMA.names:
                assert table[col].null_count == 0

            # Corresponding transport validation JSON exists
            transport_json = p.parent / p.name.replace("_candidate_oof.parquet", "_transport_validation.json")
            assert transport_json.is_file()

            import json
            data = json.loads(transport_json.read_text())
            assert data["schema_version"] == 1
            assert "best_params_per_fold" in data
            assert data["validation_status"] == "passed"

            # Corresponding manifest JSON exists
            manifest_json = p.parent / p.name.replace("_candidate_oof.parquet", "_manifest.json")
            assert manifest_json.is_file()

    def test_full_cv_dry_run_no_hash_mismatch(self, tmp_path: Path) -> None:
        """Assert run_gbdt_hpo control runs across all 3 repeats and 4 folds without hash mismatch."""


        b_ids = [f"b_{i}" for i in range(8)]
        split_records = []
        for rep in [0, 1, 2]:
            for b_idx, b_id in enumerate(b_ids):
                fold = b_idx % 4
                split_records.append({
                    "building_id": b_id,
                    "split": "dev",
                    "repeat": rep,
                    "fold": fold,
                })
        split_df = pl.DataFrame(split_records)
        split_tmp = tmp_path / "splits.parquet"
        split_df.write_parquet(split_tmp)

        train_records = []
        base_time = datetime.datetime(2016, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
        for b_id in b_ids:
            for hour in range(4):
                train_records.append({
                    "building_id": b_id,
                    "timestamp": base_time + datetime.timedelta(hours=hour),
                    "meter_reading": 10.0,
                    "square_feet": 1000.0,
                    "primary_use": "Education",
                    "anomaly": hour % 2,
                })
        train_df = pl.DataFrame(train_records).with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        train_tmp = tmp_path / "train.parquet"
        train_df.write_parquet(train_tmp)

        cfg_tmp = tmp_path / "campaign.yaml"
        cfg = {
            "target": "anomaly",
            "split_manifest": str(split_tmp),
            "data_path": str(train_tmp),
            "features": {
                "predictors": ["meter_reading", "square_feet", "primary_use"],
            },
            "models": {
                "lightgbm": {
                    "model_family": "lightgbm",
                    "recipe_id": "recipe_gbdt_hpo_lightgbm",
                    "model_config_id": "cfg_gbdt_hpo_lightgbm_default",
                    "params": {"learning_rate": 0.05, "n_estimators": 10},
                }
            },
        }
        with cfg_tmp.open("w") as f:
            yaml.dump(cfg, f)

        out_dir = tmp_path / "output"
        paths = run_gbdt_hpo(
            family="lightgbm",
            arm="control",
            config=cfg,
            config_path=cfg_tmp,
            output_dir=out_dir,
            smoke=False,
            dry_run=True,
        )

        assert len(paths) == 1
        oof = pl.read_parquet(paths[0])
        assert set(oof["repeat"].unique().to_list()) == {0, 1, 2}
        assert set(oof["fold"].unique().to_list()) == {0, 1, 2, 3}
        # Assert fixed canonical model_fit_config_sha256 across all repeats and folds
        unique_hashes = oof["model_fit_config_sha256"].unique().to_list()
        assert len(unique_hashes) == 1
        assert len(unique_hashes[0]) == 64
