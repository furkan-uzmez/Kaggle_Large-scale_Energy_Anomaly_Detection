"""Unit tests for GBDT feature selection module, CLI runner, and configuration.

Verifies:
1. Strict fold-local selection isolation (val_df never affects selected features).
2. Configuration parsing and CLI argument handling.
3. Dry-run / mock execution across all 3 families (LightGBM, CatBoost, XGBoost) in both regimes.
4. Schema conformity of output OOF records to CANDIDATE_OOF_SCHEMA and transport validation JSON.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml
from scripts.train_gbdt_feature_selection import (
    DEFAULT_CONFIG_PATH,
    compute_selection_stability,
    is_forbidden_predictor,
    load_config,
    parse_args,
    run_gbdt_feature_selection,
    validate_predictor_cols,
)

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.gbdt_feature_selection import (
    select_top_k_features,
    train_fs_partition,
)


@pytest.fixture
def synthetic_train_val_data() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Create synthetic datasets with informative and noise features."""
    rng = np.random.default_rng(42)
    n_train = 60
    n_val = 30

    # 10 features: feat_0 to feat_3 are signal-correlated with anomaly, feat_4 to feat_9 are pure noise
    train_anomaly = [1 if i % 4 == 0 else 0 for i in range(n_train)]
    val_anomaly = [1 if i % 3 == 0 else 0 for i in range(n_val)]

    train_data: dict[str, list] = {
        "building_id": [1] * (n_train // 2) + [2] * (n_train // 2),
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(hours=i)
            for i in range(n_train)
        ],
        "anomaly": train_anomaly,
    }
    val_data: dict[str, list] = {
        "building_id": [3] * n_val,
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC) + datetime.timedelta(hours=i)
            for i in range(n_val)
        ],
        "anomaly": val_anomaly,
    }

    feature_cols = [f"feat_{i}" for i in range(10)]
    for i, col in enumerate(feature_cols):
        if i < 4:
            # Strong signal in train
            train_data[col] = [
                float(y * 10.0 + rng.normal(0, 0.5)) for y in train_anomaly
            ]
            val_data[col] = [
                float(y * 10.0 + rng.normal(0, 0.5)) for y in val_anomaly
            ]
        else:
            # Pure noise
            train_data[col] = rng.normal(0, 1.0, size=n_train).tolist()
            val_data[col] = rng.normal(0, 1.0, size=n_val).tolist()

    return pl.DataFrame(train_data), pl.DataFrame(val_data), feature_cols


class TestFoldLocalIsolation:
    """Tests ensuring zero data leakage in feature selection."""

    def test_select_top_k_never_sees_validation_data(
        self, synthetic_train_val_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """Assert select_top_k_features signature and execution depends strictly on train_df."""
        train_df, val_df, feature_cols = synthetic_train_val_data

        # Selection on train_df
        selected_1 = select_top_k_features(
            model_family="lightgbm",
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="anomaly",
            k=4,
            params={"mock": True},
        )
        assert len(selected_1) == 4
        assert isinstance(selected_1, list)
        assert all(isinstance(f, str) for f in selected_1)

        # Mutate validation data completely (reverse target, add spurious correlations)
        mutated_val_df = val_df.with_columns(
            (1 - pl.col("anomaly")).alias("anomaly"),
            (pl.col("feat_9") * 1000.0).alias("feat_9"),
        )
        assert mutated_val_df.height == val_df.height

        # Selection again on train_df - must be byte-for-byte identical because val_df is not used
        selected_2 = select_top_k_features(
            model_family="lightgbm",
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="anomaly",
            k=4,
            params={"mock": True},
        )
        assert selected_1 == selected_2

    def test_train_fs_partition_isolation(
        self, synthetic_train_val_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """Assert train_fs_partition selects features without validation data influencing selection."""
        train_df, val_df, feature_cols = synthetic_train_val_data

        preds1, feats1 = train_fs_partition(
            model_family="lightgbm",
            regime="selected",
            train_df=train_df,
            val_df=val_df,
            all_features=feature_cols,
            target_col="anomaly",
            k=5,
            params={"mock": True},
            dry_run=True,
        )

        # Mutate val_df
        poisoned_val = val_df.with_columns(
            (pl.col("feat_8") + 999.0).alias("feat_8"),
            pl.lit(1).alias("anomaly"),
        )
        preds2, feats2 = train_fs_partition(
            model_family="lightgbm",
            regime="selected",
            train_df=train_df,
            val_df=poisoned_val,
            all_features=feature_cols,
            target_col="anomaly",
            k=5,
            params={"mock": True},
            dry_run=True,
        )

        assert feats1 == feats2
        assert len(feats1) == 5
        assert preds1.shape == (val_df.height,)
        assert preds2.shape == (poisoned_val.height,)

    def test_passthrough_regime_keeps_all_features(
        self, synthetic_train_val_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """Passthrough regime must retain 100% of available features."""
        train_df, val_df, feature_cols = synthetic_train_val_data
        preds, feats = train_fs_partition(
            model_family="xgboost",
            regime="passthrough",
            train_df=train_df,
            val_df=val_df,
            all_features=feature_cols,
            target_col="anomaly",
            k=3,
            params={"mock": True},
            dry_run=True,
        )
        assert feats == feature_cols
        assert preds.shape == (val_df.height,)


class TestConfigurationAndCLI:
    """Tests covering YAML config structure and CLI parsing."""

    def test_yaml_configuration_contract(self) -> None:
        """Validate configs/experiments/gbdt_feature_selection_campaign.yaml format."""
        config_path = DEFAULT_CONFIG_PATH
        assert config_path.is_file(), f"Missing campaign config: {config_path}"

        config = load_config(config_path)
        assert config["schema_version"] == 1
        assert config["k"] == 30
        assert "passthrough" in config["regimes"]
        assert "selected" in config["regimes"]

        # Ensure all 57 available columns from train_joined are declared
        available_cols = config.get("available_columns", [])
        assert len(available_cols) == 57
        assert "building_id" in available_cols
        assert "timestamp" in available_cols
        assert "anomaly" in available_cols

        # Predictor pool excludes metadata, target, site_id, and unverified gte family
        predictors = config.get("features", {}).get("predictors", [])
        assert "anomaly" not in predictors
        assert "building_id" not in predictors
        assert "site_id" not in predictors
        assert "timestamp" not in predictors
        assert "building_hour" not in predictors
        assert "building_meter" not in predictors
        assert len(predictors) == 31
        assert not any(p.startswith("gte_") for p in predictors)
        assert not any(p.endswith("_id") for p in predictors)

        numeric = config.get("features", {}).get("numeric", [])
        categorical = config.get("features", {}).get("categorical", [])
        assert len(numeric) == 30
        assert len(categorical) == 1
        assert set(predictors) == set(numeric) | set(categorical)
        # Models declared
        for fam in ("lightgbm", "catboost", "xgboost"):
            assert fam in config["models"]
            assert "params" in config["models"][fam]

    def test_cli_parsing(self) -> None:
        """Verify CLI argument handling."""
        args = parse_args(["--family", "lightgbm", "--regime", "selected", "--k", "25", "--dry-run", "--smoke"])
        assert args.family == "lightgbm"
        assert args.regime == "selected"
        assert args.k == 25
        assert args.dry_run is True
        assert args.smoke is True

        args_all = parse_args(["--family", "all", "--regime", "both"])
        assert args_all.family == "all"
        assert args_all.regime == "both"


    def test_predictor_pool_validation_guards(self) -> None:
        """Verify validate_predictor_cols rejects forbidden/leakage columns."""
        valid = ["meter_reading", "air_temperature", "wind_speed"]
        assert validate_predictor_cols(valid) == valid
        assert is_forbidden_predictor("site_id") is True
        assert is_forbidden_predictor("gte_hour") is True
        assert is_forbidden_predictor("building_id") is True
        assert is_forbidden_predictor("meter_reading") is False
        assert is_forbidden_predictor("building_hour") is True
        assert is_forbidden_predictor("building_meter") is True

        # Reject unverified gte_* family
        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            validate_predictor_cols(["meter_reading", "gte_hour"])

        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            validate_predictor_cols(["gte_site_id"])

        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            validate_predictor_cols(["gte_meter_building_id"])

        # Reject ID columns
        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            validate_predictor_cols(["meter_reading", "site_id"])

        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            validate_predictor_cols(["meter_reading", "building_id"])

        # Reject metadata / target
        for reserved in ("timestamp", "anomaly", "target", "row_index", "split"):
            with pytest.raises(ValueError, match="violates anti-leakage contract"):
                validate_predictor_cols(["meter_reading", reserved])

        # Reject empty or duplicate
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_predictor_cols([])

        with pytest.raises(ValueError, match="Duplicate predictor column"):
            validate_predictor_cols(["meter_reading", "meter_reading"])

    def test_runner_rejects_forbidden_predictor_config(self, tmp_path: Path) -> None:
        """Verify run_gbdt_feature_selection fails fast if config introduces forbidden columns."""
        config = load_config(DEFAULT_CONFIG_PATH)
        bad_config = dict(config)
        bad_features = dict(config.get("features", {}))
        bad_features["predictors"] = list(bad_features["predictors"]) + ["gte_leak_test"]
        bad_config["features"] = bad_features

        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            run_gbdt_feature_selection(
                family="lightgbm",
                regime="selected",
                k=30,
                config=bad_config,
                config_path=DEFAULT_CONFIG_PATH,
                output_dir=tmp_path / "bad_test",
                smoke=True,
                dry_run=True,
            )

        # Also test injection of site_id
        bad_features["predictors"] = ["meter_reading", "site_id"]
        with pytest.raises(ValueError, match="violates anti-leakage contract"):
            run_gbdt_feature_selection(
                family="lightgbm",
                regime="selected",
                k=30,
                config=bad_config,
                config_path=DEFAULT_CONFIG_PATH,
                output_dir=tmp_path / "bad_test_site_id",
                smoke=True,
                dry_run=True,
            )

class TestMockExecutionMatrix:
    """Verify execution across all 3 families in both regimes."""

    @pytest.mark.parametrize("family", ["lightgbm", "catboost", "xgboost"])
    @pytest.mark.parametrize("regime", ["passthrough", "selected"])
    def test_all_families_and_regimes_dry_run(
        self,
        family: str,
        regime: str,
        synthetic_train_val_data: tuple[pl.DataFrame, pl.DataFrame, list[str]],
    ) -> None:
        """Each family/regime combination executes successfully in dry-run mode."""
        train_df, val_df, feature_cols = synthetic_train_val_data
        preds, feats = train_fs_partition(
            model_family=family,
            regime=regime,
            train_df=train_df,
            val_df=val_df,
            all_features=feature_cols,
            target_col="anomaly",
            k=4,
            params={"mock": True},
            dry_run=True,
        )
        assert preds.shape == (val_df.height,)
        assert np.all((preds >= 0.0) & (preds <= 1.0))
        if regime == "passthrough":
            assert len(feats) == len(feature_cols)
        else:
            assert len(feats) == 4


class TestRunnerAndSchemaConformity:
    """Test full script dry-run execution, OOF parquet schema, and stability JSON."""

    def test_full_script_dry_run_conforms_to_schema(self, tmp_path: Path) -> None:
        """Execute full runner in dry-run smoke mode and validate CANDIDATE_OOF_SCHEMA."""
        config_path = DEFAULT_CONFIG_PATH
        config = load_config(config_path)

        out_dir = tmp_path / "test_gbdt_fs_output"
        result_paths = run_gbdt_feature_selection(
            family="lightgbm",
            regime="both",
            k=30,
            config=config,
            config_path=config_path,
            output_dir=out_dir,
            smoke=True,
            dry_run=True,
        )

        assert len(result_paths) >= 1
        for oof_path in result_paths:
            assert oof_path.is_file()
            table = pq.read_table(oof_path)
            assert table.schema.names == CANDIDATE_OOF_SCHEMA.names
            for field in CANDIDATE_OOF_SCHEMA:
                assert table.schema.field(field.name).type == field.type
            # Canonical OOF validation check
            assert table.schema.metadata is not None
            assert table.schema.metadata.get(b"candidate_schema_version") == b"1"
            for field in CANDIDATE_OOF_SCHEMA:
                assert table.column(field.name).null_count == 0
            preds = table.column("prediction").to_numpy()
            assert np.all((preds >= 0.0) & (preds <= 1.0))
            assert set(table.column("execution_mode").to_pylist()) == {"synthetic_smoke"}
            assert all(table.column("non_comparative").to_pylist())
            assert not any(table.column("n03_eligible").to_pylist())
            assert set(table.column("score_kind").to_pylist()) == {"positive_probability"}
            assert set(table.column("positive_label").to_pylist()) == {1}
            # Check that transport validation JSON exists
            manifest_path = oof_path.parent / oof_path.name.replace("_candidate_oof.parquet", "_transport_validation.json")
            assert manifest_path.is_file()
            manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            assert manifest_data["schema_version"] == 1
            assert manifest_data["non_comparative"] is True
            assert manifest_data["n03_eligible"] is False
            assert "stability" in manifest_data
            assert "per_fold_selected_features" in manifest_data

            # Ensure no gte_* or *_id columns entered fold-selected feature sets
            per_fold = manifest_data["per_fold_selected_features"]
            assert len(per_fold) > 0
            for feats in per_fold.values():
                assert "site_id" not in feats
                assert "building_id" not in feats
                assert not any(f.startswith("gte_") for f in feats)
                assert not any(f.endswith("_id") for f in feats)
                assert all(f in config["features"]["predictors"] for f in feats)
    def test_selection_stability_computation(self) -> None:
        """Test Jaccard similarity and selection frequencies computation."""
        fold_features = {
            "0:0": ["f1", "f2", "f3"],
            "0:1": ["f1", "f2", "f4"],
            "0:2": ["f1", "f3", "f5"],
        }
        stability = compute_selection_stability(fold_features)
        assert "mean_pairwise_jaccard" in stability
        assert "jaccard_matrix" in stability
        assert "selection_frequency" in stability
        freq = stability["selection_frequency"]
        assert freq["f1"] == 1.0
        assert 0.0 <= stability["mean_pairwise_jaccard"] <= 1.0
