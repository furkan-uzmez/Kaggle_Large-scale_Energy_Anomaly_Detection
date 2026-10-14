"""Unit tests for GBDT Loss Objectives architecture, custom loss functions, and CLI runner.

Verifies:
1. Mathematical properties of focal loss gradients:
   - Easy correct samples have near-zero gradient.
   - Hard samples (false negatives / false positives) have large gradients.
   - Hessian is finite and strictly positive (floored at 1e-6).
   - Numerical finite-difference verification matches analytical gradient.
2. Custom objective and evaluation callbacks for LightGBM, CatBoost, and XGBoost.
3. Configuration loading and CLI argument handling.
4. Dry-run / mock execution across all 3 families in all 3 loss arms (logloss, weighted, focal).
5. Canonical schema conformity of output OOF records to CANDIDATE_OOF_SCHEMA and transport validation JSON.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from scripts.train_gbdt_loss_objectives import (
    DEFAULT_CONFIG_PATH,
    is_forbidden_predictor,
    load_config,
    parse_args,
    run_gbdt_loss_objectives,
    validate_predictor_cols,
)

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.gbdt_loss_functions import (
    HESSIAN_MIN_FLOOR,
    SUPPORTED_FAMILIES,
    SUPPORTED_LOSS_ARMS,
    CatBoostFocalObjective,
    binary_focal_loss,
    binary_focal_loss_grad_hess,
    focal_loss_lgb,
    focal_loss_lgb_eval,
    focal_loss_xgb,
    focal_loss_xgb_eval,
    get_loss_arm_params,
    make_focal_loss_lgb,
    make_focal_loss_xgb,
    mock_predict_loss_partition,
    normalize_family,
    normalize_loss,
    train_loss_objective_partition,
)
from project.models.gbdt_standalone import (
    MissingGbdtDependencyError,
    is_gbdt_available,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def synthetic_data() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Generate synthetic train and validation DataFrames."""
    rng = np.random.default_rng(42)
    n_train = 60
    n_val = 30

    train_anomaly = [1 if i % 5 == 0 else 0 for i in range(n_train)]
    val_anomaly = [1 if i % 4 == 0 else 0 for i in range(n_val)]

    train_dict: dict[str, Any] = {
        "building_id": [1] * 30 + [2] * 30,
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC)
            + datetime.timedelta(hours=i)
            for i in range(n_train)
        ],
        "anomaly": train_anomaly,
    }
    val_dict: dict[str, Any] = {
        "building_id": [3] * n_val,
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC)
            + datetime.timedelta(hours=i)
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
    ]
    for col in feature_cols:
        train_dict[col] = rng.normal(50.0, 10.0, size=n_train).tolist()
        val_dict[col] = rng.normal(50.0, 10.0, size=n_val).tolist()

    return pl.DataFrame(train_dict), pl.DataFrame(val_dict), feature_cols


@pytest.fixture
def synthetic_split_manifest() -> pl.DataFrame:
    """Generate a minimal synthetic split manifest covering buildings 1, 2, 3."""
    records = []
    for rep in [0]:
        for b_id, fld in [(1, 0), (2, 0), (3, 1)]:
            records.append(
                {
                    "building_id": b_id,
                    "repeat": rep,
                    "fold": fld,
                    "split": "dev",
                    "seed": 42 + rep,
                    "site_id": 1,
                    "primary_use": "Education",
                    "anomaly_ratio": 0.1,
                    "record_count": 100,
                }
            )
    return pl.DataFrame(records)


# ==============================================================================
# Suite 1: Mathematical Properties of Focal Loss Gradients
# ==============================================================================


class TestFocalLossMathematics:
    """Rigorous mathematical tests for binary focal loss gradient and hessian."""

    def test_easy_correct_samples_have_near_zero_gradient(self) -> None:
        """Easy confident samples (y=1 with large +z, y=0 with large -z) produce near-zero gradients."""
        # y=1 with large positive logits (high predicted probability)
        easy_pos_logits = np.array([3.0, 4.0, 5.0, 6.0])
        easy_pos_targets = np.ones_like(easy_pos_logits)
        grad_pos, hess_pos = binary_focal_loss_grad_hess(
            easy_pos_logits, easy_pos_targets, alpha=0.25, gamma=2.0
        )

        assert np.all(np.abs(grad_pos) < 1e-2), (
            f"Easy positive samples should have near-zero gradient: {grad_pos}"
        )
        assert np.all(hess_pos >= HESSIAN_MIN_FLOOR)

        # y=0 with large negative logits (low predicted probability)
        easy_neg_logits = np.array([-3.0, -4.0, -5.0, -6.0])
        easy_neg_targets = np.zeros_like(easy_neg_logits)
        grad_neg, hess_neg = binary_focal_loss_grad_hess(
            easy_neg_logits, easy_neg_targets, alpha=0.25, gamma=2.0
        )

        assert np.all(np.abs(grad_neg) < 1e-2), (
            f"Easy negative samples should have near-zero gradient: {grad_neg}"
        )
        assert np.all(hess_neg >= HESSIAN_MIN_FLOOR)

    def test_hard_samples_have_large_gradient(self) -> None:
        """Hard misclassified samples (false negatives and false positives) receive large gradients."""
        # y=1 with negative logit (missed anomaly / false negative)
        hard_pos_logits = np.array([-2.0, -1.0])
        hard_pos_targets = np.ones_like(hard_pos_logits)
        grad_pos, _ = binary_focal_loss_grad_hess(
            hard_pos_logits, hard_pos_targets, alpha=0.25, gamma=2.0
        )

        # Gradient should be negative (pushing logit upwards) and substantial in magnitude
        assert np.all(grad_pos < 0.0)
        assert np.all(np.abs(grad_pos) > 0.15), (
            f"Hard positive samples must receive large gradients: {grad_pos}"
        )

        # y=0 with positive logit (false alarm / false positive)
        hard_neg_logits = np.array([1.0, 2.0])
        hard_neg_targets = np.zeros_like(hard_neg_logits)
        grad_neg, _ = binary_focal_loss_grad_hess(
            hard_neg_logits, hard_neg_targets, alpha=0.25, gamma=2.0
        )

        # Gradient should be positive (pushing logit downwards) and substantial in magnitude
        assert np.all(grad_neg > 0.0)
        assert np.all(np.abs(grad_neg) > 0.3), (
            f"Hard negative samples must receive large gradients: {grad_neg}"
        )

    def test_hessian_strictly_positive_and_finite(self) -> None:
        """Hessian is strictly positive (floored at 1e-6) and finite across wide logit grid."""
        grid_z = np.linspace(-30.0, 30.0, num=300)
        for y_val in [0.0, 1.0]:
            y_arr = np.full_like(grid_z, y_val)
            grad, hess = binary_focal_loss_grad_hess(grid_z, y_arr)

            assert np.all(np.isfinite(grad)), "Gradients must all be finite"
            assert np.all(np.isfinite(hess)), "Hessians must all be finite"
            assert np.all(hess >= HESSIAN_MIN_FLOOR), (
                f"Hessian must be strictly >= {HESSIAN_MIN_FLOOR}"
            )

    def test_numerical_finite_difference_agreement(self) -> None:
        """Analytical gradient matches central finite difference of focal loss with high precision."""
        eps = 1e-6
        test_z_points = [-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0]
        alpha = 0.25
        gamma = 2.0

        for y_val in [0.0, 1.0]:
            for z0 in test_z_points:
                # Analytical gradient
                g_ana, _ = binary_focal_loss_grad_hess(
                    np.array([z0]), np.array([y_val]), alpha=alpha, gamma=gamma
                )
                # Numerical central difference: (L(z + eps) - L(z - eps)) / (2 * eps)
                l_plus = binary_focal_loss(
                    np.array([z0 + eps]), np.array([y_val]), alpha=alpha, gamma=gamma
                )
                l_minus = binary_focal_loss(
                    np.array([z0 - eps]), np.array([y_val]), alpha=alpha, gamma=gamma
                )
                g_num = (l_plus - l_minus) / (2.0 * eps)

                np.testing.assert_allclose(
                    g_ana[0],
                    g_num,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Discrepancy at z={z0}, y={y_val}",
                )

    def test_extreme_inputs_numerical_stability(self) -> None:
        """Extreme values (z=+-10000) do not cause overflow, NaN, or underflow."""
        extreme_z = np.array([-10000.0, -100.0, 0.0, 100.0, 10000.0])
        extreme_y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        grad, hess = binary_focal_loss_grad_hess(extreme_z, extreme_y)

        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))
        assert np.all(hess >= HESSIAN_MIN_FLOOR)

    def test_shape_mismatch_raises_error(self) -> None:
        """Mismatched preds and targets lengths raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            binary_focal_loss_grad_hess(np.array([1.0, 2.0]), np.array([1.0]))


# ==============================================================================
# Suite 2: Framework Custom Objectives & Callbacks
# ==============================================================================


class TestFrameworkCallbacks:
    """Tests for LightGBM, XGBoost, and CatBoost custom objective callbacks."""

    def test_lgb_custom_objective_and_eval(self) -> None:
        """Verify focal_loss_lgb and focal_loss_lgb_eval with mock LightGBM Dataset."""

        class MockLgbDataset:
            def __init__(self, labels: np.ndarray) -> None:
                self._labels = labels

            def get_label(self) -> np.ndarray:
                return self._labels

        preds = np.array([-1.5, 0.0, 1.5])
        dataset = MockLgbDataset(np.array([0.0, 1.0, 1.0]))

        grad, hess = focal_loss_lgb(preds, dataset)
        assert grad.shape == (3,)
        assert hess.shape == (3,)
        assert np.all(np.isfinite(grad))
        assert np.all(hess >= HESSIAN_MIN_FLOOR)

        # Eval metric
        name, val, is_higher_better = focal_loss_lgb_eval(preds, dataset)
        assert name == "focal_loss"
        assert isinstance(val, float)
        assert val > 0.0
        assert is_higher_better is False

    def test_xgb_custom_objective_and_eval(self) -> None:
        """Verify focal_loss_xgb and focal_loss_xgb_eval with mock XGBoost DMatrix."""

        class MockDMatrix:
            def __init__(self, labels: np.ndarray) -> None:
                self._labels = labels

            def get_label(self) -> np.ndarray:
                return self._labels

        preds = np.array([-2.0, 0.5, 2.0])
        dmatrix = MockDMatrix(np.array([0.0, 1.0, 0.0]))

        grad, hess = focal_loss_xgb(preds, dmatrix)
        assert grad.shape == (3,)
        assert hess.shape == (3,)
        assert np.all(np.isfinite(grad))
        assert np.all(hess >= HESSIAN_MIN_FLOOR)

        name, val = focal_loss_xgb_eval(preds, dmatrix)
        assert name == "focal_loss"
        assert isinstance(val, float)
        assert val > 0.0

    def test_catboost_custom_objective(self) -> None:
        """Verify CatBoostFocalObjective calc_ders_range convention."""
        obj = CatBoostFocalObjective(alpha=0.25, gamma=2.0)
        approxes = [-1.0, 0.0, 1.0]
        targets = [0.0, 1.0, 1.0]
        ders = obj.calc_ders_range(approxes, targets)

        assert len(ders) == 3
        for d1, d2 in ders:
            # CatBoost convention: ders1 = -grad, ders2 = -hess
            assert isinstance(d1, float)
            assert isinstance(d2, float)
            assert d2 <= -HESSIAN_MIN_FLOOR

    def test_factories_custom_hyperparameters(self) -> None:
        """Verify make_focal_loss_lgb and make_focal_loss_xgb accept custom alpha and gamma."""
        custom_lgb = make_focal_loss_lgb(alpha=0.5, gamma=3.0)
        custom_xgb = make_focal_loss_xgb(alpha=0.5, gamma=3.0)

        targets = np.array([0.0, 1.0])
        preds = np.array([0.0, 0.0])
        # Sklearn objective signature: obj(y_true, y_pred)
        g_lgb, h_lgb = custom_lgb(targets, preds)
        g_xgb, h_xgb = custom_xgb(targets, preds)

        g_expected, h_expected = binary_focal_loss_grad_hess(
            preds=preds, targets=targets, alpha=0.5, gamma=3.0
        )

        np.testing.assert_array_equal(g_lgb, g_expected)
        np.testing.assert_array_equal(h_lgb, h_expected)
        np.testing.assert_array_equal(g_xgb, g_expected)
        np.testing.assert_array_equal(h_xgb, h_expected)


# ==============================================================================
# Suite 3: Loss Arm Configuration & CLI Argument Handling
# ==============================================================================


class TestConfigurationAndCLI:
    """Tests for YAML configuration loading, schema verification, and CLI argument parsing."""

    def test_config_file_structure(self) -> None:
        """Verify gbdt_loss_objectives_campaign.yaml structure and declarations."""
        assert DEFAULT_CONFIG_PATH.is_file(), (
            f"Config file missing at {DEFAULT_CONFIG_PATH}"
        )
        config = load_config(DEFAULT_CONFIG_PATH)

        assert config["target"] == "anomaly"
        assert config["positive_label"] == 1
        assert config["folds"] == 4
        assert config["repeats"] == 3
        assert set(config["split_seeds"]) == {42, 43, 44}
        assert config["loss_arms"] == ["logloss", "weighted", "focal"]

        # Verify safe verified 31 predictors
        predictors = config["features"]["predictors"]
        assert len(predictors) == 31, (
            f"Expected exactly 31 predictors, found {len(predictors)}"
        )
        assert "building_hour" not in predictors
        assert "building_meter" not in predictors
        assert is_forbidden_predictor("building_hour") is True
        assert is_forbidden_predictor("building_meter") is True

        # Assert no forbidden columns entered predictors
        for col in predictors:
            assert not is_forbidden_predictor(col), (
                f"Forbidden column in predictors: {col}"
            )
            assert not col.startswith("gte_"), f"gte_* column in predictors: {col}"
            assert not col.endswith("_id"), f"*_id column in predictors: {col}"

        # Verify all 3 models have all 3 loss arms
        models = config["models"]
        for fam in ("lightgbm", "catboost", "xgboost"):
            assert fam in models, f"Missing family '{fam}' in models config"
            arms = models[fam]["arms"]
            for arm in ("logloss", "weighted", "focal"):
                assert arm in arms, f"Missing arm '{arm}' in {fam} config"

        assert (
            models["catboost"]["arms"]["focal"]["loss_function"]
            == "Focal:focal_alpha=0.25;focal_gamma=2.0"
        )

    def test_validate_predictor_cols_guards(self) -> None:
        """Verify validate_predictor_cols enforces non-empty, unique, non-leaking columns."""
        valid_cols = ["meter_reading", "square_feet", "hour"]
        assert validate_predictor_cols(valid_cols) == valid_cols

        # Empty
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_predictor_cols([])

        # Duplicate
        with pytest.raises(ValueError, match="Duplicate predictor"):
            validate_predictor_cols(["meter_reading", "meter_reading"])

        # Forbidden gte_
        with pytest.raises(ValueError, match="Forbidden predictor column"):
            validate_predictor_cols(["meter_reading", "gte_building_id"])

        # Forbidden _id
        with pytest.raises(ValueError, match="Forbidden predictor column"):
            validate_predictor_cols(["meter_reading", "site_id"])

        # Forbidden target
        with pytest.raises(ValueError, match="Forbidden predictor column"):
            validate_predictor_cols(["meter_reading", "anomaly"])

    def test_get_loss_arm_params_dispatch(self) -> None:
        """Verify get_loss_arm_params sets proper objectives and weights across families and arms."""
        for fam in SUPPORTED_FAMILIES:
            # logloss
            p_log = get_loss_arm_params(fam, "logloss")
            assert isinstance(p_log, dict)

            # weighted
            p_wt = get_loss_arm_params(fam, "weighted", pos_weight=8.0)
            if fam in ("lightgbm", "xgboost"):
                assert p_wt.get("scale_pos_weight") == 8.0
            elif fam == "catboost":
                assert p_wt.get("class_weights") == [1.0, 8.0]

            # focal
            p_foc = get_loss_arm_params(fam, "focal", alpha=0.25, gamma=2.0)
            if fam == "catboost":
                assert (
                    p_foc.get("loss_function")
                    == "Focal:focal_alpha=0.25;focal_gamma=2.0"
                )
            else:
                assert callable(p_foc.get("objective"))

    def test_normalize_identifiers(self) -> None:
        """Verify normalizers handle aliases and reject invalid inputs."""
        assert normalize_family("LightGBM") == "lightgbm"
        assert normalize_family("CB") == "catboost"
        assert normalize_family("xgb") == "xgboost"
        with pytest.raises(ValueError, match="Unsupported GBDT family"):
            normalize_family("random_forest")

        assert normalize_loss("log_loss") == "logloss"
        assert normalize_loss("class_weighted") == "weighted"
        assert normalize_loss("focal_loss") == "focal"
        with pytest.raises(ValueError, match="Unsupported loss arm"):
            normalize_loss("hinge")

    def test_cli_argument_parsing(self) -> None:
        """Verify CLI argument defaults and overrides."""
        args = parse_args(
            ["--family", "lightgbm", "--loss", "focal", "--smoke", "--dry-run"]
        )
        assert args.family == "lightgbm"
        assert args.loss == "focal"
        assert args.smoke is True
        assert args.dry_run is True
        assert args.config == DEFAULT_CONFIG_PATH
        assert args.output_dir is None

        # Invalid family
        with pytest.raises(SystemExit):
            parse_args(["--family", "invalid_fam"])

        # Invalid loss
        with pytest.raises(SystemExit):
            parse_args(["--loss", "invalid_loss"])


# ==============================================================================
# Suite 4: Mock / Dry-Run Partition Training
# ==============================================================================


class TestMockPartitionTraining:
    """Tests for dry-run partition training across all families and loss arms."""

    def test_dry_run_partition_all_arms_all_families(
        self, synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """Dry-run partition training succeeds across all 9 combinations."""
        train_df, val_df, feature_cols = synthetic_data

        for fam in SUPPORTED_FAMILIES:
            for arm in SUPPORTED_LOSS_ARMS:
                preds = train_loss_objective_partition(
                    family=fam,
                    loss=arm,
                    train_df=train_df,
                    val_df=val_df,
                    feature_cols=feature_cols,
                    target_col="anomaly",
                    params={"seed": 42},
                    dry_run=True,
                )

                assert isinstance(preds, np.ndarray)
                assert preds.dtype == np.float64
                assert preds.shape == (val_df.height,)
                assert np.all(np.isfinite(preds))
                assert np.all((preds >= 0.0) & (preds <= 1.0))

                # Deterministic reproducibility check
                preds_repeat = train_loss_objective_partition(
                    family=fam,
                    loss=arm,
                    train_df=train_df,
                    val_df=val_df,
                    feature_cols=feature_cols,
                    target_col="anomaly",
                    params={"seed": 42},
                    dry_run=True,
                )
                np.testing.assert_array_equal(preds, preds_repeat)

    def test_mock_predict_loss_partition_prior_shift(self) -> None:
        """Weighted loss arm shifts probabilities upward reflecting false-negative penalty."""
        y_train = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        val_len = 20

        p_log = mock_predict_loss_partition(y_train, val_len, seed=42, loss="logloss")
        p_wt = mock_predict_loss_partition(y_train, val_len, seed=42, loss="weighted")

        assert np.mean(p_wt) > np.mean(p_log), (
            "Weighted loss should have higher mean probability than unweighted logloss"
        )

    def test_dry_run_false_without_dependency_raises(
        self, synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]]
    ) -> None:
        """In active environment without GBDT installed, dry_run=False raises MissingGbdtDependencyError."""
        train_df, val_df, feature_cols = synthetic_data

        for fam in SUPPORTED_FAMILIES:
            if not is_gbdt_available(fam):
                with pytest.raises(MissingGbdtDependencyError):
                    train_loss_objective_partition(
                        family=fam,
                        loss="focal",
                        train_df=train_df,
                        val_df=val_df,
                        feature_cols=feature_cols,
                        target_col="anomaly",
                        params={"seed": 42},
                        dry_run=False,
                    )


# ==============================================================================
# Suite 5: End-to-End Dry-Run & Canonical Schema Conformity
class TestEndToEndSchemaConformity:
    """Verifies end-to-end execution generates valid Parquet conforming to CANDIDATE_OOF_SCHEMA."""

    def test_full_script_dry_run_conforms_to_schema(self, tmp_path: Path) -> None:
        """Execute run_gbdt_loss_objectives in dry-run smoke mode on campaign dataset."""
        config = load_config(DEFAULT_CONFIG_PATH)
        out_dir = tmp_path / "test_gbdt_loss_output"

        # Execute for lightgbm across all 3 loss arms
        oof_paths = run_gbdt_loss_objectives(
            family="lightgbm",
            loss="all",
            config=config,
            config_path=DEFAULT_CONFIG_PATH,
            output_dir=out_dir,
            smoke=True,
            dry_run=True,
        )

        assert len(oof_paths) == 3, (
            f"Expected 3 OOF files (one per loss arm), got {len(oof_paths)}"
        )

        for oof_path in oof_paths:
            assert oof_path.is_file(), f"OOF file missing at {oof_path}"

            # 1. Check PyArrow schema conformity to CANDIDATE_OOF_SCHEMA
            table = pq.read_table(oof_path)
            assert table.schema.names == CANDIDATE_OOF_SCHEMA.names, (
                f"Field names mismatch: {table.schema.names} vs {CANDIDATE_OOF_SCHEMA.names}"
            )
            for field in CANDIDATE_OOF_SCHEMA:
                assert table.schema.field(field.name).type == field.type, (
                    f"Type mismatch for {field.name}: {table.schema.field(field.name).type} vs {field.type}"
                )

            # 2. Check metadata
            assert table.schema.metadata is not None
            assert table.schema.metadata.get(b"candidate_schema_version") == b"1"

            # 3. Check zero nulls across all columns
            for field in CANDIDATE_OOF_SCHEMA:
                assert table.column(field.name).null_count == 0, (
                    f"Column '{field.name}' contains nulls"
                )

            # 4. Check contract values
            df = pl.read_parquet(oof_path)
            assert set(df["score_kind"].to_list()) == {"positive_probability"}
            assert set(df["positive_label"].to_list()) == {1}
            assert set(df["execution_mode"].to_list()) == {"synthetic_smoke"}
            assert set(df["non_comparative"].to_list()) == {True}
            assert set(df["n03_eligible"].to_list()) == {False}

            preds = df["prediction"].to_numpy()
            assert np.all((preds >= 0.0) & (preds <= 1.0))

            # 5. Check corresponding transport validation JSON
            dest_dir = oof_path.parent
            transport_files = list(dest_dir.glob("*_transport_validation.json"))
            assert len(transport_files) == 1, (
                f"Expected exactly 1 transport validation JSON in {dest_dir}"
            )

            manifest_content = json.loads(transport_files[0].read_text())
            assert manifest_content["schema_version"] == 1
            assert manifest_content["non_comparative"] is True
            assert manifest_content["n03_eligible"] is False
            assert manifest_content["records_count"] == table.num_rows
            assert manifest_content["oof_parquet"] == str(oof_path)

    def test_synthetic_in_memory_dry_run_conforms_to_schema(
        self,
        tmp_path: Path,
        synthetic_data: tuple[pl.DataFrame, pl.DataFrame, list[str]],
        synthetic_split_manifest: pl.DataFrame,
    ) -> None:
        """Execute run_gbdt_loss_objectives with in-memory synthetic DataFrames."""
        train_df, _, feature_cols = synthetic_data
        config = load_config(DEFAULT_CONFIG_PATH)
        config["features"]["predictors"] = feature_cols
        out_dir = tmp_path / "test_gbdt_loss_synthetic_output"
        config["split_manifest"] = "outputs/splits/nonexistent_split.parquet"
        config["data_path"] = "outputs/prepared/nonexistent_data.parquet"

        oof_paths = run_gbdt_loss_objectives(
            family="catboost",
            loss="focal",
            config=config,
            config_path=DEFAULT_CONFIG_PATH,
            output_dir=out_dir,
            smoke=True,
            dry_run=True,
            split_df=synthetic_split_manifest,
            train_df=train_df,
        )

        assert len(oof_paths) == 1
        oof_path = oof_paths[0]
        table = pq.read_table(oof_path)
        assert table.schema.names == CANDIDATE_OOF_SCHEMA.names
        for field in CANDIDATE_OOF_SCHEMA:
            assert table.schema.field(field.name).type == field.type
            assert table.column(field.name).null_count == 0
            df = pl.read_parquet(oof_path)
            assert set(df["score_kind"].to_list()) == {"positive_probability"}
            assert set(df["positive_label"].to_list()) == {1}
            assert set(df["execution_mode"].to_list()) == {"synthetic_smoke"}
            assert set(df["non_comparative"].to_list()) == {True}
            assert set(df["n03_eligible"].to_list()) == {False}

            preds = df["prediction"].to_numpy()
            assert np.all((preds >= 0.0) & (preds <= 1.0))

            # 4b. Check valid 64-char hex SHA-256 fallback placeholders
            expected_split_sha = hashlib.sha256(b"mock_split").hexdigest()
            expected_data_sha = hashlib.sha256(b"mock_data").hexdigest()
            assert df["split_sha256"][0] == expected_split_sha
            assert df["data_sha256"][0] == expected_data_sha
            # 5. Check corresponding transport validation JSON
            dest_dir = oof_path.parent
            transport_files = list(dest_dir.glob("*_transport_validation.json"))
            assert len(transport_files) == 1, (
                f"Expected exactly 1 transport validation JSON in {dest_dir}"
            )

            manifest_content = json.loads(transport_files[0].read_text())
            assert manifest_content["schema_version"] == 1
            assert manifest_content["non_comparative"] is True
            assert manifest_content["n03_eligible"] is False
            assert manifest_content["records_count"] == table.num_rows
            assert manifest_content["oof_parquet"] == str(oof_path)
            assert manifest_content["split_sha256"] == expected_split_sha
            assert manifest_content["data_sha256"] == expected_data_sha
