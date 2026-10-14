"""Unit tests for the three post-ensemble follow-up experiments.

Covers:
1. EXP-F01: Diurnal-Weighted Manifold Mixup (transition concentration, adaptive alpha, zero-leakage).
2. EXP-F02: Platt-Calibrated Champion (nested inner-CV, zero-leakage, monotonicity, ECE reduction).
3. EXP-F03: Skip-Connection Leaf Stacking (raw standard scaling + GBDT leaf OneHot, unseen category handling).
4. Metrics & Diagnostics (ECE, Brier, LogLoss, diurnal slice analysis).
5. OOF Parquet Schema Conformity (canonical CANDIDATE_OOF_SCHEMA, 24 non-null columns).
6. Submission Pipeline & CLI Contracts (submissions 39, 40, 41 generated and validated).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scripts.train_followup_experiments import (
    DEFAULT_CONFIG_PATH,
    load_config,
    parse_args,
    run_followup_experiments,
)

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.followup_experiments import (
    PlattCalibrator,
    SkipConnectionLeafClassifier,
    SkipConnectionLeafStackingEstimator,
    apply_fold_local_diurnal_mixup,
    compute_calibration_metrics,
    compute_diurnal_slice_metrics,
    run_nested_platt_calibration_partition,
    train_diurnal_mixup_partition,
    train_skip_leaf_partition,
)
from project.models.gbdt_standalone import is_gbdt_available
from project.submission import build_submission, validate_submission_contract

NUMERIC_COLS = ["meter_reading", "square_feet", "air_temperature", "hour"]
CAT_COLS = ["primary_use"]
FEATURE_COLS = [*NUMERIC_COLS, *CAT_COLS]
SAMPLE_USES = ["Education", "Office", "Lodging"]


def _make_sample_frame(
    n: int,
    n_pos: int,
    seed: int = 42,
    building_id: str = "b_1",
) -> pl.DataFrame:
    """Generate synthetic test DataFrame with realistic anomaly rate and features."""
    rng = np.random.default_rng(seed)
    anomalies = np.zeros(n, dtype=np.int64)
    if n_pos > 0:
        pos_indices = rng.choice(n, size=min(n, n_pos), replace=False)
        anomalies[pos_indices] = 1

    hours = rng.integers(0, 24, size=n).astype(float)
    return pl.DataFrame(
        {
            "building_id": [building_id] * n,
            "timestamp": [
                datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC)
                + datetime.timedelta(hours=i)
                for i in range(n)
            ],
            "anomaly": anomalies.tolist(),
            "meter_reading": rng.normal(120.0, 30.0, n).tolist(),
            "square_feet": rng.normal(5000.0, 500.0, n).tolist(),
            "air_temperature": rng.normal(18.0, 5.0, n).tolist(),
            "hour": hours.tolist(),
            "primary_use": rng.choice(SAMPLE_USES, n).tolist(),
        }
    )


@pytest.fixture
def mock_dataset() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train and validation partition fixtures."""
    train_df = _make_sample_frame(300, 15, seed=1, building_id="b_1")
    val_df = _make_sample_frame(100, 5, seed=2, building_id="b_2")
    return train_df, val_df


@pytest.fixture
def mock_split_manifest() -> pl.DataFrame:
    """4-building 2-fold mock split manifest."""
    return pl.DataFrame(
        [
            {"building_id": "b_1", "repeat": 0, "fold": 0, "split": "dev"},
            {"building_id": "b_2", "repeat": 0, "fold": 0, "split": "dev"},
            {"building_id": "b_3", "repeat": 0, "fold": 1, "split": "dev"},
            {"building_id": "b_4", "repeat": 0, "fold": 1, "split": "dev"},
        ]
    )


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Load and configure followup experiments YAML mapping."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg["features"]["predictors"] = FEATURE_COLS
    cfg["features"]["categorical"] = CAT_COLS
    return cfg


# ==============================================================================
# EXP-F01 Tests: Diurnal-Weighted Manifold Mixup
# ==============================================================================


def test_diurnal_mixup_increases_minority_ratio():
    """Verify diurnal mixup expands minority count to target ratio without NaNs."""
    df = _make_sample_frame(200, 10, seed=42)
    X = df.select(FEATURE_COLS).to_pandas()
    y = df["anomaly"].to_numpy()

    initial_pos_ratio = np.mean(y == 1)
    target_ratio = 0.08
    assert initial_pos_ratio < target_ratio

    X_syn, y_syn = apply_fold_local_diurnal_mixup(
        X_train=X,
        y_train=y,
        categorical_cols=CAT_COLS,
        target_ratio=target_ratio,
        alpha_base=0.4,
        alpha_trans=0.8,
        trans_hours=(0, 1, 2, 3, 4, 17, 18, 19, 20, 21),
        trans_weight=2.0,
        seed=42,
    )

    resampled_pos_ratio = np.mean(y_syn == 1)
    assert resampled_pos_ratio >= target_ratio
    assert len(X_syn) == len(y_syn)
    assert not X_syn[NUMERIC_COLS].isna().all().any()


def test_diurnal_mixup_weights_favor_transition_hours():
    """Verify that positive instances in transition hours are favored in mixup selection."""
    # Create controlled dataset:
    # - 2 transition-hour positives (hour=2, meter_reading=1000.0)
    # - 2 off-peak positives (hour=12, meter_reading=10.0)
    # - 100 negatives (hour=6, meter_reading=50.0)
    pos_trans = pd.DataFrame(
        {
            "meter_reading": [1000.0, 1000.0],
            "square_feet": [5000.0, 5000.0],
            "air_temperature": [20.0, 20.0],
            "hour": [2.0, 3.0],  # transition hours
            "primary_use": ["Office", "Office"],
        }
    )
    pos_offpeak = pd.DataFrame(
        {
            "meter_reading": [10.0, 10.0],
            "square_feet": [5000.0, 5000.0],
            "air_temperature": [20.0, 20.0],
            "hour": [12.0, 13.0],  # off-peak hours
            "primary_use": ["Office", "Office"],
        }
    )
    negatives = pd.DataFrame(
        {
            "meter_reading": [50.0] * 100,
            "square_feet": [5000.0] * 100,
            "air_temperature": [20.0] * 100,
            "hour": [6.0] * 100,
            "primary_use": ["Office"] * 100,
        }
    )
    X_ctrl = pd.concat([pos_trans, pos_offpeak, negatives], ignore_index=True)
    y_ctrl = np.array([1, 1, 1, 1] + [0] * 100, dtype=np.int64)

    # High trans_weight: synthetic rows should be dominated by transition-hour parents (~1000.0)
    X_high, _y_high = apply_fold_local_diurnal_mixup(
        X_train=X_ctrl,
        y_train=y_ctrl,
        categorical_cols=CAT_COLS,
        target_ratio=0.12,
        trans_hours=(0, 1, 2, 3, 4),
        trans_weight=100.0,
        seed=42,
    )
    syn_high = X_high.iloc[len(y_ctrl):]
    assert len(syn_high) > 0
    mean_meter_high = syn_high["meter_reading"].mean()
    assert mean_meter_high > 700.0  # parents were primarily the ~1000.0 transition positives

    # Low trans_weight: synthetic rows should be dominated by offpeak parents (~10.0)
    X_low, _y_low = apply_fold_local_diurnal_mixup(
        X_train=X_ctrl,
        y_train=y_ctrl,
        categorical_cols=CAT_COLS,
        target_ratio=0.12,
        trans_hours=(0, 1, 2, 3, 4),
        trans_weight=0.01,
        seed=42,
    )
    syn_low = X_low.iloc[len(y_ctrl):]
    assert len(syn_low) > 0
    mean_meter_low = syn_low["meter_reading"].mean()
    assert mean_meter_low < 300.0  # parents were primarily the ~10.0 off-peak positives

def test_train_diurnal_mixup_partition_dry_run(mock_dataset):
    """Test partition training for EXP-F01 with dry_run."""
    train_df, val_df = mock_dataset
    preds = train_diurnal_mixup_partition(
        train_df=train_df,
        val_df=val_df,
        feature_cols=FEATURE_COLS,
        categorical_cols=CAT_COLS,
        dry_run=True,
        seed=42,
    )
    assert len(preds) == len(val_df)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


# ==============================================================================
# EXP-F02 Tests: Platt-Calibrated Champion (Nested Zero-Leakage)
# ==============================================================================


def test_platt_calibrator_fit_and_transform_monotonicity():
    """Verify PlattCalibrator preserves rank-order monotonicity."""
    rng = np.random.default_rng(42)
    # Generate overconfident probabilities
    logits = rng.normal(0.0, 2.0, 500)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, np.clip(probs, 0.01, 0.99))

    calibrator = PlattCalibrator(c_regularization=1.0)
    calibrator.fit(probs, y)

    assert calibrator.is_fitted_
    assert calibrator.slope_ >= 1e-4

    test_probs = np.linspace(0.01, 0.99, 50)
    cal_probs = calibrator.transform(test_probs)

    # Monotonicity test: strictly non-decreasing
    assert np.all(np.diff(cal_probs) >= 0.0)
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))


def test_platt_calibrator_reduces_ece():
    """Test that Platt scaling reduces ECE on severely distorted probabilities."""
    rng = np.random.default_rng(101)
    n = 1000
    # True underlying probabilities
    true_probs = rng.uniform(0.05, 0.40, n)
    y = rng.binomial(1, true_probs)

    # Artificially overconfident predictions
    distorted_probs = np.clip(true_probs ** 0.3, 0.01, 0.99)

    pre_metrics = compute_calibration_metrics(y, distorted_probs)
    calibrator = PlattCalibrator(c_regularization=1.0)
    calibrator.fit(distorted_probs, y)
    cal_probs = calibrator.transform(distorted_probs)
    post_metrics = compute_calibration_metrics(y, cal_probs)

    assert post_metrics["ece"] < pre_metrics["ece"]
    assert post_metrics["brier_score"] <= pre_metrics["brier_score"]


def test_platt_calibrator_degenerate_constant_logits():
    """Verify PlattCalibrator handles constant/degenerate logits gracefully without NaN/Inf."""
    p_const = np.full(100, 0.05)
    y = np.zeros(100, dtype=int)
    y[:10] = 1  # 10% prevalence

    calibrator = PlattCalibrator()
    calibrator.fit(p_const, y)

    assert calibrator.is_fitted_
    cal_preds = calibrator.transform(p_const)
    assert not np.any(np.isnan(cal_preds))
    assert not np.any(np.isinf(cal_preds))
    # Calibrated probability should fall back to prevalence (~0.10)
    assert np.allclose(cal_preds, 0.10, atol=0.01)


def test_platt_calibrator_clamped_slope_well_behaved_intercept():
    """Verify PlattCalibrator clamps negative slope and re-centers intercept to data prevalence."""
    p = np.linspace(0.01, 0.99, 200)
    y = np.where(p < 0.20, 1, 0)  # ~20% positives, inverted with p
    prevalence = np.mean(y)

    calibrator = PlattCalibrator()
    calibrator.fit(p, y)

    assert calibrator.is_fitted_
    assert calibrator.slope_ >= 1e-4
    cal_preds = calibrator.transform(p)
    assert not np.any(np.isnan(cal_preds))
    assert not np.any(np.isinf(cal_preds))
    # Mean of calibrated predictions should match true prevalence (~0.20)
    assert np.isclose(np.mean(cal_preds), prevalence, atol=0.05)


def test_nested_platt_coverage_check_raises_on_zero_positive_coverage(monkeypatch):
    """Verify run_nested_platt_calibration_partition raises AssertionError if inner folds lack positive coverage."""
    train_df = pl.concat(
        [
            _make_sample_frame(100, 0, seed=1, building_id="b_neg_1"),
            _make_sample_frame(100, 0, seed=2, building_id="b_neg_2"),
            _make_sample_frame(10, 5, seed=3, building_id="b_pos"),
        ]
    )
    val_df = _make_sample_frame(50, 2, seed=4, building_id="val_outer")

    # Mock inner splitting to exclude b_pos from validation
    def fake_split(ary, indices_or_sections):
        return [np.array(["b_neg_1"]), np.array(["b_neg_2"])]

    monkeypatch.setattr(np, "array_split", fake_split)

    with pytest.raises(AssertionError, match="Inner-OOF calibration predictions lack genuine coverage"):
        run_nested_platt_calibration_partition(
            train_df=train_df,
            val_df=val_df,
            feature_cols=FEATURE_COLS,
            categorical_cols=CAT_COLS,
            inner_folds=2,
            dry_run=True,
            seed=42,
        )

def test_nested_platt_partition_dry_run_and_zero_leakage(monkeypatch):
    """Test nested inner-CV Platt calibration pipeline and assert zero leakage."""
    import project.models.followup_experiments as fe

    # Set up multi-building train and validation sets
    train_df = pl.concat(
        [
            _make_sample_frame(100, 5, seed=1, building_id="train_b1"),
            _make_sample_frame(100, 5, seed=2, building_id="train_b2"),
            _make_sample_frame(100, 5, seed=3, building_id="train_b3"),
        ]
    )
    val_df = _make_sample_frame(50, 2, seed=4, building_id="val_b_outer")
    outer_val_buildings = set(val_df["building_id"].unique().to_list())

    recorded_calls: list[dict[str, Any]] = []
    real_fn = fe.train_synthetic_sampling_partition

    def spy_fn(**kwargs: Any) -> np.ndarray:
        recorded_calls.append(kwargs)
        return real_fn(**kwargs)

    monkeypatch.setattr(fe, "train_synthetic_sampling_partition", spy_fn)

    preds = run_nested_platt_calibration_partition(
        train_df=train_df,
        val_df=val_df,
        feature_cols=FEATURE_COLS,
        categorical_cols=CAT_COLS,
        inner_folds=3,
        dry_run=True,
        seed=42,
    )

    assert len(preds) == len(val_df)
    assert np.all((preds >= 0.0) & (preds <= 1.0))

    # Assertions on recorded calls:
    # At least 3 inner CV calls + 1 outer champion call
    assert len(recorded_calls) >= 4

    # 1. Verify every call uses champion lightgbm mixup
    for call in recorded_calls:
        assert call["model_family"] == "lightgbm"
        assert call["arm"] == "mixup"

    # 2. Verify inner CV calls never leak outer validation buildings
    inner_calls = recorded_calls[:-1]
    for inner_call in inner_calls:
        in_tr_b = set(inner_call["train_df"]["building_id"].unique().to_list())
        in_va_b = set(inner_call["val_df"]["building_id"].unique().to_list())

        # Inner train and inner val must be disjoint
        assert in_tr_b.isdisjoint(in_va_b)

        # Crucial zero-leakage check: outer val buildings must NEVER appear in inner train or val
        assert in_tr_b.isdisjoint(outer_val_buildings)
        assert in_va_b.isdisjoint(outer_val_buildings)

    # 3. Outer call fits on full train_df and evaluates on val_df
    outer_call = recorded_calls[-1]
    outer_tr_b = set(outer_call["train_df"]["building_id"].unique().to_list())
    outer_va_b = set(outer_call["val_df"]["building_id"].unique().to_list())
    assert outer_tr_b == {"train_b1", "train_b2", "train_b3"}
    assert outer_va_b == {"val_b_outer"}

# ==============================================================================
# EXP-F03 Tests: Skip-Connection Leaf Stacking & Unseen Categories
# ==============================================================================


def test_skip_connection_leaf_stacking_unseen_category_alignment():
    """Verify category vocabulary persistence and unseen value alignment without GBDT dependency."""
    estimator = SkipConnectionLeafStackingEstimator(categorical_cols=["primary_use"])
    train_df = pd.DataFrame({"primary_use": ["Office", "Lodging"], "meter_reading": [10.0, 20.0]})
    val_df = pd.DataFrame({"primary_use": ["Education", "Office"], "meter_reading": [15.0, 25.0]})

    _df_tr, _, _ = estimator._prepare_frames_fit(train_df)
    assert "primary_use" in estimator.cat_categories_
    assert set(estimator.cat_categories_["primary_use"]) == {"Office", "Lodging"}

    df_va, _, _ = estimator._prepare_frames_transform(val_df)
    assert isinstance(df_va["primary_use"].dtype, pd.CategoricalDtype)
    assert list(df_va["primary_use"].cat.categories) == list(estimator.cat_categories_["primary_use"])
    # Unseen category "Education" is safely coded as NaN
    assert pd.isna(df_va["primary_use"].iloc[0])
    assert df_va["primary_use"].iloc[1] == "Office"


@pytest.mark.skipif(not is_gbdt_available("lightgbm"), reason="LightGBM not installed")
def test_skip_connection_leaf_stacking_unseen_category_fit_predict():
    """Verify that unseen categorical values in validation are handled during full fit/predict."""
    train_df = pl.DataFrame(
        {
            "meter_reading": [10.0, 20.0, 30.0, 40.0],
            "square_feet": [100.0, 200.0, 300.0, 400.0],
            "primary_use": ["Office", "Office", "Lodging", "Lodging"],
            "anomaly": [0, 0, 1, 1],
        }
    )
    val_df = pl.DataFrame(
        {
            "meter_reading": [15.0, 25.0],
            "square_feet": [150.0, 250.0],
            "primary_use": ["Education", "Office"],
        }
    )

    estimator = SkipConnectionLeafStackingEstimator(
        categorical_cols=["primary_use"],
        seed=42,
    )
    X_tr = train_df.select(["meter_reading", "square_feet", "primary_use"]).to_pandas()
    y_tr = train_df["anomaly"].to_numpy()
    X_va = val_df.to_pandas()

    estimator.fit(X_tr, y_tr)
    assert "primary_use" in estimator.cat_categories_

    probs = estimator.predict_proba(X_va)
    assert len(probs) == len(val_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


@pytest.mark.skipif(not is_gbdt_available("lightgbm"), reason="LightGBM not installed")
def test_train_diurnal_mixup_partition_real_fit(mock_dataset):
    """Verify real LightGBM fit with diurnal mixup partition training."""
    train_df, val_df = mock_dataset
    preds = train_diurnal_mixup_partition(
        train_df=train_df,
        val_df=val_df,
        feature_cols=FEATURE_COLS,
        categorical_cols=CAT_COLS,
        dry_run=False,
        params={"n_estimators": 10, "num_leaves": 7, "n_jobs": 1},
        seed=42,
    )
    assert len(preds) == len(val_df)
    assert np.all((preds >= 0.0) & (preds <= 1.0))

def test_train_skip_leaf_partition_dry_run(mock_dataset):
    """Test partition training for EXP-F03 with dry-run."""
    train_df, val_df = mock_dataset
    preds = train_skip_leaf_partition(
        train_df=train_df,
        val_df=val_df,
        feature_cols=FEATURE_COLS,
        categorical_cols=CAT_COLS,
        dry_run=True,
        seed=42,
    )
    assert len(preds) == len(val_df)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


# ==============================================================================
# Metrics & Diurnal Slice Diagnostics Tests
# ==============================================================================


def test_compute_calibration_metrics():
    """Verify calibration metrics calculation output contract."""
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = compute_calibration_metrics(y, p, n_bins=5)

    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_compute_diurnal_slice_metrics():
    """Verify diurnal slice metrics splits transition vs off-peak hours."""
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.2, 0.7, 0.3])
    hours = np.array([2, 3, 12, 18, 14])  # 2, 3, 18 are transition hours

    slice_res = compute_diurnal_slice_metrics(
        y_true=y,
        y_prob=p,
        hours=hours,
        trans_hours=(0, 1, 2, 3, 4, 17, 18, 19, 20, 21),
    )

    assert slice_res["transition_count"] == 3
    assert slice_res["offpeak_count"] == 2
    assert "transition_auc" in slice_res
    assert "transition_logloss" in slice_res


# ==============================================================================
# Runner, Schema & Submission Pipeline Tests
# ==============================================================================


def test_parse_args_defaults():
    """Test argument parsing defaults and flags."""
    args = parse_args([])
    assert args.experiment == "all"
    assert not args.smoke
    assert not args.dry_run
    assert not args.submit


def test_parse_args_custom():
    """Test argument parsing with explicit choices."""
    args = parse_args(["--experiment", "exp_f02", "--smoke", "--dry-run", "--submit"])
    assert args.experiment == "exp_f02"
    assert args.smoke
    assert args.dry_run
    assert args.submit


def test_run_followup_experiments_dry_run_conforms_to_schema(
    mock_split_manifest, mock_config, tmp_path
):
    """Verify that run_followup_experiments produces canonical OOF parquets."""
    # Prepare mock training data matching the manifest buildings
    all_b_dfs = [
        _make_sample_frame(80, 4, seed=10 + i, building_id=f"b_{i}")
        for i in range(1, 5)
    ]
    train_dev = pl.concat(all_b_dfs)

    sample_sub = pl.DataFrame(
        {"row_id": list(range(50)), "anomaly": [0.0] * 50}
    )
    test_dev = _make_sample_frame(50, 0, seed=999, building_id="test_b")

    output_dir = tmp_path / "models"
    results = run_followup_experiments(
        experiment_key="all",
        config=mock_config,
        config_path=Path("mock_config.yaml"),
        output_dir=output_dir,
        smoke=True,
        dry_run=True,
        submit=True,
        split_df=mock_split_manifest,
        train_df=train_dev,
        test_df=test_dev,
        sample_sub_df=sample_sub,
    )

    # Verify OOF parquets
    oof_paths = results["oof_paths"]
    assert len(oof_paths) == 3  # exp_f01, exp_f02, exp_f03

    for oof_p in oof_paths:
        assert oof_p.is_file()
        oof_pl = pl.read_parquet(oof_p)
        assert len(oof_pl) > 0
        # Check all 24 columns in CANDIDATE_OOF_SCHEMA
        for col_name in CANDIDATE_OOF_SCHEMA.names:
            assert col_name in oof_pl.columns
            assert oof_pl[col_name].null_count() == 0

    # Verify submission paths
    sub_paths = results["submission_paths"]
    assert len(sub_paths) == 3
    expected_sub_names = {
        "submission_39_diurnal_weighted_mixup.csv",
        "submission_40_platt_calibrated_mixup.csv",
        "submission_41_skip_connection_leaf_stacking.csv",
    }
    actual_sub_names = {p.name for p in sub_paths}
    assert actual_sub_names == expected_sub_names

    for sub_p in sub_paths:
        assert sub_p.is_file()
        sub_df = pl.read_csv(sub_p)
        validate_submission_contract(sub_df, expected_row_count=50)

        assert sub_p.parent.name == "smoke"


@pytest.mark.skipif(not is_gbdt_available("lightgbm"), reason="LightGBM not installed")
def test_skip_connection_float32_sparse_feature_construction():
    """Verify SkipConnectionLeafStackingEstimator uses float32 sparse matrices and encoders."""
    train_df = pl.DataFrame(
        {
            "meter_reading": [10.0, 20.0, 30.0, 40.0],
            "square_feet": [100.0, 200.0, 300.0, 400.0],
            "primary_use": ["Office", "Office", "Lodging", "Lodging"],
            "anomaly": [0, 1, 0, 1],
        }
    )
    estimator = SkipConnectionLeafClassifier(
        categorical_cols=["primary_use"],
        gbdt_params={"n_estimators": 5, "num_leaves": 4, "n_jobs": 1},
        seed=42,
    )
    X_tr = train_df.select(["meter_reading", "square_feet", "primary_use"]).to_pandas()
    y_tr = train_df["anomaly"].to_numpy()
    estimator.fit(X_tr, y_tr)

    assert estimator.leaf_encoder_ is not None
    assert estimator.leaf_encoder_.dtype == np.float32
    assert estimator.scaler_ is not None


def test_test_submission_explicit_row_id_alignment(tmp_path, mock_split_manifest, mock_config):
    """Verify submission generator joins on row_id even when test rows are out of order."""
    all_b_dfs = [
        _make_sample_frame(80, 4, seed=10 + i, building_id=f"b_{i}")
        for i in range(1, 5)
    ]
    train_dev = pl.concat(all_b_dfs)

    # Sample sub in natural order [0, 1, ..., 49]
    sample_sub = pl.DataFrame(
        {"row_id": list(range(50)), "anomaly": [0.0] * 50}
    )
    # Test data in reverse row_id order [49, 48, ..., 0]
    rev_row_ids = list(range(49, -1, -1))
    test_dev = _make_sample_frame(50, 0, seed=999, building_id="test_b").with_columns(
        pl.Series("row_id", rev_row_ids, dtype=pl.Int64)
    )

    out_dir = tmp_path / "models"
    sub_dir = tmp_path / "submissions"

    results = run_followup_experiments(
        experiment_key="exp_f01",
        config=mock_config,
        config_path=Path("mock_config.yaml"),
        output_dir=out_dir,
        submissions_dir=sub_dir,
        smoke=True,
        dry_run=True,
        submit=True,
        split_df=mock_split_manifest,
        train_df=train_dev,
        test_df=test_dev,
        sample_sub_df=sample_sub,
    )

    sub_path = results["submission_paths"][0]
    assert sub_path.is_file()
    sub_df = pl.read_csv(sub_path)
    validate_submission_contract(sub_df, expected_row_count=50, expected_row_ids=list(range(50)))
    # Output must strictly match sample_sub's row_id sequence [0, 1, ..., 49]
    assert sub_df["row_id"].to_list() == list(range(50))

def test_build_submission_contract():
    """Verify build_submission creates valid submissions."""
    row_ids = [0, 1, 2]
    probs = [0.1, 0.5, 0.9]
    sub = build_submission(row_ids, probs)
    assert sub.schema == {"row_id": pl.Int64, "anomaly": pl.Float64}
    assert sub["row_id"].to_list() == [0, 1, 2]
    assert np.allclose(sub["anomaly"].to_numpy(), [0.1, 0.5, 0.9])
