"""Unit tests for Kaggle test submission pipeline and submission contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from project.submission import (
    BASE_MODEL_ORDER,
    CORE_GBDT_PREDICTOR_COLUMNS,
    SAFE_PREDICTOR_COLUMNS,
    SUBMISSION_SPECS,
    audit_submission_file,
    build_submission,
    generate_all_submissions,
    predict_catboost_control,
    predict_catboost_weighted,
    predict_ensemble_meta,
    predict_extratrees,
    predict_extratrees_natural,
    predict_lightgbm_challenger,
    predict_lightgbm_champion,
    predict_lightgbm_focal,
    predict_mlp_control,
    predict_tabular_mlp,
    predict_xgboost_champion,
    predict_xgboost_control,
    save_submission,
    validate_submission_contract,
)


@pytest.fixture
def synthetic_sample_sub() -> pl.DataFrame:
    """Generate small synthetic sample submission."""
    return pl.DataFrame(
        {
            "row_id": list(range(100)),
            "anomaly": [0] * 100,
        }
    )


@pytest.fixture
def synthetic_train_test() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate small synthetic train and test DataFrames with 31 safe predictors."""
    n_train, n_test = 200, 100
    rng = np.random.default_rng(42)

    train_data = {
        "row_id": list(range(n_train)),
        "building_id": [f"b_{i % 5}" for i in range(n_train)],
        "timestamp": [f"2016-01-01 {i % 24:02d}:00:00" for i in range(n_train)],
        "anomaly": rng.integers(0, 2, size=n_train).tolist(),
    }
    test_data = {
        "row_id": list(range(n_test)),
        "building_id": [f"b_{i % 5}" for i in range(n_test)],
        "timestamp": [f"2017-01-01 {i % 24:02d}:00:00" for i in range(n_test)],
    }

    for col in SAFE_PREDICTOR_COLUMNS:
        if col == "primary_use":
            train_data[col] = [f"use_{i % 3}" for i in range(n_train)]
            test_data[col] = [f"use_{i % 3}" for i in range(n_test)]
        else:
            train_data[col] = rng.standard_normal(n_train).tolist()
            test_data[col] = rng.standard_normal(n_test).tolist()

    return pl.DataFrame(train_data), pl.DataFrame(test_data)


def test_safe_predictor_columns_count() -> None:
    """Check that SAFE_PREDICTOR_COLUMNS contains exactly the 31 required features."""
    assert len(SAFE_PREDICTOR_COLUMNS) == 31
    assert "meter_reading" in SAFE_PREDICTOR_COLUMNS
    assert "primary_use" in SAFE_PREDICTOR_COLUMNS
    assert "air_temperature_std_lag73" in SAFE_PREDICTOR_COLUMNS
    assert "anomaly" not in SAFE_PREDICTOR_COLUMNS
    assert "row_id" not in SAFE_PREDICTOR_COLUMNS
    assert "building_id" not in SAFE_PREDICTOR_COLUMNS


def test_core_gbdt_predictor_columns() -> None:
    """Check that CORE_GBDT_PREDICTOR_COLUMNS contains exactly the 9 core features."""
    expected_9 = {
        "meter_reading",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
        "primary_use",
    }
    assert len(CORE_GBDT_PREDICTOR_COLUMNS) == 9
    assert set(CORE_GBDT_PREDICTOR_COLUMNS) == expected_9

    # All 9 core features must also be a subset of the 31 SAFE_PREDICTOR_COLUMNS
    for col in CORE_GBDT_PREDICTOR_COLUMNS:
        assert col in SAFE_PREDICTOR_COLUMNS

def test_submission_specs_integrity() -> None:
    """Check that all 8 champion submission specifications are defined with expected metadata."""
    assert len(SUBMISSION_SPECS) == 8

    expected_files = {
        "submission_01_ensemble_logit_lr_0.9008.csv",
        "submission_02_lightgbm_champion_0.8946.csv",
        "submission_03_ensemble_caruana_0.8908.csv",
        "submission_04_xgboost_champion_0.8937.csv",
        "submission_05_lightgbm_focal_0.8937.csv",
        "submission_06_catboost_weighted_0.8746.csv",
        "submission_07_extratrees_0.8740.csv",
        "submission_08_tabular_mlp_0.8302.csv",
    }
    actual_files = {spec["filename"] for spec in SUBMISSION_SPECS.values()}
    assert actual_files == expected_files

    # Check model order consistency for ensembles
    assert len(BASE_MODEL_ORDER) == 6
    assert BASE_MODEL_ORDER == (
        "lightgbm_champion",
        "xgboost_champion",
        "catboost_weighted",
        "extratrees",
        "lightgbm_focal",
        "tabular_mlp",
    )

    # Check focal loss has alpha=0.25, gamma=2.0
    focal_desc = SUBMISSION_SPECS["submission_05_lightgbm_focal"]["description"]
    assert "alpha=0.25" in focal_desc
    assert "gamma=2.0" in focal_desc

    # Check catboost description specifies 9 core features and no class weights
    cb_desc = SUBMISSION_SPECS["submission_06_catboost_weighted"]["description"]
    assert "no class weights" in cb_desc
    assert "9 core features" in cb_desc


def test_validate_submission_contract_valid() -> None:
    """Check valid submission passes contract validation."""
    valid_df = pl.DataFrame(
        {
            "row_id": [0, 1, 2],
            "anomaly": [0.05, 0.50, 0.95],
        }
    )
    # Should not raise
    validate_submission_contract(
        valid_df, expected_row_count=3, expected_row_ids=[0, 1, 2]
    )


def test_validate_submission_contract_invalid_columns() -> None:
    """Check validation fails on invalid or extra column names."""
    df_wrong_col = pl.DataFrame({"row_id": [0, 1], "prediction": [0.1, 0.2]})
    with pytest.raises(ValueError, match="Expected exact columns"):
        validate_submission_contract(df_wrong_col)

    df_extra_col = pl.DataFrame(
        {"row_id": [0, 1], "anomaly": [0.1, 0.2], "extra": [1, 2]}
    )
    with pytest.raises(ValueError, match="Expected exact columns"):
        validate_submission_contract(df_extra_col)


def test_validate_submission_contract_invalid_row_count() -> None:
    """Check validation fails when row count does not match expectation."""
    df = pl.DataFrame({"row_id": [0, 1], "anomaly": [0.1, 0.2]})
    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_submission_contract(df, expected_row_count=5)


def test_validate_submission_contract_nan_inf_null() -> None:
    """Check validation fails on null, NaN, or inf values."""
    df_nan = pl.DataFrame({"row_id": [0, 1], "anomaly": [0.1, float("nan")]})
    with pytest.raises(ValueError, match="contains NaN or null"):
        validate_submission_contract(df_nan, expected_row_count=2)

    df_inf = pl.DataFrame({"row_id": [0, 1], "anomaly": [0.1, float("inf")]})
    with pytest.raises(ValueError, match="contains infinite"):
        validate_submission_contract(df_inf, expected_row_count=2)

    df_null = pl.DataFrame({"row_id": [0, 1], "anomaly": [0.1, None]})
    with pytest.raises(ValueError, match="contains NaN or null"):
        validate_submission_contract(df_null, expected_row_count=2)


def test_validate_submission_contract_bounds() -> None:
    """Check validation fails when probabilities are out of [0.0, 1.0]."""
    df_low = pl.DataFrame({"row_id": [0, 1], "anomaly": [-0.01, 0.5]})
    with pytest.raises(ValueError, match="Probabilities out of bounds"):
        validate_submission_contract(df_low, expected_row_count=2)

    df_high = pl.DataFrame({"row_id": [0, 1], "anomaly": [0.5, 1.05]})
    with pytest.raises(ValueError, match="Probabilities out of bounds"):
        validate_submission_contract(df_high, expected_row_count=2)


def test_validate_submission_contract_row_ids_mismatch() -> None:
    """Check validation fails if row_id order or values do not match expected sequence."""
    df = pl.DataFrame({"row_id": [1, 0], "anomaly": [0.1, 0.2]})
    with pytest.raises(ValueError, match="row_id sequence mismatch"):
        validate_submission_contract(df, expected_row_count=2, expected_row_ids=[0, 1])


def test_build_submission() -> None:
    """Check build_submission constructs properly typed Polars DataFrame."""
    row_ids = [10, 20, 30]
    probs = [0.1, 0.4, 0.9]
    sub = build_submission(row_ids, probs)

    assert sub.columns == ["row_id", "anomaly"]
    assert sub.schema["row_id"] == pl.Int64
    assert sub.schema["anomaly"] == pl.Float64
    assert sub.height == 3
    assert sub["row_id"].to_list() == [10, 20, 30]
    assert sub["anomaly"].to_list() == pytest.approx([0.1, 0.4, 0.9])


def test_build_submission_rejects_non_finite() -> None:
    """Check that build_submission rejects NaN, Inf, and -Inf rather than replacing them."""
    with pytest.raises(ValueError, match="Predicted probabilities contain non-finite values"):
        build_submission([0, 1], [0.5, np.nan])

    with pytest.raises(ValueError, match="Predicted probabilities contain non-finite values"):
        build_submission([0, 1], [np.inf, 0.5])

    with pytest.raises(ValueError, match="Predicted probabilities contain non-finite values"):
        build_submission([0, 1], [-np.inf, 0.5])


def test_audit_submission_file(tmp_path: Path) -> None:
    """Check that audit_submission_file validates contract against reference sample_submission."""
    ref_file = tmp_path / "sample_sub.csv"
    pl.DataFrame({"row_id": [0, 1, 2], "anomaly": [0, 0, 0]}).write_csv(ref_file)

    good_file = tmp_path / "good_sub.csv"
    pl.DataFrame({"row_id": [0, 1, 2], "anomaly": [0.1, 0.4, 0.8]}).write_csv(good_file)

    # Valid audit
    assert audit_submission_file(good_file, ref_file) is True

    # Length mismatch
    bad_len_file = tmp_path / "bad_len.csv"
    pl.DataFrame({"row_id": [0, 1], "anomaly": [0.1, 0.4]}).write_csv(bad_len_file)
    with pytest.raises(ValueError, match="Row count mismatch"):
        audit_submission_file(bad_len_file, ref_file)

    # row_id sequence mismatch
    bad_seq_file = tmp_path / "bad_seq.csv"
    pl.DataFrame({"row_id": [2, 1, 0], "anomaly": [0.1, 0.4, 0.8]}).write_csv(bad_seq_file)
    with pytest.raises(ValueError, match="row_id sequence mismatch"):
        audit_submission_file(bad_seq_file, ref_file)


def test_model_candidate_aliases() -> None:
    """Check that model function aliases match canonical OOF generator names."""
    assert predict_lightgbm_challenger is predict_lightgbm_champion
    assert predict_xgboost_control is predict_xgboost_champion
    assert predict_catboost_control is predict_catboost_weighted
    assert predict_extratrees_natural is predict_extratrees
    assert predict_mlp_control is predict_tabular_mlp


def test_generate_all_submissions_smoke_dir_routing(
    synthetic_train_test: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """Check that default output_dir routes to smoke subdirectory in dry-run mode."""
    train_df, test_df = synthetic_train_test
    results = generate_all_submissions(
        train_df=train_df,
        test_df=test_df,
        models_mode="single",
        dry_run=True,
    )
    for path in results.values():
        assert "outputs/submissions/smoke" in str(path)
        assert path.is_file()

def test_save_submission(tmp_path: Path) -> None:
    """Check save_submission writes valid CSV conforming to contract."""
    sub = build_submission([1, 2], [0.3, 0.7])
    out_file = tmp_path / "test_sub.csv"
    saved = save_submission(sub, out_file, expected_row_count=2)

    assert saved.is_file()
    read_back = pl.read_csv(saved)
    assert read_back.columns == ["row_id", "anomaly"]
    assert read_back.height == 2
    validate_submission_contract(
        read_back, expected_row_count=2, expected_row_ids=[1, 2]
    )


def test_predict_ensemble_meta_on_synthetic_oof() -> None:
    """Check Level-2 meta-learner predictions using Logit LR and Caruana."""
    n_oof = 500
    n_test = 50
    rng = np.random.default_rng(42)

    # Synthetic OOF predictions for 6 models
    oof_cols = [f"pred_m{i}" for i in range(6)]
    oof_dict: dict[str, list[float] | list[int]] = {
        col: rng.uniform(0.01, 0.99, size=n_oof).tolist() for col in oof_cols
    }
    oof_dict["target"] = rng.integers(0, 2, size=n_oof).tolist()
    oof_df = pl.DataFrame(oof_dict)

    # Synthetic test probabilities for 6 models
    test_probs = rng.uniform(0.01, 0.99, size=(n_test, 6))

    # Test Logit LR
    lr_preds = predict_ensemble_meta(
        oof_df=oof_df,
        oof_pred_cols=oof_cols,
        test_prob_matrix=test_probs,
        arm_id="challenger_lr",
        random_state=42,
    )
    assert len(lr_preds) == n_test
    assert np.all((lr_preds >= 0.0) & (lr_preds <= 1.0))

    # Test Caruana
    caruana_preds = predict_ensemble_meta(
        oof_df=oof_df,
        oof_pred_cols=oof_cols,
        test_prob_matrix=test_probs,
        arm_id="control",
        random_state=42,
    )
    assert len(caruana_preds) == n_test
    assert np.all((caruana_preds >= 0.0) & (caruana_preds <= 1.0))


def test_single_model_predict_dry_run(
    synthetic_train_test: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """Check that all 6 single models produce bounded probabilities in dry-run mode."""
    train_df, test_df = synthetic_train_test
    n_test = test_df.height

    p_lgb = predict_lightgbm_champion(train_df, test_df, dry_run=True)
    assert len(p_lgb) == n_test
    assert np.all((p_lgb >= 0.0) & (p_lgb <= 1.0))

    p_xgb = predict_xgboost_champion(train_df, test_df, dry_run=True)
    assert len(p_xgb) == n_test
    assert np.all((p_xgb >= 0.0) & (p_xgb <= 1.0))

    p_focal = predict_lightgbm_focal(train_df, test_df, dry_run=True)
    assert len(p_focal) == n_test
    assert np.all((p_focal >= 0.0) & (p_focal <= 1.0))

    p_cb = predict_catboost_weighted(train_df, test_df, dry_run=True)
    assert len(p_cb) == n_test
    assert np.all((p_cb >= 0.0) & (p_cb <= 1.0))

    p_et = predict_extratrees(train_df, test_df, dry_run=True)
    assert len(p_et) == n_test
    assert np.all((p_et >= 0.0) & (p_et <= 1.0))

    p_mlp = predict_tabular_mlp(train_df, test_df, dry_run=True)
    assert len(p_mlp) == n_test
    assert np.all((p_mlp >= 0.0) & (p_mlp <= 1.0))


def test_generate_all_submissions_dry_run(
    synthetic_train_test: tuple[pl.DataFrame, pl.DataFrame],
    tmp_path: Path,
) -> None:
    """Check generate_all_submissions generates all 8 compliant files in dry-run mode."""
    train_df, test_df = synthetic_train_test

    results = generate_all_submissions(
        train_df=train_df,
        test_df=test_df,
        output_dir=tmp_path,
        models_mode="all",
        dry_run=True,
    )

    for path in results.values():
        assert path.is_file(), f"Submission file {path} was not created"
        df = pl.read_csv(path)
        validate_submission_contract(
            df,
            expected_row_count=test_df.height,
            expected_row_ids=test_df["row_id"].to_list(),
        )
